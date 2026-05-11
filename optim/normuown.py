"""
Muown: Muon optimizer with internal Weight Normalization.

This optimizer implements weight normalization dynamics purely within the optimizer,
without requiring parametrizations.weight_norm() on the model. It maintains an implicit
(g, v) parameterization for 2D weights, applying Muon to the direction component and
AdamW to the magnitude component and all 1D/0D parameters.
"""

import os
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim.optimizer import Optimizer

USE_DDP = "RANK" in os.environ
RANK = int(os.environ["RANK"]) if USE_DDP else 0
WORLD_SIZE = int(os.environ["WORLD_SIZE"]) if USE_DDP else 1


def zeropower_via_newtonschulz5_torch(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """
    Newton-Schulz iteration using the more efficient addmm factorization from torch.optim.Muon.

    Computes (aI + bA + cA²)X by first computing bA + cA² in the smaller m×m space,
    then applying it to X. This saves FLOPs for rectangular matrices compared to
    computing AX and A(AX) in the larger m×n space.

    Reference: https://github.com/pytorch/pytorch/blob/main/torch/optim/muon.py
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X.div_(X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, B, X, beta=a)
    if G.size(0) > G.size(1):
        X = X.T
    return X


def zeropower_via_svd(G: Tensor, steps: Optional[int] = None) -> Tensor:
    """Compute orthogonalization via SVD (exact but slow)."""
    U, S, V = G.svd()
    return U @ V.T


@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Uses a quintic iteration whose coefficients are selected to maximize the slope at zero.
    This produces something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5),
    which works well in practice for optimization.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


# Backend registry
ZEROPOWER_BACKENDS = {
    "svd": zeropower_via_svd,
    "newtonschulz5": zeropower_via_newtonschulz5,
    "newtonschulz5_torch": zeropower_via_newtonschulz5_torch,
}

def normuon_update(grad, momentum, second_momentum, beta=0.95, beta2=0.95, ns_steps=5, nesterov=True):
    """Muon-style orthogonalized update with per-row Adam-like second-moment rescaling.

    The orthogonalized direction is rescaled by an exponentially-decayed per-row
    second moment of the update magnitudes, then re-normalized so that the total
    update norm matches the pre-rescaling orthogonal update (norm preservation).
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    original_shape = None
    if update.ndim == 4:  # for the case of conv filters
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update = update.to(grad.dtype)

    if original_shape is not None:
        update = update.reshape(original_shape)
    ################ NorMuon added ###################
    vnorm = update.norm(dim=(-2, -1), keepdim=True)
    v_mean = torch.mean(update * update, dim=-1, keepdim=True)
    second_momentum.lerp_(v_mean, 1 - beta2)
    step_size = 1 / second_momentum.sqrt().add_(1e-10)
    update.mul_(step_size)
    vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
    update.mul_(vnorm / (vnorm_new.add_(1e-10)))  # keep update norm the same as pre-normalization
    ##################################################
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class SingleDeviceNorMuon(torch.optim.Optimizer):
    """
    NorMuon variant for usage in non-distributed settings.

    Applies Muon-style Newton-Schulz orthogonalization with an additional per-row
    Adam-like second-moment rescaling step (while preserving the update norm).
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, beta2=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                had_grad = p.grad is not None
                if not had_grad:
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                update = normuon_update(
                    p.grad,
                    state["momentum_buffer"],
                    state["second_momentum_buffer"],
                    beta=group["momentum"],
                    beta2=group["beta2"],
                )
                if group["weight_decay"] and had_grad:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


# -----------------------------------------------------------------------------
# Muown optimizer


@torch.compile
def _normuon_rescale(update: torch.Tensor, second_momentum: torch.Tensor, beta2: float) -> torch.Tensor:
    """Per-row NorMuon rescaling applied to an already-orthogonalized update.

    Divides the update by the sqrt of an EMA of per-row mean-square magnitudes,
    then renormalizes so the overall Frobenius norm matches the pre-rescaling
    orthogonal update. Mutates ``second_momentum`` in place.
    """
    vnorm = update.norm(dim=(-2, -1), keepdim=True)
    v_mean = torch.mean(update * update, dim=-1, keepdim=True)
    second_momentum.lerp_(v_mean, 1 - beta2)
    step_size = 1 / second_momentum.sqrt().add(1e-10)
    update = update * step_size
    vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
    update = update * (vnorm / vnorm_new.add(1e-10))
    return update


@torch.compile
def _wn_pre_ns(W: torch.Tensor, g: torch.Tensor, v_norm: torch.Tensor, grad_W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused reconstruct_v + compute_wn_gradients.

    Reconstructs direction v from (W, g, v_norm), then computes the weight norm
    Jacobian to get gradients for g and v from grad_W.
    """
    v = (v_norm / g) * W
    u = v / v_norm
    grad_g = (grad_W * u).sum(dim=1, keepdim=True)
    grad_v = (g / v_norm) * (grad_W - u * grad_g)
    return v, grad_g, grad_v


@torch.compile
def _wn_recompose(W: torch.Tensor, g: torch.Tensor, v_new: torch.Tensor) -> torch.Tensor:
    """Fused recompose W[:] = g * v_new / ||v_new||, writing directly into W."""
    v_norm_new = v_new.norm(dim=1, keepdim=True)
    W.copy_((g / v_norm_new) * v_new)
    return v_norm_new


class NorMuown(torch.optim.Optimizer):
    """
    NorMuown: NorMuon optimizer with internal Weight Normalization.

    This optimizer implements weight normalization dynamics purely within the optimizer,
    without requiring `parametrizations.weight_norm()` on the model. For 2D weight matrices,
    it maintains an implicit (g, v) parameterization where:
    - g is the magnitude (per output row)
    - v is the direction parameter
    - W = g * v / ||v|| is the composed weight used in forward

    The optimizer applies:
    - Muon (momentum + Newton-Schulz orthogonalization) to the direction component v
    - AdamW to the magnitude component g and all 1D/0D parameters

    Arguments:
        params: Parameters for the optimizer.
        lr: Base learning rate (shared between AdamW and Muon due to 0.2 scaling).
        momentum: Momentum factor for Muon algorithm.
        nesterov: Whether to use Nesterov-style momentum in Muon.
        betas: Tuple of (beta1, beta2) for AdamW updates on g and 1D params.
        weight_decay: Decoupled weight decay coefficient (applied to magnitude g).
        adam_eps: Small epsilon for AdamW denominator.
        ns_steps: Number of Newton-Schulz iterations for orthogonalization.
        backend: Backend for orthogonalization ('newtonschulz5', 'newtonschulz5_torch', or 'svd').
        use_normuon: If True, apply NorMuon-style per-row second-moment rescaling to the
            direction update after Newton-Schulz (preserving the Frobenius norm of the
            orthogonal update). If False, use the standard Muon direction update.
        normuon_beta2: EMA coefficient for the per-row second moment used by NorMuon.
        shape_scaling_type: Controls the per-step shape-dependent scale factor applied
            to the direction update, independently of ``use_normuon``. One of:

            - ``None`` (default): mirror ``use_normuon`` (preserves original behavior).
            - ``"normuon"``: NorMuon-style scale ``0.2 * max(1, out/in) ** 0.5``
              (and ``0.2 * max(1, chunk_size/in) ** 0.5`` for QKV chunks).
            - ``"muon"``: standard Muon scale ``0.2 * max(out, in) ** 0.5``
              (and ``0.2 * chunk_size ** 0.5`` for QKV chunks).
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        adam_eps: float = 1e-8,
        ns_steps: int = 5,
        backend: str = "newtonschulz5",
        use_normuon: bool = False,
        normuon_beta2: float = 0.95,
        shape_scaling_type: Optional[str] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 <= normuon_beta2 < 1.0:
            raise ValueError(f"Invalid normuon_beta2: {normuon_beta2}")
        if backend not in ZEROPOWER_BACKENDS:
            raise ValueError(
                f"Invalid backend: {backend}. Choose from {list(ZEROPOWER_BACKENDS.keys())}",
            )
        if shape_scaling_type not in (None, "normuon", "muon"):
            raise ValueError(
                f"Invalid shape_scaling_type: {shape_scaling_type!r}. "
                "Choose from None, 'normuon', or 'muon'."
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            betas=betas,
            weight_decay=weight_decay,
            adam_eps=adam_eps,
            ns_steps=ns_steps,
            backend=backend,
            use_normuon=use_normuon,
            normuon_beta2=normuon_beta2,
            shape_scaling_type=shape_scaling_type,
        )
        super().__init__(params, defaults)

        self._zeropower_fn = ZEROPOWER_BACKENDS[backend]

    def _init_state_2d(self, p: torch.Tensor, state: dict, use_normuon: bool = False) -> None:
        """Initialize weight norm state for a 2D parameter."""
        w_norm = p.data.norm(dim=1, keepdim=True)
        state["g"] = w_norm.clone()
        state["v_norm"] = w_norm.clone()
        state["m_v"] = torch.zeros_like(p.data)
        state["m_g"] = torch.zeros_like(w_norm)
        state["v_g"] = torch.zeros_like(w_norm)
        if use_normuon:
            # per-row second moment of the orthogonalized direction update
            state["v_normuon"] = torch.zeros_like(w_norm)
        state["step"] = 0

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            adam_eps = group["adam_eps"]
            ns_steps = group["ns_steps"]
            use_normuon = group["use_normuon"]
            normuon_beta2 = group["normuon_beta2"]
            shape_scaling_type = group["shape_scaling_type"]
            if shape_scaling_type is None:
                shape_scaling_type = "normuon" if use_normuon else "muon"
            use_ratio_scaling = shape_scaling_type == "normuon"

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    self._init_state_2d(p, state, use_normuon=use_normuon)
                elif use_normuon and "v_normuon" not in state:
                    # support toggling on use_normuon after initialization
                    state["v_normuon"] = torch.zeros_like(state["g"])

                state["step"] += 1
                step = state["step"]

                g = state["g"]
                v_norm = state["v_norm"]
                m_v = state["m_v"]
                m_g = state["m_g"]
                v_g = state["v_g"]
                if weight_decay != 0.0:
                    W_old = p.data.clone()

                # Fused: reconstruct v + compute weight norm gradients
                v, grad_g, grad_v = _wn_pre_ns(p.data, g, v_norm, grad)

                # Muon update on v: momentum + orthogonalization
                m_v.mul_(momentum).add_(grad_v)
                if nesterov:
                    update = grad_v.add(m_v, alpha=momentum)
                else:
                    update = m_v.clone()

                is_qkv = p.size(0) == 3 * p.size(1)
                if is_qkv:
                    chunk_size = update.size(1)
                    chunks = update.split(chunk_size, dim=0)
                    ortho_chunks = [self._zeropower_fn(chunk, steps=ns_steps) for chunk in chunks]
                    update = torch.cat(ortho_chunks, dim=0)
                    if use_normuon:
                        update = _normuon_rescale(update, state["v_normuon"], normuon_beta2)
                    if use_ratio_scaling:
                        # NorMuon scaling: max(1, out/in)**0.5. For QKV chunks the
                        # in-feature dim matches the weight's in-feature dim, so use
                        # per-chunk shape for the scale.
                        scale = 0.2 * max(1.0, chunk_size / update.size(1)) ** 0.5
                    else:
                        scale = 0.2 * chunk_size**0.5
                else:
                    update = self._zeropower_fn(update, steps=ns_steps)
                    if use_normuon:
                        update = _normuon_rescale(update, state["v_normuon"], normuon_beta2)
                    if use_ratio_scaling:
                        scale = 0.2 * max(1.0, update.size(-2) / update.size(-1)) ** 0.5
                    else:
                        scale = 0.2 * max(update.size(0), update.size(1)) ** 0.5

                v_new = v.add(update, alpha=-lr * scale)

                # Adam update on g (small [out_features, 1] vectors)
                beta1, beta2 = betas
                m_g.mul_(beta1).add_(grad_g, alpha=1 - beta1)
                v_g.mul_(beta2).addcmul_(grad_g, grad_g, value=1 - beta2)
                bc1 = 1 - beta1**step
                bc2 = 1 - beta2**step

                g.addcdiv_(m_g / bc1, (v_g / bc2).sqrt().add_(adam_eps), value=-lr)

                # Fused: recompose W = g * v_new / ||v_new||, writes directly into p.data
                state["v_norm"] = _wn_recompose(p.data, g, v_new)
                if weight_decay != 0.0:
                    p.data.add_(W_old, alpha=-lr * weight_decay)

        return loss

