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


@torch.compile
def _wn_pre_ns(W: Tensor, g: Tensor, v_norm: Tensor, grad_W: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Fused reconstruct_v + compute_wn_gradients.

    Reconstructs direction v from (W, g, v_norm), then computes the weight norm
    Jacobian to get gradients for g and v from grad_W.

    Uses numerically stable ordering: the invariant W_i = g_i * v_hat_i means
    W / g is O(1) per element, while v_norm / g or 1 / g can overflow for tiny
    |g|.
    """
    u = W / g
    v = u * v_norm
    grad_g = (grad_W * u).sum(dim=1, keepdim=True)
    grad_v = (g / v_norm) * (grad_W - u * grad_g)
    return v, grad_g, grad_v


@torch.compile
def _wn_recompose(W: Tensor, g: Tensor, v_new: Tensor) -> Tensor:
    """Fused recompose W[:] = g * v_new / ||v_new||, writing directly into W."""
    v_norm_new = v_new.norm(dim=1, keepdim=True)
    W.copy_(g * (v_new / v_norm_new))
    return v_norm_new


class Muown(Optimizer):
    """
    Muown: Muon optimizer with internal Weight Normalization.

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
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        adam_eps: float = 1e-8,
        ns_steps: int = 5,
        backend: str = "newtonschulz5",
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
        if backend not in ZEROPOWER_BACKENDS:
            raise ValueError(
                f"Invalid backend: {backend}. Choose from {list(ZEROPOWER_BACKENDS.keys())}",
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
        )
        super().__init__(params, defaults)

        self._zeropower_fn = ZEROPOWER_BACKENDS[backend]

    def _init_state_2d(self, p: Tensor, state: dict) -> None:
        """Initialize weight norm state for a 2D parameter."""
        w_norm = p.data.norm(dim=1, keepdim=True)
        state["g"] = w_norm.clone()
        state["v_norm"] = w_norm.clone()
        state["m_v"] = torch.zeros_like(p.data)
        state["m_g"] = torch.zeros_like(w_norm)
        state["v_g"] = torch.zeros_like(w_norm)
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

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    self._init_state_2d(p, state)

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
                    scale = 0.2 * chunk_size**0.5
                else:
                    update = self._zeropower_fn(update, steps=ns_steps)
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
                    # Decoupled WD perturbs p.data off the v_new direction, so the row
                    # norms of p.data no longer equal g. Resync g so the (g, v_norm)
                    # state stays consistent with the invariant ||p.data[i]|| == g[i].
                    g.copy_(p.data.norm(dim=1, keepdim=True))

        return loss


def _param_to_complexity(p: Tensor) -> int:
    """Approximate NS complexity for load-balanced sorting."""
    m, n = p.shape[0], p[0].numel()
    return 2 * (m**2) * n + m**3


class MuownDP(Muown):
    """
    Data-parallel Muown: distributes Newton-Schulz orthogonalization across ranks.

    Parameters are sorted by NS complexity (descending) and processed in blocks
    of WORLD_SIZE. Each rank computes the full Muown update for its assigned
    parameters, then all_gather synchronizes the updated weights.

    Comms: one all-gather per block -> ~#params/WORLD_SIZE comms.
    Space: O(largest_param); O(WORLD_SIZE * largest_param) transient during all_gather.
    """

    rank_sharded = True

    def __init__(self, params, **kwargs):
        if not isinstance(params, list):
            params = list(params)

        if not dist.is_initialized():
            raise ValueError("Using MuownDP in a non-distributed run.")

        if isinstance(params[0], dict):
            for group in params:
                group["params"] = sorted(group["params"], key=_param_to_complexity, reverse=True)
        else:
            params = sorted(params, key=_param_to_complexity, reverse=True)

        super().__init__(params, **kwargs)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        allgather_handles = []
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            adam_eps = group["adam_eps"]
            ns_steps = group["ns_steps"]
            params = group["params"]

            pad = (WORLD_SIZE - len(params) % WORLD_SIZE) % WORLD_SIZE
            params_pad = params + [torch.empty_like(params[-1]) for _ in range(pad)]

            for block_start in range(0, len(params), WORLD_SIZE):
                if block_start + RANK < len(params):
                    p = params[block_start + RANK]
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    grad = p.grad
                    state = self.state[p]

                    if len(state) == 0:
                        self._init_state_2d(p, state)

                    state["step"] += 1
                    step = state["step"]

                    g = state["g"]
                    v_norm = state["v_norm"]
                    m_v = state["m_v"]
                    m_g = state["m_g"]
                    v_g = state["v_g"]
                    if weight_decay != 0.0:
                        W_old = p.data.clone()

                    v, grad_g, grad_v = _wn_pre_ns(p.data, g, v_norm, grad)

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
                        scale = 0.2 * chunk_size**0.5
                    else:
                        update = self._zeropower_fn(update, steps=ns_steps)
                        scale = 0.2 * max(update.size(0), update.size(1)) ** 0.5

                    v_new = v.add(update, alpha=-lr * scale)

                    beta1, beta2 = betas
                    m_g.mul_(beta1).add_(grad_g, alpha=1 - beta1)
                    v_g.mul_(beta2).addcmul_(grad_g, grad_g, value=1 - beta2)
                    bc1 = 1 - beta1**step
                    bc2 = 1 - beta2**step
                    if weight_decay != 0.0:
                        g.mul_(1 - lr * weight_decay)
                    g.addcdiv_(m_g / bc1, (v_g / bc2).sqrt().add_(adam_eps), value=-lr)

                    state["v_norm"] = _wn_recompose(p.data, g, v_new)
                    if weight_decay != 0.0:
                        p.data.add_(W_old, alpha=-lr * weight_decay)
                        g.copy_(p.data.norm(dim=1, keepdim=True))

                handle = dist.all_gather(
                    # output container for all-gathered weights
                    params_pad[block_start : block_start + WORLD_SIZE], 
                    # local input tensor, containing recomposed weights
                    params_pad[block_start + RANK], 
                    async_op=True,
                )
                allgather_handles.append(handle)

        for handle in allgather_handles:
            handle.wait()

        return loss

MAGNITUDE_OPTIMS = {"fixed", "adam", "sgd", "signum", "lion"}


class MuownMagnitudeExp(Optimizer):
    """
    Muown variant with configurable magnitude optimizer.

    This optimizer implements weight normalization dynamics purely within the optimizer,
    without requiring `parametrizations.weight_norm()` on the model. For 2D weight matrices,
    it maintains an implicit (g, v) parameterization where:
    - g is the magnitude (per output row)
    - v is the direction parameter
    - W = g * v / ||v|| is the composed weight used in forward

    The optimizer applies Muon (momentum + Newton-Schulz orthogonalization) to the
    direction component v. The magnitude component g is updated according to
    `magnitude_optim`.

    Arguments:
        params: Parameters for the optimizer.
        lr: Base learning rate (shared between AdamW and Muon due to 0.2 scaling).
        momentum: Momentum factor for Muon algorithm.
        nesterov: Whether to use Nesterov-style momentum in Muon.
        betas: Tuple of (beta1, beta2) for AdamW updates on g and 1D params.
        weight_decay: Decoupled weight decay coefficient (applied to magnitude g).
        magnitude_optim: Optimizer for the magnitude g. One of:
            - 'adam': Adam update with bias correction (default). Uses betas and adam_eps.
            - 'fixed': g is frozen at initialization; no buffers allocated.
            - 'sgd': vanilla SGD (g -= lr * grad_g).
            - 'signum': sign of EMA momentum (g -= lr * sign(m_g)) with beta1 smoothing.
            - 'lion': Lion update with hardcoded betas (0.9, 0.99).
        adam_eps: Small epsilon for AdamW denominator.
        ns_steps: Number of Newton-Schulz iterations for orthogonalization.
        backend: Backend for orthogonalization ('newtonschulz5', 'newtonschulz5_torch', or 'svd').
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        magnitude_optim: str = "adam",
        adam_eps: float = 1e-8,
        ns_steps: int = 5,
        backend: str = "newtonschulz5",
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
        if magnitude_optim not in MAGNITUDE_OPTIMS:
            raise ValueError(
                f"Invalid magnitude_optim: {magnitude_optim}. Choose from {sorted(MAGNITUDE_OPTIMS)}",
            )
        if backend not in ZEROPOWER_BACKENDS:
            raise ValueError(
                f"Invalid backend: {backend}. Choose from {list(ZEROPOWER_BACKENDS.keys())}",
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            betas=betas,
            weight_decay=weight_decay,
            magnitude_optim=magnitude_optim,
            adam_eps=adam_eps,
            ns_steps=ns_steps,
            backend=backend,
        )
        super().__init__(params, defaults)

        self._zeropower_fn = ZEROPOWER_BACKENDS[backend]

    def _init_state_2d(self, p: Tensor, state: dict, magnitude_optim: str = "adam") -> None:
        """Initialize weight norm state for a 2D parameter."""
        w_norm = p.data.norm(dim=1, keepdim=True)
        state["g"] = w_norm.clone()
        state["v_norm"] = w_norm.clone()
        state["m_v"] = torch.zeros_like(p.data)
        if magnitude_optim == "adam":
            state["m_g"] = torch.zeros_like(w_norm)
            state["v_g"] = torch.zeros_like(w_norm)
        elif magnitude_optim in ("signum", "lion"):
            state["m_g"] = torch.zeros_like(w_norm)
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
            magnitude_optim = group["magnitude_optim"]
            adam_eps = group["adam_eps"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    self._init_state_2d(p, state, magnitude_optim)

                state["step"] += 1
                step = state["step"]

                g = state["g"]
                v_norm = state["v_norm"]
                m_v = state["m_v"]
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
                    scale = 0.2 * chunk_size**0.5
                else:
                    update = self._zeropower_fn(update, steps=ns_steps)
                    scale = 0.2 * max(update.size(0), update.size(1)) ** 0.5
                v_new = v.add(update, alpha=-lr * scale)

                # Magnitude update on g
                if magnitude_optim == "adam":
                    m_g = state["m_g"]
                    v_g = state["v_g"]
                    beta1, beta2 = betas
                    m_g.mul_(beta1).add_(grad_g, alpha=1 - beta1)
                    v_g.mul_(beta2).addcmul_(grad_g, grad_g, value=1 - beta2)
                    bc1 = 1 - beta1**step
                    bc2 = 1 - beta2**step
                    g.addcdiv_(m_g / bc1, (v_g / bc2).sqrt().add_(adam_eps), value=-lr)
                elif magnitude_optim == "sgd":
                    g.add_(grad_g, alpha=-lr)
                elif magnitude_optim == "signum":
                    m_g = state["m_g"]
                    beta1 = betas[0]
                    m_g.mul_(beta1).add_(grad_g, alpha=1 - beta1)
                    g.add_(m_g.sign(), alpha=-lr)
                elif magnitude_optim == "lion":
                    m_g = state["m_g"]
                    beta1, beta2 = 0.9, 0.99
                    update_g = m_g.mul(beta1).add_(grad_g, alpha=1 - beta1)
                    g.add_(update_g.sign_(), alpha=-lr)
                    m_g.mul_(beta2).add_(grad_g, alpha=1 - beta2)

                # Fused: recompose W = g * v_new / ||v_new||, writes directly into p.data
                state["v_norm"] = _wn_recompose(p.data, g, v_new)
                if weight_decay != 0.0:
                    p.data.add_(W_old, alpha=-lr * weight_decay)
                    g.copy_(p.data.norm(dim=1, keepdim=True))

        return loss
