"""
Plain Muon optimizer (for runtime benchmarking against Muown).

This file provides a plain Muon implementation (`Muon`) and its data-parallel
variant (`MuonDP`) that mirror the structure of `optim.muown.Muown` /
`optim.muown.MuownDP` as closely as possible, minus the internal weight-norm
(g, v) parameterization. They use the same Newton-Schulz backends, the same
QKV chunking, the same shape-based Muon scaling, and the same decoupled weight
decay so that step-time comparisons isolate the cost of the weight-norm
bookkeeping in Muown.

NOTE: These `Muon` / `MuonDP` implementations are ONLY used for runtime
comparisons against Muown. All other Muon experiments in the repository use
the official torch implementation (selected via `cfg.optim == "muon_torch"`,
which instantiates `torch.optim.Muon`).

Also exports `split_params_muon_adam`, the shared utility that splits a model's
parameters into the Muon group (2D matrices, excluding embeddings / lm_head)
and the Adam group (1D params, embeddings, lm_head).
"""

import os
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim.optimizer import Optimizer

from utils import print_master

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


class Muon(Optimizer):
    """
    Plain Muon (momentum + Newton-Schulz orthogonalization).

    NOTE: This implementation exists purely for runtime comparisons against
    `optim.muown.Muown`. It mirrors Muown's structure (same NS backends, same
    QKV chunking, same shape-based scaling, same decoupled weight decay) with
    the weight-norm (g, v) bookkeeping removed, so the step-time diff isolates
    Muown's weight-norm overhead. All other Muon experiments in this repo use
    `torch.optim.Muon` via `cfg.optim == "muon_torch"`.

    Arguments:
        params: Parameters for the optimizer.
        lr: Learning rate.
        momentum: Momentum factor for Muon algorithm.
        nesterov: Whether to use Nesterov-style momentum in Muon.
        weight_decay: Decoupled weight decay coefficient.
        ns_steps: Number of Newton-Schulz iterations for orthogonalization.
        backend: Backend for orthogonalization ('newtonschulz5', 'newtonschulz5_torch', or 'svd').
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        backend: str = "newtonschulz5",
        **kwargs,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
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
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            backend=backend,
        )
        super().__init__(params, defaults)

        self._zeropower_fn = ZEROPOWER_BACKENDS[backend]

    def _init_state_2d(self, p: Tensor, state: dict) -> None:
        """Initialize momentum buffer for a 2D parameter."""
        state["m"] = torch.zeros_like(p.data)
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
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    self._init_state_2d(p, state)

                state["step"] += 1

                m = state["m"]

                # Muon update: momentum + orthogonalization
                m.mul_(momentum).add_(grad)
                if nesterov:
                    update = grad.add(m, alpha=momentum)
                else:
                    update = m.clone()

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

                if weight_decay != 0.0:
                    p.data.mul_(1 - lr * weight_decay)
                p.data.add_(update, alpha=-lr * scale)

        return loss


def _param_to_complexity(p: Tensor) -> int:
    """Approximate NS complexity for load-balanced sorting."""
    m, n = p.shape[0], p[0].numel()
    return 2 * (m**2) * n + m**3


class MuonDP(Muon):
    """
    Data-parallel Muon: distributes Newton-Schulz orthogonalization across ranks.

    Mirrors `optim.muown.MuownDP`'s structure (sort by NS complexity, block-per-
    WORLD_SIZE processing, async all-gather) with the weight-norm bookkeeping
    removed. See the module docstring and `Muon` for why this exists: it is ONLY
    used for runtime comparisons against MuownDP. All other Muon experiments
    use `torch.optim.Muon` via `cfg.optim == "muon_torch"`.

    Parameters are sorted by NS complexity (descending) and processed in blocks
    of WORLD_SIZE. Each rank computes the Muon update for its assigned
    parameters, then all_gather synchronizes the updated weights.

    Comms: one all-gather per block -> ~#params/WORLD_SIZE comms.
    Space: O(largest_param); O(WORLD_SIZE * largest_param) transient during all_gather.
    """

    rank_sharded = True

    def __init__(self, params, **kwargs):
        if not isinstance(params, list):
            params = list(params)

        if not dist.is_initialized():
            raise ValueError("Using MuonDP in a non-distributed run.")

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
            weight_decay = group["weight_decay"]
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

                    m = state["m"]

                    m.mul_(momentum).add_(grad)
                    if nesterov:
                        update = grad.add(m, alpha=momentum)
                    else:
                        update = m.clone()

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

                    if weight_decay != 0.0:
                        p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(update, alpha=-lr * scale)

                handle = dist.all_gather(
                    # output container for all-gathered weights
                    params_pad[block_start : block_start + WORLD_SIZE],
                    # local input tensor, containing updated weights
                    params_pad[block_start + RANK],
                    async_op=True,
                )
                allgather_handles.append(handle)

        for handle in allgather_handles:
            handle.wait()

        return loss


def split_params_muon_adam(model):
    """Split parameters:
    - Muon: all matrix params (ndim ≥ 2) except embeddings
    - Adam: 1D params, all embeddings
    """

    muon_params, adam_params = [], []
    muon_infos, adam_infos = [], []  # for logging purposes

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Assign embeddings to Adam (wmt, criteo)
        if "embed" in n.lower():
            adam_params.append(p)
            adam_infos.append(f"{n} (ndim={p.ndim})")
        elif "lm_head" in n.lower():
            adam_params.append(p)
            adam_infos.append(f"{n} (ndim={p.ndim})")
        elif p.ndim >= 2:
            muon_params.append(p)
            muon_infos.append(f"{n} (ndim={p.ndim})")
        else:
            adam_params.append(p)
            adam_infos.append(f"{n} (ndim={p.ndim})")

    print_master("Muon params:\n\t" + "\n\t".join(muon_infos))
    print_master("Adam params:\n\t" + "\n\t".join(adam_infos))

    return muon_params, adam_params
