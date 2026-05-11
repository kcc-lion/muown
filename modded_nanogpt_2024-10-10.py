"""Taken and adapted from https://github.com/KellerJordan/modded-nanogpt/blob/993c1bd071b98909ed27ea3bc5f4da3b932b8c41/records/track_1_short/2024-10-10_Muon/train_gpt2.py"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import glob
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Callable, Optional, Tuple
# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                if g.size(0) == 3 * g.size(1): # split grouped QKV parameters
                    g = torch.cat([zeropower_backend(g1, steps=group['backend_steps']) for g1 in g.split(g.size(1))])
                    scale = g.size(1)**0.5
                else:
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    scale = max(g.size(0), g.size(1))**0.5 # scale to have update.square().mean() == 1
                p.data.add_(g, alpha=-lr * scale)

# -----------------------------------------------------------------------------
# Muown optimizer


@torch.compile
def _wn_pre_ns(W: torch.Tensor, g: torch.Tensor, v_norm: torch.Tensor, grad_W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
def _wn_recompose(W: torch.Tensor, g: torch.Tensor, v_new: torch.Tensor) -> torch.Tensor:
    """Fused recompose W[:] = g * v_new / ||v_new||, writing directly into W."""
    v_norm_new = v_new.norm(dim=1, keepdim=True)
    W.copy_(g * (v_new / v_norm_new))
    return v_norm_new


class Muown(torch.optim.Optimizer):
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
        betas: tuple[float, float] = (0.9, 0.95),
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
        if backend not in zeropower_backends:
            raise ValueError(
                f"Invalid backend: {backend}. Choose from {list(zeropower_backends.keys())}",
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

        self._zeropower_fn = zeropower_backends[backend]

    def _init_state_2d(self, p: torch.Tensor, state: dict) -> None:
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

        return loss



# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = (1 / (2 * config.n_layer)**0.5)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50257
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device) # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams; paths are resolved relative to DATA_DIR env var
    input_bin : str = 'fineweb10B/fineweb_train_*.bin'
    input_val_bin : str = 'fineweb10B/fineweb_val_*.bin'
    # optimization hyperparams
    batch_size : int = 8*64 # batch size, in sequences, across all devices
    device_batch_size : int = 64 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 6200 # number of iterations to run
    learning_rate : float = 0.0036
    warmup_iters : int = 0
    warmdown_iters : int = 1800 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # model hyperparams
    n_embd : int = 768
    # optimizer selection: 'muon' or 'muown'
    optimizer : str = 'muon'
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
    logdir : str = os.environ.get('LOGDIR', '')
args = Hyperparameters()
assert args.logdir, "LOGDIR environment variable must be set"

# Resolve data paths against DATA_DIR.
data_dir = os.environ.get('DATA_DIR', '')
assert data_dir, "DATA_DIR environment variable must be set (base directory containing the data shards)"
args.input_bin = os.path.join(data_dir, args.input_bin)
args.input_val_bin = os.path.join(data_dir, args.input_val_bin)
if 'OPTIMIZER' in os.environ:
    args.optimizer = os.environ['OPTIMIZER']
if 'LR' in os.environ:
    args.learning_rate = float(os.environ['LR'])
if 'N_EMBD' in os.environ:
    args.n_embd = int(os.environ['N_EMBD'])
# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.
if master_process:
    print(f"config: optimizer={args.optimizer} lr={args.learning_rate} n_embd={args.n_embd}")

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# init the model from scratch
num_vocab = 50257
n_embd = args.n_embd
n_head = n_embd // 64 # head_dim = 64 in original 2024-10-10 config
assert n_embd % 64 == 0, f"n_embd={n_embd} must be divisible by 64 (head_dim)"
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=n_head, n_embd=n_embd))
model = model.cuda()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# init the optimizer(s)
optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=True)
if args.optimizer == 'muown':
    optimizer2 = Muown(raw_model.transformer.h.parameters(), lr=args.learning_rate, momentum=0.95)
else:
    optimizer2 = Muon(raw_model.transformer.h.parameters(), lr=0.1*args.learning_rate, momentum=0.95)
optimizers = [optimizer1, optimizer2]
# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# begin logging
if master_process:
    import csv
    logdir = args.logdir.rstrip('/') + '/'
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, 'log.txt')
    val_loss_csv = os.path.join(logdir, 'val_losses.csv')
    # Anonymize user-identifying path prefixes in the config dump so the log
    # is safe to share: DATA_DIR and LOGDIR are replaced with placeholders.
    def _anonymize(value):
        if not isinstance(value, str):
            return value
        if data_dir and value.startswith(data_dir):
            return '<DATA_DIR>' + value[len(data_dir):]
        if args.logdir and value.startswith(args.logdir):
            return '<LOGDIR>' + value[len(args.logdir):]
        return value
    anonymized_config = {k: _anonymize(v) for k, v in vars(args).items()}
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        # log the final resolved config (with paths anonymized)
        f.write('Resolved config:\n')
        for k, v in anonymized_config.items():
            f.write(f'  {k} = {v!r}\n')
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')
    with open(val_loss_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'val_loss', 'training_time_ms'])

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with torch.no_grad(): # of course, we'd like to use ctx here too, but that creates a torch.compile error for some reason
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
            with open(val_loss_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, f'{val_loss:.6f}', f'{training_time_ms:.0f}'])
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        #torch.save(log, os.path.join(logdir, f'state_step{step:06d}.pt'))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    summary_path = os.path.join(logdir, 'summary.csv')
    fields = ['optimizer', 'learning_rate', 'n_embd', 'n_layer', 'n_head',
              'num_iterations', 'batch_size', 'val_loss', 'training_time_ms']
    values = [args.optimizer, args.learning_rate, args.n_embd, 12, n_head,
              args.num_iterations, args.batch_size, f'{val_loss:.6f}', f'{training_time_ms:.0f}']
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerow(values)
    print(f"Summary written to {summary_path}")

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()