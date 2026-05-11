import time

import torch
import wandb

@torch.no_grad()
def compute_weight_norm_stats(model):
    """Compute norm statistics for all 2D (matrix-shaped) weight parameters.

    Norms are computed along the axis matching the optimizer's weight-norm
    decomposition: rows when out_features <= in_features, columns otherwise.
    """
    skip = {"embed", "lm_head"}
    stats = {}
    for name, param in model.named_parameters():
        if param.ndim != 2 or any(s in name for s in skip):
            continue

        if name.endswith("w_qkv.weight"):
            base = name.removesuffix("w_qkv.weight")
            chunks = param.data.float().chunk(3, dim=0)
            sub_weights = [(f"{base}w_q.weight", chunks[0]),
                           (f"{base}w_k.weight", chunks[1]),
                           (f"{base}w_v.weight", chunks[2])]
        else:
            sub_weights = [(name, param.data.float())]

        for sub_name, W in sub_weights:
            norm_dim = 1
            label = "row"

            norms = W.norm(dim=norm_dim)
            D = W / norms.clamp(min=1e-8).unsqueeze(norm_dim)

            C = D @ D.T
            p = norms / norms.max().clamp(min=1e-8)
            PCP = p.unsqueeze(1) * C * p.unsqueeze(0)

            prefix = f"weight_norms/{sub_name}"
            stats[f"{prefix}/max_{label}_norm"] = norms.max().item()
            stats[f"{prefix}/spectral_norm"] = torch.linalg.matrix_norm(W, ord=2).item()
            C.fill_diagonal_(0)
            stats[f"{prefix}/lambda_max_PCP"] = torch.linalg.eigvalsh(PCP)[-1].item()
    return stats


def _maybe_log_weight_norms(model, cfg, master_process):
    """Log weight norm statistics to wandb if log_weight_norms is enabled."""
    if getattr(cfg, "log_weight_norms", False) and master_process and cfg.use_wandb:
        wandb.log(compute_weight_norm_stats(model), commit=False)


@torch.no_grad()
def compute_attn_logit_stats(model, dataloader, seq_len, device, ctx):
    """Compute max pre-softmax attention logit per layer over the entire dataloader."""
    model.eval()
    n_layers = len(model.layers)
    global_max = [float("-inf")] * n_layers

    for layer in model.layers:
        layer.attn._log_attn_stats = True

    for batch in dataloader:
        inputs = batch["input_ids"][:, :seq_len].to(device)
        with ctx:
            model(inputs)
        for i, layer in enumerate(model.layers):
            global_max[i] = max(global_max[i], layer.attn._attn_logit_max)

    for layer in model.layers:
        layer.attn._log_attn_stats = False

    stats = {f"attn_logits/layer_{i}/max": global_max[i] for i in range(n_layers)}
    return stats


class StepTimer:
    """Tracks wall-clock step timing with warmup and per-interval averaging.

    Averages all step times accumulated between consecutive ``flush()`` calls,
    so the value logged to wandb reflects the mean over the logging interval
    rather than a single noisy sample.
    """

    def __init__(self, warmup_steps: int = 50, tokens_per_step: int = 0):
        self.warmup_steps = warmup_steps
        self.tokens_per_step = tokens_per_step
        self._t0: float | None = None
        self._step_count = 0
        self._interval_times: list[float] = []
        self._all_times: list[float] = []

    def step(self) -> None:
        """Call once per optimizer step (i.e. every ``grad_accumulation_steps`` micro-steps)."""
        now = time.perf_counter()
        self._step_count += 1
        if self._t0 is not None and self._step_count > self.warmup_steps:
            dt_ms = (now - self._t0) * 1000.0
            self._interval_times.append(dt_ms)
            self._all_times.append(dt_ms)
        self._t0 = now

    def reset_after_eval(self) -> None:
        """Reset the clock after eval so eval latency is excluded from the next step."""
        self._t0 = time.perf_counter()

    def flush(self) -> dict[str, float] | None:
        """Return averaged metrics since the last flush, then clear the interval buffer."""
        if not self._interval_times:
            return None
        avg_ms = sum(self._interval_times) / len(self._interval_times)
        result: dict[str, float] = {"train/step_time_ms": avg_ms}
        if self.tokens_per_step:
            result["train/tokens_per_sec"] = self.tokens_per_step / (avg_ms / 1000.0)
        self._interval_times.clear()
        return result

    def summary_dict(self) -> dict[str, float] | None:
        """Return overall timing statistics as a dict (for wandb summary)."""
        if not self._all_times:
            return None
        s = sorted(self._all_times)
        n = len(s)
        d: dict[str, float] = {
            "timing/median_step_time_ms": s[n // 2],
            "timing/mean_step_time_ms": sum(s) / n,
            "timing/p25_step_time_ms": s[n // 4],
            "timing/p75_step_time_ms": s[3 * n // 4],
            "timing/num_measured_steps": n,
        }
        if self.tokens_per_step:
            d["timing/median_tokens_per_sec"] = self.tokens_per_step / (s[n // 2] / 1000.0)
            d["timing/mean_tokens_per_sec"] = self.tokens_per_step / ((sum(s) / n) / 1000.0)
        return d

    def summary(self) -> str | None:
        """Return a formatted summary string for end-of-training logging."""
        if not self._all_times:
            return None
        s = sorted(self._all_times)
        n = len(s)
        tok_str = (
            f"\n  tokens/sec (median): {self.tokens_per_step / (s[n // 2] / 1000.0):.0f}"
            if self.tokens_per_step
            else ""
        )
        return (
            f"=== Step Timing ({n} steps after {self.warmup_steps}-step warmup) ===\n"
            f"  median: {s[n // 2]:.2f} ms/step\n"
            f"  mean:   {sum(s) / n:.2f} ms/step\n"
            f"  p25:    {s[n // 4]:.2f} ms  |  p75: {s[3 * n // 4]:.2f} ms"
            f"{tok_str}"
        )


def _maybe_log_attn_stats(model, validloader, cfg, engine, master_process):
    """Log attention logit statistics to wandb if enabled."""
    if getattr(cfg, "log_attn_logits", False) and master_process and cfg.use_wandb:
        stats = compute_attn_logit_stats(
            model,
            validloader,
            cfg.seq_len,
            engine.device,
            engine.ctx,
        )
        wandb.log(stats, commit=False)


class MemoryTracker:
    """Tracks CUDA memory usage with warmup, periodic logging, and end-of-run summary.

    Resets peak memory stats after ``warmup_steps`` so that transient
    torch.compile / CUDA-cache warmup peaks are excluded from the
    steady-state measurement.
    """

    def __init__(self, device: torch.device, warmup_steps: int = 50):
        self.device = device
        self.warmup_steps = warmup_steps
        self._step_count = 0
        self._warmup_done = False
        self._peak_allocated_mb: float = 0.0
        self._peak_reserved_mb: float = 0.0

    def step(self) -> None:
        """Call once per optimizer step, in lockstep with StepTimer."""
        self._step_count += 1
        if not self._warmup_done and self._step_count >= self.warmup_steps:
            torch.cuda.reset_peak_memory_stats(self.device)
            self._warmup_done = True

    def flush(self) -> dict[str, float] | None:
        """Return current memory snapshot for periodic wandb logging."""
        if not self._warmup_done:
            return None
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        peak_alloc = torch.cuda.max_memory_allocated(self.device)
        peak_res = torch.cuda.max_memory_reserved(self.device)
        self._peak_allocated_mb = peak_alloc / 2**20
        self._peak_reserved_mb = peak_res / 2**20
        return {
            "memory/allocated_mb": allocated / 2**20,
            "memory/reserved_mb": reserved / 2**20,
            "memory/peak_allocated_mb": self._peak_allocated_mb,
            "memory/peak_reserved_mb": self._peak_reserved_mb,
        }

    def summary_dict(self) -> dict[str, float] | None:
        """Return summary metrics for wandb.summary (post-warmup peaks)."""
        if not self._warmup_done:
            return None
        peak_alloc = torch.cuda.max_memory_allocated(self.device)
        peak_res = torch.cuda.max_memory_reserved(self.device)
        return {
            "memory/peak_allocated_mb": peak_alloc / 2**20,
            "memory/peak_reserved_mb": peak_res / 2**20,
        }

    def summary(self) -> str | None:
        """Return a formatted summary string for end-of-training logging."""
        if not self._warmup_done:
            return None
        peak_alloc = torch.cuda.max_memory_allocated(self.device)
        peak_res = torch.cuda.max_memory_reserved(self.device)
        return (
            f"=== GPU Memory (after {self.warmup_steps}-step warmup) ===\n"
            f"  peak allocated: {peak_alloc / 2**20:.0f} MB\n"
            f"  peak reserved:  {peak_res / 2**20:.0f} MB"
        )
