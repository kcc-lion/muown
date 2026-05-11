import json
import os
import re
import shutil

import torch

import utils

_CKPT_PREFIX = "ckpt_step_"
_CKPT_PAT = re.compile(rf"^{re.escape(_CKPT_PREFIX)}(\d+)$")


def _list_ckpt_dirs(base_dir: str) -> list[str]:
    """Return checkpoint directories sorted by step (ascending)."""
    if not os.path.isdir(base_dir):
        return []
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and _CKPT_PAT.match(d)]
    dirs.sort(key=lambda d: int(_CKPT_PAT.match(d).group(1)))
    return [os.path.join(base_dir, d) for d in dirs]


def _latest_ckpt_dir(ckpt_dir: str, **_) -> str | None:
    """Return latest checkpoint directory, or None."""
    dirs = _list_ckpt_dirs(ckpt_dir)
    return dirs[-1] if dirs else None


def save_checkpoint(step, model, engine, cfg, metrics, rank, job_idx=None):
    exp_dir = utils.get_exp_dir_path(cfg, job_idx)
    save_dir = os.path.join(exp_dir, f"{_CKPT_PREFIX}{step}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving checkpoint to {save_dir}")

    if rank == 0:
        # step index
        with open(os.path.join(save_dir, "step.txt"), "w") as f:
            f.write(str(step))

        # metrics
        metrics_path = os.path.join(exp_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(dict(metrics), f)

        # model
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_state.pth"))

        # schedulers
        for n, scheduler in engine.schedulers.items():
            torch.save(
                scheduler.state_dict() if scheduler else {},
                os.path.join(save_dir, f"scheduler_{n}_state.pth"),
            )

    # optimizers
    for n, optimizer in engine.optimizers.items():
        if getattr(optimizer, "rank_sharded", False):
            torch.save(
                optimizer.state_dict(),
                os.path.join(save_dir, f"optimizer_{n}_rank{rank}_state.pth"),
            )
        elif rank == 0:
            torch.save(optimizer.state_dict(), os.path.join(save_dir, f"optimizer_{n}_state.pth"))

    # Rolling cleanup: keep only the most recent max_checkpoints
    max_ckpts = getattr(cfg, "max_checkpoints", None)
    if rank == 0 and max_ckpts is not None:
        all_ckpts = _list_ckpt_dirs(exp_dir)
        to_delete = all_ckpts[:-max_ckpts] if len(all_ckpts) > max_ckpts else []
        for old_dir in to_delete:
            print(f"Removing old checkpoint: {old_dir}")
            shutil.rmtree(old_dir)


def resolve_resume_step(cfg) -> int:
    """Read the step from the latest (or specified) checkpoint without loading tensors."""
    resume_exp = cfg.resume_exp_name or cfg.exp_name
    base = os.path.join(cfg.out_dir, resume_exp)

    if cfg.resume_step is not None:
        ckpt_dir = os.path.join(base, f"ckpt_step_{cfg.resume_step}")
    else:
        ckpt_dir = _latest_ckpt_dir(base, prefix="ckpt_step_")

    if ckpt_dir is None or not os.path.isdir(ckpt_dir):
        raise ValueError(f"No checkpoint directory found in: {base}")

    with open(os.path.join(ckpt_dir, "step.txt"), "r") as f:
        return int(f.read().strip())


def load_checkpoint(cfg):
    """
    returns:
      {
      "step": 1200,
      "model_state": <state_dict>,
      "scheduler_adamw_state": <state_dict>,
      "optimizer_zero1adamw_state": <state_dict>,
      "optimizer_zero1adamw_rank0_state": <state_dict>,
      "optimizer_zero1adamw_rank1_state": <state_dict>,
      ...
    """
    if not cfg.resume:
        raise ValueError("No checkpoint to resume from.")

    # resume from a specified exp or from the same exp
    # notice that we can resume from `resume_exp_name`, but save to a different `exp_name`
    resume_exp = cfg.resume_exp_name or cfg.exp_name
    base = os.path.join(cfg.out_dir, resume_exp)

    if cfg.resume_step is not None:
        ckpt_dir = os.path.join(base, f"ckpt_step_{cfg.resume_step}")
    else:
        ckpt_dir = _latest_ckpt_dir(base, prefix="ckpt_step_")

    if ckpt_dir is None or not os.path.isdir(ckpt_dir):
        raise ValueError(f"No checkpoint directory found in: {base}")

    print(f"Resuming from {ckpt_dir}")
    out = {}

    # step
    with open(os.path.join(ckpt_dir, "step.txt"), "r") as f:
        out["step"] = int(f.read().strip())

    # load model, optimizers, schedulers
    for fn in os.listdir(ckpt_dir):
        if not fn.endswith(".pth"):
            continue

        path = os.path.join(ckpt_dir, fn)

        key = fn[:-4]  # drop .pth
        out[key] = torch.load(path, map_location="cpu")

    return out
