"""Pretrain a Transformer on language modeling with Hydra config."""

import dataclasses
import json
import os
from collections import defaultdict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import utils
import wandb
from checkpoint_utils import resolve_resume_step, save_checkpoint
from data import get_dataloaders
from engine import TorchEngine
from models import construct_model
from torch_utils import destroy_ddp, pytorch_setup
from utils import print_master

from logging_utils import (
    MemoryTracker,
    StepTimer,
    _maybe_log_attn_stats,
    _maybe_log_weight_norms,
)


@hydra.main(version_base="1.3", config_path="config", config_name="dev/debug")
def main(cfg: DictConfig):
    # Disable struct mode to allow attribute access for missing keys (getattr with defaults)
    OmegaConf.set_struct(cfg, False)
    # Resolve interpolations
    OmegaConf.resolve(cfg)

    rank, world_size, device, master_process = pytorch_setup(cfg)

    if cfg.resume and cfg.resume_step is None:
        cfg.resume_step = resolve_resume_step(cfg)
        print_master(f"Auto-resolved resume_step={cfg.resume_step} from checkpoint")

    # Save resolved config to run directory (on master only)
    if master_process and hasattr(cfg, "out_dir"):
        os.makedirs(cfg.out_dir, exist_ok=True)
        config_path = os.path.join(cfg.out_dir, "config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(cfg, f)
        print_master(f"Config saved to: {config_path}")

    if master_process:
        utils.maybe_make_dir(cfg)

    if cfg.use_wandb and master_process:
        utils.init_wandb(cfg)

    # Dataset
    trainloader, validloader = get_dataloaders(cfg)

    # Model
    model, model_cfg = construct_model(cfg)

    # Save resolved model config to run directory (on master only)
    if master_process and hasattr(cfg, "out_dir"):
        model_cfg_path = os.path.join(cfg.out_dir, "model_config.json")
        if dataclasses.is_dataclass(model_cfg):
            with open(model_cfg_path, "w") as f:
                json.dump(dataclasses.asdict(model_cfg), f, indent=2)
        elif hasattr(model_cfg, "to_json_file"):
            model_cfg.to_json_file(model_cfg_path)
        else:
            with open(model_cfg_path, "w") as f:
                json.dump(str(model_cfg), f, indent=2)
        print_master(f"Model config saved to: {model_cfg_path}")

    # Engine
    engine = TorchEngine(model, cfg, device)

    step_start = cfg.resume_step if cfg.resume else 0
     
    # If we are just cooling down, we set budget = resume + cooldown
    steps_budget = (
        cfg.steps_budget
        if cfg.scheduler != "linear_cooldown"
        else cfg.resume_step + engine.scheduler.cooldown_steps
    )
    micro_step_budget = steps_budget * cfg.grad_accumulation_steps
    if micro_step_budget > len(trainloader):
        raise ValueError("trainloader too short!")

    _save_at_steps = set(getattr(cfg, "save_at_steps", []) or [])

    micro_step_start = step_start * cfg.grad_accumulation_steps
    print_master(
        f"=== Start Training from step: {step_start}/{steps_budget}, micro_step: {micro_step_start}/{micro_step_budget} ===",
    )

    # Bookkeeping
    metrics = defaultdict(list)

    # Progress bar (only on master)
    tokens_per_step = cfg.seq_len * cfg.micro_batch_size * cfg.grad_accumulation_steps * world_size
    total_tokens = steps_budget * tokens_per_step
    pbar = tqdm(
        total=steps_budget - step_start,
        initial=0,
        desc="Training",
        disable=not master_process,
        unit="step",
        dynamic_ncols=True,
    )
    last_step = step_start

    # Eval before training (skip on resume — checkpoint step was already evaluated)
    if cfg.eval and not cfg.resume:
        print_master("Evaluating before training")
        valid_loss = engine.eval(validloader)
        _maybe_log_weight_norms(model, cfg, master_process)
        _maybe_log_attn_stats(model, validloader, cfg, engine, master_process)
        metrics["valid/loss"].append(valid_loss)
        if master_process:
            metrics["step"].append(step_start)
            metrics["micro_step"].append(micro_step_start)
            metrics["tokens"].append(0)
            for n, optim in engine.optimizers.items():
                metrics[f"{n}_lr"].append(optim.param_groups[0]["lr"])
            utils.log(cfg, metrics)

    # Step timing & memory tracking (shared warmup period)
    warmup_steps = getattr(cfg, "timing_warmup_steps", 50)
    step_timer = StepTimer(
        warmup_steps=warmup_steps,
        tokens_per_step=tokens_per_step,
    )
    mem_tracker = MemoryTracker(device=device, warmup_steps=warmup_steps)

    # Training
    for micro_step, micro_batch in enumerate(trainloader, micro_step_start + 1):
        step = micro_step // cfg.grad_accumulation_steps
        is_step = micro_step % cfg.grad_accumulation_steps == 0
        if step > steps_budget and is_step:
            break

        # Train
        train_loss = engine.step(micro_batch)

        # Update progress bar and record step timing
        if is_step and step > last_step:
            step_timer.step()
            mem_tracker.step()

            tokens_so_far = step * tokens_per_step
            pbar.update(step - last_step)
            pbar.set_postfix(
                {
                    "loss": f"{train_loss.item():.4f}",
                    "tokens": f"{tokens_so_far / 1e9:.2f}B/{total_tokens / 1e9:.1f}B",
                },
            )
            last_step = step

        # Eval
        valid_loss = None
        if cfg.eval and step % cfg.eval_every_steps == 0 and is_step:
            pbar.set_description("Evaluating")
            valid_loss = engine.eval(validloader)
            _maybe_log_weight_norms(model, cfg, master_process)
            _maybe_log_attn_stats(model, validloader, cfg, engine, master_process)
            pbar.set_description("Training")
            step_timer.reset_after_eval()
        metrics["valid/loss"].append(valid_loss)

        # Log
        if master_process and (step % cfg.log_every_steps == 0 or step == step_start + 1 or step >= steps_budget) and is_step:
            metrics["step"].append(step)
            metrics["micro_step"].append(micro_step)
            metrics["tokens"].append(step * tokens_per_step)
            metrics["train/loss"].append(train_loss.item())
            timing = step_timer.flush()
            if timing:
                for k, v in timing.items():
                    metrics[k].append(v)
            mem = mem_tracker.flush()
            if mem:
                for k, v in mem.items():
                    metrics[k].append(v)
            for n, optim in engine.optimizers.items():
                metrics[f"{n}_lr"].append(optim.param_groups[0]["lr"])
            utils.log(cfg, metrics)

        # Checkpoint
        if cfg.save_intermediate_checkpoints and is_step and (
            step % cfg.save_every_steps == 0 or step in _save_at_steps
        ):
            save_checkpoint(step, model, engine, cfg, metrics, rank)

    pbar.close()

    # Eval at the end
    if getattr(cfg, "eval_when_finished", True):
        print_master("Evaluating on validation set")
        valid_loss = engine.eval(validloader)
        _maybe_log_weight_norms(model, cfg, master_process)
        _maybe_log_attn_stats(model, validloader, cfg, engine, master_process)
        metrics["valid/loss"].append(valid_loss)
        if master_process:
            utils.log(cfg, metrics)

    # Step timing & memory summary
    timing_summary = step_timer.summary()
    if timing_summary and master_process:
        print_master(timing_summary)
        if cfg.use_wandb:
            wandb.summary.update(step_timer.summary_dict())

    mem_summary = mem_tracker.summary()
    if mem_summary and master_process:
        print_master(mem_summary)
        if cfg.use_wandb:
            wandb.summary.update(mem_tracker.summary_dict())

    # End of training: log and save checkpoint
    print_master("=== Training Completed! ===")
    if cfg.save_last_checkpoint:
        save_checkpoint(step, model, engine, cfg, metrics, rank)

    # DDP slaughtering
    destroy_ddp()


if __name__ == "__main__":
    main()
