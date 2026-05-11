import math
import os
import shutil
from collections import namedtuple
from itertools import product

import yaml

import wandb

try:
    from omegaconf import DictConfig, OmegaConf

    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False


def cfg_to_dict(cfg):
    """Convert config to dict, supporting both namedtuple and OmegaConf."""
    if HAS_OMEGACONF and isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    elif hasattr(cfg, "_asdict"):
        return cfg._asdict()
    else:
        return dict(cfg)


def load_config(path, job_idx=None):
    """
    Parse a yaml file and return the correspondent config as a namedtuple.
    If the config files has multiple entries, returns the one corresponding to job_idx.
    """

    with open(path, "r") as file:
        config_dict = yaml.safe_load(file)
    Config = namedtuple("Config", config_dict.keys())

    if job_idx is None:
        cfg = config_dict
        sweep_size = 1

    else:
        keys = list(config_dict.keys())
        values = [val if isinstance(val, list) else [val] for val in config_dict.values()]
        combinations = list(product(*values))

        sweep_size = len(combinations)
        if job_idx >= sweep_size:
            raise ValueError("job_idx exceeds the total number of hyperparam combinations.")

        combination = combinations[job_idx]
        cfg = {keys[i]: combination[i] for i in range(len(keys))}

    return Config(**cfg), sweep_size


def init_wandb(cfg):
    """Initalizes a wandb run, optionally resuming an existing one via wandb_run_id."""
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    os.environ["WANDB_SILENT"] = "true"
    wandb_kwargs = dict(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        dir=cfg.wandb_dir,
        config=cfg_to_dict(cfg),
    )

    wandb_run_id = getattr(cfg, "wandb_run_id", None)
    if wandb_run_id:
        wandb_kwargs["id"] = wandb_run_id
        wandb_kwargs["resume"] = "allow"

    wandb.init(**wandb_kwargs)
    wandb.define_metric("*", step_metric="step")

def get_exp_dir_path(cfg, job_idx=None):
    """Build a exp_dir path from config. It supports job arrays."""
    exp_dir = os.path.join(cfg.out_dir, cfg.exp_name)
    if job_idx is not None:  # subfolder for each job in the sweep
        exp_dir = os.path.join(exp_dir, f"job_idx_{job_idx}")
    return exp_dir


def maybe_make_dir(cfg, job_idx=None):
    """Creates an experiment directory if checkpointing is enabled"""
    if not cfg.save_intermediate_checkpoints and not cfg.save_last_checkpoint:
        return
    if cfg.resume and cfg.resume_exp_name is None:  # if resuming from the same exp
        return

    exp_dir = get_exp_dir_path(cfg, job_idx)

    if os.path.exists(exp_dir):
        if not cfg.over_write:
            raise ValueError(f"Found existing exp_dir at {exp_dir}.")
        print(f"Removing experiment dir: {exp_dir}")
        shutil.rmtree(exp_dir)

    print(f"Creating experiment directory: {exp_dir}")
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.yaml"), "w") as file:
        yaml.dump(cfg_to_dict(cfg), file, default_flow_style=False)


def log(cfg, metrics):
    """Print metrics, log them on wandb."""
    if cfg.print_progress:
        msg = " | ".join(
            f"{k}: {float(v):.3e}"
            if isinstance(v, (int, float))
            else f"{k}: {v[-1]:.3e}"
            if isinstance(v[-1], float)
            else f"{k}: {v[-1]}"
            for k, v in metrics.items()
        )
        print(msg)

    if cfg.use_wandb:
        log_dict = {k: v[-1] for k, v in metrics.items() if v[-1] is not None}
        wandb.log(log_dict)


def print_master(msg):
    """Prints only in master process if using multiple GPUs."""
    rank = os.environ.get("RANK", -1)  # global rank
    ddp = int(rank) != -1
    master_process = (not ddp) or (int(rank) == 0)
    if master_process:
        print(msg)
