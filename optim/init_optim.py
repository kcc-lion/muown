"""Intialize optimizer and scheduler."""

import torch

from models import get_param_groups

from .lr_schedule import WSD, LinearCooldown, WarmupConstant, WarmupCosine


def intialize_optimizer(model, cfg):
    """
    Intialize an optimizer.
    NOTE: we pass weight_decay to optim, but it gets overwritten by the weight_decay in param_groups!
    """
    optimizers = {}

    if cfg.optim == "adamw":
        param_groups = get_param_groups(model, cfg.adamw_weight_decay)
        optimizers[cfg.optim] = torch.optim.AdamW(
            param_groups,
            lr=cfg.lr,
            betas=[cfg.adamw_beta1, cfg.adamw_beta2],
            weight_decay=cfg.adamw_weight_decay,
            fused=getattr(cfg, "fused_optim", True),
            eps=getattr(cfg, "eps", 1e-8),
        )

    elif cfg.optim == "sgd":
        param_groups = get_param_groups(model, cfg.weight_decay)
        optimizers[cfg.optim] = torch.optim.SGD(
            param_groups,
            lr=cfg.lr,
            momentum=cfg.beta1,
            dampening=cfg.dampening,
            weight_decay=cfg.adamw_weight_decay,
        )

    elif cfg.optim == "muon_torch":
        from torch.optim import Muon as MuonTorch

        from optim.muon import split_params_muon_adam

        muon_params, adam_params = split_params_muon_adam(model)

        assert cfg.adjust_lr == "match_adam", "MuonTorch only supports 'match_adam' adjust_lr"
        optimizers["muon"] = MuonTorch(
            muon_params,
            lr=cfg.lr,
            weight_decay=cfg.muon_weight_decay,
            momentum=cfg.muon_beta,
            nesterov=cfg.muon_nesterov,
            adjust_lr_fn="match_rms_adamw" if cfg.adjust_lr == "match_adam" else None,
            ns_steps=cfg.muon_ns_steps,
            eps=getattr(cfg, "eps", 1e-8),
        )
        optimizers["adamw"] = torch.optim.AdamW(
            adam_params,
            lr=cfg.lr,
            weight_decay=cfg.adamw_weight_decay,
            betas=(cfg.adamw_beta1, cfg.adamw_beta2),
            eps=getattr(cfg, "eps", 1e-8),
            fused=getattr(cfg, "fused_optim", True),
        )
    elif cfg.optim == "muown":
        from optim.muon import split_params_muon_adam
        from optim.muown import Muown

        muon_params, adam_params = split_params_muon_adam(model)
        assert cfg.adjust_lr == "match_adam", "Muown only supports 'match_adam' adjust_lr"
        optimizers["muown"] = Muown(
            muon_params,
            lr=cfg.lr,
            momentum=cfg.muon_beta,
            nesterov=cfg.muon_nesterov,
            # apply same betas as adamw to implicit magnitude params
            betas=(cfg.adamw_beta1, cfg.adamw_beta2),
            weight_decay=cfg.muon_weight_decay,
            ns_steps=cfg.muon_ns_steps,
            adam_eps=getattr(cfg, "eps", 1e-8),
            backend=getattr(cfg, "muown_backend", "newtonschulz5"),
        )
        optimizers["adamw"] = torch.optim.AdamW(
            adam_params,
            lr=cfg.lr,
            weight_decay=cfg.adamw_weight_decay,
            betas=(cfg.adamw_beta1, cfg.adamw_beta2),
            fused=getattr(cfg, "fused_optim", True),
            eps=getattr(cfg, "eps", 1e-8),
        )

    elif cfg.optim == "muown_dp":
        from optim.muon import split_params_muon_adam
        from optim.muown import MuownDP

        muon_params, adam_params = split_params_muon_adam(model)
        assert cfg.adjust_lr == "match_adam", "MuownDP only supports 'match_adam' adjust_lr"
        optimizers["muown_dp"] = MuownDP(
            muon_params,
            lr=cfg.lr,
            momentum=cfg.muon_beta,
            nesterov=cfg.muon_nesterov,
            betas=(cfg.adamw_beta1, cfg.adamw_beta2),
            weight_decay=cfg.muon_weight_decay,
            ns_steps=cfg.muon_ns_steps,
            adam_eps=getattr(cfg, "eps", 1e-8),
            backend=getattr(cfg, "muown_backend", "newtonschulz5"),
        )
        optimizers["adamw"] = torch.optim.AdamW(
            adam_params,
            lr=cfg.lr,
            weight_decay=cfg.adamw_weight_decay,
            betas=(cfg.adamw_beta1, cfg.adamw_beta2),
            fused=getattr(cfg, "fused_optim", True),
            eps=getattr(cfg, "eps", 1e-8),
        )

    elif cfg.optim == "normuon":
        from optim.muon import split_params_muon_adam
        from optim.normuown import SingleDeviceNorMuon

        muon_params, adam_params = split_params_muon_adam(model)
        optimizers["normuon"] = SingleDeviceNorMuon(
            muon_params,
            lr=cfg.lr,
            weight_decay=cfg.muon_weight_decay,
            momentum=cfg.muon_beta,
            beta2=cfg.muon_beta,
        )
        optimizers["adamw"] = torch.optim.AdamW(
            adam_params,
            lr=cfg.lr,
            weight_decay=cfg.adamw_weight_decay,
            betas=(cfg.adamw_beta1, cfg.adamw_beta2),
            fused=getattr(cfg, "fused_optim", True),
            eps=getattr(cfg, "eps", 1e-8),
        )

    elif cfg.optim in ("normuown", "normuown_ratio"):
        from optim.muon import split_params_muon_adam
        from optim.normuown import NorMuown

        muon_params, adam_params = split_params_muon_adam(model)
        assert cfg.adjust_lr == "match_adam", "NorMuown only supports 'match_adam' adjust_lr"
        # `normuown` uses standard Muon scaling (max(out, in) ** 0.5);
        # `normuown_ratio` uses NorMuon-style ratio scaling (max(1, out/in) ** 0.5).
        shape_scaling_type = "normuon" if cfg.optim == "normuown_ratio" else "muon"
        optimizers[cfg.optim] = NorMuown(
            muon_params,
            lr=cfg.lr,
            momentum=cfg.muon_beta,
            nesterov=cfg.muon_nesterov,
            betas=(cfg.adamw_beta1, cfg.adamw_beta2),
            weight_decay=cfg.muon_weight_decay,
            ns_steps=cfg.muon_ns_steps,
            adam_eps=getattr(cfg, "eps", 1e-8),
            backend=getattr(cfg, "muown_backend", "newtonschulz5"),
            use_normuon=True,
            normuon_beta2=cfg.muon_beta,
            shape_scaling_type=shape_scaling_type,
        )
        optimizers["adamw"] = torch.optim.AdamW(
            adam_params,
            lr=cfg.lr,
            weight_decay=cfg.adamw_weight_decay,
            betas=(cfg.adamw_beta1, cfg.adamw_beta2),
            fused=getattr(cfg, "fused_optim", True),
            eps=getattr(cfg, "eps", 1e-8),
        )

    elif cfg.optim == "muon_bench":
        # Plain Muon from `optim.muon`, used ONLY for runtime comparisons against
        # Muown. Other Muon experiments use `muon_torch` (torch.optim.Muon).
        from optim.muon import Muon, split_params_muon_adam

        muon_params, adam_params = split_params_muon_adam(model)
        assert cfg.adjust_lr == "match_adam", "Muon (bench) only supports 'match_adam' adjust_lr"
        optimizers["muon"] = Muon(
            muon_params,
            lr=cfg.lr,
            momentum=cfg.muon_beta,
            nesterov=cfg.muon_nesterov,
            weight_decay=cfg.muon_weight_decay,
            ns_steps=cfg.muon_ns_steps,
            backend=getattr(cfg, "muown_backend", "newtonschulz5"),
        )
        optimizers["adamw"] = torch.optim.AdamW(
            adam_params,
            lr=cfg.lr,
            weight_decay=cfg.adamw_weight_decay,
            betas=(cfg.adamw_beta1, cfg.adamw_beta2),
            fused=getattr(cfg, "fused_optim", True),
            eps=getattr(cfg, "eps", 1e-8),
        )

    elif cfg.optim == "muon_dp_bench":
        # Plain MuonDP from `optim.muon`, used ONLY for runtime comparisons against
        # MuownDP. Other Muon experiments use `muon_torch` (torch.optim.Muon).
        from optim.muon import MuonDP, split_params_muon_adam

        muon_params, adam_params = split_params_muon_adam(model)
        assert cfg.adjust_lr == "match_adam", "MuonDP (bench) only supports 'match_adam' adjust_lr"
        optimizers["muon"] = MuonDP(
            muon_params,
            lr=cfg.lr,
            momentum=cfg.muon_beta,
            nesterov=cfg.muon_nesterov,
            weight_decay=cfg.muon_weight_decay,
            ns_steps=cfg.muon_ns_steps,
            backend=getattr(cfg, "muown_backend", "newtonschulz5"),
        )
        optimizers["adamw"] = torch.optim.AdamW(
            adam_params,
            lr=cfg.lr,
            weight_decay=cfg.adamw_weight_decay,
            betas=(cfg.adamw_beta1, cfg.adamw_beta2),
            fused=getattr(cfg, "fused_optim", True),
            eps=getattr(cfg, "eps", 1e-8),
        )

    elif cfg.optim == "lion":
        from optim.lion import Lion

        param_groups = get_param_groups(model, cfg.muon_weight_decay)
        optimizers["lion"] = Lion(
            param_groups,
            lr=cfg.lr,
            betas=(cfg.adamw_beta1, cfg.adamw_beta2),
            weight_decay=cfg.muon_weight_decay,
        )

    elif cfg.optim == "soap":
        from optim.soap import SOAP

        param_groups = get_param_groups(model, cfg.adamw_weight_decay)
        optimizers["soap"] = SOAP(
            param_groups,
            lr=cfg.lr,
            betas=(cfg.adamw_beta1, cfg.adamw_beta2),
            shampoo_beta=getattr(cfg, "soap_shampoo_beta", -1),
            eps=getattr(cfg, "eps", 1e-8),
            weight_decay=cfg.muon_weight_decay,
            precondition_frequency=getattr(cfg, "soap_precondition_frequency", 10),
            max_precond_dim=getattr(cfg, "soap_max_precond_dim", 10000),
            merge_dims=getattr(cfg, "soap_merge_dims", False),
            precondition_1d=getattr(cfg, "soap_precondition_1d", False),
            normalize_grads=getattr(cfg, "soap_normalize_grads", False),
            correct_bias=getattr(cfg, "soap_correct_bias", True),
        )

    elif cfg.optim == "muown_magnitude_exp":
        from optim.muon import split_params_muon_adam
        from optim.muown import MuownMagnitudeExp

        muon_params, adam_params = split_params_muon_adam(model)
        assert cfg.adjust_lr == "match_adam", "MuownFixedMagnitude only supports 'match_adam' adjust_lr"
        optimizers["muown_magnitude_exp"] = MuownMagnitudeExp(
            muon_params,
            lr=cfg.lr,
            momentum=cfg.muon_beta,
            nesterov=cfg.muon_nesterov,
            betas=(cfg.adamw_beta1, cfg.adamw_beta2),
            weight_decay=cfg.muon_weight_decay,
            magnitude_optim=getattr(cfg, "muown_magnitude_optim", "fixed"),
            ns_steps=cfg.muon_ns_steps,
            adam_eps=getattr(cfg, "eps", 1e-8),
        )
        optimizers["adamw"] = torch.optim.AdamW(
            adam_params,
            lr=cfg.lr,
            weight_decay=cfg.adamw_weight_decay,
            betas=(cfg.adamw_beta1, cfg.adamw_beta2),
            fused=getattr(cfg, "fused_optim", True),
            eps=getattr(cfg, "eps", 1e-8),
        )
    else:
        raise NotImplementedError(f"Not implemented optim: {cfg.optim}.")

    return optimizers


def initialize_scheduler(optimizer, cfg):
    if cfg.scheduler is None:
        return None

    ## Number of warmup steps
    # either specified directly (int) or as a fraction of steps_budget (float)
    if getattr(cfg, "warmup_steps", None) is not None:
        warmup_steps = (
            cfg.warmup_steps
            if isinstance(cfg.warmup_steps, int)
            else int(cfg.warmup_steps * cfg.steps_budget)
        )

    ## Number of cooldown steps
    # either specified directly (int) or as a fraction of steps_budget (float)
    if getattr(cfg, "cooldown_steps", None) is not None:
        cooldown_steps = (
            cfg.cooldown_steps
            if isinstance(cfg.cooldown_steps, int)
            else int(cfg.cooldown_steps * cfg.steps_budget)
        )

    ## Final LR of the schedule
    # either specified directly via `lr_end` or as a fraction of top lr via `lr_end_pct`
    if getattr(cfg, "lr_end", None) is not None or getattr(cfg, "lr_end_pct", None) is not None:
        lr_end = cfg.lr_end if (cfg.lr_end is not None) else (cfg.lr_end_pct * cfg.lr)

    if cfg.scheduler == "warmup_cosine":
        scheduler = WarmupCosine(
            optimizer,
            lr_start=cfg.lr_start,
            lr_max=cfg.lr,
            lr_end=lr_end,
            warmup_steps=warmup_steps,
            T=cfg.steps_budget,
        )

    elif cfg.scheduler == "wsd":
        cooldown_start_step = cfg.steps_budget - cooldown_steps
        scheduler = WSD(
            optimizer,
            lr_start=cfg.lr_start,
            lr_max=cfg.lr,
            lr_end=lr_end,
            warmup_steps=warmup_steps,
            cooldown_start_step=cooldown_start_step,
            cooldown_steps=cooldown_steps,
        )

    elif cfg.scheduler == "warmup_constant":
        scheduler = WarmupConstant(
            optimizer,
            lr_start=cfg.lr_start,
            lr_max=cfg.lr,
            warmup_steps=warmup_steps,
        )

    elif cfg.scheduler == "linear_cooldown":
        cooldown_start_step = cfg.resume_step
        scheduler = LinearCooldown(
            optimizer,
            lr_max=cfg.lr,
            lr_end=lr_end,
            cooldown_start_step=cooldown_start_step,
            cooldown_steps=cooldown_steps,
        )

    else:
        raise NotImplementedError(f"Not implemented scheduler: {cfg.scheduler}.")

    return scheduler
