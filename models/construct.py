import math
from fractions import Fraction

import torch
import wandb


def construct_model(cfg):
    """Initalize a model from config. Counts parameters."""

    # Transformer++
    if cfg.model == "transformer":
        from .transformer import ModelConfig, Transformer

        model_cfg = ModelConfig(
            vocab_size=cfg.vocab_size,
            dim=cfg.d_model,
            expand=float(Fraction(cfg.expand)),
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            rmsnorm_eps=1e-6,
            mlp=cfg.mlp_class,
            seq_len=cfg.seq_len,
            tie_embeddings=cfg.tie_embeddings,
        )
        model = Transformer(model_cfg)

    # Pythia
    elif cfg.model.startswith("pythia"):
        from transformers import AutoConfig, AutoModelForCausalLM

        model_cfg = AutoConfig.from_pretrained(f"EleutherAI/{cfg.model}")
        model = AutoModelForCausalLM.from_config(model_cfg)  # NOTE: vocab_size=50304 here!
        model.init_weights()  # explict init, since I am not sure it is done in 'from_config'
    elif cfg.model.lower().startswith("qwen"):
        from transformers import AutoModelForCausalLM, AutoConfig
        model_cfg = AutoConfig.from_pretrained(f"Qwen/{cfg.model}")
        # Optional override of RoPE base frequency. Qwen2-0.5B ships with
        # rope_theta=1_000_000, raised from 10,000 only in the long-context
        # phase that extended pre-training from 4K to 32K tokens (Qwen2
        # report, sec. 3.2). Setting ++rope_theta=10000 from Hydra reverts
        # to the value used for the bulk of Qwen2 pre-training, which
        # matches our short-context regime.
        #
        # NOTE: transformers >=5 stores this inside `rope_parameters` (a dict
        # consumed by the rotary module). Setting the legacy flat
        # `rope_theta` attribute is silently ignored by the model, so we mutate
        # `rope_parameters` and fall back to the flat attr only on older
        # versions that still expose it.
        rope_theta = cfg.get("rope_theta", None) if hasattr(cfg, "get") else getattr(cfg, "rope_theta", None)
        if rope_theta is not None:
            if hasattr(model_cfg, "rope_parameters"):
                rp = dict(model_cfg.rope_parameters or {"rope_type": "default"})
                old = rp.get("rope_theta")
                rp["rope_theta"] = float(rope_theta)
                model_cfg.rope_parameters = rp
            else:
                old = getattr(model_cfg, "rope_theta", None)
                model_cfg.rope_theta = float(rope_theta)
            print(f"Overriding rope_theta: {old} -> {rope_theta}")
        model = AutoModelForCausalLM.from_config(model_cfg)
        # Force fp32 master weights. The published Qwen2 HF JSON sets
        # `dtype: bfloat16`, which `from_config` honors and constructs the
        # model directly in bf16. Our engine does mixed precision via
        # `torch.amp.autocast` and assumes master weights live in fp32; if
        # they're already in bf16, autocast is a no-op, the optimizer state
        # is bf16, and weight updates are quantized to the bf16 grid.
        model = model.float()
    else:
        raise NotImplementedError(f"Not implemented model: {cfg.model}.")

    if hasattr(model, "count_params"):
        n_params = model.count_params(non_embedding=False)
        n_params_no_embed = model.count_params(non_embedding=True)
    else:
        # Fallback for HF models (no count_params method): sum parameter numels
        # and subtract input (and untied output) embedding weights.
        n_params = sum(p.numel() for p in model.parameters())
        n_params_no_embed = n_params
        in_emb = model.get_input_embeddings()
        if in_emb is not None and hasattr(in_emb, "weight"):
            n_params_no_embed -= in_emb.weight.numel()
            out_emb = model.get_output_embeddings()
            if (
                out_emb is not None
                and hasattr(out_emb, "weight")
                and out_emb.weight is not in_emb.weight  # skip when tied
            ):
                n_params_no_embed -= out_emb.weight.numel()

    print(f"Number of parameters: {n_params:_}")
    print(f"Number of non-embedding parameters: {n_params_no_embed:_}")

    # Log master-parameter dtypes so a regression to bf16 (or any unintended
    # precision change in the construction path) shows up as a wandb config
    # diff.
    master_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    print(f"Master parameter dtypes: {master_dtypes}")
    if wandb.run is not None:
        wandb.log({"n_params": n_params, "n_params_no_embed": n_params_no_embed})
        wandb.config.update(
            {
                "model_master_dtypes": master_dtypes,
                "model_construction_path": (
                    "transformer" if cfg.model == "transformer"
                    else "hf_from_config_pythia" if cfg.model.startswith("pythia")
                    else "hf_from_pretrained_qwen" if cfg.model.lower().startswith("qwen")
                    else "unknown"
                ),
            },
            allow_val_change=True,
        )

    return model, model_cfg


def get_param_groups(model, weight_decay):
    """
    Create param groups for a Transformer model.
    Bias and normalization layers are excluded from weight decay.
    """

    # filter out parameters that do not require grad
    named_param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    names = named_param_dict.keys()

    # special param names
    norm = [n for n in names if "norm" in n]
    bias = [n for n in names if "bias" in n and n not in norm]
    embedlm = [
        n for n in names if ("embed" in n or "lm_head" in n) and n not in norm and n not in bias
    ]

    # special params
    special_param_names = norm + bias + embedlm
    special_params = [p for n, p in named_param_dict.items() if n in special_param_names]

    # all the ohers params
    other_param_names = [n for n in names if n not in special_param_names]
    other_params = [p for n, p in named_param_dict.items() if n in other_param_names]

    # assemble param grosups
    param_groups = [
        dict(
            params=special_params,
            weight_decay=0.0,
        ),
        dict(
            params=other_params,
            weight_decay=weight_decay,
        ),
    ]

    # # sanity check
    # print("norm:\n\t" + "\n\t".join(norm))
    # print("bias:\n\t" + "\n\t".join(bias))
    # print("embedlm:\n\t" + "\n\t".join(embedlm))
    # print("other_param_names:\n\t" + "\n\t".join(other_param_names))

    return param_groups
