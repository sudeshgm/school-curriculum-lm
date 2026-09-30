"""Load configs/default.yaml and build a model config for one variant."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sclm.locks import MATCH_TOKENS, TARGET_N_EMBD, TARGET_N_HEAD, TARGET_N_LAYER


@dataclass
class ModelConfig:
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    vocab_size: int = 50257
    dropout: float = 0.0
    bias: bool = True


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} is not a mapping")
    return cfg


def model_config_from_dict(cfg: dict[str, Any], variant: str) -> ModelConfig:
    if variant not in ("debug", "target"):
        raise ValueError(f"Unknown model variant {variant!r}; expected 'debug' or 'target'")
    model = cfg["model"]
    spec = model[variant]
    mc = ModelConfig(
        n_layer=int(spec["n_layer"]),
        n_head=int(spec["n_head"]),
        n_embd=int(spec["n_embd"]),
        block_size=int(model["block_size"]),
        vocab_size=int(model["vocab_size"]),
        dropout=float(model["dropout"]),
        bias=bool(model["bias"]),
    )
    if mc.n_embd % mc.n_head != 0:
        raise ValueError(f"n_embd ({mc.n_embd}) must be divisible by n_head ({mc.n_head})")
    if variant == "target":
        if (mc.n_layer, mc.n_embd, mc.n_head) != (
            TARGET_N_LAYER,
            TARGET_N_EMBD,
            TARGET_N_HEAD,
        ):
            raise ValueError(
                "Target architecture does not match the locked 12-layer / 768-dim / 12-head model"
            )
    if int(cfg["decontamination"]["match_tokens"]) != MATCH_TOKENS:
        raise ValueError(
            f"decontamination.match_tokens must stay {MATCH_TOKENS} (12-gram overlap filter)"
        )
    return mc
