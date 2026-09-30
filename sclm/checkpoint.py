"""From-scratch checkpoint save/load.

A checkpoint must set from_scratch=True. This loader refuses anything else,
including a file that merely contains raw weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sclm.config import ModelConfig
from sclm.model import GPT


def save_checkpoint(path: str | Path, model: GPT, extra: dict[str, Any]) -> None:
    if not getattr(model, "from_scratch", False):
        raise RuntimeError("Refusing to save a model that is not marked from_scratch")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "model_config": {
            "n_layer": model.config.n_layer,
            "n_head": model.config.n_head,
            "n_embd": model.config.n_embd,
            "block_size": model.config.block_size,
            "vocab_size": model.config.vocab_size,
            "dropout": model.config.dropout,
            "bias": model.config.bias,
        },
        "from_scratch": True,
        **extra,
    }
    torch.save(payload, path)


def load_from_scratch_checkpoint(path: str | Path, map_location: str = "cpu") -> tuple[GPT, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or not payload.get("from_scratch", False):
        raise RuntimeError(
            "Refusing to load a checkpoint that is not marked from_scratch. "
            "Pretrained LM weights are not allowed."
        )
    cfg = ModelConfig(**payload["model_config"])
    model = GPT(cfg)
    model.load_state_dict(payload["state_dict"])
    return model, payload
