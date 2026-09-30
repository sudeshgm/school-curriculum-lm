#!/usr/bin/env python3
"""WP0: overfit one chapter for 400 steps. Loss on a fixed batch must fall.

Default text is the built-in dummy chapter (not NCERT, not the scored MCQ
bank). Pass --chapter to train on a local text file instead. The learner is
constructed from random weights. Pretrained LM weights are never loaded.

No critic, no teacher LLM, no RL.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sclm.checkpoint import save_checkpoint
from sclm.config import load_yaml, model_config_from_dict
from sclm.dummy_chapter import dummy_chapter, text_sha256
from sclm.locks import ALLOW_PRETRAINED_LM_WEIGHTS, ALLOW_RL
from sclm.model import GPT
from sclm.tokenizer import GPT2Tokenizer


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_chapter(path: str | None) -> tuple[str, str]:
    if path is None:
        return dummy_chapter(), "dummy"
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        raise SystemExit(f"Chapter file is empty: {path}")
    return text, "file"


def encode_repeated(tokenizer: GPT2Tokenizer, text: str, min_tokens: int) -> torch.Tensor:
    ids = tokenizer.encode(text)
    if not ids:
        raise SystemExit("Chapter tokenized to zero tokens")
    while len(ids) < min_tokens:
        ids = ids + ids
    return torch.tensor(ids, dtype=torch.long)


def make_batch(
    data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(data.numel())
    if n <= seq_len:
        raise RuntimeError("Not enough tokens to form a batch")
    starts = torch.randint(0, n - seq_len, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in starts])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def batch_loss(model: GPT, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    _logits, loss = model(x, y)
    model.train()
    return float(loss.item())


def main() -> int:
    if ALLOW_PRETRAINED_LM_WEIGHTS or ALLOW_RL:
        raise SystemExit("Protocol locks violated: pretrained weights and RL stay off")

    parser = argparse.ArgumentParser(description="Overfit one chapter for WP0")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--chapter", type=str, default=None, help="Optional UTF-8 chapter file")
    parser.add_argument("--variant", type=str, default=None, choices=["debug", "target"])
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    wp0 = cfg["wp0"]
    train = cfg["train"]
    variant = args.variant or wp0["model_variant"]
    steps = int(args.steps if args.steps is not None else wp0["overfit_steps"])
    if steps < 1:
        raise SystemExit("--steps must be >= 1")

    seed = int(cfg["seed"])
    set_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = model_config_from_dict(cfg, variant)
    # Random init. There is no pretrained load path.
    model = GPT(model_cfg).to(device)
    if not model.from_scratch:
        raise SystemExit("Model was not marked from_scratch")

    tokenizer = GPT2Tokenizer()
    if tokenizer.vocab_size != model_cfg.vocab_size:
        raise SystemExit(
            f"Tokenizer vocab {tokenizer.vocab_size} != config vocab {model_cfg.vocab_size}"
        )

    text, source = load_chapter(args.chapter)
    seq_len = int(wp0["sequence_length"])
    batch_size = int(wp0["batch_size"])
    data = encode_repeated(tokenizer, text, min_tokens=seq_len + 2)

    lr = float(wp0["lr"])
    weight_decay = float(wp0["weight_decay"])
    betas = tuple(train["betas"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,  # type: ignore[arg-type]
        eps=float(train["eps"]),
        weight_decay=weight_decay,
    )
    grad_clip = float(train["grad_clip"])

    set_seed(seed)
    fixed_x, fixed_y = make_batch(data, batch_size, seq_len, device)
    loss_before = batch_loss(model, fixed_x, fixed_y)

    model.train()
    losses: list[float] = []
    log_every = int(wp0["log_every"])
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = make_batch(data, batch_size, seq_len, device)
        _logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        value = float(loss.item())
        losses.append(value)
        if step == 1 or step % log_every == 0 or step == steps:
            print(f"step {step:4d}/{steps}  train_loss {value:.4f}", flush=True)

    loss_after = batch_loss(model, fixed_x, fixed_y)
    elapsed = time.time() - t0
    fell = loss_after < loss_before
    params = model.num_parameters()

    ckpt_path = ROOT / wp0["checkpoint"]
    log_path = ROOT / wp0["loss_log"]
    extra = {
        "variant": variant,
        "steps": steps,
        "seed": seed,
        "chapter_source": source,
        "chapter_sha256": text_sha256(text),
        "loss_before": loss_before,
        "loss_after": loss_after,
        "loss_fell": fell,
        "train_losses": losses,
        "num_parameters": params,
        "lr": lr,
        "sequence_length": seq_len,
        "batch_size": batch_size,
    }
    save_checkpoint(ckpt_path, model, extra)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in extra.items() if k != "train_losses"}
    summary["train_loss_first"] = losses[0]
    summary["train_loss_last"] = losses[-1]
    summary["seconds"] = elapsed
    log_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("")
    print("WP0 OVERFIT ACCEPTANCE")
    print(f"variant: {variant}")
    print(f"from_scratch: true")
    print(f"params: {params}")
    print(f"steps: {steps}")
    print(f"chapter_source: {source}")
    print(f"fixed_batch_loss_before: {loss_before:.4f}")
    print(f"fixed_batch_loss_after: {loss_after:.4f}")
    print(f"train_loss_first: {losses[0]:.4f}")
    print(f"train_loss_last: {losses[-1]:.4f}")
    print(f"loss_fell: {str(fell).lower()}")
    print(f"seconds: {elapsed:.1f}")
    print(f"checkpoint: {ckpt_path}")
    print("PASS" if fell else "FAIL")
    return 0 if fell else 1


if __name__ == "__main__":
    raise SystemExit(main())
