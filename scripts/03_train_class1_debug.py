#!/usr/bin/env python3
"""WP3: train the debug model on Class 1 packed blocks only.

From-scratch next-token loss. Corpus BPE. No critic, no RL, no pretrained
LM weights, no Classes 2–6, no target model, no curriculum matrix.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sclm.checkpoint import save_checkpoint
from sclm.class1_heldout import CLASS1_HELDOUT
from sclm.config import load_yaml, model_config_from_dict
from sclm.dummy_chapter import dummy_chapter
from sclm.locks import ALLOW_CRITIC_IN_LOOP, ALLOW_PRETRAINED_LM_WEIGHTS, ALLOW_RL
from sclm.model import GPT
from sclm.overlap import shares_ngram
from sclm.scoring import render_mcq, score_mcqs
from sclm.tokenizer import load_tokenizer


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def assert_heldout_is_clean(root: Path) -> None:
    blobs = [dummy_chapter()]
    for path in sorted((root / "data" / "clean" / "class_1").glob("chapter_*.txt")):
        blobs.append(path.read_text(encoding="utf-8"))
    for item in CLASS1_HELDOUT:
        rendered = render_mcq(item, include_answer=True)
        for blob in blobs:
            if shares_ngram(blob, rendered):
                raise SystemExit(f"Held-out item {item['id']} shares a 12-gram with training text")
    letters = [item["answer"] for item in CLASS1_HELDOUT]
    if len(CLASS1_HELDOUT) != 8:
        raise SystemExit("WP3 held-out set must be 8 items")
    if sorted(letters.count(letter) for letter in "ABCD") != [2, 2, 2, 2]:
        raise SystemExit(f"Gold letters are not balanced: {letters}")


def main() -> int:
    if ALLOW_PRETRAINED_LM_WEIGHTS or ALLOW_RL or ALLOW_CRITIC_IN_LOOP:
        raise SystemExit("Protocol locks violated")
    cfg = load_yaml(ROOT / "configs" / "default.yaml")
    if cfg["curriculum"]["p_new"] is not None:
        raise SystemExit("p_new must stay null")
    wp3 = cfg["wp3"]
    if int(wp3["class"]) != 1 or wp3["model_variant"] != "debug":
        raise SystemExit("WP3 trains only the debug model on Class 1")
    if cfg["tokenizer"]["name"] != "corpus_bpe":
        raise SystemExit("WP3 requires the corpus BPE")

    assert_heldout_is_clean(ROOT)
    set_seed(int(cfg["seed"]))
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(cfg, ROOT)
    model_cfg = model_config_from_dict(cfg, "debug")
    if model_cfg.vocab_size != tokenizer.vocab_size or model_cfg.block_size != 512:
        raise SystemExit("Debug model is not on the corpus BPE 512-token setup")
    model = GPT(model_cfg).to(device)
    if not model.from_scratch:
        raise SystemExit("Model was not marked from_scratch")

    blocks = np.load(ROOT / "data" / "packed" / "class_1.npy")
    if blocks.ndim != 2 or blocks.shape[1] != 512:
        raise SystemExit(f"Expected Class 1 blocks [n, 512], got {blocks.shape}")
    data = torch.from_numpy(blocks.astype(np.int64))
    if int(data.max()) >= model_cfg.vocab_size:
        raise SystemExit("Packed ids are outside the corpus vocab")
    x = data[:, :-1].contiguous().to(device)
    y = data[:, 1:].contiguous().to(device)

    train = cfg["train"]
    steps = int(wp3["steps"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train["lr"]),
        betas=tuple(train["betas"]),  # type: ignore[arg-type]
        eps=float(train["eps"]),
        weight_decay=float(train["weight_decay"]),
    )
    grad_clip = float(train["grad_clip"])

    @torch.no_grad()
    def full_loss() -> float:
        model.eval()
        _logits, loss = model(x, y)
        model.train()
        return float(loss.item())

    loss_before = full_loss()
    print(
        f"params {model.num_parameters()} blocks {blocks.shape[0]} "
        f"loss_before {loss_before:.4f}",
        flush=True,
    )
    model.train()
    t0 = time.time()
    last_loss = None
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        _logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        last_loss = float(loss.item())
        if step == 1 or step % 10 == 0 or step == steps:
            print(f"step {step} train_loss {last_loss:.4f}", flush=True)
    loss_after = full_loss()
    token_steps = steps * int(y.numel())
    elapsed = time.time() - t0

    model.eval()
    scored = score_mcqs(model, tokenizer, CLASS1_HELDOUT)
    print(
        f"loss_after {loss_after:.4f} token_steps {token_steps} "
        f"mcq {scored['correct']}/{scored['n']} accuracy {scored['accuracy']:.3f} "
        f"beats_chance {scored['beats_chance']}",
        flush=True,
    )
    for row in scored["rows"]:
        print(f"item {row['id']} gold {row['answer']} pred {row['pred']}", flush=True)

    out_dir = ROOT / "artifacts" / "wp3"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        out_dir / "class1_debug.pt",
        model,
        extra={
            "wp": 3,
            "class": 1,
            "tokenizer": "corpus_bpe",
            "steps": steps,
        },
    )
    report = {
        "variant": "debug",
        "params": model.num_parameters(),
        "tokenizer": "corpus_bpe",
        "vocab_size": tokenizer.vocab_size,
        "block_size": model_cfg.block_size,
        "class": 1,
        "packed_blocks": int(blocks.shape[0]),
        "packed_tokens": int(blocks.size),
        "steps": steps,
        "token_steps": token_steps,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "train_loss_last": last_loss,
        "mcq_correct": scored["correct"],
        "mcq_n": scored["n"],
        "mcq_accuracy": scored["accuracy"],
        "chance": scored["chance"],
        "beats_chance": scored["beats_chance"],
        "seconds": elapsed,
        "p_new": cfg["curriculum"]["p_new"],
    }
    (out_dir / "class1_debug.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_report(report)
    return 0


def write_report(report: dict) -> None:
    path = ROOT / "reports" / "wp3_class1.md"
    lines = [
        "# WP3 Class 1 debug",
        "",
        "Debug model only, from scratch, next-token loss, corpus BPE.",
        "Classes 2–6 were not trained. The 12-layer model was not trained.",
        "`p_new` is null. No critic.",
        "",
        f"Parameters: {report['params']}",
        f"Tokenizer: {report['tokenizer']} vocab {report['vocab_size']}",
        f"Packed Class 1 blocks: {report['packed_blocks']} ({report['packed_tokens']} tokens)",
        f"Optimizer steps: {report['steps']}",
        f"Token steps (next-token predictions): {report['token_steps']}",
        f"Full-batch loss before: {report['loss_before']:.4f}",
        f"Full-batch loss after: {report['loss_after']:.4f}",
        f"Last train loss: {report['train_loss_last']:.4f}",
        f"Held-out MCQ accuracy: {report['mcq_correct']}/{report['mcq_n']} = {report['mcq_accuracy']:.3f}",
        f"Chance: {report['chance']}",
        f"Beats chance: {report['beats_chance']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
