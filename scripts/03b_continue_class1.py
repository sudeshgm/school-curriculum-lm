#!/usr/bin/env python3
"""WP3b: baseline, continue the Class 1 debug checkpoint, then rescore.

Does not train Classes 2–6 or the 12-layer model. No critic. No pretrained
LM weights. p_new stays null. The checkpoint is not a new random init.
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

from sclm.checkpoint import load_from_scratch_checkpoint, save_checkpoint
from sclm.class1_heldout import CLASS1_FORMAT_EXAMPLES, CLASS1_HELDOUT
from sclm.config import load_yaml, model_config_from_dict
from sclm.dummy_chapter import dummy_chapter
from sclm.locks import ALLOW_CRITIC_IN_LOOP, ALLOW_PRETRAINED_LM_WEIGHTS, ALLOW_RL
from sclm.model import GPT
from sclm.overlap import shares_ngram
from sclm.scoring import render_mcq, score_mcqs
from sclm.tokenizer import load_tokenizer

CKPT = ROOT / "artifacts" / "wp3" / "class1_debug.pt"
EXTRA_STEPS = 200
LOSS_STOP = 2.0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def format_prefix() -> str:
    return "\n\n".join(render_mcq(item, include_answer=True) for item in CLASS1_FORMAT_EXAMPLES) + "\n\n"


def assert_items_are_clean(root: Path) -> None:
    scored_ids = {item["id"] for item in CLASS1_HELDOUT}
    example_ids = {item["id"] for item in CLASS1_FORMAT_EXAMPLES}
    if scored_ids & example_ids:
        raise SystemExit("Format examples overlap the scored set")
    if len(CLASS1_FORMAT_EXAMPLES) != 4 or len(CLASS1_HELDOUT) != 8:
        raise SystemExit("WP3b needs 4 format examples and 8 scored items")
    blobs = [dummy_chapter()]
    blobs.extend(render_mcq(item, include_answer=True) for item in CLASS1_HELDOUT)
    for path in sorted((root / "data" / "clean" / "class_1").glob("chapter_*.txt")):
        blobs.append(path.read_text(encoding="utf-8"))
    for item in CLASS1_FORMAT_EXAMPLES:
        rendered = render_mcq(item, include_answer=True)
        for blob in blobs:
            if shares_ngram(blob, rendered):
                raise SystemExit(f"Format example {item['id']} shares a 12-gram with held-out or training text")


def rows_text(scored: dict) -> list[str]:
    lines = []
    for row in scored["rows"]:
        lines.append(f"{row['id']} gold {row['answer']} pred {row['pred']}")
        print(f"item {row['id']} gold {row['answer']} pred {row['pred']}", flush=True)
    return lines


def main() -> int:
    if ALLOW_PRETRAINED_LM_WEIGHTS or ALLOW_RL or ALLOW_CRITIC_IN_LOOP:
        raise SystemExit("Protocol locks violated")
    if not CKPT.is_file():
        raise SystemExit(f"Class 1 checkpoint missing: {CKPT}")
    cfg = load_yaml(ROOT / "configs" / "default.yaml")
    if cfg["curriculum"]["p_new"] is not None:
        raise SystemExit("p_new must stay null")
    if cfg["tokenizer"]["name"] != "corpus_bpe":
        raise SystemExit("WP3b requires the corpus BPE")
    assert_items_are_clean(ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    tokenizer = load_tokenizer(cfg, ROOT)
    prefix = format_prefix()

    set_seed(int(cfg["seed"]))
    untrained = GPT(model_config_from_dict(cfg, "debug")).to(device)
    untrained.eval()
    base = score_mcqs(untrained, tokenizer, CLASS1_HELDOUT)
    print(
        f"untrained {base['correct']}/{base['n']} accuracy {base['accuracy']:.3f}",
        flush=True,
    )
    print("untrained_items", flush=True)
    base_rows = rows_text(base)
    del untrained

    model, payload = load_from_scratch_checkpoint(CKPT, map_location=device)
    if payload.get("class") != 1 or payload.get("tokenizer") != "corpus_bpe":
        raise SystemExit("Checkpoint is not the Class 1 corpus-BPE debug run")
    if model.config.n_layer != 4 or model.config.vocab_size != 4096:
        raise SystemExit("Checkpoint is not the debug 4096-vocab model")
    model.to(device)

    blocks = np.load(ROOT / "data" / "packed" / "class_1.npy")
    data = torch.from_numpy(blocks.astype(np.int64))
    x = data[:, :-1].contiguous().to(device)
    y = data[:, 1:].contiguous().to(device)
    train = cfg["train"]
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

    loss_start = full_loss()
    print(f"resume_loss {loss_start:.4f} prior_steps {payload.get('steps')}", flush=True)
    model.train()
    t0 = time.time()
    taken = 0
    last_loss = loss_start
    stopped_reason = "step_cap"
    for step in range(1, EXTRA_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        _logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        taken = step
        last_loss = float(loss.item())
        if step == 1 or step % 10 == 0 or last_loss < LOSS_STOP:
            print(f"extra_step {step} train_loss {last_loss:.4f}", flush=True)
        if last_loss < LOSS_STOP:
            stopped_reason = "loss_under_2"
            break
    loss_after = full_loss()
    if loss_after < LOSS_STOP:
        stopped_reason = "loss_under_2"
    elapsed = time.time() - t0
    print(f"loss_after {loss_after:.4f} extra_steps {taken} reason {stopped_reason}", flush=True)

    model.eval()
    zero = score_mcqs(model, tokenizer, CLASS1_HELDOUT)
    few = score_mcqs(model, tokenizer, CLASS1_HELDOUT, prompt_prefix=prefix)
    print(
        f"zero_shot {zero['correct']}/{zero['n']} accuracy {zero['accuracy']:.3f}",
        flush=True,
    )
    print("zero_shot_items", flush=True)
    zero_rows = rows_text(zero)
    print(
        f"few_shot {few['correct']}/{few['n']} accuracy {few['accuracy']:.3f}",
        flush=True,
    )
    print("few_shot_items", flush=True)
    few_rows = rows_text(few)

    out = ROOT / "artifacts" / "wp3"
    save_checkpoint(
        out / "class1_debug_wp3b.pt",
        model,
        extra={
            "wp": "3b",
            "class": 1,
            "tokenizer": "corpus_bpe",
            "resumed_from_steps": payload.get("steps"),
            "extra_steps": taken,
        },
    )
    report = {
        "tokenizer": "corpus_bpe",
        "vocab_size": 4096,
        "p_new": cfg["curriculum"]["p_new"],
        "resumed_from_steps": payload.get("steps"),
        "extra_steps": taken,
        "total_steps": int(payload.get("steps") or 0) + taken,
        "token_steps_extra": taken * int(y.numel()),
        "loss_at_resume": loss_start,
        "loss_after": loss_after,
        "train_loss_last": last_loss,
        "stop_reason": stopped_reason,
        "untrained_accuracy": base["accuracy"],
        "untrained_correct": base["correct"],
        "zero_shot_accuracy": zero["accuracy"],
        "zero_shot_correct": zero["correct"],
        "few_shot_accuracy": few["accuracy"],
        "few_shot_correct": few["correct"],
        "chance": 0.25,
        "untrained_rows": base_rows,
        "zero_shot_rows": zero_rows,
        "few_shot_rows": few_rows,
        "seconds": elapsed,
        "optimizer_moments": "restarted",
    }
    (out / "wp3b.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_report(report)
    return 0


def write_report(report: dict) -> None:
    path = ROOT / "reports" / "wp3b_class1.md"
    lines = [
        "# WP3b Class 1 debug",
        "",
        "Continued the step-40 Class 1 checkpoint. Not a new random init.",
        "Classes 2–6 were not trained. The 12-layer model was not trained.",
        "Corpus BPE, vocab 4096. `p_new` is null. No critic. No pretrained weights.",
        "AdamW moments restarted because the checkpoint did not store them.",
        "",
        f"Resume loss: {report['loss_at_resume']:.4f}",
        f"Extra steps: {report['extra_steps']} (stop: {report['stop_reason']})",
        f"Total optimizer steps: {report['total_steps']}",
        f"Extra token steps: {report['token_steps_extra']}",
        f"Full-batch loss after: {report['loss_after']:.4f}",
        f"Last train loss: {report['train_loss_last']:.4f}",
        "",
        f"Untrained baseline: {report['untrained_correct']}/8 = {report['untrained_accuracy']:.3f}",
        f"Zero-shot after training: {report['zero_shot_correct']}/8 = {report['zero_shot_accuracy']:.3f}",
        f"Four-example in-context: {report['few_shot_correct']}/8 = {report['few_shot_accuracy']:.3f}",
        "Chance: 0.25",
        "",
        "Untrained gold vs predicted:",
        *[f"- {row}" for row in report["untrained_rows"]],
        "",
        "Zero-shot gold vs predicted:",
        *[f"- {row}" for row in report["zero_shot_rows"]],
        "",
        "In-context gold vs predicted:",
        *[f"- {row}" for row in report["few_shot_rows"]],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
