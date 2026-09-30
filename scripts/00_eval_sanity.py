#!/usr/bin/env python3
"""WP0: letter-logprob sanity on 8 dummy MCQs whose correct letter is always B.

Loads the from-scratch checkpoint written by scripts/00_overfit_chapter.py.
Scores each item by the log-probability of the letter continuation. Pass iff
accuracy beats chance (0.25). Does not generate, does not use a critic, and
does not update weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sclm.checkpoint import load_from_scratch_checkpoint
from sclm.config import load_yaml
from sclm.dummy_chapter import dummy_chapter, dummy_items, text_sha256
from sclm.scoring import score_mcqs
from sclm.tokenizer import GPT2Tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="WP0 letter-logprob sanity check")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ckpt_path = args.checkpoint or (ROOT / cfg["wp0"]["checkpoint"])
    if not ckpt_path.is_file():
        raise SystemExit(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run scripts/00_overfit_chapter.py first (default dummy chapter)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_from_scratch_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()

    if payload.get("chapter_source") != "dummy":
        raise SystemExit(
            "This sanity check scores the 8 dummy MCQs and requires the default "
            "dummy-chapter checkpoint. A --chapter overfit proves loss only."
        )
    expected_hash = text_sha256(dummy_chapter())
    if payload.get("chapter_sha256") != expected_hash:
        raise SystemExit(
            "Checkpoint chapter hash does not match the built-in dummy chapter. "
            "Refusing to treat a different text as the WP0 sanity set."
        )

    items = dummy_items()
    if len(items) != 8 or any(item["answer"] != "B" for item in items):
        raise SystemExit("Dummy MCQ set must be 8 items whose correct letter is B")

    tokenizer = GPT2Tokenizer()
    result = score_mcqs(model, tokenizer, items)

    print("WP0 EVAL SANITY")
    print(f"from_scratch: {str(bool(payload.get('from_scratch'))).lower()}")
    print(f"variant: {payload.get('variant')}")
    print(f"params: {payload.get('num_parameters')}")
    print(f"n: {result['n']}")
    for row in result["rows"]:
        lps = " ".join(f"{k}={row['logprobs'][k]:.3f}" for k in ("A", "B", "C", "D"))
        flag = "ok" if row["correct"] else "miss"
        print(f"  {row['id']}: pred={row['pred']} gold={row['answer']} {flag}  {lps}")
    print(f"correct: {result['correct']}")
    print(f"accuracy: {result['accuracy']:.4f}")
    print(f"chance: {result['chance']:.4f}")
    print(f"beats_chance: {str(result['beats_chance']).lower()}")
    print("PASS" if result["beats_chance"] else "FAIL")
    return 0 if result["beats_chance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
