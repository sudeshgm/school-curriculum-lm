#!/usr/bin/env python3
"""WP2: train a corpus-native BPE and pack chapters into 512-token blocks.

Reads data/clean/class_{k}/chapter_*.txt only. Prelims are not included.
Does not train a language model, does not load pretrained LM weights, and
does not build an exam bank.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sclm.corpus_bpe import train_corpus_bpe
from sclm.locks import ALLOW_PRETRAINED_LM_WEIGHTS
from sclm.pack import pack_chapters

CLEAN = ROOT / "data" / "clean"
TOKENIZER_PATH = ROOT / "data" / "tokenizer" / "corpus_bpe.json"
PACKED = ROOT / "data" / "packed"
REPORT = ROOT / "reports" / "wp2_token_table.md"


def chapter_files(class_dir: Path) -> list[Path]:
    files = sorted(class_dir.glob("chapter_*.txt"))
    if not files:
        raise SystemExit(f"no chapter files in {class_dir}")
    return files


def main() -> int:
    if ALLOW_PRETRAINED_LM_WEIGHTS:
        raise SystemExit("protocol lock violated: pretrained LM weights stay off")
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    wp2 = cfg["wp2"]
    vocab_target = int(wp2["vocab_size"])
    block_size = int(wp2["block_size"])
    min_frequency = int(wp2["min_frequency"])
    if block_size != 512:
        raise SystemExit("WP2 block_size is 512")

    classes: list[tuple[int, list[tuple[str, str]]]] = []
    train_texts: list[str] = []
    for class_no in range(1, 7):
        rows = []
        for path in chapter_files(CLEAN / f"class_{class_no}"):
            text = path.read_text(encoding="utf-8")
            chapter_id = path.stem.replace("chapter_", "")
            rows.append((chapter_id, text))
            train_texts.append(text)
        classes.append((class_no, rows))

    tokenizer = train_corpus_bpe(train_texts, vocab_size=vocab_target, min_frequency=min_frequency)
    for text in train_texts:
        if tokenizer.decode(tokenizer.encode(text)) != text:
            raise SystemExit("BPE did not round-trip a training chapter")
    tokenizer.save(TOKENIZER_PATH)

    PACKED.mkdir(parents=True, exist_ok=True)
    index_lines = ["class\tblock\tchapter\ttokens"]
    summary = []
    for class_no, rows in classes:
        encoded = [(chapter_id, tokenizer.encode(text)) for chapter_id, text in rows]
        blocks, index_rows, tail, n_tokens = pack_chapters(encoded, block_size)
        chars = sum(len(text) for _chapter_id, text in rows)
        array = np.asarray(blocks, dtype=np.int32)
        if n_tokens >= block_size:
            if array.ndim != 2 or array.shape[1] != block_size:
                raise SystemExit(f"class {class_no} blocks are not {block_size} wide")
        np.save(PACKED / f"class_{class_no}.npy", array)
        for block, chapter_id, n_from in index_rows:
            index_lines.append(f"{class_no}\t{block}\t{chapter_id}\t{n_from}")
        summary.append(
            {
                "class": class_no,
                "chapters": len(rows),
                "chars": chars,
                "tokens": n_tokens,
                "blocks": len(blocks),
                "tail": tail,
            }
        )
        print(
            f"class {class_no} chapters {len(rows)} chars {chars} "
            f"tokens {n_tokens} blocks {len(blocks)} tail {tail}",
            flush=True,
        )

    (PACKED / "chapter_index.tsv").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    meta = {
        "kind": "corpus_bpe",
        "vocab_size_target": vocab_target,
        "vocab_size": tokenizer.vocab_size,
        "min_frequency": min_frequency,
        "block_size": block_size,
        "prelims_included": False,
        "pretrained_lm_weights": False,
        "classes": summary,
    }
    (PACKED / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    write_report(meta)
    print(f"vocab {tokenizer.vocab_size} target {vocab_target}")
    print(f"tokenizer {TOKENIZER_PATH}")
    print(f"report {REPORT}")
    return 0


def write_report(meta: dict) -> None:
    lines = [
        "# WP2 token table",
        "",
        "Corpus-native byte-level BPE trained on `data/clean/class_*/chapter_*.txt`.",
        "Prelims are not included. No language-model weights were loaded.",
        "Blocks are 512 tokens. A remainder shorter than 512 is a tail, not a block.",
        "",
        f"Vocab target {meta['vocab_size_target']}, actual {meta['vocab_size']}, min frequency {meta['min_frequency']}.",
        "",
        "| class | chapters | chars | bpe tokens | blocks of 512 | tail tokens |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    totals = {"chapters": 0, "chars": 0, "tokens": 0, "blocks": 0, "tail": 0}
    for row in meta["classes"]:
        lines.append(
            f"| {row['class']} | {row['chapters']} | {row['chars']} | {row['tokens']} | {row['blocks']} | {row['tail']} |"
        )
        for key in totals:
            totals[key] += row[key]
    lines.append(
        f"| total | {totals['chapters']} | {totals['chars']} | {totals['tokens']} | {totals['blocks']} | {totals['tail']} |"
    )
    lines.append("")
    lines.append("Chapter index: `data/packed/chapter_index.tsv` (`class`, `block`, `chapter`, `tokens`).")
    lines.append("Token arrays: `data/packed/class_{k}.npy`, shape `[n_blocks, 512]`.")
    lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
