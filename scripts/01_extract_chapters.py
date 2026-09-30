#!/usr/bin/env python3
"""Extract and clean chapter text from the local NCERT PDFs.

Reads data/raw_pdf/MANIFEST.tsv. Writes:
  data/clean/class_{k}/chapter_{id}.txt
  data/text/                         (same text, previous path)
  reports/data_stats.md
  data/audit/                        (12 seeded-random pages)

Does not train and does not build an exam bank.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sclm.ncert_clean import clean_document, clean_page_lines, join_wrapped_lines
from sclm.tokenizer import GPT2Tokenizer

MANIFEST = ROOT / "data" / "raw_pdf" / "MANIFEST.tsv"
TEXT_ROOT = ROOT / "data" / "text"
CLEAN_ROOT = ROOT / "data" / "clean"
AUDIT_ROOT = ROOT / "data" / "audit"
STATS_PATH = ROOT / "reports" / "data_stats.md"
AUDIT_SEED = 1337


def page_strings(pdf_path: Path) -> list[str]:
    document = pymupdf.open(pdf_path)
    return [page.get_text("text") for page in document]


def cleaned_pages(raw_pages: list[str]) -> list[str]:
    """Per-page cleaned text, using the same rules as the chapter join."""
    from sclm.ncert_clean import drop_running_headers

    pages = drop_running_headers([clean_page_lines(text) for text in raw_pages])
    return [join_wrapped_lines(lines) for lines in pages]


def main() -> int:
    if not MANIFEST.is_file():
        raise SystemExit(f"Missing {MANIFEST}. Run scripts/01_fetch_ncert_maths.py first.")
    tokenizer = GPT2Tokenizer()
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8"), delimiter="\t"))
    TEXT_ROOT.mkdir(parents=True, exist_ok=True)
    CLEAN_ROOT.mkdir(parents=True, exist_ok=True)
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    for old in AUDIT_ROOT.glob("*.txt"):
        old.unlink()

    by_class: dict[int, list[dict]] = {}
    for row in rows:
        by_class.setdefault(int(row["class"]), []).append(row)

    summary = []
    audit_chunks = []
    failures: list[str] = []
    rng = random.Random(AUDIT_SEED)
    chapter_index = [
        "class\tchapter\tbook_code\ttitle_line\tchars\twhitespace_tokens\tgpt2_tokens\tsource_pdf"
    ]
    for class_no in range(1, 7):
        items = by_class.get(class_no)
        if not items:
            failures.append(f"class {class_no}: no manifest rows")
            summary.append(_empty_summary(class_no))
            continue
        items = sorted(items, key=lambda row: int(row["chapter"]))
        text_dir = TEXT_ROOT / f"class{class_no}"
        clean_dir = CLEAN_ROOT / f"class_{class_no}"
        text_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)
        chapter_chars = 0
        chapter_ws = 0
        chapter_tokens = 0
        n_chapters = 0
        front_chars = 0
        chapter_pages: list[tuple[dict, list[str], list[str]]] = []
        for row in items:
            pdf_path = ROOT / row["local_path"]
            try:
                raw_pages = page_strings(pdf_path)
            except Exception as exc:
                failures.append(f"class {class_no} {row['filename']}: {exc}")
                continue
            if not raw_pages:
                failures.append(f"class {class_no} {row['filename']}: zero pages")
                continue
            clean_each = cleaned_pages(raw_pages)
            text = clean_document(raw_pages)
            if row["role"] == "prelims":
                (text_dir / "front_matter.txt").write_text(text + "\n", encoding="utf-8")
                (clean_dir / "prelims.txt").write_text(text + "\n", encoding="utf-8")
                front_chars = len(text)
                if len(text) < 50:
                    failures.append(f"class {class_no} prelims: only {len(text)} chars")
                continue
            n_chapters += 1
            chapter_no = int(row["chapter"])
            (text_dir / f"ch{chapter_no:02d}.txt").write_text(text + "\n", encoding="utf-8")
            (clean_dir / f"chapter_{chapter_no:02d}.txt").write_text(text + "\n", encoding="utf-8")
            ws = len(text.split())
            tokens = len(tokenizer.encode(text)) if text else 0
            chapter_chars += len(text)
            chapter_ws += ws
            chapter_tokens += tokens
            title = next((line.strip() for line in text.splitlines() if line.strip()), "")
            chapter_index.append(
                "\t".join(
                    [
                        str(class_no),
                        str(chapter_no),
                        row["book_code"],
                        title.replace("\t", " ")[:120],
                        str(len(text)),
                        str(ws),
                        str(tokens),
                        row["filename"],
                    ]
                )
            )
            chapter_pages.append((row, raw_pages, clean_each))
            if len(text) < 200:
                failures.append(
                    f"class {class_no} chapter {chapter_no}: only {len(text)} chars"
                )
        if n_chapters == 0:
            failures.append(f"class {class_no}: no chapter text")
        book = items[0].get("title") or items[0].get("book_title") or ""
        summary.append(
            {
                "class": class_no,
                "book": book,
                "code": items[0]["book_code"],
                "chapters": n_chapters,
                "chars": chapter_chars,
                "ws": chapter_ws,
                "tokens": chapter_tokens,
                "front_chars": front_chars,
            }
        )
        if chapter_pages:
            audit_chunks.extend(pick_audit(class_no, chapter_pages, rng))

    (TEXT_ROOT / "chapters.tsv").write_text("\n".join(chapter_index) + "\n", encoding="utf-8")
    summary_lines = [
        "class\tbook\tbook_code\tchapters\tchars\twhitespace_tokens\tgpt2_tokens\tfront_matter_chars"
    ]
    for row in summary:
        summary_lines.append(
            f"{row['class']}\t{row['book']}\t{row['code']}\t{row['chapters']}\t"
            f"{row['chars']}\t{row['ws']}\t{row['tokens']}\t{row['front_chars']}"
        )
    (TEXT_ROOT / "SUMMARY.tsv").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    write_stats(summary, failures)

    if len(audit_chunks) != 12:
        failures.append(f"expected 12 audit pages, got {len(audit_chunks)}")
    combined = []
    index_lines = ["index\tclass\tfilename\tpage\tseed"]
    for index, chunk in enumerate(audit_chunks, start=1):
        name = f"{index:02d}_class{chunk['class']}_{chunk['filename']}_p{chunk['page']:03d}.txt"
        body = format_audit(chunk)
        (AUDIT_ROOT / name).write_text(body, encoding="utf-8")
        combined.append(body)
        index_lines.append(
            f"{index}\t{chunk['class']}\t{chunk['filename']}\t{chunk['page']}\t{AUDIT_SEED}"
        )
    (AUDIT_ROOT / "INDEX.tsv").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    notice = (
        "Local extraction audit of official NCERT mathematics PDFs.\n"
        f"Twelve pages, two per class, sampled with seed {AUDIT_SEED}.\n"
        "These pages are for checking the cleaner inside this project.\n"
        "They are not a public dataset and must not be republished.\n\n"
    )
    (AUDIT_ROOT / "SAMPLE_12_PAGES.txt").write_text(
        notice + ("\n\n" + ("=" * 72) + "\n\n").join(combined),
        encoding="utf-8",
    )
    print("CLASS\tBOOK\tCHAPTERS\tCHARS\tWS_TOKENS")
    for row in summary:
        print(
            f"{row['class']}\t{row['book']}\t{row['chapters']}\t{row['chars']}\t{row['ws']}"
        )
    print(f"audit_pages: {len(audit_chunks)}")
    print(f"stats: {STATS_PATH}")
    if failures:
        print("FAILURES")
        for item in failures:
            print(item)
        return 1
    print("failures: none")
    return 0


def _empty_summary(class_no: int) -> dict:
    return {
        "class": class_no,
        "book": "",
        "code": "",
        "chapters": 0,
        "chars": 0,
        "ws": 0,
        "tokens": 0,
        "front_chars": 0,
    }


def write_stats(summary: list[dict], failures: list[str]) -> None:
    lines = [
        "# NCERT English mathematics, Classes 1–6",
        "",
        "Local extract of official PDFs from https://ncert.nic.in/textbook/pdf/.",
        "Not a public dataset. Do not copy this text out of the project.",
        "",
        "Counts are cleaned chapter text only. Prelims PDFs are saved as `prelims.txt` and are not included.",
        "Whitespace tokens are `str.split()` tokens.",
        "",
        "| class | book | chapters | chars | whitespace tokens |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['class']} | {row['book']} | {row['chapters']} | {row['chars']} | {row['ws']} |"
        )
    total_ch = sum(row["chapters"] for row in summary)
    total_chars = sum(row["chars"] for row in summary)
    total_ws = sum(row["ws"] for row in summary)
    lines.append(f"| total |  | {total_ch} | {total_chars} | {total_ws} |")
    lines.append("")
    if failures:
        lines.append("Extraction failures:")
        lines.extend(f"- {item}" for item in failures)
    else:
        lines.append("Extraction failures: none.")
    lines.append("")
    STATS_PATH.write_text("\n".join(lines), encoding="utf-8")


def pick_audit(
    class_no: int,
    chapters: list[tuple[dict, list[str], list[str]]],
    rng: random.Random,
) -> list[dict]:
    """Two random readable pages from two different chapters."""
    by_file: dict[str, list[tuple[dict, int, str, str]]] = {}
    for row, raw_pages, clean_pages in chapters:
        bucket = by_file.setdefault(row["filename"], [])
        for page_index, (raw, clean) in enumerate(zip(raw_pages, clean_pages)):
            if len(clean.strip()) >= 200:
                bucket.append((row, page_index, raw, clean))
    eligible = [name for name, pages in by_file.items() if pages]
    if len(eligible) < 2:
        raise SystemExit(f"class {class_no} does not have two readable chapters to audit")
    picked = []
    for name in rng.sample(eligible, 2):
        picked.append(rng.choice(by_file[name]))
    chosen = []
    for row, page_index, raw, clean in picked:
        chosen.append(
            {
                "class": class_no,
                "book": row.get("title") or row.get("book_title"),
                "filename": row["filename"],
                "url": row["source_url"],
                "sha256": row["sha256"],
                "page": page_index + 1,
                "raw": raw.strip(),
                "clean": clean.strip(),
            }
        )
    return chosen


def format_audit(chunk: dict) -> str:
    return (
        f"class: {chunk['class']}\n"
        f"book: {chunk['book']}\n"
        f"file: {chunk['filename']}\n"
        f"page: {chunk['page']}\n"
        f"source: {chunk['url']}\n"
        f"sha256: {chunk['sha256']}\n"
        "\n--- cleaned ---\n"
        f"{chunk['clean']}\n"
        "\n--- raw ---\n"
        f"{chunk['raw']}\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
