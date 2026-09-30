"""Clean text extracted from official NCERT mathematics PDFs.

Drops running furniture (reprint stamps, edge page numbers, production
marks, repeated headers). Does not rewrite the book's sentences.
"""

from __future__ import annotations

import re
from collections import Counter

REPRINT = re.compile(r"^Reprint\s+\d{4}-\d{2}$")
PAGE_NO = re.compile(r"^\d{1,3}$")
STAMP = re.compile(
    r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)?$",
    re.IGNORECASE,
)
INDD = re.compile(r"\.indd\b", re.IGNORECASE)
PDCODE = re.compile(r"^PD\s+\d+T\b")
NUMBERED_ITEM = re.compile(r"^\d+[\.\)]\s")
SECTION_HEAD = re.compile(r"^\d+(?:\.\d+)+\s+\S")
BARE_ITEM = re.compile(r"^\d+\.$")
BULLET = ("•", "●", "■", "–", "-", "*")
TERMINAL = set(".?!;:\"'”’")


def normalize_line(line: str) -> str:
    line = (
        line.replace("\u00a0", " ")
        .replace("\u2009", " ")
        .replace("\u202f", " ")
        .replace("\u00ad", "")
    )
    return re.sub(r"[ \t]+", " ", line).strip()


def clean_page_lines(text: str) -> list[str]:
    raw = [normalize_line(line) for line in text.splitlines()]
    nonempty = [i for i, line in enumerate(raw) if line]
    edge_page_no: set[int] = set()
    if nonempty:
        edge_page_no.add(nonempty[0])
        edge_page_no.add(nonempty[-1])
        # A running page number often sits on the second line under a short header.
        if len(nonempty) >= 4 and len(raw[nonempty[0]]) <= 40:
            edge_page_no.add(nonempty[1])
    kept: list[str] = []
    for index, line in enumerate(raw):
        if not line:
            kept.append("")
            continue
        if REPRINT.match(line) or INDD.search(line) or PDCODE.match(line):
            continue
        if STAMP.match(line):
            continue
        if PAGE_NO.match(line) and index in edge_page_no:
            continue
        if kept and kept[-1] == line:
            continue
        kept.append(line)
    return kept


def drop_running_headers(pages: list[list[str]]) -> list[list[str]]:
    """Drop short lines that repeat on many pages, keeping the first copy."""
    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update({line for line in lines if line and len(line) <= 60})
    threshold = max(3, int(0.4 * len(pages))) if pages else 3
    repeated = {
        line
        for line, count in counts.items()
        if count >= threshold and line[-1] not in TERMINAL
    }
    seen: set[str] = set()
    output: list[list[str]] = []
    for lines in pages:
        new: list[str] = []
        for line in lines:
            if line in repeated:
                if line in seen:
                    continue
                seen.add(line)
            new.append(line)
        output.append(new)
    return output


def join_wrapped_lines(lines: list[str]) -> str:
    merged: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line:
            merged.append("")
            index += 1
            continue
        while index + 1 < len(lines) and lines[index + 1]:
            nxt = lines[index + 1]
            if BARE_ITEM.match(line):
                line = line + " " + nxt
                index += 1
                continue
            if line.endswith("-") and nxt[:1].islower():
                line = line[:-1] + nxt
                index += 1
                continue
            if SECTION_HEAD.match(nxt) or NUMBERED_ITEM.match(nxt):
                break
            if (
                len(line) >= 48
                and line[-1] not in TERMINAL
                and not nxt.startswith(BULLET)
                and not NUMBERED_ITEM.match(nxt)
            ):
                line = line + " " + nxt
                index += 1
                continue
            break
        merged.append(line)
        index += 1
    text = "\n".join(merged)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_document(page_texts: list[str]) -> str:
    pages = [clean_page_lines(text) for text in page_texts]
    pages = drop_running_headers(pages)
    chunks = [join_wrapped_lines(lines) for lines in pages]
    chunks = [chunk for chunk in chunks if chunk]
    return "\n\n".join(chunks).strip()
