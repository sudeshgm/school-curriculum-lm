#!/usr/bin/env python3
"""Download official NCERT English mathematics PDFs for Classes 1–6.

Source: https://ncert.nic.in/textbook/pdf/
Files stay under data/raw_pdf/. This script does not train and does not
write a public dataset.
"""

from __future__ import annotations

import hashlib
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw_pdf"
MANIFEST = OUT_DIR / "MANIFEST.tsv"
UA = {"User-Agent": "school-curriculum-lm local research download"}

# Current English-medium mathematics textbooks on ncert.nic.in (2026-27 reprint).
# Codes are the official textbook.php slugs. Chapter files are {code}1NN.pdf.
BOOKS = [
    {
        "class": 1,
        "code": "aejm1",
        "title": "Joyful Mathematics",
        "chapters": list(range(1, 14)),
    },
    {
        "class": 2,
        "code": "bejm1",
        "title": "Joyful Mathematics",
        "chapters": list(range(1, 12)),
    },
    {
        "class": 3,
        "code": "cemm1",
        "title": "Maths Mela",
        "chapters": list(range(1, 15)),
    },
    {
        "class": 4,
        "code": "demm1",
        "title": "Maths Mela",
        "chapters": list(range(1, 15)),
    },
    {
        "class": 5,
        "code": "eemm1",
        "title": "Maths Mela",
        "chapters": list(range(1, 16)),
    },
    {
        "class": 6,
        "code": "fegp1",
        "title": "Ganita Prakash",
        "chapters": list(range(1, 11)),
    },
]


def jobs() -> list[dict]:
    rows = []
    for book in BOOKS:
        code = book["code"]
        rows.append(
            {
                **book,
                "role": "prelims",
                "chapter": 0,
                "filename": f"{code}ps.pdf",
                "url": f"https://ncert.nic.in/textbook/pdf/{code}ps.pdf",
            }
        )
        for number in book["chapters"]:
            filename = f"{code}{number:02d}.pdf"
            rows.append(
                {
                    **book,
                    "role": "chapter",
                    "chapter": number,
                    "filename": filename,
                    "url": f"https://ncert.nic.in/textbook/pdf/{filename}",
                }
            )
    return rows


def download(job: dict) -> dict:
    dest_dir = OUT_DIR / f"class{job['class']}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / job["filename"]
    if dest.is_file() and dest.stat().st_size > 1000 and dest.read_bytes()[:5] == b"%PDF-":
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        return {
            **job,
            "path": dest.relative_to(ROOT).as_posix(),
            "sha256": digest,
            "bytes": dest.stat().st_size,
        }
    last_error = None
    for attempt in range(1, 5):
        try:
            req = urllib.request.Request(job["url"], headers=UA)
            hasher = hashlib.sha256()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(req, timeout=120) as response:
                with tmp.open("wb") as handle:
                    while True:
                        chunk = response.read(1 << 16)
                        if not chunk:
                            break
                        hasher.update(chunk)
                        handle.write(chunk)
            tmp.replace(dest)
            data = dest.read_bytes()[:5]
            if data != b"%PDF-":
                raise RuntimeError(f"not a PDF: {dest}")
            return {
                **job,
                "path": dest.relative_to(ROOT).as_posix(),
                "sha256": hasher.hexdigest(),
                "bytes": dest.stat().st_size,
            }
        except Exception as exc:  # network flake on ncert.nic.in
            last_error = exc
            time.sleep(attempt)
    raise RuntimeError(f"failed {job['url']}: {last_error}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pending = jobs()
    done: list[dict] = []
    print(f"downloading {len(pending)} official PDFs", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(download, job) for job in pending]
        for future in as_completed(futures):
            row = future.result()
            done.append(row)
            print(
                f"{len(done):02d}/{len(pending)} class {row['class']} {row['filename']} {row['bytes']}",
                flush=True,
            )
    done.sort(key=lambda row: (row["class"], row["chapter"], row["filename"]))
    lines = [
        "class\tfilename\tsource_url\tbytes\tsha256\ttitle\tbook_code\trole\tchapter\tlocal_path"
    ]
    for row in done:
        lines.append(
            "\t".join(
                [
                    str(row["class"]),
                    row["filename"],
                    row["url"],
                    str(row["bytes"]),
                    row["sha256"],
                    row["title"],
                    row["code"],
                    row["role"],
                    str(row["chapter"]),
                    row["path"],
                ]
            )
        )
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
