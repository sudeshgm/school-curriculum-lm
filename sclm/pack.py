"""Pack a class of chapter token streams into fixed blocks.

A block is exactly `block_size` tokens. A short tail is not padded and is
not a block. The chapter index records how many tokens in each block came
from each chapter, in chapter order.
"""

from __future__ import annotations


def pack_chapters(
    chapters: list[tuple[str, list[int]]], block_size: int
) -> tuple[list[list[int]], list[tuple[int, str, int]], int, int]:
    """Return blocks, index rows, tail length, and total token count.

    Index rows are `(block, chapter_id, n_tokens)`.
    """
    if block_size < 1:
        raise ValueError("block_size must be positive")
    stream: list[int] = []
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for chapter_id, ids in chapters:
        start = cursor
        stream.extend(ids)
        cursor += len(ids)
        if cursor > start:
            spans.append((chapter_id, start, cursor))
    n_tokens = len(stream)
    n_blocks = n_tokens // block_size
    tail = n_tokens - n_blocks * block_size
    blocks: list[list[int]] = []
    rows: list[tuple[int, str, int]] = []
    for block in range(n_blocks):
        start = block * block_size
        end = start + block_size
        blocks.append(stream[start:end])
        for chapter_id, span_start, span_end in spans:
            overlap = min(end, span_end) - max(start, span_start)
            if overlap > 0:
                rows.append((block, chapter_id, overlap))
    return blocks, rows, tail, n_tokens
