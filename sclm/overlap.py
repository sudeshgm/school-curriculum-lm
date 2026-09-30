"""12-gram word-overlap filter.

A training document is dropped when it shares any word 12-gram with a held-out
eval string. Normalization is lowercase, punctuation stripped to spaces.
Shorter-than-12 eval strings have no 12-gram and do not contaminate.

MATCH_TOKENS is locked at 12. Passing a different n is only for tests of the
boundary (11 must not count). Production callers use the default.
"""

from __future__ import annotations

import re

from sclm.locks import MATCH_TOKENS

_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")


def word_tokens(text: str) -> list[str]:
    lowered = text.lower()
    cleaned = _NON_ALNUM.sub(" ", lowered)
    return [tok for tok in cleaned.split() if tok]


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if n <= 0:
        raise ValueError(f"n-gram order must be positive, got {n}")
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def shares_ngram(train_text: str, eval_text: str, n: int = MATCH_TOKENS) -> bool:
    """True iff train_text and eval_text share at least one word n-gram."""
    eval_grams = ngrams(word_tokens(eval_text), n)
    if not eval_grams:
        return False
    train_grams = ngrams(word_tokens(train_text), n)
    return not train_grams.isdisjoint(eval_grams)


def filter_documents(
    documents: list[str],
    eval_texts: list[str],
    n: int = MATCH_TOKENS,
) -> tuple[list[str], list[int]]:
    """Drop documents that share any n-gram with any eval text.

    Returns (kept_documents, dropped_indices). Documents are not rewritten.
    """
    banned: set[tuple[str, ...]] = set()
    for eval_text in eval_texts:
        banned |= ngrams(word_tokens(eval_text), n)
    kept: list[str] = []
    dropped: list[int] = []
    for index, doc in enumerate(documents):
        grams = ngrams(word_tokens(doc), n)
        if banned and not grams.isdisjoint(banned):
            dropped.append(index)
        else:
            kept.append(doc)
    return kept, dropped
