"""Built-in WP0 chapter and 8 dummy MCQs.

These items are not the scored exam bank. Every correct letter is B so a
letter-logprob scorer that always emits A cannot pass after a real overfit,
and a scorer that reads log-probabilities can.

The chapter text includes each item with its answer line so next-token
training can memorize the local pattern used at scoring time.
"""

from __future__ import annotations

import hashlib

from sclm.scoring import render_mcq

DUMMY_ITEMS: list[dict] = [
    {
        "id": "d1",
        "question": "What is the oldest boat at River School named?",
        "choices": {"A": "Lotus", "B": "Mango", "C": "Cedar", "D": "Pearl"},
        "answer": "B",
    },
    {
        "id": "d2",
        "question": "How many boats does River School keep?",
        "choices": {"A": "two", "B": "four", "C": "seven", "D": "nine"},
        "answer": "B",
    },
    {
        "id": "d3",
        "question": "What colour is the river beside the school?",
        "choices": {"A": "green", "B": "blue", "C": "grey", "D": "red"},
        "answer": "B",
    },
    {
        "id": "d4",
        "question": "When do classes begin?",
        "choices": {"A": "seven", "B": "nine", "C": "noon", "D": "dusk"},
        "answer": "B",
    },
    {
        "id": "d5",
        "question": "Who is the head teacher?",
        "choices": {"A": "Mr. Das", "B": "Ms. Iyer", "C": "Mr. Khan", "D": "Ms. Roy"},
        "answer": "B",
    },
    {
        "id": "d6",
        "question": "Which day is library day?",
        "choices": {"A": "Monday", "B": "Thursday", "C": "Saturday", "D": "Sunday"},
        "answer": "B",
    },
    {
        "id": "d7",
        "question": "Which animal lives by the school dock?",
        "choices": {"A": "heron", "B": "otter", "C": "turtle", "D": "crane"},
        "answer": "B",
    },
    {
        "id": "d8",
        "question": "What is the school motto?",
        "choices": {"A": "count the stars", "B": "read the river", "C": "climb the hill", "D": "mind the gate"},
        "answer": "B",
    },
]


PROSE = (
    "Chapter 1. River School.\n"
    "River School sits beside the blue river. The school keeps four boats. "
    "The oldest boat is named Mango. Classes begin at nine. "
    "The head teacher is Ms. Iyer. Library day is Thursday. "
    "An otter lives by the school dock. The school motto is read the river.\n"
)


def dummy_items() -> list[dict]:
    return [dict(item) for item in DUMMY_ITEMS]


def dummy_chapter(repeats: int = 12) -> str:
    """One short chapter: prose plus the dummy items with answers, repeated."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    blocks = [render_mcq(item, include_answer=True) for item in DUMMY_ITEMS]
    body = "\n\n".join(blocks)
    repeated = "\n\n".join([body] * repeats)
    return PROSE + "\n" + repeated + "\n"


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
