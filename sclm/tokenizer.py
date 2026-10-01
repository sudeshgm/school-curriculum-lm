"""Tokenizers for the learner.

GPT2Tokenizer is the tiktoken vocabulary used by the WP0 harness. It is a
vocabulary, not a pretrained language model. The training learner after
amendment 002 uses CorpusBPE via `load_tokenizer`.
"""

from __future__ import annotations

from pathlib import Path

import tiktoken

from sclm.corpus_bpe import CorpusBPE


class GPT2Tokenizer:
    def __init__(self) -> None:
        self.enc = tiktoken.get_encoding("gpt2")

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)


def load_tokenizer(cfg: dict, root: str | Path):
    """Build the tokenizer named in the config. No language-model weights."""
    spec = cfg["tokenizer"]
    name = spec["name"]
    if name == "gpt2":
        tokenizer = GPT2Tokenizer()
    elif name == "corpus_bpe":
        tokenizer = CorpusBPE.load(Path(root) / spec["path"])
    else:
        raise ValueError(f"Unknown tokenizer {name!r}")
    if tokenizer.vocab_size != int(spec["vocab_size"]):
        raise ValueError(
            f"Tokenizer vocab {tokenizer.vocab_size} != config {spec['vocab_size']}"
        )
    if tokenizer.vocab_size != int(cfg["model"]["vocab_size"]):
        raise ValueError("Tokenizer vocab does not match model.vocab_size")
    return tokenizer
