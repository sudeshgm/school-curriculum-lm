"""GPT-2 byte-level BPE tokenizer.

The encoding is a vocabulary, not a pretrained language model. No weights
are loaded here.
"""

from __future__ import annotations

import tiktoken


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
