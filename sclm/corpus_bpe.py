"""Corpus-native byte-level BPE.

The merges are learned from the text you pass in. This file does not load a
pretrained language model and does not load a pretrained tokenizer file
unless you point it at a model trained by `train_corpus_bpe`.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


class CorpusBPE:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def save(self, path: str | Path) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(dest))

    @classmethod
    def load(cls, path: str | Path) -> "CorpusBPE":
        return cls(Tokenizer.from_file(str(path)))


def train_corpus_bpe(texts: list[str], vocab_size: int, min_frequency: int = 2) -> CorpusBPE:
    if vocab_size < 256:
        raise ValueError("byte-level BPE needs vocab_size >= 256")
    if not texts or not any(text.strip() for text in texts):
        raise ValueError("refusing to train a BPE on empty text")
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[],
        show_progress=False,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return CorpusBPE(tokenizer)
