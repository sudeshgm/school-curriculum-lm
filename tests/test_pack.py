"""WP2 packing and a tiny corpus-native BPE. No textbook text, no LM weights."""

from sclm.corpus_bpe import train_corpus_bpe
from sclm.locks import ALLOW_PRETRAINED_LM_WEIGHTS
from sclm.pack import pack_chapters


def test_pack_fills_512_and_indexes_chapters():
    chapters = [("01", list(range(600))), ("02", list(range(100, 500)))]
    blocks, rows, tail, n_tokens = pack_chapters(chapters, 512)
    assert n_tokens == 1000
    assert len(blocks) == 1
    assert len(blocks[0]) == 512
    assert tail == 488
    assert rows == [(0, "01", 512)]
    assert sum(n for _block, _chapter, n in rows) == 512


def test_pack_splits_a_block_across_chapters():
    chapters = [("01", [1] * 400), ("02", [2] * 200)]
    blocks, rows, tail, n_tokens = pack_chapters(chapters, 512)
    assert n_tokens == 600
    assert tail == 88
    assert rows == [(0, "01", 400), (0, "02", 112)]
    assert blocks[0].count(1) == 400
    assert blocks[0].count(2) == 112


def test_empty_chapter_does_not_appear_in_the_index():
    blocks, rows, tail, n_tokens = pack_chapters([("01", []), ("02", [3] * 10)], 4)
    assert n_tokens == 10
    assert len(blocks) == 2
    assert tail == 2
    assert rows == [(0, "02", 4), (1, "02", 4)]


def test_corpus_bpe_roundtrip_does_not_load_an_lm():
    assert ALLOW_PRETRAINED_LM_WEIGHTS is False
    text = "2 + 2 = 4. Count the sides.\n"
    tokenizer = train_corpus_bpe([text] * 30, vocab_size=300, min_frequency=2)
    assert 256 <= tokenizer.vocab_size <= 300
    assert tokenizer.decode(tokenizer.encode(text)) == text
