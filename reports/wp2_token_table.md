# WP2 token table

Corpus-native byte-level BPE trained on `data/clean/class_*/chapter_*.txt`.
Prelims are not included. No language-model weights were loaded.
Blocks are 512 tokens. A remainder shorter than 512 is a tail, not a block.

Vocab target 4096, actual 4096, min frequency 2.

| class | chapters | chars | bpe tokens | blocks of 512 | tail tokens |
|---:|---:|---:|---:|---:|---:|
| 1 | 13 | 49316 | 15036 | 29 | 188 |
| 2 | 11 | 66306 | 22024 | 43 | 8 |
| 3 | 14 | 127574 | 38823 | 75 | 423 |
| 4 | 14 | 149003 | 48375 | 94 | 247 |
| 5 | 15 | 166334 | 53970 | 105 | 210 |
| 6 | 10 | 382103 | 115547 | 225 | 347 |
| total | 77 | 940636 | 293775 | 571 | 1423 |

Chapter index: `data/packed/chapter_index.tsv` (`class`, `block`, `chapter`, `tokens`).
Token arrays: `data/packed/class_{k}.npy`, shape `[n_blocks, 512]`.
