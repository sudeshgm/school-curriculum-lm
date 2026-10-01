# Amendments

Protocol changes to `p_new`, the target architecture, or `MATCH_TOKENS` must be written here before the code changes. WP0 does not make such a change.

## 000 — Plan files were not in the workspace (2026-09-30)

The handoff said to read, in order:

- `README.md`
- `05_IMPLEMENTATION_CHECKLIST.md`
- `00_MASTER_PLAN.md`
- `01_EXPERIMENTAL_PROTOCOL.md`
- `02_DATA_PIPELINE.md`
- `03_EVALUATION_PROTOCOL.md`
- `08_HYPERPARAMETERS.md`

None of those files were present anywhere in the workspace. No NCERT corpus was downloaded. No 300-item exam was generated. Class 1–6 training was not started.

What WP0 locked from the handoff text itself, without renumbering:

| Lock | Value used |
| --- | --- |
| Initialization | Random GPT-2 init. No pretrained LM weights. |
| Objective | Next-token cross-entropy only. |
| Critic / teacher LLM | Absent. |
| RL / PPO / GRPO / exam-as-reward | Absent. |
| Vision / page scans / ViT | Absent. |
| Scored MCQ bank | Not trained on. WP0 uses 8 dummy items only. |
| Target architecture | 12 layers, 12 heads, 768 dim, tied embeddings, GPT-2 vocab 50257, block size 1024. This is the standard GPT-2 small count (~124M), i.e. the ~100M-class model named in the handoff. Depth and width were not changed to force a round 100M. |
| Debug flag | 4 layers, 4 heads, 256 dim, same vocab and block size. Count is inside the required 15–30M band. |
| `MATCH_TOKENS` | 12. This is the planned 12-gram word-overlap filter named in the handoff, not a new width. |
| `p_new` | **Unset** (`null` in `configs/default.yaml`). No number was invented. |

`train.*` in `configs/default.yaml` is the public GPT-2 small AdamW recipe (lr `6e-4`, betas `0.9/0.95`, weight decay `0.1`) so the file is runnable. It is not claimed to be a verbatim copy of the missing `08_HYPERPARAMETERS.md`. The WP0 harness (`wp0.*`) is the 400-step overfit run specified in the handoff, with lr `1e-3` and weight decay `0` so a single chapter can memorize. If the plan file is restored and disagrees, update only through a new amendment.

NCERT text is not redistributed. Later packages may download official PDFs and extract them locally.

## 001 — WP2 corpus BPE vocab target (2026-10-01)

WP2 asks for a corpus-native BPE and 512-token blocks. No plan file states the BPE vocab size. `model.vocab_size` stays 50257. Target depth, width, heads, `p_new`, and `MATCH_TOKENS` are unchanged.

| WP2 data choice | Value |
| --- | --- |
| Tokenizer | Byte-level BPE trained on `data/clean/class_*/chapter_*.txt` only. |
| Vocab target | 4096, min frequency 2. Not the GPT-2 vocabulary. |
| Block length | 512. A shorter tail is counted and not stored as a block. |
| Language-model weights | Not loaded. Not trained. |

## 002 — Learner uses the corpus BPE (2026-10-01)

WP3 switches the learner off tiktoken GPT-2 and onto the WP2 corpus BPE. This is a vocabulary and context-length change. It is not a curriculum matrix.

| Lock | After 002 |
| --- | --- |
| Tokenizer | `corpus_bpe`, file `data/tokenizer/corpus_bpe.json`. Vocab 4096. |
| `model.vocab_size` | 4096. Was 50257. |
| `model.block_size` | 512. Was 1024. `train.sequence_length` is 512 so a batch cannot exceed the block. |
| Debug depth and width | Unchanged: 4 layers, 4 heads, 256 dim. |
| Target depth and width | Unchanged: 12 layers, 12 heads, 768 dim. Not trained in WP3. |
| `p_new` | Still null. |
| `MATCH_TOKENS` | Still 12. |
| Objective | Next-token cross-entropy only. No critic. No RL. No pretrained LM weights. |

The old ~16M debug count and ~124M target count included a 50257-word embedding table. Those counts are not preserved. Depth and width were not widened to chase them.

## 003 — WP3b continues the Class 1 debug checkpoint (2026-10-01)

WP3b does not change `p_new`, `MATCH_TOKENS`, depth, width, vocab, or block size. It does not train Classes 2–6 or the 12-layer model.

| WP3b choice | Value |
| --- | --- |
| Start | `artifacts/wp3/class1_debug.pt` after 40 steps. Not a new random init. |
| Optimizer | AdamW settings from `train.*`. Moments restart because the checkpoint did not store them. |
| Stop | Full-batch loss under 2.0, or 200 more steps, whichever comes first. |
| Baseline | The same debug stack at seed 1337, before any update. |
| Format examples | Four original items, ids f1–f4. They are not scored. |
| Objective | Next-token cross-entropy only. No critic. No pretrained LM weights. |

