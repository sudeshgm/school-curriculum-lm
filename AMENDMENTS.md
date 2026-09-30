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
