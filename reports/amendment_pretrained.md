# Amendment — pretrained learner for WP12

WP12 does not train the 4.3M debug model and does not resume any earlier checkpoint.

| Lock | WP12 |
| --- | --- |
| Learner | `HuggingFaceTB/SmolLM2-360M` |
| Checkpoint type | Base pretrained model. Not a math-tuned checkpoint. Not random weights. |
| Not used | `HuggingFaceTB/SmolLM2-360M-Instruct`, any math-tuned checkpoint, the 4.3M debug model |
| Objective | Next-token cross-entropy only |
| Critic | None |
| `p_new` | Still null |
| Earlier runs | Unchanged |

The SmolLM2 tokenizer is used only in this condition. The corpus BPE file is not replaced.

Continue-pretrain uses AdamW, learning rate 2e-5, betas 0.9 and 0.95, weight decay 0.1. Sequences are 512 tokens. Class 1 is trained, then Class 2, with no Class 1 replay. Each class stops when full-batch loss improves by less than 0.01 over 20 steps, or at 200 steps.
