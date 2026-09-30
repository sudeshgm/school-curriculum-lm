# School-Curriculum LM — WP0 status report

Date: 2026-09-30  
Scope executed: **WP0 only**. Stopped after it passed.  
This is a research training experiment, not a web app. The preinstalled Node/TanStack scaffold in this workspace was left untouched and is not part of the experiment.

## 1. What was asked

Take over the School-Curriculum LM from the plan files in the workspace, in this order:

1. `README.md`
2. `05_IMPLEMENTATION_CHECKLIST.md`
3. `00_MASTER_PLAN.md`
4. `01_EXPERIMENTAL_PROTOCOL.md`
5. `02_DATA_PIPELINE.md`
6. `03_EVALUATION_PROTOCOL.md`
7. `08_HYPERPARAMETERS.md`

Then implement **WP0 only**:

1. A GPT-2-like PyTorch model with a debug flag (~15–30M) and a target 12-layer / 768-dim ~100M model.
2. `configs/default.yaml` matching `08_HYPERPARAMETERS.md`.
3. `scripts/00_overfit_chapter.py`: 400 steps on one chapter; loss must fall.
4. `scripts/00_eval_sanity.py`: 8 dummy MCQs whose correct letter is always B; after overfit, letter-logprob scoring must beat chance.
5. Pytest for letter-logprob scoring and the planned 12-gram overlap filter.

Hard constraints that were followed:

- Learner starts from random weights. No pretrained LM weights.
- No critic or teacher LLM in the training loop.
- No vision, page scans, or ViT.
- No RL, PPO, GRPO, or exam-as-reward.
- Do not train on the scored MCQ bank.
- Do not change `p_new`, the target architecture, or `MATCH_TOKENS` without an amendment.
- Do not republish NCERT text. Do not download the full NCERT corpus until WP0 passes.
- Do not generate 300 exam items.
- Do not start Class 1–6 training.
- Do not invent a new method.

## 2. What could not be achieved

### 2.1 The plan files were not in the workspace

A full filesystem search found none of the seven named plan documents. They were also not recoverable from the public web by the distinctive protocol terms (`MATCH_TOKENS`, letter-logprob school-curriculum protocol, `00_MASTER_PLAN.md`, and so on).

Because of that, the following were **not** done, on purpose:

| Not done | Why |
| --- | --- |
| Checklist items after WP0 | The checklist file was missing, and the handoff said to stop after WP0. |
| Numeric `p_new` | Named as locked, but no value was in the handoff. Left `null`. Inventing one would have been a new method. |
| A yaml that is a verbatim copy of `08_HYPERPARAMETERS.md` | That file was absent. See §4 for what was written instead. |
| Exact debug depth/width from the plan | The handoff only required the 15–30M band. Dims were chosen inside that band (amendment 000). |
| Data pipeline (PDF download, chapter split, curriculum order) | `02_DATA_PIPELINE.md` was absent. No NCERT PDFs were fetched. |
| The 300-item scored exam | Explicitly out of WP0. Not generated. |
| Class 1–6 training, the target-model training run | Explicitly out of WP0. The 124M model was only constructed to check its parameter count. It was not trained. |
| Held-out exam evaluation | The sanity set is 8 dummy items whose answer lines are inside the overfit chapter. That checks the scorer, not generalization. |
| Applying the 12-gram filter to a real train/eval split | The filter is implemented and unit-tested. It has not been run on a corpus. |
| A real-chapter overfit | `00_overfit_chapter.py` accepts `--chapter`, but the acceptance run used the built-in dummy chapter so the letter-logprob check had something to memorize. |
| GPU training | The machine had 2 CPU cores and no CUDA. Torch 2.14.1+cpu was installed for the run. |

### 2.2 Defaults that are not from the missing hyperparameter file

These are documented in [AMENDMENTS.md](/workspace/AMENDMENTS.md) (amendment 000) so they are not silent changes:

- `train.*` uses the public GPT-2 small AdamW recipe: lr `6e-4`, betas `(0.9, 0.95)`, weight decay `0.1`, grad clip `1.0`, warmup `100`. This is **not** claimed to be `08_HYPERPARAMETERS.md`.
- The WP0 harness uses lr `1e-3`, weight decay `0`, warmup `0`, sequence length `64`, batch size `8`, so one short chapter can be memorized in 400 steps on CPU.
- Dropout `0.0`, bias on, block size `1024`, GPT-2 vocab `50257`, tied token embeddings.
- 12-gram normalization (chosen because the data-pipeline note was missing): lowercase, punctuation stripped to spaces, hyphens split, whitespace-collapsed word tokens. An 11-gram does not count. An eval string shorter than 12 words cannot contaminate.

If the plan files are restored and disagree, those values should be updated only through a new amendment. `p_new`, target depth/width/heads, and `MATCH_TOKENS` were not given new numbers.

## 3. What was achieved

WP0 acceptance passed on 2026-09-30.

### 3.1 Model

[sclm/model.py](/workspace/sclm/model.py) is a GPT-2-like pre-norm decoder:

- Token embeddings + learned position embeddings, dropout, pre-norm blocks, causal self-attention, 4× MLP, tanh-approximate GELU, final LayerNorm, tied `lm_head`.
- Init std `0.02`, with residual `c_proj` scaled by `1/sqrt(2 * n_layer)`.
- `from_scratch = True` at construction. There is no `from_pretrained` path. [sclm/checkpoint.py](/workspace/sclm/checkpoint.py) refuses any payload not marked `from_scratch`.

| Variant | Layers | Heads | Dim | Block | Vocab | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `debug` | 4 | 4 | 256 | 1024 | 50257 | **16,287,488** (inside 15–30M) |
| `target` | 12 | 12 | 768 | 1024 | 50257 | **124,439,808** (standard GPT-2 small; the ~100M-class model named in the handoff) |

Depth and width of the target were not reduced to force a round 100M. 124,439,808 is the usual tied-embedding count for 12 × 768.

### 3.2 Config and locks

- [configs/default.yaml](/workspace/configs/default.yaml)
- [sclm/locks.py](/workspace/sclm/locks.py): `MATCH_TOKENS = 12`, pretrained weights off, critic off, RL off, scored-MCQ-bank training off, target 12 / 768 / 12.
- `curriculum.p_new` is `null`.

### 3.3 Overfit (400 steps)

Command: `python scripts/00_overfit_chapter.py`  
Text: built-in dummy chapter (original prose plus 8 MCQs with `Answer: B`, repeated). Not NCERT. Not a scored exam bank.  
Optimizer for this harness: AdamW, lr `1e-3`, weight decay `0`, grad clip `1.0`, seed `1337`.  
Device: CPU. Wall time: **197.4 s**.

| Metric | Value |
| --- | ---: |
| Fixed-batch loss before any update | 10.9167 |
| Fixed-batch loss after 400 steps | 0.0546 |
| Train loss step 1 | 10.8985 |
| Train loss step 50 | 1.2200 |
| Train loss step 100 | 0.3497 |
| Train loss step 200 | 0.0716 |
| Train loss step 400 | 0.0660 |
| Loss fell | **yes** |
| Verdict | **PASS** |

Checkpoint: [artifacts/wp0/overfit_chapter.pt](/workspace/artifacts/wp0/overfit_chapter.pt)  
Loss summary: [artifacts/wp0/overfit_loss.json](/workspace/artifacts/wp0/overfit_loss.json)  
Chapter sha256: `17d1f3dc166b8f905216e54a2b2b03f294e50939edd6dd2804e6f5f48adddfd2`

The curve is not monotone (step 300 was 0.0855, step 350 was 0.1473). The acceptance test is the fixed batch held out of the comparison noise: 10.92 → 0.055.

### 3.4 Letter-logprob sanity

Command: `python scripts/00_eval_sanity.py`  
Eight dummy items, gold letter **B** on every item. No weight update. No generation. No critic.

The score of a letter is the teacher-forced log-probability of the continuation `" {letter}"` after a prompt that ends at `Answer:`. BPE alignment is checked: the continuation must extend the prompt’s token ids with no merge back into the prompt. Ties would break to the earlier letter. Chance for four letters is 0.25, and “beats chance” is strict (`>`).

| Item | Pred | log p(A) | log p(B) | log p(C) | log p(D) |
| --- | --- | ---: | ---: | ---: | ---: |
| d1 | B | −23.293 | −0.009 | −23.659 | −23.341 |
| d2 | B | −22.229 | −0.002 | −22.657 | −22.264 |
| d3 | B | −23.475 | −0.001 | −23.936 | −23.720 |
| d4 | B | −22.177 | −0.001 | −22.711 | −22.096 |
| d5 | B | −23.812 | −0.001 | −24.285 | −24.000 |
| d6 | B | −22.451 | −0.001 | −22.948 | −22.652 |
| d7 | B | −22.393 | −0.001 | −22.811 | −22.422 |
| d8 | B | −22.592 | −0.001 | −23.076 | −22.555 |

Accuracy **8/8 = 1.00** vs chance **0.25**. Verdict: **PASS**.

This result is memorization of `Answer: B` in the dummy chapter. It does not show that the model can answer unseen questions.

The eval script refuses a checkpoint that was not trained on this exact dummy chapter, so a `--chapter` loss run cannot be silently scored as the sanity set.

### 3.5 Tests

`python -m pytest tests -q` → **24 passed** in about 3 seconds.

Scoring ([tests/test_scoring.py](/workspace/tests/test_scoring.py)):

- Favored-letter logits match full-vocab log-softmax, not raw logits.
- Tie breaks to the earlier letter.
- Chance comparison is strict at 0.25.
- `score_mcqs` counts only the gold letter.
- Multi-token continuations sum log-probabilities.
- The rendered prompt has no answer letter until `include_answer=True`.

12-gram filter ([tests/test_overlap.py](/workspace/tests/test_overlap.py)):

- `MATCH_TOKENS` is 12.
- A shared 12-word gram is contamination, including in the middle of a long document.
- An 11-gram is not.
- Case and punctuation are normalized; hyphens split; repeated whitespace does not create empty tokens; digits count as words.
- Documents shorter than 12 words on the eval side do not match.
- `filter_documents` drops only contaminated docs, and a hit against any eval item is enough.
- The default order is 12, so an 11-gram-only overlap is kept.

Locks and shape ([tests/test_model_and_locks.py](/workspace/tests/test_model_and_locks.py)):

- Protocol flags stay off.
- Yaml target is 12 / 12 / 768, `match_tokens` is 12, `p_new` is null, overfit steps are 400, scoring is `letter_logprob`.
- Debug parameter count is in 15–30M; target is in 100–140M; embeddings are tied; both models are marked from-scratch.
- Package sources do not call `from_pretrained(`, `AutoModelForCausalLM`, `import trl`, or a PPO optimizer.

### 3.6 Files added for WP0

| Path | Role |
| --- | --- |
| [sclm/model.py](/workspace/sclm/model.py) | GPT-2-like LM |
| [sclm/config.py](/workspace/sclm/config.py) | Yaml → `ModelConfig`; rejects a drifted target or `match_tokens` |
| [sclm/locks.py](/workspace/sclm/locks.py) | Named constants |
| [sclm/scoring.py](/workspace/sclm/scoring.py) | Letter log-prob MCQ scoring |
| [sclm/overlap.py](/workspace/sclm/overlap.py) | 12-gram filter |
| [sclm/tokenizer.py](/workspace/sclm/tokenizer.py) | GPT-2 BPE via `tiktoken` (vocabulary only, no weights) |
| [sclm/dummy_chapter.py](/workspace/sclm/dummy_chapter.py) | 8 original dummy items, all gold B |
| [sclm/checkpoint.py](/workspace/sclm/checkpoint.py) | From-scratch checkpoint gate |
| [configs/default.yaml](/workspace/configs/default.yaml) | Defaults and WP0 harness |
| [scripts/00_overfit_chapter.py](/workspace/scripts/00_overfit_chapter.py) | 400-step overfit |
| [scripts/00_eval_sanity.py](/workspace/scripts/00_eval_sanity.py) | 8-item sanity |
| [tests/](/workspace/tests) | Pytest |
| [AMENDMENTS.md](/workspace/AMENDMENTS.md) | Amendment 000 |
| [README.md](/workspace/README.md) | How to re-run WP0 |
| [requirements.txt](/workspace/requirements.txt), [pyproject.toml](/workspace/pyproject.toml) | Python deps |

Dependencies installed in this environment: `torch==2.14.1+cpu`, `tiktoken`, `pyyaml`, `pytest`. Python 3.10.

## 4. How to re-run

From the workspace root:

```bash
python -m pytest tests -q
python scripts/00_overfit_chapter.py
python scripts/00_eval_sanity.py
```

Expected: pytest 24 passed, overfit prints `PASS` with fixed-batch loss down, eval prints `PASS` with accuracy above 0.25.

## 5. What the next package still needs

WP0 is done. The next package should not start until the missing plan files are restored, because the handoff forbade inventing the method. In particular, before any Class 1–6 run:

1. Restore `08_HYPERPARAMETERS.md` and set `p_new` only from that file, via an amendment if the yaml changes.
2. Restore the data-pipeline note before any NCERT download. Download official PDFs locally. Do not publish the book text.
3. Build the scored MCQ bank as eval-only, then run the 12-gram filter so training text that shares a 12-gram with an item is dropped.
4. Train the from-scratch target (or the debug model) with next-token loss only.

Until those files exist, `p_new` stays null and no curriculum run should be launched.
