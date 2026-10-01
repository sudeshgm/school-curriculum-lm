#!/usr/bin/env python3
"""WP5: new debug model, Class 6 blocks only. No letter scoring.

A first start uses random weights. If artifacts/wp5/class6_debug.pt exists,
a later start resumes those weights. The Class 1 checkpoint is never loaded.
Microbatches are 8 sequences. One optimizer step still covers all 225 blocks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sclm.checkpoint import load_from_scratch_checkpoint, save_checkpoint
from sclm.config import load_yaml, model_config_from_dict
from sclm.locks import ALLOW_CRITIC_IN_LOOP, ALLOW_PRETRAINED_LM_WEIGHTS, ALLOW_RL
from sclm.model import GPT
from sclm.overlap import shares_ngram
from sclm.tokenizer import load_tokenizer

BLOCKS = ROOT / "data" / "packed" / "class_6.npy"
OUT = ROOT / "artifacts" / "wp5"
CKPT = OUT / "class6_debug.pt"
OPT = OUT / "optimizer.pt"
STATE = OUT / "wp5_loss.json"
MAX_STEPS = 80
LOSS_STOP = 2.0
CHUNK = 8

QUESTIONS = [
    {"id": "q1", "prompt": "7 times 8 is", "number": 56},
    {"id": "q2", "prompt": "96 divided by 12 is", "number": 8},
    {"id": "q3", "prompt": "29 added to 46 is", "number": 75},
    {"id": "q4", "prompt": "3 fourths of 36 is", "number": 27},
    {"id": "q5", "prompt": "15 percent of 80 is", "number": 12},
    {"id": "q6", "prompt": "Negative 7 added to 12 is", "number": 5},
    {"id": "q7", "prompt": "A rectangle 9 by 6 has area", "number": 54},
    {"id": "q8", "prompt": "The number 11 squared is", "number": 121},
]


def number_target(tokenizer, prompt: str, number: int) -> tuple[list[int], list[int]]:
    prefix = tokenizer.encode(prompt)
    full = tokenizer.encode(prompt + " " + str(number))
    if full[: len(prefix)] != prefix:
        raise SystemExit(f"Token boundary mismatch for {prompt!r}")
    target = full[len(prefix) :]
    if not target:
        raise SystemExit(f"Number produced no tokens for {prompt!r}")
    return prefix, target


def assert_questions_clean(tokenizer) -> None:
    if len(QUESTIONS) != 8:
        raise SystemExit("WP5 needs 8 numeric prompts")
    blobs = [
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "data" / "clean" / "class_6").glob("chapter_*.txt"))
    ]
    if not blobs:
        raise SystemExit("Class 6 chapter text is missing")
    for item in QUESTIONS:
        sentence = f"{item['prompt']} {item['number']}"
        if any(mark in sentence for mark in ("A.", "B.", "C.", "D.", "Answer:")):
            raise SystemExit("A scored prompt contains a letter exam mark")
        number_target(tokenizer, item["prompt"], item["number"])
        for blob in blobs:
            if shares_ngram(blob, sentence) or shares_ngram(blob, item["prompt"]):
                raise SystemExit(f"{item['id']} shares a 12-gram with Class 6 text")


def chunks(data: torch.Tensor, device: torch.device):
    for start in range(0, data.shape[0], CHUNK):
        batch = data[start : start + CHUNK]
        x = batch[:, :-1].contiguous().to(device)
        y = batch[:, 1:].contiguous().to(device)
        yield x, y


@torch.no_grad()
def full_batch_loss(model, data: torch.Tensor, device: torch.device) -> float:
    model.eval()
    total = 0.0
    tokens = 0
    for x, y in chunks(data, device):
        _logits, loss = model(x, y)
        n = int(y.numel())
        total += float(loss) * n
        tokens += n
        del _logits, loss
    model.train()
    return total / tokens


def backward_full(model, data: torch.Tensor, device: torch.device) -> float:
    n_tokens = int(data.shape[0] * (data.shape[1] - 1))
    total = 0.0
    model.train()
    for x, y in chunks(data, device):
        logits, loss = model(x, y)
        n = int(y.numel())
        value = float(loss.detach())
        del logits
        (loss * (n / n_tokens)).backward()
        del loss
        total += value * n
    return total / n_tokens


@torch.no_grad()
def exact_rows(model, tokenizer, device: torch.device) -> list[dict]:
    model.eval()
    rows = []
    for item in QUESTIONS:
        prefix, target = number_target(tokenizer, item["prompt"], item["number"])
        ids = list(prefix)
        for _ in range(len(target)):
            step = torch.tensor([ids], dtype=torch.long, device=device)
            logits, _loss = model(step)
            ids.append(int(logits[0, -1].argmax()))
        guess = ids[len(prefix) :]
        rows.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "number": item["number"],
                "match": guess == target,
                "guess": tokenizer.decode(guess),
            }
        )
    return rows


def write_state(payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_partial(state: dict) -> None:
    path = ROOT / "reports" / "wp5_class6_partial.md"
    lines = [
        "# WP5 Class 6 partial",
        "",
        "The run did not finish. exact_after was not scored.",
        "A local checkpoint was saved. The Class 1 checkpoint was not loaded.",
        "No book text is printed.",
        "",
        f"Steps saved: {state['steps']}",
        f"Full-batch loss before: {state['loss_before']:.4f}",
        f"Full-batch loss at last saved step: {state['loss_last']:.4f}",
        f"Exact match before: {state['exact_before']}/8",
        "Exact match after: not scored",
        "",
    ]
    for row in state["before"]:
        lines.append(
            f"{row['id']} prompt {row['prompt']!r} gold {row['number']} "
            f"before {row['match']} guess {row['guess']!r}"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_progress(model, optimizer, state: dict) -> None:
    save_checkpoint(
        CKPT,
        model,
        extra={
            "wp": "5",
            "class": 6,
            "tokenizer": "corpus_bpe",
            "steps": state["steps"],
            "fresh_init": False,
            "resumed_from_fresh_debug": True,
        },
    )
    torch.save({"step": state["steps"], "optimizer": optimizer.state_dict()}, OPT)
    write_state(state)
    write_partial(state)
    print(f"saved step {state['steps']} loss {state['loss_last']:.4f}", flush=True)


def write_final(state: dict) -> None:
    path = ROOT / "reports" / "wp5_class6.md"
    lines = [
        "# WP5 Class 6 debug",
        "",
        "Debug model trained on Class 6 packed blocks only.",
        "Microbatches of 8. One optimizer step covers all 225 blocks.",
        "Next-token loss. Corpus BPE. Block size 512. No letter scoring.",
        "No book text is printed. p_new is null. No critic. No RL. No pretrained weights.",
        "Classes 1-5 were not trained. The 12-layer model was not trained.",
        "",
        f"Parameters: {state['params']}",
        f"Steps: {state['steps']}",
        f"Stop: {state['stop_reason']}",
        f"Full-batch loss before: {state['loss_before']:.4f}",
        f"Full-batch loss after: {state['loss_after']:.4f}",
        f"Exact match before: {state['exact_before']}/8",
        f"Exact match after: {state['exact_after']}/8",
        "",
    ]
    after_by_id = {row["id"]: row for row in state["after"]}
    for row in state["before"]:
        nxt = after_by_id[row["id"]]
        lines.append(
            f"{row['id']} prompt {row['prompt']!r} gold {row['number']} "
            f"before {row['match']} guess {row['guess']!r} "
            f"after {nxt['match']} guess {nxt['guess']!r}"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    write_paste(state)


def write_paste(state: dict) -> None:
    """Plain text for the other machine to paste back. Path is reports/wp5_gpu_paste.txt."""
    path = ROOT / "reports" / "wp5_gpu_paste.txt"
    lines = [
        "WP5_DONE: yes",
        f"device: {state.get('device', 'unknown')}",
        f"steps: {state['steps']}",
        f"stop: {state['stop_reason']}",
        f"loss_before: {state['loss_before']:.4f}",
        f"loss_after: {state['loss_after']:.4f}",
        f"exact_before: {state['exact_before']}/8",
        f"exact_after: {state['exact_after']}/8",
        "letter_scoring: no",
        "p_new: null",
        "pretrained_lm_weights: no",
        "critic: no",
    ]
    after_by_id = {row["id"]: row for row in state["after"]}
    for row in state["before"]:
        nxt = after_by_id[row["id"]]
        lines.append(
            f"{row['id']} gold {row['number']} before_guess {row['guess']!r} "
            f"before_match {row['match']} after_guess {nxt['guess']!r} after_match {nxt['match']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"paste_file {path}", flush=True)


def main() -> int:
    if ALLOW_PRETRAINED_LM_WEIGHTS or ALLOW_RL or ALLOW_CRITIC_IN_LOOP:
        raise SystemExit("Protocol locks violated")
    cfg = load_yaml(ROOT / "configs" / "default.yaml")
    if cfg["curriculum"]["p_new"] is not None:
        raise SystemExit("p_new must stay null")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    tokenizer = load_tokenizer(cfg, ROOT)
    assert_questions_clean(tokenizer)

    data = torch.from_numpy(np.load(BLOCKS).astype(np.int64))
    if tuple(data.shape) != (225, 512):
        raise SystemExit(f"Expected Class 6 blocks [225, 512], got {tuple(data.shape)}")
    if int(data.max()) >= 4096:
        raise SystemExit("Class 6 ids are outside the corpus vocab")

    train = cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    resumed = False
    if CKPT.is_file() and STATE.is_file():
        model, payload = load_from_scratch_checkpoint(CKPT, map_location="cpu")
        if payload.get("wp") != "5" or payload.get("class") != 6:
            raise SystemExit("Refusing to resume a checkpoint that is not WP5 Class 6")
        state = json.loads(STATE.read_text(encoding="utf-8"))
        start_step = int(state["steps"])
        resumed = True
        model.to(device)
        print(f"resume step {start_step} loss {state['loss_last']:.4f}", flush=True)
    else:
        torch.manual_seed(int(cfg["seed"]))
        model = GPT(model_config_from_dict(cfg, "debug"))
        if not model.from_scratch or model.config.n_layer != 4 or model.config.vocab_size != 4096:
            raise SystemExit("WP5 requires a new debug model")
        model.to(device)
        before = exact_rows(model, tokenizer, device)
        loss_before = full_batch_loss(model, data, device)
        state = {
            "steps": 0,
            "loss_before": loss_before,
            "loss_last": loss_before,
            "loss_curve": [],
            "exact_before": sum(int(row["match"]) for row in before),
            "before": before,
            "params": model.num_parameters(),
            "resumed": False,
            "device": str(device),
            "p_new": None,
        }
        write_state(state)
        start_step = 0
        print(
            f"fresh_init params {state['params']} loss_before {loss_before:.4f} "
            f"exact_before {state['exact_before']}/8",
            flush=True,
        )
        for row in before:
            print(f"before {row['id']} gold {row['number']} match {row['match']} guess {row['guess']!r}", flush=True)

    state["device"] = str(device)
    if model.config.n_layer != 4 or model.config.n_embd != 256 or model.config.n_head != 4:
        raise SystemExit("Loaded model is not the debug stack")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train["lr"]),
        betas=tuple(train["betas"]),  # type: ignore[arg-type]
        eps=float(train["eps"]),
        weight_decay=float(train["weight_decay"]),
    )
    if resumed and OPT.is_file():
        optimizer.load_state_dict(torch.load(OPT, map_location=device, weights_only=False)["optimizer"])
    elif resumed:
        print("optimizer moments restarted", flush=True)

    taken = start_step
    reason = "step_cap"
    for step in range(start_step + 1, MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        last_loss = backward_full(model, data, device)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train["grad_clip"]))
        optimizer.step()
        taken = step
        state["steps"] = step
        state["loss_last"] = last_loss
        state["loss_curve"].append({"step": step, "loss": last_loss})
        print(f"step {step} full_batch_loss {last_loss:.4f}", flush=True)
        if step % 10 == 0 or last_loss < LOSS_STOP:
            save_progress(model, optimizer, state)
        if last_loss < LOSS_STOP:
            reason = "loss_under_2"
            break

    loss_after = full_batch_loss(model, data, device)
    if loss_after < LOSS_STOP:
        reason = "loss_under_2"
    after = exact_rows(model, tokenizer, device)
    state.update(
        {
            "steps": taken,
            "stop_reason": reason,
            "loss_after": loss_after,
            "exact_after": sum(int(row["match"]) for row in after),
            "after": after,
            "finished": True,
        }
    )
    save_progress(model, optimizer, state)
    write_final(state)
    print(
        f"loss_after {loss_after:.4f} steps {taken} reason {reason} "
        f"exact_after {state['exact_after']}/8",
        flush=True,
    )
    for row in after:
        print(f"after {row['id']} gold {row['number']} match {row['match']} guess {row['guess']!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
