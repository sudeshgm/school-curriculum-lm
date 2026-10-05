#!/usr/bin/env python3
"""WP12: continue-pretrain SmolLM2-360M base on Class 1 then Class 2.

Does not load the 4.3M debug model. Does not resume an earlier work-package
checkpoint. Class 3 text is scored and is not trained on.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sclm.config import load_yaml
from sclm.locks import ALLOW_CRITIC_IN_LOOP, ALLOW_RL
from sclm.overlap import shares_ngram
from sclm.tokenizer import load_tokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
MAX_STEPS = 200
PLATEAU_STEPS = 20
PLATEAU_DROP = 0.01
BLOCK = 512
LR = 2.0e-5
OUT = ROOT / "artifacts" / "wp12"
QUESTIONS = [
    {"id": "n1", "prompt": "3 buttons and 2 buttons make", "number": 5},
    {"id": "n2", "prompt": "5 cups and 5 cups make", "number": 10},
    {"id": "n3", "prompt": "6 leaves and 4 leaves make", "number": 10},
    {"id": "n4", "prompt": "7 stones and 6 stones make", "number": 13},
    {"id": "n5", "prompt": "8 birds and 7 birds make", "number": 15},
    {"id": "n6", "prompt": "9 dots and 2 dots make", "number": 11},
    {"id": "n7", "prompt": "8 kites and 3 kites make", "number": 11},
    {"id": "n8", "prompt": "9 hats and 9 hats make", "number": 18},
]


def load_wp7():
    spec = importlib.util.spec_from_file_location("wp7", ROOT / "scripts" / "07_forgetting.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chapter_ids(tokenizer, klass: int) -> list[int]:
    folder = ROOT / "data" / "clean" / f"class_{klass}"
    paths = sorted(folder.glob("chapter_*.txt"))
    if not paths:
        raise SystemExit(f"Class {klass} text is missing")
    ids = []
    for path in paths:
        ids.extend(tokenizer.encode(path.read_text(encoding="utf-8"), add_special_tokens=False))
        ids.extend(tokenizer.encode("\n\n", add_special_tokens=False))
    if len(ids) < BLOCK:
        raise SystemExit(f"Class {klass} is shorter than one block")
    return ids


def pack(ids: list[int]) -> torch.Tensor:
    rows = len(ids) // BLOCK
    data = torch.tensor(ids[: rows * BLOCK], dtype=torch.long).view(rows, BLOCK)
    return data


def number_target(tokenizer, prompt: str, number: int) -> tuple[list[int], list[int]]:
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    full = tokenizer.encode(prompt + " " + str(number), add_special_tokens=False)
    if full[: len(prefix)] != prefix or len(full) == len(prefix):
        raise SystemExit(f"Token boundary mismatch for {prompt!r}")
    return prefix, full[len(prefix) :]


def assert_held_out() -> None:
    blobs = []
    for klass in (1, 2):
        paths = sorted((ROOT / "data" / "clean" / f"class_{klass}").glob("chapter_*.txt"))
        blobs.extend(path.read_text(encoding="utf-8") for path in paths)
    for item in QUESTIONS:
        sentence = f"{item['prompt']} {item['number']}"
        for blob in blobs:
            if shares_ngram(blob, sentence) or shares_ngram(blob, item["prompt"]):
                raise SystemExit(f"{item['id']} shares a 12-gram with Class 1 or 2 text")


def class3_spans(corpus, smol) -> list[dict]:
    wp7 = load_wp7()
    blocks = wp7.load_blocks(3)
    spans = [[int(block), int(start)] for block, start in wp7.choose_spans(blocks.shape[0])]
    masked = int((wp7.heldout_targets(blocks, spans) == -1).sum().item())
    if len(spans) != 8 or masked != 1024:
        raise SystemExit("Class 3 spans do not match the WP9 geometry")
    pieces = []
    for block, start in spans:
        window = blocks[block, start : start + 128].astype(int).tolist()
        prefix_text = corpus.decode(window[:120])
        hidden_text = corpus.decode(window[120:])
        if corpus.decode(window) != prefix_text + hidden_text:
            raise SystemExit("Corpus span did not split cleanly")
        batch = smol(
            prefix_text + hidden_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        cut = len(prefix_text)
        ids = list(batch["input_ids"])
        offsets = list(batch["offset_mapping"])
        target_at = [index for index, pair in enumerate(offsets) if pair[0] >= cut]
        if not target_at or target_at[0] == 0:
            raise SystemExit("SmolLM span has no hidden tokens")
        first = target_at[0]
        if target_at != list(range(first, len(ids))):
            raise SystemExit("SmolLM hidden tokens are not a suffix")
        pieces.append({"prefix": ids[:first], "target": ids[first:]})
    return pieces


def load_model(device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    if "instruct" in MODEL_ID.lower() or getattr(model.config, "_name_or_path", "").lower().endswith("instruct"):
        raise SystemExit("Refusing an instruct checkpoint")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to(device)
    return tokenizer, model


@torch.no_grad()
def full_batch_loss(model, blocks: torch.Tensor, device: torch.device) -> float:
    model.eval()
    total = 0.0
    tokens = 0
    for row in blocks:
        batch = row.unsqueeze(0).to(device)
        out = model(input_ids=batch, labels=batch)
        n = int(batch.numel() - 1)
        total += float(out.loss) * n
        tokens += n
        del out
    model.train()
    return total / tokens


def backward_full(model, blocks: torch.Tensor, device: torch.device) -> float:
    n_tokens = int(blocks.shape[0] * (blocks.shape[1] - 1))
    total = 0.0
    model.train()
    for row in blocks:
        batch = row.unsqueeze(0).to(device)
        out = model(input_ids=batch, labels=batch)
        n = int(batch.numel() - 1)
        (out.loss * (n / n_tokens)).backward()
        total += float(out.loss.detach()) * n
        del out
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
            logits = model(input_ids=step).logits
            ids.append(int(logits[0, -1].argmax()))
            del logits
        guess_ids = ids[len(prefix) :]
        rows.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "number": item["number"],
                "match": guess_ids == target,
                "guess": tokenizer.decode(guess_ids),
            }
        )
    model.train()
    return rows


@torch.no_grad()
def span_accuracy(model, spans: list[dict], device: torch.device) -> tuple[int, int]:
    model.eval()
    hits = 0
    total = 0
    for span in spans:
        ids = span["prefix"] + span["target"]
        batch = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(input_ids=batch).logits[0]
        start = len(span["prefix"]) - 1
        pred = logits[start : start + len(span["target"])].argmax(dim=-1).tolist()
        hits += sum(int(a == b) for a, b in zip(pred, span["target"]))
        total += len(span["target"])
        del logits
    model.train()
    return hits, total


def train_phase(name: str, model, optimizer, blocks: torch.Tensor, device: torch.device, state: dict) -> dict:
    if state.get("finished"):
        print(f"{name} already finished at step {state['steps']}", flush=True)
        return state
    losses = list(state.get("losses", []))
    reason = "step_cap"
    for step in range(int(state["steps"]) + 1, MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        last = backward_full(model, blocks, device)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(last)
        state["steps"] = step
        state["loss_last"] = last
        state["losses"] = losses
        print(f"{name} step {step} full_batch_loss {last:.4f}", flush=True)
        if step % 10 == 0:
            save_state(name, model, optimizer, state)
        if step >= PLATEAU_STEPS and min(losses[-5:]) > losses[-PLATEAU_STEPS] - PLATEAU_DROP:
            reason = "plateau"
            break
    state["stop_reason"] = reason
    state["loss_after"] = full_batch_loss(model, blocks, device)
    state["finished"] = True
    save_state(name, model, optimizer, state)
    print(f"{name} loss_after {state['loss_after']:.4f} reason {reason}", flush=True)
    return state


def save_state(name: str, model, optimizer, state: dict) -> None:
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "wp": "12", "model_id": MODEL_ID, "phase": name},
        folder / "model.pt",
    )
    torch.save({"optimizer": optimizer.state_dict(), "step": state["steps"]}, folder / "optimizer.pt")
    public = {key: value for key, value in state.items() if key != "losses"}
    public["loss_tail"] = state.get("losses", [])[-5:]
    (folder / "state.json").write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")


def new_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)


def score_pair(model, tokenizer, spans, device: torch.device) -> dict:
    rows = exact_rows(model, tokenizer, device)
    hits, total = span_accuracy(model, spans, device)
    exact = sum(int(row["match"]) for row in rows)
    print(f"numeric {exact}/8 spans {hits}/{total}", flush=True)
    for row in rows:
        print(f"{row['id']} gold {row['number']} match {row['match']} guess {row['guess']!r}", flush=True)
    return {"exact": exact, "rows": rows, "span_hits": hits, "span_total": total}


def conclusion(exact_after: int) -> str:
    if exact_after == 0:
        return "The books are the bottleneck."
    return "The from-scratch failure was language."


def write_report(meta: dict, before: dict, class1: dict, class2: dict, after: dict) -> None:
    lines = [
        "# WP12 SmolLM2-360M",
        "",
        "Learner is HuggingFaceTB/SmolLM2-360M base. Not a math-tuned checkpoint. Not random weights. Not the 4.3M model.",
        "No earlier checkpoint was resumed. Class 1 chapter text, then Class 2 chapter text, retokenized with the SmolLM2 tokenizer and packed to 512. No replay. Class 3 text was not trained on.",
        "Numeric prompts are the eight held-out WP6 prompts. The Class 3 spans are the same eight windows as WP9. Span text is not printed.",
        "No critic. p_new is null. The matrix was not started.",
        "",
        f"Parameters: {meta['params']}",
        f"Class 1 blocks: {meta['class1_blocks']}",
        f"Class 2 blocks: {meta['class2_blocks']}",
        f"Class 1 steps: {class1['steps']}",
        f"Class 1 stop: {class1['stop_reason']}",
        f"Class 1 loss before: {class1['loss_before']:.4f}",
        f"Class 1 loss after: {class1['loss_after']:.4f}",
        f"Class 2 steps: {class2['steps']}",
        f"Class 2 stop: {class2['stop_reason']}",
        f"Class 2 loss before: {class2['loss_before']:.4f}",
        f"Class 2 loss after: {class2['loss_after']:.4f}",
        "",
        f"Numeric exact match before: {before['exact']}/8",
        f"Numeric exact match after: {after['exact']}/8",
        f"Class 3 span tokens before: {before['span_hits']}/{before['span_total']}",
        f"Class 3 span tokens after: {after['span_hits']}/{after['span_total']}",
        "",
        conclusion(after["exact"]),
        "",
        "Numeric gold versus guess:",
    ]
    after_rows = {row["id"]: row for row in after["rows"]}
    for row in before["rows"]:
        nxt = after_rows[row["id"]]
        lines.append(
            f"{row['id']} prompt {row['prompt']!r} gold {row['number']} "
            f"before {row['match']} guess {row['guess']!r} "
            f"after {nxt['match']} guess {nxt['guess']!r}"
        )
    lines.append("")
    path = ROOT / "reports" / "wp12_smollm.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    paste = [
        f"numeric_before: {before['exact']}/8",
        f"numeric_after: {after['exact']}/8",
        f"spans_before: {before['span_hits']}/{before['span_total']}",
        f"spans_after: {after['span_hits']}/{after['span_total']}",
        f"class1_steps: {class1['steps']} {class1['stop_reason']}",
        f"class1_loss: {class1['loss_before']:.4f} to {class1['loss_after']:.4f}",
        f"class2_steps: {class2['steps']} {class2['stop_reason']}",
        f"class2_loss: {class2['loss_before']:.4f} to {class2['loss_after']:.4f}",
        conclusion(after["exact"]),
    ]
    (ROOT / "reports" / "wp12_smollm_paste.txt").write_text("\n".join(paste) + "\n", encoding="utf-8")


def main() -> int:
    if ALLOW_RL or ALLOW_CRITIC_IN_LOOP:
        raise SystemExit("Protocol locks violated")
    cfg = load_yaml(ROOT / "configs" / "default.yaml")
    if cfg["curriculum"]["p_new"] is not None:
        raise SystemExit("p_new must stay null")
    if not (ROOT / "reports" / "amendment_pretrained.md").is_file():
        raise SystemExit("Write the pretrained amendment before training")
    assert_held_out()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    corpus = load_tokenizer(cfg, ROOT)
    tokenizer, model = load_model(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"params {params} model {MODEL_ID}", flush=True)
    spans = class3_spans(corpus, tokenizer)
    class1 = pack(chapter_ids(tokenizer, 1))
    class2 = pack(chapter_ids(tokenizer, 2))
    print(f"class1 blocks {class1.shape[0]} class2 blocks {class2.shape[0]}", flush=True)
    before = score_pair(model, tokenizer, spans, device)
    (OUT / "before.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "before.json").write_text(json.dumps(before, indent=2) + "\n", encoding="utf-8")
    if device.type != "cuda":
        raise SystemExit(
            "Stopped before the first update. AdamW for SmolLM2-360M does not fit in this CPU memory. "
            "Run this script on a GPU and paste reports/wp12_smollm_paste.txt."
        )

    optimizer = new_optimizer(model)
    loss_before = full_batch_loss(model, class1, device)
    state1 = {"steps": 0, "loss_before": loss_before, "loss_last": loss_before, "losses": [], "finished": False}
    print(f"class1 loss_before {loss_before:.4f}", flush=True)
    state1 = train_phase("class1", model, optimizer, class1, device, state1)

    loss_before = full_batch_loss(model, class2, device)
    state2 = {"steps": 0, "loss_before": loss_before, "loss_last": loss_before, "losses": [], "finished": False}
    print(f"class2 loss_before {loss_before:.4f}", flush=True)
    state2 = train_phase("class2", model, optimizer, class2, device, state2)
    after = score_pair(model, tokenizer, spans, device)
    write_report(
        {"params": params, "class1_blocks": int(class1.shape[0]), "class2_blocks": int(class2.shape[0])},
        before,
        state1,
        state2,
        after,
    )
    print("report reports/wp12_smollm.md", flush=True)
    print(conclusion(after["exact"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
