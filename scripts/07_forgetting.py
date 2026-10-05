#!/usr/bin/env python3
"""WP7: Class 1 recall after Class 2 training, with no Class 1 replay.

Fresh debug init, seed 1337. Does not load a WP6 checkpoint.
The 8 eval spans are held out of the Class 1 loss and are never printed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sclm.checkpoint import load_from_scratch_checkpoint, save_checkpoint
from sclm.config import load_yaml, model_config_from_dict
from sclm.locks import ALLOW_CRITIC_IN_LOOP, ALLOW_PRETRAINED_LM_WEIGHTS, ALLOW_RL
from sclm.model import GPT

MAX_STEPS = 80
LOSS_STOP = 2.0
CHUNK = 8
SEED = 1337
PREFIX = 120
HIDDEN = 8
WIDTH = PREFIX + HIDDEN
OUT = ROOT / "artifacts" / "wp7"


def load_blocks(klass: int) -> np.ndarray:
    data = np.load(ROOT / "data" / "packed" / f"class_{klass}.npy")
    if data.ndim != 2 or data.shape[1] != 512:
        raise SystemExit(f"Bad Class {klass} blocks: {data.shape}")
    if int(data.max()) >= 4096:
        raise SystemExit(f"Class {klass} ids are outside the corpus vocab")
    return data


def choose_spans(n_blocks: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(SEED)
    blocks = rng.choice(n_blocks, size=8, replace=False)
    starts = rng.integers(0, 512 - WIDTH, size=8)
    return [(int(block), int(start)) for block, start in zip(blocks, starts)]


def heldout_targets(blocks: np.ndarray, spans: list[tuple[int, int]]) -> torch.Tensor:
    targets = blocks[:, 1:].astype(np.int64).copy()
    for block, start in spans:
        left = max(0, start - 1)
        right = min(targets.shape[1], start + WIDTH - 1)
        targets[block, left:right] = -1
    return torch.from_numpy(targets)


def chunks(inputs: torch.Tensor, targets: torch.Tensor, device: torch.device):
    for start in range(0, inputs.shape[0], CHUNK):
        x = inputs[start : start + CHUNK].to(device)
        y = targets[start : start + CHUNK].to(device)
        yield x, y


def token_loss(model, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, int]:
    logits, _loss = model(x)
    n = int((y != -1).sum().item())
    summed = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1, reduction="sum")
    del logits
    return summed, n


@torch.no_grad()
def full_batch_loss(model, inputs: torch.Tensor, targets: torch.Tensor, device: torch.device) -> float:
    model.eval()
    total = 0.0
    tokens = 0
    for x, y in chunks(inputs, targets, device):
        summed, n = token_loss(model, x, y)
        total += float(summed)
        tokens += n
        del summed
    model.train()
    if tokens == 0:
        raise SystemExit("No training tokens")
    return total / tokens


def backward_full(model, inputs: torch.Tensor, targets: torch.Tensor, device: torch.device) -> float:
    n_tokens = int((targets != -1).sum().item())
    total = 0.0
    model.train()
    for x, y in chunks(inputs, targets, device):
        summed, n = token_loss(model, x, y)
        if n:
            (summed / n_tokens).backward()
        total += float(summed.detach())
        del summed
    return total / n_tokens


@torch.no_grad()
def teacher_forced(model, blocks: np.ndarray, spans: list[tuple[int, int]], device: torch.device) -> tuple[int, int]:
    model.eval()
    hits = 0
    for block, start in spans:
        window = blocks[block, start : start + WIDTH].astype(np.int64)
        ids = torch.from_numpy(window).unsqueeze(0).to(device)
        logits, _loss = model(ids)
        pred = logits[0, PREFIX - 1 : PREFIX - 1 + HIDDEN].argmax(dim=-1).tolist()
        gold = window[PREFIX:].tolist()
        hits += sum(int(a == b) for a, b in zip(pred, gold))
        del logits
    model.train()
    return hits, 8 * HIDDEN


def fresh_model(cfg, device: torch.device) -> GPT:
    torch.manual_seed(SEED)
    model = GPT(model_config_from_dict(cfg, "debug"))
    if (
        not model.from_scratch
        or model.config.n_layer != 4
        or model.config.n_head != 4
        or model.config.n_embd != 256
        or model.config.vocab_size != 4096
        or model.config.block_size != 512
    ):
        raise SystemExit("WP7 requires a fresh debug model")
    model.to(device)
    return model


def save_progress(phase: str, model, optimizer, state: dict) -> None:
    folder = OUT / phase
    folder.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        folder / "debug.pt",
        model,
        extra={"wp": "7", "phase": phase, "steps": state["steps"], "from_scratch": True},
    )
    torch.save({"step": state["steps"], "optimizer": optimizer.state_dict()}, folder / "optimizer.pt")
    (folder / "state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"{phase} saved step {state['steps']} loss {state['loss_last']:.4f}", flush=True)


def train_phase(
    phase: str,
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    cfg,
    device: torch.device,
    optimizer,
    state: dict,
) -> dict:
    if state.get("finished"):
        print(f"{phase} already finished at step {state['steps']}", flush=True)
        return state
    train = cfg["train"]
    start = int(state["steps"])
    reason = "step_cap"
    for step in range(start + 1, MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        last_loss = backward_full(model, inputs, targets, device)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train["grad_clip"]))
        optimizer.step()
        state["steps"] = step
        state["loss_last"] = last_loss
        print(f"{phase} step {step} full_batch_loss {last_loss:.4f}", flush=True)
        if step % 10 == 0 or last_loss < LOSS_STOP:
            save_progress(phase, model, optimizer, state)
        if last_loss < LOSS_STOP:
            reason = "loss_under_2"
            break
    loss_after = full_batch_loss(model, inputs, targets, device)
    if loss_after < LOSS_STOP:
        reason = "loss_under_2"
    state["stop_reason"] = reason
    state["loss_after"] = loss_after
    state["finished"] = True
    save_progress(phase, model, optimizer, state)
    print(f"{phase} loss_after {loss_after:.4f} reason {reason}", flush=True)
    return state


def new_optimizer(model, cfg):
    train = cfg["train"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(train["lr"]),
        betas=tuple(train["betas"]),  # type: ignore[arg-type]
        eps=float(train["eps"]),
        weight_decay=float(train["weight_decay"]),
    )


def load_phase(phase: str, cfg, device: torch.device):
    folder = OUT / phase
    model, payload = load_from_scratch_checkpoint(folder / "debug.pt", map_location="cpu")
    if payload.get("wp") != "7" or payload.get("phase") != phase:
        raise SystemExit(f"Refusing to load a checkpoint that is not WP7 {phase}")
    model.to(device)
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    optimizer = new_optimizer(model, cfg)
    opt_path = folder / "optimizer.pt"
    if opt_path.is_file():
        optimizer.load_state_dict(torch.load(opt_path, map_location=device, weights_only=False)["optimizer"])
    return model, optimizer, state


def write_report(class1: dict, class2: dict, hits_after_1: int, hits_after_2: int) -> None:
    acc1 = hits_after_1 / 64
    acc2 = hits_after_2 / 64
    if hits_after_2 < hits_after_1:
        claim = "Class 1 recall dropped after Class 2 training with no replay. That is the forgetting result."
    else:
        claim = "Class 1 recall did not drop after Class 2 training."
    lines = [
        "# WP7 forgetting",
        "",
        "Fresh debug model, seed 1337. Class 1 blocks, then Class 2 blocks only. No Class 1 replay.",
        "Eight Class 1 spans were held out of the Class 1 loss. The spans are not printed.",
        "Score is teacher-forced accuracy on the last 8 tokens of each span.",
        "No addition exam. No critic. No pretrained weights. p_new is null.",
        "Classes 3-6 were not trained. The 12-layer model was not trained.",
        "",
        f"Parameters: {class1['params']}",
        f"Class 1 steps: {class1['steps']}",
        f"Class 1 stop: {class1['stop_reason']}",
        f"Class 1 loss before: {class1['loss_before']:.4f}",
        f"Class 1 loss after: {class1['loss_after']:.4f}",
        f"Class 1 teacher-forced accuracy: {hits_after_1}/64 = {acc1:.4f}",
        "",
        f"Class 2 steps: {class2['steps']}",
        f"Class 2 stop: {class2['stop_reason']}",
        f"Class 2 loss before: {class2['loss_before']:.4f}",
        f"Class 2 loss after: {class2['loss_after']:.4f}",
        f"Same spans after Class 2: {hits_after_2}/64 = {acc2:.4f}",
        "",
        claim,
        "",
    ]
    path = ROOT / "reports" / "wp7_forgetting.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if ALLOW_PRETRAINED_LM_WEIGHTS or ALLOW_RL or ALLOW_CRITIC_IN_LOOP:
        raise SystemExit("Protocol locks violated")
    cfg = load_yaml(ROOT / "configs" / "default.yaml")
    if cfg["curriculum"]["p_new"] is not None:
        raise SystemExit("p_new must stay null")
    if int(cfg["seed"]) != SEED:
        raise SystemExit("Config seed is not 1337")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)

    class1 = load_blocks(1)
    class2 = load_blocks(2)
    if class1.shape[0] != 29 or class2.shape[0] != 43:
        raise SystemExit(f"Unexpected block counts {class1.shape[0]} {class2.shape[0]}")
    spans = choose_spans(class1.shape[0])
    # Coordinates stay in the local state. They are not book text and are not printed.
    inputs1 = torch.from_numpy(class1[:, :-1].astype(np.int64)).contiguous()
    targets1 = heldout_targets(class1, spans)
    masked = int((targets1 == -1).sum().item())
    print(f"class1 blocks {class1.shape[0]} heldout_targets {masked}", flush=True)

    phase1 = OUT / "class1"
    if (phase1 / "debug.pt").is_file() and (phase1 / "state.json").is_file():
        model, optimizer, state1 = load_phase("class1", cfg, device)
        if state1.get("spans") != spans:
            raise SystemExit("Saved Class 1 spans do not match this seed")
        print(f"class1 resume step {state1['steps']}", flush=True)
    else:
        model = fresh_model(cfg, device)
        loss_before = full_batch_loss(model, inputs1, targets1, device)
        state1 = {
            "steps": 0,
            "loss_before": loss_before,
            "loss_last": loss_before,
            "params": model.num_parameters(),
            "spans": spans,
            "finished": False,
        }
        optimizer = new_optimizer(model, cfg)
        print(f"fresh params {state1['params']} loss_before {loss_before:.4f}", flush=True)
    state1 = train_phase("class1", model, inputs1, targets1, cfg, device, optimizer, state1)
    hits1, total = teacher_forced(model, class1, spans, device)
    state1["teacher_forced_hits"] = hits1
    state1["teacher_forced_total"] = total
    save_progress("class1", model, optimizer, state1)
    print(f"after_class1 teacher_forced {hits1}/{total}", flush=True)

    inputs2 = torch.from_numpy(class2[:, :-1].astype(np.int64)).contiguous()
    targets2 = torch.from_numpy(class2[:, 1:].astype(np.int64))
    phase2 = OUT / "class2"
    if (phase2 / "debug.pt").is_file() and (phase2 / "state.json").is_file():
        model, optimizer, state2 = load_phase("class2", cfg, device)
        print(f"class2 resume step {state2['steps']}", flush=True)
    else:
        loss_before = full_batch_loss(model, inputs2, targets2, device)
        state2 = {
            "steps": 0,
            "loss_before": loss_before,
            "loss_last": loss_before,
            "params": state1["params"],
            "finished": False,
        }
        print(f"class2 continue loss_before {loss_before:.4f}", flush=True)
    state2 = train_phase("class2", model, inputs2, targets2, cfg, device, optimizer, state2)
    hits2, total2 = teacher_forced(model, class1, spans, device)
    if total2 != total:
        raise SystemExit("Span token count changed")
    state2["teacher_forced_hits"] = hits2
    save_progress("class2", model, optimizer, state2)
    print(f"after_class2 teacher_forced {hits2}/{total2}", flush=True)
    write_report(state1, state2, hits1, hits2)
    print("report reports/wp7_forgetting.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
