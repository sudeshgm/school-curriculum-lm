"""Letter-logprob scoring for 4-way MCQs.

The score of a letter is the summed next-token log-probability of the
continuation " {letter}" given the prompt. It is not a generation, not a
reward, and not a critic judgement. Ties break toward the earlier letter.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

LETTERS: tuple[str, ...] = ("A", "B", "C", "D")
CHANCE_ACCURACY = 0.25


def continuation_ids(tokenizer, prompt: str, continuation: str) -> tuple[list[int], list[int]]:
    """Token ids of `prompt` and of `continuation`, requiring a clean BPE boundary."""
    prefix = list(tokenizer.encode(prompt))
    full = list(tokenizer.encode(prompt + continuation))
    if full[: len(prefix)] != prefix:
        raise ValueError(
            "Tokenization boundary mismatch between prompt and continuation. "
            "Refusing to score a misaligned letter."
        )
    cont = full[len(prefix) :]
    if not cont:
        raise ValueError("Continuation produced no tokens")
    return prefix, cont


@torch.no_grad()
def continuation_logprob(model, prefix_ids: list[int], continuation_ids_: list[int]) -> float:
    """Sum of teacher-forced log-probabilities of continuation tokens."""
    if not continuation_ids_:
        raise ValueError("empty continuation")
    device = next(model.parameters()).device
    ids = prefix_ids + list(continuation_ids_)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _loss = model(idx)
    # logits[t] predicts token t+1. Drop the unused last position.
    logprobs = F.log_softmax(logits[0, :-1, :], dim=-1)
    total = 0.0
    start = len(prefix_ids)
    for offset, token_id in enumerate(continuation_ids_):
        position = start + offset
        total += float(logprobs[position - 1, token_id].item())
    return total


@torch.no_grad()
def letter_logprobs(
    model,
    tokenizer,
    prompt: str,
    letters: tuple[str, ...] = LETTERS,
) -> dict[str, float]:
    """Log-probability of each letter continuation given `prompt`."""
    scores: dict[str, float] = {}
    for letter in letters:
        prefix, cont = continuation_ids(tokenizer, prompt, " " + letter)
        scores[letter] = continuation_logprob(model, prefix, cont)
    return scores


def predict_letter(scores: dict[str, float], letters: tuple[str, ...] = LETTERS) -> str:
    """Argmax letter. On an exact tie, the earliest letter in `letters` wins."""
    best_letter = None
    best_lp = None
    for letter in letters:
        lp = scores[letter]
        if best_lp is None or lp > best_lp:
            best_letter = letter
            best_lp = lp
    if best_letter is None:
        raise ValueError("no letters to score")
    return best_letter


def beats_chance(accuracy: float, chance: float = CHANCE_ACCURACY) -> bool:
    return accuracy > chance


def render_mcq(item: dict, include_answer: bool) -> str:
    """Render one MCQ. The scored prompt ends at 'Answer:' with no letter yet."""
    lines = [f"Question: {item['question']}"]
    for letter in LETTERS:
        lines.append(f"{letter}. {item['choices'][letter]}")
    text = "\n".join(lines) + "\nAnswer:"
    if include_answer:
        text += f" {item['answer']}"
    return text


def score_mcqs(model, tokenizer, items: list[dict], prompt_prefix: str = "") -> dict:
    """Score items by letter log-probability. Returns accuracy against chance.

    `prompt_prefix` is optional in-context text. It is not a critic.
    """
    rows = []
    correct = 0
    for item in items:
        prompt = prompt_prefix + render_mcq(item, include_answer=False)
        scores = letter_logprobs(model, tokenizer, prompt)
        pred = predict_letter(scores)
        ok = pred == item["answer"]
        correct += int(ok)
        rows.append(
            {
                "id": item["id"],
                "answer": item["answer"],
                "pred": pred,
                "correct": ok,
                "logprobs": scores,
            }
        )
    n = len(items)
    accuracy = correct / n if n else 0.0
    return {
        "n": n,
        "correct": correct,
        "accuracy": accuracy,
        "chance": CHANCE_ACCURACY,
        "beats_chance": beats_chance(accuracy),
        "rows": rows,
    }
