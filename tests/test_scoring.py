"""Letter-logprob scoring. No training, no pretrained weights."""

import math

import torch
import torch.nn as nn

from sclm.scoring import (
    CHANCE_ACCURACY,
    beats_chance,
    continuation_logprob,
    letter_logprobs,
    predict_letter,
    render_mcq,
    score_mcqs,
)


class StubTokenizer:
    def encode(self, text: str):
        table = {
            "Q": [1],
            "Q A": [1, 10],
            "Q B": [1, 11],
            "Q C": [1, 12],
            "Q D": [1, 13],
            "Q AA": [1, 10, 10],
        }
        if text not in table:
            raise KeyError(text)
        return list(table[text])


class FavoredLetterModel(nn.Module):
    """Every position assigns a high logit to `favored_id` and 0 elsewhere."""

    def __init__(self, vocab: int, favored_id: int, favored_logit: float = 5.0):
        super().__init__()
        self.vocab = vocab
        self.favored_id = favored_id
        self.favored_logit = favored_logit
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, idx, targets=None):
        bsz, seq = idx.shape
        logits = torch.zeros(bsz, seq, self.vocab)
        logits[..., self.favored_id] = self.favored_logit
        return logits, None


def test_letter_logprob_picks_favored_letter_and_is_a_log_probability():
    vocab = 20
    favored = 11  # token id of " B" in StubTokenizer
    model = FavoredLetterModel(vocab, favored, favored_logit=5.0)
    scores = letter_logprobs(model, StubTokenizer(), "Q")
    assert predict_letter(scores) == "B"
    expected = 5.0 - math.log(math.exp(5.0) + (vocab - 1))
    assert abs(scores["B"] - expected) < 1e-5
    for letter in ("A", "C", "D"):
        assert scores[letter] < scores["B"]
        assert scores[letter] < 0.0
    assert scores["B"] < 0.0


def test_predict_letter_tie_breaks_to_earlier_letter():
    scores = {"A": -1.0, "B": -1.0, "C": -1.0, "D": -2.0}
    assert predict_letter(scores) == "A"


def test_beats_chance_is_strict():
    assert CHANCE_ACCURACY == 0.25
    assert beats_chance(0.25) is False
    assert beats_chance(0.2500001) is True
    assert beats_chance(0.0) is False
    assert beats_chance(1.0) is True


def test_score_mcqs_counts_only_gold_letter():
    model = FavoredLetterModel(20, 11)
    items = [
        {
            "id": "a",
            "question": "q",
            "choices": {"A": "1", "B": "2", "C": "3", "D": "4"},
            "answer": "B",
        },
        {
            "id": "b",
            "question": "q2",
            "choices": {"A": "1", "B": "2", "C": "3", "D": "4"},
            "answer": "A",
        },
    ]

    class AlwaysQ(StubTokenizer):
        def encode(self, text: str):
            # render_mcq prompts are longer than "Q". Map every prompt to the
            # stub context, and letter continuations by their final character.
            if text.endswith(" A"):
                return [1, 10]
            if text.endswith(" B"):
                return [1, 11]
            if text.endswith(" C"):
                return [1, 12]
            if text.endswith(" D"):
                return [1, 13]
            return [1]

    result = score_mcqs(model, AlwaysQ(), items)
    assert result["n"] == 2
    assert result["correct"] == 1
    assert result["accuracy"] == 0.5
    assert result["beats_chance"] is True
    assert result["rows"][0]["pred"] == "B"
    assert result["rows"][1]["pred"] == "B"


def test_multitoken_continuation_sums_logprobs():
    class TwoToken(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))

        def forward(self, idx, targets=None):
            bsz, seq = idx.shape
            logits = torch.full((bsz, seq, 20), -100.0)
            # Make token 10 the only likely token at every position.
            logits[..., 10] = 0.0
            return logits, None

    # prefix [1], continuation [10, 10]. Each step's logprob is log_softmax of
    # a one-hot-at-10 distribution, i.e. about 0, so the sum is about 0.
    total = continuation_logprob(TwoToken(), [1], [10, 10])
    assert abs(total - 0.0) < 1e-4


def test_render_mcq_prompt_has_no_answer_until_requested():
    item = {
        "id": "d1",
        "question": "Name?",
        "choices": {"A": "no", "B": "yes", "C": "maybe", "D": "later"},
        "answer": "B",
    }
    prompt = render_mcq(item, include_answer=False)
    answered = render_mcq(item, include_answer=True)
    assert prompt.endswith("Answer:")
    assert not prompt.endswith("B")
    assert answered.endswith("Answer: B")
    assert "B. yes" in prompt
