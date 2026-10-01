"""Architecture flag, parameter bands, and protocol locks."""

from pathlib import Path

import yaml

from sclm.config import load_yaml, model_config_from_dict
from sclm.locks import (
    ALLOW_CRITIC_IN_LOOP,
    ALLOW_PRETRAINED_LM_WEIGHTS,
    ALLOW_RL,
    MATCH_TOKENS,
    TRAIN_ON_SCORED_MCQ_BANK,
)
from sclm.model import GPT

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_locks_are_off():
    assert ALLOW_PRETRAINED_LM_WEIGHTS is False
    assert ALLOW_CRITIC_IN_LOOP is False
    assert ALLOW_RL is False
    assert TRAIN_ON_SCORED_MCQ_BANK is False
    assert MATCH_TOKENS == 12


def test_default_yaml_locks_target_and_leaves_p_new_unset():
    cfg = load_yaml(ROOT / "configs" / "default.yaml")
    assert cfg["model"]["target"] == {"n_layer": 12, "n_head": 12, "n_embd": 768}
    assert cfg["model"]["debug"] == {"n_layer": 4, "n_head": 4, "n_embd": 256}
    assert cfg["model"]["vocab_size"] == 4096
    assert cfg["model"]["block_size"] == 512
    assert cfg["tokenizer"]["name"] == "corpus_bpe"
    assert cfg["decontamination"]["match_tokens"] == 12
    assert cfg["curriculum"]["p_new"] is None
    assert cfg["wp0"]["overfit_steps"] == 400
    assert cfg["wp3"]["class"] == 1
    assert cfg["wp3"]["model_variant"] == "debug"
    assert cfg["eval"]["scoring"] == "letter_logprob"
    assert cfg["model"]["tie_embeddings"] is True


def test_debug_and_target_parameter_bands():
    cfg = load_yaml(ROOT / "configs" / "default.yaml")
    debug = GPT(model_config_from_dict(cfg, "debug"))
    target = GPT(model_config_from_dict(cfg, "target"))
    debug_n = debug.num_parameters()
    target_n = target.num_parameters()
    # Amendment 002: same depth and width, vocab 4096, block 512.
    # Counts are lower than the old 50257-vocab models. Do not widen to chase them.
    assert debug_n == 4_339_200, debug_n
    assert target_n == 88_594_944, target_n
    assert debug.from_scratch is True
    assert target.from_scratch is True
    assert debug.lm_head.weight is debug.transformer.wte.weight
    assert target.lm_head.weight is target.transformer.wte.weight


def test_model_source_has_no_pretrained_loader():
    src = (ROOT / "sclm" / "model.py").read_text(encoding="utf-8")
    assert "from_pretrained" not in src
    forbidden = ("PPO", "GRPO", "AutoModelForCausalLM")
    for token in forbidden:
        assert token not in src


def test_package_sources_do_not_call_rl_or_pretrained_loaders():
    blob = []
    for path in (ROOT / "sclm").glob("*.py"):
        blob.append(path.read_text(encoding="utf-8"))
    for path in (ROOT / "scripts").glob("00_*.py"):
        blob.append(path.read_text(encoding="utf-8"))
    text = "\n".join(blob)
    assert "from_pretrained(" not in text
    assert "AutoModelForCausalLM" not in text
    assert "import trl" not in text
    assert "torch.optim.PPO" not in text


def test_yaml_roundtrip_matches_loader():
    raw = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    assert raw["model"]["debug"]["n_embd"] == 256
    assert raw["model"]["debug"]["n_layer"] == 4
