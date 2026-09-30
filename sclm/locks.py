"""Locked protocol constants.

Do not change MATCH_TOKENS, the target architecture, or p_new without an
amendment note. p_new is intentionally not given a numeric value here:
08_HYPERPARAMETERS.md was not in the workspace at takeover (see AMENDMENTS.md).
"""

# 12-gram word overlap used to keep scored exam text out of training.
MATCH_TOKENS = 12

# Learner is always randomly initialized. Never load pretrained LM weights.
ALLOW_PRETRAINED_LM_WEIGHTS = False

# No critic / teacher LLM in the training loop.
ALLOW_CRITIC_IN_LOOP = False

# No RL, PPO, GRPO, or exam-as-reward training.
ALLOW_RL = False

# The scored MCQ bank is evaluation-only. WP0's 8 dummy items are not that bank.
TRAIN_ON_SCORED_MCQ_BANK = False

TARGET_N_LAYER = 12
TARGET_N_EMBD = 768
TARGET_N_HEAD = 12
