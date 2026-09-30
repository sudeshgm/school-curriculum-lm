"""School-Curriculum LM (from-scratch GPT-2-like learner).

Protocol locks live in sclm.locks. WP0 does not train on a scored MCQ bank,
does not load pretrained LM weights, and does not run a critic or RL.
"""

__version__ = "0.0.1"
