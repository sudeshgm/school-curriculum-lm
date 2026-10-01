# WP3b Class 1 debug

Continued the step-40 Class 1 checkpoint. Not a new random init.
Classes 2–6 were not trained. The 12-layer model was not trained.
Corpus BPE, vocab 4096. `p_new` is null. No critic. No pretrained weights.
AdamW moments restarted because the checkpoint did not store them.

Resume loss: 4.2995
Extra steps: 49 (stop: loss_under_2)
Total optimizer steps: 89
Extra token steps: 726131
Full-batch loss after: 1.9422
Last train loss: 1.9865

Untrained baseline: 4/8 = 0.500
Zero-shot after training: 2/8 = 0.250
Four-example in-context: 2/8 = 0.250
Chance: 0.25

Untrained gold vs predicted:
- c1 gold B pred B
- c2 gold C pred C
- c3 gold A pred C
- c4 gold D pred C
- c5 gold A pred C
- c6 gold B pred C
- c7 gold C pred C
- c8 gold D pred D

Zero-shot gold vs predicted:
- c1 gold B pred C
- c2 gold C pred C
- c3 gold A pred C
- c4 gold D pred C
- c5 gold A pred C
- c6 gold B pred C
- c7 gold C pred C
- c8 gold D pred C

In-context gold vs predicted:
- c1 gold B pred C
- c2 gold C pred C
- c3 gold A pred C
- c4 gold D pred C
- c5 gold A pred C
- c6 gold B pred C
- c7 gold C pred C
- c8 gold D pred C
