# WP3c Class 1 probe

Recall and letter prior use the WP3b checkpoint before any further steps.
No decoded textbook span is included. Classes 2–6 and the 12-layer model were not trained.
Corpus BPE. `p_new` is null. No critic. No pretrained weights.

Spans: 8 windows, 120 tokens of context, last 8 hidden. Seed 1337.
Coordinates are packed-block index and start. They are not book text.
Span coordinates: [(20, 131), (24, 106), (4, 190), (14, 216), (12, 344), (27, 91), (10, 2), (17, 200)]

Untrained exact match: 0/8 = 0.000
Untrained greedy token accuracy: 0.000
Untrained teacher-forced token accuracy: 0.016
Checkpoint exact match: 0/8 = 0.000
Checkpoint greedy token accuracy: 0.141
Checkpoint teacher-forced token accuracy: 0.391

Logprob of the continuation after the prompt Answer:
prior_untrained: A -9.2824, B -7.6656, C -8.0898, D -8.1188
prior_checkpoint: A -8.8196, B -6.9373, C -8.6128, D -9.2266

Then 40 steps resumed that checkpoint. AdamW moments restarted.
Each step mixed 29 Class 1 blocks with 29 blocks made by repeating the 20 exemplar sentences.
No new textbook text was added.
Last mixed-batch loss: 0.9149
Zero-shot after the mix: 2/8 = 0.250
Chance: 0.25

Zero-shot gold vs predicted:
- c1 gold B pred C
- c2 gold C pred C
- c3 gold A pred C
- c4 gold D pred C
- c5 gold A pred C
- c6 gold B pred C
- c7 gold C pred C
- c8 gold D pred C
