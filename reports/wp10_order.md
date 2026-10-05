# WP10 order

Same 8 Class 3 spans as WP9. The spans are not printed. Class 3 text was not in training.
S is a fresh debug model, seed 1337, trained on Class 1 then Class 2 with no replay.
O is a fresh debug model, seed 1337. Class 1 and Class 2 blocks are shuffled once and repeated.
O uses the same optimizer-step count as S. Each O step has the same number of blocks, and therefore the same number of tokens, as the matching S step.
No critic. No pretrained weights. p_new is null. Classes 4-6 were not trained. The matrix was not started.

WP9 reference, not rescored here: R 0/64, S 8/64, T 16/64.

S Class 1 steps: 80
S Class 1 stop: step_cap
S Class 1 loss before: 8.4022
S Class 1 loss after: 2.2476
S Class 2 steps: 63
S Class 2 stop: loss_under_2
S Class 2 loss before: 6.0953
S Class 2 loss after: 1.9443
S teacher-forced accuracy: 8/64

O steps: 143
O tokens: 2569819
S tokens: 2569819
O loss before: 8.4059
O loss after: 2.2974
O teacher-forced accuracy: 5/64

Order helped.
