# WP11 seed 21

Same 8 Class 3 spans as WP9 and WP10. The spans are not printed. Class 3 text was not in training.
S is a fresh debug model, seed 21, trained on Class 1 then Class 2 with no replay.
O is a fresh debug model, seed 21. Class 1 and Class 2 blocks are shuffled once with seed 21 and repeated.
O matches S token for token: the same optimizer-step count, and the same number of blocks on each step.
No critic. No pretrained weights. p_new is null. Classes 4-6 were not trained. The matrix was not started.

WP10 pair: staged 8/64, shuffled 5/64.

S Class 1 steps: 80
S Class 1 stop: step_cap
S Class 1 loss before: 8.3189
S Class 1 loss after: 2.1201
S Class 2 steps: 60
S Class 2 stop: loss_under_2
S Class 2 loss before: 6.1720
S Class 2 loss after: 1.9587
S teacher-forced accuracy: 6/64

O steps: 140
O tokens: 2503900
S tokens: 2503900
O loss before: 8.3217
O loss after: 2.4202
O teacher-forced accuracy: 4/64

The WP10 gap did not repeat.
