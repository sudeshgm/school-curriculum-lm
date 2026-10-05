# WP9 forward transfer

Eight Class 3 spans, 120 tokens of context and 8 hidden tokens. The spans are not printed.
R is a fresh debug model, seed 1337, with no training.
S is a fresh debug model, seed 1337, trained on Class 1 then Class 2 with no replay. Class 3 text was not in the training data.
T is a fresh debug model, seed 1337, trained on Class 3 only, with those spans masked out of the loss.
The score is teacher-forced accuracy. This is not an exam result.
No critic. No pretrained weights. p_new is null. Classes 4-6 were not trained. The matrix was not started.

R teacher-forced accuracy: 0/64

S Class 1 steps: 80
S Class 1 stop: step_cap
S Class 1 loss before: 8.4022
S Class 1 loss after: 2.2476
S Class 2 steps: 63
S Class 2 stop: loss_under_2
S Class 2 loss before: 6.0953
S Class 2 loss after: 1.9443
S teacher-forced accuracy: 8/64

T steps: 80
T stop: step_cap
T loss before: 8.4024
T loss after: 3.4496
T teacher-forced accuracy: 16/64

Earlier classes helped but did not replace Class 3.
This is not an exam result.
