# WP7 forgetting

Fresh debug model, seed 1337. Class 1 blocks, then Class 2 blocks only. No Class 1 replay.
Eight Class 1 spans were held out of the Class 1 loss. The spans are not printed.
Score is teacher-forced accuracy on the last 8 tokens of each span.
No addition exam. No critic. No pretrained weights. p_new is null.
Classes 3-6 were not trained. The 12-layer model was not trained.

Parameters: 4339200
Class 1 steps: 80
Class 1 stop: step_cap
Class 1 loss before: 8.4012
Class 1 loss after: 2.0290
Class 1 teacher-forced accuracy: 16/64 = 0.2500

Class 2 steps: 62
Class 2 stop: loss_under_2
Class 2 loss before: 6.1589
Class 2 loss after: 1.9247
Same spans after Class 2: 9/64 = 0.1406

Class 1 recall dropped after Class 2 training with no replay. That is the forgetting result.
