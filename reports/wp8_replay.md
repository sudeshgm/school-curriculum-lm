# WP8 replay

Fresh debug model, seed 1337. The WP7 checkpoint was not loaded.
The same 8 Class 1 spans were held out of the loss. The spans are not printed.
After Class 1, training continued on a mix: 70 percent of the loss is all Class 2 blocks, 30 percent is all Class 1 blocks with the held-out targets still masked.
The mix stops on Class 2 full-batch loss, not on the mixed loss.
No critic. No pretrained weights. The tokenizer was not changed. p_new is null.
Classes 3-6 were not trained. The matrix was not started.

Parameters: 4339200
Class 1 steps: 80
Class 1 stop: step_cap
Class 1 loss before: 8.4012
Class 1 loss after: 2.0290
Class 1 teacher-forced accuracy: 16/64

Mix steps: 64
Mix stop: loss_under_2
Class 2 loss before the mix: 6.1589
Class 2 loss after the mix: 1.9351
Same spans after the mix: 9/64

WP7 no replay: 16/64 then 9/64.
WP8 replay drop: 16/64 to 9/64.

30 percent replay did not help at this budget.
