# WP5 locked

Class 6 debug model, fresh random init, 4339200 params, corpus BPE vocab 4096, block size 512.

80 optimizer steps, each step covering all 225 Class 6 blocks. Stop was the step cap, not loss below 2.

CPU loss 8.3863 to 3.9353, exact match 0/8 to 0/8.

GPU loss 8.3863 to 3.9623, exact match 0/8 to 0/8. After training, 6 of 8 guesses were "the".

No critic, no pretrained weights, no letter scoring, p_new null, Classes 1–5 not trained, 12-layer model not trained.

Book text and checkpoints were not pushed.
