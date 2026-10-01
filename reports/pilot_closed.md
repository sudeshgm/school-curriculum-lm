# Pilot closed

Class 1 debug: loss 8.40 to 1.94, teacher-forced recall 1/64 to 25/64, letter exam 2/8 with every prediction C, numeric exam 1/8.

Class 6 debug, fresh init, 80 steps over 225 blocks: CPU loss 8.3863 to 3.9353, GPU loss 8.3863 to 3.9623, numeric exact match 0/8 to 0/8. After the GPU run, 6 of 8 guesses were "the".

Both runs were from scratch, corpus BPE vocab 4096, 4339200 params. No critic, no pretrained weights, p_new null, 12-layer model not trained.

Conclusion: next-token loss on the textbook fell, and the unseen numeric question stayed at chance.
