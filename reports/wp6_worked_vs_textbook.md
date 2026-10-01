# WP6 worked solutions versus textbook

Fresh debug models, seed 1337. Classes 1 and 2 only. No letter scoring.
Condition A mixes Class 1 and Class 2 evenly: Class 1 is repeated from 29 blocks to 43, then concatenated with 43 Class 2 blocks.
Condition B uses those same blocks plus an equal number of tokens from 40 original worked sentences.
No book text is printed. p_new is null. No critic. No pretrained weights. The 12-layer model was not trained.

Parameters: 4339200
A blocks per step: 86
B blocks per step: 172

A steps: 80
A stop: step_cap
A loss before: 8.4051
A loss after: 3.3384
A exact match before: 0/8
A exact match after: 0/8

B steps: 80
B stop: step_cap
B loss before: 8.4055
B loss after: 2.1577
B loss is the mixed batch, including the repeated sentences. It is not textbook-only loss.
B exact match before: 0/8
B exact match after: 0/8

The claim is dead at this scale.

Condition A gold versus guess:
A n1 prompt '3 buttons and 2 buttons make' gold 5 before False guess ' make' after False guess '.'
A n2 prompt '5 cups and 5 cups make' gold 10 before False guess ' make' after False guess ' the'
A n3 prompt '6 leaves and 4 leaves make' gold 10 before False guess ' make' after False guess ' the'
A n4 prompt '7 stones and 6 stones make' gold 13 before False guess ' make' after False guess ' a'
A n5 prompt '8 birds and 7 birds make' gold 15 before False guess ' make' after False guess ' a'
A n6 prompt '9 dots and 2 dots make' gold 11 before False guess ' make' after False guess ' 6'
A n7 prompt '8 kites and 3 kites make' gold 11 before False guess ' make' after False guess ' a'
A n8 prompt '9 hats and 9 hats make' gold 18 before False guess ' make' after False guess ' the'

Condition B gold versus guess:
B n1 prompt '3 buttons and 2 buttons make' gold 5 before False guess ' make' after False guess ' 9'
B n2 prompt '5 cups and 5 cups make' gold 10 before False guess ' make' after False guess ' 9'
B n3 prompt '6 leaves and 4 leaves make' gold 10 before False guess ' make' after False guess ' 9'
B n4 prompt '7 stones and 6 stones make' gold 13 before False guess ' make' after False guess ' the'
B n5 prompt '8 birds and 7 birds make' gold 15 before False guess ' make' after False guess ' 9'
B n6 prompt '9 dots and 2 dots make' gold 11 before False guess ' make' after False guess ' 5'
B n7 prompt '8 kites and 3 kites make' gold 11 before False guess ' make' after False guess ' 10'
B n8 prompt '9 hats and 9 hats make' gold 18 before False guess ' make' after False guess ' 15'
