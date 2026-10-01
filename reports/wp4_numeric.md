# WP4 numeric exact match

No letter scoring. Resumed the WP3c Class 1 checkpoint for 40 steps.
Each step mixed 29 Class 1 blocks with 29 blocks made by repeating 20 original sentences.
The 8 prompts are not those sentences. Exact match is the greedy continuation of the number.
Corpus BPE. p_new is null. No critic. No pretrained weights.
Classes 2-6 were not trained. The 12-layer model was not trained.

Exact match before: 1/8
Exact match after: 1/8
Last mixed-batch loss: 0.4047
The mixed loss includes the repeated sentences, so it is not textbook loss.

n1 prompt '3 buttons and 2 buttons make' gold 5 before False guess ' 9' after False guess ' 9'
n2 prompt '1 cups and 1 cups make' gold 2 before False guess ' 9' after False guess ' 7'
n3 prompt '4 leaves and 3 leaves make' gold 7 before False guess ' 9' after True guess ' 7'
n4 prompt '5 stones and 3 stones make' gold 8 before False guess ' 9' after False guess ' 7'
n5 prompt '5 birds and 4 birds make' gold 9 before True guess ' 9' after False guess ' 5'
n6 prompt '6 dots and 2 dots make' gold 8 before False guess ' 9' after False guess ' 6'
n7 prompt '2 kites and 4 kites make' gold 6 before False guess ' 5' after False guess '\n'
n8 prompt '1 hats and 6 hats make' gold 7 before False guess ' the' after False guess ' 3'
