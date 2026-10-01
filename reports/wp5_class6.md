# WP5 Class 6 debug

Debug model trained on Class 6 packed blocks only.
Microbatches of 8. One optimizer step covers all 225 blocks.
Next-token loss. Corpus BPE. Block size 512. No letter scoring.
No book text is printed. p_new is null. No critic. No RL. No pretrained weights.
Classes 1-5 were not trained. The 12-layer model was not trained.

Parameters: 4339200
Steps: 80
Stop: step_cap
Full-batch loss before: 8.3863
Full-batch loss after: 3.9353
Exact match before: 0/8
Exact match after: 0/8

q1 prompt '7 times 8 is' gold 56 before False guess ' multiple' after False guess ' the'
q2 prompt '96 divided by 12 is' gold 8 before False guess ' short' after False guess ' the'
q3 prompt '29 added to 46 is' gold 75 before False guess 'aman' after False guess ' the'
q4 prompt '3 fourths of 36 is' gold 27 before False guess 'lebr' after False guess ' the'
q5 prompt '15 percent of 80 is' gold 12 before False guess ' is' after False guess ' the'
q6 prompt 'Negative 7 added to 12 is' gold 5 before False guess ' oper' after False guess ' the'
q7 prompt 'A rectangle 9 by 6 has area' gold 54 before False guess ' pur' after False guess ' of'
q8 prompt 'The number 11 squared is' gold 121 before False guess 'alkalk' after False guess ' the number'
