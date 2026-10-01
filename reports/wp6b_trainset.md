# WP6 condition B train-set score

No training. The condition B checkpoint, step 80, was scored on the 40 worked sentences it trained on. The prompt is the sentence up to "make". The gold is the trained number. The 8 held-out prompts were not scored again.

Exact match: 19/40.
Class 1 sentences: 10/20.
Class 2 sentences: 9/20.
Held-out exam from WP6: 0/8.

The model did not memorize every worked sentence. It did memorize 19 of 40. The held-out exam was 0/8, so the failure on new questions is generalization.

No book text. No critic. No pretrained weights. p_new is null.

c1_01 gold 3 guess " 3" match yes
c1_02 gold 4 guess " 4" match yes
c1_03 gold 4 guess " 6" match no
c1_04 gold 5 guess " 5" match yes
c1_05 gold 5 guess " 6" match no
c1_06 gold 6 guess " 7" match no
c1_07 gold 6 guess " 7" match no
c1_08 gold 6 guess " 7" match no
c1_09 gold 7 guess " 9" match no
c1_10 gold 7 guess " 7" match yes
c1_11 gold 7 guess " 5" match no
c1_12 gold 8 guess " 8" match yes
c1_13 gold 8 guess " 8" match yes
c1_14 gold 8 guess " 14" match no
c1_15 gold 8 guess " 9" match no
c1_16 gold 9 guess " 9" match yes
c1_17 gold 9 guess " 9" match yes
c1_18 gold 9 guess " 9" match yes
c1_19 gold 5 guess " 5" match yes
c1_20 gold 9 guess " 14" match no
c2_01 gold 12 guess " 12" match yes
c2_02 gold 14 guess " 14" match yes
c2_03 gold 13 guess " 14" match no
c2_04 gold 14 guess " 14" match yes
c2_05 gold 14 guess " 11" match no
c2_06 gold 11 guess " 11" match yes
c2_07 gold 12 guess " 12" match yes
c2_08 gold 12 guess " 10" match no
c2_09 gold 11 guess " 3" match no
c2_10 gold 10 guess " 10" match yes
c2_11 gold 10 guess " 12" match no
c2_12 gold 10 guess " 15" match no
c2_13 gold 15 guess " 15" match yes
c2_14 gold 15 guess " 15" match yes
c2_15 gold 17 guess " 15" match no
c2_16 gold 15 guess " 15" match yes
c2_17 gold 16 guess " 11" match no
c2_18 gold 11 guess " 12" match no
c2_19 gold 12 guess " 13" match no
c2_20 gold 13 guess " 3" match no
