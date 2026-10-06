# WP12 SmolLM2-360M

Learner is HuggingFaceTB/SmolLM2-360M base. Not a math-tuned checkpoint. Not random weights. Not the 4.3M model. No earlier checkpoint was resumed.

Class 1 chapter text, then Class 2 chapter text, retokenized with the SmolLM2 tokenizer and packed to 512. No replay. Class 3 text was not trained on. Numeric prompts are the eight held-out WP6 prompts. The Class 3 spans are the same eight windows as WP9. Span text is not printed.

No critic. p_new is null. The matrix was not started. This run is the GPU paste. The CPU machine did not finish an update.

Class 1 steps: 200
Class 1 stop: step_cap
Class 1 loss before: 2.2669
Class 1 loss after: 1.8054
Class 2 steps: 200
Class 2 stop: step_cap
Class 2 loss before: 1.8008
Class 2 loss after: 1.4949

Numeric exact match before: 2/8
Numeric exact match after: 1/8
Class 3 span tokens before: 56/83
Class 3 span tokens after: 53/83

Continue-pretraining did not raise either score. The base model was already at 2/8 before any textbook update. The from-scratch failure was language.
