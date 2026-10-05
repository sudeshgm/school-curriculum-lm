# A school-book curriculum does not produce exam skill at this scale.

Exam gate. Class 1 letter exam 2/8, and every prediction was C. Commit 268180205e79594efb33e91fa1bfb5fe5deebf3b. Class 1 numeric exam 1/8. Commit 274ed7816f501a6b021226377a362d0e3cf6fa18. Class 6 numeric exam 0/8 on CPU and 0/8 on GPU. Commit eb8a19a6b031c8c587986f8d945983b41f87e00a.

WP6. Textbook-only and textbook plus worked sentences both scored 0/8. Commit 4e05c292dbc62a39c2c0a9fd8cba7ff42539ea6c. WP6b. The worked-sentence model memorized 19/40 of the trained sums. Commit 3cfa1c2c16f6102bb2ff45c284415b9e9383eb71.

WP7. Class 1 recall fell from 16/64 to 9/64 after Class 2 with no replay. Commit ae893d208d7a966553283e29e6d2def621a8ddc8.

WP8. With 30 percent replay, the drop was the same, 16/64 to 9/64. Commit 4c83e34a8e0de389b8d969eacdd1b5dd37ce7702.

WP9. Class 3 spans scored 0/64 untrained, 8/64 after Class 1 then Class 2, and 16/64 after Class 3. Commit 1694c0519015ddfcb83b967d2a7221a1de597fb6.

WP10. Staged training scored 8/64 and the shuffled run scored 5/64, at the same token count. Commit 2439f101bb8f3c91d4e2fdca7c8bdce699ffa90e.

WP11. Seed 21 scored 6/64 staged and 4/64 shuffled. The gap did not repeat. Commit 1a7b0a3a99d1b7e9bc2665818faa92a32d1dc4c5.

The testbed failed the exam gate, forward transfer of next-token recall is the result that moved, and order did not clear a second seed.
