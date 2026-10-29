# Smaller model (SM73) stage 1: smaller stored state for the leak-free small model (SM72s, cool-down from epoch 20)

732 set, leak-free scoring, review-weighted LogLoss; bar +0.50% curve / +0.40% instant vs the big model (1 leak-free epoch). Delta = paired vs the int8-state release (user bootstrap 2,000, seed 0, 95% CI).

| state | bytes per card + note | users | curve | vs big | delta vs int8 | instant | vs big | delta vs int8 | bar |
|---|---:|---:|---:|---:|---|---:|---:|---|---|
| w8+s4 | 3,018 | - | diverged on user 3876: int4 note state grows geometrically on a 9539-step note sequence (largest WKV output ~1e32 by review step 6000, inf at 6843; float 5, int8 6.5): no zero/inf scale, the rounding error compounds; ticket 000754 stopped after 37 users. int4 card + int8 note (w8+s4c8n) is finite on this user (curve 0.29217 vs int8 0.28974) | | | | | | FAIL |
| w8+s4c8n | 3,530 | 722 | 0.30023 | +0.42% | +0.000039 [+0.000026, +0.000055] | 0.27669 | +0.34% | +0.000051 [+0.000039, +0.000065] | PASS |
