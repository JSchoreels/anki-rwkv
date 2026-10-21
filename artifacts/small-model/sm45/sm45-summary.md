# SM45: int4 card state + int8 note state, SM36 (best model), w8 weights

Review-weighted benchmark LogLoss; delta = variant - SM36 w8+s8 [95% paired user bootstrap]; % vs the big model (fork w8+s8) on the same users; curve = ahead.

- 732 pool, w8+s4c8n, 723 users: curve 0.30338 (+0.66% vs big; s8 0.30333 +0.64%) delta +0.00005 [+0.00003, +0.00006]; instant 0.27965 (+0.64% vs big; s8 0.27960 +0.63%) delta +0.00005 [+0.00004, +0.00006]
- 732 pool, w8+s4, 723 users: curve 0.30343 (+0.68% vs big; s8 0.30333 +0.64%) delta +0.00010 [+0.00007, +0.00013]; instant 0.27976 (+0.69% vs big; s8 0.27960 +0.63%) delta +0.00017 [+0.00014, +0.00020]
- 464 pool, w8+s4c8n, 458 users: curve 0.29809 (+0.76% vs big; s8 0.29804 +0.74%) delta +0.00005 [+0.00003, +0.00007]; instant 0.27402 (+0.66% vs big; s8 0.27397 +0.64%) delta +0.00005 [+0.00003, +0.00007]
- 464 pool, w8+s4, 458 users: curve 0.29814 (+0.78% vs big; s8 0.29804 +0.74%) delta +0.00010 [+0.00007, +0.00013]; instant 0.27413 (+0.70% vs big; s8 0.27397 +0.64%) delta +0.00016 [+0.00012, +0.00021]
