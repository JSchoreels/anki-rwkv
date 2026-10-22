# SM59 kd-control fine-tune from SM36: s-w64kd-sm36-kd-control/w8+s8 vs s-w64kd-gate-sc-lanes-dec12-1ep32k-muon/w8+s8

w8+s8, review-weighted, paired user bootstrap 2,000 (seed 0) 95% CI; delta = arm - SM36; 9 never-fit users excluded.

732 pool (723 users): arm curve 0.30329 (+0.63% vs fork) instant 0.27957 (+0.62%); SM36 curve 0.30333 (+0.64%) instant 0.27960 (+0.63%); arm - SM36 curve -0.00004 [-0.00016, +0.00006] (-0.012%, 350 users better) instant -0.00003 [-0.00005, -0.00001] (-0.011%, 428 users better)

464 pool (458 users): arm curve 0.29799 (+0.73% vs fork) instant 0.27394 (+0.63%); SM36 curve 0.29804 (+0.74%) instant 0.27397 (+0.64%); arm - SM36 curve -0.00005 [-0.00023, +0.00009] (-0.017%, 222 users better) instant -0.00003 [-0.00006, -0.00001] (-0.012%, 273 users better)

training: 81,297,101 reviews, 3108 updates, 20,196 reviews/s, 67.1 min training (67.2 min case)
