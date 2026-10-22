# SM59 kd-100 fine-tune from SM36: s-w64kd-sm36-kd-100/w8+s8 vs s-w64kd-gate-sc-lanes-dec12-1ep32k-muon/w8+s8

w8+s8, review-weighted, paired user bootstrap 2,000 (seed 0) 95% CI; delta = arm - SM36; 9 never-fit users excluded.

732 pool (723 users): arm curve 0.30365 (+0.75% vs fork) instant 0.27971 (+0.67%); SM36 curve 0.30333 (+0.64%) instant 0.27960 (+0.63%); arm - SM36 curve +0.00032 [+0.00013, +0.00050] (+0.106%, 227 users better) instant +0.00011 [+0.00009, +0.00014] (+0.041%, 175 users better)

464 pool (458 users): arm curve 0.29834 (+0.84% vs fork) instant 0.27408 (+0.68%); SM36 curve 0.29804 (+0.74%) instant 0.27397 (+0.64%); arm - SM36 curve +0.00030 [-0.00000, +0.00053] (+0.099%, 145 users better) instant +0.00010 [+0.00007, +0.00013] (+0.038%, 111 users better)

training: 81,297,101 reviews, 3108 updates, 20,139 reviews/s, 67.3 min training (67.4 min case)
