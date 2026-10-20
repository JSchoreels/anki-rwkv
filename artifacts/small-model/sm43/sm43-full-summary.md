# SM43 full: data diversity vs epochs (equal reviews seen, with LR decay)

- Arm A (sm43full-a-1250u-11ep): 1,250 gate users x 10 constant-LR + 1 decay epoch(s), 96,446,289 reviews/epoch, 1,060,909,179 total; 11 epochs done, 2.87 h training, 102,599 reviews/s.
- Arm B (sm43full-b-125u-110ep): 125 of those users x 100 constant-LR + 10 decay epochs, 9,618,195 reviews/epoch, 1,058,001,450 total; 110 epochs done, 2.87 h training, 102,296 reviews/s.
- Recipe: scale-w64kd-gate-cached-ctx2048 options (2,048-review windows, cached KD), seed 2025, order seed 20261013, from scratch. Decay: linear 7e-4 -> 0 over the decay epochs (scripts/lr_tail_decay.py), KD alpha 0.5 while decaying.
- Scoring: 732 pool, w8+s8, 723 users with rows for A, B and the fork. Paired, review-weighted; 95% user bootstrap (2,000 resamples, seed 0).

| head | A LogLoss | B LogLoss | B vs A | 95% CI | A vs fork | B vs fork | B vs A light / mid / heavy |
|---|---|---|---|---|---|---|---|
| curve | 0.3086 | 0.3110 | +0.77% | [+0.58%, +1.00%] | +2.40% | +3.19% | +0.97% / +0.76% / +0.76% |
| instant | 0.2846 | 0.2862 | +0.55% | [+0.47%, +0.65%] | +2.43% | +3.00% | +0.56% / +0.62% / +0.54% |

Verdict (B as the candidate, A as the reference; margin +0.25%): **FAIL**. PASS upper95 <= +0.25% on both heads; EXTEND point <= +0.25% on both, an upper above; FAIL point > +0.25% on either.
Mini (10-19, 1,000 users x 1 epoch vs 100 x 10, no decay): B was +0.36% curve / +0.46% instant worse (FAIL).
