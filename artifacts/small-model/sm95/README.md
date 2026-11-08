# The grade order by construction (SM95): result

**Verdict: cost** (informal read-out on 699 users; the pre-registered rule needs 700 and stays "incomplete").

- 24 users with 260k–530k reviews do not fit in GPU memory in rectified scoring, even with 1 worker at cap 0.92 (`gpu/retry-rect.log`). So 699 of 722 users have all four labels.
- The read-out ran the same rule code with the minimum lowered to 699, writing to `readout-699/`. One more user cannot flip it: the curve cost's CI is [+0.34, +0.62].

## Numbers (points = % of the big model (1 leak-free epoch) LogLoss, paired user bootstrap 95% CI)

| comparison | curve | instant |
|---|---|---|
| arm rectified vs reference raw (primary) | +0.46 [+0.34, +0.62] | +0.00 [-0.00, +0.01] |
| arm rectified vs reference rectified (training effect) | -0.47 [-0.67, -0.33] | +0.00 |
| reference rectified vs reference raw (pool today's model, no training) | +0.93 [+0.67, +1.27] | -0.00 |
| arm raw vs reference raw | +0.00 [-0.01, +0.02] | +0.00 |
| grade-order hinge (SM33) arm vs reference raw | +0.14 [+0.10, +0.19] | +0.00 |

The order measure covers 722 users x 3,000 sampled reviews (2,166,000 reviews).

| measure | arm rectified | reference raw | grade-order hinge (SM33) |
|---|---|---|---|
| displayed inverted pair at 0.90 | 0.00% (0 reviews) | 19.08% | 0.94% |
| Good changed by the fork's correction at 0.90 | 0.00% | 10.52% | 0.29% |

## Reading
- The rectifier gives an exact order, as proven (zero breaks everywhere). But pooling the four curves costs curve accuracy:
  - +0.93 points when today's model is pooled without training;
  - training on 8% of reviews halves that, to +0.46.
- The soft hinge (SM33) remains the cheaper trade: +0.14 points for 0.94% displayed breaks.
- The fork keeps its display-only correction (PAVA on intervals), which costs nothing in LogLoss.

Files: `readout-699/sm95-decision.{json,md}`, `readout-699/run.out`, `gpu/retry-rect.log`, `gpu/launch-sm95.log`, `gpu/sm95-neighbour.jsonl`.
