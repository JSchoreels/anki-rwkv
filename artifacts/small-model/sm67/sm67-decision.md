# Data augmentation (SM67) stage 1: note-split cool-down vs the leak-free small model's (SM72s) cool-down

Arm `sm67-aug-dec25-1ep32k-muon/w8+s8`, control `s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/w8+s8` (same start epoch, same reviews, order and options except the split). Leak-free scoring, w8+s8, review-weighted LogLoss, user bootstrap 2,000 seed 0, 95% CI; deltas in points of % of the big model (1 leak-free epoch).

| pool | users | head | arm | control | delta | points [95% CI] | users better | arm vs big | control vs big |
|---|---|---|---|---|---|---|---|---|---|
| 732 (rule) | 722 | curve | 0.30007 | 0.30003 | +0.000036 | +0.01 [-0.02, +0.06] | 377 | +0.37% | +0.36% |
| 732 (rule) | 722 | instant | 0.27652 | 0.27654 | -0.000015 | -0.01 [-0.01, -0.00] | 364 | +0.28% | +0.28% |
| 732+long | 725 | curve | 0.30208 | 0.30199 | +0.000097 | +0.03 [-0.02, +0.10] | 378 | +0.28% | +0.25% |
| 732+long | 725 | instant | 0.27741 | 0.27742 | -0.000011 | -0.00 [-0.01, +0.00] | 364 | +0.30% | +0.31% |

Verdict: **stop** (both heads gain under 0.10 points).
Ship check: not a release candidate.
