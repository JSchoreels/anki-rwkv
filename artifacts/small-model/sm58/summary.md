# SM58 disagreement test: small vs big model (UM+ and slope)

UM+ lower is better; slope closer to 0 is better. In this construction the two slopes always differ by exactly 1 (slope small = slope big + 1), so a tie is +0.5 for the small model and -0.5 for the big one (the sign of the half depends on which model is x's plus side; here x = p_small - p_big). x = p_small - p_big.

## 732-user pool (benchmark users, w8+s8 deployed variants)

### Muon best small model (SM36) (sm36) vs big model (722 users used; skipped: ['1479', '4379', '4501', '4852', '4922', '650', '70'])

| Head | Sample | Reviews | UM+ small | slope small | UM+ big | slope big | UM+ diff (small - big) [95% CI] | LogLoss small | LogLoss big | LogLoss delta [95% CI] |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| curve | all | 30,126,145 | 0.0251 | +0.723 | 0.0106 | -0.277 | +0.0145 [+0.0102, +0.0189] | 0.3014 | 0.2994 | +0.65% [+0.44%, +0.89%] |
| curve | cross_day | 19,287,209 | 0.0242 | +0.673 | 0.0130 | -0.327 | +0.0112 [+0.0052, +0.0173] | 0.2951 | 0.2935 | +0.55% [+0.27%, +0.88%] |
| instant | all | 30,126,145 | 0.0219 | +0.796 | 0.0059 | -0.204 | +0.0159 [+0.0148, +0.0170] | 0.2776 | 0.2758 | +0.63% [+0.58%, +0.68%] |
| instant | cross_day | 19,287,209 | 0.0195 | +0.757 | 0.0065 | -0.243 | +0.0130 [+0.0118, +0.0140] | 0.2667 | 0.2653 | +0.54% [+0.49%, +0.60%] |

### 8k small model (SM55) (sm55) vs big model (722 users used; skipped: ['1479', '4379', '4501', '4852', '4922', '650', '70'])

| Head | Sample | Reviews | UM+ small | slope small | UM+ big | slope big | UM+ diff (small - big) [95% CI] | LogLoss small | LogLoss big | LogLoss delta [95% CI] |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| curve | all | 30,126,145 | 0.0229 | +0.683 | 0.0114 | -0.317 | +0.0115 [+0.0072, +0.0155] | 0.3009 | 0.2994 | +0.49% [+0.33%, +0.67%] |
| curve | cross_day | 19,287,209 | 0.0210 | +0.613 | 0.0141 | -0.387 | +0.0069 [+0.0018, +0.0121] | 0.2945 | 0.2935 | +0.34% [+0.13%, +0.56%] |
| instant | all | 30,126,145 | 0.0205 | +0.768 | 0.0065 | -0.232 | +0.0140 [+0.0129, +0.0150] | 0.2773 | 0.2758 | +0.54% [+0.49%, +0.58%] |
| instant | cross_day | 19,287,209 | 0.0180 | +0.724 | 0.0071 | -0.276 | +0.0110 [+0.0099, +0.0119] | 0.2665 | 0.2653 | +0.44% [+0.40%, +0.49%] |

## user's own collection (aggregate only; full history, float32 CPU)

### Muon best small model (SM36) (sm36) vs big model

| Head | Sample | Reviews | UM+ small | slope small | UM+ big | slope big | UM+ diff (small - big) [95% CI] | LogLoss small | LogLoss big | LogLoss delta [95% CI] |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| curve | all | 173,690 | 0.0179 | +0.534 | 0.0181 | -0.466 | -0.0001 | 0.4412 | 0.4417 | -0.10% |
| curve | cross_day | 141,775 | 0.0173 | +0.504 | 0.0184 | -0.496 | -0.0011 | 0.4602 | 0.4606 | -0.09% |
| instant | all | 173,690 | 0.0189 | +0.577 | 0.0157 | -0.423 | +0.0032 | 0.4369 | 0.4366 | +0.07% |
| instant | cross_day | 141,775 | 0.0167 | +0.533 | 0.0166 | -0.467 | +0.0001 | 0.4556 | 0.4555 | +0.01% |

### 8k small model (SM55) (sm55) vs big model

| Head | Sample | Reviews | UM+ small | slope small | UM+ big | slope big | UM+ diff (small - big) [95% CI] | LogLoss small | LogLoss big | LogLoss delta [95% CI] |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| curve | all | 173,690 | 0.0199 | +0.559 | 0.0181 | -0.441 | +0.0018 | 0.4414 | 0.4417 | -0.05% |
| curve | cross_day | 141,775 | 0.0197 | +0.541 | 0.0186 | -0.459 | +0.0011 | 0.4604 | 0.4606 | -0.05% |
| instant | all | 173,690 | 0.0201 | +0.608 | 0.0150 | -0.392 | +0.0051 | 0.4370 | 0.4366 | +0.10% |
| instant | cross_day | 141,775 | 0.0181 | +0.557 | 0.0160 | -0.443 | +0.0021 | 0.4557 | 0.4555 | +0.05% |

