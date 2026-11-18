# Time head for the show-time two-head cool-down for Tiny (SM126): result

**What answer time means.** The answer time is the whole time on the card, front and back together: from the moment the question is shown to the moment a grade button is pressed, capped at the preset's maximum answer seconds. Every "within S s" below means answered within S s, reading the back included.

An answer-time head (the answer-time head on richer frozen features (SM82c) recipe) trained alone on each frozen family model, on show-time inputs with same-day reviews included, then scored on the 732 set (722 users), show-time inputs, leak-free. Pre-registered rules: `design.md`.

## Gate per head (grade unchanged bitwise on 5 users; the head beats both answer-time baselines, CI clear of zero)

| model | users | grade unchanged | time LogLoss (head / ablation / card's last / user median) | head vs user median | mean abs ln error (head / user median) | gate |
|---|---|---|---|---|---|---|
| the show-time two-head cool-down for Tiny (SM124) | 722 | True | 1.9749 / 2.0420 / 2.4751 / 2.3849 | -0.4101 [-0.4368, -0.3856] (-17.2%) | 0.4776 / 0.6533 | pass |

Same numbers on the later-day subset (the rows of the earlier answer-time heads, SM82 rule), for comparison with the shipped head of the show-time retrain (SM92) (gated at 1.9580 on normal inputs):

| model | rows | head | card's last | user median | shipped head |
|---|---|---|---|---|---|
| the show-time two-head cool-down for Tiny (SM124) | 21,585,938 | 1.9611 | 2.4691 | 2.3652 |  |

## P(recall within S seconds)

p = sum over Hard, Good, Easy of P(grade) x P(t <= S | grade); label: grade >= Hard and recorded time <= S. Baseline (a): the model's own P(grade) x the user's running share of earlier answers of that grade under S; baseline (b): the user's running rate of the event. LogLoss deltas are head minus baseline, review-weighted, 95% CI from 2,000 user bootstraps (seed 0).

### the show-time two-head cool-down for Tiny (SM124)

| S | rows | event rate | LogLoss head | (a) | (b) | head vs (a) | head vs (b) | Brier head | ECE head | ECE (a) | AUC head | AUC (a) | AUC (b) | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 s | 30,093,623 | 0.199 | 0.3047 | 0.4022 | 0.4268 | -0.0975 [-0.1069, -0.0889] (-24.2%) | -0.1221 [-0.1330, -0.1120] (-28.6%) | 0.0946 | 0.0159 | 0.0305 | 0.8966 | 0.8029 | 0.7664 | no |
| 5 s | 30,093,623 | 0.366 | 0.4304 | 0.5419 | 0.5789 | -0.1115 [-0.1193, -0.1040] (-20.6%) | -0.1486 [-0.1566, -0.1402] (-25.7%) | 0.1404 | 0.0122 | 0.0300 | 0.8694 | 0.7751 | 0.7273 | no |
| 8 s | 30,093,623 | 0.518 | 0.4727 | 0.5708 | 0.6182 | -0.0981 [-0.1052, -0.0909] (-17.2%) | -0.1455 [-0.1526, -0.1381] (-23.5%) | 0.1559 | 0.0049 | 0.0227 | 0.8529 | 0.7709 | 0.7115 | yes |
| 10 s | 30,093,623 | 0.580 | 0.4728 | 0.5585 | 0.6103 | -0.0857 [-0.0926, -0.0790] (-15.3%) | -0.1375 [-0.1444, -0.1304] (-22.5%) | 0.1556 | 0.0039 | 0.0188 | 0.8468 | 0.7730 | 0.7069 | yes |
| 15 s | 30,093,623 | 0.670 | 0.4546 | 0.5158 | 0.5749 | -0.0611 [-0.0672, -0.0553] (-11.8%) | -0.1203 [-0.1270, -0.1135] (-20.9%) | 0.1481 | 0.0036 | 0.0122 | 0.8371 | 0.7793 | 0.6993 | yes |
| 20 s | 30,093,623 | 0.717 | 0.4345 | 0.4802 | 0.5439 | -0.0458 [-0.0511, -0.0408] (-9.5%) | -0.1095 [-0.1161, -0.1030] (-20.1%) | 0.1402 | 0.0031 | 0.0088 | 0.8323 | 0.7858 | 0.6952 | yes |
| 30 s | 30,093,623 | 0.769 | 0.3976 | 0.4237 | 0.4949 | -0.0261 [-0.0307, -0.0217] (-6.2%) | -0.0972 [-0.1041, -0.0903] (-19.6%) | 0.1262 | 0.0023 | 0.0069 | 0.8318 | 0.8028 | 0.6954 | yes |

Reliable at every S (ECE < 0.01 and LogLoss below both baselines, CI clear of zero): **no**.

## Reliability tables (head, 20 equal-mass bins: rows, mean predicted, observed)

In `sm117-result.json` (`<model>.rus.<S>.predictors.model.reliability`); the largest bin gap per S is below.

| model | 3 s | 5 s | 8 s | 10 s | 15 s | 20 s | 30 s |
|---|---|---|---|---|---|---|---|
| the show-time two-head cool-down for Tiny (SM124) | 0.066 | 0.030 | 0.010 | 0.008 | 0.006 | 0.005 | 0.011 |
