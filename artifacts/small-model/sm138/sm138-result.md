# Answer-time heads for the six-layer Tiny (SM138): result

**Verdict.** Both heads pass the pre-registered gate. The six-layer Tiny two-head model (SM130): grade unchanged bitwise on 5 users, time LogLoss 1.9931 against 2.4751 for the card's last answer time and 2.3849 for the user's median (-16.4% vs the user median, CI clear of zero), P(recall within 10 s) LogLoss 0.4772 with ECE 0.0027: PASS. The six-layer Tiny Instant-Only model (SM130): 1.9877, P(recall within 10 s) LogLoss 0.4757, ECE 0.0027: PASS. Neither head is reliable at every S (the 3 s and 5 s thresholds miss ECE < 0.01), the same as the heads of the time heads for the new candidates (SM119); this is informational, not a gate.

**What answer time means.** The answer time is the whole time on the card, front and back together: from the moment the question is shown to the moment a grade button is pressed, capped at the preset's maximum answer seconds. The "Show answer" moment is not stored, so front-only and back-only times cannot be computed. "Recall within S seconds" therefore means recalled (Hard, Good or Easy) and answered within S seconds, reading the back included.

**Run notes.** Each head trained on 1,991 of 1,991 users (one user skipped, as in SM117 and SM119), 14,314 steps over 117.3M rows: two-head 40.7 min, mean time loss 1.9389 / ablation 1.9987; Instant-Only 40.5 min, 1.9353 / 1.9987; the trunk unchanged for both. Both scored 723 of 723 users of the 732 set (722 with time rows). The whole launcher ran from 09:34 to 11:26 on Sat 11-21 (queue tickets 001412 to 001419, about 112 GPU minutes).

**Head width (packaging).** Each head takes **346 inputs**, not 474: the six-layer Tiny has 2 schedule rounds instead of 4, so the inputs are 3 trunk snapshots x 64 + the instant head's hidden layer (128) + 26 explicit time features (head 52,672 values instead of 69,056). The fork's head runtime must accept a 346-wide head for these two models when they are packaged for the family release.

**Against the four-round Tiny heads (information only).** The head of the longer cool-down for Tiny (SM115) in SM119 reached time LogLoss 1.9621 and P(recall within 10 s) LogLoss 0.4706. The six-layer heads are 0.026 (Instant-Only, +1.3%) and 0.031 (two-head, +1.6%) higher on time LogLoss and 0.005 to 0.007 (+1.1% to +1.4%) higher at 10 s: a shallower trunk gives the head fewer and weaker features. Both still beat the answer-time baselines by a wide margin.

An answer-time head (the answer-time head on richer frozen features (SM82c) recipe) trained alone on each frozen family model, on show-time inputs with same-day reviews included, then scored on the 732 set (722 users), show-time inputs, leak-free. Pre-registered rules: `design.md`. Answer time = front + back together, from the moment the question is shown to the moment a grade button is pressed, capped at the maximum answer seconds; recall within S s = recalled (Hard, Good or Easy) and answered within S s, reading the back included.

## Gate per head (grade unchanged bitwise on 5 users; the head beats both answer-time baselines, CI clear of zero)

| model | users | grade unchanged | time LogLoss (head / ablation / card's last / user median) | head vs user median | mean abs ln error (head / user median) | gate |
|---|---|---|---|---|---|---|
| the six-layer Tiny two-head model (SM130) | 722 | True | 1.9931 / 2.0413 / 2.4751 / 2.3849 | -0.3919 [-0.4186, -0.3681] (-16.4%) | 0.4844 / 0.6533 | pass |
| the six-layer Tiny Instant-Only model (SM130) | 722 | True | 1.9877 / 2.0413 / 2.4751 / 2.3849 | -0.3972 [-0.4241, -0.3733] (-16.6%) | 0.4817 / 0.6533 | pass |

Same numbers on the later-day subset (the rows of the earlier answer-time heads, SM82 rule), for comparison with the shipped head of the show-time retrain (SM92) (gated at 1.9580 on normal inputs):

| model | rows | head | card's last | user median | shipped head |
|---|---|---|---|---|---|
| the six-layer Tiny two-head model (SM130) | 21,585,938 | 1.9801 | 2.4691 | 2.3652 |  |
| the six-layer Tiny Instant-Only model (SM130) | 21,585,938 | 1.9746 | 2.4691 | 2.3652 |  |

## P(recall within S seconds)

p = sum over Hard, Good, Easy of P(grade) x P(t <= S | grade); label: grade >= Hard and recorded time <= S. Baseline (a): the model's own P(grade) x the user's running share of earlier answers of that grade under S; baseline (b): the user's running rate of the event. LogLoss deltas are head minus baseline, review-weighted, 95% CI from 2,000 user bootstraps (seed 0).

### the six-layer Tiny two-head model (SM130)

| S | rows | event rate | LogLoss head | (a) | (b) | head vs (a) | head vs (b) | Brier head | ECE head | ECE (a) | AUC head | AUC (a) | AUC (b) | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 s | 30,093,623 | 0.199 | 0.3095 | 0.4024 | 0.4268 | -0.0929 [-0.1021, -0.0844] (-23.1%) | -0.1173 [-0.1280, -0.1074] (-27.5%) | 0.0960 | 0.0173 | 0.0306 | 0.8928 | 0.8028 | 0.7664 | no |
| 5 s | 30,093,623 | 0.366 | 0.4365 | 0.5422 | 0.5789 | -0.1057 [-0.1131, -0.0985] (-19.5%) | -0.1425 [-0.1503, -0.1345] (-24.6%) | 0.1425 | 0.0129 | 0.0302 | 0.8653 | 0.7748 | 0.7273 | no |
| 8 s | 30,093,623 | 0.518 | 0.4779 | 0.5712 | 0.6182 | -0.0933 [-0.1001, -0.0866] (-16.3%) | -0.1403 [-0.1471, -0.1332] (-22.7%) | 0.1578 | 0.0039 | 0.0229 | 0.8496 | 0.7705 | 0.7115 | yes |
| 10 s | 30,093,623 | 0.580 | 0.4772 | 0.5590 | 0.6103 | -0.0818 [-0.0883, -0.0754] (-14.6%) | -0.1331 [-0.1398, -0.1262] (-21.8%) | 0.1572 | 0.0027 | 0.0191 | 0.8438 | 0.7725 | 0.7069 | yes |
| 15 s | 30,093,623 | 0.670 | 0.4579 | 0.5164 | 0.5749 | -0.0585 [-0.0644, -0.0529] (-11.3%) | -0.1170 [-0.1235, -0.1104] (-20.4%) | 0.1493 | 0.0036 | 0.0125 | 0.8346 | 0.7788 | 0.6993 | yes |
| 20 s | 30,093,623 | 0.717 | 0.4371 | 0.4809 | 0.5439 | -0.0438 [-0.0489, -0.0390] (-9.1%) | -0.1069 [-0.1132, -0.1004] (-19.6%) | 0.1411 | 0.0031 | 0.0091 | 0.8301 | 0.7852 | 0.6952 | yes |
| 30 s | 30,093,623 | 0.769 | 0.3994 | 0.4244 | 0.4949 | -0.0251 [-0.0294, -0.0211] (-5.9%) | -0.0955 [-0.1021, -0.0888] (-19.3%) | 0.1269 | 0.0027 | 0.0072 | 0.8301 | 0.8020 | 0.6954 | yes |

Reliable at every S (ECE < 0.01 and LogLoss below both baselines, CI clear of zero): **no**.

### the six-layer Tiny Instant-Only model (SM130)

| S | rows | event rate | LogLoss head | (a) | (b) | head vs (a) | head vs (b) | Brier head | ECE head | ECE (a) | AUC head | AUC (a) | AUC (b) | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 s | 30,093,623 | 0.199 | 0.3077 | 0.4023 | 0.4268 | -0.0946 [-0.1039, -0.0861] (-23.5%) | -0.1191 [-0.1299, -0.1092] (-27.9%) | 0.0954 | 0.0161 | 0.0303 | 0.8939 | 0.8027 | 0.7664 | no |
| 5 s | 30,093,623 | 0.366 | 0.4340 | 0.5421 | 0.5789 | -0.1081 [-0.1157, -0.1007] (-19.9%) | -0.1450 [-0.1529, -0.1368] (-25.0%) | 0.1417 | 0.0115 | 0.0297 | 0.8668 | 0.7748 | 0.7273 | no |
| 8 s | 30,093,623 | 0.518 | 0.4760 | 0.5710 | 0.6182 | -0.0950 [-0.1020, -0.0880] (-16.6%) | -0.1422 [-0.1491, -0.1349] (-23.0%) | 0.1571 | 0.0032 | 0.0224 | 0.8507 | 0.7706 | 0.7115 | yes |
| 10 s | 30,093,623 | 0.580 | 0.4757 | 0.5587 | 0.6103 | -0.0831 [-0.0898, -0.0766] (-14.9%) | -0.1347 [-0.1414, -0.1276] (-22.1%) | 0.1567 | 0.0027 | 0.0183 | 0.8448 | 0.7727 | 0.7069 | yes |
| 15 s | 30,093,623 | 0.670 | 0.4568 | 0.5161 | 0.5749 | -0.0592 [-0.0652, -0.0536] (-11.5%) | -0.1180 [-0.1247, -0.1114] (-20.5%) | 0.1489 | 0.0058 | 0.0121 | 0.8354 | 0.7791 | 0.6993 | yes |
| 20 s | 30,093,623 | 0.717 | 0.4362 | 0.4805 | 0.5439 | -0.0443 [-0.0495, -0.0394] (-9.2%) | -0.1077 [-0.1142, -0.1012] (-19.8%) | 0.1408 | 0.0042 | 0.0091 | 0.8309 | 0.7855 | 0.6952 | yes |
| 30 s | 30,093,623 | 0.769 | 0.3986 | 0.4240 | 0.4949 | -0.0255 [-0.0299, -0.0214] (-6.0%) | -0.0963 [-0.1029, -0.0896] (-19.5%) | 0.1266 | 0.0014 | 0.0075 | 0.8309 | 0.8025 | 0.6954 | yes |

Reliable at every S (ECE < 0.01 and LogLoss below both baselines, CI clear of zero): **no**.

## Reliability tables (head, 20 equal-mass bins: rows, mean predicted, observed)

In `sm117-result.json` (`<model>.rus.<S>.predictors.model.reliability`); the largest bin gap per S is below.

| model | 3 s | 5 s | 8 s | 10 s | 15 s | 20 s | 30 s |
|---|---|---|---|---|---|---|---|
| the six-layer Tiny two-head model (SM130) | 0.070 | 0.037 | 0.007 | 0.006 | 0.007 | 0.005 | 0.008 |
| the six-layer Tiny Instant-Only model (SM130) | 0.058 | 0.030 | 0.008 | 0.010 | 0.009 | 0.007 | 0.005 |
