# Answer-time expressiveness (SM142): result

Answer time = front + back together, from the moment the question is shown to the moment a grade button is pressed, capped at the maximum answer seconds; "answered within S s" means answered within S s, reading the back included. Rows: the 732 set (722 users with time rows), show-time inputs, leak-free, fp32; reviews on a later day than the card's previous review whose card has an earlier timed review. Pre-registered methods: `design.md`. Every CI: 2,000 paired user bootstraps, seed 0, 95%. Description, not a gate.

## Read-out in plain language (the user's question: how expressive are the predictions, beyond the average?)

The question: if the answer-time head only produced 3 or 4 kinds of curve saying "half of the answers within 2 s, 4 s", it would not be exploitable even if precise on average. The result is that it is not that. All five heads separate the fast cards from the slow cards of one user clearly better than the card's own history does, and each user gets hundreds of distinguishable curves. But the heads spread the cards about 1.7 times too widely for what they actually know, and the shape of the curve (as opposed to its location) carries only a weak signal. By the pre-registered rule no model is template-like (E1 holds for every model) and none is card-expressive (E2 fails: the between-card slope is 0.59, below the required 0.8), so every model gets the label "partly expressive" (E1, E3 and E4 hold, E2 fails); none can be shape-expressive, because that label needs card-expressive first. The five models are close to each other: the show-time instant-only big model (SM99) and the longer Small Instant-Only model (SM113) are at the top, the six-layer Tiny models (SM130) are about 0.01 AUC and 2% to 3% of explained card variance lower.

Answer time means front plus back ("answered within S s, reading the back included"). Everything is within one user and one true grade, so the head gets no credit for knowing the grade. The show-time instant-only big model (SM99) has 719 of the 722 users (3 users could not be dumped in two passes at GPU cap 0.92); the card baselines on its users agree with the 722-user values to the third decimal, and no separate common-user table was made.

Within-user, within-grade AUC of "answered within S s", mean over users. The card's own running median (a) scores 0.690 / 0.719 / 0.727 / 0.729 at 2 / 3 / 4 / 6 s, the card's last time 0.674 / 0.696 / 0.700 / 0.698, and the user's running median per grade about 0.55 (the "template" reading). Gain over the card's own median with the 95% CI (the gain is above 0 at 2, 3, 4 and 6 s for every model; the CIs at 3 s and 6 s are in section 1):

| model | AUC 2 s | AUC 3 s | AUC 4 s | AUC 6 s | gain vs the card's median at 2 s | at 4 s |
|---|---|---|---|---|---|---|
| the six-layer Tiny two-head model (SM130) | 0.828 | 0.824 | 0.817 | 0.807 | +0.138 [0.129, 0.148] | +0.090 [0.084, 0.095] |
| the six-layer Tiny Instant-Only model (SM130) | 0.828 | 0.826 | 0.818 | 0.808 | +0.139 [0.129, 0.148] | +0.092 [0.086, 0.097] |
| the longer Small Instant-Only model (SM113) | 0.839 | 0.834 | 0.825 | 0.815 | +0.149 [0.140, 0.158] | +0.099 [0.093, 0.104] |
| the show-time two-head cool-down for Small (SM118) | 0.836 | 0.831 | 0.823 | 0.813 | +0.146 [0.137, 0.155] | +0.096 [0.091, 0.101] |
| the show-time instant-only big model (SM99) | 0.836 | 0.834 | 0.826 | 0.816 | +0.146 [0.137, 0.156] | +0.099 [0.094, 0.105] |

The head also beats the explicit-features-only head (+0.018 to +0.033 AUC).

Range and card-level knowledge (out of sample: predictions on a card's first half, actual times on its second half):

| model | review-level slope | between-card slope | spread ratio | persistent card differences explained (head) | gain vs the card's median (44.1%) | gain vs the explicit-features-only head (47.3%) |
|---|---|---|---|---|---|---|
| the six-layer Tiny two-head model (SM130) | 0.930 [0.921, 0.939] | 0.589 [0.562, 0.616] | 1.109 [1.086, 1.133] | 50.6% [48.8, 52.4] | +6.5% [5.4, 7.7] | +3.3% [2.8, 3.8] |
| the six-layer Tiny Instant-Only model (SM130) | 0.930 [0.922, 0.939] | 0.590 [0.563, 0.617] | 1.114 [1.091, 1.138] | 51.2% [49.4, 53.0] | +7.1% [6.0, 8.3] | +3.9% [3.4, 4.4] |
| the longer Small Instant-Only model (SM113) | 0.932 [0.924, 0.940] | 0.592 [0.565, 0.618] | 1.135 [1.112, 1.159] | 53.0% [51.2, 54.8] | +8.9% [7.7, 10.1] | +5.7% [5.1, 6.2] |
| the show-time two-head cool-down for Small (SM118) | 0.929 [0.921, 0.938] | 0.590 [0.563, 0.616] | 1.131 [1.108, 1.155] | 52.6% [50.8, 54.4] | +8.6% [7.4, 9.7] | +5.3% [4.8, 5.9] |
| the show-time instant-only big model (SM99) | 0.929 [0.920, 0.938] | 0.589 [0.563, 0.614] | 1.147 [1.125, 1.172] | 53.5% [51.7, 55.3] | +9.4% [8.2, 10.6] | +6.4% [5.8, 7.0] |

Reading: within a (user, grade) the review-level slope is 0.93, nearly calibrated. Between cards the head's spread is 1.1 to 1.15 times the SD of the persistent real card differences, but a head that explains about 53% of them should have a spread of only about 0.73 times, so the head spreads the cards too widely and the between-card slope is 0.59 (the explicit-features-only head has the same 0.61, so this comes from the training target, not from the frozen model). In plain words: the ordering of the cards is useful, the size of the card-to-card gap is overstated by about 1.7 times, and a client that uses the head's card differences should shrink them toward the user's average (a slope of about 0.6). The heads explain 51% to 54% of the persistent card differences against 44% for the card's own median, 38% for the card's last time and 7% for the user's running median per grade. Of a user's within-grade variance, 34% (median user) is persistent card-to-card difference, the rest is review-to-review noise nobody can predict (within-card variance explained: 1.8% for the card's median, 20% to 22% for the heads, mildly in-sample).

Distinct curves per user (10% grid, true-grade distribution, median user): location included 681 to 804 curves (p10 to p90 about 320 to 1,750), location removed (shape only) 74 to 87 shapes (p10 to p90 about 30 to 275). On the coarser 20% grid: 160 to 180 and 20 to 23. The per-user SD of the predicted log median is 0.57 to 0.58 ln-units (a factor of about 1.8) and of the log interquartile range 0.19. So the head does not produce 3 or 4 templates. The pooled PCA of the location-normalised shapes needs 7 to 8 components for 95% of the variance (participation ratio 2.8 to 2.9, so two or three components dominate), and 3 components when the location is included (participation ratio 1.65: the location is the main axis).

Shape calibration: the predicted width is informative but only weakly. Card level, the within-user Spearman correlation of the predicted width with the realised spread of the card's later times is 0.086 [0.077, 0.095] for the six-layer Tiny two-head model (SM130), 0.085 [0.076, 0.094] for the six-layer Tiny Instant-Only model (SM130), 0.093 [0.084, 0.102] for the longer Small Instant-Only model (SM113), 0.090 [0.081, 0.099] for the show-time two-head cool-down for Small (SM118) and 0.100 [0.091, 0.110] for the show-time instant-only big model (SM99). All are above 0 but small, and none is better than the card's own earlier spread (0.093 to 0.094; difference -0.008 to +0.007, every CI contains 0 except the six-layer Tiny Instant-Only model (SM130) at -0.008 [-0.016, -0.000]). At review level the correlation of the width with the absolute error is 0.20 to 0.22, and the interquartile range covers 52% to 54% of the answers in every width quintile (ideal 50%): the widths are right on average, but a wider prediction is only slightly more often a more spread review. So the useful part of the head is the location of each card; its individual spread carries modest information.

Life of a card (section 5): all grades, the head's median is 0.03 to 0.04 ln-units (3% to 4%) too low on average at every card age, and for Good answers 0.04 to 0.06 too low (up to 0.088 for the show-time instant-only big model (SM99) on cards with 32 or more earlier reviews). The predicted interquartile range is about 8% wider than the realised one (0.75 to 0.93 against 0.69 to 0.84 ln-units) and shrinks after the first review, as the real one does.

Worth noting: the five models differ little in expressiveness, the gain over the explicit-features-only head is small (+0.02 to +0.03 AUC, +3% to +6% explained card variance), and the over-wide card spread (between-card slope 0.59) is also there in the explicit-features-only head.

## Summary

| model | label | E1 AUC vs card median | E2 slope / between-card slope | E3 between-card explained head vs (a) | E4 shapes (PCA k95, card shape calibration) |
|---|---|---|---|---|---|
| the six-layer Tiny two-head model (SM130) | partly expressive | True | False (0.93 / 0.59) | True (50.6% vs 44.1%) | True (8, 0.086 [0.077, 0.095]) |
| the six-layer Tiny Instant-Only model (SM130) | partly expressive | True | False (0.93 / 0.59) | True (51.2% vs 44.1%) | True (8, 0.085 [0.076, 0.094]) |
| the longer Small Instant-Only model (SM113) | partly expressive | True | False (0.93 / 0.59) | True (53.0% vs 44.1%) | True (8, 0.093 [0.084, 0.102]) |
| the show-time two-head cool-down for Small (SM118) | partly expressive | True | False (0.93 / 0.59) | True (52.6% vs 44.1%) | True (8, 0.090 [0.081, 0.099]) |
| the show-time instant-only big model (SM99) | partly expressive | True | False (0.93 / 0.59) | True (53.5% vs 44.1%) | True (7, 0.100 [0.091, 0.110]) |

## 1. Within-user, within-grade AUC (P(answered within S s | true grade))

Each user equal; strata = (user, true grade). (a) the card's running median, (b) the card's last time, (c) the user's running median of the grade; ablation = the explicit-features-only head.

### the six-layer Tiny two-head model (SM130)

| S | users | head | ablation | (a) card median | (b) card last | (c) user grade median | head - (a) | head - ablation | head - (b) |
|---|---|---|---|---|---|---|---|---|---|
| 2 s | 660 | 0.8279 | 0.8104 | 0.6896 | 0.6739 | 0.5578 | 0.1383 [0.1292, 0.1478] | 0.0175 [0.0156, 0.0193] | 0.1540 [0.1455, 0.1624] |
| 3 s | 711 | 0.8243 | 0.8022 | 0.7185 | 0.6961 | 0.5559 | 0.1058 [0.0995, 0.1122] | 0.0222 [0.0207, 0.0236] | 0.1283 [0.1223, 0.1340] |
| 4 s | 721 | 0.8165 | 0.7934 | 0.7266 | 0.6998 | 0.5542 | 0.0899 [0.0844, 0.0951] | 0.0231 [0.0217, 0.0245] | 0.1167 [0.1115, 0.1215] |
| 6 s | 722 | 0.8069 | 0.7830 | 0.7288 | 0.6984 | 0.5491 | 0.0781 [0.0734, 0.0825] | 0.0239 [0.0228, 0.0249] | 0.1084 [0.1043, 0.1126] |
| 8 s | 722 | 0.7994 | 0.7753 | 0.7256 | 0.6939 | 0.5455 | 0.0738 [0.0695, 0.0778] | 0.0240 [0.0230, 0.0250] | 0.1055 [0.1017, 0.1092] |
| 10 s | 721 | 0.7941 | 0.7705 | 0.7222 | 0.6900 | 0.5434 | 0.0719 [0.0680, 0.0756] | 0.0236 [0.0227, 0.0245] | 0.1041 [0.1004, 0.1074] |

Secondary, grades pooled within user: head on the true grade / the head's own grade mix vs (a), (b), the user's running median:

| S | head (true grade) | head - (a) | head (grade mix) | grade mix - (a) | (c) all-grade user median |
|---|---|---|---|---|---|
| 2 s | 0.8383 | 0.1502 [0.1412, 0.1595] | 0.8287 | 0.1406 [0.1311, 0.1502] | 0.5511 |
| 3 s | 0.8396 | 0.1189 [0.1123, 0.1253] | 0.8303 | 0.1096 [0.1029, 0.1161] | 0.5511 |
| 4 s | 0.8344 | 0.1043 [0.0991, 0.1096] | 0.8236 | 0.0936 [0.0882, 0.0990] | 0.5514 |
| 6 s | 0.8262 | 0.0934 [0.0886, 0.0979] | 0.8140 | 0.0812 [0.0765, 0.0859] | 0.5479 |
| 8 s | 0.8194 | 0.0894 [0.0850, 0.0936] | 0.8066 | 0.0767 [0.0724, 0.0808] | 0.5453 |
| 10 s | 0.8140 | 0.0871 [0.0831, 0.0909] | 0.8014 | 0.0746 [0.0705, 0.0782] | 0.5436 |

### the six-layer Tiny Instant-Only model (SM130)

| S | users | head | ablation | (a) card median | (b) card last | (c) user grade median | head - (a) | head - ablation | head - (b) |
|---|---|---|---|---|---|---|---|---|---|
| 2 s | 660 | 0.8281 | 0.8104 | 0.6896 | 0.6739 | 0.5578 | 0.1386 [0.1293, 0.1481] | 0.0177 [0.0158, 0.0196] | 0.1542 [0.1458, 0.1626] |
| 3 s | 711 | 0.8263 | 0.8022 | 0.7185 | 0.6961 | 0.5559 | 0.1078 [0.1013, 0.1140] | 0.0241 [0.0225, 0.0257] | 0.1302 [0.1243, 0.1360] |
| 4 s | 721 | 0.8183 | 0.7934 | 0.7266 | 0.6998 | 0.5542 | 0.0917 [0.0863, 0.0968] | 0.0249 [0.0234, 0.0263] | 0.1185 [0.1134, 0.1233] |
| 6 s | 722 | 0.8082 | 0.7830 | 0.7288 | 0.6984 | 0.5491 | 0.0794 [0.0748, 0.0839] | 0.0252 [0.0241, 0.0263] | 0.1098 [0.1057, 0.1139] |
| 8 s | 722 | 0.8004 | 0.7753 | 0.7256 | 0.6939 | 0.5455 | 0.0748 [0.0705, 0.0789] | 0.0250 [0.0240, 0.0260] | 0.1064 [0.1028, 0.1102] |
| 10 s | 721 | 0.7949 | 0.7705 | 0.7222 | 0.6900 | 0.5434 | 0.0727 [0.0688, 0.0763] | 0.0244 [0.0234, 0.0253] | 0.1049 [0.1012, 0.1083] |

Secondary, grades pooled within user: head on the true grade / the head's own grade mix vs (a), (b), the user's running median:

| S | head (true grade) | head - (a) | head (grade mix) | grade mix - (a) | (c) all-grade user median |
|---|---|---|---|---|---|
| 2 s | 0.8382 | 0.1501 [0.1409, 0.1595] | 0.8294 | 0.1413 [0.1319, 0.1508] | 0.5511 |
| 3 s | 0.8410 | 0.1203 [0.1136, 0.1269] | 0.8324 | 0.1117 [0.1050, 0.1183] | 0.5511 |
| 4 s | 0.8360 | 0.1060 [0.1007, 0.1112] | 0.8253 | 0.0953 [0.0899, 0.1005] | 0.5514 |
| 6 s | 0.8274 | 0.0946 [0.0899, 0.0992] | 0.8152 | 0.0824 [0.0777, 0.0871] | 0.5479 |
| 8 s | 0.8203 | 0.0903 [0.0860, 0.0944] | 0.8074 | 0.0774 [0.0731, 0.0817] | 0.5453 |
| 10 s | 0.8147 | 0.0878 [0.0837, 0.0916] | 0.8021 | 0.0752 [0.0711, 0.0789] | 0.5436 |

### the longer Small Instant-Only model (SM113)

| S | users | head | ablation | (a) card median | (b) card last | (c) user grade median | head - (a) | head - ablation | head - (b) |
|---|---|---|---|---|---|---|---|---|---|
| 2 s | 660 | 0.8385 | 0.8106 | 0.6896 | 0.6739 | 0.5578 | 0.1489 [0.1401, 0.1582] | 0.0278 [0.0259, 0.0299] | 0.1646 [0.1564, 0.1727] |
| 3 s | 711 | 0.8336 | 0.8018 | 0.7185 | 0.6961 | 0.5559 | 0.1151 [0.1088, 0.1214] | 0.0318 [0.0302, 0.0335] | 0.1376 [0.1317, 0.1433] |
| 4 s | 721 | 0.8254 | 0.7930 | 0.7266 | 0.6998 | 0.5542 | 0.0988 [0.0932, 0.1038] | 0.0323 [0.0308, 0.0338] | 0.1256 [0.1206, 0.1303] |
| 6 s | 722 | 0.8149 | 0.7827 | 0.7288 | 0.6984 | 0.5491 | 0.0861 [0.0815, 0.0907] | 0.0322 [0.0311, 0.0333] | 0.1165 [0.1125, 0.1207] |
| 8 s | 722 | 0.8068 | 0.7751 | 0.7256 | 0.6939 | 0.5455 | 0.0812 [0.0769, 0.0853] | 0.0316 [0.0305, 0.0327] | 0.1128 [0.1091, 0.1167] |
| 10 s | 721 | 0.8012 | 0.7703 | 0.7222 | 0.6900 | 0.5434 | 0.0790 [0.0750, 0.0828] | 0.0309 [0.0298, 0.0319] | 0.1112 [0.1074, 0.1146] |

Secondary, grades pooled within user: head on the true grade / the head's own grade mix vs (a), (b), the user's running median:

| S | head (true grade) | head - (a) | head (grade mix) | grade mix - (a) | (c) all-grade user median |
|---|---|---|---|---|---|
| 2 s | 0.8495 | 0.1614 [0.1524, 0.1702] | 0.8390 | 0.1509 [0.1419, 0.1602] | 0.5511 |
| 3 s | 0.8485 | 0.1278 [0.1213, 0.1341] | 0.8390 | 0.1183 [0.1118, 0.1246] | 0.5511 |
| 4 s | 0.8426 | 0.1126 [0.1074, 0.1179] | 0.8316 | 0.1016 [0.0961, 0.1068] | 0.5514 |
| 6 s | 0.8336 | 0.1008 [0.0960, 0.1054] | 0.8214 | 0.0886 [0.0838, 0.0934] | 0.5479 |
| 8 s | 0.8263 | 0.0963 [0.0919, 0.1006] | 0.8135 | 0.0836 [0.0792, 0.0879] | 0.5453 |
| 10 s | 0.8206 | 0.0937 [0.0896, 0.0976] | 0.8081 | 0.0812 [0.0771, 0.0850] | 0.5436 |

### the show-time two-head cool-down for Small (SM118)

| S | users | head | ablation | (a) card median | (b) card last | (c) user grade median | head - (a) | head - ablation | head - (b) |
|---|---|---|---|---|---|---|---|---|---|
| 2 s | 660 | 0.8356 | 0.8106 | 0.6896 | 0.6739 | 0.5578 | 0.1460 [0.1369, 0.1554] | 0.0249 [0.0229, 0.0269] | 0.1616 [0.1535, 0.1698] |
| 3 s | 711 | 0.8309 | 0.8018 | 0.7185 | 0.6961 | 0.5559 | 0.1124 [0.1060, 0.1187] | 0.0291 [0.0274, 0.0309] | 0.1349 [0.1291, 0.1405] |
| 4 s | 721 | 0.8228 | 0.7930 | 0.7266 | 0.6998 | 0.5542 | 0.0963 [0.0906, 0.1014] | 0.0298 [0.0283, 0.0313] | 0.1231 [0.1180, 0.1279] |
| 6 s | 722 | 0.8127 | 0.7827 | 0.7288 | 0.6984 | 0.5491 | 0.0839 [0.0793, 0.0884] | 0.0300 [0.0288, 0.0311] | 0.1142 [0.1102, 0.1185] |
| 8 s | 722 | 0.8048 | 0.7751 | 0.7256 | 0.6939 | 0.5455 | 0.0792 [0.0749, 0.0833] | 0.0296 [0.0285, 0.0307] | 0.1108 [0.1071, 0.1147] |
| 10 s | 721 | 0.7994 | 0.7703 | 0.7222 | 0.6900 | 0.5434 | 0.0771 [0.0731, 0.0809] | 0.0290 [0.0280, 0.0300] | 0.1093 [0.1056, 0.1127] |

Secondary, grades pooled within user: head on the true grade / the head's own grade mix vs (a), (b), the user's running median:

| S | head (true grade) | head - (a) | head (grade mix) | grade mix - (a) | (c) all-grade user median |
|---|---|---|---|---|---|
| 2 s | 0.8454 | 0.1573 [0.1484, 0.1664] | 0.8349 | 0.1468 [0.1377, 0.1563] | 0.5511 |
| 3 s | 0.8457 | 0.1250 [0.1183, 0.1315] | 0.8360 | 0.1153 [0.1085, 0.1217] | 0.5511 |
| 4 s | 0.8401 | 0.1101 [0.1049, 0.1153] | 0.8290 | 0.0990 [0.0935, 0.1043] | 0.5514 |
| 6 s | 0.8315 | 0.0987 [0.0940, 0.1033] | 0.8193 | 0.0865 [0.0817, 0.0913] | 0.5479 |
| 8 s | 0.8242 | 0.0942 [0.0899, 0.0985] | 0.8115 | 0.0816 [0.0772, 0.0859] | 0.5453 |
| 10 s | 0.8186 | 0.0917 [0.0876, 0.0956] | 0.8062 | 0.0793 [0.0752, 0.0831] | 0.5436 |

### the show-time instant-only big model (SM99)

| S | users | head | ablation | (a) card median | (b) card last | (c) user grade median | head - (a) | head - ablation | head - (b) |
|---|---|---|---|---|---|---|---|---|---|
| 2 s | 657 | 0.8360 | 0.8103 | 0.6897 | 0.6741 | 0.5579 | 0.1463 [0.1372, 0.1556] | 0.0257 [0.0234, 0.0279] | 0.1619 [0.1537, 0.1697] |
| 3 s | 708 | 0.8337 | 0.8020 | 0.7185 | 0.6961 | 0.5559 | 0.1152 [0.1091, 0.1214] | 0.0317 [0.0301, 0.0334] | 0.1376 [0.1319, 0.1431] |
| 4 s | 718 | 0.8259 | 0.7932 | 0.7266 | 0.6999 | 0.5543 | 0.0994 [0.0940, 0.1046] | 0.0327 [0.0314, 0.0341] | 0.1261 [0.1213, 0.1308] |
| 6 s | 719 | 0.8156 | 0.7829 | 0.7289 | 0.6986 | 0.5492 | 0.0868 [0.0824, 0.0915] | 0.0328 [0.0317, 0.0339] | 0.1171 [0.1132, 0.1212] |
| 8 s | 719 | 0.8077 | 0.7752 | 0.7257 | 0.6941 | 0.5456 | 0.0820 [0.0780, 0.0863] | 0.0325 [0.0315, 0.0335] | 0.1136 [0.1100, 0.1175] |
| 10 s | 718 | 0.8022 | 0.7704 | 0.7224 | 0.6902 | 0.5435 | 0.0798 [0.0760, 0.0836] | 0.0318 [0.0308, 0.0328] | 0.1120 [0.1086, 0.1154] |

Secondary, grades pooled within user: head on the true grade / the head's own grade mix vs (a), (b), the user's running median:

| S | head (true grade) | head - (a) | head (grade mix) | grade mix - (a) | (c) all-grade user median |
|---|---|---|---|---|---|
| 2 s | 0.8468 | 0.1585 [0.1493, 0.1674] | 0.8370 | 0.1487 [0.1395, 0.1577] | 0.5513 |
| 3 s | 0.8482 | 0.1275 [0.1211, 0.1339] | 0.8388 | 0.1181 [0.1117, 0.1248] | 0.5512 |
| 4 s | 0.8429 | 0.1129 [0.1077, 0.1183] | 0.8318 | 0.1018 [0.0965, 0.1071] | 0.5516 |
| 6 s | 0.8344 | 0.1015 [0.0970, 0.1062] | 0.8220 | 0.0891 [0.0846, 0.0940] | 0.5481 |
| 8 s | 0.8271 | 0.0970 [0.0929, 0.1013] | 0.8143 | 0.0842 [0.0799, 0.0886] | 0.5455 |
| 10 s | 0.8215 | 0.0944 [0.0904, 0.0983] | 0.8089 | 0.0819 [0.0779, 0.0858] | 0.5438 |

## 2-3. Range and variance explained

Review-level slope: actual log time on the predicted log median within (user, grade), each user weighted by its predicted variance. Card level (out of sample: predictions on a card's first half, actual times on its second half): between-card slope, spread ratio = SD of predicted card means / SD of the persistent actual card means, explained share of the persistent between-card variance. Within cards: share of the within-card variance explained.

| model | predictor | review slope | median user slope | between-card slope | spread ratio | between-card explained | within-card explained |
|---|---|---|---|---|---|---|---|
| the six-layer Tiny two-head model (SM130) | head | 0.930 [0.921, 0.939] | 0.926 | 0.589 [0.562, 0.616] | 1.109 [1.086, 1.133] | 0.506 [0.488, 0.524] | 0.205 [0.178, 0.233] |
| the six-layer Tiny two-head model (SM130) | ablation | 0.922 [0.910, 0.934] | 0.928 | 0.610 [0.581, 0.639] | 1.027 [1.004, 1.051] | 0.473 [0.455, 0.492] | 0.155 [0.129, 0.182] |
| the six-layer Tiny two-head model (SM130) | (a) card median | 0.488 [0.471, 0.504] | 0.510 | 0.381 [0.364, 0.398] | 1.565 [1.531, 1.597] | 0.441 [0.423, 0.459] | 0.018 [0.016, 0.020] |
| the six-layer Tiny two-head model (SM130) | (b) card last | 0.313 [0.301, 0.325] | 0.321 | 0.325 [0.311, 0.341] | 1.679 [1.643, 1.718] | 0.383 [0.366, 0.401] | 0.012 [0.011, 0.014] |
| the six-layer Tiny two-head model (SM130) | (c) user grade median | 0.630 [0.522, 0.749] | 0.720 | 0.194 [0.085, 0.309] | 0.303 [0.276, 0.332] | 0.072 [0.060, 0.087] | 0.026 [0.020, 0.033] |
| the six-layer Tiny two-head model (SM130) | head - (a), head - ablation (between-card explained) | | | | | 0.065 [0.054, 0.077]; 0.033 [0.028, 0.038] | |
| the six-layer Tiny two-head model (SM130) | head, grades pooled within user | 0.939 [0.930, 0.947] | | | | | |
| the six-layer Tiny Instant-Only model (SM130) | head | 0.930 [0.922, 0.939] | 0.926 | 0.590 [0.563, 0.617] | 1.114 [1.091, 1.138] | 0.512 [0.494, 0.530] | 0.207 [0.180, 0.234] |
| the six-layer Tiny Instant-Only model (SM130) | ablation | 0.922 [0.910, 0.934] | 0.928 | 0.610 [0.581, 0.639] | 1.027 [1.004, 1.051] | 0.473 [0.455, 0.492] | 0.155 [0.129, 0.182] |
| the six-layer Tiny Instant-Only model (SM130) | (a) card median | 0.488 [0.471, 0.504] | 0.510 | 0.381 [0.364, 0.398] | 1.565 [1.531, 1.597] | 0.441 [0.423, 0.459] | 0.018 [0.016, 0.020] |
| the six-layer Tiny Instant-Only model (SM130) | (b) card last | 0.313 [0.301, 0.325] | 0.321 | 0.325 [0.311, 0.341] | 1.679 [1.643, 1.718] | 0.383 [0.366, 0.401] | 0.012 [0.011, 0.014] |
| the six-layer Tiny Instant-Only model (SM130) | (c) user grade median | 0.630 [0.522, 0.749] | 0.720 | 0.194 [0.085, 0.309] | 0.303 [0.276, 0.332] | 0.072 [0.060, 0.087] | 0.026 [0.020, 0.033] |
| the six-layer Tiny Instant-Only model (SM130) | head - (a), head - ablation (between-card explained) | | | | | 0.071 [0.060, 0.083]; 0.039 [0.034, 0.044] | |
| the six-layer Tiny Instant-Only model (SM130) | head, grades pooled within user | 0.939 [0.931, 0.947] | | | | | |
| the longer Small Instant-Only model (SM113) | head | 0.932 [0.924, 0.940] | 0.921 | 0.592 [0.565, 0.618] | 1.135 [1.112, 1.159] | 0.530 [0.512, 0.548] | 0.219 [0.192, 0.246] |
| the longer Small Instant-Only model (SM113) | ablation | 0.923 [0.911, 0.936] | 0.931 | 0.612 [0.583, 0.641] | 1.023 [1.001, 1.047] | 0.473 [0.455, 0.492] | 0.155 [0.129, 0.181] |
| the longer Small Instant-Only model (SM113) | (a) card median | 0.488 [0.471, 0.504] | 0.510 | 0.381 [0.364, 0.398] | 1.565 [1.531, 1.597] | 0.441 [0.423, 0.459] | 0.018 [0.016, 0.020] |
| the longer Small Instant-Only model (SM113) | (b) card last | 0.313 [0.301, 0.325] | 0.321 | 0.325 [0.311, 0.341] | 1.679 [1.643, 1.718] | 0.383 [0.366, 0.401] | 0.012 [0.011, 0.014] |
| the longer Small Instant-Only model (SM113) | (c) user grade median | 0.630 [0.522, 0.749] | 0.720 | 0.194 [0.085, 0.309] | 0.303 [0.276, 0.332] | 0.072 [0.060, 0.087] | 0.026 [0.020, 0.033] |
| the longer Small Instant-Only model (SM113) | head - (a), head - ablation (between-card explained) | | | | | 0.089 [0.077, 0.101]; 0.057 [0.051, 0.062] | |
| the longer Small Instant-Only model (SM113) | head, grades pooled within user | 0.939 [0.931, 0.948] | | | | | |
| the show-time two-head cool-down for Small (SM118) | head | 0.929 [0.921, 0.938] | 0.923 | 0.590 [0.563, 0.616] | 1.131 [1.108, 1.155] | 0.526 [0.508, 0.544] | 0.217 [0.190, 0.244] |
| the show-time two-head cool-down for Small (SM118) | ablation | 0.923 [0.911, 0.936] | 0.931 | 0.612 [0.583, 0.641] | 1.023 [1.001, 1.047] | 0.473 [0.455, 0.492] | 0.155 [0.129, 0.181] |
| the show-time two-head cool-down for Small (SM118) | (a) card median | 0.488 [0.471, 0.504] | 0.510 | 0.381 [0.364, 0.398] | 1.565 [1.531, 1.597] | 0.441 [0.423, 0.459] | 0.018 [0.016, 0.020] |
| the show-time two-head cool-down for Small (SM118) | (b) card last | 0.313 [0.301, 0.325] | 0.321 | 0.325 [0.311, 0.341] | 1.679 [1.643, 1.718] | 0.383 [0.366, 0.401] | 0.012 [0.011, 0.014] |
| the show-time two-head cool-down for Small (SM118) | (c) user grade median | 0.630 [0.522, 0.749] | 0.720 | 0.194 [0.085, 0.309] | 0.303 [0.276, 0.332] | 0.072 [0.060, 0.087] | 0.026 [0.020, 0.033] |
| the show-time two-head cool-down for Small (SM118) | head - (a), head - ablation (between-card explained) | | | | | 0.086 [0.074, 0.097]; 0.053 [0.048, 0.059] | |
| the show-time two-head cool-down for Small (SM118) | head, grades pooled within user | 0.938 [0.930, 0.946] | | | | | |
| the show-time instant-only big model (SM99) | head | 0.929 [0.920, 0.938] | 0.930 | 0.589 [0.563, 0.614] | 1.147 [1.125, 1.172] | 0.535 [0.517, 0.553] | 0.219 [0.194, 0.247] |
| the show-time instant-only big model (SM99) | ablation | 0.928 [0.916, 0.940] | 0.936 | 0.612 [0.583, 0.641] | 1.020 [0.997, 1.044] | 0.471 [0.453, 0.489] | 0.157 [0.132, 0.184] |
| the show-time instant-only big model (SM99) | (a) card median | 0.487 [0.472, 0.503] | 0.510 | 0.380 [0.363, 0.397] | 1.565 [1.533, 1.599] | 0.441 [0.423, 0.458] | 0.018 [0.016, 0.020] |
| the show-time instant-only big model (SM99) | (b) card last | 0.314 [0.302, 0.326] | 0.321 | 0.325 [0.310, 0.341] | 1.680 [1.643, 1.718] | 0.383 [0.366, 0.401] | 0.012 [0.010, 0.014] |
| the show-time instant-only big model (SM99) | (c) user grade median | 0.630 [0.518, 0.747] | 0.720 | 0.195 [0.080, 0.307] | 0.304 [0.277, 0.333] | 0.072 [0.059, 0.087] | 0.026 [0.020, 0.032] |
| the show-time instant-only big model (SM99) | head - (a), head - ablation (between-card explained) | | | | | 0.094 [0.082, 0.106]; 0.064 [0.058, 0.070] | |
| the show-time instant-only big model (SM99) | head, grades pooled within user | 0.937 [0.929, 0.946] | | | | | |

Persistent between-card share of the within-(user, grade) variance (median user): the six-layer Tiny two-head model (SM130) 33.8%, the six-layer Tiny Instant-Only model (SM130) 33.8%, the longer Small Instant-Only model (SM113) 33.8%, the show-time two-head cool-down for Small (SM118) 33.8%, the show-time instant-only big model (SM99) 33.9%

## 4. Shapes

Effective number of distinct curves = exp(entropy) of the cells of the quantile vectors (q10, q25, q50, q75, q90 of the true-grade distribution) rounded to 0.1 ln-units (about 10% in time) / 0.2; location-normalised = quantiles minus the median. Per user: median [p10, p90] over users. PCA on CDF grids of a pooled sample (200 rows per user).

| model | head | per-user curves, location incl. 0.1 / 0.2 | per-user shapes, normalised 0.1 / 0.2 | pooled PCA normalised k90/k95/k99, PR | pooled PCA location incl. k90/k95/k99, PR | per-user SD of predicted log median / of log IQR |
|---|---|---|---|---|---|---|
| the six-layer Tiny two-head model (SM130) | head | 717 [344, 1515] / 163 | 81 [31, 249] / 20 | 5/8/24, 2.93 | 2/3/5, 1.65 | 0.571 / 0.185 |
| the six-layer Tiny two-head model (SM130) | ablation | 807 [357, 1758] / 176 | 98 [38, 268] / 24 | 5/8/24, 3.04 | 2/3/4, 1.61 |  |
| the six-layer Tiny Instant-Only model (SM130) | head | 709 [354, 1427] / 163 | 76 [31, 231] / 20 | 5/8/24, 2.88 | 2/3/5, 1.64 | 0.573 / 0.186 |
| the six-layer Tiny Instant-Only model (SM130) | ablation | 807 [357, 1758] / 176 | 98 [38, 268] / 24 | 5/8/24, 3.04 | 2/3/4, 1.61 |  |
| the longer Small Instant-Only model (SM113) | head | 711 [350, 1493] / 166 | 80 [32, 239] / 21 | 4/8/23, 2.79 | 2/3/5, 1.65 | 0.581 / 0.191 |
| the longer Small Instant-Only model (SM113) | ablation | 843 [371, 1828] / 183 | 103 [41, 278] / 25 | 5/8/24, 3.07 | 2/3/4, 1.61 |  |
| the show-time two-head cool-down for Small (SM118) | head | 681 [321, 1469] / 161 | 74 [30, 224] / 20 | 5/8/23, 2.86 | 2/3/5, 1.65 | 0.576 / 0.188 |
| the show-time two-head cool-down for Small (SM118) | ablation | 843 [371, 1828] / 183 | 103 [41, 278] / 25 | 5/8/24, 3.07 | 2/3/4, 1.61 |  |
| the show-time instant-only big model (SM99) | head | 804 [354, 1748] / 182 | 87 [33, 275] / 23 | 4/7/22, 2.78 | 2/3/5, 1.66 | 0.580 / 0.195 |
| the show-time instant-only big model (SM99) | ablation | 801 [345, 1837] / 178 | 95 [37, 275] / 23 | 5/8/24, 3.04 | 2/3/4, 1.61 |  |

Median scored rows per user: the six-layer Tiny two-head model (SM130) 12561, the six-layer Tiny Instant-Only model (SM130) 12561, the longer Small Instant-Only model (SM113) 12561, the show-time two-head cool-down for Small (SM118) 12561, the show-time instant-only big model (SM99) 12461

Shape calibration: card level = Spearman, within user, of the predicted IQR (first half of the card) with the realised SD of the card's later times (second half), against the card's own earlier SD; review level = Spearman of the predicted IQR with |actual - predicted median|; coverage = share of answers inside the predicted IQR by within-user quintile of the predicted IQR (ideal 50%).

| model | card: head | card: own earlier SD | head - own | review level | coverage by width quintile (narrow to wide) | coverage all |
|---|---|---|---|---|---|---|
| the six-layer Tiny two-head model (SM130) | 0.086 [0.077, 0.095] | 0.094 [0.087, 0.100] | -0.008 [-0.016, 0.000] | 0.199 [0.191, 0.206] | 54% / 52% / 52% / 53% / 54% | 53.1% |
| the six-layer Tiny Instant-Only model (SM130) | 0.085 [0.076, 0.094] | 0.094 [0.087, 0.100] | -0.008 [-0.016, -0.000] | 0.200 [0.193, 0.208] | 54% / 52% / 52% / 53% / 54% | 53.0% |
| the longer Small Instant-Only model (SM113) | 0.093 [0.084, 0.102] | 0.094 [0.087, 0.100] | -0.001 [-0.008, 0.007] | 0.213 [0.205, 0.221] | 54% / 53% / 52% / 53% / 53% | 53.1% |
| the show-time two-head cool-down for Small (SM118) | 0.090 [0.081, 0.099] | 0.094 [0.087, 0.100] | -0.003 [-0.011, 0.005] | 0.207 [0.199, 0.215] | 54% / 53% / 52% / 52% / 53% | 53.0% |
| the show-time instant-only big model (SM99) | 0.100 [0.091, 0.110] | 0.093 [0.087, 0.100] | 0.007 [-0.001, 0.015] | 0.220 [0.212, 0.227] | 54% / 53% / 52% / 52% / 53% | 53.0% |

## 5. Life of a card

Answer count = earlier reviews of the card. Levels as % of the user's typical time (geometric mean of its scored answers); each user equal.

### the six-layer Tiny two-head model (SM130)

| grades | earlier reviews | users | predicted median | actual mean | predicted - actual (ln) | predicted IQR (ln) | realised IQR of residuals (ln) | Again / Hard / Good / Easy share |
|---|---|---|---|---|---|---|---|---|
| all | 1 | 698 | -4% | +0% | -0.040 [-0.048, -0.033] | 0.927 | 0.831 | 11% / 11% / 55% / 22% |
| all | 2-3 | 722 | -6% | -3% | -0.032 [-0.037, -0.027] | 0.819 | 0.764 | 11% / 15% / 63% / 11% |
| all | 4-7 | 722 | -3% | +1% | -0.038 [-0.044, -0.032] | 0.801 | 0.744 | 12% / 18% / 62% / 7% |
| all | 8-15 | 693 | +4% | +8% | -0.036 [-0.044, -0.029] | 0.810 | 0.748 | 15% / 25% / 55% / 6% |
| all | 16-31 | 580 | +6% | +10% | -0.037 [-0.046, -0.027] | 0.818 | 0.747 | 16% / 27% / 52% / 5% |
| all | 32+ | 377 | +1% | +5% | -0.038 [-0.051, -0.024] | 0.825 | 0.768 | 15% / 28% / 53% / 3% |
| good | 1 | 668 | -7% | -2% | -0.055 [-0.064, -0.047] | 0.932 | 0.844 | 0% / 0% / 100% / 0% |
| good | 2-3 | 714 | -13% | -10% | -0.038 [-0.044, -0.031] | 0.811 | 0.764 | 0% / 0% / 100% / 0% |
| good | 4-7 | 708 | -13% | -9% | -0.045 [-0.052, -0.039] | 0.786 | 0.730 | 0% / 0% / 100% / 0% |
| good | 8-15 | 662 | -8% | -3% | -0.052 [-0.060, -0.044] | 0.793 | 0.732 | 0% / 0% / 100% / 0% |
| good | 16-31 | 517 | -7% | -2% | -0.054 [-0.064, -0.044] | 0.790 | 0.717 | 0% / 0% / 100% / 0% |
| good | 32+ | 313 | -8% | -2% | -0.061 [-0.076, -0.046] | 0.798 | 0.722 | 0% / 0% / 100% / 0% |

### the six-layer Tiny Instant-Only model (SM130)

| grades | earlier reviews | users | predicted median | actual mean | predicted - actual (ln) | predicted IQR (ln) | realised IQR of residuals (ln) | Again / Hard / Good / Easy share |
|---|---|---|---|---|---|---|---|---|
| all | 1 | 698 | -4% | +0% | -0.045 [-0.052, -0.037] | 0.918 | 0.832 | 11% / 11% / 55% / 22% |
| all | 2-3 | 722 | -6% | -3% | -0.037 [-0.043, -0.032] | 0.819 | 0.761 | 11% / 15% / 63% / 11% |
| all | 4-7 | 722 | -3% | +1% | -0.039 [-0.045, -0.033] | 0.798 | 0.742 | 12% / 18% / 62% / 7% |
| all | 8-15 | 693 | +4% | +8% | -0.041 [-0.048, -0.034] | 0.803 | 0.743 | 15% / 25% / 55% / 6% |
| all | 16-31 | 580 | +5% | +10% | -0.049 [-0.058, -0.040] | 0.806 | 0.744 | 16% / 27% / 52% / 5% |
| all | 32+ | 377 | -1% | +5% | -0.058 [-0.070, -0.046] | 0.810 | 0.764 | 15% / 28% / 53% / 3% |
| good | 1 | 668 | -7% | -2% | -0.054 [-0.064, -0.046] | 0.919 | 0.842 | 0% / 0% / 100% / 0% |
| good | 2-3 | 714 | -13% | -10% | -0.042 [-0.048, -0.035] | 0.809 | 0.757 | 0% / 0% / 100% / 0% |
| good | 4-7 | 708 | -13% | -9% | -0.044 [-0.050, -0.037] | 0.783 | 0.727 | 0% / 0% / 100% / 0% |
| good | 8-15 | 662 | -8% | -3% | -0.053 [-0.061, -0.045] | 0.783 | 0.727 | 0% / 0% / 100% / 0% |
| good | 16-31 | 517 | -8% | -2% | -0.064 [-0.074, -0.054] | 0.775 | 0.714 | 0% / 0% / 100% / 0% |
| good | 32+ | 313 | -9% | -2% | -0.076 [-0.089, -0.062] | 0.778 | 0.720 | 0% / 0% / 100% / 0% |

### the longer Small Instant-Only model (SM113)

| grades | earlier reviews | users | predicted median | actual mean | predicted - actual (ln) | predicted IQR (ln) | realised IQR of residuals (ln) | Again / Hard / Good / Easy share |
|---|---|---|---|---|---|---|---|---|
| all | 1 | 698 | -3% | +0% | -0.036 [-0.042, -0.029] | 0.902 | 0.811 | 11% / 11% / 55% / 22% |
| all | 2-3 | 722 | -7% | -3% | -0.040 [-0.045, -0.034] | 0.805 | 0.742 | 11% / 15% / 63% / 11% |
| all | 4-7 | 722 | -3% | +1% | -0.034 [-0.040, -0.029] | 0.781 | 0.723 | 12% / 18% / 62% / 7% |
| all | 8-15 | 693 | +5% | +8% | -0.029 [-0.036, -0.022] | 0.783 | 0.723 | 15% / 25% / 55% / 6% |
| all | 16-31 | 580 | +6% | +10% | -0.034 [-0.044, -0.025] | 0.790 | 0.725 | 16% / 27% / 52% / 5% |
| all | 32+ | 377 | +1% | +5% | -0.046 [-0.058, -0.033] | 0.807 | 0.740 | 15% / 28% / 53% / 3% |
| good | 1 | 668 | -7% | -2% | -0.048 [-0.057, -0.040] | 0.909 | 0.822 | 0% / 0% / 100% / 0% |
| good | 2-3 | 714 | -14% | -10% | -0.046 [-0.052, -0.040] | 0.796 | 0.741 | 0% / 0% / 100% / 0% |
| good | 4-7 | 708 | -13% | -9% | -0.045 [-0.051, -0.038] | 0.763 | 0.710 | 0% / 0% / 100% / 0% |
| good | 8-15 | 662 | -7% | -3% | -0.043 [-0.051, -0.036] | 0.763 | 0.709 | 0% / 0% / 100% / 0% |
| good | 16-31 | 517 | -7% | -2% | -0.052 [-0.062, -0.042] | 0.758 | 0.700 | 0% / 0% / 100% / 0% |
| good | 32+ | 313 | -8% | -2% | -0.067 [-0.081, -0.053] | 0.771 | 0.701 | 0% / 0% / 100% / 0% |

### the show-time two-head cool-down for Small (SM118)

| grades | earlier reviews | users | predicted median | actual mean | predicted - actual (ln) | predicted IQR (ln) | realised IQR of residuals (ln) | Again / Hard / Good / Easy share |
|---|---|---|---|---|---|---|---|---|
| all | 1 | 698 | -2% | +0% | -0.023 [-0.030, -0.016] | 0.897 | 0.816 | 11% / 11% / 55% / 22% |
| all | 2-3 | 722 | -6% | -3% | -0.033 [-0.039, -0.027] | 0.808 | 0.747 | 11% / 15% / 63% / 11% |
| all | 4-7 | 722 | -3% | +1% | -0.033 [-0.039, -0.028] | 0.787 | 0.728 | 12% / 18% / 62% / 7% |
| all | 8-15 | 693 | +4% | +8% | -0.034 [-0.041, -0.027] | 0.789 | 0.730 | 15% / 25% / 55% / 6% |
| all | 16-31 | 580 | +6% | +10% | -0.036 [-0.045, -0.027] | 0.791 | 0.729 | 16% / 27% / 52% / 5% |
| all | 32+ | 377 | +2% | +5% | -0.034 [-0.046, -0.021] | 0.803 | 0.747 | 15% / 28% / 53% / 3% |
| good | 1 | 668 | -6% | -2% | -0.039 [-0.048, -0.030] | 0.900 | 0.827 | 0% / 0% / 100% / 0% |
| good | 2-3 | 714 | -14% | -10% | -0.042 [-0.048, -0.036] | 0.799 | 0.745 | 0% / 0% / 100% / 0% |
| good | 4-7 | 708 | -13% | -9% | -0.045 [-0.051, -0.039] | 0.771 | 0.713 | 0% / 0% / 100% / 0% |
| good | 8-15 | 662 | -8% | -3% | -0.053 [-0.060, -0.045] | 0.769 | 0.714 | 0% / 0% / 100% / 0% |
| good | 16-31 | 517 | -7% | -2% | -0.057 [-0.066, -0.047] | 0.757 | 0.702 | 0% / 0% / 100% / 0% |
| good | 32+ | 313 | -8% | -2% | -0.062 [-0.075, -0.049] | 0.762 | 0.709 | 0% / 0% / 100% / 0% |

### the show-time instant-only big model (SM99)

| grades | earlier reviews | users | predicted median | actual mean | predicted - actual (ln) | predicted IQR (ln) | realised IQR of residuals (ln) | Again / Hard / Good / Easy share |
|---|---|---|---|---|---|---|---|---|
| all | 1 | 695 | -3% | +0% | -0.039 [-0.046, -0.032] | 0.882 | 0.806 | 11% / 12% / 55% / 22% |
| all | 2-3 | 719 | -6% | -3% | -0.036 [-0.042, -0.031] | 0.800 | 0.739 | 11% / 15% / 63% / 11% |
| all | 4-7 | 719 | -4% | +1% | -0.042 [-0.047, -0.036] | 0.782 | 0.719 | 12% / 18% / 62% / 7% |
| all | 8-15 | 690 | +4% | +8% | -0.039 [-0.046, -0.032] | 0.788 | 0.720 | 15% / 26% / 53% / 6% |
| all | 16-31 | 577 | +5% | +10% | -0.046 [-0.054, -0.037] | 0.790 | 0.722 | 16% / 29% / 50% / 5% |
| all | 32+ | 374 | -1% | +5% | -0.060 [-0.072, -0.047] | 0.796 | 0.743 | 16% / 31% / 50% / 3% |
| good | 1 | 665 | -7% | -2% | -0.048 [-0.056, -0.039] | 0.884 | 0.814 | 0% / 0% / 100% / 0% |
| good | 2-3 | 711 | -14% | -10% | -0.045 [-0.051, -0.039] | 0.790 | 0.737 | 0% / 0% / 100% / 0% |
| good | 4-7 | 705 | -13% | -9% | -0.053 [-0.059, -0.046] | 0.763 | 0.705 | 0% / 0% / 100% / 0% |
| good | 8-15 | 659 | -8% | -3% | -0.057 [-0.065, -0.049] | 0.763 | 0.703 | 0% / 0% / 100% / 0% |
| good | 16-31 | 514 | -8% | -2% | -0.068 [-0.077, -0.059] | 0.751 | 0.693 | 0% / 0% / 100% / 0% |
| good | 32+ | 310 | -10% | -2% | -0.088 [-0.101, -0.074] | 0.750 | 0.699 | 0% / 0% / 100% / 0% |

## 6. Plain-language read-out

- The six-layer Tiny two-head model (SM130): a typical user gets about 717 distinguishable time curves (location included, 10% grid) and about 81 distinguishable shapes once the location is removed. Within a grade, the head ranks this user's reviews by speed with AUC 0.816 at 4 s against 0.727 for the card's own running median (+0.138 at 2 s, +0.090 at 4 s, +0.074 at 8 s). It explains 51% of the persistent card-to-card differences against 44% for the card's own median and 47% for the explicit-features-only head. Label: partly expressive.
- The six-layer Tiny Instant-Only model (SM130): a typical user gets about 709 distinguishable time curves (location included, 10% grid) and about 76 distinguishable shapes once the location is removed. Within a grade, the head ranks this user's reviews by speed with AUC 0.818 at 4 s against 0.727 for the card's own running median (+0.139 at 2 s, +0.092 at 4 s, +0.075 at 8 s). It explains 51% of the persistent card-to-card differences against 44% for the card's own median and 47% for the explicit-features-only head. Label: partly expressive.
- The longer Small Instant-Only model (SM113): a typical user gets about 711 distinguishable time curves (location included, 10% grid) and about 80 distinguishable shapes once the location is removed. Within a grade, the head ranks this user's reviews by speed with AUC 0.825 at 4 s against 0.727 for the card's own running median (+0.149 at 2 s, +0.099 at 4 s, +0.081 at 8 s). It explains 53% of the persistent card-to-card differences against 44% for the card's own median and 47% for the explicit-features-only head. Label: partly expressive.
- The show-time two-head cool-down for Small (SM118): a typical user gets about 681 distinguishable time curves (location included, 10% grid) and about 74 distinguishable shapes once the location is removed. Within a grade, the head ranks this user's reviews by speed with AUC 0.823 at 4 s against 0.727 for the card's own running median (+0.146 at 2 s, +0.096 at 4 s, +0.079 at 8 s). It explains 53% of the persistent card-to-card differences against 44% for the card's own median and 47% for the explicit-features-only head. Label: partly expressive.
- The show-time instant-only big model (SM99): a typical user gets about 804 distinguishable time curves (location included, 10% grid) and about 87 distinguishable shapes once the location is removed. Within a grade, the head ranks this user's reviews by speed with AUC 0.826 at 4 s against 0.727 for the card's own running median (+0.146 at 2 s, +0.099 at 4 s, +0.082 at 8 s). It explains 53% of the persistent card-to-card differences against 44% for the card's own median and 47% for the explicit-features-only head. Label: partly expressive.
