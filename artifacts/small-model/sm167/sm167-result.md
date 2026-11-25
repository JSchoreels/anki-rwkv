# Links between cards (SM167): result

Label T24 (`w8+lr2.16.4q8x8@e2s-query`), training users 100, 732 set users 723, 30,479,310 target reviews. Design: design.md (pre-registered).

## Coverage (732 set, share of target reviews)

| link type | has a linked card in the window | with a matched control | linked cards per covered review |
|---|---:|---:|---:|
| co-failure (a), C >= 2 and lift >= 2 | 10.10% | 9.88% | 5.45 |
| co-failure loose (a1), C >= 1 | 28.59% | 27.52% | 8.93 |
| confusable order (b), A -> X | 1.32% | 1.29% | 1.20 |

## Small Instant-Only (SM113)

Raw benchmark instant LogLoss 0.27811 on 723 users.

| link type | target reviews | corr(linked) | corr(matched control) | difference [95% CI] | slope linked | slope control |
|---|---:|---:|---:|---:|---:|---:|
| co-failure (a), C >= 2 and lift >= 2 | 3,011,915 | +0.0056 | -0.0001 | +0.0057 [+0.0037, +0.0078] | +0.0084 | -0.0001 |
| co-failure loose (a1), C >= 1 | 8,387,268 | +0.0039 | -0.0002 | +0.0041 [+0.0029, +0.0053] | +0.0074 | -0.0004 |
| confusable order (b), A -> X | 393,135 | +0.0027 | +0.0013 | +0.0014 [-0.0037, +0.0062] | +0.0030 | +0.0014 |

| correction (beta fit on the 100 training users) | LogLoss change [95% CI] |
|---|---:|
| links beyond session baseline (M1 vs M0), the go statistic | -0.0017% [-0.0024%, -0.0009%] |
| session baseline vs raw model (M0 vs raw), information | +0.0015% [+0.0006%, +0.0025%] |
| everything vs raw model (M1 vs raw), information | -0.0002% [-0.0015%, +0.0011%] |
| links beyond session, beta fit on the 732 set itself (in-sample upper bound) | -0.0017% [-0.0024%, -0.0009%] |

Betas M1 (training users): f_sess +0.1345, f_deck +0.0293, f_a +0.0477, i_a +0.0211, f_b +0.0077, i_b -0.0003.

## Small two-head (SM118), instant head

Raw benchmark instant LogLoss 0.27876 on 723 users.

| link type | target reviews | corr(linked) | corr(matched control) | difference [95% CI] | slope linked | slope control |
|---|---:|---:|---:|---:|---:|---:|
| co-failure (a), C >= 2 and lift >= 2 | 3,011,915 | +0.0066 | +0.0002 | +0.0064 [+0.0044, +0.0086] | +0.0100 | +0.0003 |
| co-failure loose (a1), C >= 1 | 8,387,268 | +0.0047 | -0.0001 | +0.0048 [+0.0036, +0.0060] | +0.0089 | -0.0001 |
| confusable order (b), A -> X | 393,135 | +0.0034 | +0.0013 | +0.0021 [-0.0029, +0.0068] | +0.0038 | +0.0015 |

| correction (beta fit on the 100 training users) | LogLoss change [95% CI] |
|---|---:|
| links beyond session baseline (M1 vs M0), the go statistic | -0.0009% [-0.0014%, -0.0004%] |
| session baseline vs raw model (M0 vs raw), information | +0.0017% [+0.0005%, +0.0030%] |
| everything vs raw model (M1 vs raw), information | +0.0007% [-0.0007%, +0.0022%] |
| links beyond session, beta fit on the 732 set itself (in-sample upper bound) | -0.0009% [-0.0014%, -0.0004%] |

Betas M1 (training users): f_sess +0.1561, f_deck +0.0357, f_a +0.0560, i_a +0.0105, f_b +0.0107, i_b -0.0017.

## Big Instant-Only (SM99)

Raw benchmark instant LogLoss 0.27585 on 718 users.

| link type | target reviews | corr(linked) | corr(matched control) | difference [95% CI] | slope linked | slope control |
|---|---:|---:|---:|---:|---:|---:|
| co-failure (a), C >= 2 and lift >= 2 | 2,791,360 | +0.0065 | +0.0004 | +0.0061 [+0.0041, +0.0081] | +0.0099 | +0.0007 |
| co-failure loose (a1), C >= 1 | 7,672,249 | +0.0052 | +0.0006 | +0.0046 [+0.0034, +0.0057] | +0.0098 | +0.0010 |
| confusable order (b), A -> X | 368,766 | +0.0038 | +0.0019 | +0.0020 [-0.0032, +0.0070] | +0.0042 | +0.0021 |

| correction (beta fit on the 100 training users) | LogLoss change [95% CI] |
|---|---:|
| links beyond session baseline (M1 vs M0), the go statistic | -0.0013% [-0.0022%, -0.0005%] |
| session baseline vs raw model (M0 vs raw), information | -0.0009% [-0.0019%, +0.0002%] |
| everything vs raw model (M1 vs raw), information | -0.0021% [-0.0035%, -0.0008%] |
| links beyond session, beta fit on the 732 set itself (in-sample upper bound) | -0.0014% [-0.0021%, -0.0007%] |

Betas M1 (training users): f_sess +0.1379, f_deck +0.0352, f_a +0.0387, i_a -0.0231, f_b +0.0096, i_b -0.0132.

## Decision: NO GO

Go rule (design.md): for Small Instant-Only (SM113), the link gain M1 vs M0 needs its 95% CI upper bound below 0 and a point gain of at least 0.05%.
