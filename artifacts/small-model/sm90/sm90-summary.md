# Codebook state compression sweep (SM90): finalists on the 732 set (722 users)

Released small model (the leak-free small model (SM72s), cool-down from epoch 25), w8 weights, leak-free inputs. Delta vs int4 cards + int8 notes (`w8+s4c8n`, the fork default) on the same users; points = % of the big model (1 leak-free epoch) LogLoss; [ ] = 95% paired user bootstrap (2,000, seed 0).

- int4 cards + int8 notes: curve 0.30007 (+0.37% vs big), instant 0.27659 (+0.30% vs big); state 48.1 MB (fork format)

| Recipe | Card / note bits | State MB | Codebooks MB (f16) | Encode ms / review (est.) | GPU reviews/s | Curve vs s4c8n | Instant vs s4c8n | Nearly free | Practical |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| rel/w8+vqC4430 | 4430 / 12536 | 28.4 | 9.0 | 2.72 | 13,140 | +0.000065 [+0.000046, +0.000086] = +0.022 pts [+0.015, +0.029]; +0.39% vs big | +0.000078 [+0.000062, +0.000097] = +0.028 pts [+0.022, +0.035]; +0.33% vs big | no | no |
| rel/w8+vqC4430N3020 | - / - | 0.0 | 0.0 | 0.00 | 5,301 | +0.000126 [+0.000099, +0.000154] = +0.042 pts [+0.034, +0.052]; +0.41% vs big | +0.000180 [+0.000155, +0.000210] = +0.065 pts [+0.056, +0.076]; +0.37% vs big | no | no |
| rel/w8+vqC5902 | 5902 / 12536 | 31.0 | 12.0 | 3.59 | 11,295 | +0.000014 [-0.000002, +0.000030] = +0.005 pts [-0.001, +0.010]; +0.37% vs big | +0.000012 [-0.000002, +0.000026] = +0.004 pts [-0.001, +0.009]; +0.31% vs big | yes | no |
| rel/w8+vqN3020 | 15704 / 3020 | 32.4 | 11.7 | 1.92 | 8,529 | +0.000054 [+0.000038, +0.000071] = +0.018 pts [+0.013, +0.024]; +0.39% vs big | +0.000091 [+0.000077, +0.000108] = +0.033 pts [+0.028, +0.039]; +0.34% vs big | no | no |

Smallest nearly-free card budget: rel/w8+vqC5902 (practical: False)
Smallest nearly-free note budget: none of the finalists
Smallest nearly-free both budget: none of the finalists
Rule: extra cost <= 0.02 points on both heads and total <= +0.5% curve / +0.4% instant vs the big model (point estimates); practical = codebooks <= 10.0 MB at float16 and encode <= 1.0 ms per review (est.).
