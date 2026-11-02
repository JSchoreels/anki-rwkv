# Codebook state compression sweep (SM90): screen (40 users of the 732 set)

Released small model (the leak-free small model (SM72s), cool-down from epoch 25), w8 weights, leak-free inputs. Delta vs int4 cards + int8 notes (`w8+s4c8n`, the fork default) on the same users; points = % of the big model (1 leak-free epoch) LogLoss; [ ] = 95% paired user bootstrap (2,000, seed 0). Screen threshold 0.04 points (the final rule is 0.02).

- w8+s8 vs the recorded released rows: {'users': 40, 'max_abs_delta': {'curve': 0.0, 'instant': 0.0}, 'passes_1e-5': True}
- int4 cards + int8 notes vs big: curve +0.87%, instant +0.33%

| Recipe | Card / note bits | State MB | Codebooks MB (f16) | Encode ms / review (est.) | GPU reviews/s | Curve vs s4c8n | Instant vs s4c8n |
|---|---:|---:|---:|---:|---:|---|---|
| rel/w8+s8 | - / - | 0.0 | 0.0 | 0.00 | 104,307 | -0.000016 [-0.000043, +0.000008] = -0.006 pts [-0.016, +0.003]; +0.86% vs big | -0.000040 [-0.000088, -0.000004] = -0.017 pts [-0.037, -0.002]; +0.31% vs big |
| rel/w8+vq5902x3028 | 5902 / 3028 | 15.3 | 11.9 | 5.40 | 4,995 | +0.000014 [-0.000077, +0.000082] = +0.005 pts [-0.031, +0.030]; +0.87% vs big | +0.000058 [-0.000001, +0.000116] = +0.024 pts [-0.001, +0.047]; +0.35% vs big |
| rel/w8+vqC5902 | 5902 / 12536 | 31.0 | 12.0 | 3.59 | 11,102 | -0.000009 [-0.000084, +0.000039] = -0.003 pts [-0.034, +0.014]; +0.87% vs big | +0.000001 [-0.000053, +0.000047] = +0.000 pts [-0.022, +0.020]; +0.33% vs big |
| rel/w8+vqC4430 | 4430 / 12536 | 28.4 | 9.0 | 2.72 | 13,377 | +0.000024 [-0.000090, +0.000102] = +0.009 pts [-0.036, +0.036]; +0.88% vs big | +0.000068 [+0.000018, +0.000125] = +0.028 pts [+0.008, +0.048]; +0.36% vs big |
| rel/w8+vqC2958 | 2958 / 12536 | 25.8 | 6.0 | 1.85 | 16,851 | +0.000049 [-0.000215, +0.000211] = +0.019 pts [-0.087, +0.078]; +0.89% vs big | +0.000176 [+0.000106, +0.000256] = +0.073 pts [+0.045, +0.104]; +0.40% vs big |
| rel/w8+vqC1934 | 1934 / 12536 | 24.1 | 4.2 | 1.25 | 19,824 | +0.000248 [-0.000166, +0.000548] = +0.093 pts [-0.063, +0.195]; +0.96% vs big | +0.000463 [+0.000292, +0.000648] = +0.192 pts [+0.128, +0.254]; +0.52% vs big |
| rel/w8+vqC1470 | 1470 / 12536 | 23.2 | 2.9 | 0.94 | 22,708 | +0.000457 [-0.000102, +0.000868] = +0.172 pts [-0.041, +0.310]; +1.04% vs big | +0.000881 [+0.000616, +0.001190] = +0.366 pts [+0.271, +0.461]; +0.69% vs big |
| rel/w8+vqN3964 | 15704 / 3964 | 34.0 | 13.2 | 2.36 | 7,102 | +0.000007 [-0.000007, +0.000022] = +0.003 pts [-0.003, +0.008]; +0.87% vs big | +0.000014 [+0.000001, +0.000031] = +0.006 pts [+0.001, +0.012]; +0.33% vs big |
| rel/w8+vqN3020 | 15704 / 3020 | 32.4 | 11.7 | 1.92 | 8,321 | +0.000014 [-0.000020, +0.000042] = +0.005 pts [-0.008, +0.015]; +0.87% vs big | +0.000038 [+0.000013, +0.000070] = +0.016 pts [+0.005, +0.029]; +0.34% vs big |
| rel/w8+vqN1524 | 15704 / 1524 | 29.9 | 5.8 | 0.97 | 12,243 | +0.000036 [-0.000053, +0.000118] = +0.014 pts [-0.021, +0.041]; +0.88% vs big | +0.000130 [+0.000074, +0.000196] = +0.054 pts [+0.032, +0.078]; +0.38% vs big |

Screen extra cost, max of the two heads (points): vqC5902 +0.000, vqC4430 +0.028, vqC2958 +0.073, vqC1934 +0.192, vqC1470 +0.366, vqN3964 +0.006, vqN3020 +0.016, vqN1524 +0.054
Finalists (card: smallest rung within 0.04 points = vqC4430; both streams coded: vqC4430 + vqN3020; card safety rung vqC5902; note: smallest rung within 0.04 points = vqN3020): ['rel/w8+s4c8n', 'rel/w8+vqC4430', 'rel/w8+vqC4430N3020', 'rel/w8+vqC5902', 'rel/w8+vqN3020']
