# SM48b timing: lanes with int state variants

40 users of the 732 pool, SM36 w8+s8, w8+s4c8n, w8+s8hx8, w8+s4c8nhx8 = 9,022,128 review-evals per run; rate = scorer counter; mem = nvidia-smi max / torch peak allocated, reserved.

| Run | Flags | review-evals/s | vs lanes 1 | process s | GPU util % | nvidia-smi max MiB | torch alloc / reserved GiB | rows | err |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| 1-l1 | `default` | 84,535 | 1.17x | 110.6 | 65.5 | 13853 | 6.81 / 11.33 | 160 | 0 |
| 2-l2 | `--lanes 2` | 94,576 | 1.31x | 99.3 | 72.0 | 14608 | 6.82 / 12.07 | 160 | 0 |
| 3-l4 | `--lanes 4` | 98,355 | 1.36x | 95.3 | 74.8 | 15248 | 6.86 / 12.71 | 160 | 0 |
| 4-l4b250 | `--lanes 4 --lane-review-budget 250000` | 89,919 | 1.24x | 104.4 | 69.5 | 15179 | 6.88 / 12.71 | 160 | 0 |
| 5-l4b150 | `--lanes 4 --lane-review-budget 150000` | 88,190 | 1.22x | 106.2 | 67.5 | 15045 | 6.88 / 12.64 | 160 | 0 |
| 6-l4lock | `--lanes 4` (lock) | 85,111 | 1.18x | 111.2 | 69.3 | 14861 | 6.86 / 12.71 | 160 | 0 |
| 7-l1 | `default` | 60,258 | 0.83x | 157.3 | 55.3 | 13539 | 6.81 / 11.33 | 160 | 0 |

Row compare vs run 1 (AGENT.md evaluation tolerance):

- 2-l2: 160/160 pairs bitwise, passed True, only_a 0 only_b 0
- 3-l4: 160/160 pairs bitwise, passed True, only_a 0 only_b 0
- 4-l4b250: 160/160 pairs bitwise, passed True, only_a 0 only_b 0
- 5-l4b150: 160/160 pairs bitwise, passed True, only_a 0 only_b 0
- 6-l4lock: 160/160 pairs bitwise, passed True, only_a 0 only_b 0
- 7-l1: 160/160 pairs bitwise, passed True, only_a 0 only_b 0
