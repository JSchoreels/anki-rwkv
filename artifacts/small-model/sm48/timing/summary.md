# SM48 timing: held-out scoring speed A/B

40 users of the 732 pool (artifacts/small-model/sm48/timing-users.json), SM36 w8+s8 + w8+s4c8n = 4,511,064 review-evals per run; rate = the scorer's own counter (loop only); process = wall incl. startup; GPU util = nvidia-smi mean over the process.

| Arm | Flags | Runs | review-evals/s (runs) | vs old | process s | GPU util % | GPU mem max MiB |
|---|---|---:|---|---:|---|---|---|
| old | `--legacy-metrics` | 2 | 46,144 (46,118, 46,169) | 1.00x | 102.0, 100.9 | 44.9, 45.9 | 13106.0, 13121.0 |
| arr | `default` | 2 | 61,430 (61,438, 61,421) | 1.33x | 76.6, 76.6 | 53.8, 55.6 | 13121.0, 13123.0 |
| l2 | `--lanes 2` | 2 | 66,484 (66,335, 66,632) | 1.44x | 70.9, 70.6 | 60.2, 61.7 | 14518.0, 14534.0 |
| l4 | `--lanes 4` | 2 | 72,074 (72,183, 71,964) | 1.56x | 65.5, 65.6 | 65.8, 67.3 | 14370.0, 14502.0 |

Row compare vs run 1-old (AGENT.md evaluation tolerance; bitwise = every compared value equal):

- run 2-arr: 80/80 pairs bitwise, passed True; error rows 0
- run 3-l2: 80/80 pairs bitwise, passed True; error rows 0
- run 4-l4: 80/80 pairs bitwise, passed True; error rows 0
- run 5-l4: 80/80 pairs bitwise, passed True; error rows 0
- run 6-l2: 80/80 pairs bitwise, passed True; error rows 0
- run 7-arr: 80/80 pairs bitwise, passed True; error rows 0
- run 8-old: 80/80 pairs bitwise, passed True; error rows 0
