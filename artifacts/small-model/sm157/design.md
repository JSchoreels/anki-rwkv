# Output-aware low-rank truncation (SM157): design

Written Tue 11-24, before any probe row of a weighted label was scored or read. Branch `sm157`, worktree `C:/Users/jonat/workspace/wt/sm157`, cut from `origin/sm152` (19360679, the launch of smaller shift vectors and factors on T24 states (SM152)). CPU only, no GPU job, no training. Idea 2 of `wt/ideas-1123/artifacts/ideas-1123/OPPORTUNITIES.md`.

## Question

The card and note WKV state S of a head (K x K; K = 32 for Big, 16 for Small and Tiny) is only used through products on its key side: the read `S r`, the removal term `S kd` and the decay on the key axis (`S diag(w)`). The rank-r codec of the low-rank int8 states test (SM148) keeps the Frobenius-optimal truncation, which treats every key direction as equally important. Does a truncation that minimises the expected read error, `E||(S - S_hat) x||^2 = ||(S - S_hat) M^(1/2)||_F^2` with `M = E[x x^T]`, cut the cost of rank 2 against int8 rows by at least a third, at unchanged stored bytes?

## Method (`scripts/state_lowrank.py`, SM157 block)

- Weighted Eckart-Young: the best rank-r fit in the M metric is the rank-r truncation of `S M^(1/2)`, multiplied back by `M^(-1/2)`. `S M^(1/2)` has the column space of S, so the factored solve of SM148 (`lowrank_factors`, basis span(P) of the previous left factors and this write's v) applies unchanged to `S M^(1/2)`.
- Stored format unchanged: left factors (value side) and right factors, int8 per vector with a float16 scale, as SM148. The right factors are quantized in the weighted space (the rows of `sqrt(sigma) V^T` of `S M^(1/2)`) and decoded by `x M^(-1/2)`, so the quantization error is also spread in the M metric. Stored state used by the next step: `L (R_q M^(-1/2))`.
- Label suffix `w<kind>` on any SM148 codec label with a rank tier: `lr2q8x8wr`, `lr2.16.4q8x8wg`, and so on. Without the suffix every code path is the old one (expressions unchanged; the CUDA fused kernel is used only for plain labels).
- Three weight kinds, per (model, stream, layer, head):
  - (a) `r`: `M = R = E[r r^T]`.
  - (b) `rk`: `M = R + beta E[kd kd^T]`, with `beta = tr R / tr E[kd kd^T]` (equal trace, fixed a priori, no tuning).
  - (c) `g`: the discounted observability Gramian `M = R + D M D`, `D = diag(d)`, d = the head's mean per-channel decay `E[w]`. With a diagonal D the Stein equation is solved elementwise: `M_ij = R_ij / (1 - d_i d_j)`. The decays are clamped to at most `1 - 1/64` (a channel with d close to 1 would otherwise get an unbounded weight; some channels have a mean w of 0.9999).
- Every M is then normalised to a mean eigenvalue of 1 (`M / (tr M / K)`), given a ridge of `1e-3 I`, and rounded to float16 (the shipped precision); `M^(1/2)` and `M^(-1/2)` come from its eigh in float64, used in float32. Ridge and clamp are fixed here and not tuned.

## Fit (`sm157_fit.py`, training users only)

- Users: 100 training users of the 8k split with id >= 5000 (`users-fit.json`, `random.Random(157).sample` of the 469 such users with 4,000 to 8,000 reviews). Users with id >= 5000 in `train_users` are training users of every release model, the fork-architecture Big models included. The script checks that none is a validation, quick-validation or final-test user and none is in the 732, 2,225 or 1,500 pools. Deviation from "a few hundred": the smallest training user has about 4,900 reviews and the median 30,838; 100 users (627,234 reviews) keep the fit at about 25 minutes of wall time on two processes. A K x K second moment from about 600,000 writes per layer is well determined; the split-half check below measures it.
- Split halves: half A (every other user of the sorted list) and half B are fitted separately and summed. The report gives the relative difference of M between halves and the overlap of their top-2 eigenvector subspaces.
- Forward passes of the scorer's CPU model (w8 weights, show-time inputs `e2s-query`, leak-free `--deleted-cards neighbour`: the scorer's `LeakBatches`), with the stored state under `w8+lrRq8x8` (REF's int8 rows through the SM148 step loop, the reference the costs are measured against). At every write of every card and note layer, `record_stats` accumulates `r r^T`, `kd kd^T`, `w` and `log w` per head. The training users have no rows in the label DB, so the scorer itself cannot run on them; the fit script only runs `model.get_loss` and discards the output.
- Output: `sm157-stats.json` (sums per model and "stream,layer", aggregates only, no ids), `sm157-stats-a.json`, `sm157-stats-b.json`. The weighted labels read the file named by `SM157_STATS`; its sha256 is recorded in every weighted row.

## Labels (`labels.sh`)

| Name | Label | Meaning |
|---|---|---|
| REF | `w8+s8hx8` | int8 card and note rows (the reference) |
| Q2 | `w8+lr2q8x8` | rank 2 everywhere, int8 factors, int8 shift vectors (plain) |
| T24 | `w8+lr2.16.4q8x8` | the shipped format, rank 2 below 16 answers then rank 4 (plain) |
| Q2R, Q2RK, Q2G | `w8+lr2q8x8w{r,rk,g}` | Q2 with weighted truncation (a), (b), (c) |
| T24R, T24RK, T24G | `w8+lr2.16.4q8x8w{r,rk,g}` | T24 with weighted truncation (a), (b), (c) |

Stored bytes are those of Q2 and T24 in SM148's table (unchanged). M ships with the model: one K x K matrix per layer and head per kind; the report gives its size at float16 (full and symmetric storage). Big: 5 layers x 4 heads x 32 x 32 x 2 bytes = 40,960 bytes full (21,120 symmetric); Small: 3 x 4 x 16 x 16 x 2 = 6,144 (3,264 symmetric); six-layer Tiny: 2 x 4 x 16 x 16 x 2 = 4,096 (2,176 symmetric). If the fork ships `M^(1/2)` and `M^(-1/2)` instead of computing them at load, double these.

## Proofs (CPU, `cpu-proof.sh`, `sm157_check.py cpu` -> `check-cpu-sm157.json`)

- K1: the selftests of SM147, SM148 and SM152 unchanged, plus the SM157 identities (the weighted factored fit equals the float64 weighted optimum; the weighted error is never above plain truncation's and the plain error never below; young states with n <= r answers exact with weights; M = I gives the plain fit; the Gramian solves `M = R + D M D`; grammar).
- K2: REF, Q2 and T24 rescored on the 3 smoke users (2388, 4676, 1866) x the 6 models equal SM148's committed CPU smoke rows in every metric and descriptor field (identical JSON values, not a tolerance).
- K3: a pipeline check on user 2388 x the 6 models x Q2R, Q2RK, T24G with throwaway statistics from 2 training users (`users-fit-test.json`): rows are finite, name the weight kind, and differ from the plain label. Only these three properties are checked; the values are not read or reported.
- Report only: the per-write cost of the codec with and without weights (microbenchmark, one thread).

## Run (`run-sm157.sh`, detached)

1. Fit, halves A and B, two processes (two-head models; Instant-Only models), then merge.
2. Probe scoring: the 8 users of the low-rank structure probe (SM147) (`users-probe.json`: 2388, 4676, 1866, 2883, 3882, 1215, 4407, 966; validation users of the 732 set, never used for the fit) x the six release models (Big two-head (SM91s2), Big Instant-Only (SM99), Small two-head (SM118), Small Instant-Only (SM113), six-layer Tiny two-head (SM130), six-layer Tiny Instant-Only (SM130)) x the 9 labels, CPU, show-time inputs, leak-free, `--nonfinite-rows`, `--prior-buckets`. Two processes, memory guard 5.9 GB each.
3. `sm157_report.py report` -> `sm157-result.{json,md}`.

## Rule (pre-registered, `sm157_report.py`; its selftest passes)

- Cost of a label: `100 x (sum of benchmark LogLoss x size over the 8 users / the same for REF - 1)`, review-weighted, paired user bootstrap 2,000 draws, seed 0, 95% CI. Cells: (model, head); curve and instant for the three two-head models, instant for the three Instant-Only models: 9 cells.
- For a weight kind v, on Q2: `reduction = 1 - cost(Q2 weighted v) / cost(Q2 plain)`, computed for the cells whose plain cost is at least 0.01% (informative cells).
- PASS(v) when there are at least 5 informative cells, the median reduction over them is at least one third, and no cell of the 9 is worse: the direct paired % of Q2 weighted v against Q2 plain is never above +0.01%.
- The pointer passes when at least one v passes. The GPU follow-up then scores the passing v with the largest median reduction (ties: r, then rk, then g). Fewer than 5 informative cells: inconclusive, no GPU run.
- T24 is reported with the same numbers, outside the rule. Eight users and three kinds make this a pointer, not a gate: a pass only buys the 732-set GPU run, judged there by the 11-23 budgets.

## GPU follow-up (prepared, NOT started; `launch-sm157-gpu.sh`)

In the pattern of the rank-1 int8 states test (SM150) and SM152: `scripts/r01_queue.py` tickets (owner sm157, cap 0.8, PYTORCH_JIT=0), a GPU smoke on the 3 smoke users against the CPU rows first, then one ticket per size on the 732 set (723 users; Big 718) with the two weighted labels of the chosen kind (Q2 and T24); REF, Q2 and T24 rows are SM148's. About 3.5 GPU hours (SM150's measured 2-label passes: Tiny 50, Small 75, Big 130 minutes, plus the smoke). The weighted codec runs the torch path on the GPU (the fused kernel is for plain labels only), so the Big pass may take longer than SM150's.

## Speed

- Extra cost per write, as implemented here: `S M^(1/2)` (K^3 multiply-adds per head) and `R_q M^(-1/2)` (r K^2). In the fork's factored form the product is `(Q^T S) M^(1/2)`, m K^2 with m = r + 1, plus the decode r K^2: for Big at rank 2, 3,072 + 2,048 multiply-adds per head and write, against the factored solve's existing products of the same order. The microbenchmark in `check-cpu-sm157.json` gives the measured CPU time per write.

## Expectation (not a rule)

The receptance statistics of the 2-user pipeline fit are strongly anisotropic (the top 2 of K eigen-directions of R carry about 60% to 80% of its trace), so a real gain is plausible; the risk is that the out group norm makes the output scale-invariant per head and that errors in weakly read directions still reach the output through the decay and the removal term.

## Files

`labels.sh`, `models.sh` (copy of SM152's), `users-fit.json`, `users-fit-a.json`, `users-fit-b.json`, `users-fit-test.json`, `users-probe.json`, `users-smoke.json`, `users-pipe.json`, `sm157_fit.py`, `sm157_check.py`, `sm157_report.py`, `cpu-proof.sh`, `run-sm157.sh`, `launch-sm157-gpu.sh`, `memguard.ps1`; outputs `check-cpu-sm157.json`, `sm157-stats*.json`, `cpu/`, `sm157-result.{json,md}`, `RESULT.md`, `HANDOFF.md`.
