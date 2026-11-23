# Low-rank int8 card and note states, post-training (SM148): design

Pre-registered Mon 11-23 (about 02:00), before any scoring of the new labels on the 732 set or the long-card users (the only rows looked at before this file: feasibility timings on 3 smoke users and 8 timing users of the 732 set, used for speed only, and the CPU proof rows of the 3 smoke users). Branch `sm148`, worktree `C:/Users/jonat/workspace/wt/sm148`, cut from `origin/sm147` (e4f5d238, the result of the low-rank structure of the card and note states (SM147)). No training: the six release-candidate models are scored as they are. The setup mirrors the int3 young tier on the early switch (SM145) (scorer flags, users, rules, launcher, memory guard, per-card read-out).

## Question

The low-rank structure of the card and note states (SM147) (CPU pointer, 8 users, float32 factors) found rank 4 per head free within noise on all six release-candidate models (-0.01% to +0.01%, at most +0.09% in the 32-127 bucket), rank 2 about free for Small and +0.04% to +0.08% for Big and the six-layer Tiny instant heads, rank 1 +0.16% to +0.42%. **Question: on the real evaluation scale, does a post-training low-rank state with int8 factors pass our rules, and which rank per size?**

## The codec (`scripts/state_lowrank.py`, variants `w8+lr<spec><q>x8`)

- **Where:** the same placement as every int codec since SM47: the answer's own output uses the float update S' = S G + v k^T (G = diag(w) - kd (a kd)^T, the RWKV-7 step); the stored state, read by the entity's next review, is the codec's reconstruction. n = the entity's update count, this write included (the non-skipped steps, counted as the tier codec of SM141 counts them).
- **Truncation, exact and cheap:** the stored state is a rank-r product L R, so col(S') lies in span(L, v) (S G keeps the column space of S, the new term adds v). With Q an orthonormal basis of span([L, v]) (m = r + 1 columns; column-normalised Gram, eigen-decomposition, directions with a Gram eigenvalue below 1e-5 x the largest dropped), S' = Q Q^T S', so the best rank-r approximation of S' is Q U_r Sigma_r V_r^T from the small m x K matrix B = Q^T S' (eigen-decomposition of B B^T). No K x K SVD. Float32 data; the CPU uses LAPACK `eigh`, the GPU a fused CUDA kernel (one warp per matrix, cyclic Jacobi for the m x m eigen problems, `scripts/state_lowrank_cuda.py`). This equals the SVD truncation up to float32 rounding (proof K2, G2, K3 F4X0).
- **Exact rank for young states:** a state with n answers has rank at most n, so the codec keeps rank min(n, r): exact for n <= r, and such an entity stores only n factor pairs.
- **Factors:** component i is split evenly, left_i = u_i sqrt(sigma_i), right_i = sqrt(sigma_i) v_i^T. **int8 quantization:** symmetric, one float16 scale per factor vector (per column of L and per row of R: scale = max |x| / 127 rounded to float16, codes round-half-even in [-127, 127]), the same arithmetic as REF's row scales. Per-vector scales because the components differ in size by up to a factor 100 (one scale per factor matrix, as in the bytes estimate of SM147, would round the small components to almost nothing); the cost is 4 bytes per component instead of 4 per head. The stored state is the product of the decoded factors (float32), and its column space is span(L), which the next write uses.
- **Shift vectors** stay REF's int8 (one float16 scale per vector, `x8`), card and note. A non-finite matrix is stored as it is (the scorer's non-finite handling sees it).
- Tier syntax: `lr<kind0>[.<N1>.<kind1>]...`, kind = rank r or `R` (REF's int8 rows, last tier only), kind_i for answer counts from N_i. `q8` = int8 factors, `f` = float32 factors (proof); `x0` = float32 shifts (proof only).

## Labels (`labels.sh`) and bytes (`sm148_bytes.py` -> `sm148-bytes.json`)

Bytes in the fork's compact layout of SM145 (`sm145_bytes.entity_bytes`): header, per-layer lengths and int8 shift vectors unchanged; one head's WKV at rank c = 2 c K int8 codes + 2 c float16 scales (REF: K rows of K codes + a float16 scale); plus 1 byte per entity for the stored rank. Mean per answered card / note with answers on the 723 users of the 732 set (end-of-history answer counts of `../sm145/sm145-dist.json`: 9.8% / 9.4% have 1 answer, 13.1% / 12.8% have 2, 12.6% / 12.0% have 3, 8.0% / 10.7% have 16 or more).

| label | variant | state | Big card / note (B) | Small card / note (B) | Tiny card / note (B) | bytes vs int8: Big / Small / Tiny |
|---|---|---|---:|---:|---:|---|
| REF | `w8+s8hx8` | int8 rows (the release-candidate format, reference of every rule) | 13,891 / 9,267 | 2,545 / 1,315 | 1,315 / 1,315 | 0 |
| Q4 | `w8+lr4q8x8` | rank 4, int8 factors | 3,544 / 2,382 | 1,198 / 645 | 642 / 645 | -74.4% / -52.3% / -51.1% |
| T24 | `w8+lr2.16.4q8x8` | rank 2 below 16 answers, rank 4 from 16 | 2,519 / 1,717 | 836 / 469 | 461 / 469 | -81.7% / -66.3% / -64.7% |
| Q2 | `w8+lr2q8x8` | rank 2, int8 factors | 2,388 / 1,601 | 790 / 439 | 438 / 439 | -82.8% / -68.2% / -66.7% |

For scale: the current best tiered int formats save -43.5% (Big), -38.6% (Small) and -42.7% (Tiny) against int8 rows (the int3 young tier on the early switch (SM145) and the aggressive state tiers for Tiny (SM146) go picks).

Why these three: Q4 is the label SM147 found free everywhere; Q2 is the smallest form worth testing (SM147: about free for Small, +0.04% to +0.08% for Big and the six-layer Tiny instant heads, so likely over the +0.05% ceiling there but maybe not for Small); T24 is the tiered form in between: SM147's rank-2 cost sits in the cards and notes with 32 or more answers (within +0.1% below 32 answers, +0.1% to +0.4% at 32-127), so rank 2 for the young states and rank 4 from 16 answers (the repo's usual switch point, SM144) keeps most of Q2's saving (it costs 1.1% to 2.0% of the int8 bytes over Q2) while protecting the long-lived states. Not scored (GPU time, below): rank 4 with int8 rows from 128 answers (`lr4.128.Rq8x8`, -74.4% for Big, almost the same bytes as Q4 because 0.04% of cards and 0.08% of notes reach 128 answers), the fallback if Q4 failed only the long-card rule; rank 1 (-88.1% Big; SM147 +0.16% to +0.42%, clearly over the ceiling without a cool-down); int4 factors (not asked). Proof-only labels: U8 (`w8+lrRq8x8`: the SM148 code path with every write in REF's int8 rows, must equal REF), F4 (`w8+lr4fx8`: rank 4 with float32 factors, the truncation without the factor rounding), F4X0 (`w8+lr4fx0`: float32 factors and shifts) and LR4 (SM147's `w8+lr4`, float64 SVD, CPU only; F4X0 must equal it to float rounding).

## Feasibility (measured 11-23, before the design was fixed)

- **CPU:** a full float64 SVD per head per write (SM147's probe) costs about 100 microseconds per 32 x 32 matrix (4 threads, batched); the factored solve about 3 to 5 microseconds. In the scorer the factored codec costs Big 3.79 s against REF 3.41 s on the smallest smoke user (SM147's SVD: 9.55 s), Small 2.55 s against 2.50 s.
- **GPU:** the per-time-step torch loop is launch-bound (about 90 small kernels per step for the codec in plain torch, plus host syncs: cusolver's batched `eigh` checks its result on the host and refuses batches of about 32k matrices). Fixes: the cyclic Jacobi kernel for the small eigen problems and then one fused encode kernel per step (one warp per matrix, float32): per step 0.42 ms at 256 rows (the float step alone 0.19 ms) and 2.1 ms at 4,096 rows with K = 32. GPU vs CPU: float factors agree to 1e-6 relative; int8 factors differ by rounding flips (discrete-codec rule, below).
- **GPU time on the 732 set:** timed on 8 users of the 732 set at the review-count quantiles 10 / 30 / 50 / 70 / 85 / 95 / 99 / 100% (4396, 722, 1286, 3208, 3873, 3829, 276, 3735; REF and Q4 of Big two-head (SM91s2), Small two-head (SM118) and six-layer Tiny two-head (SM130)), weighted to the quantile mass: a low-rank label takes about 2.25 s per user for Big (REF 0.87 s), 1.15 s for Small (REF 0.46 s) and 0.83 s for Tiny (REF 0.20 s), so about 27 / 14 / 10 minutes per label and model over 723 users. Three low-rank labels plus REF, two models per size: about 185 min (Big) + 95 min (Small) + 65 min (Tiny) = **about 5.7 GPU hours** (range about 4.5 to 7.5 hours, the 8-user estimate is rough). That is inside the 4 to 6 hour budget at its upper edge, so **the full 732 set (723 users, 722 rule users) is kept**, no reduced user set. A fourth low-rank label would add about 1.7 hours, so the design stays at three.
- The largest user (3735, 531k reviews) ran out of memory at cap 0.8 for Big (REF too, as in SM144 / SM145); the retry pass at cap 0.92 (memory only, same numerics) of SM145's launcher covers it.

## Rules (pre-registered, `sm148_report.py`; selftest passes)

Statistic: % = 100 x (sum of the label's LogLoss / sum of the reference's LogLoss - 1), review-weighted; paired user bootstrap, 2,000 draws, seed 0, 95% CI (`showtime_report.compare`, as every 732 read-out). Heads: two-head models curve and instant, Instant-Only models instant.

- **R1 finite:** no non-finite row and no non-finite LogLoss, on the 732 set (GPU) and on the long-card users (CPU).
- **R2 dataset-wide:** benchmark LogLoss on the 722 rule users of the 732 set vs the same model's int8 rows (REF `w8+s8hx8`), CI upper bound <= **+0.05%** per head.
- **R2T, the user's Tiny rule (the six-layer Tiny models only, user 11-22):** the Tiny label vs the matching Small model at int8 (`w8+s8hx8`, scored in this run: Small two-head (SM118) for six-layer Tiny two-head (SM130), Small Instant-Only (SM113) for six-layer Tiny Instant-Only (SM130)), paired by user on the same 722 rule users, CI upper bound <= **+1.00%** per head.
- **R3 long cards:** on the 50 long-card users (CPU, every label scored directly), reviews whose card had 128 or more earlier answers: point estimate <= **+0.5%** and CI upper bound <= **+1.0%** per head vs REF (the user's 11-22 long-card tolerance).
- **G per-card guard:** on the same 50 long-card users, the share of cards whose worst review has |dR| > 0.10 (dR = p(label) - p(REF) on the same review; every predicted review, `sm148_dr.py` = SM145's `sm145_dr.py` read-out) is at most **1%** per head.
- **R5 coverage:** at least 720 of the 722 rule users for REF and every label on the GPU, at least 48 of the 50 long-card users on the CPU, and a complete per-review read-out (at least 48 users per model, label and head); otherwise the report is `pending` (exit 2) and the main session decides.
- **Go per size:** a label passes for a size when it passes R1, R2, R3 and G for both models of the size; for the six-layer Tiny models R2T replaces R2 (as the aggressive state tiers for Tiny (SM146) used it). The size takes the passing label with the smallest state bytes (Q2 < T24 < Q4); if none passes, int8 stays (REF). Both Tiny verdicts are reported: the go under R2T and the strict go under R2 (+0.05% vs Tiny's own int8).

### Reported, no verdict

- The cost of every label vs REF per answer-count bucket (`--prior-buckets`, edges 0, 1, 3, 5, 9, 16, 32, 128, 300, 1000: buckets 1-2, 3-4, 5-8, 9-15, 16-31, 32-127, 128+ answers before the review), on the GPU rows (all users) and on the CPU long-card rows; the all-scope total; Tiny int8 vs Small int8 (the starting point of R2T).
- The per-review and per-card |dR| tables (`sm148-dr.json`): per review by the card's earlier answers (n, mean |dR|, P50 / P90 / P99 / P99.9 / max, share above 0.05 and 0.10), per card (the card's worst review), per user. Aggregates only, no user or card ids, no review rows.
- **Speed cost per write:** review evaluations per second (scorer model time) of every label vs REF, on the GPU rows and on the CPU long-card rows; the CPU per-write microbenchmark of `sm148_check.py` (codec part of one head write, 1 thread, batched and single-entity). The CPU ratio is the closer proxy of what the fork would pay at every write (the fork runs on the CPU and would keep the factors: the write is L (R G) + v k^T in factor form, the m x m solves and the factor rounding; O(K m^2) per head instead of K^2 for int8 rows).
- Integrity: REF rows vs SM145's committed rows (GPU 732 set, CPU long-card users; SM146 found them bitwise stable); GPU vs CPU on the long-card users (REF, 128+ bucket).

## GPU or CPU per user (as SM145)

The 732 set (723 users; R2, R2T) runs on the GPU; R3 and the per-review read-out run on the CPU on the same 50 long-card users as SM141 / SM144 / SM145 / SM146 (`users-longcard.json`). GPU tickets through the queue (owner sm148, cap 0.8, 0.92 only for a retry pass without progress), `PYTORCH_JIT=0` (the fork model's eager modules, needed for the wrapped reference WKV; the REF path of the fork stays its CUDA kernel). CPU: one fresh process per (size, user), at most 2 at a time while the GPU chain runs (3 after it ended), a user only starts when the free RAM fits, memory guard 5.9 GB per process (`memguard.ps1`), `--dump-scope all` into `work/sm148-dump/` (never committed).

GPU smoke rule (first ticket, STOP on failure): `sm148_check.py gpu-selftest` (G1 the Jacobi kernel vs `torch.linalg.eigh`; G2 `selftest_sm148` on CUDA through the fused kernel; G3 the fused kernel vs the CPU loop, float factors within 1e-5 relative); then 3 smoke users x 6 models x REF, Q4, T24, Q2, U8, F4 vs the CPU proof rows: REF and U8 within 1e-5 per (user, head) (the strict rule of the int8 variants), the low-rank labels by the discrete-codec rule (per user |delta benchmark LogLoss| <= 3e-4, review-weighted mean signed delta <= 1e-4): int8 factor codes flip between devices as the int4 / int3 codes do (the feasibility trials saw at most about 2e-6 on the smoke users).

## Proofs (CPU, before any GPU ticket; `cpu-proof.sh`, `sm148_check.py cpu` -> `check-cpu-sm148.json`)

- K1 bytes: REF's entity bytes equal SM145's int8 sizes; the byte order Q2 < T24 < Q4 < REF holds for every size.
- K2 synthetic, seeded (`state_lowrank.selftest_sm148`, CPU): the factored truncation equals the float64 SVD truncation (relative Frobenius error <= 5e-5; float32 measured about 2e-6 to 9e-6) and keeps the singular values; states with n <= r answers are kept exactly (float factors, relative error about 1e-6, at most n stored components); the int8 factor round trip (codes in [-127, 127], error <= scale / 2, sign symmetric, idempotent); the tier selection; SM147's selftest.
- K3 scorer rows, the 3 smoke users (2388, 4676, 1866) x the 6 models x 8 labels (REF, Q4, T24, Q2, U8, F4, F4X0, LR4), CPU: U8 == REF in every metric field; REF == SM145's committed CPU smoke rows in every metric field; F4X0 (the factored solve with float factors and float shifts) == SM147's LR4 (float64 SVD) within 2e-5 per user and head; Q4, T24, Q2 differ from REF on every (user, model); T24 differs from Q4 on every (user, model) and from Q2 on at least one user per model; every row finite.
- K5 the dump: a file per (user, model, label); the dumped reviews per bucket equal the rows' all-scope counts; equal review sets across labels; |dR| U8 vs REF exactly 0, Q4 vs REF above 0; `sm148_dr.py` aggregates the smoke dumps.

## Run plan (`launch-sm148.sh`, one detached launcher; DRYRUN=1 for a dry run)

| step | where | expected |
|---|---|---|
| CPU proofs (`cpu-proof.sh`) | CPU, before the launch | about 15 min |
| GPU smoke: self-test + 3 users x 6 models x 6 labels | 1 queue ticket | about 5 min |
| scoring Tiny, Small, Big: 723 users x 2 models x 4 labels, `--prior-buckets` | 3 queue tickets (+ short retry passes) | about 65 / 95 / 185 min, about 5.7 GPU hours in all |
| long-card users: 50 users x 3 sizes x 2 models x 4 labels, with the dump | CPU, 2 processes (3 after the GPU chain) | about 8 to 9 h (SM145: 5 labels, about 9 to 10 h) |
| per-review read-out (`sm148_dr.py aggregate`), report | CPU | a few minutes |

## Expectation (written before scoring, not a rule)

From SM147's 8-user pointer: Q4 passes R2 everywhere (its point estimates were within +/-0.01%; with int8 factors on top maybe +0.01% more) and R3 (at most +0.09% at 32-127 answers; the 128+ bucket was noise). Q2 likely fails R2 for Big (+0.04% to +0.08% point estimates) and the six-layer Tiny instant heads under the strict rule, may pass for Small, and passes R2T for Tiny (Tiny starts at +0.47% to +0.57% against Small at int8). T24 should land close to Q2 on the young-card buckets and close to Q4 on the long cards; it may pass for Big where Q2 fails. The per-card guard has no earlier reference for low rank; rank truncation moves predictions more smoothly than int3 / int2 codes, so it should stay far under 1% for Q4.

## Files

- `labels.sh`, `models.sh` (the six release-candidate models), `users-smoke.json`, `users-longcard.json` (copies of SM145's), `memguard.ps1` (SM145's).
- `sm148_bytes.py` -> `sm148-bytes.json`; `sm148_check.py` (CPU proofs, GPU self-test); `sm148_report.py` (smoke, rules, selftest); `sm148_dr.py` (the per-review read-out, SM145's code with the SM148 labels); `cpu-proof.sh`; `launch-sm148.sh`.
- Source: `scripts/state_lowrank.py` (codec), `scripts/state_lowrank_cuda.py` (CUDA kernels), two hooks in `scripts/state_vq.py`, the variant parse and per-label setup in `scripts/small_model_heldout.py` (commit 379e548a).
- Outputs: `cpu/smoke-cpu-rows.jsonl`, `cpu/smoke-dr.json`, `check-cpu-sm148.json`, `gpu/` (tickets, smoke, rows of the 732 set), `cpu/longcard/` (one file per user and size), `cpu/sm148-longcard-rows.jsonl`, `sm148-dr.json`, `sm148-result.{json,md}`. Not committed: `work/sm148-dump/`, `work/sm148-dump-smoke/` (per-review values).

## Proof results

- `check-cpu-sm148.json`: **CPU proofs ALL OK** (16 checks; `cpu-proof.sh` 01:31 to 01:40 Mon 11-23, log `cpu-proof.out`, source commit 379e548a, clean). K1 bytes OK; K2 the synthetic identities (factored truncation vs float64 SVD at most 8.6e-6 relative, young states kept exactly to 9.5e-7, int8 round trip, tiers); K3 on 3 users x 6 models x 8 labels (144 rows, all finite): U8 == REF in every metric field, REF == SM145's committed CPU smoke rows in every metric field, F4X0 == SM147's LR4 to 0.0 at the rows' 6 decimals, Q4 / T24 / Q2 differ from REF on every user and model, T24 differs from Q4 everywhere and from Q2 on at least one user per model; K5 the dump (90 files, counts, equal review sets, |dR| U8 vs REF exactly 0, Q4 vs REF above 0, `sm148_dr.py` aggregates the smoke dumps).
- GPU self-test trial (11-23 before the launch, same code; the launcher reruns it in the smoke ticket): G1 to G3 OK, float factors fused kernel vs CPU loop at most 8.5e-7 relative.
- CPU speed on the smoke rows (scorer model time, review evaluations per second, report only): Big two-head (SM91s2) REF 910, Q4 776 (0.85x), T24 783, Q2 802; Small two-head (SM118) REF 1,430, Q4 1,261 (0.88x); six-layer Tiny two-head (SM130) REF 3,497, Q4 2,824 (0.81x); SM147's float64 SVD truncation (LR4) was 358 / 1,028 / 2,143. Per head write, codec part only (1 thread, 256 entities per call): int8 rows 1.2 / 1.5 microseconds (K = 16 / 32), rank 4 int8 factors 7.3 / 11.6, rank 2 4.9 / 8.6; one entity per call (as the fork writes) is dominated by torch's per-op overhead here (7 vs 72 microseconds) and says little about a Rust implementation.

## Launch notes

- First launch 11-23 01:42: the smoke ticket (001491) ended with exit 127 within 11 s because the memory guard killed the GPU self-test. The guard copied from SM145 (`memguard.ps1`) unrolled a single `-Match` group into its words, so any python.exe whose command line held "sm148" matched, and a CUDA process on Windows shows a working set of 10 to 12 GB (the self-test, and the agent's own diagnostic runs). Fixed in `memguard.ps1` (the group array is built explicitly; checked: the GPU self-test and a `--device cuda` scorer line no longer match, a `--device cpu` scorer line does). The CPU chain had finished 14 Tiny long-card users by then (their rows are kept and skipped on resume; the guard killed no scorer). Launcher stopped with `gpu/STOP`, log kept as `gpu/launch-sm148-launch1.log`, kill log as `cpu/memguard-kill-launch1.txt`; relaunched after the fix. Scripts, rules and labels unchanged.
