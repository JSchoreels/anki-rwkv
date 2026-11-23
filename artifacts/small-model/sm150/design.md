# Rank-1 int8 states (SM150): design

Pre-registered Mon 11-23, before any scoring of the new labels (R1, R12) on the 732 set or the long-card users. Branch `sm150`, worktree `C:/Users/jonat/workspace/wt/sm150`, cut from `origin/sm148` (cea0f931, the low-rank int8 states (SM148) with the main-session decision). No training, no source change: the codec (`scripts/state_lowrank.py`, label grammar `lr<spec>q8x8`) already parses `lr1q8x8` and `lr1.16.2q8x8`, the CUDA kernel handles any rank up to 7 (m = rank + 1 <= 8). Everything else is a narrow copy of the low-rank int8 states (SM148) tree with new labels and new rules.

## Question

With the low-rank int8 states (SM148), rank 2 (Q2, `w8+lr2q8x8`) already passes the user's new rules (11-23): Small two-head about +0.26% / +0.24% and Tiny two-head about +0.77% / +0.81% against Big two-head (SM91s2) at int8, Big two-head +0.080% / +0.109% and Big Instant-Only (SM99) +0.118% against their own int8. **Question: does rank 1 pass them too on the real evaluation scale, and if not, does rank 1 below 16 answers then rank 2 (R12)?** A CPU probe on 8 users (SM147) put rank 1 at +0.16% to +0.42% against int8.

## Labels (`labels.sh`) and bytes (`sm150_bytes.py` -> `sm150-bytes.json`)

| label | variant | state | scored here |
|---|---|---|---|
| REF | `w8+s8hx8` | int8 cards and notes | SM148's rows are reused everywhere; rescored on the 3 smoke users only (integrity) |
| Q4 | `w8+lr4q8x8` | rank 4, int8 factors | SM148's rows, comparison only |
| Q2 | `w8+lr2q8x8` | rank 2, int8 factors | SM148's rows, comparison only |
| R12 | `w8+lr1.16.2q8x8` | rank 1 below 16 answers, rank 2 from 16 | **new** |
| R1 | `w8+lr1q8x8` | rank 1 everywhere | **new** |

(SM148's rank 2 then 4 label (T24) is not a candidate any more: Q2 beats it in bytes and the new rules do not need it.) Proof-only labels: F1 (`w8+lr1fx8`, float32 factors, int8 shifts), F1X0 (`w8+lr1fx0`, float32 factors and shifts) and LR1 (`w8+lr1`, SM147's float64 SVD rank-1 truncation, CPU only).

Bytes per answered card / note (the fork's compact layout of SM148, `sm148_bytes.py`; the stored-rank byte included, so one byte more than the plain factor count of 1,651 / 1,107 for Big, 529 / 307 for Small and 307 / 307 for Tiny), means over the 723 users' answer counts, and the total vs int8 rows:

| label | Big card / note (B) | Small card / note (B) | Tiny card / note (B) | bytes vs int8: Big / Small / Tiny |
|---|---:|---:|---:|---|
| REF (int8 rows) | 13,891 / 9,267 | 2,545 / 1,315 | 1,315 / 1,315 | 0 |
| Q4 | 3,544 / 2,382 | 1,198 / 645 | 642 / 645 | -74.4% / -52.3% / -51.1% |
| Q2 | 2,388 / 1,601 | 790 / 439 | 438 / 439 | -82.8% / -68.2% / -66.7% |
| R12 | 1,717 / 1,166 | 553 / 323 | 320 / 323 | -87.6% / -77.4% / -75.6% |
| R1 | 1,652 / 1,108 | 530 / 308 | 308 / 308 | -88.1% / -78.4% / -76.6% |

Fully stored rank-1 entity (answers >= 2): Big 1,652 / 1,108, Small 530 / 308, Tiny 308 / 308. The byte order R1 < R12 < Q2 < Q4 < REF holds for every size (checked in `sm150_bytes.py`).

## Rules (pre-registered, `sm150_report.py`; selftest passes)

Statistic: % = 100 x (sum of the label's LogLoss / sum of the reference's LogLoss - 1), review-weighted, paired user bootstrap, 2,000 draws, seed 0, 95% CI (`showtime_report.compare`, as every 732 read-out), on the benchmark LogLoss of the 722 rule users of the 732 set. Heads: two-head models curve and instant, Instant-Only models instant. All scoring on show-time inputs, leak-free, as SM148.

- **Big (two-head (SM91s2) and Instant-Only (SM99)):** the label vs the same model at int8 (REF): CI upper bound <= **+0.15%** per head.
- **Small and Tiny (user 11-23): TOTAL cost vs the matching Big model at int8.** Matching: Small two-head (SM118) and six-layer Tiny two-head (SM130) vs Big two-head (SM91s2), both heads; Small Instant-Only (SM113) and six-layer Tiny Instant-Only (SM130) vs Big Instant-Only (SM99), instant head only. CI upper bound per head: Small <= **+0.5%**, Tiny <= **+1.0%**. The model's own cost vs its own int8 is reported, not ruled.
- **Conditions:** every row finite (GPU and CPU); at least **715 users** scored on the GPU for the label, its REF and, for Small / Tiny, the matching Big REF. SM148's GPU scorer could not fit Big's 5 largest users (424k to 532k reviews, out of memory at cap 0.92), so Big is expected at 718 users and the matching comparison for Small and Tiny at 718 too; below 715 the report is `pending` (exit 2) and the main session decides.
- **Go per size:** the label with the smallest state bytes that passes for BOTH models of the size, among REF, Q4, Q2 (SM148's rows), R12 and R1 (byte order R1 < R12 < Q2 < Q4 < REF); REF (int8) if none passes. Q4 and Q2 take part with SM148's rows, so the go is decided on the same users and the same rule for all five labels. One verdict per size, no strict-rule variant.
- **Warnings (reported, never a verdict):** (a) long cards, reviews whose card had 128 or more earlier answers, on the 50 long-card users (CPU, Small and Tiny only): the label vs its own int8, point and CI upper bound per head, flagged above +0.5% (point) or +1.0% (upper bound), the old long-card tolerance; (b) per-review and per-card |dR| = |p(label) - p(REF)| on the same review: the share of cards with a review |dR| > 0.10, flagged above 1%, and the per-review quantile tables (aggregates only); (c) cost by answer-count bucket (`--prior-buckets`, edges 0, 1, 3, 5, 9, 16, 32, 128, 300, 1000) on the GPU rows; (d) review evaluations per second per label (GPU and CPU rows), quoted with reviews/s; (e) the starting point of Small and Tiny at int8 vs Big at int8.
- **Big long-card users are not run.** In SM148, 13 of the 48 Big long-card users (87k to 151k reviews) were killed by the 5.9 GB per-process memory guard on every attempt (the padded card by time tensors of the scorer's forward pass: a probe showed the peak is a single label's forward pass), and the long-card read-out is only a warning now, so the Big long-card and Big per-card read-outs stay out of SM150 (SM148's Big rank 4 and rank 2 read-outs remain the reference). The warnings are therefore Small and Tiny only.

## Reused rows and the integrity check

REF, Q4 and Q2 come from SM148's committed rows (`../sm148/gpu/sm148-rows-732.jsonl`, `../sm148/cpu/sm148-longcard-rows.jsonl`); the report merges them with this run's rows by label. Integrity: before anything else, REF is rescored on SM148's 3 smoke users (2388, 4676, 1866) on the CPU for all six models and must equal SM148's committed smoke rows to **1e-6** in benchmark LogLoss per user and head with equal review counts (`sm150_check.py cpu`, check K3); if not, **stop and report**. The same smoke ticket on the GPU rescored REF too (REF GPU vs CPU within 1e-5 per user and head, as SM148).

For the per-review |dR| read-out, the new labels' dumps (`work/sm150-dump`) are compared with SM148's REF dumps (`work/sm148-dump`, same code, same inputs, REF stable to 1e-6), copied next to them for the aggregate step; no REF rescoring on the long-card users.

## Proofs (CPU, before any GPU ticket; `cpu-proof.sh`, `sm150_check.py cpu` -> `check-cpu-sm150.json`)

- K1 bytes (`sm150_bytes.py`): REF's entity bytes equal SM145's int8 sizes; the byte order R1 < R12 < Q2 < Q4 < REF.
- K2 the SM147 and SM148 selftests of `scripts/state_lowrank.py` on the CPU (unchanged codec).
- K3 scorer rows on the 3 smoke users x 6 models x REF, R1, R12, F1, F1X0, LR1 (CPU): every row finite; **REF equals SM148's smoke rows to 1e-6**; **F1X0 (rank 1 through the factored solve, float factors and shifts) equals SM147's rank-1 truncation (LR1, float64 SVD) within 2e-5 per user and head**; R1 and R12 differ from REF on every user and model, R12 differs from R1 on at least one user per model.
- K5 the dump: a file per (user, model, label) for REF, R1, R12; dumped reviews per bucket equal the rows' all-scope counts; equal review sets across labels; |dR| R1 vs REF above 0; `sm150_dr.py` aggregates the smoke dumps.
- GPU smoke ticket (first ticket, STOP on failure): `sm150_check.py gpu-selftest` (Jacobi kernel, fused encode kernel; now also rank 1 and rank 1-then-2 against the CPU loop) + the 3 smoke users x 6 models x REF, R1, R12, F1 vs the CPU rows: REF within 1e-5; the low-rank labels by the discrete-codec rule (per user |delta| <= 3e-4, review-weighted mean signed delta <= 1e-4).

## Run plan (`launch-sm150.sh`, one detached launcher; DRYRUN=1 for a dry run)

| step | where | expected |
|---|---|---|
| CPU proofs | CPU | about 10 min |
| GPU smoke ticket | queue, cap 0.8 | about 5 min |
| scoring Tiny / Small / Big: 723 users x 2 models x {R12, R1}, `--prior-buckets` | 3 tickets, cap 0.8 (retry at 0.92 only after a pass without progress) | about 40 / 60 / 100 min (SM148: 69 / 100 / 167 min for 4 labels, REF is the cheap one), about 3.5 GPU hours in all |
| long-card users: 50 users x {Tiny, Small} x 2 models x {R12, R1}, with the dump | CPU, 2 processes (3 after the GPU chain), memory guard 5.9 GB, one fresh process per (size, user) | about 2 to 3 h (SM148 Tiny about 1 min per user and label) |
| per-review read-out, report | CPU | a few minutes |

Memory guard: SM148's fixed `memguard.ps1` with `-Match "small_model_heldout.py&--device cpu&sm150"` (a single group naming only the CPU scorer command words; GPU tickets never match). Quiet windows (`quiet.json`) are respected by both chains as in SM148. Resources: CPU processes < 6 GB, GPU cap 0.8 (0.92 only for retry passes).

## Expectation (written before scoring, not a rule)

From SM148's rows and the 8-user probe: rank 2 costs +0.04% to +0.12% against own int8 and passes with room (Small and Tiny far below their ceilings against Big). Rank 1 should cost roughly +0.15% to +0.4% against own int8: likely too much for Big's +0.15% (maybe R12 passes), probably fine for Small (+0.26% at rank 2 against a +0.5% ceiling) and Tiny (+0.77% / +0.81% at rank 2 against +1.0%; Tiny is the tightest one). The cost sits in cards and notes with many answers (SM147), which R12 protects from 16 answers on.

## Files

`labels.sh`, `models.sh` (the six release-candidate models), `users-smoke.json`, `users-longcard.json`, `memguard.ps1` (SM148's fixed copy); `sm150_bytes.py` -> `sm150-bytes.json`; `sm150_check.py`; `sm150_report.py` (smoke, rules, selftest, pins); `sm150_dr.py` (SM148's with the SM150 labels and a `--sizes` filter); `cpu-proof.sh`; `launch-sm150.sh`; `HANDOFF.md`. Outputs: `cpu/smoke-cpu-rows.jsonl`, `check-cpu-sm150.json`, `gpu/` (tickets, smoke, rows of the 732 set), `cpu/longcard/<size>/` (one file per user), `cpu/sm150-longcard-rows.jsonl`, `sm150-dr.json`, `sm150-result.{json,md}`. Not committed: `work/sm150-dump*` (per-review values).
