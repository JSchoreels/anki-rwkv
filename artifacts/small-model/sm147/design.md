# Low-rank structure of the card and note states (SM147): design

Written Mon 11-23, before any number of the probe was looked at. Branch `sm147`, worktree `C:/Users/jonat/workspace/wt/sm147`, cut from `origin/sm146` (8d0f8e56, the result of the aggressive state tiers for Tiny (SM146)). CPU only, no GPU ticket, no training. A pointer, not a gate: no verdict rules, numbers and a short reading only.

## Why

Outside research ships a state recipe that stores each card / note WKV matrix (one head, K = 32) as rank-2 (card) / rank-1 (note) int8 factors plus int4 token shifts, trained with a short quantization-aware cool-down. Their diagnostic: full int8 without truncation is nearly free and the whole cost comes from the rank truncation. Our state-format work (the int3 young tier (SM145), the aggressive state tiers for Tiny (SM146)) rounds rows of the full matrix and never truncates the rank. Our models differ from theirs: Big (fork architecture, 4 heads x 32, card 3 layers, note 2), Small (4 x 16, card 2, note 1), six-layer Tiny (4 x 16, card 1, note 1). Earlier evidence on an older student only: the state quantization test (SM32) found rank-1 truncation alone at about +0.0003 curve / +0.0009 instant LogLoss on 8 training users of the 8-head w64 student of that time, and the large state budget (SM46) kept 98.9% of the head energy at rank 4 and 99.8% at rank 6 on that student's states. Nothing has been measured on the six release-candidate models.

## Questions

1. **Spectrum.** For the stored float32 card and note WKV matrices of each release-candidate model, per head and layer: the share of Frobenius energy (sum of squared singular values) kept by the best rank 1, 2, 4, 8 approximation, and how it depends on the entity's answer count (buckets 1-2, 3-4, 5-8, 9-15, 16-31, 32-127, 128+).
2. **Post-training rank truncation.** If every write stores the best rank-r approximation of each head's matrix (r = 1, 2, 4; float32 factors, no quantization), what does it cost in benchmark LogLoss per head (curve, instant) and by answer-count bucket, against the int8 reference format and against the float state? Would a rank-r state be worth a proper GPU test, with or without a cool-down?

## Models (`models.sh`, a copy of the SM145 file)

The six release-candidate models: Big two-head (SM91s2) and Big Instant-Only (SM99) on the fork architecture; Small two-head (SM118) and Small Instant-Only (SM113); six-layer Tiny two-head (SM130 `th`) and six-layer Tiny Instant-Only (SM130 `io`). Heads read: curve and instant for the two-head models, instant for the Instant-Only models (their curve head is reported but not read).

## Users (`users.json`, 8 public benchmark users, about 100k reviews)

- The 3 smoke users of SM141 to SM146 (`users-smoke.json` there): 2388 (3,685 reviews), 4676 (3,716), 1866 (5,673; it has a card with 128+ answers and is also a long-card user).
- 5 more long-card users from `users-longcard.json` (users of the 732 set with a card of 128 or more answers), the smallest ones after 1866, so the run stays short: 2883 (9,451), 3882 (12,325), 1215 (17,626), 4407 (23,053), 966 (24,396).

Eight users is a handful: the numbers are a pointer with wide uncertainty, not a dataset-wide estimate. Every bucket table lists its review counts.

## Method

Scorer: `scripts/small_model_heldout.py` on the CPU reference path, the inputs of every SM14x CPU run (`--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets 0,1,3,5,9,16,32,128,300,1000`), int8 weights (`w8`). New, additive (`scripts/state_lowrank.py`; default runs unchanged):

| label | variant | state |
|---|---|---|
| F32 | `w8+lrsv` | float32 state (same arithmetic as `w8`); records the spectrum of every write |
| REF | `w8+s8hx8` | int8 cards and notes, float16 row scales, int8 shift vectors (the release-candidate format) |
| R1 / R2 / R4 | `w8+lr1` / `w8+lr2` / `w8+lr4` | after each write, each head's stored matrix replaced by its best rank-r approximation; shift vectors float32 |

- **Where:** the same placement as the int codecs (SM47 onward): the answer's own output uses the float update; the stored state, read by the entity's next review, is the processed one. n = the entity's update count so far, this write included (the non-skipped steps of its stream, counted as the tier codec of SM141 counts them); buckets by n.
- **Spectrum (F32):** per write and head, the singular values of the float32 matrix (float64 `svdvals`), energy share kept by rank r = sum of the r largest squared singular values / sum of all. Per (model, stream, layer, head, bucket): writes, sum of the shares (mean), a histogram of the residual 1 - share on a 0.1-decade log grid (quantiles P10 / P50 / P90 of the kept share are read from it, to one grid step). Zero matrices are counted apart. Every write counts once (a write is the state the entity's next review reads), so long cards weigh by their number of answers.
- **Truncation (R1 / R2 / R4):** SVD in float64 of the float32 matrix, factors L = U_r sqrt(S_r) and R = sqrt(S_r) V_r^T rounded to float32, stored = L R in float32. No quantization of the factors, shift vectors float32 (the reference REF has int8 shift vectors; F32 is the paired float control of the truncation alone). For K = 16 (Small, Tiny) rank 4 is a quarter of the full rank, for K = 32 (Big) an eighth.
- **Fork model (Big):** its patched state module (`scripts/compress_quant.py`) is TorchScript; the probe runs with `PYTORCH_JIT=0` (eager modules) and wraps the patched `reference_rwkv7_q` (same loop as the fork's `reference_rwkv7`, plus the probe on the written rows). The expertium models (Small, Tiny) go through `state_vq.Hooks`, as every CPU state codec.
- **Read-out (`sm147_report.py`):** per model and label, review-weighted benchmark LogLoss over the 8 users and the % difference vs REF and vs F32 (100 x (sum LogLoss label / sum LogLoss base - 1), review-weighted; paired user bootstrap 95% CI, 2,000 draws, seed 0, only as a rough width with 8 users); the all-scope LogLoss by answer-count bucket of the card's earlier answers (`prior_buckets`, buckets 0, 1-2, 3-4, 5-8, 9-15, 16-31, 32-127, 128+: the benchmark set stops at 127 earlier answers, so 128+ only exists in the all scope). Per-user aggregate rows stay in `cpu/sm147-rows.jsonl`; no card ids, no review rows.
- **Bytes, for scale only (not a format proposal):** the int8 card / note state of the fork compact layout (SM145) is Big 13,891 / 9,267 B, Small 2,545 / 1,315 B, Tiny 1,315 / 1,315 B per entity. If rank-r factors were stored as int8 with one float16 scale per factor (as the outside recipe), the WKV part of one head would take 2 r K + 4 B instead of K x K + 2 K (int8 rows with float16 row scales); `sm147_report.py` prints the resulting entity sizes with the shift vectors unchanged.

## Checks (before reading any result; `sm147_check.py` -> `check-sm147.json`)

- C1 `scripts/state_lowrank.py` selftest: the truncation has rank r and its Frobenius error equals the norm of the dropped singular values; the spectrum accumulator counts and bins a synthetic rank-1 batch and a zero matrix correctly; a non-finite matrix is left as it is.
- C2 F32 equals plain `w8` in every metric field (curve / instant LogLoss, benchmark, prior buckets) on the 3 smoke users for all six models (the probe changes no arithmetic).
- C3 REF on this tree, eager fork modules included, equals the committed CPU smoke rows of the int3 young tier (SM145) (`../sm145/.../cpu/smoke-cpu-rows.jsonl`) in every metric field, on the 3 smoke users for all six models.
- C4 every row finite; the spectrum's write counts per (model, stream) are equal over layers and heads.

## Run (`run-sm147.sh`)

CPU only, at most 2 processes, memory guard 5.9 GB per process (`memguard.ps1`, a copy of SM145's). Step 1: the smoke proof rows (3 smoke users x 6 models x `w8`, F32, REF), a few minutes. Step 2: 8 users x 6 models x 5 labels in 2 processes (process A: the two Big models; process B: the two Small and two Tiny models). Estimate from a one-user trial (user 2388: Big 3.5 s REF, 5.6 s F32, 9.7 s per truncation label; Tiny 1.0 / 1.2 / 1.6 s) scaled by the SM145 long-card timings: about 60 to 80 minutes wall, so it runs in the background of this session (no detached handoff needed; one is written if the estimate breaks 2 hours).

## Expectation (written before the run, not a rule)

From the older student's numbers (SM32, SM46): rank 4 probably keeps about 99% of the energy on the K = 16 heads and is nearly free; rank 1 is likely visible (order +0.2% to +1%), more on instant than curve, and grows with the answer count (long cards accumulate more directions). Big's K = 32 heads may be less low-rank per head than the 16-wide ones in relative terms (rank 2 of 32 versus 2 of 16).
