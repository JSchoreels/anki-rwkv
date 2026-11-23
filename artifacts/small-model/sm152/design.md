# Smaller shift vectors and factors on T24 states (SM152): design

Pre-registered Mon 11-23, before any scoring of the new labels (S4, F6, F6S4) on the 732 set or the long-card users. Branch `sm152`, worktree `C:/Users/jonat/workspace/wt/sm152`, cut from `origin/sm150` (d37cc9d4, the rank-1 int8 states (SM150) with its result). No training. Everything is post-training and a narrow copy of the SM150 tree.

## Question

On 11-23 the user chose T24 for every size: card and note states stored as low-rank int8 factors, rank 2 below 16 answers and rank 4 from 16 (`w8+lr2.16.4q8x8`), which saves -81.7% / -66.3% / -64.7% of the int8 state bytes for Big / Small / Tiny. After T24, the header and the shift vectors are about a third of what is left per entity (Big card about 836 of 2,519 bytes, Small card about 242 of 836). Outside research stores the shift vectors at int4 and reports a factor precision ladder (int8 factors free, int6 a little cost, int5 / int4 the knee). **Question: can T24 take int4 shift vectors (S4), int6 factors (F6), or both (F6S4), within the user's budgets?**

## Codec change (`scripts/state_lowrank.py`, `scripts/state_lowrank_cuda.py`, one line of `scripts/small_model_heldout.py`)

- Label grammar `lr<spec><q>x<s>`: `q` = `q8` (int8 factors, unchanged), **`q6`** (int6 factors: codes in [-31, 31], one float16 scale per factor vector, the same arithmetic as q8 with qmax 31) or `f`; `s` = `8` (REF's int8 shift vectors, unchanged), **`4`** (int4 shift vectors: codes in [-7, 7], one float16 scale per vector, REF's shift format at width 4, the existing `x4` path of the int-state labels and of the fork's patched module) or `0` (float, proof only).
- The fused CUDA encode kernel takes the factor qmax as its `quant` argument (127 for q8, 31 for q6, 0 for float) instead of a 0 / 1 flag; with 127 the expressions are the old ones (IEEE division by 127.0f).
- The scorer's parse_variant calls `LowRankCodec.set_ref(rows_bits, shift_bits)` instead of assigning REF's bits; for `x4` the shift bits become `4 | REF's flags` (260), otherwise REF's (264).
- int6 shift vectors (`x6`) are **not** added: the int-state rounding (`state_vq.int_row_quant`, the fork's `_quantize_rows`) maps every width other than 8 to qmax 7, so `x6` would need a change in two shared code paths; it is not one of the scored labels.

## Labels (`labels.sh`) and bytes (`sm152_bytes.py` -> `sm152-bytes.json`)

Bytes per answered card / note in the fork's compact layout of SM148 with **bit-packed codes**: header (magic, precision byte, layer count, three u32 lengths per layer, 1 stored-rank byte), shift vectors (float16 scale + ceil(D x bits / 8) bytes each), WKV (per layer and head at rank c: 2c factor vectors, float16 scale + ceil(K x bits / 8) bytes each). Means over the 723 users' answer counts (`../sm145/sm145-dist.json`), as SM148 / SM150; REF, T24 and Q2 equal SM148's table (checked).

| label | variant | Big card / note (B) | Small card / note (B) | Tiny card / note (B) | bytes vs int8: Big / Small / Tiny | bytes vs T24: Big / Small / Tiny | scored here |
|---|---|---:|---:|---:|---|---|---|
| REF | `w8+s8hx8` | 13,891 / 9,267 | 2,545 / 1,315 | 1,315 / 1,315 | 0 | - | SM148's rows; rescored on the 3 smoke users only |
| T24 | `w8+lr2.16.4q8x8` | 2,518.5 / 1,716.9 | 835.8 / 469.2 | 460.9 / 469.2 | -81.72% / -66.26% / -64.65% | 0 | SM148's rows; rescored on the 3 smoke users only |
| S4 | `w8+lr2.16.4q8x4` | 2,134.5 / 1,460.9 | 739.8 / 405.2 | 396.9 / 405.2 | -84.49% / -70.38% / -69.52% | -15.12% / -12.20% / -13.77% | **new** |
| F6 | `w8+lr2.16.4q6x8` | 2,122.6 / 1,445.6 | 703.9 / 401.4 | 394.9 / 401.4 | -84.60% / -71.44% / -69.74% | -15.75% / -15.34% / -14.38% | **new** |
| F6S4 | `w8+lr2.16.4q6x4` | 1,738.6 / 1,189.6 | 607.9 / 337.4 | 330.9 / 337.4 | -87.36% / -75.56% / -74.60% | -30.87% / -27.55% / -28.15% | **new** |

Byte order F6S4 < F6 < S4 < T24 < REF for every size (checked in `sm152_bytes.py`). Q2 (`w8+lr2q8x8`, rank 2 everywhere) is rescored on the 3 smoke users only, as a third integrity label.

## Rules (pre-registered, `sm152_report.py`; selftest passes)

Statistic: % = 100 x (sum of the label's LogLoss / sum of the reference's LogLoss - 1), review-weighted, paired user bootstrap, 2,000 draws, seed 0, 95% CI (`showtime_report.compare`, as every 732 read-out), on the benchmark LogLoss of the 722 rule users of the 732 set. Heads: two-head models curve and instant, Instant-Only models instant. All scoring on show-time inputs, leak-free, as SM148 / SM150.

- **Big (two-head (SM91s2) and Instant-Only (SM99)):** the label vs the same model at int8 (REF): CI upper bound <= **+0.15%** per head.
- **Small and Tiny: TOTAL cost vs the matching Big model at int8.** Small two-head (SM118) and six-layer Tiny two-head (SM130) vs Big two-head (SM91s2), both heads; Small Instant-Only (SM113) and six-layer Tiny Instant-Only (SM130) vs Big Instant-Only (SM99), instant head only. CI upper bound per head: Small <= **+0.5%**, Tiny <= **+1.0%**.
- **Conditions:** every row finite (GPU, and CPU when the long-card step runs); at least **715 users** on the GPU for the label, its REF and, for Small / Tiny, the matching Big REF (Big is expected at 718 users: its 5 largest users never fit the scorer); below 715 the report is `pending` (exit 2) and the main session decides.
- **Go per size:** the new label with the smallest state bytes that passes for BOTH models of the size, in the order F6S4, F6, S4; **T24 (the user's 11-23 choice) if none passes**. T24 is evaluated with SM148's rows on the same users and rules and its pass is reported, but the fallback does not re-decide it.
- **Report only:** every label vs T24 (the cost of this extra step alone), per head with its CI; the cost by answer-count bucket (`--prior-buckets`, edges 0, 1, 3, 5, 9, 16, 32, 128, 300, 1000) vs own int8 and vs T24; the starting point of Small and Tiny at int8 vs Big at int8; review evaluations per second per label.
- **Warnings (never a verdict), later step:** (a) long cards, reviews whose card had 128 or more earlier answers, on the 50 long-card users (CPU, Small and Tiny only), the label vs its own int8, flagged above +0.5% (point) or +1.0% (upper bound); (b) per-review and per-card |dR| = |p(label) - p(REF)|, the share of cards with a review |dR| > 0.10, flagged above 1%. **These are not run with the GPU part**: a CPU speed measurement for the fork is due, so the CPU long-card chain is a separate later launch (`PART=longcard`, HANDOFF.md). Until then the report says "not run yet" and the go is decided on the rules alone. Big long-card users are not run (13 users exceed the 6 GB per-process limit, SM148 / SM150).

## Reused rows and the integrity checks

REF and T24 come from SM148's committed rows (`../sm148/gpu/sm148-rows-732.jsonl`, and for the later long-card step `../sm148/cpu/sm148-longcard-rows.jsonl` and SM148's REF dumps). The codec change must leave every existing label unchanged:

- CPU (K3, before the GPU): REF, T24 and Q2 rescored on SM148's 3 smoke users (2388, 4676, 1866) for all six models equal SM148's committed CPU smoke rows to **1e-6** in benchmark LogLoss per user and head, with equal review counts. If not, **stop and report**.
- GPU (smoke ticket): REF and T24 on the GPU equal SM148's GPU smoke rows (`../sm148/gpu/smoke-gpu-rows.jsonl`) within **1e-5** per user and head (the fused kernel with qmax 127 is the old kernel).

## Proofs (CPU, before any GPU ticket; `cpu-proof.sh`, `sm152_check.py cpu` -> `check-cpu-sm152.json`)

- K1 bytes and bit packing (`sm152_bytes.py`): REF / T24 / Q2 bytes equal SM148's table; byte order; **pack / unpack round trip** of int8, int6 and int4 codes on vectors of 16, 32, 64 and 128 codes (packed length = the layout's byte count); real int6 factor vectors (`quant_vectors`, qmax 31) and int4 shift vectors (`int_row_quant(4 | float16 scale)`) decode from their packed codes and float16 scale to exactly the codec's floats.
- K2 `state_lowrank.selftest` and `selftest_sm148` on the CPU: the SM148 identities, plus the int6 round trip (codes in [-31, 31], error <= half a step, idempotent, qmax 127 = the int8 path) and the SM152 grammar (q6, x4; set_ref gives shift bits 260 for x4 and 264 for x8; q8 / x8 rows keep SM148's descriptor; `x6` and `q4` refused).
- K3 scorer rows on the 3 smoke users x 6 models x REF, T24, Q2, S4, F6, F6S4 (CPU): every row finite; **REF, T24 and Q2 equal SM148's rows to 1e-6**; S4, F6 and F6S4 differ from T24 on every (user, model); F6S4 differs from S4 and from F6 on at least one user per model; the rows' codec descriptors (spec 2.16.4, q8 / q6, shift bits 264 / 260).
- K5 the dump: a file per (user, model, label) for REF, S4, F6, F6S4; dumped reviews per bucket equal the rows' all-scope counts; equal review sets; |dR| S4 vs REF above 0; `sm152_dr.py` aggregates the smoke dumps.
- GPU smoke ticket (first ticket, STOP on failure): `sm152_check.py gpu-selftest` (Jacobi kernel; `selftest_sm148` on CUDA including the int6 round trip; the fused kernel finite and equal to the CPU loop for float factors at 1e-5; **G4: on the same inputs the fused kernel's int6 output differs from its int8 output and is closer to the CPU int6 loop than to the CPU int8 loop**) + the 3 smoke users x 6 models x REF, T24, S4, F6, F6S4 vs the CPU rows: REF within 1e-5; the low-rank labels by the discrete-codec rule (per user |delta| <= 3e-4, review-weighted mean signed delta <= 1e-4); REF and T24 vs SM148's GPU smoke rows within 1e-5.

## Run plan (`launch-sm152.sh`; DRYRUN=1 for a dry run)

| step | where | expected |
|---|---|---|
| CPU proofs | CPU | about 15 min (done before the launch) |
| GPU smoke ticket | queue, cap 0.8 | about 2 to 5 min |
| scoring Tiny / Small / Big: 723 users x 2 models x {S4, F6, F6S4}, `--prior-buckets` | 3 tickets, cap 0.8 (retry at 0.92 only after a pass without progress) | about 60 to 65 / 85 to 90 / 140 to 160 min (SM150: 43 / 59 / 105 min for 2 labels), about 5 GPU hours in all |
| report | CPU | a few minutes |
| later, separate: long-card users, 50 users x {Tiny, Small} x 2 models x {S4, F6, F6S4}, with the dump; dR read-out; report again | CPU, 2 processes, memory guard 5.9 GB, `PART=longcard` | about 3 to 4 h (not started now) |

Quiet windows (`quiet.json`) are respected by both parts as in SM150. Resources: GPU cap 0.8 (0.92 only for retry passes), `PYTORCH_JIT=0`, order Tiny, Small, Big.

## Expectation (written before scoring, not a rule)

T24's margins under the rules (SM148 rows, upper bound vs the ceiling): Big two-head (SM91s2) +0.043% / +0.059% and Big Instant-Only (SM99) +0.065% against +0.15%; Small two-head (SM118) +0.405% / +0.227% against +0.5%; six-layer Tiny two-head (SM130) +0.899% / +0.802% against +1.0%. The tightest heads (Small two-head and six-layer Tiny two-head curve) leave about +0.1%. int6 factors should cost little (int6 has about four times the int8 rounding error, still small next to the rank truncation): perhaps +0.01% to +0.05%. int4 shift vectors are the bigger unknown (16x the int8 step on every stored token-shift vector, every layer, card and note): perhaps +0.05% to +0.3%. F6 likely passes Big and probably Small and Tiny; S4 and F6S4 may fail the Small and Tiny two-head curve heads.

## Files

`labels.sh`, `models.sh` (the six release-candidate models), `users-smoke.json`, `users-longcard.json`, `memguard.ps1` (SM150's copy, for the later long-card step); `sm152_bytes.py` -> `sm152-bytes.json`; `sm152_check.py`; `sm152_report.py` (smoke, rules, selftest, pins); `sm152_dr.py` (SM150's with the SM152 labels); `cpu-proof.sh`; `launch-sm152.sh`; `HANDOFF.md`. Outputs: `cpu/smoke-cpu-rows.jsonl`, `check-cpu-sm152.json`, `gpu/` (tickets, smoke, rows of the 732 set), `sm152-result.{json,md}`; later `cpu/longcard/<size>/`, `cpu/sm152-longcard-rows.jsonl`, `sm152-dr.json`. Not committed: `work/sm152-dump*` (per-review values).
