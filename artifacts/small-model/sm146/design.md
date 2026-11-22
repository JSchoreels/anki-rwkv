# Aggressive state tiers for Tiny (SM146): design

Pre-registered Sun 11-22 (about 23:00), before any scoring of the new labels (no row of an int3 or int2 young tier of this experiment exists yet). Branch `sm146`, worktree `C:/Users/jonat/workspace/wt/sm146`, cut from branch `sm145` (f2b872c2, the int3 young tier on the early switch (SM145)). No training: the two six-layer Tiny release-candidate models (the six-layer Tiny two-head model (SM130) and the six-layer Tiny Instant-Only model (SM130)) are scored as they are, with the scorer of SM145 (unchanged, including `--dump-scope all`).

## Question and the user's decision

User decision 11-22 (translated from French): for the Tiny model it is acceptable to take a bigger change for int3 and int4 as long as the overall accuracy loss does not go past the global 1% value. Our reading, pre-registered here: for the two six-layer Tiny models the dataset-wide rule becomes "the Tiny model with that state format against the matching Small model at int8 (`w8+s8hx8`), benchmark LogLoss paired by user on the same 723 users (722 rule users), 95% CI upper bound <= +1.00% per head". The matching Small models: the show-time two-head cool-down for Small (SM118) for the two-head Tiny model (curve and instant heads) and the longer Small Instant-Only model (SM113) for the Tiny Instant-Only model (instant head).

At int8 the six-layer Tiny models are already +0.467% [+0.398, +0.546] (curve) and +0.524% [+0.484, +0.567] (instant) for the two-head model and +0.565% [+0.523, +0.608] (instant) for the Instant-Only model against Small (SM130 RESULT.md; recomputed here from the rows, same numbers). With the int4 switch at 16 answers (AN16) they are +0.479% / +0.543% / +0.595%, so about 0.4% of room is left below the +1.00% line.

**Question: how far can the young-card tiers go below AN16 (int3 or ternary int2 for the first answers of a card or note) before the six-layer Tiny models leave +1.00% against Small, and how many bytes does that save?** Plus the user's earlier concern (no card without a real retention prediction) as one per-card guard.

## Labels (`labels.sh`; Tiny only, cards and notes alike)

Spec `b0.T.b1.N.b2` = width b0 for answer counts n < T, b1 for T <= n < N, b2 from N (n = the stored state's non-skipped updates, as in SM141 / SM144 / SM145). Shift vectors stay int8 in every tier. Every scored label keeps the AN16 switch to int8 at 16 answers. Int2 is ternary (codes -1, 0, 1) as in SM141.

| label | state format | variant (scorer) |
|---|---|---|
| REF | int8 | `w8+s8hx8` (reference of the cost vs Tiny int8, of R3 and of the per-card read-out) |
| AN16 | int4 below 16 answers, int8 from 16 | `w8+t4.16.8c4.16.8nhx8` (the paired baseline, SM144's AN16) |
| I3a | int3 for 1-4, int4 for 5-15, int8 from 16 | `w8+t3.5.4.16.8c3.5.4.16.8nhx8` |
| I3b | int3 for 1-8, int4 for 9-15, int8 from 16 | `w8+t3.9.4.16.8c3.9.4.16.8nhx8` |
| I2a | int2 for 1-4, int4 for 5-15, int8 from 16 | `w8+t2.5.4.16.8c2.5.4.16.8nhx8` |
| I2b | int2 for 1-4, int3 for 5-15, int8 from 16 | `w8+t2.5.3.16.8c2.5.3.16.8nhx8` |

Proof-only labels in the CPU / GPU smoke: U8 (every tier int8, must equal REF) and P4 (`w8+t4.5.4.16.8c4.5.4.16.8nhx8`, the 5-part spec with the young tier at int4, must equal AN16).

**Selection by bytes (spread rule: each label saves at least 1.5% of the int8 bytes more than the next less aggressive one).** State bytes vs int8 everywhere, six-layer Tiny, cards + notes, 732 set (`sm146_bytes.py` -> `sm146-bytes.json`; method of SM145: fork compact layout, Tiny entity 1,315 / 803 / 675 / 547 bytes at int8 / int4 / int3 / int2, weighted by the end-of-history tier shares of the 723 users; an update counter in the fork would add 0.02% to 0.4%, not included):

| label | bytes vs int8 | extra saving vs the next less aggressive label |
|---|---:|---:|
| AN16 | -35.34% | - |
| I3a | -39.80% | 4.46% |
| I3b | -42.71% | 2.91% |
| I2a | -44.27% | 1.56% |
| I2b | -48.64% | 4.37% |
| (not scored) int3 below 16 (`t3.16.8`) | -44.17% | 1.46% vs I3b, 0.10% vs I2a: dropped by the rule (it saves the same bytes as I2a; I2a keeps the ternary step in the ladder) |
| (not scored) int2 for 1-8, int3 to 15 (`t2.9.3.16.8`) / int4 to 15 (`t2.9.4.16.8`) | -51.54% / -50.08% | beyond the budget of 6 labels and, at int2 for 1-8 answers, read by 62% to 69% of all predictions |

Four aggressive labels plus the two baselines = 6 labels, so the GPU part stays at about 1 hour. The step between I3b and I2a is the smallest that passes the 1.5% rule (1.56%).

## Rules (pre-registered, `sm146_report.py`; selftest passes)

Statistic: % = 100 x (sum of the label's LogLoss / sum of the reference's LogLoss - 1), review-weighted; paired user bootstrap, 2,000 draws, seed 0, 95% CI (`showtime_report.compare`, as every 732 read-out). Heads: two-head model curve and instant, Instant-Only model instant.

- **R1 finite:** no non-finite row and no non-finite LogLoss, on the 732 set (GPU) and on the long-card users (CPU).
- **R2 dataset-wide (the user's rule, new):** the label's benchmark LogLoss on the 722 rule users of the 732 set vs the matching Small model at int8 (`w8+s8hx8`), same scorer, same inputs, same users, 95% CI upper bound <= **+1.00%** per head.
- **R3 long cards (the user's 11-22 long-card tolerance, as SM144 / SM145):** on the 50 long-card users (CPU, every label scored directly), reviews whose card had 128 or more earlier answers: vs Tiny int8 (REF), point estimate <= **+0.5%** and CI upper bound <= **+1.0%** per head.
- **G per-card guard (pre-registered, new):** on the same 50 long-card users, the share of cards whose worst review has |dR| > 0.10 (dR = p(label) - p(REF) on the same review; all cards of the long-card users, every predicted review) is at most **1%** per head.
- **R5 coverage:** at least 720 of the 722 rule users on the GPU for REF, AN16 and every label, and for the Small int8 rows; at least 48 of the 50 long-card users on the CPU; a complete per-review read-out (at least 48 users per model, label and head); otherwise the report is `pending` (exit 2) and the main session decides.
- **Go:** a label passes when it passes R1, R2, R3 and G for both models. The go pick is the passing label with the smallest state bytes (I2b < I2a < I3b < I3a < AN16); if none passes, int8 stays (REF).

The Small int8 reference rows come from SM145's GPU file (`gpu/small-ref-rows.jsonl`, 1,446 rows = 2 Small models x 723 users, sha256 pinned in the report). They are identical, field by field and prior-bucket by prior-bucket, to the rows of SM144 (same scorer, same show-time inputs `e2s-query`, leak-free `--deleted-cards neighbour`, same 732 set minus 9 never-fit users): max difference 0.0 on every user. REF and AN16 of Tiny reproduce bitwise across SM144 and SM145 as well (0.0 on the GPU and on the CPU long-card rows), so the Small int8 rows are not scored again here; this run scores Tiny REF and AN16 again anyway and the report prints the comparison with SM144 and SM145 (integrity, report only).

### Reported, no verdict

- The cost of every label vs Tiny int8: dataset-wide (GPU, 722 rule users) and per answer-count bucket (`--prior-buckets`, edges 0, 1, 3, 5, 9, 16, 32, 128, 300, 1000: buckets 1-2, 3-4, 5-8, 9-15, 16-31, 32-127, 128+ answers before the review), on the GPU rows (all users) and on the CPU long-card rows; the all-scope total; the extra cost of every label vs AN16 (paired).
- Tiny int8 vs Small int8 (the starting point of R2) and each label vs Small int8 for both heads.
- The per-card worst-case read-out of `sm146_dr.py` on the 50 long-card users (CPU rows, every label's per-review predictions dumped by `--dump-scope all` into `work/sm146-dump/`, never committed): per review, by the card's answers before the review (0, 1-2, 3-4, 5-8, 9-15, 16-31, 32-127, 128+, all): n, mean |dR|, P50 / P90 / P99 / P99.9 / max, share and count of reviews with |dR| above 0.05 and 0.10; per card (the card's worst review, by the card's answers in the whole history): cards, P50 / P90 / P99 / max, share and count of cards above 0.05 and 0.10; per user: users with a review above 0.05 / 0.10. Committed: aggregates only (`sm146-dr.json`, tables in `sm146-result.md`), no card ids, no review rows.
- Integrity: REF and AN16 rows vs SM144's and SM145's committed rows (GPU 732 set, CPU long-card users; expected 0.0); GPU vs CPU on the long-card users (REF, 128+ bucket).

## GPU or CPU per user

The 732 set (723 users; R2) runs on the GPU; R3 and the per-review read-out run on the CPU on the same 50 long-card users as SM141 / SM144 / SM145 (`users-longcard.json`). GPU through the queue (owner sm146, cap 0.8, 0.92 only for a retry pass without progress); the SM145 tickets had all ended before this launch. The CPU chain starts a user only when fewer than 3 CPU scoring processes (any experiment) run on the machine, fewer than 3 of this chain run, and the free RAM fits; memory guard 5.9 GB per process (`memguard.ps1`).

GPU smoke rule (first ticket, STOP on failure; as SM144 / SM145): the tier kernel with every tier at one width equals the stock kernel bit for bit (`gpu-selftest`, which also runs 3/4/8, 2/4/8, 2/3/8 and 3/8 tier specs); REF and U8 GPU vs CPU within 1e-5 on every (user, head); U8 vs REF on the GPU within 2e-6 per field; the tiered labels by the discrete-codec rule (per user |delta benchmark LogLoss| <= 3e-4, review-weighted mean signed delta <= 1e-4); SM141's smoke saw at most 3e-6 for its int2 label on Tiny.

## Proofs (CPU, before any GPU ticket; `cpu-proof.sh`, `sm146_check.py cpu` -> `check-cpu-sm146.json`; short Tiny-only version of SM145's)

- K1 bytes: the formula equals SM136's Tiny encoder sizes; the label order and the 1.5% steps hold.
- K2 synthetic, seeded (unchanged tier code): with every tier at one width the tier code equals the stock code bit for bit; with 3/4/8, 2/4/8, 2/3/8 and 3/8 tier specs every row gets the width of its own update count (manual per-step loop), on the state quantizer, the recurrence and the fork's patched reference.
- K3 scorer rows, the 3 smoke users (2388, 4676, 1866) x the 2 Tiny models x 8 labels, CPU: U8 == REF and P4 == AN16 in every metric field; each aggressive label differs from AN16 and from REF on every user; the pairs (I3a, I3b), (I2a, I2b), (I2a, I3a), (I2b, I3b) differ; every row finite; the GPU code path's CPU twin (`--emulate-gpu-path`) gives U8 == REF.
- K4: this tree's REF and AN16 CPU rows equal SM145's committed CPU smoke rows in every field.
- K5 the dump: a file per (user, model, label); the dumped reviews per bucket equal the rows' all-scope counts; equal review sets across labels; |dR| of U8 vs REF exactly 0 and of AN16 vs REF above 0; `sm146_dr.py` aggregates the smoke dumps (finite, complete).

## Run plan (`launch-sm146.sh`, one detached launcher; DRYRUN=1 for a dry run)

| step | where | expected |
|---|---|---|
| CPU proofs (`cpu-proof.sh`) | CPU, before the launch | a few minutes |
| GPU smoke: kernel self-test + 3 users x 2 models x 7 labels | 1 queue ticket | about 2 min |
| scoring Tiny: 723 users x 2 models x 6 labels, `--prior-buckets` | 1 queue ticket (+ short retry passes) | about 50 min (SM144 Tiny, 6 labels: 49 + 1 min) |
| long-card users: 50 users x 2 models x 6 labels, with the dump | CPU, up to 3 processes | about 3.2 process-hours, about 1 to 1.5 h wall (SM145 Tiny, 5 labels: 2.65 process-hours) |
| per-review read-out (`sm146_dr.py aggregate`), report | CPU | a few minutes |

Expected speed: GPU about 130k review evaluations per second over the two models and six labels (SM144 / SM145 Tiny rates); CPU about 3,000 to 4,500 per process.

## Expectation (written before scoring, not a rule)

From SM141's ternary label (int2 for 1-4 answers with int4 to 127 answers: +0.162% curve and +0.261% instant on the two-head model, dataset-wide vs int8; int3 for 1-8 answers: +0.103% instant on the two-head model and +0.150% on the Instant-Only model) on top of AN16, I2a lands around +0.7% to +0.9% vs Small, close to the +1.00% line on the instant heads; I2b (int3 for 5-15 as well) is likely over it. I3a and I3b are likely inside. R3 should look like AN16's (the long cards are int8 from 16 in every label). The per-card guard has no earlier reference; AN16 is the baseline it is read against.

## Fork work if a label is chosen (estimate; nothing changed in the fork here)

AN16 first (the codec choice at encode time by the entity's update count, SM144). On top: Int3Compact (3-bit packing, 8 codes in 3 bytes) and, for the int2 labels, Int2Compact (ternary, 4 codes per byte); parity vectors from `sm145_bytes.py` (`pack` / `unpack`) and the scorer's `TierQuant`; tests at the tier boundaries. The Tiny Instant-Only model and the Tiny two-head model would both take the chosen label; the Small and Big sizes keep their own choice (AN16 under SM144 / SM145's results).

## Files

- `labels.sh`, `models.sh` (the two Tiny models), `users-smoke.json`, `users-longcard.json` (copies of SM145's), `sm146-dist.json` (SM145's tier shares, aggregates only), `gpu/small-ref-rows.jsonl` (Small int8 rows, see above).
- `sm146_bytes.py` -> `sm146-bytes.json`, `sm146_check.py` (CPU proofs, GPU self-test), `sm146_report.py` (smoke, rules, selftest), `sm146_dr.py` (per-review read-out, selftest), `cpu-proof.sh`, `launch-sm146.sh`, `memguard.ps1`, `cpu-count.ps1`.
- Outputs: `cpu/smoke-cpu-rows.jsonl`, `cpu/emulate-rows.jsonl`, `cpu/smoke-dr.json`, `check-cpu-sm146.json`, `gpu/` (tickets, smoke, rows of the 732 set), `cpu/longcard/tiny/` (one file per user), `cpu/sm146-longcard-rows.jsonl`, `sm146-dr.json`, `sm146-result.{json,md}`. Not committed: `work/sm146-dump/`, `work/sm146-dump-smoke/` (per-review values).
