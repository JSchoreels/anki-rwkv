# Earlier int8 switch (SM144): design

Pre-registered Sun 11-22 (about 01:30), before any scoring of the new labels. Branch `sm144`, worktree `C:/Users/jonat/workspace/wt/sm144`, cut from `origin/sm141` (a73cecce), the result of the tiered card state by answer count (SM141). No training, no source change: the six release-candidate models are scored as they are, with the scorer of SM141 (`scripts/`, `expertium/`, `rwkv/` unchanged; the label spec `b0.N.b1` already supports any threshold).

## Question

The tiered card state by answer count (SM141) found that no scheme passes, so int8 cards and notes (`w8+s8hx8`) stay for every size. Its reading: int4 cards below 128 answers (scheme A) passes the dataset-wide rule for all six models and fails only on cards with 128 or more answers, and A's cost there was never measured directly (taken from scheme AN by dominance). The cost of int4 lands on long cards even though they switch to int8 at 128 answers, so the damage is done during the int4 period (state error that accumulates while the card is stored as int4 stays in the int8 state afterwards).

**Question: if a card (and a note) switches from int4 to int8 earlier, at 16 or 32 answers instead of 128, does the long-card cost disappear while most of the saving stays?** On the 732 set only 1.9% of answered cards reach 32 answers (7.97% reach 16), so the byte saving should stay close to scheme A's.

## Schemes and labels (`labels.sh`)

Spec `b0.N.b1` = width b0 for answer counts below N, b1 from N (n = the stored state's updates; the scorer counts non-skipped steps, as in SM141). Shift vectors stay int8 in every tier.

| label | cards | notes | variant (scorer) |
|---|---|---|---|
| REF | int8 | int8 | `w8+s8hx8` (the release-candidate format) |
| A | int4 below 128, int8 from 128 | int8 | `w8+t4.128.8c8nhx8` (scheme A of SM141, now measured directly on the long-card users) |
| A32 | int4 below 32, int8 from 32 | int8 | `w8+t4.32.8c8nhx8` |
| A16 | int4 below 16, int8 from 16 | int8 | `w8+t4.16.8c8nhx8` |
| AN32 | int4 below 32, int8 from 32 | same as cards | `w8+t4.32.8c4.32.8nhx8` |
| AN16 | int4 below 16, int8 from 16 | same as cards | `w8+t4.16.8c4.16.8nhx8` |

Proof-only labels in the smoke: U8 (every tier int8, must equal REF), U4 (int4 cards, int8 notes in every tier, must equal C4), C4 (`w8+s4c8nhx8`, the stock compact int4 code).

## Bytes (`sm144_bytes.py` -> `sm144-bytes.json`, `sm144_dist.py` -> `sm144-dist.json`)

The method of SM141: fork compact layout, per-entity bytes at int8 / int4 (Big card 13,891 / 7,747, note 9,267 / 5,171; Small card 2,545 / 1,521, note 1,315 / 803; Tiny card 1,315 / 803, note 1,315 / 803; the formula reproduces the fork encoder's sizes of SM136 exactly), weighted by the end-of-history answer counts of the 723 users of the 732 set (6,136,137 answered cards, 5,461,876 notes with answers; never-answered cards store nothing). Share of answered entities in the int8 tier: cards 0.04% (128), 1.90% (32), 7.97% (16); notes 0.08%, 2.91%, 10.67%. Share of predicted reviews reading an int8 card state: 0.24%, 5.93%, 16.05%.

State bytes vs int8 everywhere (cards + notes, 732 set; an update counter in the fork would add 0.02% to 0.4%, not included):

| scheme | Big | Small | Tiny |
|---|---:|---:|---:|
| A (int8 from 128; notes int8) | -27.74% | -27.55% | -20.59% |
| A32 | -27.22% | -27.04% | -20.21% |
| A16 | -25.54% | -25.36% | -18.96% |
| AN32 | -43.21% | -38.94% | -38.01% |
| AN16 | -40.25% | -36.32% | -35.34% |

(AN, int8 from 128, was -44.2% / -39.8% / -38.9%.) Mean bytes per answered card / note are in `sm144-bytes.json`. So an earlier switch costs about 0.5% (32) to 2.2% (16) of the bytes for the A family and 1% to 4% for the AN family against the 128 switch.

## Rules (pre-registered, `sm144_report.py`; selftest passes)

The same rules as SM141. Statistic: % = 100 x (sum of the scheme's LogLoss / sum of REF's LogLoss - 1), review-weighted; paired user bootstrap, 2,000 draws, seed 0, 95% CI. Heads: two-head models curve and instant, Instant-Only models instant.

- **R1 finite:** no non-finite row and no non-finite LogLoss, on the 732 set (GPU) and on the long-card users (CPU).
- **R2 dataset-wide:** benchmark LogLoss on the 722 rule users of the 732 set, CI upper bound <= **+0.05%** per head.
- **R3 long cards:** on the long-card users (CPU), reviews whose card had 128 or more earlier answers, CI upper bound <= **+0.15%** per head. **Every label is scored directly on the CPU; there is no dominance shortcut** (SM141 took the A and B4 values from AN and B8).
- **R5 coverage:** at least 720 of the 722 rule users for REF and every scheme on the GPU, and at least 48 of the 50 long-card users on the CPU; otherwise the report is `pending` (exit 2) and the main session decides.
- (There is no R4: no young tier.)
- **Go per size:** a scheme passes for a size when it passes R1, R2, R3 for both models of the size (two-head and Instant-Only). The size takes the passing scheme with the smallest total bytes on the 732 set (order for every size: AN32 < AN16 < A < A32 < A16); if none passes, int8 cards and notes stay.
- **Reported, no verdict:** the cost per answer-count bucket (`--prior-buckets`, edges 0, 1, 3, 5, 9, 16, 32, 128, 300, 1000: buckets 1-2, 3-4, 5-8, 9-15, 16-31, 32-127, 128+ answers before the review) for every label, on the GPU rows (all users) and on the CPU long-card rows; the all-scope total; REF vs SM139's GPU rows (integrity); GPU vs CPU on the long-card users (REF, 128+ bucket).

Why the SM141 limits: R2 +0.05% is the repo's minimum effect and the release-candidate format ceiling; R3 +0.15% sits just above the int8 noise level measured on these 50 users (+0.12% CI upper bound for int8 cards against the published format, SM136) and rejects anything near int4's damage (+0.18% to +0.74% for int4 cards at 128+).

## GPU or CPU per user (as SM141)

The 732 set (723 users, R2) runs on the GPU; the **long-card rule R3 runs on the CPU** on the same 50 long-card users as SM141 (`users-longcard.json`: users of the 732 set with a card of 128+ answers, scored completely under the 5.9 GB guard), because GPU rounding flips are amplified there. **Two CPU processes at a time** (SM141 ran three; user 1714 was killed by the memory guard at 5.9 GB), memory guard kept at 5.9 GB per process, a user only starts when the free RAM fits.

GPU smoke rule (first ticket, STOP on failure; includes the SM141 smoke tolerance fix, `TOL_EPS` 1e-9, and the U8 vs REF row comparison within 2e-6 per field, kept as is): the tier kernel with every tier at one width equals the stock kernel bit for bit; REF and U8 GPU vs CPU within 1e-5 on every (user, head); the tiered labels by the discrete-codec rule (per user |delta benchmark LogLoss| <= 3e-4, review-weighted mean signed delta <= 1e-4). The report fix of 11-22 (skip the scalar fields of the benchmark dict in the integrity loop) is kept.

## Run plan (`launch-sm144.sh`, one detached launcher; DRYRUN=1 for a dry run)

| step | where | expected |
|---|---|---|
| CPU proofs (`cpu-proof.sh`) | CPU, now | about 15 min |
| GPU smoke: kernel self-test + 3 users x 6 models x 7 labels | 1 queue ticket (owner sm144, cap 0.8) | about 8 min |
| scoring Tiny, Small, Big: 723 users x 2 models x 6 labels, `--prior-buckets` | 3 queue tickets (+ short retry passes) | about 65 / 120 / 150 min (SM141, same 6 labels, all passes: 92 / 123 / 198 min) |
| long-card users: 50 users x 3 sizes x 2 models x 6 labels | CPU, 2 processes at a time | about 12-16 h (SM141: 4 labels, 3 processes, 7 h) |
| report | CPU | 1 min |

The per-user adaptation pilot (SM143) holds the GPU until about 02:00; the queue is FIFO, so the smoke ticket waits behind it. GPU about 7 h after that; the long-card CPU chain is the long pole (result expected about 14:00 to 18:00 Sun 11-22).

## Fork work if a scheme passes (estimate; nothing changed in the fork here)

The codecs for int4 and int8 exist (precision bytes 0x14 / 0x18, decoded per state by its own byte). The work is the choice at encode time: int4 below N answers, int8 from N, for cards (A family) and notes (AN family); the encoder needs the entity's update count (replay or a counter in the state, +2 to 4 B). Tests: the boundary (answer N-1 int4, N int8), mixed-width decode, the bulk path, the synthetic long-card probe of the fork compact state check (SM133). About 0.5 to 1 agent-day.

## Files

- `sm144_dist.py` -> `sm144-dist.json` (answer-count distribution and tier shares for switch points 16, 32, 64, 128, 256; aggregates only), `sm144_bytes.py` -> `sm144-bytes.json`.
- `labels.sh`, `models.sh` (the six release-candidate models, role rc), `users-smoke.json`, `users-longcard.json` (copies of SM141's).
- `sm144_check.py` (CPU proofs, GPU self-test), `sm144_report.py` (smoke, rules, selftest), `cpu-proof.sh`, `launch-sm144.sh`, `memguard.ps1`.
- Outputs: `cpu/smoke-cpu-rows.jsonl`, `cpu/emulate-rows.jsonl`, `check-cpu-sm144.json`, `gpu/` (tickets, smoke, rows of the 732 set), `cpu/longcard/` (one file per user and size), `sm144-result.{json,md}`.

## Proof results

- `check-cpu-sm144.json`: **CPU proofs ALL OK** (33 checks, `cpu-proof.sh` 01:00 to 01:08 Sun 11-22). K1 bytes OK (formula == the SM136 fork encoder sizes; the A rows of `sm144-bytes.json` equal SM141's). K2 every synthetic identity bitwise (unchanged tier code). K3 on 3 users x 6 models (162 rows, all finite): U8 == REF and U4 == compact int4 in every metric field; **the thresholds take effect where expected**: the largest card answer count read by a review is 10 (user 2388), 17 (user 4676) and 147 (user 1866), and A16 differs from compact int4 only where a review reads a state of 16 or more updates (4676 and 1866, not 2388), A32 only where 32 or more (1866 only), A where 128 or more (1866 only); on the long-card user A16, A32 and A differ from each other, AN16 differs from A16, AN32 from A32 (notes tier) and AN16 from AN32; REF and A rows equal SM141's committed CPU rows bit for bit (12 rows); the GPU code path's CPU twin gives U8 == REF (tiered rows equal the Hooks path to 0.0 in benchmark LogLoss); K4 the REF rows equal SM139's committed CPU rows. The 3-part spec `4.32.8` is accepted by the parser (A already used `4.128.8`; `4.16.8` and `4.32.8` ran for all six models).

