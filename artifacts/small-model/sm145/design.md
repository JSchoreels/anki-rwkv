# Int3 young tier on the early switch (SM145): design

Pre-registered Sun 11-22 (about 12:50), before any scoring of the new labels on the 732 set or the long-card users. Branch `sm145`, worktree `C:/Users/jonat/workspace/wt/sm145`, cut from `origin/sm144` (8c662c04, the result of the earlier int8 switch (SM144)). No training: the six release-candidate models are scored as they are, with the scorer of the earlier int8 switch (SM144) plus one additive flag (`--dump-scope all`, below; default output unchanged).

## Question

The earlier int8 switch (SM144) found that with the int8 switch at 16 answers for cards and notes (AN16, `w8+t4.16.8c4.16.8nhx8`: int4 below 16 answers, int8 from 16), every one of the six release-candidate models passes the dataset-wide rule and, under the user's 11-22 long-card tolerance, the long-card rule; bytes vs int8 are -40.3% Big, -36.3% Small, -35.3% Tiny. The tiered card state by answer count (SM141) measured int3 for cards with 1-4 answers at about +0.01% to +0.03% on top of int4 (switch at 128), and it passed the young-card rule everywhere, but int3 plus int4 notes with the switch at 128 went over +0.05% dataset-wide (the long-card damage of the 128 switch was part of that). Int2 failed clearly and is not tested again.

**Question: does int3 for the youngest cards (and maybe notes) on top of AN16 still pass, and how many bytes does it add?** And, new, the user's concern: does any card end up without a real retention prediction (a per-card worst case, not only averages)?

## Labels (`labels.sh`)

Spec `b0.T.b1.N.b2` = width b0 for answer counts n < T, b1 for T <= n < N, b2 from N (n = the stored state's updates, counted as in SM141 / SM144: the non-skipped steps of the entity's stream). Shift vectors stay int8 in every tier. Every int3 label keeps the AN16 switch to int8 at 16 answers.

| label | cards | notes | variant (scorer) |
|---|---|---|---|
| REF | int8 | int8 | `w8+s8hx8` (the release-candidate format; reference of every rule) |
| AN16 | int4 below 16, int8 from 16 | same as cards | `w8+t4.16.8c4.16.8nhx8` (SM144's AN16; the paired baseline of the int3 labels) |
| Y2N | int3 for 1-2, int4 for 3-15, int8 from 16 | same as cards | `w8+t3.3.4.16.8c3.3.4.16.8nhx8` |
| Y4C | int3 for 1-4, int4 for 5-15, int8 from 16 | as AN16 (int4 below 16, int8 from 16) | `w8+t3.5.4.16.8c4.16.8nhx8` |
| Y4N | int3 for 1-4, int4 for 5-15, int8 from 16 | same as cards | `w8+t3.5.4.16.8c3.5.4.16.8nhx8` |

Why these: 1-4 answers is the cut-off SM141 measured (just below the median card, P50 = 5 answers); Y4C (cards only) separates the cards' int3 cost from the notes', since in SM141 the cost of int3 on cards and notes together was the part that pushed the six-layer Tiny Instant-Only model (SM130) over the dataset-wide ceiling; Y2N (1-2 answers, P25) is the cautious step for the sizes with the least margin (the six-layer Tiny Instant-Only model (SM130) has AN16 at +0.030% [+0.027, +0.033], so an int3 increment like SM141's +0.029% would not fit). Int3 for 1-8 answers is not scored (it reads 69% of all predictions and SM141's B8 failed for Small and Tiny); its bytes are listed below as a possible follow-up for Big if Y4N passes with room.

Proof-only labels in the CPU / GPU smoke: U8 (every tier int8, must equal REF), U4 (int4 cards, int8 notes in every tier, must equal C4), C4 (`w8+s4c8nhx8`, the stock compact int4 code), P4 (`w8+t4.5.4.16.8c4.5.4.16.8nhx8`: the 5-part spec with the young tier at int4, must equal AN16).

## Bytes (`sm145_bytes.py` -> `sm145-bytes.json`, `sm145_dist.py` -> `sm145-dist.json`)

The method of SM141 / SM144: fork compact layout, bytes per entity and width (Big card 13,891 / 7,747 / 6,211 B at int8 / int4 / int3, note 9,267 / 5,171 / 4,147; Small card 2,545 / 1,521 / 1,265, note 1,315 / 803 / 675; Tiny card and note 1,315 / 803 / 675; the formula reproduces the fork encoder's sizes of SM136 for int8 and int4, and the 3-bit packing round-trips bitwise), weighted by the end-of-history answer counts of the 723 users of the 732 set (6,136,137 answered cards, 5,461,876 notes with answers; never-answered cards store nothing).

Tier shares with the switch at 16 (cards / notes): int3 for 1-2 answers holds 22.9% / 22.2% of the answered entities and is read by 28.0% / 24.0% of the predicted reviews; int3 for 1-4 holds 46.6% / 45.1% and is read by 47.7% / 41.6%; int8 (16+) holds 8.0% / 10.7% and is read by 16.1% / 21.7%.

State bytes vs int8 everywhere (cards + notes, 732 set; an update counter in the fork would add 0.02% to 0.4%, not included):

| label | Big | Small | Tiny | extra vs AN16 (Big / Small / Tiny) |
|---|---:|---:|---:|---|
| AN16 | -40.25% | -36.32% | -35.34% | - |
| Y2N | -42.75% | -38.57% | -37.53% | -2.50% / -2.25% / -2.19% |
| Y4C | -43.48% | -39.53% | -37.74% | -3.23% / -3.21% / -2.40% |
| Y4N | -45.34% | -40.91% | -39.80% | -5.09% / -4.59% / -4.46% |
| (not scored) Y8N, int3 for 1-8 | -48.65% | -43.91% | -42.70% | -8.40% / -7.59% / -7.36% |

(The "extra" column is in % of the int8 bytes.) The byte order is the same for every size: Y4N < Y4C < Y2N < AN16 (`sm145_report.py check` asserts it).

## Rules (pre-registered, `sm145_report.py`; selftest passes)

Statistic: % = 100 x (sum of the label's LogLoss / sum of REF's LogLoss - 1), review-weighted; paired user bootstrap, 2,000 draws, seed 0, 95% CI. Heads: two-head models curve and instant, Instant-Only models instant.

- **R1 finite:** no non-finite row and no non-finite LogLoss, on the 732 set (GPU) and on the long-card users (CPU).
- **R2 dataset-wide:** benchmark LogLoss on the 722 rule users of the 732 set (`showtime_report.compare`), CI upper bound <= **+0.05%** per head vs REF.
- **R3 long cards:** on the 50 long-card users (CPU, every label scored directly), reviews whose card had 128 or more earlier answers: point estimate <= **+0.5%** and CI upper bound <= **+1.0%** per head vs REF (the user's 11-22 tolerance: "If really long cards have to somewhat lose 0.5 or 1% precision but just them, it's ok").
- **R4 young cards (int3 labels):** on the 732 set (GPU), reviews whose card had 1 to M earlier answers (M = 2 for Y2N, 4 for Y4C and Y4N; the reviews that read an int3 card state), CI upper bound <= **+0.10%** per head vs REF.
- **R5 coverage:** at least 720 of the 722 rule users for REF and every label on the GPU, at least 48 of the 50 long-card users on the CPU, and a complete per-review read-out (at least 48 users per model, label and head); otherwise the report is `pending` (exit 2) and the main session decides.
- **Go per size:** a label passes for a size when it passes R1-R4 for both models of the size (two-head and Instant-Only). The size takes the passing label with the smallest total bytes (Y4N < Y4C < Y2N < AN16); if none passes, int8 cards and notes stay (REF).

### Worth its extra bytes (pre-registered criterion; the recommendation on top of the go)

An int3 label needs a new Int3Compact codec in the fork (about 1 to 1.5 agent-days on top of AN16, plus one more tier to test and keep). For a size, an int3 label is **worth its extra bytes** when all of these hold:

1. it passes R1-R4 for both models of the size;
2. it saves at least **2%** of the int8 state bytes beyond AN16 (all three labels do, table above);
3. its paired extra cost vs AN16 on the 732 set (benchmark LogLoss, % of AN16's) is at most **+0.02%** (point estimate) and its CI upper bound at most **+0.03%**, per head and for both models (less than half the repo's 0.05% minimum effect, so the int3 step does not eat most of the dataset-wide ceiling);
4. per-card worst case (long-card users, below): the share of cards whose worst review has |dR| > 0.10 is at most **twice AN16's** share (or at most 0.01% of cards when twice AN16's is below that), per head and for both models.

**Recommended per size** = the passing label with the smallest bytes among AN16 and the int3 labels worth their bytes; REF if neither AN16 nor any int3 label passes. The go pick and the recommendation are both reported; they differ only when the smallest passing int3 label is not worth its bytes.

### Reported, no verdict

- The extra cost of each int3 label vs AN16 (paired, % of AN16's LogLoss): dataset-wide (GPU), the young bucket 1-M (GPU), cards with 128+ answers (CPU).
- The cost per answer-count bucket (`--prior-buckets`, edges 0, 1, 3, 5, 9, 16, 32, 128, 300, 1000: buckets 1-2, 3-4, 5-8, 9-15, 16-31, 32-127, 128+ answers before the review) for every label vs REF, on the GPU rows (all users) and on the CPU long-card rows; the all-scope total.
- Integrity: REF and AN16 rows vs SM144's committed rows (GPU 732 set, CPU long-card users; expected 0 on the CPU, GPU deterministic as SM144 found vs SM141); GPU vs CPU on the long-card users (REF, 128+ bucket).

## New measurement: per-review and per-card |dR| (long-card users, CPU; `sm145_dr.py` -> `sm145-dr.json`)

The user cares that no card ends up without a real retention prediction, not only that averages hold. So, on the 50 long-card users (the CPU rows of R3, the same processes), every label's per-review predictions are dumped and compared with REF's on the same review:

- **Source:** the scorer's `--dump` wrote only the benchmark reviews, which stop at 127 earlier answers of a card (`prior_bucket_fields`), so long cards would be missing. New flag `--dump-scope all` (`scripts/small_model_heldout.py`, additive, default unchanged): one file per (user, model, label) with, per head, the review_th of every review the scorer predicts (exactly the "all" scope of `--prior-buckets`: keys with a rating) and the head's probability as float32 (the heads' own precision, lossless). The dumps go to `work/sm145-dump/` (never committed; per-review values). A killed or failed user's dumps are deleted, so the read-out uses only users with complete rows.
- **dR** = p(label) - p(REF) on the same review; curve = the two-head models' curve head, instant = the instant head; labels AN16, Y2N, Y4C, Y4N.
- **Per review** (pooled over the users), by the card's answers before the review (0, 1-2, 3-4, 5-8, 9-15, 16-31, 32-127, 128+, all): n, mean |dR|, mean signed dR, P50 / P90 / P99 / P99.9 / max of |dR|, share and count of reviews with |dR| > 0.05 and > 0.10.
- **Per card**: each card's worst review (max |dR| over its predicted reviews), by the card's answers in the whole history (1-2, ..., 128+, all): cards, P50 / P90 / P99 / max, share and count of cards above 0.05 and 0.10.
- **Per user**: users with at least one review above 0.05 / 0.10, the largest per-user share above 0.05.
- Committed: aggregates only (`sm145-dr.json`, tables in `sm145-result.md`): no user or card ids, no review rows. The per-card share above 0.10 enters criterion 4 of "worth its bytes"; everything else is report only.

## GPU or CPU per user (as SM141 / SM144)

The 732 set (723 users; R2, R4) runs on the GPU; the long-card rule R3 and the per-review read-out run on the CPU on the same 50 long-card users as SM141 / SM144 (`users-longcard.json`), where GPU rounding flips are amplified. Two CPU processes at a time, memory guard 5.9 GB per process (`memguard.ps1`), a user only starts when the free RAM fits. GPU tickets through the queue (owner sm145, cap 0.8, 0.92 only for a retry pass without progress).

GPU smoke rule (first ticket, STOP on failure; as SM144): the tier kernel with every tier at one width equals the stock kernel bit for bit (`gpu-selftest`, which also runs a 3/4/8 tier spec); REF and U8 GPU vs CPU within 1e-5 on every (user, head), U8 vs REF on the GPU within 2e-6 per field; the tiered labels by the discrete-codec rule (per user |delta benchmark LogLoss| <= 3e-4, review-weighted mean signed delta <= 1e-4), the int3 tiers included as in SM141.

## Proofs (CPU, before any GPU ticket; `cpu-proof.sh`, `sm145_check.py cpu` -> `check-cpu-sm145.json`)

- K1 bytes: the formula equals SM136's encoder sizes; 3-bit pack round trip.
- K2 synthetic, seeded (unchanged tier code of SM141): with every tier at one width the tier code equals the stock code bit for bit on all four paths; with 3/4/8 tiers every row gets the width of its own update count (manual per-step loop).
- K3 scorer rows, the 3 smoke users (2388, 4676, 1866) x the 6 models x 9 labels, CPU: U8 == REF and U4 == C4 in every metric field; P4 == AN16 in every metric field (the 5-part parse with the switch at 16 is the AN16 code); Y2N, Y4C, Y4N each differ from AN16 on every user (the young tier takes effect) and Y4N differs from Y4C (notes tier) and from Y2N (cut-off); every row finite; the GPU code path's CPU twin (`--emulate-gpu-path`, Small and Tiny) gives U8 == REF.
- K4: this tree's REF and AN16 CPU rows equal SM144's committed CPU smoke rows in every field (prior buckets included).
- K5 the dump: a file per (user, model, label); per head the dumped reviews per answer-count bucket equal the row's all-scope counts (the dump is exactly the scored scope); the review sets are equal across labels; |dR| of U8 vs REF is exactly 0 and of AN16 vs REF is above 0; `sm145_dr.py` aggregates the smoke dumps (finite, complete; `cpu/smoke-dr.json`).

## Run plan (`launch-sm145.sh`, one detached launcher; DRYRUN=1 for a dry run)

| step | where | expected |
|---|---|---|
| CPU proofs (`cpu-proof.sh`) | CPU, before the launch | about 12 min |
| GPU smoke: kernel self-test + 3 users x 6 models x 6 labels | 1 queue ticket | about 5 min |
| scoring Tiny, Small, Big: 723 users x 2 models x 5 labels, `--prior-buckets` | 3 queue tickets (+ short retry passes) | about 41 / 73 / 110 min (SM144, 6 labels: 49 / 88 / 132 min), about 3.9 GPU hours in all |
| long-card users: 50 users x 3 sizes x 2 models x 5 labels, with the dump | CPU, 2 processes at a time | about 9-10 h (SM144: 6 labels, 11.4 h) |
| per-review read-out (`sm145_dr.py aggregate`) | CPU | a few minutes |
| report | CPU | 1 min |

Expected speed: GPU about 130k (Tiny), 73k (Small), 48k (Big) review evaluations per second over the two models and five labels (SM144's rates); CPU about 1,000 to 1,650 review evaluations per second per process. The queue was empty at 12:55 and quiet.json has no window ahead, so the GPU chain should end about 17:00 to 17:30 Sun 11-22 and the CPU chain (the long pole) about 22:30 to 00:00.

## Expectation (written before scoring, not a rule)

From SM141's int3 increments (B4 vs AN at the 128 switch: +0.008% to +0.029% dataset-wide) on top of SM144's AN16 (+0.007% to +0.030%): Y4N likely passes R2 for Big and probably Small, and is at risk for the six-layer Tiny Instant-Only model (SM130) (about +0.06%); Y4C and Y2N should be cheaper, roughly half of Y4N's increment. R3 should look like AN16's (the long cards are int8 from 16 in every label). The per-card read-out has no earlier reference; int4 vs int8 is the baseline it is read against (AN16).

## Fork work if an int3 label is recommended (estimate; nothing changed in the fork here)

AN16 first (the codec choice at encode time by the entity's update count, about 0.5 to 1 agent-day, SM144). For an int3 label on top: a new Int3Compact codec (precision byte e.g. 0x13): `encode_matrix` / `decode_matrix` / `row_bytes` with 3-bit packing (8 codes in 3 bytes, codes -3..3), the same float16 row scale and int8 shifts; parity vectors from `sm145_bytes.py` (`pack` / `unpack`) and the scorer's `TierQuant`; tests at the tier boundaries (answer M int3, M+1 int4, 15 int4, 16 int8). About 1 to 1.5 agent-days on top of AN16.

## Files

- `labels.sh`, `models.sh` (the six release-candidate models), `users-smoke.json`, `users-longcard.json` (copies of SM144's).
- `sm145_dist.py` -> `sm145-dist.json` (answer-count distribution and tier shares for young cut-offs 2 / 4 / 8 and switch points 16 / 32 / 128; aggregates only), `sm145_bytes.py` -> `sm145-bytes.json`.
- `sm145_check.py` (CPU proofs, GPU self-test), `sm145_report.py` (smoke, rules, worth criterion, selftest), `sm145_dr.py` (per-review read-out, selftest), `cpu-proof.sh`, `launch-sm145.sh`, `memguard.ps1`.
- Outputs: `cpu/smoke-cpu-rows.jsonl`, `cpu/emulate-rows.jsonl`, `cpu/smoke-dr.json`, `check-cpu-sm145.json`, `gpu/` (tickets, smoke, rows of the 732 set), `cpu/longcard/` (one file per user and size), `cpu/sm145-longcard-rows.jsonl`, `sm145-dr.json`, `sm145-result.{json,md}`. Not committed: `work/sm145-dump/`, `work/sm145-dump-smoke/` (per-review values).

## Proof results

- `check-cpu-sm145.json`: **CPU proofs ALL OK** (36 checks, `cpu-proof.sh` 12:45 to 12:54 Sun 11-22; log `cpu-proof.out`). K1 bytes OK; K2 every synthetic identity bitwise (unchanged tier code, 3/4/8 tiers included); K3 on 3 users x 6 models x 9 labels (162 rows, all finite): U8 == REF, U4 == C4 and P4 == AN16 in every metric field, Y2N / Y4C / Y4N each differ from AN16 on every user, Y4N differs from Y4C and from Y2N; the GPU code path's CPU twin gives U8 == REF; K4 the REF and AN16 rows equal SM144's committed CPU smoke rows in every field (36 rows); K5 the dump: 162 files, the dumped reviews per bucket equal the rows' all-scope counts for every (user, model, label, head), equal review sets across labels, |dR| U8 vs REF exactly 0, `sm145_dr.py` aggregates the smoke dumps (`cpu/smoke-dr.json`, finite, complete).
- Smoke dumps: 4.7 MB for 162 files (3 small users); the long-card dumps should be about 1 to 2 GB in `work/sm145-dump/`.
- CPU speed (check only): 1,440 to 1,670 review evaluations per second per process with the dump (SM144 without it: 1,400 to 1,650), 1,860 to 2,160 on the emulated GPU path.
- Smoke dR on the 3 smoke users (8,481 predicted reviews per model; a proof of the read-out, not a result): AN16 P99 0.004 to 0.008, max 0.016 to 0.036; the int3 labels P99 0.007 to 0.015, max 0.018 to 0.077, no review above 0.10.
