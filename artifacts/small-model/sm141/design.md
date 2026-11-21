# Tiered card state by answer count (SM141): design

Pre-registered Sat 11-21 (about 18:00), before any scoring of the schemes. Branch `sm141`, worktree `C:/Users/jonat/workspace/wt/sm141`, cut from `origin/sm139` (2bae63e8), which holds the scorer with the compact formats of the compact state for Small and Big (SM131), the long-card tooling of the long-card state drift on real users (SM136) (`--prior-buckets`, `--nonfinite-rows`) and the release-candidate checks (SM139). No training: the six release-candidate models are scored as they are.

## Question

The fork stores, for every card and every note, the model's recurrent state. The release-candidate format the user chose on 11-21 stores int8 rows with float16 row scales and int8 shift vectors for cards and notes (scorer variant `w8+s8hx8`): Big 13,891 / 9,267 B per card / note, Small 2,545 / 1,315 B, Tiny 1,315 / 1,315 B. Int4 card rows (`w8+s4c8nhx8`, 7,747 / 1,521 / 803 B per card) were dropped because they cost LogLoss on reviews of cards with 128 or more earlier answers (the long-card state drift on real users (SM136)).

The user's idea (11-21): choose the precision per card by the card's own answer count. Young cards get a small state (int3, maybe int2), normal cards int4, and a card switches to int8 once it passes N answers (N = 128). Every write is decode, float32 update, encode, so a switch only encodes the fresh float32 state with more bits. The same may apply to notes by the note's answer count.

**Question in the user's terms: can young cards keep a smaller state and long cards int8, so that every card keeps a valid R and the predictions cost no measurable LogLoss against int8 everywhere?** The user's hypothesis (11-21): many cards are easy and rarely reviewed, so a lot of state could be compressed.

## What earlier experiments found about fewer bits

- No earlier experiment rounded the WKV state to 3 or 2 bits per value. The only values below 4 bits were the 1-bit norms of the codebook state (SM32), which cost most of its loss. 2- and 3-bit results exist for weights only.
- Int4 card rows with int8 notes (SM45, 723 users) cost about +0.02% on both heads; float16 row scales and int8 shift vectors (SM47) were free on top.
- Int4 notes and int4 shift vectors (SM49, 723 users): int4 notes about +0.03% curve / +0.06% instant; int4 shifts too about +0.04% / +0.11%; one scale per head instead of per row +0.22% / +0.39% (too coarse on the instant head). Int4 is the lowest width with an acceptable cost so far.
- Codebook state (SM32, SM44, SM46, SM90): 400-800 bits per card cost +0.4% to +1.2%; about 6,000 bits were nearly free but needed 12-30 MB of codebooks and were 4-11x slower. Not adopted.
- Drift of int4 on long cards: the older int4 drift check (SM78, 110 users) grew to +0.129% on the instant head for cards with 32-63 earlier answers. The long-card drift test (SM136, 51 users) found compact int4 cards at +0.18% (Big), +0.74% (Small), +0.65% (Tiny) on reviews of cards with 128+ earlier answers (mean, two-head curve), while int8 cards stayed within +0.12% (CI upper bound) everywhere. The old plain int4 layout with float32 scales overflowed on one user; the compact layouts saturate instead and stay finite.
- On whole pools, compact int4 cards were about +0.015% vs int8 (SM131, 722 users) and the release-candidate format was at most +0.002% vs the published int8 format (SM139, 1,487 users).

What this means for the tiers: int4 below 128 answers is expected to cost about what SM131 measured (+0.015%), because the cards where int4 hurt are exactly the ones switched to int8. Int3 and int2 are untested; young states have seen only a few updates, so their rounding error has had little time to accumulate, but they are read by a large share of reviews (next section).

## How the fork stores the state (read from `wt/fork-rc-state`, not changed)

- Each stored card or note state carries its own precision byte (`precision.rs`: 0x18 Int8Compact, 0x14 Int4Compact), so states of different widths can live side by side and decode without any other change.
- A card with no answer has no stored state: the card map has no entry and the model starts from a fresh zero state (`engine.rs` `state_ref`). Never-answered cards cost 0 bytes in every scheme.

## Answer counts on the 732 set (`sm141_dist.py` -> `sm141-dist.json`, aggregates only)

The 723 fit users of the 732 set, at the end of each user's history. A card's answers = its revlog rows (every revlog row is one answer and one state update; the scorer counts the same way). A note's answers = the answers of all its cards (the note state is updated on every answer of any of its cards). Only cards and notes with at least one answer are counted.

| entities | count | P10 | P25 | P50 | P75 | P90 | P95 | P99 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| answered cards, pooled | 6,136,137 | 2 | 3 | 5 | 8 | 14 | 20 | 42 | 7.0 |
| answered cards, per-user median of each percentile | 723 users | 2 | 3 | 5 | 8 | 11 | 14 | 22 | |
| answered cards still in the cards table (not deleted), pooled | 4,197,883 | 2 | 3 | 5 | 9 | 15 | 22 | 46 | |
| ... per-user median | | 2 | 4 | 5 | 8 | 11 | 14 | 22 | |
| answered notes, pooled | 5,461,876 | 2 | 3 | 5 | 9 | 16 | 24 | 50 | 7.8 |
| answered notes, per-user median | | 2 | 4 | 5 | 8 | 13 | 16 | 27 | |
| predicted reviews, by the card's earlier answers | 36,675,119 | 1 | 2 | 5 | 10 | 22 | 35 | 76 | |
| predicted reviews, by the note's earlier answers | | 1 | 3 | 6 | 14 | 29 | 45 | 101 | |

- Never-answered cards: 10,932,675 cards in the cards tables have no answer (64% of all cards; 4 users have no cards table). They store nothing.
- Deleted cards: 1,938,254 answered cards are in the revlog but no longer in the cards table. They are counted as cards (the scorer scores their reviews); for notes each counts as its own note. Their distribution is close to the others (table).
- Share of answered cards with at most k answers: 1: 9.8%, 2: 22.9%, 4: 46.6%, 8: 77.2%, 16: 92.9%, 32: 98.2%, 127: 99.96%. Cards with 128+ answers are 0.04% of cards, but 0.24% of predicted reviews read such a card's state.
- Share of predicted reviews whose card had at most k earlier answers: 1: 15.1%, 2: 28.0%, 4: 47.7%, 8: 69.4%, 16: 85.1%. So a young tier is not a niche: int3 for 1-4 answers touches about half of all predictions, int3 for 1-8 about 70%.

## Are low-answer cards easy, or recently added? (aggregates by the card's end-of-history answer count)

| answers | share of answered cards | lapsed at least once | only Good / Easy answers | first answer in the user's last 30 days | median age (days) | median interval before the last answer (days) | median days since the last answer |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.8% | 0.0% | 84.9% | 6.1% | 567 | new | 567 |
| 2 | 13.1% | 0.2% | 75.9% | 6.0% | 564 | 0 | 545 |
| 3-4 | 23.7% | 3.6% | 54.7% | 4.4% | 615 | 5 | 570 |
| 5-8 | 30.6% | 16.8% | 36.1% | 2.4% | 669 | 17 | 547 |
| 9-16 | 15.7% | 52.6% | 10.9% | 1.2% | 751 | 30 | 444 |
| 17-32 | 5.3% | 85.3% | 2.5% | 0.7% | 797 | 28 | 362 |
| 33-127 | 1.7% | 96.4% | 0.0% | 0.4% | 1,017 | 33 | 181 |
| 128+ | 0.04% | 99.1% | 0.0% | 0.1% | 1,653 | 38 | 32 |

Lapsed = an Again on a card in review state. The user's hypothesis holds in part. Low-answer cards are mostly easy (few lapses, mostly Good / Easy) and are not recently added: only about 6% had their first answer in the user's last 30 days. But they are also dormant: the median card with at most 8 answers was last answered about 1.5 years before the end of the history, after short intervals. They look like cards that were suspended, deleted or abandoned early, more than mature easy cards with long intervals. For storage this does not matter (they are stored either way); for accuracy it means their states are read rarely after the last answer, while the young states of the cards that are still being learned are read by about half of all predictions.

## Tier shares and bytes for a few cut-offs (cards and notes tiered alike; bytes = cards + notes of the 732 set vs int8)

| young tier | int8 from | cards in young / int4 / int8 tier | reviews reading a young / int4 / int8 card state | Big | Small | Tiny |
|---|---|---|---|---:|---:|---:|
| none | 64 | 0.0 / 99.7 / 0.33% | 0.0 / 98.5 / 1.54% | -44.0% | -39.7% | -38.8% |
| none | 128 | 0.0 / 100.0 / 0.04% | 0.0 / 99.8 / 0.24% | -44.2% | -39.8% | -38.9% |
| int3 for 1-2 | 128 | 22.9 / 77.1 / 0.04% | 28.0 / 71.8 / 0.24% | -46.7% | -42.1% | -41.1% |
| int3 for 1-4 | 128 | 46.6 / 53.4 / 0.04% | 47.7 / 52.0 / 0.24% | -49.3% | -44.4% | -43.4% |
| int3 for 1-8 | 128 | 77.2 / 22.8 / 0.04% | 69.4 / 30.4 / 0.24% | -52.6% | -47.4% | -46.3% |
| int3 for 1-16 | 128 | 92.9 / 7.0 / 0.04% | 85.1 / 14.6 / 0.24% | -54.4% | -49.0% | -47.8% |
| int2 for 1-2 | 128 | 22.9 / 77.1 / 0.04% | 28.0 / 71.8 / 0.24% | -49.2% | -44.3% | -43.3% |
| int2 for 1-4 | 128 | 46.6 / 53.4 / 0.04% | 47.7 / 52.0 / 0.24% | -54.4% | -49.0% | -47.8% |

(int8 from 64 instead of 128 changes the bytes by 0.1-0.2%; the full grid is in `sm141-dist.json` "tiers".) Most of the saving comes from int4 below 128 answers (-39% to -44%). A young int3 tier adds 2.5% (1-2 answers) to 8% (1-8 answers) more.

## Pre-registered schemes and cut-offs (chosen from the tables above, before any scoring)

- **N = 128 for int8, one value.** It is the bucket where SM136 measured the int4 damage, and moving it to 64 saves only 0.1-0.2% more bytes; with N = 128 only 0.04% of cards are int8, so the int8 tier costs almost nothing.
- **Young cut-offs M = 4 and M = 8.** M = 4 is just below the median card (P50 = 5 answers; 47% of cards, 48% of reviews), M = 8 is the third quartile (P75 = 8 to 9; 77% of cards, 69% of reviews). M = 2 (P25) saves only about 2.5% and M = 16 (P90) puts 85% of reviews on int3, so neither is tested.
- **int3 is the primary young width** (the user's 11-21 request); **int2 is secondary**, at M = 4 only, so it compares directly with int3 at the same cut-off. Int2 here is ternary: codes -1, 0, +1 times the row scale (symmetric, like int8 at ±127 and int4 at ±7; the fourth 2-bit code is unused).
- Shift vectors stay int8 (with one float16 scale) in every tier; only the WKV matrix rows change width.

| scheme | cards | notes | variant (scorer) |
|---|---|---|---|
| REF | int8 | int8 | `w8+s8hx8` (the release-candidate format) |
| A | int4 below 128 answers, int8 from 128 | int8 | `w8+t4.128.8c8nhx8` |
| AN | int4 below 128, int8 from 128 | same as cards | `w8+t4.128.8c4.128.8nhx8` |
| B4 | int3 for 1-4, int4 for 5-127, int8 from 128 | same as cards | `w8+t3.5.4.128.8c3.5.4.128.8nhx8` |
| B8 | int3 for 1-8, int4 for 9-127, int8 from 128 | same as cards | `w8+t3.9.4.128.8c3.9.4.128.8nhx8` |
| D4 | int2 for 1-4, int4 for 5-127, int8 from 128 | same as cards | `w8+t2.5.4.128.8c2.5.4.128.8nhx8` (secondary) |

Notes can be tiered: the scorer counts note answers the same way as card answers (below), and the fork updates the note state on every answer of the note's cards. The young schemes tier cards and notes together; A vs AN shows what tiering notes costs on its own.

### How the answers are counted (scorer = what the fork can see)

The answer count n of a stored state is the number of updates it has received: in each card (or note) stream of the scorer, the number of non-skipped steps so far, the current one included (skipped steps are the show-time query rows, which read the state but never write it). After the n-th answer the fresh float32 state is encoded at width young if n <= M, old if n >= N, else mid. A review whose card had p earlier answers reads the state written after update p, so its tier is the tier of p. For a card this n equals the card's revlog rows so far (`card_prior_answers`, SM136's check K5); for a note it is the answers of all its cards so far, by the scorer's note assignment (deleted cards: `--deleted-cards neighbour`).

## Bytes per entity and tier (`sm141_bytes.py` -> `sm141-bytes.json`)

Compact layout (the fork's `ModuleState`): header 19 B, 12 B lengths per layer, each shift vector 2 + D bytes (int8 in every tier), each matrix row 2 + ceil(K x bits / 8) bytes (float16 scale + packed codes: int3 packs 8 codes in 3 bytes, int2 4 codes per byte). The formula reproduces SM136's encoded sizes for int8 and int4 exactly; 3- and 2-bit packing round-trips bitwise.

| size | card int8 / int4 / int3 / int2 | note int8 / int4 / int3 / int2 |
|---|---|---|
| Big (4 heads x 32, card 3 layers, note 2 layers) | 13,891 / 7,747 / 6,211 / 4,675 | 9,267 / 5,171 / 4,147 / 3,123 |
| Small (4 x 16, card 2 layers, note 1) | 2,545 / 1,521 / 1,265 / 1,009 | 1,315 / 803 / 675 / 547 |
| Tiny (4 x 16, card 1 layer, note 1) | 1,315 / 803 / 675 / 547 | 1,315 / 803 / 675 / 547 |

Expected bytes of each scheme on the 732 set (end-of-history counts; mean bytes per answered card / note; total cards + notes vs REF):

| scheme | Big | Small | Tiny |
|---|---|---|---|
| REF | 13,891 / 9,267, 0% | 2,545 / 1,315, 0% | 1,315 / 1,315, 0% |
| A | 7,749 / 9,267, -27.7% | 1,521 / 1,315, -27.6% | 803 / 1,315, -20.6% |
| AN | 7,749 / 5,174, -44.2% | 1,521 / 803, -39.8% | 803 / 803, -38.9% |
| B4 | 7,034 / 4,713, -49.3% | 1,402 / 746, -44.4% | 744 / 746, -43.4% |
| B8 | 6,563 / 4,416, -52.6% | 1,324 / 709, -47.4% | 704 / 709, -46.3% |
| D4 | 6,318 / 4,252, -54.4% | 1,283 / 688, -49.0% | 684 / 688, -47.8% |

Note on totals: the 723 users together hold 6.1M answered cards and 5.5M answered notes, so REF is 135.9 GB for Big, 22.8 GB for Small and 15.3 GB for Tiny over all 723 collections (about 190 MB, 32 MB and 21 MB per average user). The fork may need an update counter per entity to choose the tier (see "Fork work"); a 2-byte counter would add 0.02-0.4% and is not included.

## Scorer support (this branch; additive, default output unchanged)

- `scripts/small_model_wkv.py`: `TIER` flag (1024) on the state bits, `tier_bits(young, M, mid, N, old)` packs the tiers into the same int (low byte = mid width, so every existing `bits & 255` stays valid); a separate CUDA kernel `wkv_q_tier` built from the same text counts the non-skipped steps per (row, head) and picks qmax 127 / 7 / 3 / 1 per write; the CPU `reference` does the same. Without the flag the stock kernel and code run unchanged.
- `scripts/state_vq.py`: `TierQuant` for the CPU (Hooks) path of the Small and Tiny models; `_reference_wkv_q` passes each row's update count to a tiered quantizer only.
- `scripts/compress_quant.py`: the patched fork model (Big) gets `_quantize_tier` and counts updates per row in `reference_rwkv7_q` on the CPU; on the GPU the fork model already routes flagged bits to `wkv_long`, so it uses the same tier kernel.
- `scripts/small_model_heldout.py`: variant grammar `w8+t<card spec>c<note spec>nhx8`, spec `b0`, `b0.N.b1` or `b0.T.b1.N.b2` (b0 below T answers, b1 from T, b2 from N); always the tier code path, so a uniform spec tests the tier code itself; rows get `state_tiers`.

## Proofs (CPU, before any GPU ticket; `cpu-proof.sh`, `sm141_check.py cpu` -> `check-cpu-sm141.json`)

- K1 bytes: the table above (formula = SM136's encoder sizes; 3- / 2-bit pack round trip).
- K2 synthetic, seeded: with every tier at one width, the tier code equals the stock code bit for bit (state_vq `TierQuant` vs `int_row_quant`, the fork `_quantize_tier` vs `_quantize_rows`, `small_model_wkv.reference` with TIER bits vs the plain flags, the fork `reference_rwkv7_q` TIER vs plain, K 16 and 32, int8 and int4; zero, tiny and huge rows); with real tiers every row gets the width of its own update count (manual per-step loop, bitwise); the fork's tiered recurrence equals `small_model_wkv.reference`'s bit for bit.
- K3 scorer rows, the 3 smoke users (2388, 4676: the two smallest of the 732 fit list; 1866: 5,673 reviews with a card of 128+ answers) x the 6 models, CPU: U8 (`w8+t8.5.8.128.8c8.5.8.128.8nhx8`) equals REF in every metric field; U4 (`w8+t4.5.4.128.4c8.5.8.128.8nhx8`) equals compact int4 `w8+s4c8nhx8`; A equals `w8+s4c8nhx8` exactly when no review reads a state of 128+ updates and differs when one does (the threshold works); B4 / B8 / D4 differ from AN (the young tier works); the GPU code path's CPU twin (`--emulate-gpu-path`, Small and Tiny) also gives U8 == REF.
- K4: this tree's REF CPU rows equal the committed CPU rows of the release-candidate checks (SM139) for the common users.
- Result: see "Proof results" at the end (filled in after the run, before the launch).

## GPU or CPU per user (decision)

- The GPU smoke of SM136 agreed with the CPU on the benchmark LogLoss within 4.4e-5 even on its long-card user 1866, but the all-scope means of the 128+ bucket differed by up to 1.85e-3 for the quantized formats (limit 1e-4). This is the known code-flip noise of discrete codecs (AGENT.md "Discrete state codecs"): a float32 summation-order difference flips a rounding, and the recurrence carries the flip along a long card.
- **Decision:** the 732 set (723 users, the dataset-wide rule R2 and the young-card rule R4) runs on the GPU: these readings are averages over many reviews, where flips have random sign, and the smoke checks them. The **long-card rule R3 runs on the CPU** on the 50 long-card users (`users-longcard.json`: the users of the 732 set with a card of 128+ answers that SM136 scored completely under the 5.9 GB memory guard; the same list SM137 used), because that bucket is where GPU flips are amplified. The CPU path is the reference arithmetic (sequential float32 per row, like the fork). The report also lists the GPU-vs-CPU difference on the 128+ bucket for these users (report only).
- GPU smoke rule (first ticket, STOP on failure): the tier kernel with every tier at one width equals the stock kernel bit for bit (random inputs, K 16 and 32) and U8 equals REF exactly in the GPU rows; REF and U8 GPU vs CPU within 1e-5 on every (user, head); the tiered labels by the discrete-codec rule (per user |Δ benchmark LogLoss| <= 3e-4, review-weighted mean signed Δ <= 1e-4 in absolute value, per head). Using the discrete rule for the int3 / int2 tiers extends the 10-21 rule (approved for int4 notes and shifts in SM49) to coarser codes, for which the strict 1e-5 rule already fails at int4; flagged for the user (open question 3).

## Rules (`sm141_report.py`, selftest passes; fixed now)

Statistic: % = 100 x (sum of the scheme's LogLoss / sum of REF's LogLoss - 1), review-weighted; paired user bootstrap, 2,000 draws, seed 0, 95% CI. Heads: two-head models curve and instant, Instant-Only models instant.

- **R1 finite:** no non-finite row and no non-finite LogLoss for the scheme, on the 732 set (GPU) and on the long-card users (CPU).
- **R2 dataset-wide:** benchmark LogLoss on the 722 rule users of the 732 set (`showtime_report.compare`, as every 732 readout), CI upper bound <= **+0.05%** per head. Why: the same ceiling as the release-candidate format check (SM139 C2) and the repo's minimum effect (0.05%); int4 below 128 is expected near SM131's +0.015%, so the ceiling leaves room for noise but not for a real step back.
- **R3 long cards:** on the long-card users (CPU), the reviews whose card had 128 or more earlier answers (all scope), CI upper bound <= **+0.15%** per head. Why: in that bucket int8 cards themselves reached +0.12% (CI upper bound) against the published int8 format (SM136, Small two-head), which is the resolution of 50 users and about 30,000 reviews; compact int4 cost +0.74% (point). A scheme whose cards are int8 from 128 answers should sit at the int8 noise level ("no extra cost beyond int8's"), and +0.15% rejects anything near int4's damage. CPU labels: AN, B8, D4 (and REF). A and B4 are not scored there: at every update they store each state at the same or a higher width than AN (A: int8 notes) or B8 (B4: int4 instead of int3 for answers 5-8), so their R3 is taken from AN and B8 (precision dominance, a pre-registered proxy; measuring them is a 2-label CPU add-on of about 3 h).
- **R4 young cards (B4, B8, D4):** on the 732 set (GPU), the reviews whose card had 1 to M earlier answers (all scope; M = 4 or 8), CI upper bound <= **+0.10%** per head. Why: these reviews are about half (M = 4) to 70% (M = 8) of all predictions, so a cost concentrated there would show at about half its size in R2; +0.10% keeps the young bucket consistent with R2's +0.05% and catches a cost the average would hide.
- **R5 coverage:** at least 720 of the 722 rule users for REF and the scheme on the GPU, and at least 48 of the 50 long-card users on the CPU; otherwise the report is `pending` (exit 2) and the main session decides.
- **Go per size:** a scheme passes for a size when it passes R1-R4 for both models of the size (two-head and Instant-Only). The size takes the passing scheme with the smallest total card + note bytes on the 732 set (order for every size: D4 < B8 < B4 < AN < A); if none passes, REF stays.
- Reported, no verdict: every bucket (1-2, 3-4, 5-8, 9-16, 17-32, 33-127, 128+) per scheme on the GPU rows, the all-scope total, REF vs SM139's GPU rows (integrity), GPU vs CPU on the long-card users, CPU speed (review evaluations per second in the logs, a check only: the fork's encode cost per tier is not measured here). Training is unaffected (no model is retrained).

## Run plan (`launch-sm141.sh`, one detached launcher; DRYRUN=1 for a dry run)

| step | where | expected |
|---|---|---|
| CPU proofs (`cpu-proof.sh`) | CPU, now | ~15 min |
| GPU smoke: kernel self-test + 3 users x 6 models x 7 labels | 1 queue ticket | ~5-15 min |
| scoring Tiny, Small, Big: 723 users x 2 models x 6 labels, `--prior-buckets` | 3 queue tickets (+ short retry passes) | ~60 / 80 / 130 min (SM139: 31 / 42 / 43 min for 6 / 6 / 4 labels) |
| long-card users: 50 users x 3 sizes x 2 models x 4 labels | CPU, 3 processes at a time, < 5.9 GB each, started only when the free RAM fits | ~6-8 h |
| report | CPU | 1 min |

- The GPU is busy with the rolling recalibration for the six-layer Tiny (SM140) until about 19:30; the queue is FIFO, so the smoke ticket waits its turn. GPU done about 00:30 Sun; the CPU chain starts right away and ends about 01:00-03:00 Sun.
- The extension pool (771 users) is not scored: it would add about 5 GPU hours for a report-only reading.

## Fork work if a scheme passes (estimate; nothing changed in the fork here)

- **A / AN (int4 + int8 only):** the codecs exist (0x14 / 0x18). The work is the choice at encode time: card (and note) state written with Int4Compact below 128 answers and Int8Compact from 128. The encoder needs the entity's update count: either the replay counts it (cards: revlog entries replayed; notes: answers of the note's cards), or the state carries a counter (a u16 or u32 in the header, +2-4 B). Plus tests: the boundary (answer 127 int4, 128 int8), decode of mixed collections, the bulk path, SM133's synthetic long-card probe. About 0.5-1 agent-day.
- **B4 / B8 (int3):** a new Int3Compact codec (precision byte e.g. 0x13): `encode_matrix` / `decode_matrix` / `row_bytes` with 3-bit packing (8 codes in 3 bytes, codes -3..3), the same float16 row scale and int8 shifts; parity vectors from `sm141_bytes.py` (`pack` / `unpack`) and from the scorer's `TierQuant`. About 1-1.5 agent-days on top of A.
- **D4 (int2):** the same for Int2Compact (0x12, ternary codes, 4 per byte). About 0.5 agent-day more if int3 is done.

## Open questions for the user (recommendations; nothing in the run waits on them)

1. **Int2 as ternary.** Symmetric 2-bit codes use only -1, 0, +1 (the fourth code is unused), consistent with the int4 / int8 layouts. An asymmetric -2..+1 would use all four codes but break the symmetric rule of every other width. Recommendation: keep ternary for this test; only revisit if D4 is close to passing.
2. **A and B4 long-card check by dominance.** R3 for A and B4 comes from AN and B8. Recommendation: accept; if A or B4 becomes the pick, run the 2-label CPU add-on (about 3 h) before the fork work.
3. **The discrete-codec GPU rule for int3 / int2 tiers.** The smoke applies the 10-21 discrete rule (3e-4 per user, 1e-4 mean) to the tiered labels; the strict 1e-5 rule stays for REF and U8. Recommendation: approve, since int4 already needs it.
4. **Notes of deleted cards.** In the distribution each deleted card counts as its own note (the scorer's neighbour rule is not reproduced there), which slightly inflates the note count; the scored rows use the scorer's own assignment.

## Files

- `sm141_dist.py` -> `sm141-dist.json` (answer-count distribution, tiers, easy vs recent; aggregates only), `sm141_bytes.py` -> `sm141-bytes.json`.
- `labels.sh`, `models.sh` (the six release-candidate models of SM139, role rc), `users-smoke.json`, `users-longcard.json` (copy of SM137's list).
- `sm141_check.py` (CPU proofs, GPU self-test), `sm141_report.py` (smoke, rules, selftest), `cpu-proof.sh`, `launch-sm141.sh`, `memguard.ps1`.
- Outputs: `cpu/smoke-cpu-rows.jsonl`, `cpu/emulate-rows.jsonl`, `check-cpu-sm141.json`, `gpu/` (tickets, smoke, rows of the 732 set), `cpu/longcard/` (one file per user and size), `sm141-result.{json,md}`.

## Proof results (11-21 17:35, before the launch)

- `check-cpu-sm141.json`: **CPU proofs ALL OK** (28 checks). K1 bytes OK; K2 every synthetic identity bitwise (tier code with one width == stock code on all four paths; real tiers pick each row's width; the fork's tiered recurrence == `small_model_wkv.reference`); K3 on 3 users x 6 models (162 rows, all finite): U8 == REF and U4 == compact int4 in every metric field, A == compact int4 except on the long-card user (threshold works), the young tiers change the rows, and the GPU code path's CPU twin gives U8 == REF too (its tiered rows equal the Hooks path's to 0.0 in benchmark LogLoss); K4 the REF rows equal SM139's committed CPU rows (12 rows).
- One check was first reported as failed: the emulated path's rows carry a descriptor field (`state_vq_path`, the bits it ran) that differs by construction; it was added to the skipped descriptor fields (like `variant`), the metrics were already equal. No source change.
- Launcher dry run: `gpu/launch-sm141-dryrun.log` (DRYRUN OK: 1 smoke + 3 scoring tickets, 3 x 50 long-card users).
- CPU speed (check only): the CPU smoke ran at 1,400-1,650 review evaluations per second per process (4 threads), 1,800-2,100 on the emulated GPU path for Small and Tiny. Model seconds per label on the 3 smoke users are within noise of the int8 and int4 labels (Small two-head 9.2-9.6 s, Tiny 3.6-3.9 s, Big two-head 12.6-14.2 s; the first Big label, 17.0 s, includes the TorchScript warm-up). This is the scorer's emulation, not the fork's encoder.
