# Deleted-card leak check (SM71)

Question (user, 2026-10-24 ~17:05, after the feature discovery SM70): the dataset tells both models which cards were deleted before the export. How much does each model rely on it, and does the small model's gap to the big model change once it is removed? Models: the big model (the fork's 2.76M RWKV, `fork/w8+s8`, the reference) and the current best small model, the full retrain's decay from epoch 10 (SM56, `scale-w64kd-final-sc-lanes-dec10-1ep32k-muon`, w64, w8+s8); the 8k model (SM55) on GPU too.

## Result so far

CPU estimate only (GPU scoring on the 464 set is queued, see below). Both models use the deleted-card signals, the big model a little more, and removing them barely moves the small model's gap to the big model:

- View 1, inference without the leak (40 users, 1.30 M test reviews, 22% on deleted cards). Flags set to 0: instant LogLoss rises by +0.08% for the big model (+0.000255 [+0.000118, +0.000429]) and +0.05% for the small model (+0.000138 [+0.000073, +0.000221]), all of it on the deleted-card reviews (+0.42% and +0.23% there); curve change about 0 with wide intervals. Flags 0 and the cards moved into the nearest existing card's deck (`neighbour`): big +0.20% curve (CI includes 0) / +0.15% instant, small +0.15% / +0.13%; part of this is on existing cards, whose real deck lanes now carry the deleted cards (the reseed control shows the re-drawn id codes alone move LogLoss by at most 0.00003).
- Gap (small vs big, % vs big model): today curve -0.10% [-1.15, +0.56], instant +0.35% [+0.27, +0.46]. Flags 0: curve -0.07%, instant +0.31% (change -0.04 points [-0.08, -0.01]). Neighbour: curve -0.14% (-0.05 [-0.13, +0.02]), instant +0.33% (-0.02 [-0.07, +0.02]).
- View 2, existing cards only (no input surgery; 200 users, 7.06 M test reviews, from SM70's dumps): the gap widens slightly, curve +0.11% -> +0.28% (+0.17 points [+0.02, +0.34]), instant +0.44% -> +0.47% (+0.03 [+0.01, +0.07]); on deleted cards alone the small model is relatively better (curve -0.49%, instant +0.31%).
- So: the leak is worth about 0.1-0.2% of LogLoss to each model, and the small-vs-big gap moves by at most about 0.2 points under either view (instant: -0.04 to +0.03 points). Numbers are a 40-user / 200-user CPU estimate; the GPU rows on 458 users replace them.

## The leak

`rwkv/data_processing.get_rwkv_data` merges the revlogs with the export's cards table (`card_id -> note_id, deck_id`) and decks table (`deck_id -> preset_id`; `parent_id` is dropped). A card deleted before the export is missing from the cards table. In live Anki a card always exists at review time, so every input that differs for these cards is future information. 21.4% of the benchmark test reviews of SM70's 200 users (1,507,382 of 7,058,890), in 85.5% of those users, are on such cards; one user (4789) has 97% of its reviews on deleted cards.

Inventory of every input that differs for a review of a deleted card (all other inputs come from the revlogs alone and are unchanged: ratings, durations, states, elapsed times, new-card and review counters, day offsets, card id, user id):

| input | deleted card gets | a normal card gets | can a deployed model restore it? | `flags` mode | `neighbour` mode |
|---|---|---|---|---|---|
| `note_id_is_nan` (card feature, on review and query rows) | 1 | 0 | yes (0) | 0 | 0 |
| `deck_id_is_nan` | 1 | 0 | yes | 0 | 0 |
| `preset_id_is_nan` | 1 | 0 | yes | 0 | 0 |
| note id (note RWKV lane, random 12-dim id code) | a note of its own (`ID_PLACEHOLDER + card_id`) | its real note, shared with sibling cards | no: the real note and siblings are lost; 20% of reviews are on notes with several cards (SM70 inventory) | unchanged | unchanged (singleton note) |
| deck id (deck lane, 8-dim code) | ONE placeholder deck shared by all deleted cards of the user, across their real decks | its real deck | no; stand-in: the deck of the existing card nearest by card_id (creation time) | unchanged | nearest existing card's deck |
| preset id (preset lane, 8-dim code) | ONE placeholder preset | its deck's preset | as the deck | unchanged | that deck's preset |
| random id/day codes | drawn over the set of ids | same | n/a | bit-identical (ids unchanged) | re-drawn: removing the placeholder deck shifts the code draws of decks, presets and day offsets (a reseed); measured by the `reseed` control |

So the leak is more than a flag: the deleted cards of a user also share one deck and one preset lane of their own, i.e. the model gets a running "deleted-cards deck" whose state summarises exactly the cards that will be deleted (recalled more often: test reviews 0.870 vs 0.857, SM70), and those cards are missing from their real deck lanes. The deck of a deleted card cannot be recovered from the export (revlogs have no deck), and a whole deleted deck really did hold only those cards, so no stand-in is exact: `flags` removes only the flag, `neighbour` also puts the cards into a plausible real deck. 117 of 10.1 M reviews are on existing cards whose deck is missing from the decks table: flag set to 0 in both modes, placeholder preset kept.

## Method

- Scorer `scripts/small_model_heldout.py` (opt-in, default off = today's rows): `--deleted-cards keep|flags|neighbour` (transform in `scripts/sm71_leak.py`, applied to the `get_rwkv_data` frame before the batch is built), `--seed-offset N` (id/day codes drawn with seed 2025 + user + N; noise control for `neighbour`), `--split-existing` (adds `benchmark_existing` and `benchmark_deleted`: the benchmark metrics restricted to test reviews of cards in / missing from the cards table).
- View 1 (inference without the leak): `neighbour` (and `flags`) rows, all test reviews, vs the same model's `keep` rows. View 2 (no input surgery): `keep` rows, existing-card reviews only.
- Report `scripts/sm71_report.py`: review-weighted LogLoss (curve and instant heads), each model's change vs its own `keep`, the small-vs-big gap (% vs big model) per view and its change vs today's all-review gap, user bootstrap 2,000 resamples (seed 0), 95% intervals on paired differences.

## Checks (CPU)

- Default off: SM56 on 2 users (4676, 1745) with the SM71 scorer gives rows identical in every field (except timings and provenance) to SM70's CPU rows of the same checkpoint (`checks/off-sm56.jsonl`).
- Inputs (`scripts/sm71_check.py`, users 4676, 516, 4789, 1058; `check-inputs.json`): `flags` changes only the three flag columns, only on rows (reviews and queries) of deleted cards; ids identical; the prepared batch differs in exactly 3 entries per deleted-card row (582 rows -> 1,746 entries for user 4676), everything else bit-identical. `neighbour`: same feature changes; card, note and user ids identical; deck and preset ids change only on deleted-card rows (e.g. user 516: 490 deleted cards over 15 neighbour decks; user 1058: all 607 into one deck). Counts per user: 4676 353 of 3,716 reviews deleted, 516 1,711 of 4,601, 4789 4,934 of 5,102, 1058 2,380 of 5,224.
- Spawned data-loader workers (the GPU path) build `neighbour` batches (one-user test); the runbook was dry-run with the queue replaced by `echo` (18 tickets, commands and expected minutes as listed below).

## CPU estimate

View 1 on CPU: the first 40 users of the SM56 458-user list (stratified over history-length deciles; SM70's part 0), modes `flags`, `neighbour` and the `reseed` control (keep inputs, `--seed-offset 1`), 6 processes x 2 threads, about 35 min for the small model and 50 min for the big model (`cpu/run-cpu.sh`, `cpu/run-cpu-reseed.sh`). Baselines (`keep`): SM70's CPU dump of SM56 (`wt/sm70/artifacts/small-model/sm70/dump`) and the SM24 GPU dump of the big model (`wt/small-model/artifacts/small-model/sm24-dump`; GPU and CPU agree within about 1e-5 per user). Rows `cpu/*.jsonl`, report `cpu/report-40.md` / `.json`:


#### LogLoss

| series | all curve | all instant | existing curve | existing instant | deleted curve | deleted instant |
|---|---|---|---|---|---|---|
| big/keep | 0.33237 | 0.302404 | 0.33438 | 0.306713 | 0.325167 | 0.286957 |
| small/keep | 0.332049 | 0.303461 | 0.334765 | 0.307891 | 0.322313 | 0.287581 |
| big/flags | 0.332247 | 0.302659 | 0.33438 | 0.306706 | 0.324605 | 0.288152 |
| small/flags | 0.332004 | 0.303599 | 0.334738 | 0.307887 | 0.322203 | 0.288233 |
| big/neighbour | 0.33303 | 0.302859 | 0.335376 | 0.306842 | 0.324622 | 0.288584 |
| small/neighbour | 0.332553 | 0.303845 | 0.335353 | 0.308077 | 0.322519 | 0.288676 |
| big/reseed | 0.332375 | 0.302422 | 0.334377 | 0.306735 | 0.3252 | 0.286965 |
| small/reseed | 0.332065 | 0.303466 | 0.334793 | 0.307913 | 0.322287 | 0.287526 |

#### Change vs the same model with today's inputs (delta [95% CI], % of today)

| series | subset | curve | instant |
|---|---|---|---|
| big/flags | all | -0.000123 [-0.001358, +0.000626] (-0.04%) | +0.000255 [+0.000118, +0.000429] (+0.08%) |
| big/flags | deleted | -0.000562 [-0.008541, +0.002857] (-0.17%) | +0.001195 [+0.000907, +0.001628] (+0.42%) |
| big/flags | existing | +0.000000 [-0.000099, +0.000083] (+0.00%) | -0.000007 [-0.000051, +0.000019] (-0.00%) |
| big/neighbour | all | +0.000660 [-0.000189, +0.001584] (+0.20%) | +0.000456 [+0.000246, +0.000670] (+0.15%) |
| big/neighbour | deleted | -0.000545 [-0.010099, +0.003555] (-0.17%) | +0.001627 [+0.001291, +0.002269] (+0.57%) |
| big/neighbour | existing | +0.000996 [+0.000020, +0.002286] (+0.30%) | +0.000129 [+0.000006, +0.000292] (+0.04%) |
| big/reseed | all | +0.000005 [-0.000116, +0.000128] (+0.00%) | +0.000019 [-0.000004, +0.000041] (+0.01%) |
| big/reseed | deleted | +0.000033 [-0.000230, +0.000484] (+0.01%) | +0.000007 [-0.000020, +0.000057] (+0.00%) |
| big/reseed | existing | -0.000003 [-0.000137, +0.000135] (-0.00%) | +0.000022 [-0.000004, +0.000046] (+0.01%) |
| small/flags | all | -0.000045 [-0.000523, +0.000224] (-0.01%) | +0.000138 [+0.000073, +0.000221] (+0.05%) |
| small/flags | deleted | -0.000110 [-0.002863, +0.001135] (-0.03%) | +0.000651 [+0.000463, +0.000975] (+0.23%) |
| small/flags | existing | -0.000027 [-0.000060, +0.000005] (-0.01%) | -0.000005 [-0.000018, +0.000007] (-0.00%) |
| small/neighbour | all | +0.000505 [-0.000214, +0.001375] (+0.15%) | +0.000384 [+0.000161, +0.000653] (+0.13%) |
| small/neighbour | deleted | +0.000206 [-0.002991, +0.001891] (+0.06%) | +0.001095 [+0.000670, +0.001913] (+0.38%) |
| small/neighbour | existing | +0.000588 [-0.000026, +0.001567] (+0.18%) | +0.000186 [+0.000019, +0.000427] (+0.06%) |
| small/reseed | all | +0.000016 [-0.000068, +0.000138] (+0.01%) | +0.000005 [-0.000011, +0.000023] (+0.00%) |
| small/reseed | deleted | -0.000026 [-0.000207, +0.000275] (-0.01%) | -0.000055 [-0.000096, -0.000019] (-0.02%) |
| small/reseed | existing | +0.000028 [-0.000054, +0.000131] (+0.01%) | +0.000021 [+0.000000, +0.000049] (+0.01%) |

#### Small vs big model (% vs big model [95% CI]; change vs today's all-review gap in points)

| pair / inputs | subset | curve gap | instant gap | curve change | instant change |
|---|---|---|---|---|---|
| small-big/flags | all | -0.07% [-0.89, +0.48] | +0.31% [+0.22, +0.43] | +0.02 [-0.13, +0.27] | -0.04 [-0.08, -0.01] |
| small-big/flags | deleted | -0.74% [-2.89, +0.08] | +0.03% [-0.16, +0.21] | -0.64 [-2.02, -0.16] | -0.32 [-0.51, -0.16] |
| small-big/flags | existing | +0.11% [-0.57, +0.69] | +0.39% [+0.30, +0.51] | +0.20 [-0.05, +0.64] | +0.04 [-0.01, +0.10] |
| small-big/keep | all | -0.10% [-1.15, +0.56] | +0.35% [+0.27, +0.46] | - | - |
| small-big/keep | deleted | -0.88% [-4.48, +0.42] | +0.22% [+0.06, +0.41] | -0.78 [-3.62, +0.14] | -0.13 [-0.30, +0.03] |
| small-big/keep | existing | +0.12% [-0.54, +0.67] | +0.38% [+0.30, +0.50] | +0.21 [-0.03, +0.65] | +0.04 [-0.01, +0.10] |
| small-big/neighbour | all | -0.14% [-1.15, +0.49] | +0.33% [+0.23, +0.44] | -0.05 [-0.13, +0.02] | -0.02 [-0.07, +0.02] |
| small-big/neighbour | deleted | -0.65% [-2.42, +0.14] | +0.03% [-0.17, +0.24] | -0.55 [-1.61, -0.03] | -0.32 [-0.53, -0.12] |
| small-big/neighbour | existing | -0.01% [-0.94, +0.72] | +0.40% [+0.32, +0.53] | +0.09 [-0.07, +0.29] | +0.05 [-0.00, +0.14] |
| small-big/reseed | all | -0.09% [-1.12, +0.53] | +0.34% [+0.27, +0.45] | +0.00 [-0.04, +0.05] | -0.01 [-0.01, +0.01] |
| small-big/reseed | deleted | -0.90% [-4.56, +0.42] | +0.20% [+0.04, +0.38] | -0.80 [-3.70, +0.13] | -0.15 [-0.31, +0.01] |
| small-big/reseed | existing | +0.12% [-0.49, +0.65] | +0.38% [+0.30, +0.50] | +0.22 [-0.04, +0.70] | +0.03 [-0.01, +0.10] |


View 2 on CPU from the same dumps, all 200 SM70 users (`cpu/view2-200.md`):

#### LogLoss

| series | all curve | all instant | existing curve | existing instant | deleted curve | deleted instant |
|---|---|---|---|---|---|---|
| big/keep | 0.297819 | 0.270512 | 0.297348 | 0.270947 | 0.29955 | 0.26891 |
| small/keep | 0.298148 | 0.271692 | 0.298168 | 0.272222 | 0.298074 | 0.269741 |

#### Change vs the same model with today's inputs (delta [95% CI], % of today)

| series | subset | curve | instant |
|---|---|---|---|

#### Small vs big model (% vs big model [95% CI]; change vs today's all-review gap in points)

| pair / inputs | subset | curve gap | instant gap | curve change | instant change |
|---|---|---|---|---|---|
| small-big/keep | all | +0.11% [-0.25, +0.42] | +0.44% [+0.36, +0.53] | - | - |
| small-big/keep | deleted | -0.49% [-1.40, +0.11] | +0.31% [+0.23, +0.40] | -0.60 [-1.34, -0.08] | -0.13 [-0.23, -0.02] |
| small-big/keep | existing | +0.28% [-0.10, +0.61] | +0.47% [+0.37, +0.59] | +0.17 [+0.02, +0.34] | +0.03 [+0.01, +0.07] |


## Queued on GPU

Runbook `gpu/run-sm71.sh` (log `gpu/run-sm71.log`, scorer output `gpu/run-sm71.out`), launched detached and hidden; it waits for the SM56 launcher's epoch-14 scoring (or 20:30), checks `quiet.json` before every enqueue, and enqueues one queue ticket per model and mode, all `--split-existing`, outputs `gpu/<model>-<mode>.jsonl` (new files only):

| order | model | inputs | users | expected min |
|---|---|---|---|---|
| 1-2 | big, small (SM56) | keep (gives view 2 and the baselines) | 464 set (458) | 20, 10 |
| 3-4 | big, small | neighbour | 458 | 20, 10 |
| 5-6 | big, small | flags | 458 | 20, 10 |
| 7-8 | big, small | reseed (seed offset 1) | 458 | 20, 10 |
| 9-10 | 8k model (SM55) | keep, neighbour | 458 | 10, 10 |
| 11-18 | big, small | keep, neighbour | 732 pool: 265 more fit users, then the 3 long users at the 0.92 cap | 14 / 8, 13 / 8 |

Big-model time: 909 s of model time for 464 users on GPU (SM24 rows), about 17-20 min per ticket with startup. About 140 min for the 464 set: from about 19:25, roughly tickets 1-6 finish before the 21:00 break; the queue holds the rest until 01:30. Done lines: `[sm71 ...] 464 set done`, then `[sm71 ...] done`. Read-out commands: `HANDOFF-sm71.md`.

## Files

`scripts/sm71_leak.py`, `scripts/sm71_check.py`, `scripts/sm71_report.py`, scorer options in `scripts/small_model_heldout.py`; `check-inputs.json`, `checks/` (default-off and smoke rows), `cpu/` (CPU runs, `run-cpu.sh`, `run-cpu-reseed.sh`, reports), `gpu/run-sm71.sh` and its outputs, `HANDOFF-sm71.md` at the worktree root.
