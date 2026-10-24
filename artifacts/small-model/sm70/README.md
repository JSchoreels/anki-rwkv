# Feature discovery (SM70): which new inputs could lower the small model's LogLoss?

Question (user, 2026-10-24): which other features could we take from the users, or construct, to help the current best small model? CPU-only offline analysis, no training. Model: the full retrain's one-epoch Muon cool-down from epoch 10 (SM56, `scale-w64kd-final-sc-lanes-dec10-1ep32k-muon`, w64, scored w8+s8). Big model = the fork's 2.76M RWKV (same users, its dumps from `wt/small-model/artifacts/small-model/sm24-dump`).

## Answer in one paragraph

None of the about 55 candidate inputs that are cleanly available before the answer measurably lowers SM56's LogLoss when fed to a recalibration on top of its prediction. On the instant head the probe is tight: within users, the lower end of every clean single feature's 95% interval is above -0.000024 (2% of the SM56 - big model gap of 0.00118 on these users); the best clean pooled result, the recall rate of cards created next to this card (creation-order neighbours, `k_nb_ret`), is -0.000019 [-0.000036, -0.000002], about 1.6% of the gap, and it does not repeat within users (+0.000035). The only input with a clear gain is the pause between the previous answer and this card's estimated show time (`t_since_any`, from the timestamped `-id` export): instant -0.000357 [-0.000688, -0.000072] within users and -0.000393 [-0.000584, -0.000211] pooled, the same size for the big model. It is not deployable as measured: the export's show time is the answer time minus this review's recorded duration, so a pause appears exactly when the current review's real thinking time exceeded what was recorded (60 s cap, paused timer), which is information about the answer itself; in live Anki the next card shows right after the previous answer, so this gap is about zero inside a session. Its past-only versions (the previous review's pause, a break over 30 min, the session counters as of the previous review) show nothing. The positive control works: the big model's own prediction, used as a feature, recovers -0.0011 curve / -0.0014 instant within users, so the probe can see a real signal of the size of the gap. Curve-head intervals are wide (±0.0005 to ±0.001) and include zero for every clean feature. Where SM56 loses to the big model is not a missing input: the gap concentrates on same-day re-reviews of a card and on cards and users with many failures (see "Where the small model loses"), i.e. situations both models see the same inputs for.

## Method

1. Per-review predictions: `scripts/small_model_heldout.py --device cpu --dump` on the first 200 users of the SM56 458-user list (`wt/sm56/artifacts/small-model/sm56/eval-users-464-fit.json`, stratified over history-length deciles), 5 processes x 2 threads, about 40 min (`run-dump.sh`). 7,058,890 benchmark test reviews. Check: per-user benchmark LogLoss equals the GPU rows of the same checkpoint (`wt/sm56/artifacts/small-model/heldout-sm56-pools.jsonl`) within 4e-6 on all 200 users; weighted curve 0.298148 / instant 0.271692 both ways.
2. Features (`scripts/sm70_features.py`): built from the raw tables of `data/anki-revlogs-10k-original-20261011` (the training data) and the timestamped `data/anki-revlogs-10k` (`-id` export: raw Anki ids, which are creation times in ms, and `review_time`), aligned row by row (original card ids are the ranks of the sorted raw ids; rating match 100%). Every feature uses only reviews before the current one. The curve head is fixed when the card's previous review is answered, so for it every time-varying feature is taken as of the card's previous review (`@c` versions); card-history, static and interval features are the same for both heads.
3. Within-user probe (primary, `scripts/sm70_within.py`): per user, the first half of the test reviews (in time) fits logit = logit(p) + effect(feature bins) (quintiles of that half, own bin for missing, ridge 20), the second half scores it. Delta = LogLoss with - without, review-weighted (the SM56 headline aggregation), 2,000-resample user bootstrap. A per-user Platt variant is also stored (`with-platt` in the logs); per-user Platt alone loses on the second half (curve +0.0010, instant +0.00005), so the offset form is the reported one.
4. Pooled probe (`scripts/sm70_probe.py`): 5-fold cross-validation grouped by user, logistic recalibration [1, z, bins(x), bins(x)·z] fitted with equal weight per training user (6 bins), scored review-weighted on the held-out users; baseline = Platt [1, z]. Plus a gradient-boosted tree on z and all deployable features against the same tree on z alone. A shift-only variant (no bins·z terms) is in `pooled-sm56-shift.json`.
5. Same probes on the big model's predictions (`within-fork.json`, `pooled-fork.json`).
6. Gap breakdown (`scripts/sm70_gap.py`): SM56 - big model per-review LogLoss by feature quintile.

Run: `run-dump.sh`, then `run-probes.sh` (inventory, features, probes, gap; about 50 min CPU), then `scripts/sm70_report.py --dir artifacts/small-model/sm70` for the tables below.

## Inventory: what the data has, what the model sees

| Field | Table | Model sees it? | Available before the answer in Anki? |
|---|---|---|---|
| card_id, note_id, deck_id, preset_id | revlogs, cards, decks | as stream routing (card, note, deck, preset, user streams), not as values | yes |
| day_offset | revlogs | day_offset_diff, day_of_week (relative to the user's first day), per-day counters | yes |
| rating | revlogs | past ratings (one-hot); masked on query rows | no for the current review |
| state (learning / review / relearning / filtered) | revlogs | past rows only; masked on query rows | yes for the current review (tested as `s_state`: no gain) |
| duration | revlogs | past durations; masked on query rows | no for the current review |
| elapsed_days, elapsed_seconds | revlogs | scaled, cumulative, sin/cos of the time of day of the interval | yes (answer-to-answer intervals include the current review's duration: known leak, the catch-up notes, section 4) |
| parent_id (deck hierarchy) | decks | no | yes |
| cards never reviewed, deck and note sizes | cards | no | yes in Anki, but the export-time table also reflects later additions and deletions ("snapshot" rows) |
| review_time (show time, ms) | `-id` revlogs only | no | the clock yes; the export's show time is answer time minus the recorded duration (see the leak above) |
| raw ids = creation times (card, note, deck) | `-id` cards, decks | no | yes |
| note type, template (card ord), fields, tags, flags, scheduled interval / due, ease, preset options (desired retention, steps), suspended / buried | - | not in either export | yes in Anki, not measurable here |

So the planned "card template and note type as content proxy" (SM62) cannot be tested on this dataset: neither export has note types or templates. Inventory numbers on the 200 users (`inventory.json`): 10.13 M reviews; 20% of reviews are on notes with several cards; 62 users use more than one preset, 156 have a deck hierarchy. 26% of all reviews (21.4% of the test reviews, in 85.5% of the users) are on cards missing from the export's cards table, i.e. cards deleted before the export. The model sees this through `note_id_is_nan` (and the placeholder note/deck routing): future information, available to both models. Those cards are recalled more often (test reviews 0.870 vs 0.857). The size of this leak was not measured here.

## Ranked table

Generated by `scripts/sm70_report.py` (also in `tables.md`). Ranked by the better of the two within-user deltas; negative = the feature lowers LogLoss. "Deployable" is about the instant-head version; every curve-head value already uses the as-of-previous-review version. Feature names: `c_` card history, `n_` note siblings, `d_`/`p_` deck and preset, `k_` creation order, `u_` user running stats, `t_` clock and session (`-id` export), `r_` real creation times (`-id` export), `s_state` review type.

Within-user probe (primary), SM56, 200 users, 3,529,494 evaluation reviews (second halves). Delta = LogLoss with the feature - without, review-weighted; 95% user bootstrap.

| Rank | Feature | Plain description | Curve delta | Instant delta | Interval (curve; instant) | Deployable | Implementation cost |
|---:|---|---|---:|---:|---|---|---|
| 1 | big-model prediction (control) | positive control: the big model's own prediction | -0.001118 | -0.001398 | [-0.001977, -0.000208]; [-0.001656, -0.001197] | - | - |
| 2 | t_since_any | pause between the previous answer and this card's estimated show time | +0.000368 | -0.000357 | [-0.000291, +0.001106]; [-0.000688, -0.000072] | no: the show time is answer time minus the recorded (capped / paused) duration of this review | - |
| 3 | c_lapses | lapses so far (Again in review state) | -0.000198 | +0.000010 | [-0.000948, +0.000222]; [-0.000007, +0.000026] | yes | low: one input column |
| 4 | clock-leaky group | all clock-leaky features together | +0.001461 | -0.000168 | [+0.000627, +0.002795]; [-0.000508, +0.000158] | see rows | see rows |
| 5 | c_fail_frac | share of Again among the card's labelled reviews | -0.000076 | +0.000053 | [-0.000874, +0.000428]; [+0.000028, +0.000082] | yes | low |
| 6 | k_backlog | cards never studied yet (export-time table) | -0.000063 | +0.000000 | [-0.000223, +0.000132]; [+0.000000, +0.000000] | snapshot: export-time table | low |
| 7 | t_break | break before this review, if over 30 min | -0.000001 | -0.000008 | [-0.000012, +0.000010]; [-0.000024, +0.000005] | yes | medium: -id build |
| 8 | n_sib_last_y | result of the last sibling review | +0.000799 | +0.000008 | [+0.000069, +0.002082]; [-0.000013, +0.000029] | yes | low |
| 9 | s_state | review type of this review (learning / review / relearning / filtered) | +0.000153 | +0.000013 | [-0.000593, +0.000824]; [-0.000020, +0.000040] | yes | low: unmask the state column on query rows |
| 10 | state group | all state features together | +0.000153 | +0.000013 | [-0.000593, +0.000824]; [-0.000020, +0.000040] | see rows | see rows |
| 11 | n_size | cards in the note (export-time cards table) | +0.000304 | +0.000017 | [+0.000006, +0.000769]; [-0.000002, +0.000036] | snapshot: export-time table | low (Anki knows the note's cards) |
| 12 | c_today | earlier reviews of the card today | +0.000103 | +0.000020 | [+0.000009, +0.000235]; [-0.000006, +0.000047] | yes | low |
| 13 | n_sib_fail | Again share of sibling reviews | +0.000590 | +0.000023 | [+0.000033, +0.001632]; [+0.000002, +0.000046] | yes | low |
| 14 | t_sib_gap_h | hours since a sibling was shown | +0.000677 | +0.000024 | [+0.000024, +0.001780]; [-0.000003, +0.000054] | yes | medium: -id build |
| 15 | c_first_rating | rating of the card's first review | +0.000461 | +0.000024 | [-0.000454, +0.001930]; [-0.000003, +0.000056] | yes | low |
| 16 | n_sib_n | earlier reviews of sibling cards | +0.000241 | +0.000026 | [+0.000008, +0.000525]; [+0.000007, +0.000046] | yes | low |
| 17 | p_ndecks | decks sharing the preset | +0.000968 | +0.000029 | [+0.000042, +0.002478]; [+0.000006, +0.000057] | snapshot: export-time table | low |
| 18 | u_gap | days since the user's previous study day | +0.000150 | +0.000033 | [-0.000012, +0.000317]; [+0.000019, +0.000048] | yes | low |
| 19 | k_nb_ret | recall rate of cards created next to this one | +0.000342 | +0.000035 | [-0.000422, +0.001229]; [+0.000004, +0.000066] | yes | medium: needs creation order |
| 20 | n_sib_gap | days since a sibling was reviewed | +0.000654 | +0.000038 | [-0.000002, +0.001768]; [+0.000010, +0.000074] | yes | low |
| 21 | c_easy_frac | share of Easy among the card's successes | +0.000048 | +0.000040 | [-0.000407, +0.000426]; [+0.000015, +0.000066] | yes | low |
| 22 | p_size | cards under the preset (export-time table) | +0.000961 | +0.000040 | [+0.000035, +0.002478]; [+0.000012, +0.000074] | snapshot: export-time table | low |
| 23 | c_max_ratio | this interval / the longest earlier interval (log) | +0.000573 | +0.000042 | [-0.000567, +0.001789]; [+0.000015, +0.000071] | yes | low |
| 24 | c_dur_last | answer time of the card's previous review | +0.000351 | +0.000042 | [-0.000284, +0.001229]; [+0.000008, +0.000076] | yes | none: duration is an input |
| 25 | u_dur_mean | user's mean answer time | +0.000669 | +0.000043 | [+0.000194, +0.001283]; [+0.000019, +0.000070] | yes | low |
| 26 | u_easy | user's share of Easy among successes | +0.000791 | +0.000044 | [+0.000101, +0.001674]; [+0.000018, +0.000079] | yes | low |
| 27 | k_order_gap | share of earlier-learned cards created after this one | +0.000390 | +0.000049 | [-0.000418, +0.001383]; [+0.000008, +0.000093] | yes | medium: needs creation order |
| 28 | r_batch | cards created within 60 s of this one (import / batch size) | +0.000150 | +0.000054 | [-0.000589, +0.001061]; [+0.000021, +0.000092] | yes | medium: -id build |
| 29 | u_ret | user's running recall rate | +0.000685 | +0.000055 | [-0.000083, +0.001528]; [+0.000030, +0.000086] | yes | low |
| 30 | t_sess_fail | Again share in the session (boundary from this review's show time) | +0.000514 | +0.000056 | [-0.000136, +0.001492]; [+0.000031, +0.000086] | no: same show-time leak | - |
| 31 | u_hard | user's share of Hard among successes | +0.000274 | +0.000057 | [-0.001266, +0.001599]; [+0.000027, +0.000091] | yes | low |
| 32 | c_dur_mean | mean answer time of the card's earlier reviews | +0.000328 | +0.000058 | [-0.000446, +0.001358]; [+0.000032, +0.000085] | yes | low |
| 33 | u_ret_rev | user's running recall rate in review state | +0.000505 | +0.000061 | [+0.000147, +0.000938]; [+0.000036, +0.000091] | yes | low |
| 34 | u_dur_last | answer time of the user's previous review (any card) | +0.000379 | +0.000062 | [-0.000241, +0.001277]; [+0.000040, +0.000089] | yes | low |
| 35 | t_sess_fail_prev | Again share in the session (as of the previous review) | +0.000522 | +0.000066 | [-0.000126, +0.001513]; [+0.000041, +0.000096] | yes | medium: -id build |
| 36 | d_ret | deck's running recall rate (other cards) | +0.000749 | +0.000066 | [-0.000177, +0.001849]; [+0.000040, +0.000095] | yes | low |
| 37 | c_streak | successes since the card's last Again | +0.000907 | +0.000067 | [+0.000195, +0.001734]; [+0.000030, +0.000103] | yes | low |
| 38 | c_entropy | entropy of the card's rating history | +0.000807 | +0.000069 | [+0.000027, +0.001924]; [+0.000043, +0.000099] | yes | low |
| 39 | c_growth | this interval / the previous interval (log) | +0.001026 | +0.000071 | [-0.000142, +0.002512]; [+0.000034, +0.000112] | yes | low |
| 40 | u_yday | reviews on the previous study day | +0.000489 | +0.000071 | [-0.000272, +0.001274]; [+0.000045, +0.000101] | yes | low |
| 41 | c_hard_frac | share of Hard among the card's successes | +0.000264 | +0.000072 | [-0.000536, +0.000950]; [+0.000043, +0.000104] | yes | low |
| 42 | t_gap_prev | pause before the previous review (past only) | +0.000392 | +0.000073 | [-0.000256, +0.001111]; [+0.000042, +0.000107] | yes | medium: -id build |
| 43 | k_nb_n | labelled reviews of cards created next to this one | +0.000746 | +0.000073 | [-0.000261, +0.001822]; [+0.000038, +0.000113] | yes | medium |
| 44 | k_intro_jump | creation-order jump from the previously introduced card | +0.000449 | +0.000074 | [+0.000036, +0.001031]; [+0.000045, +0.000102] | yes | medium |
| 45 | r_wait_new | days between creation and first review | +0.000411 | +0.000075 | [-0.000293, +0.001387]; [+0.000031, +0.000125] | yes | medium: -id build |
| 46 | d_depth | depth of the deck in the hierarchy | +0.001187 | +0.000075 | [-0.000029, +0.002861]; [+0.000032, +0.000125] | snapshot: export-time table | low |
| 47 | t_tod | local time of day (from review times) | +0.000681 | +0.000075 | [-0.000086, +0.001658]; [+0.000046, +0.000109] | yes (Anki has the clock) | medium: review timestamps in training data (-id build) and at inference |
| 48 | t_sess_pos_prev | position in the session (as of the previous review) | +0.000420 | +0.000076 | [-0.000216, +0.001287]; [+0.000051, +0.000104] | yes | medium: -id build |
| 49 | u_wd_ret | user's recall rate on this weekday minus overall | +0.000669 | +0.000077 | [-0.000171, +0.001715]; [+0.000051, +0.000107] | yes | low |
| 50 | t_sess_min | minutes into the session (this review's show time) | +0.000500 | +0.000078 | [-0.000136, +0.001342]; [+0.000051, +0.000108] | no: same show-time leak | - |
| 51 | note group | all note features together | +0.001029 | +0.000079 | [+0.000143, +0.002571]; [+0.000030, +0.000129] | see rows | see rows |
| 52 | d_today | reviews in the deck today so far | +0.000370 | +0.000079 | [-0.000396, +0.001354]; [+0.000055, +0.000105] | yes | low |
| 53 | t_sess_pos | position in the session (boundary from this review's show time) | +0.000435 | +0.000079 | [-0.000198, +0.001304]; [+0.000052, +0.000109] | no: same show-time leak | - |
| 54 | t_sess_min_prev | minutes into the session (as of the previous review) | +0.000475 | +0.000080 | [-0.000162, +0.001309]; [+0.000055, +0.000108] | yes | medium: -id build |
| 55 | u_today_fail | user's Again share today so far | +0.000397 | +0.000081 | [-0.000286, +0.001413]; [+0.000053, +0.000111] | yes | low |
| 56 | u_load | today's reviews so far / the user's mean daily reviews | +0.000461 | +0.000082 | [-0.000267, +0.001335]; [+0.000053, +0.000113] | yes | low |
| 57 | k_rank | position of the card in creation order (export-time) | +0.000675 | +0.000082 | [-0.000104, +0.001552]; [+0.000049, +0.000122] | snapshot: uses later cards | low |
| 58 | u_n | user's reviews so far | +0.000480 | +0.000084 | [-0.001066, +0.001726]; [+0.000044, +0.000130] | yes | none: the user stream counts them |
| 59 | u_ret_100 | user's recall rate over the last 100 labelled reviews | +0.000806 | +0.000087 | [-0.000063, +0.001998]; [+0.000062, +0.000116] | yes | low |
| 60 | d_size | cards in the deck (export-time table) | +0.000874 | +0.000087 | [+0.000002, +0.001936]; [+0.000050, +0.000130] | snapshot: export-time table | low |
| 61 | c_n | number of earlier reviews of the card | +0.000570 | +0.000089 | [-0.000362, +0.001523]; [+0.000054, +0.000132] | yes | none: the card stream counts them |
| 62 | r_note_age | days since the note was created | +0.001082 | +0.000092 | [+0.000314, +0.002186]; [+0.000045, +0.000142] | yes | medium: -id build |
| 63 | r_card_age | days since the card was created (raw Anki id) | +0.000842 | +0.000093 | [-0.000268, +0.002214]; [+0.000047, +0.000140] | yes | medium: -id build |
| 64 | c_age | days since the card's first review | +0.001556 | +0.000108 | [+0.000072, +0.003217]; [+0.000062, +0.000160] | yes | none: elapsed_days_cumulative is an input |
| 65 | r_deck_age | days since the deck was created | +0.001806 | +0.000110 | [+0.000693, +0.003347]; [+0.000065, +0.000161] | yes | medium: -id build |
| 66 | deck group | all deck features together | +0.000999 | +0.000150 | [-0.000198, +0.002515]; [+0.000115, +0.000191] | see rows | see rows |
| 67 | creation group | all creation features together | +0.001466 | +0.000238 | [+0.000266, +0.003031]; [+0.000170, +0.000312] | see rows | see rows |
| 68 | snapshot group | all snapshot features together | +0.002785 | +0.000280 | [+0.001170, +0.004841]; [+0.000198, +0.000375] | see rows | see rows |
| 69 | clock group | all clock features together | +0.001953 | +0.000369 | [+0.000929, +0.003467]; [+0.000287, +0.000464] | see rows | see rows |
| 70 | created group | all created features together | +0.003200 | +0.000416 | [+0.001403, +0.005531]; [+0.000305, +0.000540] | see rows | see rows |

Supplementary: pooled cross-user probe (SM56) and the same probes on the big model.

| Feature | Pooled SM56 curve | Pooled SM56 instant | Within big model curve | Within big model instant | Pooled big model curve | Pooled big model instant |
|---|---:|---:|---:|---:|---:|---:|
| control:big-model-prediction | -0.000498 [-0.000739, -0.000262] | -0.000424 [-0.000492, -0.000362] | - | - | - | - |
| t_since_any | +0.000043 [+0.000010, +0.000079] | -0.000393 [-0.000584, -0.000211] | +0.000548 [-0.000015, +0.001233] | -0.000349 [-0.000659, -0.000087] | +0.000039 [-0.000012, +0.000089] | -0.000377 [-0.000558, -0.000200] |
| c_lapses | +0.000014 [-0.000005, +0.000034] | +0.000009 [+0.000005, +0.000014] | +0.000187 [-0.000045, +0.000451] | +0.000025 [-0.000000, +0.000052] | +0.000024 [-0.000001, +0.000053] | +0.000006 [+0.000002, +0.000010] |
| group:clock-leaky | +0.000137 [+0.000076, +0.000203] | -0.000371 [-0.000581, -0.000171] | +0.001842 [+0.000484, +0.003630] | -0.000156 [-0.000481, +0.000150] | +0.000071 [-0.000047, +0.000182] | -0.000345 [-0.000545, -0.000152] |
| c_fail_frac | +0.000011 [-0.000011, +0.000033] | +0.000015 [+0.000007, +0.000022] | +0.000202 [-0.000131, +0.000527] | +0.000054 [+0.000021, +0.000087] | -0.000050 [-0.000122, +0.000018] | +0.000013 [+0.000004, +0.000023] |
| k_backlog | +0.000066 [+0.000014, +0.000122] | +0.000011 [+0.000003, +0.000019] | -0.000126 [-0.000369, +0.000134] | +0.000000 [+0.000000, +0.000000] | +0.000046 [-0.000041, +0.000130] | +0.000009 [+0.000001, +0.000017] |
| t_break | +0.000001 [-0.000009, +0.000011] | -0.000000 [-0.000005, +0.000005] | +0.000002 [-0.000009, +0.000012] | -0.000000 [-0.000013, +0.000010] | +0.000002 [-0.000011, +0.000017] | -0.000002 [-0.000006, +0.000004] |
| n_sib_last_y | +0.000000 [-0.000030, +0.000025] | +0.000003 [-0.000002, +0.000008] | +0.000689 [-0.000056, +0.001858] | +0.000015 [-0.000013, +0.000044] | +0.000005 [-0.000019, +0.000026] | +0.000005 [-0.000002, +0.000012] |
| s_state | +0.000039 [-0.000030, +0.000137] | +0.000005 [-0.000005, +0.000016] | +0.000748 [-0.000076, +0.001658] | -0.000005 [-0.000060, +0.000041] | -0.000010 [-0.000082, +0.000060] | +0.000011 [+0.000000, +0.000025] |
| group:state | +0.000035 [-0.000024, +0.000109] | +0.000003 [-0.000007, +0.000013] | +0.000748 [-0.000076, +0.001658] | -0.000005 [-0.000060, +0.000041] | -0.000009 [-0.000071, +0.000055] | +0.000011 [+0.000000, +0.000025] |
| n_size | +0.000035 [+0.000006, +0.000073] | +0.000010 [+0.000003, +0.000018] | +0.000245 [-0.000090, +0.000702] | +0.000033 [+0.000009, +0.000059] | +0.000032 [+0.000002, +0.000065] | +0.000009 [+0.000001, +0.000018] |
| c_today | +0.000017 [-0.000014, +0.000046] | +0.000007 [-0.000001, +0.000014] | +0.000073 [+0.000002, +0.000169] | +0.000006 [-0.000033, +0.000038] | -0.000015 [-0.000067, +0.000036] | -0.000002 [-0.000018, +0.000010] |
| n_sib_fail | +0.000038 [+0.000011, +0.000066] | +0.000011 [+0.000003, +0.000020] | +0.000496 [-0.000040, +0.001418] | +0.000018 [-0.000009, +0.000045] | +0.000033 [+0.000008, +0.000061] | +0.000011 [+0.000003, +0.000020] |
| t_sib_gap_h | +0.000000 [-0.000024, +0.000025] | +0.000001 [-0.000012, +0.000016] | +0.000542 [-0.000186, +0.001581] | +0.000027 [-0.000007, +0.000063] | -0.000001 [-0.000025, +0.000026] | +0.000002 [-0.000008, +0.000014] |
| c_first_rating | +0.000005 [-0.000032, +0.000041] | +0.000002 [-0.000005, +0.000008] | +0.001158 [-0.000103, +0.002919] | +0.000004 [-0.000022, +0.000032] | -0.000047 [-0.000112, +0.000016] | +0.000002 [-0.000005, +0.000008] |
| n_sib_n | +0.000015 [-0.000006, +0.000036] | +0.000007 [+0.000002, +0.000013] | +0.000216 [-0.000091, +0.000537] | +0.000024 [+0.000005, +0.000047] | +0.000009 [-0.000011, +0.000030] | +0.000005 [+0.000000, +0.000011] |
| p_ndecks | +0.000066 [+0.000005, +0.000137] | +0.000017 [+0.000010, +0.000025] | +0.001891 [+0.000199, +0.004024] | +0.000021 [-0.000011, +0.000057] | +0.000141 [+0.000037, +0.000262] | +0.000010 [+0.000001, +0.000020] |
| u_gap | +0.000019 [-0.000004, +0.000043] | +0.000002 [-0.000002, +0.000007] | +0.000078 [-0.000170, +0.000291] | +0.000041 [+0.000026, +0.000057] | +0.000016 [-0.000008, +0.000043] | +0.000004 [-0.000001, +0.000010] |
| k_nb_ret | +0.000016 [-0.000019, +0.000052] | -0.000019 [-0.000036, -0.000002] | +0.000780 [-0.000514, +0.002288] | +0.000020 [-0.000020, +0.000059] | -0.000045 [-0.000136, +0.000038] | -0.000016 [-0.000037, +0.000006] |
| n_sib_gap | +0.000019 [-0.000012, +0.000052] | +0.000014 [-0.000004, +0.000035] | +0.000512 [-0.000200, +0.001551] | +0.000040 [+0.000002, +0.000081] | +0.000020 [-0.000007, +0.000049] | +0.000006 [-0.000006, +0.000018] |
| c_easy_frac | +0.000029 [+0.000004, +0.000053] | +0.000009 [+0.000002, +0.000016] | +0.000068 [-0.000233, +0.000382] | +0.000039 [+0.000006, +0.000072] | +0.000016 [-0.000022, +0.000059] | +0.000007 [+0.000000, +0.000015] |
| p_size | +0.000131 [+0.000069, +0.000201] | +0.000014 [+0.000004, +0.000022] | +0.001875 [+0.000173, +0.004020] | +0.000036 [-0.000001, +0.000079] | +0.000121 [+0.000029, +0.000227] | +0.000015 [+0.000005, +0.000024] |
| c_max_ratio | +0.000036 [-0.000025, +0.000097] | +0.000012 [+0.000005, +0.000019] | +0.000785 [-0.000014, +0.001738] | +0.000032 [-0.000001, +0.000065] | +0.000038 [-0.000030, +0.000110] | +0.000011 [+0.000004, +0.000018] |
| c_dur_last | +0.000022 [-0.000001, +0.000044] | +0.000011 [+0.000003, +0.000021] | +0.000418 [-0.000240, +0.001219] | +0.000037 [+0.000003, +0.000073] | +0.000018 [-0.000013, +0.000045] | +0.000010 [+0.000001, +0.000019] |
| u_dur_mean | +0.000072 [+0.000028, +0.000119] | +0.000004 [-0.000002, +0.000011] | +0.000659 [+0.000127, +0.001408] | +0.000054 [+0.000024, +0.000086] | +0.000032 [-0.000015, +0.000074] | +0.000001 [-0.000010, +0.000011] |
| u_easy | +0.000047 [-0.000097, +0.000156] | +0.000007 [-0.000002, +0.000015] | +0.000686 [+0.000056, +0.001451] | +0.000045 [+0.000020, +0.000077] | +0.000008 [-0.000128, +0.000135] | +0.000003 [-0.000007, +0.000012] |
| k_order_gap | +0.000040 [-0.000018, +0.000103] | +0.000009 [-0.000003, +0.000020] | +0.000675 [-0.000291, +0.001780] | +0.000071 [+0.000013, +0.000138] | +0.000037 [-0.000014, +0.000091] | +0.000006 [-0.000005, +0.000018] |
| r_batch | +0.000043 [-0.000001, +0.000091] | +0.000045 [+0.000004, +0.000123] | +0.000253 [-0.000894, +0.001387] | +0.000015 [-0.000021, +0.000057] | +0.000050 [-0.000010, +0.000124] | +0.000011 [-0.000002, +0.000028] |
| u_ret | +0.000056 [+0.000002, +0.000106] | +0.000006 [-0.000001, +0.000012] | +0.001305 [+0.000531, +0.002248] | +0.000053 [+0.000024, +0.000088] | +0.000102 [-0.000002, +0.000206] | +0.000006 [-0.000002, +0.000014] |
| t_sess_fail | +0.000053 [+0.000010, +0.000103] | +0.000012 [+0.000007, +0.000019] | +0.000617 [-0.000153, +0.001556] | +0.000047 [+0.000016, +0.000079] | -0.000013 [-0.000101, +0.000069] | +0.000008 [+0.000002, +0.000015] |
| u_hard | +0.000032 [-0.000036, +0.000099] | +0.000010 [-0.000000, +0.000025] | +0.000411 [-0.000887, +0.001686] | +0.000078 [+0.000030, +0.000137] | +0.000070 [+0.000002, +0.000141] | +0.000008 [+0.000000, +0.000018] |
| c_dur_mean | -0.000002 [-0.000054, +0.000045] | +0.000010 [+0.000005, +0.000016] | +0.000533 [-0.000378, +0.001580] | +0.000055 [+0.000024, +0.000089] | +0.000006 [-0.000053, +0.000057] | +0.000015 [+0.000008, +0.000022] |
| u_ret_rev | +0.000028 [-0.000010, +0.000068] | +0.000016 [+0.000009, +0.000024] | +0.001329 [+0.000344, +0.002747] | +0.000058 [+0.000027, +0.000096] | +0.000009 [-0.000060, +0.000080] | +0.000010 [+0.000003, +0.000017] |
| u_dur_last | +0.000009 [-0.000018, +0.000035] | +0.000015 [+0.000011, +0.000020] | +0.000445 [-0.000148, +0.001226] | +0.000053 [+0.000026, +0.000085] | +0.000005 [-0.000025, +0.000031] | +0.000013 [+0.000009, +0.000018] |
| t_sess_fail_prev | +0.000061 [+0.000012, +0.000115] | +0.000013 [+0.000007, +0.000020] | +0.000614 [-0.000158, +0.001540] | +0.000061 [+0.000028, +0.000095] | -0.000004 [-0.000086, +0.000078] | +0.000010 [+0.000004, +0.000015] |
| d_ret | +0.000016 [-0.000028, +0.000057] | +0.000012 [+0.000007, +0.000018] | +0.001351 [+0.000537, +0.002412] | +0.000068 [+0.000038, +0.000101] | -0.000012 [-0.000090, +0.000061] | +0.000007 [+0.000001, +0.000012] |
| c_streak | +0.000063 [+0.000024, +0.000100] | +0.000014 [+0.000006, +0.000021] | +0.001461 [+0.000005, +0.003256] | +0.000062 [+0.000025, +0.000100] | +0.000020 [-0.000055, +0.000095] | +0.000011 [+0.000004, +0.000018] |
| c_entropy | +0.000021 [-0.000003, +0.000042] | +0.000013 [+0.000006, +0.000019] | +0.001270 [+0.000160, +0.002756] | +0.000071 [+0.000035, +0.000108] | +0.000028 [-0.000005, +0.000061] | +0.000010 [+0.000004, +0.000017] |
| c_growth | +0.000003 [-0.000053, +0.000059] | +0.000014 [+0.000006, +0.000024] | +0.001298 [+0.000211, +0.002708] | +0.000057 [+0.000017, +0.000101] | +0.000009 [-0.000042, +0.000066] | +0.000012 [+0.000005, +0.000020] |
| u_yday | +0.000028 [-0.000013, +0.000068] | +0.000012 [+0.000007, +0.000017] | +0.000765 [+0.000319, +0.001355] | +0.000072 [+0.000036, +0.000112] | +0.000030 [+0.000002, +0.000057] | +0.000007 [+0.000002, +0.000012] |
| c_hard_frac | +0.000021 [-0.000018, +0.000057] | +0.000010 [+0.000002, +0.000017] | +0.000652 [+0.000125, +0.001281] | +0.000074 [+0.000040, +0.000108] | +0.000017 [-0.000033, +0.000069] | +0.000009 [+0.000003, +0.000016] |
| t_gap_prev | +0.000053 [+0.000021, +0.000084] | +0.000013 [+0.000007, +0.000020] | +0.000593 [+0.000003, +0.001315] | +0.000052 [+0.000011, +0.000093] | +0.000051 [+0.000011, +0.000094] | +0.000009 [+0.000002, +0.000017] |
| k_nb_n | +0.000054 [+0.000024, +0.000088] | +0.000012 [+0.000005, +0.000019] | +0.001175 [+0.000359, +0.002166] | +0.000073 [+0.000030, +0.000120] | +0.000057 [+0.000026, +0.000090] | +0.000012 [+0.000005, +0.000019] |
| k_intro_jump | +0.000008 [-0.000010, +0.000026] | +0.000007 [+0.000002, +0.000013] | +0.000368 [-0.000044, +0.000889] | +0.000055 [+0.000024, +0.000089] | +0.000019 [-0.000003, +0.000041] | +0.000007 [+0.000002, +0.000012] |
| r_wait_new | +0.000010 [-0.000048, +0.000069] | -0.000007 [-0.000019, +0.000006] | +0.000527 [-0.000500, +0.001739] | +0.000063 [+0.000016, +0.000119] | +0.000010 [-0.000051, +0.000074] | -0.000007 [-0.000020, +0.000005] |
| d_depth | +0.000036 [-0.000020, +0.000096] | +0.000013 [+0.000003, +0.000025] | +0.001317 [+0.000289, +0.002756] | +0.000064 [+0.000012, +0.000120] | +0.000071 [+0.000006, +0.000152] | +0.000015 [+0.000003, +0.000029] |
| t_tod | +0.000014 [-0.000014, +0.000045] | +0.000007 [+0.000001, +0.000012] | +0.001141 [+0.000336, +0.002211] | +0.000066 [+0.000032, +0.000107] | +0.000006 [-0.000021, +0.000035] | +0.000003 [-0.000003, +0.000010] |
| t_sess_pos_prev | +0.000030 [-0.000005, +0.000065] | +0.000013 [+0.000006, +0.000019] | +0.000970 [+0.000076, +0.002194] | +0.000083 [+0.000048, +0.000120] | +0.000025 [-0.000003, +0.000053] | +0.000011 [+0.000006, +0.000017] |
| u_wd_ret | +0.000029 [+0.000014, +0.000046] | +0.000010 [+0.000006, +0.000015] | +0.001142 [+0.000380, +0.002173] | +0.000067 [+0.000033, +0.000106] | +0.000018 [-0.000002, +0.000039] | +0.000009 [+0.000004, +0.000014] |
| t_sess_min | +0.000019 [+0.000006, +0.000033] | +0.000012 [+0.000006, +0.000019] | +0.000811 [+0.000171, +0.001656] | +0.000073 [+0.000037, +0.000113] | +0.000031 [+0.000014, +0.000048] | +0.000012 [+0.000005, +0.000019] |
| group:note | +0.000031 [-0.000006, +0.000075] | +0.000024 [+0.000007, +0.000043] | +0.000919 [-0.000028, +0.002331] | +0.000083 [+0.000030, +0.000141] | +0.000028 [-0.000009, +0.000069] | +15532.632503 [+5978.045441, +28821.994287] |
| d_today | +0.000027 [-0.000016, +0.000061] | +0.000013 [+0.000006, +0.000020] | +0.000740 [-0.000034, +0.001794] | +0.000081 [+0.000046, +0.000115] | +0.000036 [+0.000006, +0.000064] | +0.000012 [+0.000005, +0.000019] |
| t_sess_pos | +0.000017 [-0.000019, +0.000052] | +0.000015 [+0.000007, +0.000023] | +0.000982 [+0.000084, +0.002211] | +0.000085 [+0.000050, +0.000124] | +0.000014 [-0.000014, +0.000039] | +0.000012 [+0.000005, +0.000021] |
| t_sess_min_prev | +0.000017 [+0.000004, +0.000031] | +0.000010 [+0.000005, +0.000016] | +0.000786 [+0.000152, +0.001621] | +0.000075 [+0.000041, +0.000113] | +0.000034 [+0.000017, +0.000053] | +0.000009 [+0.000003, +0.000015] |
| u_today_fail | +0.000056 [+0.000010, +0.000110] | +0.000016 [+0.000008, +0.000024] | +0.000540 [-0.000193, +0.001451] | +0.000067 [+0.000037, +0.000101] | -0.000024 [-0.000110, +0.000056] | +0.000017 [+0.000009, +0.000025] |
| u_load | +0.000044 [+0.000024, +0.000069] | +0.000007 [+0.000003, +0.000012] | +0.000898 [+0.000223, +0.001735] | +0.000088 [+0.000045, +0.000133] | +0.000039 [+0.000017, +0.000063] | +0.000007 [+0.000002, +0.000012] |
| k_rank | +0.000058 [+0.000004, +0.000114] | +0.000011 [+0.000006, +0.000016] | +0.000971 [-0.000053, +0.002013] | +0.000092 [+0.000052, +0.000139] | +0.000064 [+0.000003, +0.000131] | +0.000011 [+0.000005, +0.000016] |
| u_n | +0.000048 [-0.000002, +0.000099] | +0.000006 [+0.000002, +0.000011] | +0.000509 [-0.000747, +0.001597] | +0.000096 [+0.000044, +0.000154] | +0.000022 [-0.000054, +0.000102] | -0.000004 [-0.000011, +0.000002] |
| u_ret_100 | +0.000074 [-0.000002, +0.000166] | +0.000019 [+0.000013, +0.000025] | +0.001557 [-0.000510, +0.004382] | +0.000066 [+0.000032, +0.000100] | -0.000047 [-0.000278, +0.000164] | +0.000018 [+0.000011, +0.000025] |
| d_size | +0.000049 [-0.000019, +0.000128] | +0.000008 [+0.000002, +0.000015] | +0.001378 [+0.000389, +0.002544] | +0.000097 [+0.000055, +0.000148] | +0.000111 [+0.000034, +0.000214] | +0.000010 [+0.000003, +0.000019] |
| c_n | +0.000070 [+0.000036, +0.000107] | +0.000017 [+0.000009, +0.000024] | +0.001100 [+0.000306, +0.002045] | +0.000084 [+0.000040, +0.000132] | +0.000076 [+0.000038, +0.000119] | +0.000017 [+0.000010, +0.000025] |
| r_note_age | +0.000080 [+0.000010, +0.000169] | +0.000021 [+0.000012, +0.000029] | +0.001727 [+0.000424, +0.003499] | +0.000084 [+0.000039, +0.000132] | +0.000041 [-0.000027, +0.000111] | +0.000014 [+0.000005, +0.000023] |
| r_card_age | +0.000070 [+0.000006, +0.000140] | +0.000012 [+0.000004, +0.000022] | +0.000996 [+0.000004, +0.002311] | +0.000081 [+0.000030, +0.000136] | +0.000063 [-0.000007, +0.000141] | +0.000010 [-0.000000, +0.000020] |
| c_age | +0.000113 [+0.000039, +0.000196] | +0.000021 [+0.000014, +0.000027] | +0.001794 [+0.000241, +0.003467] | +0.000084 [+0.000035, +0.000135] | +0.000094 [+0.000005, +0.000199] | +0.000022 [+0.000016, +0.000028] |
| r_deck_age | +0.000067 [+0.000018, +0.000121] | +0.000009 [+0.000001, +0.000017] | +0.001822 [+0.000457, +0.003654] | +0.000134 [+0.000086, +0.000190] | +0.000095 [+0.000038, +0.000159] | +0.000012 [+0.000004, +0.000021] |
| group:deck | +0.000018 [-0.000065, +0.000086] | +0.000024 [+0.000016, +0.000032] | +0.001494 [+0.000476, +0.002878] | +0.000156 [+0.000113, +0.000205] | +0.000003 [-0.000077, +0.000083] | +0.000020 [+0.000011, +0.000029] |
| group:creation | +0.000063 [-0.000008, +0.000137] | -0.000007 [-0.000027, +0.000014] | +0.002053 [+0.000465, +0.004033] | +0.000260 [+0.000184, +0.000341] | -0.000000 [-0.000123, +0.000121] | -0.000003 [-0.000024, +0.000020] |
| group:snapshot | +0.000424 [+0.000262, +0.000587] | +0.000119 [+0.000077, +0.000178] | +0.003094 [+0.001034, +0.005628] | +0.000280 [+0.000193, +0.000380] | +0.000549 [+0.000349, +0.000755] | +0.000120 [+0.000079, +0.000173] |
| group:clock | +0.000161 [+0.000102, +0.000225] | +0.000057 [+0.000040, +0.000074] | +0.002499 [+0.000898, +0.004512] | +0.000368 [+0.000279, +0.000471] | +0.000074 [-0.000059, +0.000197] | +0.000052 [+0.000035, +0.000067] |
| group:created | +0.000327 [+0.000172, +0.000510] | +0.000157 [+0.000032, +0.000373] | +0.003656 [+0.001240, +0.006571] | +0.000409 [+0.000307, +0.000523] | +0.000295 [+0.000115, +0.000496] | +0.000106 [+0.000023, +0.000248] |

Pooled curve: raw 0.29815, Platt 0.29811; all-deployable +0.002557 [+0.002118, +0.003025]; tree:z-only +0.000155 [+0.000114, +0.000205]; tree:all-deployable +0.002040 [+0.000923, +0.003424]; tree:all-minus-z +0.001885 [+0.000770, +0.003277]
Pooled instant: raw 0.27169, Platt 0.27168; all-deployable +0.001553 [+0.001363, +0.001779]; tree:z-only +0.000095 [+0.000078, +0.000115]; tree:all-deployable +0.000239 [+0.000177, +0.000311]; tree:all-minus-z +0.000145 [+0.000089, +0.000208]

Within curve: raw 0.30165, per-user Platt 0.30267 (Platt - raw +0.001023 [-0.000730, +0.003373])

Within instant: raw 0.27269, per-user Platt 0.27274 (Platt - raw +0.000050 [+0.000009, +0.000100])

## Where the small model loses (gap breakdown)

SM56 - big model per-review LogLoss on the 200 users: curve +0.00033 (0.29815 vs 0.29782, +0.11%), instant +0.00118 (0.27169 vs 0.27051, +0.44%). Share of the gap by feature bin (`gap.json`):

| Situation | Share of test reviews | Curve gap share | Instant gap share |
|---|---:|---:|---:|
| card already reviewed earlier the same day (`c_today` >= 1, learning / relearning steps) | 30% | 185% | 46% |
| card not reviewed earlier that day | 70% | -85% (SM56 better) | 54% |
| first review after an Again (`c_streak` = 0) | 30% | 172% | 50% |
| cards with >= 34% Again in their history (`c_fail_frac`) | 17% | 135% | 37% |
| users with running recall below 0.79 (`u_ret`, lowest fifth) | 20% | 163% | 38% |
| learning-state reviews (`s_state` = 1) | 29% | 135% | 42% |

The small model loses mostly on short-term and failure-heavy situations (same-day steps, just-failed cards, weak users) and is already better than the big model on the curve head for ordinary spaced reviews. These are situations both models see with the same inputs, so the gap looks like capacity or training (state resolution for short intervals) rather than a missing input.

## Caveats

- A recalibration probe bounds what a simple correction on top of the prediction can extract from a feature, not what a network trained with the input could learn (it can combine the input with the card's whole history and the user's state). The outside research measured about -0.0005 curve / -0.0008 instant for timestamp features trained end to end, part of it from the same show-time clock; this probe neither confirms nor rules out a training gain for the clean ones.
- Pooled fits must transfer across users. They hurt for most features (user-level and creation-age features describe the user rather than the review); the gradient-boosted tree on all features is worse than the tree on the prediction alone by +0.0019 curve. Within-user fits must transfer across time; features that trend over a history (ages, counts) get a positive delta from drift.
- Curve-head estimates are noisy (heavy-tailed per-review losses of far-ahead predictions): intervals of ±0.0005 to ±0.001, so curve gains below about 0.0005 cannot be seen with 200 users.
- 200 users (7.06 M test reviews), one checkpoint, the first 200 of the 458-user list. Instant-head intervals are tight enough to exclude gains above about 0.0001 for each clean feature.
- Time of day uses the user's day rollover found from `day_offset` (middle of the nightly no-review window), so it is local time relative to the quiet hours, not wall-clock time.

## Recommendation

- No feature deserves its own GPU test on this evidence. In particular, do not train with the `-id` show-time gap (`t_since_any`) or session counters derived from it: their gain is the current review's untimed thinking time.
- If a GPU slot is spent on inputs anyway, the cleanest candidate bundle is the planned real-timestamp test (SM37) restricted to past-only clock columns (`t_tod`, `t_break`, `t_gap_prev`, `t_sess_*_prev`, `r_card_age`, `r_wait_new`, `r_batch`) plus creation-order neighbours (`k_nb_ret`), as new input columns initialised at zero on a fine-tune from the SM56 checkpoint, measured against a same-length fine-tune without them. Expected effect from this probe: below 0.0001 instant; curve unknown (below about 0.0005). It needs the `-id` build in training and the clock at inference, and the show times must be fixed first (use the answer time of the previous review, never the current review's estimated show time).
- Separate data-quality question for the user: the deleted-card flag (`note_id_is_nan`, 21% of test reviews) is future information in both models' inputs. Measuring it needs a retrain or a fine-tune with the flag zeroed.

## Files

`run-dump.sh`, `run-probes.sh`; `within-sm56.json`, `within-fork.json` (primary probe), `pooled-sm56.json`, `pooled-sm56-shift.json`, `pooled-fork.json` (pooled), `gap.json`, `inventory.json`, `*.log`. Not committed: `dump/` (per-review predictions, 200 npz) and `features.parquet` (7.06 M rows); rebuild with the run scripts. Per-user CPU scoring rows: `heldout-sm70-part*.jsonl`.
