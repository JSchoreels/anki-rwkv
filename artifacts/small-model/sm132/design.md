# Again pressed as Hard (SM132): design (2026-11-19, pre-registered before any GPU ticket)

## Question

The user (Thu 11-19 ~15:40): "I wonder how RWKV would be worse if people would randomly use Hard for Again for different probabilities. My bet is that as long as it's lower than 20% it will be just fine, with LogLoss not more than 2% worse."

A training-free robustness test: the trained models read a history in which some Again presses were recorded as Hard, and are scored against what really happened.

| plain name (ID) | role here |
|---|---|
| the show-time two-head cool-down for Small (SM118) | the model of the readout (Small two-head candidate, `w64`, expertium kind), curve + instant |
| the show-time big model (SM91s2) | report only, at 10% and 20%: is the bigger model more or less robust? |
| the compact state for Small and Big (SM131) | the run whose scorer tree this branch is cut from (`origin/sm131` 616fcc41) |

## Corruption (exact definition; `scripts/sm31_inputs.py` `again_as_hard`, scorer flag `--again-as-hard P`)

- For each user, draw `u = numpy.random.default_rng([132, user]).random(n_reviews)` once. Review `review_th` is **flipped** iff its true grade is Again (rating 1) and `u[review_th - 1] < p`. Each Again is flipped independently with probability p; the result is deterministic per (user, p).
- **Deviation from the brief (a seed per (user, p))**: one draw per user, the threshold p varies, so the flipped sets are nested (everything flipped at 10% is flipped at 20%). This is still deterministic per (user, p), and it makes the curve over p smoother: the steps differ only by the added flips, not by a fresh random set each time. At p = 100% every Again is flipped.
- **What the model reads differently** (the update row of each flipped review, i.e. the row that carries the answer to the later reviews):
  1. the grade one-hot: `rating_1` 1 -> 0, `rating_2` 0 -> 1 (the grade reads Hard). This is the only place the grade enters the network: the model has no other grade-derived input (no lapse counter or recall bit; `y` and `rating` feed the labels only).
  2. the card state (`scaled_state`, card-state arm "anki", the default): Anki records the card state before each review (0 new, 1 learning, 2 review, 3 relearning). Only an Again in review state lapses a card into relearning; a Hard there keeps it in review. So the relearning rows that followed a flipped review-state Again are recorded as review-state rows, until an Again that is not flipped lapses the card for real (state machine `_anki_states`, unit-tested in K1). Agains in new / learning / relearning state do not change any state (Hard keeps the same state there). A relearning streak with no recorded review-state Again before it keeps its states.
- **What stays the same**: every target (the curve and instant labels come from the `rating` column, which is not touched: a flipped Again is still scored as a failure); the intervals and answer times (the real timeline, see the limitation below); the query rows (they carry no grade or state); ids, decks, presets, day offsets; the deleted-card handling (neighbour), applied before the corruption.
- **Show-time causality**: the grade of review t sits only in t's update row, which comes after t's own query row (the instant prediction of t) and after every earlier row; the network is causal, so the corrupted grade of t only reaches predictions of later reviews. K3 proves it on the model: with one benchmark Again flipped, every instant prediction up to and including t and every curve prediction made before t are bit for bit the clean ones.
- **Limitation (stated with the result)**: the timeline stays the real one. A person who pressed Hard would have got a longer next interval from the scheduler, not a relearning step of minutes; we cannot replay the scheduler without inventing reviews. So the corrupted history still shows the short interval after a hidden lapse, which the model may partly read as a lapse. The test therefore measures robustness to the recorded grade and card state on a real timeline. The grade-only arm (`--again-as-hard-state keep`, at 20%, report only) shows how much of the effect comes from the card state.
- Default behaviour is unchanged: without the flag no new code runs (K4: Big rows without the flag are bit for bit the committed CPU rows of the compact state for Small and Big (SM131)); with p = 0 the frame, the samples and the rows are bit for bit the clean ones (K2, K4).

## How often Again occurs (`sm132_again_share.py` -> `again-share.json`, 723 users, 42,811,256 reviews)

| quantity | value |
|---|---:|
| Again share of all reviews (review-weighted) | 15.91% |
| Again share, median user (10th / 90th percentile) | 14.59% (3.17% / 31.32%) |
| Again share of review-state reviews (lapses) | 12.25% |
| relearning rows, share of all reviews | 9.14% |
| share of all reviews changed at p = 5 / 10 / 20 / 30 / 50% | 0.80 / 1.59 / 3.18 / 4.77 / 7.95% |

So p = 20% changes about 3.2% of all reviews. The scorer rows also count Again and flipped reviews among the scored (benchmark) reviews per user; the decision reports both.

## Runs (no training; scorer `scripts/small_model_heldout.py`, w8 weights + int8 state = the published format `w8+s8`)

| step | model | p | label | expected GPU min |
|---|---|---|---|---:|
| smoke | SM118 p 0 and 20%, SM91s2 p 0, users 2388 and 4676 | | `gpu/smoke-gpu-rows.jsonl` | 5 |
| s0 | the show-time two-head cool-down for Small (SM118) | 0 (reference) | `sm118-e26-dec2ep32k-t91s2/w8+s8@e2s-query+aah0` | 12 |
| s5, s10, s20, s30, s50 | the show-time two-head cool-down for Small (SM118) | 5, 10, 20, 30, 50% | `...+aah5` ... `...+aah50` | 5 x 12 |
| s20g | the show-time two-head cool-down for Small (SM118), grade only (states kept) | 20% | `...+aah20g` | 12 |
| b10, b20 | the show-time big model (SM91s2) | 10, 20% | `sm91s2-big-showtime/w8+s8@e2s-query+aah10`, `+aah20` | 2 x 16 |

All: 723 fitted users of the 732 set (`wt/sm72s/.../eval-users-732-fit.json`), show-time inputs (`--inputs e2s-query`), leak-free (`--split-existing --deleted-cards neighbour`), `--nonfinite-rows`, GPU cap 0.8 (0.92 only for a retry without progress). Rows `gpu/sm132-rows.jsonl` (complete lines + fsync, unparseable lines skipped and counted). About 121 GPU minutes in ONE chained ticket (Small ~11 min per p as measured for this model in the compact state for Tiny (SM129); Big ~15 min as in the show-time big model (SM91s2)'s scoring).

- The big model at 10% and 20% adds ~32 min to ~89 min (+36%, not double), so it is in, report only.
- Reference for Small: p = 0 scored by this run in the same ticket (the same scorer process type as every p). Reference for Big: its published rows (`wt/sm91s2/.../sm91s2-inputs.jsonl`, sha256 pinned): **deviation** to save ~15 GPU min; the GPU smoke checks p = 0 of this tree against them within 1e-5, and on CPU the no-flag rows equal them within 1e-6 (K4).
- FSRS-7 baseline under the same corrupted history: **skipped**. It needs a per-user refit at each p (723 users x 6 histories) on a separate pipeline that does not use the show-time scorer's inputs; not cheap here. A possible follow-up.

## Pool, statistic, readout (`sm132_rule.py`, fixed now)

- Rule pool: the 722 users the big model's rows have (`showtime_report.py pools()`, sha256 pinned 3cdfed03afd4), restricted to the users with finite rows at p = 0 and at every p of the model; **at least 718 of the 722**, otherwise `decide` exits 2 (pending, `STOP` line) and the main session decides.
- Statistic: per head, % = 100 x (corrupted / clean - 1) of the review-weighted LogLoss on the true labels (`pct_of_ref` of `showtime_report.compare()`), paired user bootstrap 2,000, seed 0, 95% CI, the same users at every p.
- **Readout (the user's bet, pre-registered): the bet holds for a head iff the 95% CI upper bound is <= +2.0% at every p in {5%, 10%, 20%}. Verdict: "holds" if it holds for both heads (curve and instant) of the show-time two-head cool-down for Small (SM118), else "does not hold (<heads>)".** Also reported: the first p of the grid where a head's point estimate passes +2%.
- Reported, no rule:
  - p = 30% and 50% (where it starts to hurt);
  - the grade-only arm at 20% vs clean and vs the Anki-state arm at 20% (how much the card state carries);
  - the show-time big model (SM91s2) at 10% and 20% vs its published rows, with the same +2.0% test (report only) and "Big minus Small" in % with a paired bootstrap on the common users (is the bigger model more or less robust?);
  - **LogLoss against the corrupted labels** (secondary, clearly labelled as dominated by label noise): the same predictions scored with every flipped Again counted as passed (row field `benchmark_corrupt_labels`), vs the clean model on the true labels: what Anki's own metrics would show for a collection with this habit;
  - Again share and share of reviews changed per p (all reviews and the scored reviews);
  - p = 0 of this run vs the published rows of SM118 (expected 0.000%; largest per-user difference).

## CPU proofs (`sm132_check.py`, all required before the GPU ticket; -> `check-cpu-sm132.json`)

- K1 state machine on synthetic cards (8 cases + a no-flip identity on random sequences).
- K2 real users 2388, 4676, 2450: p = 0 frame and samples bit for bit unchanged; the draw is deterministic and nested; the p = 100% dry check flips every Again in the inputs (rating_1 0 / rating_2 1 on every Again update row), changes only `rating_1`, `rating_2` and `scaled_state`, only on update rows, the reported state-row count matches, and every label tensor, review index, id and day offset is bit for bit the clean one (no target changes); the grade-only arm changes the two rating columns only.
- K3 causality on the model (SM118 float32, user 2388, one benchmark Again flipped): predictions before t bit for bit equal, labels equal, later predictions change.
- K4 scorer rows on CPU (users 2388, 4676; `cpu/smoke-cpu-rows.jsonl`): no flag == p = 0 bit for bit for SM118 and SM91s2 (benchmark, benchmark_existing, benchmark_deleted); corrupted labels at p = 0 == benchmark; Big no-flag == SM131's committed CPU rows bit for bit; SM118 no-flag vs its published GPU rows 0.0 (6-decimal rows); p = 100% flips every Again, also every benchmark Again; p = 20% finite.
- K5 the readout's selftest on synthetic rows (zero cost holds; curve x 1.03 at 20% fails curve only; x 1.019 holds; x 1.021 fails; 3 missing users reported; 10 missing -> pending; torn line counted).
- First run 11-19 ~15:55: ALL OK (81 s).

GPU smoke (first step of the ticket; a mismatch = `STOP`): SM118 at p = 0 and 20% and SM91s2 at p = 0 on users 2388, 4676 on the GPU: every head within 1e-5 of the CPU rows, the corruption counts identical, p = 0 within 1e-5 of the published GPU rows.

## Decisions fixed now

1. The corruption, the card-state rule, the seed, the p grid and the labels above.
2. The readout (CI upper bound <= +2.0% at 5, 10, 20%, both heads of SM118) and the coverage floor (718 of 722).
3. Big uses its published rows as the reference; Big, p = 30 / 50%, grade-only and the corrupted-label LogLoss are report only.
4. No quiet window is planned; only quiet.json windows of scope all / gpu are honoured (re-read before every ticket).
