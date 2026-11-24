# Three-way input parity for the fork's feature builder (SM161)

CPU only, 2026-11-24. Question: does the Anki fork, building its inputs from an Anki collection, produce for every
review the same input vector the released models are scored on (the anki-revlogs-10k rows built by this repository's
pipeline, show-time inputs `e2s-query`, leak-free `--deleted-cards neighbour`)? Engine-level parity (native against
PyTorch on exported traces) cannot see this, because those traces come from the fork's own builder.

## Verdict

- **CLEAN for the canonical history builder.** Every answered review of 8 benchmark users (62,758 reviews) gets exactly
  the reference input vector: all 24 card-feature columns and the 28 day-offset inputs agree to at most 8.3e-07
  (float32 rounding), 0 mismatching rows. This is the path that builds every RWKV state at start-up, on rebuilds and in
  the state cache (Rust `rwkv_historical_review_inputs`, then `qt/aqt/rwkv_history.review_input_from_row`, then
  `FeatureState::features_for`). The card, deck and preset grouping is identical for all 8 users.
- **DRIFT on the live paths, for Learning cards shown on a later day.** When the reviewer shows (or answers) a card in a
  Learning step on a later day than its previous answer, the fork feeds the model Anki's day-learning interval, a whole
  number of days times 86,400 s, instead of the real seconds since the previous answer. The interval seconds, their
  time-of-day sine and cosine and the cumulative columns are wrong on those rows (up to 2.0 on a sine / cosine column).
  It touches 8.2% of the show-time queries of the 8 users (4,258 of 51,970, every one a Learning review on a later
  day), more than half for one user with day-long learning steps. Cost on the instant head of the Small models: +0.054% LogLoss for Small two-head (SM118) and +0.059%
  for Small Instant-Only (SM113) on 12 scored users (half of them the most affected users of the pool), +0.09% to
  +0.10% on the 6 most affected users, about +0.01% (CI includes 0) on 6 random users. The fix is one line.
- Two smaller live-path differences and one structural one are listed below; none is a builder bug in the sense of the
  question, but two have cheap fixes.

## What was run

- Users: 8 benchmark users with 5,083 to 18,443 reviews (relearning in all, filtered-deck reviews in 4, deleted cards
  in 4: 485 deleted cards, 1,712 reviews).
- Throwaway collections (under `work/`, never committed): for each user, synthetic millisecond revlog ids solved as
  difference constraints so that the published `elapsed_seconds`, `day_offset` (fixed 86,400 s days at rollover 4) and
  review order are reproduced exactly (0 mismatches on all 62,758 reviews); revlog type from the dataset state; one deck
  per dataset deck with one deck config per dataset preset; every card a review card. Deleted cards are present as
  cards (the collection as it was when they were reviewed), placed as the leak-free scorer places them (deck of the
  existing card with the nearest card id, its preset, a note of their own).
- Fork side: new worktree `wt/fork-parity`, local branch `parity-sm161` from `origin/rc-2026-11` (514f3a41d), one
  ignored Rust test (`rslib/src/rwkv/parity_sm161.rs`, a copy in `fork-test/`): creates the collection with rslib,
  calls the canonical history builder with the desktop's production request (deck-config first-review policy, no
  dynamic presets), maps rows as the desktop does, and dumps `features_for` for every answered review. It also dumps
  emulated live inputs before each answer: the show-time query by the reviewer rule (`rwkv_review_input` with no ease,
  shown at answer time minus the answer time), the same query by the batch rule (`rwkv_review_input_state` for
  learning cards, the deck-queue scorer), and the live answer (`review_answered`). The live rules are re-implemented in
  the test from the code (they read the system clock), so they are an emulation, not the fork's code path.
- Reference side: `scripts/sm31_inputs.samples(data, user, "e2s-query", deleted_cards="neighbour")`, the scorer's full
  layout. Comparison and cause attribution: `scripts/sm161_parity.py compare`; aggregation `scripts/sm161_report.py`
  (`aggregate.json`). A differing row is attributed to a raw-input difference (seconds off by one, another interval
  rule, another day, the deleted-card flags); a differing row with identical raw inputs would be a builder drift.

## Per path (8 users)

| Path | Rows | Rows differing | Unexplained | Max abs diff with identical raw inputs |
|---|---:|---:|---:|---:|
| Canonical history replay (answered reviews) | 62,758 | 0 | 0 | 8.3e-07 |
| Show-time query, reviewer rule | 51,970 | 12,468 | 0 | 7.7e-07 |
| Show-time query, batch rule (Learning cards only) | 13,212 | 8,745 | 0 | 6.0e-07 |
| Live answer | 51,970 | 12,194 | 0 | 8.3e-07 |

Causes of the differing live rows:

| Cause | Reviewer query | Batch query | Live answer |
|---|---:|---:|---:|
| Interval seconds off by 1 or 2 s (rounding of whole seconds) | 8,180 | 7,847 | 7,889 |
| Learning card on a later day: whole days instead of seconds (diff > 2 s) | 4,258 | 0 | 4,286 |
| Elapsed days: seconds divided by 86,400 instead of the day difference | 86 | 2,674 | 19 |
| Shown before the rollover, answered after (query takes the show day) | 68 | 31 | 0 |

Columns on the reviewer query path (max abs diff, rows above 1e-4): scaled_elapsed_seconds 0.26 (12,067),
elapsed_seconds_sin 1.0 (4,486), elapsed_seconds_cos 2.0 (4,317), scaled_elapsed_seconds_cumulative 0.24 (9,520),
the cumulative sine / cosine 2.0 (about 4,400), scaled_elapsed_days 0.43 (86), scaled_elapsed_days_cumulative 0.31
(55); day_offset_diff, day_of_week, cum_reviews_today, cum_new_cards_today and the day inputs only on the 68 rows shown
before the rollover. All other columns (duration, ratings, is_nan flags, diff_new_cards, diff_reviews, scaled_state,
is_query) agree on every row of every path.

## Findings, causes and fixes (fork not changed)

1. **Learning cards on a later day (the drift).** `rwkv_review_input` (qt/aqt/rwkv_scheduler.py) takes a Learning
   card's interval from Anki's scheduling state (`_scheduling_state_elapsed`), and rslib `answering/current.rs` gives a
   day-learning card `elapsed_secs = days * 86,400` (the intraday learning queue reconstructs it from the due time
   minus the fuzzed step). Review, relearning and filtered cards already use the real time since the last review.
   Fix: add `RwkvReviewState.LEARNING` to the states that call `_elapsed_since_card_last_review` (one line, plus a test
   on a day-learning card); this also gives the calendar-day difference as elapsed days. It changes the show-time query
   and the live answer; the canonical replay is already right. Default only: with "learning queues disabled" the steps
   stay in the review queue and get real seconds.
   Impact (instant head, `--inputs e2s-query` against the new `e2s-query-daylearn`, which applies the whole-day
   interval to Learning reviews on a later day, CPU, fp32 weights, 732-pool users with 2,000 to 12,000 reviews):

   | Model | Users | Benchmark reviews | Instant LogLoss change | 95% CI (user bootstrap) | Users worse |
   |---|---:|---:|---:|---:|---:|
   | Small two-head (SM118), all | 12 | 55,100 | +0.054% | +0.005% to +0.110% | 9 |
   | Small two-head (SM118), 6 most affected | 6 | 31,915 | +0.087% | +0.035% to +0.165% | 6 |
   | Small two-head (SM118), 6 random | 6 | 23,185 | +0.008% | -0.054% to +0.108% | 3 |
   | Small Instant-Only (SM113), all | 12 | 55,100 | +0.059% | +0.012% to +0.115% | 9 |
   | Small Instant-Only (SM113), 6 most affected | 6 | 31,915 | +0.095% | +0.046% to +0.181% | 6 |
   | Small Instant-Only (SM113), 6 random | 6 | 23,185 | +0.009% | -0.044% to +0.099% | 3 |

   The most affected users were picked by their share of such reviews (20% to 48%; pool median 4%), so the 12-user
   figure overstates a typical user; the curve head is unchanged (0.0). The live answer has the same error in the state
   it writes during a session, until the next canonical rebuild; that part is not scored here.
2. **Batch rule elapsed days for Learning cards.** `rwkv_review_input_state` (rslib/src/scheduler/rwkv.rs,
   `CardType::Learn`) and the stats-graph path leave elapsed days empty, so the engine uses seconds / 86,400, while the
   training rows use the calendar-day difference (2,674 of 13,212 learning queries, scaled_elapsed_days up to 0.43).
   Fix: set elapsed days from `next_day_at` as the review branch does (one line). Not scored; it moves one column of
   learning cards in queue ordering only.
3. **Whole-second rounding on the live paths.** Live intervals are `floor(now) - floor(last review)` in whole seconds;
   the history and the dataset take `floor((answer ms - previous ms) / 1000)`, and `e2s-query` floors twice (the
   interval, then the interval minus the answer time). The difference is 1 or 2 s and matters only for intervals of a
   few seconds (scaled_elapsed_seconds at most 0.21, for 0 against 2 s).
   Its frequency here follows the synthetic sub-second times, so it is not an estimate of real use. Fix if wanted:
   millisecond arithmetic from the revlog id; low priority.
4. **Shown before the rollover, answered after (68 queries).** The fork's query takes the show day, the scorer's
   `e2s-query` keeps the answer day. A definition choice of the reference; no fix proposed.
5. **Deleted cards (structural, known from the deleted-card leak check (SM71)).** In the collection as exported, cards
   deleted earlier are missing and the fork's history drops their reviews. On the 4 users with deleted cards, 1,712
   reviews are absent and 15,531 of the remaining 23,866 answered reviews (65%) get different counter or day inputs
   (day_of_week and the day inputs when the history's first day moves, diff_new_cards, diff_reviews, cum_*_today). The
   deleted history cannot be rebuilt by the fork; this is the training-to-deployment gap the leak-free retraining
   addresses, not a builder bug.
6. **Reference-side quirk (this repository, not the fork).** `rwkv/data_processing.get_rwkv_data` gives each deleted
   card the note id `ID_PLACEHOLDER + card_id` in a float64 column, where ids 64 apart round together: the 485 deleted
   cards of the 4 users collapse to 35 note ids (435 to 18 for one user), so in training and scoring they share note
   state lanes and note codes, contrary to "a note of its own". That is why the note grouping matches on 4 of 8 users
   (the 4 without deleted cards). Fix: build that id in int64 (changes training inputs; only with a retrain).

Not covered: the Python scoped / incremental history builder (`_historical_rwkv_review_rows`; it is checked in the fork
against the Rust traversal by the history hash), daylight-saving changes (the history uses fixed 86,400 s days from
today's rollover, live day numbers are local calendar days; the dataset has no time zone), random id codes and the
random day baseline (draws, compared as grouping and raw day values only), the trimming of older history at a
Learning start (the dataset is already trimmed).

## Reproduce

```
python scripts/sm161_parity.py build --users 260 330 369 275 287 364 342 293 [--variant export]
# fork worktree, branch parity-sm161, per user:
ANKI_SM161_REQUEST=work/u<user>-<variant>/spec.json cargo test -p anki --lib rwkv_sm161_input_parity_dump -- --ignored
python scripts/sm161_parity.py compare --users ... [--variant export] --out work/cmp_<variant>.json
python scripts/small_model_heldout.py --model TAG=KIND:ARCH:PATH --pool 732 --users-file work/impact-users.json \
  --device cpu --deleted-cards neighbour --inputs e2s-query | e2s-query-daylearn
python scripts/sm161_report.py
```

The fork branch `parity-sm161` (commit 2de493c83) is local only; it holds the test and no data.
