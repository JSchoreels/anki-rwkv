# RWKV "Memorised" day-to-day swings on the own collection (2026-10-26)

Question: why does the fork's RWKV Memorised graph (Cards mode, sum of each card's R at every day end) swing by hundreds of cards from one day to the next, while FSRS stays smooth?

Open first: [daily-decomposition.png](daily-decomposition.png), [swing-by-r-bucket.png](swing-by-r-bucket.png), then the numbers below. Aggregate data: `daily.csv` (one row per day), `buckets.csv`, `logit_shift_daily.csv`, `summary.json`. Scripts: `replay_memorised.py` (CPU replay), `analyse_swings.py` (aggregates and charts).

## How the fork computes the series

`qt/aqt/rwkv_scheduler.py` `_compute_rwkv_memorised_history` (fork, read-only): for each Anki day it warms the Rust runtime up on that day's reviews, then for every card seen so far it runs one **query step** of the model: the card's last review input with `is_query`, no rating or duration, and elapsed = query day − last review day in whole days. The step passes through the card, deck, note, preset and global states (`rslib/src/rwkv/mod.rs` `review_retrievability_features`) and returns the **instant head**, 1 − P(Again). The curve head (`ahead_logits`, forgetting-curve weights) is not used by this graph.

So a card's R on day d depends on its own card state and on four states shared with other cards (deck, note, preset, global), and on per-collection counters fed as features (reviews since the card's last review, new cards since, days since the previous review, reviews done today, weekday, day-offset encodings).

## Rebuild

- Data: the converted copy of 2026-10-14 (`artifacts/collection-eval/2026-10-14/collection-dataset`, 223,670 events, 13,953 cards, copied read-only from the profile on that day). Nothing in the Anki profile was opened for this analysis.
- Model: the fork's big RWKV model (`RWKV_trained_on_5000_10000.pth`, same tensors as the Rust `.bin`), run with this repo's upstream `rwkv/` RNN code in float32 on the CPU. The fork's Rust runtime is not built on this machine, so the series is a faithful port, not the fork's own bytes: same query construction and features as `features_for` in the Rust code, but different random ID encodings (the model is trained to be invariant to them) and f32 CPU arithmetic.
- For every day from 2025-11-25 (planned through 2026-10-14) and every card, four values: `full` (the graph's value), `q0` (same day, but with the states and counters of the previous evening: what the graph would show if nothing had been reviewed that day), `qg` (q0 plus the new global state), `qs` (q0 plus all new states). For cards not reviewed that day the change then splits exactly into: **time** (q0 − yesterday), **global state** (qg − q0), **deck/preset/note states** (qs − qg) and **counter features** (full − qs). Reviewed cards and newly introduced cards are their own terms.
- FSRS reference: FSRS-6 with default parameters replayed on the same events (not the user's preset parameters, so the level differs slightly from Anki's FSRS graph; the point is its smoothness).
- Totals are over unsuspended cards (suspension as of the copy). Variance shares are Cov(component, daily change) / Var(daily change), so they add up to 1.

## Status: partial (2025-12-01 to 2026-03-24, 114 days)

The replay was stopped at 20:54 for the CPU break. It covers 2025-11-25 to 2026-03-24; the rest (to 2026-10-14) resumes from a checkpoint, see [HANDOFF.md](HANDOFF.md). All numbers below are for 2025-12-01 to 2026-03-24 and will be refreshed when the replay completes.

## Results

At the window's end 12,346 cards have been seen (no suspended cards in the copy); RWKV sums to 8,366, FSRS-6 to 10,078.

| | RWKV | FSRS-6 |
|---|---|---|
| SD of the daily change | 318 cards | 37 cards |
| Mean absolute daily change | 237 cards | 25 cards |
| Largest single-day moves | +1,263 (2026-03-06), −1,173 (2026-03-23) | under ±100 |

**Share of the daily-change variance (RWKV):**

| Component | Share | SD (cards) | Mean (cards/day) |
|---|---|---|---|
| Time, cards not reviewed (one more day elapsed, plus the weekday and day-offset encodings) | 30.2% | 175 | +73 |
| Global state shift, cards not reviewed | 29.3% | 118 | +1 |
| Counter features, cards not reviewed (reviews since last review, new cards since, reviews today) | 29.7% | 165 | −94 |
| Deck/preset/note state shift, cards not reviewed | 6.2% | 117 | −6 |
| Cards reviewed that day | 3.5% | 20 | +12 |
| New cards | 1.2% | 27 | +23 |

- **About 95% of the swing comes from cards that were not reviewed that day.** Every card is re-queried through states and counters it shares with the rest of the collection, so one day's reviews move R for all of them. The "time" term is not a smooth decay either: on average it raises R (+73/day), and the counters lower it (−94/day). Two large opposite effects that do not cancel exactly each day leave a noisy remainder.
- **The swing is close to a uniform logit shift.** For each day, take the median per-card change in logit on non-reviewed cards and apply it to every card as R(1−R)·Δlogit. That reproduces the summed shift with R² = 0.98 (correlation 0.99). The typical day moves the mean logit by 0.11; the per-card SD around it is 0.16.
- **So the cards near R = 0.5–0.8 carry it.** Bucketed by the previous day's R: 0.5–0.8 holds 5,040 cards and carries 62% of the shift variance; 0.8–0.95 (4,483 cards) carries 29%; below 0.5 (554 cards) 7.5%; above 0.95 (842 cards) 1%. By elapsed time, cards last reviewed 90–365 days ago carry 45% and 30–90 days 30%. Mature cards (age ≥ 90 days) carry 72%. One deck carries 89% of the variance, but it also holds 8,328 of the cards ([buckets.csv](buckets.csv)).
- **Day statistics barely explain it.** Univariate R² with the day's reviews: 1.0%; Again count: 0.01%; Again rate: 4.5%; lapses: 0.03%; no-review day: 0.1%; share of the top deck: 0.06%. An OLS on reviews, Again, lapses, no-review and gap explains 4.1%. Weekday explains 7.8% (Fridays average +203). The size of the swing is set by the model's response to the shared inputs, not by how hard the day was.
- **Not the curve head:** the graph never evaluates the forgetting curve. It re-runs the instant head on a query row every day, so "R of a card 200 days after its review" is a fresh model output each day, not a point on a fixed curve.

**Example days** (cards; components add up to the total):

| Day | Total | FSRS | Time | Global | Deck/preset/note | Counters | Reviewed | New | Reviews / Again / lapses |
|---|---|---|---|---|---|---|---|---|---|
| 2026-03-06 | +1,263 | +38 | +280 | +360 | −189 | +709 | +60 | +42 | 182 / 85 / 65 |
| 2026-01-16 | +549 | −1 | +231 | +188 | +67 | +36 | +27 | 0 | 198 / 97 / 97 |
| 2026-03-23 | −1,173 | +17 | −421 | −266 | −117 | −354 | −22 | +6 | 168 / 93 / 77 |
| 2026-01-13 | −789 | +1 | −211 | −480 | +136 | −233 | −1 | 0 | 150 / 110 / 110 |
| 2026-01-10 | −747 | −2 | −109 | −71 | −268 | −317 | 0 | +18 | 79 / 71 / 0 |

Large up and down days have similar review loads (150–200 reviews, 70–110 Again). Each big move is mostly a collection-wide shift that hits the non-reviewed cards.

## Caveats

- This is a CPU port of the fork's model, not the fork's own Rust output, and it has not been compared with the fork's graph day by day. The ID encodings are random (seed 2025), whereas the Rust runtime uses its own, so individual days can differ from the user's screen. The mechanism (shared states and counters re-queried daily, instant head) is structural and does not depend on the encodings.
- The "time" term mixes elapsed time with the weekday and day-offset encodings. A further pass that varies only the elapsed time would separate them; this one did not.
- FSRS-6 uses default parameters, not the user's presets.
- The source is the collection as copied on 2026-10-14; reviews after that date are not included.
