# RWKV "Memorised" day-to-day swings on the own collection (2026-10-26, completed 2026-10-27)

Question: why does the fork's RWKV Memorised graph (Cards mode, sum of each card's R at every day end) swing by hundreds of cards from one day to the next, while FSRS stays smooth?

Open first: [daily-decomposition.png](daily-decomposition.png), [swing-by-r-bucket.png](swing-by-r-bucket.png), then the numbers below. Aggregate data: `daily.csv` (one row per day), `buckets.csv`, `logit_shift_daily.csv`, `summary.json`. Scripts: `replay_memorised.py` (CPU replay), `analyse_swings.py` (aggregates and charts).

## How the fork computes the series

`qt/aqt/rwkv_scheduler.py` `_compute_rwkv_memorised_history` (fork, read-only): for each Anki day it warms the Rust runtime up on that day's reviews, then for every card seen so far it runs one **query step** of the model: the card's last review input with `is_query`, no rating or duration, and elapsed = query day − last review day in whole days. The step passes through the card, deck, note, preset and global states (`rslib/src/rwkv/mod.rs` `review_retrievability_features`) and returns the **instant head**, 1 − P(Again). The curve head (`ahead_logits`, forgetting-curve weights) is not used by this graph.

So a card's R on day d depends on its own card state and on four states shared with other cards (deck, note, preset, global), and on per-collection counters fed as features (reviews since the card's last review, new cards since, days since the previous review, reviews done today, weekday, day-offset encodings).

## Rebuild

- Data: the converted copy of 2026-10-14 (`artifacts/collection-eval/2026-10-14/collection-dataset`, 223,670 events, 13,953 cards, copied read-only from the profile on that day). Nothing in the Anki profile was opened for this analysis.
- Model: the fork's big RWKV model (`RWKV_trained_on_5000_10000.pth`, same tensors as the Rust `.bin`), run with this repo's upstream `rwkv/` RNN code in float32 on the CPU. The fork's Rust runtime is not built on this machine, so the series is a faithful port, not the fork's own bytes: same query construction and features as `features_for` in the Rust code, but different random ID encodings (the model is trained to be invariant to them) and f32 CPU arithmetic.
- For every day from 2025-11-25 to 2026-10-04 (last review in the copy) and every card, four values: `full` (the graph's value), `q0` (same day, but with the states and counters of the previous evening: what the graph would show if nothing had been reviewed that day), `qg` (q0 plus the new global state), `qs` (q0 plus all new states). For cards not reviewed that day the change then splits exactly into: **time** (q0 − yesterday), **global state** (qg − q0), **deck/preset/note states** (qs − qg) and **counter features** (full − qs). Reviewed cards and newly introduced cards are their own terms.
- FSRS reference: FSRS-6 with default parameters replayed on the same events (not the user's preset parameters, so the level differs slightly from Anki's FSRS graph; the point is its smoothness).
- Totals are over unsuspended cards (suspension as of the copy). Variance shares are Cov(component, daily change) / Var(daily change), so they add up to 1.

## Status: complete (2025-12-01 to 2026-10-04, 308 days)

The replay was stopped on 2026-10-26 at 20:54 for the CPU break and resumed from its checkpoint on 2026-10-27 00:25 (same code, same seed; [HANDOFF.md](HANDOFF.md)). It covers every Anki day from 2025-11-25 to the last review in the copy, 2026-10-04; the first six days are warm-up for the day-over-day changes. A partial read-out on 2025-12-01 to 2026-03-24 (114 days, commit 735ad659) gave the same picture with slightly different shares.

## Results

At the window's end 13,953 cards have been seen (no suspended cards in the copy); RWKV sums to 10,867, FSRS-6 to 11,099.

| | RWKV | FSRS-6 |
|---|---|---|
| SD of the daily change | 388 cards | 30 cards |
| Mean absolute daily change | 296 cards | 21 cards |
| Days with a move over ±500 / ±1,000 cards | 50 / 8 | 0 / 0 |
| Largest single-day moves | +1,263 (2026-03-06), −1,263 (2026-06-21) | under ±230 |

The swings grow over the year (monthly SD of the daily change: 225 in December, 334 in January, 490 in March, 537 in June, about 300 in September and October).

**Share of the daily-change variance (RWKV):**

| Component | Share | SD (cards) | Mean (cards/day) |
|---|---|---|---|
| Global state shift, cards not reviewed | 35.0% | 153 | 0 |
| Time, cards not reviewed (one more day elapsed, plus the weekday and day-offset encodings) | 28.2% | 234 | +167 |
| Counter features, cards not reviewed (reviews since last review, new cards since, reviews today) | 27.7% | 227 | −187 |
| Deck/preset/note state shift, cards not reviewed | 5.6% | 120 | 0 |
| Cards reviewed that day | 3.1% | 25 | +19 |
| New cards | 0.4% | 20 | +13 |

- **About 96% of the swing comes from cards that were not reviewed that day.** Every card is re-queried through states and counters it shares with the rest of the collection, so one day's reviews move R for all of them. The "time" term is not a smooth decay either: on average it raises R (+167/day), and the counters lower it (−187/day). Two large opposite effects that do not cancel exactly each day leave a noisy remainder.
- **The swing is close to a uniform logit shift.** For each day, take the median per-card change in logit on non-reviewed cards and apply it to every card as R(1−R)·Δlogit. That reproduces the summed shift with R² = 0.98 (correlation 0.99). The typical day moves the mean logit by 0.13; the per-card SD around it is 0.17.
- **So the cards near R = 0.5–0.8 carry it, not the low-R cards.** Bucketed by the previous day's R: 0.5–0.8 (5,552 cards on average) carries 59% of the shift variance; 0.8–0.95 (4,839 cards) 28.5%; below 0.5 (842 cards) 10.4%; above 0.95 (1,027 cards) 2.1%. By elapsed time, cards last reviewed 90–365 days ago carry 40% and 30–90 days 29%. Mature cards (age ≥ 90 days) carry 89%. One deck carries 86% of the variance, but it also holds 9,222 of the cards on average ([buckets.csv](buckets.csv)).
- **Day statistics barely explain it.** Univariate R² with the day's reviews: 0.04%; Again count: 0.7%; Again rate: 2.0%; lapses: 0.9%; no-review day: 0.09%; share of the top deck: 0.3%. An OLS on reviews, Again, lapses, no-review and gap explains 2.8%. Weekday explains 2.4% (Fridays average +107, Tuesdays −114; the 114-day partial read-out showed a larger Friday effect that did not hold over the year). The size of the swing is set by the model's response to the shared inputs, not by how hard the day was.
- **Not the curve head:** the graph never evaluates the forgetting curve. It re-runs the instant head on a query row every day, so "R of a card 200 days after its review" is a fresh model output each day, not a point on a fixed curve.

**Example days** (the three largest up and down moves; cards; components add up to the total):

| Day | Total | FSRS | Time | Global | Deck/preset/note | Counters | Reviewed | New | Reviews / Again / lapses |
|---|---|---|---|---|---|---|---|---|---|
| 2026-03-06 | +1,263 | +38 | +280 | +360 | −189 | +709 | +60 | +42 | 182 / 85 / 65 |
| 2026-06-22 | +1,142 | −28 | +689 | +369 | −62 | +79 | +51 | +15 | 296 / 69 / 51 |
| 2026-03-26 | +1,098 | +6 | +539 | +304 | +295 | −60 | +19 | 0 | 227 / 91 / 67 |
| 2026-06-21 | −1,263 | +10 | +26 | −436 | −157 | −664 | −44 | +12 | 222 / 109 / 98 |
| 2026-03-23 | −1,173 | +17 | −421 | −266 | −117 | −354 | −22 | +6 | 168 / 93 / 77 |
| 2026-05-20 | −1,127 | +26 | +197 | −246 | −131 | −930 | −50 | +33 | 325 / 105 / 89 |

Large up and down days have similar review loads (170–330 reviews, 70–110 Again), and the biggest drop (06-21) is followed by one of the biggest rises (06-22). Each big move is mostly a collection-wide shift that hits the non-reviewed cards, and different days are driven by different shared inputs (counters on 03-06 and 05-20, time on 06-22 and 03-26, global state on 06-21).

## Caveats

- This is a CPU port of the fork's model, not the fork's own Rust output, and it has not been compared with the fork's graph day by day. The ID encodings are random (seed 2025), whereas the Rust runtime uses its own, so individual days can differ from the user's screen. The mechanism (shared states and counters re-queried daily, instant head) is structural and does not depend on the encodings.
- The "time" term mixes elapsed time with the weekday and day-offset encodings. A further pass that varies only the elapsed time would separate them; this one did not.
- FSRS-6 uses default parameters, not the user's presets.
- The source is the collection as copied on 2026-10-14; the last review in it is on 2026-10-04.
