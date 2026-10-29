# What makes cards stick (SM79), part 1: note structure, siblings, new-card load

Question: not which cards fail, but which patterns make cards **easier**, so they stick with fewer reviews.
Descriptive, CPU only (8 of 16 cores, no GPU), run 2026-10-29 21:30-21:35 inside the quiet break with the user's
one-hour grant. Script: [part1.py](part1.py); all tables with bootstrap CIs: [part1.json](part1.json).

## Data and filters

- Benchmark dataset `data/anki-revlogs-10k-original-20261011` (revlogs + cards + decks). **Training users only**:
  the 8k/2k run's `train_users` inside 5000-10000 (outside every evaluation pool: not validation, not the
  464 / 732 / 2,225 pools, not final test). Seeded sample of 1,000 (`random.Random(79)`); 11 users have no local
  revlog file, 98 have under 200 reviews or under 50 eligible cards, **891 users kept**, 53.9M reviews, **5.40M cards**.
- Reviews with rating 1-4 only (manual/reschedule rows dropped). Cards joined to notes via the cards table.
- A card is included if its **first review is in the data** (first row `elapsed_days = -1`) and that first review is
  **at least 180 days before the user's last review day** (follow-up window).
- Outcomes per card:
  - **reach21_10**: the card has a passed review (Hard/Good/Easy) at `elapsed_days >= 21` within its first 10 reviews
    (censored = 0 if never).
  - **lapses10**: number of Again ratings at `elapsed_days >= 1` (i.e. cross-day failures, a proxy for "after
    graduation") in the first 10 reviews.
  - **leech**: >= 8 such lapses over the whole history.
  - **reviews180**: reviews in the card's first 180 days (the cost side).
- Within-user comparisons: each outcome minus its mean in the user x deck stratum (A, C) or user x deck x note-size
  stratum (B, card level); bucket effects are pooled residuals. 95% CIs: 1,000 bootstrap resamples **over users**.
  "Contrast" = residual of a bucket minus residual of the reference bucket, bootstrapped jointly.

## The prize

Share of the included cards' reviews that go to each user's most-reviewed cards:

| cards | share of reviews (review-weighted) | 95% CI | median user |
|---|---|---|---|
| worst 5% | **16.6%** | 15.8-17.5% | 14.3% |
| worst 10% | **27.2%** | 26.1-28.4% | 24.4% |

So making the worst decile behave like an average card would save roughly a fifth of all reviews (27.2% down to
the ~10% they would take as average cards; an upper bound, since card age also differs).

Stickiness baseline: **43.8%** of cards reach a passed review at >= 21 days within 10 reviews (7.2% within 3, 19.4%
within 5, 36.2% within 7, 48.8% within 15); 53.0% ever do in the data; median 6 reviews among cards that do.

## A. Note structure (cards per note)

Cards per note counted from the current cards table (cloze deletions / reverse cards proxy).

Raw means and within user x deck contrast vs single-card notes:

| cards per note | cards | users | reach21_10 raw | lapses10 raw | leech raw | reviews180 raw | reach21_10 vs 1 | lapses10 vs 1 | leech vs 1 (pp) | reviews180 vs 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3,718,133 | 871 | 40.4% | 0.404 | 1.31% | 7.02 | ref | ref | ref | ref |
| 2 | 1,158,390 | 627 | 54.2% | 0.378 | 1.71% | 7.49 | **+0.9 pp** [+0.6, +1.1] | **-0.025** [-0.031, -0.020] | -0.10 [-0.14, -0.06] | **-0.17** [-0.21, -0.13] |
| 3-4 | 371,517 | 438 | 49.2% | 0.345 | 0.69% | 7.26 | +1.0 pp [+0.6, +1.4] | -0.029 [-0.037, -0.021] | -0.09 [-0.17, -0.04] | -0.20 [-0.27, -0.14] |
| 5-9 | 94,397 | 340 | 34.6% | 0.296 | 0.25% | 5.84 | +0.2 pp [-0.5, +0.7] | -0.026 [-0.037, -0.014] | -0.12 [-0.18, -0.07] | -0.21 [-0.30, -0.12] |
| 10+ | 55,592 | 201 | 35.7% | 0.406 | 0.38% | 6.10 | -0.7 pp [-1.7, +0.2] | -0.012 [-0.036, +0.007] | -0.13 [-0.23, -0.06] | -0.03 [-0.21, +0.16] |

Within user x deck, cards of 2-9 card notes lapse ~6-7% less than single-card notes (-0.025 lapses on 0.40),
are ~8% less often leeches and need ~2-3% fewer reviews in 180 days; very large notes (10+) show no gain.
Within user only (not deck) the same signs, about twice as large (part1.json `A_user`), so part of the raw gap is
deck content. Effect size is **small** next to the prize (a 27% review share).

## B. Same-day siblings

**Review level**: cross-day reviews (`elapsed_days >= 1`) of cards in notes with >= 2 cards, residual of pass within
user x log2(elapsed) x note-size bucket x review-index bucket (581 users with such notes, 11.3M reviews).

| that day, before this review | reviews | pass raw | contrast vs no sibling that day |
|---|---|---|---|
| no sibling reviewed that day | 8,650,972 | 89.2% | ref |
| a sibling earlier that day (any) | 1,490,998 | 89.5% | **+1.0 pp** [+0.5, +1.6] |
| - sibling <= 5 rows earlier, it passed | 206,018 | 91.7% | +2.3 pp [+1.7, +3.1] |
| - sibling <= 5 rows earlier, it failed | 38,960 | 57.3% | **-12.6 pp** [-14.9, -10.3] |
| - sibling > 5 rows earlier, it passed | 1,138,750 | 92.6% | +2.4 pp [+2.0, +2.9] |
| - sibling > 5 rows earlier, it failed | 107,270 | 64.2% | -11.0 pp [-12.8, -8.5] |
| a sibling only later that day | 1,118,223 | 84.4% | **-4.0 pp** [-5.0, -3.1] |

Reading: a sibling shown first is worth only +1 pp, and it **does not look like priming**: if the sibling just failed
(so the user has just seen the answer), this card fails far more (-11 to -13 pp), and the effect does not shrink with
distance (<= 5 vs > 5 rows apart). The sibling's result mostly carries **shared note difficulty** (and that day's
form). The -4 pp for the card shown first in a same-day pair is not explained here: candidates are queue order (sort
by relative overdueness or interval shows weaker cards first) and bury rules (a sibling appears later only in some
setups).

**Card level** (cards of notes with >= 2 cards, first review = introduction day, within user x deck x note size):

| on its first day | cards | reach21_10 vs none | lapses10 vs none | leech vs none (pp) | reviews180 vs none |
|---|---|---|---|---|---|
| no sibling that day | 630,006 | ref | ref | ref | ref |
| sibling earlier that day | 600,784 | -0.3 pp [-0.9, +0.4] | +0.003 [-0.009, +0.015] | +0.17 [+0.01, +0.35] | -0.07 [-0.21, +0.06] |
| sibling only later that day | 448,580 | +0.3 pp [-0.4, +0.9] | **-0.033** [-0.044, -0.021] | -0.12 [-0.27, +0.02] | +0.01 [-0.09, +0.13] |

Introducing a card after its sibling the same day has **no durable benefit** (lapses, reach-21, cost all within
noise; leech share marginally higher). The card introduced first does slightly better on lapses, mirror image of the
review-level ordering effect, so it is probably the same queue artefact (which card of a note comes first is not random).

What cannot be separated here: burying on/off (unknown per user; same-day siblings only happen with burying off,
learning siblings, or cramming / filtered decks), queue sort order, and that-day attention. Time of day is not in
the data (only day offsets and row order).

## C. New-card load (new cards started that day)

Within user x deck, contrast vs days with 1-10 new cards:

| new cards that day | cards | users | reach21_10 raw | lapses10 raw | reach21_10 vs 1-10 | lapses10 vs 1-10 | leech vs 1-10 (pp) | reviews180 vs 1-10 |
|---|---|---|---|---|---|---|---|---|
| 1-10 | 372,799 | 878 | 55.3% | 0.566 | ref | ref | ref | ref |
| 11-30 | 1,063,478 | 878 | 55.1% | 0.468 | +1.1 pp [+0.3, +1.8] | -0.024 [-0.036, -0.013] | -0.20 [-0.38, -0.02] | -0.11 [-0.25, +0.01] |
| 31-100 | 1,915,174 | 801 | 47.0% | 0.410 | +1.2 pp [+0.6, +1.8] | -0.020 [-0.032, -0.008] | -0.14 [-0.29, +0.02] | -0.10 [-0.21, -0.01] |
| 100+ | 2,046,578 | 502 | 32.8% | 0.304 | +0.8 pp [+0.1, +1.4] | -0.031 [-0.041, -0.021] | -0.19 [-0.33, -0.04] | -0.22 [-0.33, -0.12] |

Heavy new-card days do **not** look worse; if anything cards from days with 1-10 new cards lapse slightly more.
This is almost certainly selection, not a reason to add more: small-load days include re-added / hand-picked hard
cards and catch-up days, and big-load days include bulk imports of already-known material and decks later abandoned
(fewer reviews also means fewer chances to lapse; reach21_10 is the outcome least affected by that).

## Caveats (apply to everything above)

- **Correlational.** Nothing here is randomized. Scheduling gives hard cards more reviews (outcomes are partly the
  scheduler's response), and within-deck residuals still mix content differences between note types.
- **Selection**: deleted cards are missing (bad cards users deleted are invisible; note sizes are undercounted
  when siblings were deleted), suspended or abandoned cards just stop and look "cheap" on lapses10 / reviews180.
- Lapse = Again at `elapsed_days >= 1`, which also counts a next-day failure during learning (no state filter).
- The prize ranks cards by review count among cards with >= 180 days of follow-up; older cards rank higher too.
- 891 users from the training pool, one seed; within-deck effects are tight, but heterogeneity across users is large.

## What this suggests for users / scheduler defaults

- The size of the prize (worst 10% of cards = 27% of reviews) dwarfs every structural lever found here (note
  structure: ~6-7% fewer lapses, ~2-3% fewer reviews). The bigger wins are probably in handling the worst cards
  (early leech detection, rewrite / split prompts) rather than in note layout.
- Splitting knowledge into a few cards per note (2-9) is mildly associated with easier cards; 10+ card notes are not.
- Seeing a sibling earlier the same day is **not** a learning aid and not clearly harmful to retention; the
  sibling's result is a strong same-day signal of the other card's difficulty (-11 pp if it failed). For
  schedulers: same-day sibling outcomes are a useful **predictive feature**, and keeping burying on (the default)
  costs little measured stickiness.
- No evidence here that capping new cards per day makes cards stick better; do not change defaults on this data.

## What to test next

- **Interval fuzz** (natural experiment): Anki randomizes intervals by a few percent; compare cards with the same
  state whose fuzz drew longer vs shorter intervals, for later lapses and reach-21 (a clean, randomized spacing effect).
- **Missed days**: forced overdue reviews after the user's absent days (vacations) vs on-time reviews of similar cards:
  does a longer-than-planned gap that still passes make the card stickier later (desirable difficulty), and at what cost?
- **Burying inference**: classify users by whether siblings ever appear the same day (bury off) and compare their
  sibling-note cards against bury-on users matched on history length and retention.
- **Queue order**: within a day, does the first-shown card of a note fail more because it is shown first, or because
  it is weaker? Use users whose order is random (no overdueness sort signature) to separate.
- **First-review signals**: first-review rating (Easy/Good/Hard) and response duration as markers of cards that stick;
  learning-step length (preset) within user.
- Part 2 levers: deck / preset differences within user, review timing (spacing of the first 3 reviews), and how early
  leeches can be flagged from the first 5 reviews.
