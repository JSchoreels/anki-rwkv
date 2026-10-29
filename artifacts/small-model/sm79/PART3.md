# What makes cards stick (SM79), part 3: same-day relearning step

Follow-up to [part 2](PART2.md) item 3 (the same-day learning step for new cards). Question: after a **lapse** (a
graduated card answered Again), does a same-day relearning review help, compared with a lapse whose relearning review
only comes the next day because the session ended? CPU only (8 of 16 cores, no GPU), run 2026-10-29 22:02-22:03
inside the quiet break. Script: [part3.py](part3.py); every table with its CI: [part3.json](part3.json).

**Same data and filters as parts 1-2**: benchmark dataset `data/anki-revlogs-10k-original-20261011`, training users
of the 8k/2k run inside 5000-10000 (outside every evaluation pool), `random.Random(79).sample(1000)`, >= 200 reviews,
>= 50 included new cards (891 users in part 2; 16 of the 1,000 fail to load, the same 16 as in part 2). Of the 891,
**802 have >= 50 lapses with 180 days of follow-up** (53.0M reviews, **3.56M lapse events**). Ratings 1-4 only.
Lapse = Again at `elapsed_days >= 1`. 95% CIs: bootstrap over users, 1,000 resamples; every contrast is **within
user x deck** (plus the extra strata named below), strata with both groups only.

## Design

- Event = one lapse, on lapse day D, with D >= 180 days before the user's last review day. Each lapse of a card is an
  event (the CI treats users as the unit).
- **Same-day relearning** = the card has >= 1 more review on day D after the lapse. **Slip** = no further review on D.
  Of the 3.56M lapses, 13% slipped (0 same-day reviews 466,533; 1: 2,360,697; 2: 475,849; 3+: 257,977).
- **Slip design (as in part 2)**: only decks where >= 70% of that deck's own lapses (>= 10 lapses in the deck) get a
  same-day relearning review; "usually" is taken from the deck's own history. Main sample 641-711 users depending on
  the strata, 0.2M slipped vs 1.4-2.3M same-day lapses. Variants with 50% and 90% below.
- Outcomes (D+1 to D+180): pass at the next review after day D, and its log elapsed time; further lapses; cross-day
  reviews; **total reviews = those plus the same-day relearning reviews**; interval of the first passed review after
  D ("recovery interval", log elapsed); reached a passed review at >= 21 days.
- Strata: A = user x deck. B = A x number of earlier lapses of the card (10 bins, to handle the leech tail) x log2 of
  the interval before the lapse x the card's earlier cross-day pass rate (4 bins + none). C = B x the user's number
  of reviews that day (log2) x the lapse's position in the day's reviews (7 bins from the first 30% to the last 1%).

## Balance: slips are not random

Covariate difference (same-day minus slip, within user x deck; slip means in parentheses):

| covariate | A: deck only | B: + card history strata | C: + day load / position in day |
|---|---|---|---|
| earlier lapses of the card | -0.66 [-1.15, -0.27] (3.97) | -0.02 [-0.10, +0.07] (4.09) | balanced by construction (strata) |
| log interval before the lapse (ln days) | -0.32 [-0.38, -0.26] (2.57) | -0.00 | idem |
| card's earlier pass rate | -0.003 | +0.000 | idem |
| previous rating | +0.03 [+0.01, +0.05] | +0.03 [+0.02, +0.04] | not stratified |
| user's reviews that day (log2) | **+0.77** [+0.71, +0.84] (7.29) | **+0.65** [+0.60, +0.70] | strata |
| lapse position in the day (0 = first, 1 = last) | **-0.26** [-0.28, -0.24] (0.68) | **-0.24** [-0.25, -0.22] (0.69) | strata |

So slipped lapses come **late in the day's reviews** (mean position 0.69 vs 0.42) on **lighter days** (about 1.6x fewer
reviews) and for slightly **weaker cards** (more earlier lapses, shorter intervals in A; B removes that). That is the
fatigue / end-of-session pattern the question suspected; design C removes it as far as the buckets allow (it keeps 424
of the users because both groups must meet in a stratum).

## Result: all lapses, decks that usually show a same-day step (>= 70%)

Contrast same-day relearning minus slip, 95% CI; slip mean in the "base" column.

| design | users | lapses (slip vs same-day) | pass at next review | next review interval (ln days) | further lapses / 180 d | cross-day reviews | **total reviews** | recovery interval (ln days) | reach >= 21 d |
|---|---|---|---|---|---|---|---|---|---|
| A: user x deck | 711 | 0.22M vs 2.35M | +1.6 pp [-0.1, +3.3] (68.1%) | +0.39 [+0.27, +0.50] (0.70) | **+0.18** [+0.05, +0.33] (2.11) | +0.91 [+0.41, +1.50] (10.7) | **+2.28** [+1.76, +2.88] | +0.23 [+0.17, +0.29] (0.38) | +4.8 pp [+3.3, +6.7] (43.8%) |
| B: + card history | 641 | 0.22M vs 1.50M | +1.2 pp [-0.7, +3.1] (68.2%) | +0.52 [+0.39, +0.64] (0.64) | **-0.08** [-0.14, -0.01] (2.24) | -0.26 [-0.51, +0.05] (11.3) | **+0.90** [+0.63, +1.25] | +0.30 [+0.23, +0.36] (0.36) | +1.4 pp [+0.7, +2.3] (45.5%) |
| C: + day load and position | 424 | 0.09M vs 0.28M | +1.8 pp [-0.3, +3.6] (67.8%) | +0.38 [+0.24, +0.53] (0.55) | -0.06 [-0.17, +0.05] (3.05) | -0.08 [-0.41, +0.30] (14.7) | **+0.83** [+0.55, +1.18] | +0.23 [+0.15, +0.32] (0.31) | +2.5 pp [+1.6, +3.4] (48.8%) |

Raw means differ much more (A) than the adjusted ones: the same-day group is the weaker, more lapse-prone card mix
(more earlier lapses). Once the card history is held fixed the lapse effect is small and negative, the total reviews
go **up** by about 0.8-0.9.

## Result: by which lapse of the card it is (the cleanest comparison: first lapse)

Strata B-style without the lapse-count bins (user x deck x interval x earlier pass rate), and with day load and position
(C). Pre-lapse history of the **first** lapse (no earlier lapses) is the same for both groups by definition.

| lapse of the card | design | users | lapses (slip vs same-day) | pass at next review | further lapses / 180 d | cross-day reviews | **total reviews** | recovery interval (ln) | reach >= 21 d |
|---|---|---|---|---|---|---|---|---|---|
| 1st | B | 615 | 75k vs 567k | +1.8 pp [+0.3, +3.2] (73.1%) | **-0.165** [-0.200, -0.134] (1.21) | -0.78 [-0.91, -0.66] (7.8) | **+0.39** [+0.27, +0.50] | +0.35 [+0.29, +0.40] | +1.0 pp [+0.3, +1.7] (47.4%) |
| 1st | C | 380 | 29k vs 99k | +1.7 pp [+0.6, +2.7] (74.5%) | **-0.110** [-0.165, -0.065] (1.25) | -0.46 [-0.65, -0.30] (8.3) | **+0.40** [+0.22, +0.55] | +0.29 [+0.24, +0.35] | +2.0 pp [+1.1, +2.8] (53.3%) |
| 2nd | B | 468 | 40k vs 256k | +0.8 pp [-1.1, +2.5] | -0.136 [-0.169, -0.101] (1.58) | -0.57 [-0.71, -0.43] | +0.60 [+0.46, +0.75] | +0.33 [+0.27, +0.38] | +1.4 pp [+0.6, +2.2] |
| 2nd | C | 254 | 13k vs 37k | +1.2 pp [-0.2, +2.6] | -0.106 [-0.155, -0.056] (1.84) | -0.29 [-0.49, -0.08] | +0.63 [+0.41, +0.85] | +0.30 [+0.24, +0.37] | +2.5 pp [+1.4, +3.7] |
| 3rd-4th | B | 399 | 41k vs 285k | +0.6 pp [-1.6, +2.7] | -0.144 [-0.193, -0.101] (2.01) | -0.41 [-0.61, -0.23] | +0.77 [+0.58, +0.95] | +0.31 [+0.24, +0.37] | +1.4 pp [+0.3, +2.6] |
| 3rd-4th | C | 236 | 16k vs 57k | +0.9 pp [-0.8, +2.2] | -0.144 [-0.213, -0.084] (2.38) | -0.21 [-0.47, +0.05] | +0.68 [+0.42, +0.94] | +0.27 [+0.21, +0.34] | +2.6 pp [+1.2, +4.1] |
| 5th-8th | B | 319 | 29k vs 233k | +1.4 pp [-1.1, +3.5] | +0.205 [+0.000, +0.435] (2.11) | +1.28 [+0.49, +2.12] | +2.50 [+1.70, +3.38] | +0.24 [+0.17, +0.32] | +8.3 pp [+4.9, +11.8] |
| 5th-8th | C | 183 | 13k vs 80k | +2.9 pp [+0.5, +5.2] | +0.438 [+0.062, +0.770] (2.15) | +2.31 [+0.90, +3.44] | +3.38 [+1.90, +4.59] | +0.16 [+0.07, +0.25] | +12.8 pp [+7.5, +17.1] |
| 9th+ | B | 150 | 25k vs 245k | +1.5 pp [-6.4, +6.3] | +0.444 [-0.040, +1.056] (6.68) | +1.94 [+0.38, +3.28] | +3.21 [+1.53, +4.48] | +0.12 [-0.02, +0.35] | +1.5 pp [-0.2, +4.7] |

For lapses 1-4 of a card, a same-day relearning review goes with **0.11-0.17 fewer later lapses in 180 days (9-14% of
1.2-2.4)** and a higher recovery interval (+0.27-0.35 in ln, about x1.3-1.4), but the net **total reviews go up by 0.4
to 0.8**: the relearning review(s) cost more than the 0.2-0.8 cross-day reviews they save. That differs from new cards
in part 2, where the step paid for itself. For the later lapses of heavy-lapse cards (5th lapse and beyond) the sign
flips (more lapses, many more reviews): there the same-day reviews are dominated by difficulty (cards that keep failing
also fail again that day) and the slip group is a different, easier population; treat those rows as confounded.

The next-review pass rate is +1 to +2 pp but with CIs mostly including 0, and **its elapsed time is longer after the
step** (+0.4 to +0.6 ln, x1.5-1.7, partly because a slipped card's next review is the relearning step one day later),
so the pass rate is not comparable at equal interval; the recovery interval and the later lapse count are the cleaner
outcomes.

## Same-day count and pass/fail (design B, all lapse orders, 70% decks)

| group | share of lapses | pass at next review | further lapses | total reviews | recovery interval (ln) | reach >= 21 d |
|---|---|---|---|---|---|---|
| slip (ref) | 13% | 68.2% | 2.24 | 11.3 | 0.36 | 45.5% |
| 1 same-day review | 66% | +3.5 pp [+1.2, +5.6] | -0.29 [-0.39, -0.19] | -0.16 [-0.45, +0.13] | +0.32 [+0.25, +0.39] | +2.7 pp [+1.9, +3.5] |
| 2+ same-day reviews | 21% | **-5.6 pp** [-7.7, -3.6] | +0.55 [+0.25, +0.96] | +4.11 [+2.95, +5.61] | +0.23 [+0.17, +0.30] | -2.4 pp [-3.6, -1.2] |
| step, first same-day review passed | | +3.7 pp [+1.3, +5.9] | -0.31 [-0.44, -0.21] | -0.10 [-0.43, +0.20] | +0.33 [+0.26, +0.40] | +2.7 pp [+1.9, +3.5] |
| step, first same-day review Again | | -9.1 pp [-11.1, -7.0] | +0.89 [+0.45, +1.41] | +5.01 [+3.53, +6.79] | +0.15 [+0.11, +0.20] | -3.6 pp [-4.8, -2.3] |
| step, last same-day review passed | | +1.8 pp [-0.4, +4.0] | -0.15 [-0.23, -0.09] | +0.63 [+0.42, +0.84] | +0.31 [+0.24, +0.37] | +1.6 pp [+0.9, +2.4] |
| step, last same-day review Again | | -13.1 pp [-16.3, -8.9] | +1.82 [+0.66, +3.38] | +7.90 [+3.75, +12.92] | +0.04 [-0.00, +0.08] | -3.4 pp [-5.2, -1.5] |

(Share of lapses in the 70% decks, 1.63M. Step groups by last review: 3% of lapses end the day on Again.) With **exactly one**
same-day relearning review, total reviews are unchanged (-0.16, CI includes 0) while later lapses fall by 0.29 (13%),
the same pattern as the new-card step. When the first relearning review is itself failed the card is simply a harder
card (+0.9 lapses, +5 reviews, pass -9 pp); these subgroups are conditioned on a post-treatment outcome and are
descriptive, not effects.

## Sensitivity to "usually shows a same-day step"

Design B, all lapse orders: deck share threshold 50% (696 users, 0.28M vs 1.54M): further lapses -0.09 [-0.15, -0.03],
total reviews +0.81 [+0.55, +1.08], recovery +0.27 [+0.22, +0.33]; 70% (main, above): -0.08, +0.90; 90% (441 users,
55k vs 833k): +0.00 [-0.13, +0.13], total +1.03 [+0.59, +1.51]. Slightly weaker for the strictest decks (they have
fewer slips and heavier-lapse cards), same direction otherwise.

## Verdict

- A same-day relearning step after a lapse is associated with **fewer later lapses (about -0.11 to -0.17 per lapse over
  180 days, 9-14%, for the first four lapses of a card; -0.06 to -0.08, CI touching 0, for all lapses pooled)** and a
  somewhat longer recovery interval (x1.3-1.4) and +1 to +2.5 pp reaching >= 21 days.
- **Unlike new cards it does not pay for itself**: total reviews over 180 days rise by 0.4-0.9 per lapse (0.4 for the
  first lapse), because the relearning reviews cost more than the cross-day reviews they save. The exception is
  exactly one relearning review, where total reviews are unchanged and lapses are 13% lower (descriptive subgroup).
- 2+ same-day reviews and failing the relearning review mark hard cards, not harm from the step.
- Compared with part 2, the benefit is similar in relative size (about -10% lapses) but the cost side is worse for
  lapses, since a lapsed card has fewer future reviews to save.

## Caveats

- Slips are mostly end-of-session and light-day events (balance table); strata C removes the load and position
  imbalance as far as seven position buckets and the day load allow, but "the user stopped" can reflect fatigue,
  interruption or a short session that we cannot see. Order within the day comes from the log order, no timestamps.
- Preset relearning-step settings (only the current preset is known) are not controlled; decks whose relearning step
  is >= 1 day would show mostly slips and fall below the share rule, so the sample is decks with same-day steps.
- Each lapse of a card is an event; lapses of the same card are correlated and heavy-lapse (leech) cards contribute
  many events, hence the card-lapse-count strata and the ordinal split. The CI resamples users but not cards, and the
  pooled estimate depends on the strata (A vs B differ in sign for further lapses), which is why the by-ordinal table
  is the one to trust.
- Pass at the next review has different elapsed times in the two groups, so pass rates are not like for like.
- Same sampling as parts 1-2 (891 users, one seed, training pool only); the 89 users with < 50 usable lapses drop out.
- The user's own Anki collection was not used.
