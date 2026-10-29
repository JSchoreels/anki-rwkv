# What makes cards stick (SM79), part 2: early "won't stick" flag, missed days, early spacing

Follow-up to [part 1](PART1.md). CPU only (8 of 16 cores, no GPU), run 2026-10-29 21:42-21:48 inside the quiet break
at the user's request. Script: [part2.py](part2.py); every table with its bootstrap CI: [part2.json](part2.json).

**Same data as part 1**: benchmark dataset `data/anki-revlogs-10k-original-20261011`, training users of the 8k/2k run
inside 5000-10000 (outside every evaluation pool), `random.Random(79).sample(1000)`, **891 users kept**, 53.9M reviews,
5.40M cards whose first review is in the data and is >= 180 days before the user's last review day. Ratings 1-4 only.
Lapse = Again at `elapsed_days >= 1`. 95% CIs: bootstrap over users (1,000 resamples for items 2-3; 300 for
precision / recall / prize and 50 for AUC in item 1). Preset per deck from the decks table (current preset only).

Done: items 1, 2, 3. Item 4 (interval fuzz): power check only, no measurement (see the end).

## 1. Early "won't stick" flag

Labels (per user, among the included cards): **worst 10%** by total reviews (the part 1 prize), **worst 10% by
reviews in the first 180 days** (same window for every card, removes card age), and **>= 4 cross-day lapses ever**
(6.3% of cards). Features known after the card's k-th review only: ratings 1..k, intervals (days and seconds) before
reviews 2..k, days since introduction, answer times of reviews 1..k-1 relative to the user's running mean (review k's
own time left out as a leak), note size, siblings' results so far, the deck's and the user's cross-day pass rate so
far, user experience, new cards started that day. Cards sampled to 600 per user (labels computed on all cards).
**User-grouped split**: 448 train users / 443 test users. Flags are per user (each test user flags the top r% of their
cards); a card with fewer than k reviews cannot be flagged. "Prize" = share of all reviews of the test users'
cards that the flagged cards consume (part 1: 27.2% for the true worst 10%); "after k" = share of the reviews still to
come after review k (the savable part).

Label worst 10% by total reviews, logistic regression (boosting is equal or slightly worse everywhere):

| flag after | cards with k reviews | AUC | flag 5%: precision | recall | prize | flag 10%: precision | recall | prize | prize after k |
|---|---|---|---|---|---|---|---|---|---|
| 1st review | 100% | 0.701 [0.695, 0.712] | 28.8% | 14.4% | 7.4% | 26.2% [24.9, 27.6] | 26.3% | **14.6%** [14.0, 15.3] | 15.1% |
| 2nd review | 93% | 0.756 [0.749, 0.765] | 39.4% | 19.7% | 8.8% | 34.1% [32.9, 35.4] | 34.1% | **16.4%** [15.7, 17.1] | 18.0% |
| 3rd review | 84% | 0.801 [0.793, 0.809] | 46.5% | 23.3% | 9.3% | 39.6% [38.5, 40.9] | 39.6% | **17.4%** [16.8, 18.1] | 20.2% [19.3, 21.2] |
| 5th review | 65% | 0.856 [0.845, 0.865] | 58.0% | 29.0% | 10.2% | 49.4% [47.8, 50.9] | 49.4% | **18.9%** [18.3, 19.6] | 24.6% [23.2, 26.1] |
| oracle (true worst 10%) | | | 99.7% | 49.9% | 15.8% | 92.2% | 92.3% | **26.2%** [25.2, 27.2] | 32.7% |

Random flagging of 10% would capture 10% of reviews. So after the 3rd review the model captures **17.4% of reviews
with 10% of cards, about 46% of the reachable excess** (oracle 26.2%), and ~20% of the reviews still to come.
Same picture for the fixed-window label (after 3: AUC 0.812, precision@10% 41.3%, prize 17.5%, after-k 21.4%) and
for >= 4 lapses (after 3: AUC 0.860 [0.848, 0.874], precision@10% 17.7%, recall 28.4%; after 5: AUC 0.908).

**The simple rule "number of Again among the first k reviews"** (ties: lower mean rating) is almost as good on the
prize: after 3 reviews, prize@10% 16.9% [16.2, 17.5] vs 17.4% for the model, precision 36.9% vs 39.6%, AUC 0.765 vs
0.801 (AUC suffers from ties).

Ablation after the 3rd review (boosting, worst 10% label; full model AUC 0.803, prize@10% 17.0%):

| signal group | AUC without it | prize@10% without | AUC with only it | prize@10% only it |
|---|---|---|---|---|
| ratings (1..k, Again/Hard/Easy counts) | **0.753** [0.744, 0.763] | **15.5%** | **0.774** | **16.9%** |
| user context (pass rate so far, experience, new cards that day, answer-time baseline) | 0.791 [0.784, 0.800] | 16.3% | 0.669 | 12.9% |
| intervals (days / seconds between reviews) | 0.796 | 17.0% | 0.727 | 15.0% |
| past answer times | 0.799 | 17.0% | 0.665 | 12.1% |
| siblings so far | 0.801 | 17.1% | 0.621 | 11.8% |
| deck-mates' pass rate so far | 0.801 | 17.1% | 0.645 | 11.7% |

Ratings carry almost everything; user context adds a little; intervals, answer times, siblings and deck are each
informative alone but redundant once ratings are in. (Correlational, but this is prediction, so that is fine.)

## 2. Missed days (natural experiment on lateness)

Treated review: cross-day review (previous interval >= 1 day) done on the **first active day after the user had
>= 2 idle days**, card last reviewed before the gap. Control: cross-day review on a day where the user was active on
each of the 3 previous days. Comparisons **within user**, within strata previous interval (log2 bucket) x previous
rating x review index (cap 10), strata with both groups only; >= 180 days of follow-up after the review.
765 users, 11.5M on-time reviews, 1.03M late ones. Contrast vs on time, 95% CI:

| gap before the review | reviews | interval taken (x on time) | pass now | next interval, if passed (x) | next review pass, if passed | next review pass, if failed |
|---|---|---|---|---|---|---|
| on time | 11,526,578 | ref | 90.9% | ref | 91.4% | 86.2% |
| 2-3 idle days | 617,975 | x1.24 [1.20, 1.28] | **-0.5 pp** [-1.0, -0.0] | x1.11 [1.07, 1.14] | -0.5 pp [-0.8, -0.1] | -1.1 pp [-1.7, -0.6] |
| 4-7 idle days | 247,608 | x1.42 [1.37, 1.49] | **-1.6 pp** [-2.4, -0.9] | x1.20 [1.12, 1.27] | -0.8 pp [-1.3, -0.2] | -1.7 pp [-2.5, -1.1] |
| 8-30 idle days | 162,631 | x1.81 [1.71, 1.92] | **-3.5 pp** [-4.7, -2.3] | x1.34 [1.26, 1.42] | -1.7 pp [-2.4, -1.0] | -2.8 pp [-4.2, -1.3] |

- **Forgetting from lateness is gentle**: 1.8x the usual interval costs 3.5 pp of recall at that review (and the
  late group is diluted by cards that were due on the return day itself, so the true per-day cost is somewhat higher).
- **A successful late recall earns a longer next interval** (x1.11-1.34, about half of the extra lateness in log
  terms: +0.10 / +0.18 / +0.29 vs +0.20 / +0.32 / +0.53), and the next review still passes about as well (-0.5 to
  -1.7 pp). So the late success is mostly **real extra stability**, credited by the schedulers roughly correctly or
  slightly generously; a "spacing bonus" beyond that is not visible. The next interval also absorbs later gaps of
  the same users, so this is not a clean scheduler test.
- **Failed late cards recover slightly worse at the next review** (-1.1 to -2.8 pp) but cost less in total.
- **Reviews and lapses in the 180 days after are not interpretable as card effects**: after a gap the user is active
  on **11-16 fewer days** of the next 180 (on time: 138; after 2-3 / 4-7 / 8-30 idle days: -11.4 [-14.2, -8.7],
  -15.8 [-19.5, -12.6], -15.7 [-19.6, -11.8]). Raw differences (-0.24 to -0.39 reviews per card in 180 days for all
  late reviews, -0.75 to -1.4 for failed ones) mostly reflect that lower activity; per active day the late cards get
  *more* reviews (+0.36 to +0.77 per 100 active days on 4.15), consistent with backlog catch-up. Full table:
  `T_all` / `T_pass` / `T_fail` in part2.json (`rev180`, `lap180`, `act180`, `rev_per_act`).

Causal status: lateness is set by the user's absence, not by the card, so the **same-review pass drop and the
next-review comparisons are causal-plausible** (remaining biases: which cards get done first on the return day,
backlog cards in the control group, and that gap days are not random days in the user's life). The 180-day cost
outcomes are confounded by the user's future activity.

## 3. Early spacing of new cards

Second exposure **the same day** (learning step) vs **next day or later**, for new cards with >= 2 reviews.
Outcomes in the 180 days after introduction: pass at the first cross-day review, lapses, reached a passed review at
>= 21 days, total and cross-day reviews, leech (>= 8 lapses). 1.60M cards had the next exposure next day or later,
2.23M one extra same-day exposure, 1.21M two or more.

Within user x deck alone the same-day groups look worse (2+ extra: +3.7 reviews, +0.53 lapses), but that is the first
rating: cards that fail the first time get more same-day steps. Conditioning on the first rating:

| design (contrast vs next day or later) | cards | pass first cross-day | lapses / 180 d | reach >= 21 d | reviews / 180 d | cross-day reviews | leech |
|---|---|---|---|---|---|---|---|
| user x deck x first rating, 1 extra same-day | 1.51M vs 0.67M (837 users) | +2.1 pp [+1.6, +2.7] | **-0.113** [-0.131, -0.096] (ref 0.50) | -0.4 pp [-0.9, -0.1] | -0.28 [-0.36, -0.20] (ref 6.64) | -0.44 [-0.50, -0.39] | -0.33 pp [-0.43, -0.24] |
| same, first rating Good only | 1.20M vs 0.39M (769 users) | -0.4 pp [-0.8, -0.0] | -0.062 [-0.081, -0.045] (ref 0.40) | -1.0 pp [-1.6, -0.4] | -0.17 [-0.26, -0.07] | -0.46 [-0.55, -0.38] | -0.21 pp [-0.34, -0.11] |
| **slip design**: Good first, decks that show a same-day step to >= 70% of such cards; same day vs slipped to next day | 1.18M vs 0.11M (659 users) | -2.2 pp [-3.2, -1.2] | **-0.050** [-0.091, -0.015] (ref 0.46) | -0.4 pp [-1.5, +0.6] | +0.15 [-0.01, +0.30] (ref 6.97) | -0.67 [-0.79, -0.54] | -0.18 pp [-0.52, +0.07] |

- The slip design is the closest to an experiment: same deck, same preset, same first rating, and the step was
  missed because the session ended before it was due. There, the same-day step **cuts later lapses by ~11%**
  (-0.05 on 0.46) and replaces about 0.67 cross-day reviews, but itself costs a same-day review, so **total reviews
  are unchanged** (+0.15, CI includes 0). Reach >= 21 days does not move. (The lower pass at the first cross-day
  review is expected: for slipped cards that review is the 2nd exposure one day later; for the others it comes after
  two exposures at a longer interval.)
- With 2+ extra same-day exposures (Good first rating), everything is worse (+0.38 lapses, +2.7 reviews); these are
  cards that also failed a step the same day, i.e. difficulty, not spacing.
- Can't be separated: preset step lengths and preset changes (only the current preset is in the data, no history),
  why the session ended (end-of-session cards may be introduced when the user is tired or in a hurry), and the
  learning-step length itself (`elapsed_seconds` is available; not split here).

## 4. Interval fuzz: power check only (skipped)

Fuzz is a few percent (roughly +-15% at 3-7 days, +-10% at 7-20, +-5% beyond). With a forgetting curve like the FSRS
one at 90% recall, a +-5% interval change moves recall at that review by about +-0.4 pp; with ~11.5M on-time
reviews over 765 users (assume ~1M effective after clustering), the slope on log-interval would have an SE of about
0.01 per log unit vs an expected ~0.09, so **power would be ample if the fuzz were observable**. It is not: the revlogs
hold only the realized interval (scheduled x fuzz + lateness); the unfuzzed interval depends on each user's scheduler
state and parameters (FSRS weights, desired retention, SM-2 ease), which are not in the dataset. Doing it would need
replaying each user's scheduler from the logs; not attempted.

## What this suggests

**For users**
- A card that has failed once or twice in its first 3 reviews is already a strong candidate for rewriting: among
  cards flagged this way (10%), ~37-40% end up in the worst decile and they take ~17% of all reviews.
- Missing a few days is cheap for memory: a review done after a 4-7 day break passes only ~1.6 pp less, and a
  success then is credited with a longer interval. The visible cost of breaks is the backlog, not forgetting.
- A same-day learning step helps a little (about 11% fewer later lapses) and pays for itself in total reviews; it
  is not worth extra steps beyond one for cards that passed.

**For the scheduler**
- After 3 reviews the information is mostly in the ratings. A memory model already sees them, so a separate flag
  model adds little to prediction; the use is in *acting* on the flagged cards (rewrite prompt, split, extra context),
  not in better intervals.
- Late successes: current schedulers credit about half the extra lateness to the next interval and the next pass
  rate holds within 0.5-1.7 pp, so there is no evidence of under-crediting late successes.

**For an Anki feature**
- "Struggling card" early warning: after the 3rd review, flag cards with 2+ Again (or the top 5-10% by a small
  logistic model per user). At a 10% flag rate that catches ~40% of the eventual worst decile and ~20% of the reviews
  still to come, versus the leech threshold (8 lapses) which fires on ~1% of cards after most of the cost is paid.
  It would be a prompt to edit the card, not a scheduling change.

## What next

- Value of acting on the flag: we can't measure rewrite effects here (edits aren't logged); look for cards whose
  note was modified (not in this dataset) or run a user-facing pilot.
- Flag with RWKV / memory-model predicted difficulty after 3 reviews as a feature, compared against the ratings-only
  rule.
- Missed days: restrict to cards whose due date can be inferred as falling inside the gap (users with a known
  scheduler) to remove the dilution, and check whether the next-interval credit differs between FSRS and SM-2 users.
- Early spacing: split the slip design by learning-step length (`elapsed_seconds`) and by how long the slip was
  (1 day vs several).
- Interval fuzz: only via scheduler replay; low priority.

## Caveats

- Correlational except where marked causal-plausible (item 2 same-review and next-review outcomes, item 3 slip design).
- Same selection issues as part 1: deleted cards are missing; suspended / abandoned cards look cheap.
- Item 1: labels use the card's whole later history; the flag rate is per user; 600 cards per user sample.
- Item 2: the late group includes cards due on the return day (dilution); the control group can include overdue
  backlog cards of users who don't clear their queue.
- 891 users, one seed, training pool only.
