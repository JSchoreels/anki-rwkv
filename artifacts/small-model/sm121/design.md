# Timed recall on your own collection (SM121): design (pre-registered before any scoring)

User's Mac assistant, 2026-11-16: a replay of your own collection with the Small pair (button probabilities of the Small Instant-Only model (SM98), time from the answer-time head release (SM83) on the show-time retrain (SM92)) found that "recalled (Hard, Good or Easy) and answered within T s" from the time head beats a rolling 2,000-review rate at every T from 2 to 30 s, on the whole collection and on the Yomitan decks. It asked for the isolation test (does the head beat raw R, R x a historical speed share, and the grade-weighted historical speed?), then for card-history baselines, discrimination read-outs ("are the time predictions really discriminating between cards, or just a big average?", user 11-16) and an exact reproduction control of its numbers. On the 732 set the same scorer exists (the time heads for the family (SM117), fast Good/Easy prediction (SM120)); the time heads for your own collection (SM121) applies it to your collection.

**What answer time means.** The answer time is the whole time on the card, front and back together: from the moment the question is shown to the moment a grade button is pressed, capped at the preset's maximum answer seconds. The "Show answer" moment is not stored, so front-only and back-only times cannot be computed. Every "within S s" in this experiment means answered within S s, **reading the back included**.

## What runs (CPU only, no training, no GPU)

Code: `scripts/sm121_own.py` (imports `scripts/sm117_head.py` and `scripts/sm120_fast.py`, not edited). Each model is run once over the whole collection history as one sequence (the fork's replay; show-time inputs `e2s-query`, leak-free, fp32 weights, seed 2025 + user as in every collection score), below-normal priority, 4 threads, under 6 GB. The frozen trunk of each head is checked equal to the released model before scoring, so the grades are those of the release.

**Data, main analysis.** The collection copy of 2026-10-14 and its conversion (`anki-rwkv/artifacts/collection-eval/2026-10-14`, read only; reviews up to 2026-10-04; 223,670 canonical reviews; the fork's canonical replay rules), the same inputs as every own-collection score so far. The Yomitan decks are the decks whose full name contains "yomitan" (any case), read from that copy (opened `mode=ro&immutable=1`); names never leave the script. Nothing is ever opened under `AppData\Roaming\Anki2` by the scorer.

**Combinations scored** (P(grade) from the first model, P(t | grade) from the head of the second, on its own trunk):

| key | grades from | time head |
|---|---|---|
| `mac_pair` | the Small Instant-Only model (SM98) | the answer-time head release (SM83) on the show-time retrain (SM92) (the Mac's pair) |
| `small_pair_sm98` | the Small Instant-Only model (SM98) | the show-time retrain (SM92)'s SM117 head |
| `small_pair_sm113` | the longer Small Instant-Only model (SM113) | the show-time retrain (SM92)'s SM117 head (the Small pair as the fork runs it today) |
| `sm92`, `sm98`, `sm106`, `sm91s2`, `sm99` | the model itself | its own SM117 head (`sm99` = the Big pair as the fork runs it) |
| `sm113`, `sm115` | the model itself | its own SM119 head, once the time heads for the new candidates (SM119) has trained them |
| `tiny_pair_sm115` | the longer cool-down for Tiny (SM115) | the Tiny two-head model (SM106)'s SM117 head (the Tiny pair as the fork runs it today) |

Big models (the show-time big model (SM91s2), the show-time instant-only big model (SM99)) run last and are killed and skipped after 3 h each.

## Rows

Scored rows: every review with a recorded time that is not the card's first review (the SM117 rule, same-day reviews included). Slices, each on the whole collection and on the Yomitan decks:

- `bench`: the benchmark test reviews of the collection label DB (the 173,690 reviews of every collection score) that have a time: the 732-set row rule. **The reliability call is made here (whole collection), and reported for every slice.**
- `mac`: the Mac's slice: later-day reviews (at least 1 day since the card's previous review) with a prior review and a positive time, the latest 20% chronologically (of this copy).
- `mac_date`: the same filter from 28 April 2026 (the Mac's tail start) on.

## Events and predictors

Grade sets: recalled = {Hard, Good, Easy}, Good or Easy ("Good or better"), Easy. S in {2, 3, 4, 5, 6, 8, 10, 15, 20, 30} s; "set and answered within S s" = grade in the set and duration <= S (SM117 rule); S counts only below the user's cap.

1. **Untimed grade events** (their own table, never compared with the timed ones): P(set) of the model vs (b) the running rate of the set and (b2) its rolling 2,000-review rate.
2. **Timed events** P(set and within S s): the head = sum over the set of P(g) P(t <= S | g); baselines:
   - (1) raw P(set), time ignored;
   - (2) P(set) x the share of earlier answers of the set within S s;
   - (3) sum over the set of P(g) x the share of earlier answers of grade g within S s (SM117's baseline (a); rare grades, Easy above all, are pooled by pseudo-count 10 at the pooled priors);
   - (b) the running rate of the event; (b2) its rolling 2,000-review rate (the Mac's comparator);
   - (4), (5): (2) and (3) with each share adjusted by the card's history: (k + 3 x share) / (n + 3), where n = the card's last three earlier answers with a positive time (any day, any grade) that are in the set (for (4)) or of grade g (for (5)), k = those within S s;
   - the Mac's four (definitions of the reproduction control below, windows in the scope of the rows: the whole collection, or only the Yomitan decks for the Yomitan rows): R x recent, grade x recent, R x card, grade x card (card strength 3; strengths 1 and 10 are reported as a sensitivity, outside the bar).
3. **Speed given the observed grade** (diagnostic; Hard, Good, Easy rows separately): P(t <= S | g) of the head vs the recent share of grade g, its card-adjusted version, the Mac's grade x recent fraction and its card-adjusted version.

**Which earlier reviews feed each baseline.** (2), (3), (b): every earlier review with a time that is not a first review, same-day included, pseudo-count 10 at the pooled priors of SM120 (`artifacts/small-model/sm120/priors.json`, benchmark training users). (b2): the previous 2,000 of those. Card history: the card's last three earlier answers with a positive time, any day, any grade (first reviews included). The Mac's windows: the preceding 2,000 eligible reviews (seen before, at least 1 day elapsed, positive time) inside the scope, updated after each prediction.

## The bar (a report, not a go / no-go; same as SM117 / SM120)

A timed event at a given S is **reliable** iff the head's ECE (20 equal-mass bins, exact on the rows) is below 0.01 **and** its LogLoss is below **every** baseline of the bar ((1), (2), (3), (b), (b2), (4), (5) and the Mac's four at strength 3), each paired head-minus-baseline difference with the 95% CI from 2,000 day-block bootstraps (seed 0) below zero. Card-block bootstraps are reported as a sensitivity (`reliable_card_block`). The report names the strongest baseline per event and S and gives each baseline separately. Speed given the grade: the same rule with its four baselines. Untimed events: (b) and (b2). LogLoss is compared only within the same outcome and threshold. Reported with it: Brier, mean p, event rate, AUC, reliability tables with counts (20 equal-mass and 10 fixed-width bins).

## Discrimination ("is it more than a big average?")

Per combination on `bench` and `mac`, whole collection and Yomitan decks, the head against the card-history baseline (the card-adjusted shares of (5); for time distributions: the card's last-three answers of that grade per 16-bin, (counts + 3 x the recent per-grade bin distribution) / (n + 3)):

1. Spread across reviews of the predicted median time given Good, of P(t <= 4 s | Good) and of P(Good or Easy within 4 s): SD, IQR, 10th / 50th / 90th percentiles.
2. Decile lift: reviews in 10 equal-count bins of the predicted median time given the actual grade (Good only, then all grades): observed median time and observed rate within 4 s and 8 s per decile.
3. AUC given the grade: on Good reviews, P(t <= S | Good) ranking actually-within-S vs not, S = 4 and 8 s (timing only); head vs card vs recent.
4. Brier decomposition (Murphy: reliability, resolution, uncertainty; 20 equal-mass bins) for "Good or Easy within 4 s" and "within 8 s": head, (5), (3), the Mac's grade x card.
5. R^2 of the predicted mean ln time against the actual ln time on uncensored Good reviews: head, card, recent.
6. Within-card: on uncensored Good reviews of cards with at least two of them, each review's predicted ln median minus that card's mean prediction, correlated with the same for the actual ln time (does it follow a card getting faster or slower, not only which cards are slow).

Day-block bootstrap 95% CIs (2,000, seed 0) on head minus card for AUC (200 quantile bins), resolution and reliability, R^2 and the within-card correlation.

## Reproduction control (separate output `sm121-repro.md` / `.json`)

Spec from the Mac (11-16), implemented here from the spec (its scripts are not reachable): time head = the shipped answer-time head release (SM83) of the show-time retrain (SM92) (the fork's `sm92/time_head.f32.bin`, sha256 `814e9b89...`; the fork's SM92 weights `4fbc5f96...`, SM98 weights `a006dcc0...`, all three checked in the CPU check), button probabilities from the Small Instant-Only model (SM98), f32 weights, **int4 card / int8 note state** as the fork runs it. Input: the Mac used a backup of 11-16 (sha256 `2515ef66...`, 226,374 canonical reviews). No file on this computer has that sha256 (the profile's 7 backups, its collection file, the 2026-10-14 copy and two exported packages were hashed, read only); the closest copy is the profile's collection file (last written 2026-11-03, sha256 `c5380f78...`), copied once to `work/repro/` and converted with the same `convert_collection.py`. Numbers therefore cannot match to 2e-6; the control checks the method, and the Mac's backup is needed for an exact match.

- Eligible: card seen before, at least 1 day elapsed, positive time; split once over the whole collection: the latest 20% of the eligible reviews scored (the Mac: first 143,125 = prefix, last 35,782 scored, 27,431 of them in the Yomitan decks); all earlier canonical reviews (same-day included) feed the model state and the card history. Also the tail from 28 April 2026.
- T in {2, 4, 6, 8, 10, 15, 20, 30} s; outcome Hard/Good/Easy and recorded time < T (strict); predictions before the answer (show-time elapsed = floor(answer-to-answer seconds - this duration), clamped at 0, the -1 sentinel kept).
- Baselines (window = the preceding 2,000 eligible reviews in the scope, updated after each prediction): (1) R x recent: R x (fast successes + 0.5) / (successes + 1); (2) grade x recent: per grade Hard / Good / Easy, (fast_g + 10 x the pooled success fraction of (1)) / (n_g + 10), x P(g), summed; (3) R x card: the card's last three earlier positive-time answers (same-day included), (card fast successes + 3 x the fraction of (1)) / (card successes + 3), x R; (4) grade x card: per grade, the matching answers among those three, (fast + 3 x that grade's fraction of (2)) / (n + 3), x P(g), summed. Strength 3; 1 and 10 as a sensitivity. Reading of "pooled success fraction" in (2): the smoothed fraction of (1) (an ambiguity of the spec, listed in HANDOFF).
- LogLoss natural log, p clipped to [1e-7, 1 - 1e-7]; the Mac's values in the table beside ours; paired differences with day- and card-block CIs.

## CPU check (`check-cpu.json`, before the full run)

On the first 800 reviews, CPU: K1 trunk outputs finite, probabilities sum to 1, both SM92 heads load; K1b int4/int8 state moves P(grade) a little (rounding active, under 0.2); K1c the repro copy runs; K2 the recent baselines equal `sm117_head.running_baselines` on SM117's S (whole history); K3 (2), (b2), card history and the Mac's windows equal naive loops (first 800 rows, windows shortened to 100); K4 nesting (Easy <= Good or Easy <= recalled), monotone in S, head <= raw P(set), every prediction in (0, 1); K5 identical predictors give a zero delta and a zero CI; K6 deck map (decks used = decks of the dataset, at least one Yomitan deck); K7 slice sizes; K8 the evaluation and discrimination code end to end; K9 the fork files have the Mac's sha256; K10 CPU-time projection for the full history. Every output passes a privacy scan (no deck name, no per-review field).

## Privacy

Aggregate only: no review or card ids, timestamps (months at most), deck names other than the label "Yomitan decks", no per-review rows in committed files. Per-review intermediates stay in `artifacts/small-model/sm121/work/` (gitignored; never `git add -f`).
