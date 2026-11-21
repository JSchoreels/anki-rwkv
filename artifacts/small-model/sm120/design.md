# Fast Good/Easy prediction (SM120): design (pre-registered before any GPU run)

User, 2026-11-16 ~18:00: see how precisely the models predict that an answer is Good or Easy, or just Easy, and how precisely they predict those under a time limit ("how fast some Easy answer might be"), so a card could be scheduled to become, for example, "a Good within 4 s" without counting the Hard answers that may be slower; or some grade-weighted time, since each grade has a probability.

**What answer time means.** The answer time is the whole time on the card, front and back together: from the moment the question is shown to the moment a grade button is pressed, capped at the preset's maximum answer seconds. The "Show answer" moment is not stored, so front-only and back-only times cannot be computed. Every "within S s" in this experiment means answered within S s, **reading the back included**.

## What runs

Scoring only, no training. The answer-time heads (MLP -> 4 grades x 16 log-time bins, P(t | grade)) already trained by the time heads for the family (SM117) and by the time heads for the new candidates (SM119), on their frozen models, scored again on the GPU because SM117 saved only per-user histograms of its own events:

- the six SM117 heads: the Tiny two-head model (SM106), the Small Instant-Only model (SM98), the Tiny Instant-Only model (SM101), the show-time instant-only big model (SM99), the show-time big model (SM91s2), the show-time retrain (SM92);
- then the SM119 heads that exist when the time heads for the new candidates (SM119) has ended: the longer Small Instant-Only model (SM113), the longer cool-down for Tiny (SM115), and the show-time two-head cool-down for Small (SM118) only if its verdict was go.

Same rows and inputs as SM117: the 732 set (723 users; the evaluation pool of `sm117_head.pool_users()`), show-time inputs (e2s-query), leak-free (deleted cards: neighbour), fp32, the SM117 time-labelled test reviews (duration above 0 and not a first review; same-day reviews included). The forward pass and head inputs are SM117's code (`scripts/sm117_head.py`, imported, not edited); the trunk file is checked equal to the release before scoring, so grades are bitwise those of the release. Code: `scripts/sm120_fast.py`.

## Events (grade sets: Rec = {Hard, Good, Easy}, GE = {Good, Easy}, E = {Easy})

1. **Grade only.** P(Rec), P(GE), P(E) from the model's grade probabilities. Baseline (b): the user's running rate of the set (earlier time-labelled reviews, pseudo-count 10 at the pooled rate).
2. **Fast events.** P(set and answered within S s) = sum over g in the set of P(g) P(t <= S | g), for S in {2, 3, 4, 5, 6, 8, 10, 15, 20, 30} s (SM117's grid plus 2, 4, 6). Baselines as SM117: (a) the model's P(g) x the user's running share of earlier answers of grade g within S s, summed over the set; (b) the user's running rate of "set and within S s".
3. **Speed given the grade.** On reviews whose actual grade is Hard, Good or Easy (each separately; Good and Easy are the ones asked): P(t <= S | g) of the head vs the user's running share of earlier answers of that grade within S s (LogLoss, ECE, AUC per S); and the mean absolute ln error of the head's predicted median time of that grade vs SM117's user-median baseline (the user's exact running median of that grade), plus a running 16-bin distribution of the grade.
4. **Grade-weighted time.** The mixture P(t) = sum over all grades of P(g) P(t | g), and the success-only mixture P(t | GE) = sum over Good, Easy of P(g) P(t | g) / P(GE). Scored: censored LogLoss over the 16 bins, mean absolute ln error of the mixture's median and of its expected ln time (bin values: ln-midpoint of each bin, ln 60 s for the top bin; uncensored rows only). Baselines: (a) the same mixture of the user's running per-grade bin distributions with the model's P(g); (b) the user's running bin distribution (all grades for the all-grade mixture, Good and Easy only for the success mixture); the median of (b) is the user's running median read from those 16 bins (linear in ln time inside a bin). The all-grade mixture on all rows and on Good/Easy-graded rows; the success mixture on Good/Easy-graded rows.

Running baselines are causal: only the user's earlier time-labelled reviews (SM117 rule), each with pseudo-count 10 at pooled priors computed on SM117's 1,991 plan training users (none in the 732 pool; `priors.json` here; on SM117's S values these equal SM117's priors, CPU check C1).

S counts only when the user's own maximum answer seconds is above S (SM117's `valid`); a censored answer (at a lower user cap) counts as "not within S".

## The bar (a report, not a go / no-go; same as SM117)

An event at a given S is **reliable** iff the head's ECE (20 equal-mass bins) is below 0.01 **and** its LogLoss is below every baseline of that event, each paired difference review-weighted with the 95% CI (2,000 user bootstraps, seed 0) below zero. Grade-only events: one baseline (b). Speed given the grade: one baseline. Fast events: (a) and (b). For each event the report lists the values of S where it is reliable, with "Good or Easy within 4 s" on its own. For the mixtures (item 4) the report gives the deltas with CIs and "beats both" (every delta below zero with the CI below zero); there is no ECE there.

Reported with it: Brier, mean predicted, event rate, AUC, the reliability table and largest bin gap (in the json).

## CPU check (before any GPU run; `check-cpu.json`)

On 3 small users of the 732 pool with the Tiny two-head model (SM106), CPU: C1 the priors on SM117's S equal SM117's `priors.json`; C2 prior sums and nesting; C3 the Rec fast event reproduces SM117's recorded per-user R-under-S sums (head, (a), (b); row and event counts exactly, LogLoss / Brier / sum p within 1e-4 relative); C4 the running baselines equal `sm117_head.running_baselines` on SM117's S; C5 running bin distributions sum to 1; C6 labels and probabilities: E <= GE <= Rec in event rate and mean p at every S, monotone in S, fast at 30 s below grade-only, label counts match the grade counts, P sums to 1.

## Launch

`launch-sm120.sh` waits until the launcher of the time heads for the new candidates (SM119) has ended (its log's last line `done` or `STOP`), then queues one scoring ticket per model (cap 0.8, then a missing-users pass at 0.92) through `scripts/r01_queue.py`, respecting the GPU break itself (quiet.json, daily 20:00-01:00, Sunday from 16:00), then writes `sm120-result.md` / `.json` on CPU. About 15 to 25 min per model.
