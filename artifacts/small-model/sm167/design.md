# Links between cards (SM167): pre-registered design

Written 2026-11-25 before any result of this study was computed. CPU only, no GPU, no training.

## Question

Our RWKV models already read shared memories per card, note (siblings), deck, preset and user, and they see every
earlier review of the user. Is there any card-to-card signal left after the model's prediction, beyond same note,
same deck and same-session user state? Concretely: does a card's residual (y - p) correlate with the recent residuals
of other cards it is "linked" to, more than with matched unlinked cards of the same session? If not, modelling links
between cards is not worth GPU time.

## Data (read-only)

- Per-review instant predictions dumped by the deploy-time ensemble test (SM160): `wt/sm160/work/sm160-dump/e732/`
  (the 732 set) and `.../wfit/` (100 training users with id >= 5000, outside the 732, 2,225 and sealed lists), label
  T24 (`w8+lr2.16.4q8x8@e2s-query`, show-time inputs).
- Models: main model Small Instant-Only (SM113); also Small two-head (SM118) instant head and Big Instant-Only (SM99).
  Users: every user with that model's dump (Small: 723 of the 732 set, Big: 718).
- Review rows from the dataset the scorer reads (`data/anki-revlogs-10k-original-20261011`: revlogs, cards, decks).
  Dump `review_th` = 1 + the row position in the user's revlog file (checked: y matches rating > 1 on every row).
  There is no clock time: a "session" is one `day_offset`, and "recent" is measured in review positions that day.

## Definitions

- Target review: a review in the dump (a benchmark review). Residual e = y - p_instant.
- Failure: a review rated Again (1) that is not the card's first review. A card "failed on day d" if it has a failure
  on day d.
- Window: for target X at row t on day d, the earlier reviews of day d at rows t-W .. t-1 that are themselves
  benchmark reviews (so they have a residual), of another card and another note than X. Same note is always removed.
  W = 50 for the co-failure link and the session baselines, W = 10 for the confusable-order link.
- Link (a), co-failure, undirected, learned from days strictly before d: C(A, X) = number of past days on which both A
  and X failed. F_A, F_X = number of past days each failed, D = number of past days with any failure. A is linked to X
  if C >= 2 and lift = C * D / (F_A * F_X) >= 2. Days with more than 150 failed cards add no pairs (bulk noise and
  cost). Secondary, loose variant (a1): C >= 1, no lift rule.
- Link (b), confusable order, directed A -> X, learned from days strictly before d: an event is "A reviewed at row s-o
  and X reviewed at row s, same day, o = 1..10, other card and other note". N(A -> X) = past events, K(A -> X) = past
  events where X's review at s was a failure. f_X = (X's past failures + 1) / (X's past non-first reviews + 2).
  A -> X is a link if K >= 2 and K / N >= 2 * f_X.
- (c) state similarity: skipped (card states are not in the dumps).

## Controls

- Same note: removed from every window (siblings already share the note memory).
- Matched random control: for each linked card A in X's window, the control is the window card C (not linked to X
  under that link type, not X's note) that has the same "same deck as X" flag as A and is closest to A in review
  position (ties: the earlier one). The pair is kept only if such a C exists. So deck membership and time-in-session
  are matched, and a bad day or a bad deck shows up equally in both arms.
- Session state baselines (features of the correction model below): mean residual of all window cards (W = 50) and
  mean residual of the same-deck window cards.

## Measures

1. Residual correlation per link type: Pearson correlation, pooled over target reviews with at least one kept pair,
   between e_X and r_link = mean residual of X's kept linked cards; the same with r_ctrl = mean residual of their
   matched controls. Statistic: corr_link - corr_ctrl. Paired user bootstrap (2,000 draws, seed 0, users resampled,
   correlations recomputed from per-user sums), 95% CI. Also reported: regression slope of e_X on r_link and r_ctrl.
2. Coverage: share of target reviews with at least one linked card in the window (any, and with a matched control).
3. Upper-bound LogLoss gain from a simple causal correction, z = logit(p) + b . f, no intercept, f = 0 when absent:
   - M0 (session baseline): f = [mean window residual, mean same-deck window residual].
   - M1 (links): M0 features plus [mean residual of (a) linked cards, indicator any (a) link in window, mean residual
     of (b) linked cards, indicator any (b) link in window].
   - b fitted by Newton's method on the 100 training users only (review-weighted LogLoss), applied unchanged to the 732
     set. The link gain is 100 x (sum LogLoss M1 / sum LogLoss M0 - 1), in %, review-weighted, paired user bootstrap
     2,000 draws, seed 0, 95% CI. Also reported, as information: M0 vs the raw model, and an "oracle" M1 fit on the 732
     set itself (in-sample, a looser upper bound).
   - LogLoss per review as sklearn (p clipped to float64 eps), logits of p clipped to [1e-7, 1 - 1e-7].

## Go rule

Go for a follow-up modelling experiment only if, for the main model Small Instant-Only (SM113), the link gain (M1 vs
M0, trained on the training users) has its 95% CI upper bound below 0 AND the point gain is at least 0.05% (i.e. <=
-0.05%). Otherwise no go. The correlation table is supporting evidence; it cannot turn a no go into a go.

## Outputs

`artifacts/small-model/sm167/`: `sm167_links.py`, per-user score rows (`rows-*.jsonl`, aggregate sums only, no card
ids or raw rows), `sm167-result.{json,md}`, `RESULT.md`. Per-review feature arrays stay in `work/` (not committed).
