# Links between cards (SM167): RESULT

Decision: NO GO. Modelling links between cards (cards that fail together, confusable cards shown in a row) is not
worth GPU time with our current models. The pre-registered design is in design.md; the full tables are in
sm167-result.md and sm167-result.json; the per-user aggregate rows are rows-e732.jsonl and rows-wfit.jsonl.

## What was measured

After the model's own instant prediction (label T24, show-time inputs), we asked whether a card's residual (answer
minus predicted probability) moves with the recent residuals, earlier the same day, of the cards it is linked to.
Links are learned only from the user's earlier days. Same-note cards are always removed, and each linked card is
compared with a matched control card from the same session, with the same "same deck as the target" status and the
nearest review position.

## Main model: Small Instant-Only (SM113), 723 users of the 732 set, 30.5 million target reviews

- Co-failure link (both cards failed together on at least two earlier days, at least twice as often as chance): the
  residual correlation is +0.0056 with linked cards against -0.0001 with matched controls. The difference is +0.0057
  (95% CI +0.0037 to +0.0078). It is real but tiny: a linked card's residual explains about 0.003% of the target's
  residual variance. Coverage: 10.1% of reviews have such a linked card in the last 50 same-day reviews.
- Loose co-failure link (at least one earlier day failed together): difference +0.0041 (CI +0.0029 to +0.0053),
  coverage 28.6%.
- Confusable-order link (the target card failed shortly after the other card on earlier days, more than twice its
  own failure rate): difference +0.0014 (CI -0.0037 to +0.0062), not distinguishable from zero; coverage 1.3%.
- LogLoss gain of the causal correction (link features on top of a session baseline, weights fitted on the 100
  training users, evaluated on the 732 set): -0.0017% (CI -0.0024% to -0.0009%). The go rule needs at least 0.05%,
  about 30 times more. Fitting the weights on the 732 set itself (an in-sample upper bound) gives the same -0.0017%.
- The session baseline alone (mean same-day residual of the last 50 reviews and of the same deck) does not help:
  +0.0015% (worse) out of sample. The model already captures the "bad day" state.

## The other models

Small two-head (SM118), instant head: link gain -0.0009% (CI -0.0014% to -0.0004%), co-failure correlation difference
+0.0064. Big Instant-Only (SM99, 718 users): link gain -0.0013% (CI -0.0022% to -0.0005%), difference +0.0061. Same
picture for all three.

## Reading

The leftover card-to-card signal is statistically detectable for co-failure links but about 30 times below our
minimum effect, and an in-sample fit does not raise it. The shared note, deck, preset and user memories plus the
sequence state already absorb what links between cards could add from ids and answers alone. Card text (confusable
content) is not in the dataset, so this study cannot rule out a text-based link model.

## Run

CPU only, 4 worker processes with OMP_NUM_THREADS=1: features for the 100 training users in 29 s and the 732 set in
212 s, then the report. Six users have no card metadata file; they were rerun with every card as its own note and a
shared deck placeholder, as the scorer's loader does. A brute-force recount of the co-failure coverage for one user
matched the vectorized code exactly.

  python artifacts/small-model/sm167/sm167_links.py features --pool wfit --workers 4
  python artifacts/small-model/sm167/sm167_links.py features --pool e732 --workers 4
  python artifacts/small-model/sm167/sm167_links.py report
