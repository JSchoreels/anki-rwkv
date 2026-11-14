# Show-time retrain of the release (SM92): design (2026-11-03)

## Question

The released small model (cool-down from epoch 25, SM63) scores 0.30003 / 0.27654 on normal inputs but 0.30041 / 0.27772 on show-time inputs (`e2s-query`, what the Anki fork sees at query time; the duration-leak scoring (SM31)). The show-time interval retrain (SM91) step 1 showed that one epoch on show-time inputs plus a quarter-epoch cool-down wins back ~2/3 of that cost; step 3 showed that new teacher targets add nothing. Does the same recipe, applied to the release itself (no 15-20 epoch redo, the user's choice), give a ship candidate?

## What runs

| | New (SM92) | Release (read only) |
|---|---|---|
| Start | the leak-free small model (SM72s) constant-LR epoch 25 (w64), its AdamW state, update offset 551,950 | the same |
| Epoch 26 | the run's own epoch-26 protocol (`sm77_warm.py epoch --epoch 26`: the run's order rule `Random(split_seed * 1000 + 26)`, epoch 25's label-window trims, run.json options, constant LR 7e-4) + `"show_time_intervals": true` | none |
| Cool-down | the release's recipe: `sm77_warm.py cooldown --epoch 25 --fraction 1.0` (tier final, shuffle seed 1, 8,000 users, 581,942,724 reviews after trims, 32k windows, batch_chunks 1, fresh Muon `expertium-iter53`, LR 7e-4 -> 0 one_minus_sin, KD replay `big-leakfree-1ep-final8k-neighbour` alpha 0.9) + the flag, from epoch 26 | `wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon` |
| Scoring | leak-free (`--split-existing --deleted-cards neighbour`), w8+s8, 732 set (723 users cap 0.8 + 3 long users cap 0.92), `--inputs e2s-query` and normal; label `sm92-e26-dec1ep32k/w8+s8` | existing rows: SM31 `e2s-query`, SM72s normal (`small-lf-neighbour.jsonl`) |

Differences from the release besides the inputs flag: the start weights of the cool-down (epoch 26 instead of 25) and its update offset (it only enters the 5-update LR warm-up; both are > 500k, so the factor is 1). **Confound: the new model has one more constant-LR epoch than the release**; the normal-vs-normal row shows its size.

## Pre-registered rule (`sm92_rule.py`, fixed before launch)

722 users of the 732 set (those the big model (1 leak-free epoch) rows have), new - release both on `e2s-query`, review-weighted LogLoss, paired user bootstrap 2,000, seed 0, 95% CI; points = % of the big model (1 leak-free epoch) normal-input LogLoss.

- **ship candidate** if one head gains >= 0.05 points with its CI entirely below zero and no head is worse by >= 0.05 points with its CI entirely above zero;
- **no ship** otherwise.

Also reported: vs the big model (1 leak-free epoch) on normal and on `e2s-query`; vs the show-time big model on `e2s-query` (step 2 rows, `wt/sm91s2`); the new model's own show-time cost; new normal - release normal; the smaller-model bar (<= +0.5% curve / <= +0.4% instant) the three ways step 3 did; the 725-user 732+long pool. Self-test reproduces step 3's published "step 1's arm e2s - the released small model e2s" pair (+0.146 [+0.066, +0.230] / -0.062 [-0.161, +0.016]).

## CPU proofs (`sm92_check.py` -> `check-cpu-sm92.json`)

C1 epochs 21-25 rebuilt by the order rule == the protocols on disk; C2 the epoch-26 folder == epoch 25's protocol except the flag, the start, the epoch index, the run's epoch-26 order and paths (and refused against epoch 24); C3 the trainer / scorer files are byte-identical to `df9fee59`, where SM91's T1-T3 passed, and T1 + T2 (flag moves only the query intervals and curve query times; trainer rows == scorer `e2s-query` rows bitwise) re-run on 3 users of the epoch-26 protocol; C4 a cool-down prepared by this tree from epoch 25 with the flag + trim == the release's cool-down except the flag and names (the release against itself refused); C5 the rule self-test. Before the cool-down the launcher re-checks it against the release (`sm92_pair.py decay`).

## Schedule

The epoch is snapshot-resumable (`expertium_decay.py run`, snapshot stop <= 2 min before a window), so it starts once the smoke passes and pauses for the user's 17:30-19:30 window. The launcher skips the daily 20:00-01:00 break of 2026-11-03 only (date fixed in the script); quiet.json and the daily break apply from 11-04. GPU: smoke ~15 min, epoch ~111 min, cool-down ~520 min, scoring ~60 min.
