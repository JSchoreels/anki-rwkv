# Deploy-time ensemble of the two instant predictions (SM160): design

Written Tue 11-24 about 12:00 (machine clock), before any row or per-review dump of this run existed. Branch `sm160`, worktree `C:/Users/jonat/workspace/wt/sm160`, cut from `origin/sm158` (821f4845, the result commit of output-aware T24 on the 732 set (SM158)). No source change: the scorer `scripts/small_model_heldout.py` is SM157's / SM158's, used with its existing `--dump` option. GPU scoring plus a CPU read-out; no training.

## Idea

Idea 9 of the 11-23 opportunity list. In the fork's default mode each size runs both of its models on every review: the two-head model (curve and instant heads) and the Instant-Only model. The instant prediction shown is the Instant-Only model's; the two-head model's own instant prediction is computed and thrown away. Averaging the two in logit space costs nothing extra at deploy time (one weighted sum and one sigmoid per review, no extra model run, no extra stored state).

## Question

Per size (Big, Small, six-layer Tiny), at the shipped state format T24 on show-time inputs: does a fixed logit average of the two instant predictions, with one weight per size fitted on training users only, have a lower instant-head LogLoss than the Instant-Only model alone, by at least 0.05%?

## Models (`models.sh`, sha256 pinned and checked by the launcher)

| Size | Two-head model (TH) | Instant-Only model (IO) |
|---|---|---|
| Big | Big two-head (SM91s2) `sm91s2-big-showtime` | Big Instant-Only (SM99) `sm99-big-inst-e2` |
| Small | Small two-head (SM118) `sm118-e26-dec2ep32k-t91s2` | Small Instant-Only (SM113) `sm113-inst-e35st-st-dec2ep32k-t99` |
| Tiny | six-layer Tiny two-head (SM130) `sm130-th-dec2ep32k-t91s2` | six-layer Tiny Instant-Only (SM130) `sm130-io-dec2ep32k-t99` |

## Labels (`labels.sh`)

| Name | Label | Role |
|---|---|---|
| T24 | `w8+lr2.16.4q8x8@e2s-query` | the shipped format (rank 2 below 16 answers, rank 4 from 16, int8 factors): **the decision** |
| REF | `w8+s8hx8@e2s-query` | int8 card and note rows: information only, scored after T24 (dropped if the chain is stopped) |

Scorer settings as the low-rank int8 states test (SM148) and SM158: show-time inputs `--inputs e2s-query`, leak-free `--deleted-cards neighbour`, `--nonfinite-rows`, `--split-existing`, `--prior-buckets 0,1,3,5,9,16,32,128,300,1000`, `PYTORCH_JIT=0`, plus `--dump` (per benchmark review: `review_th`, `bin`, `y`, `p_ahead`, `p_instant`, into `work/sm160-dump/`, never committed). The rows should equal SM148's rows of the same labels; the smoke checks that on 3 users.

## Users

- Evaluation: the 732 fit list (`C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json`, 723 users; Big 718, the 5 largest never fit the GPU scorer, as in SM148).
- Training users for w (`users-wfit.json`, `sm160_report.py users`): 100 users drawn with `random.Random(160).sample` from the 730 validation users of the 8k split with id >= 5000 and at most 300,000 reviews (the 1,500 pool minus the 732 set; none is in the 732 set, the 2,225 decisive pool, or the sealed final-test list; the script checks this). 5,548,901 reviews, 5,323 to 291,371 per user, median 29,090. These are the only users with benchmark labels outside both evaluation pools. They are held out from the Small and six-layer Tiny models' training; for the two Big models they are training users of the fork the Big models start from (as for the training users of SM157), so the fitted Big weight may lean toward whichever Big model fits seen users better. The oracle weight on the evaluation users (report only) shows how far the fitted weight is from the best one.
- Smoke: users 2388, 4676, 1866 (`users-smoke.json`, SM158's).

## Ensemble

For each benchmark review: `E(w) = sigmoid(w * logit(p_IO) + (1 - w) * logit(p_TH))`, where p_IO is the Instant-Only model's instant prediction, p_TH the two-head model's instant prediction, and logits are taken of p clipped to [1e-7, 1 - 1e-7]. w is the weight of the Instant-Only model; w = 1 is today's deploy. Both models' dumps must have identical review sets and labels (checked per user).

Fit (`sm160_report.py fit --label L`, run by the launcher after the training users are scored and BEFORE any 732-set row of L exists): for each size, w on the grid 0.00, 0.01, ..., 1.00 that minimises the summed per-review LogLoss on the training users (review-weighted); fixed in `gpu/wfit-L.json`. The same fit for the probability average `w * p_IO + (1 - w) * p_TH` is reported, not decided on.

LogLoss per review as `sklearn.metrics.log_loss` (p clipped to the float64 epsilon, `xlogy`); the dumped predictions must reproduce every row's benchmark instant LogLoss within 2e-6 (rows are rounded to 6 decimals), else the read-out is invalid.

## Rules (pre-registered; `sm160_report.py`, its selftest passes)

Statistic: % = 100 x (sum over users of the ensemble's benchmark instant LogLoss x size / the same for the Instant-Only model alone - 1), review-weighted, paired user bootstrap 2,000 draws, seed 0, 95% CI, on the rule users of the size: users of the 732 fit list with a finite row and a dump for both models of the size.

**Go for a size** (all of): D = % of E(w_fit) against IO alone on the instant head, at T24, with
- (a) the CI upper bound of D below 0 AND D at most -0.05% (an improvement of at least 0.05%);
- (b) at least 715 rule users and no non-finite row for either model of the size;
- (c) every dump reproduces its row (above).

Each size is decided separately. A go means the fork's default mode could show E(w_fit) instead of the Instant-Only model's instant prediction for that size (a later, separate change; nothing ships from this test). The curve head is untouched by construction.

Report only (decide nothing): D at w = 0.5; D of the probability average at its fitted w; the oracle w on the evaluation users and its D; the two-head model's instant head alone against IO alone; E(w_fit) against the two-head model's instant head; the training-user curve of the logit average (aggregate % at w = 0, 0.1, ..., 1); the same tables at REF with its own fitted w (`gpu/wfit-REF.json`); GPU minutes per ticket.

## Run (`launch-sm160-gpu.sh`, detached)

`scripts/r01_queue.py` tickets (owner sm160, claim gpu, `HELDOUT_GPU_MEMORY_FRACTION` 0.8, 0.92 only for memory-only retry passes after a pass without progress). Order: preconditions (files, clean source, model sha256 pins, report selftest, training users unchanged); GPU smoke (3 smoke users x 6 models x T24 and REF with the dump: rows within 1e-5 of SM148's GPU smoke rows per user and head, dumps reproduce the rows, equal review sets within each size); then for T24 and afterwards REF: the training users (one ticket, 6 models, `--pool 1500 --users-file users-wfit.json`), the w fit, then the 732 set one ticket per size (Tiny, Small, Big), passes until every user fits. Quiet windows (`quiet.json`) are respected. Expected: smoke under 20 minutes; per label about 45 minutes of training users plus 40 / 55 / 95 minutes for Tiny / Small / Big (SM148's wall minutes per plain label x 1.3 for the dump), a budget of about 4 hours per label; SM148's actual wall times (17 / 25 / 48 minutes per label) x 1.3 suggest about 2.5 to 3 hours per label.

## Read-out (a later, fresh agent)

`sm160_report.py report` -> `sm160-result.{json,md}` (exit 0 decided, 2 pending, 3 invalid), then `RESULT.md`.

## Files

`models.sh`, `labels.sh`, `users-wfit.json`, `users-smoke.json`, `sm160_report.py`, `launch-sm160-gpu.sh`, `design.md`; outputs `gpu/` (rows, tickets, fits, logs), `sm160-result.{json,md}`, `RESULT.md`, `HANDOFF.md`. Not committed: `work/sm160-dump/` (per-review values).
