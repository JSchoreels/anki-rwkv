# Small + Tiny ensemble (SM184): design (pre-registered 11-28, before any number was computed)

CPU only, no training. Reuses the per-review dumps of the deploy-time ensemble test (SM160): `wt/sm160/work/sm160-dump/{wfit,e732}`, label T24 (`w8+lr2.16.4q8x8@e2s-query`, the deployed format; REF is not used). Loader, LogLoss, logit clipping and paired user bootstrap (2,000 draws, seed 0, ratio of sums, 95% CI) are copied from `sm160_report.py`.

## Question
Does a logit ensemble of Small and Tiny beat Small Instant-Only (SM113) alone on the instant head, and does it reach Big Instant-Only (SM99)?

## Models
Big two-head (SM91s2), Big Instant-Only (SM99), Small two-head (SM118), Small Instant-Only (SM113), six-layer Tiny two-head (SM130), six-layer Tiny Instant-Only (SM130).

## Baselines
Small Instant-Only alone; Big Instant-Only alone; Tiny shipped ensemble (0.69 Tiny Instant-Only + 0.31 Tiny two-head, the SM160 weight); Small two-model ensemble (0.78 Small Instant-Only + 0.22 Small two-head, the SM160 weight); Big two-model ensemble (0.80, reference).

## Cross-size ensembles (logit average, weights fitted on the 100 training users `wfit` only, then read on the 732 set)
- (a) Small Instant-Only + Tiny Instant-Only: weight of Small on a 0.01 grid.
- (b) 4-way logit average of Small Instant-Only, Small two-head, Tiny Instant-Only, Tiny two-head: non-negative weights summing to 1, minimising the summed training LogLoss (SLSQP).
- (c) Small two-model ensemble (0.78) + Tiny two-model ensemble (0.69): mixing weight of the Small ensemble on a 0.01 grid.
- (d) reference: Big Instant-Only + Small Instant-Only: weight of Big on a 0.01 grid.

## Read-out
Review-weighted LogLoss change (%) of the instant head vs Small Instant-Only alone and vs Big Instant-Only alone, paired user bootstrap 95% CI, on users with all six dumps (the 732 list; Big's out-of-memory users drop out). Also mean within-user AUC (instant head). Report-only: curve head (ahead prediction of the two-head models), Small two-head + Tiny two-head with a fitted weight vs Small two-head alone and Big two-head alone.

## Go rule
An ensemble is "worth considering" if D vs Small Instant-Only alone is at most -0.05% and the CI upper bound is below 0. Also reported: whether it reaches Big Instant-Only (CI upper bound vs Big at or below 0 counts as reaching; a CI containing 0 counts as matching).

## Deploy cost
Running both sizes costs the per-review compute and card-state bytes of Small + Tiny together (parameter counts / state bytes from the RESULT files if found, else about 1.5x Small).
