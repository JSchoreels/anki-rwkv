# Benchmark labels for the 2,225-user set (EVAL): design note (not built)

Written 2026-11-01 ~09:50. Decision: the label build was **not** launched. It is cheap (CPU only, about 15-30 min), but no current model can be scored validly on the 2,225 set, so the labels would sit unused. Build it when a question on gate-tier checkpoints needs the 2,225 set; the steps below are ready.

## What "the label database" is

- `artifacts/full-run/run-8k-20261013/label-db` (LMDB, main checkout): for each user, `{user}_review_ths` (the benchmark test reviews: the 5 `TimeSeriesSplit` test folds after the FSRS-6 feature filter, `SHORT`/`SECS` on) and `{user}_rmse_bins` (their RMSE(bins) bin index). Writer: `rwkv/find_equalize_test_reviews.py` `process(user)`, CPU only, no fitting (pandas row loop). The 8k run built it for its 1,500 validation + 500 final-test users only (2,000 users).
- `wt/compression/artifacts/compression/label-db-b` (LMDB): the same writer via `scripts/compress_labels.py` (toml `labels-b.toml`, same dataset copy `data/anki-revlogs-10k-original-20261011`), built 10-14 for cohort B of the compression track (train + validation users with id < 5000, stratified). 2,120 users, 150.4M reviews, 5,159 s with 2 processes, i.e. about 69 µs of process time per review.

## Why the 2,225 set scored nothing (10-19)

`small_model_heldout.py` (small-model / sm77 lineage, line ~352) opens only the run's `label-db`. The 1,493 extra users of the 2,225 set are train users of the 8k split, so they have no key there: `artifacts/small-model/heldout-pools.jsonl` in `wt/small-model` holds 4,466 `error: no labels` rows (1,493 users x 3 labels) and only the 268 new validation users were scored (266-267 per label after OOM retries).

## Coverage today

- 2,225 set (`wt/small-model/artifacts/small-model/eval-users-2225.json`): 732 validation users (all in the run's label-db) + 1,493 train users of the 8k split with id < 5000 that are not in the gate tier.
- Of the 1,493: 787 are already in `label-db-b`, 706 are in no label DB (50.6M reviews; largest user 1891 with 1.33M reviews).

## Why it is not needed now (validity)

- The 2,225 set is valid only for models trained on the gate tier (5,000 users) or less (`eval_pools.py`, rule in the pool file).
- The released small model (the leak-free small model (SM72s), label `s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon`) has `training_users: 8000`, `train_users_from: tier final` (`wt/sm72s/.../decay.json`): all 1,493 extra users are its training users. The same holds for every final-tier model since the leak-free retrain (SM72), including the data-order re-seed (SM80) and scheduling-time vs review-time accuracy (SM81) runs. Scoring them on the 2,225 set would mix in-sample users into a held-out readout, so check 3 of the brief (score the released model on the 2,225 set) was not run.
- The big model (1 leak-free epoch) is trained on users 5000-10000, so for small-vs-big comparisons the 732 set (id < 5000 validation users) is already the largest pool held out from both models.
- Models the 2,225 set can still serve: gate-tier checkpoints, e.g. `s-w64kd-gate-ctx32768c-dec10`, `s-w64kd-gate-sc-{carry,zero,lanes}-dec12`, `s-w64kd-gate-sc-lanes-dec12-1ep32k(-muon)` (SM40v2, SM42 era), and the fork model (trained on 5000-10000).

## Larger valid pool for final-tier models (no label build needed)

For small-vs-small comparisons of final-tier models, all 1,500 validation users are held out (107.6M reviews; labels already in the run's label-db). The 768 users with id >= 5000 are seen only by the fork and big models, so this pool is invalid for any comparison against them. Cost about 37 min of GPU per w64 checkpoint (11 min per 31.9M reviews), about 20 min more than the 732 set. This needs a `--pool 1500` entry in `scripts/eval_pools.py` (small-model lineage); not built.

## Build steps (when a gate-tier question needs the 2,225 set)

1. Worktree: `git -C C:/Users/jonat/workspace/anki-rwkv fetch && git -C C:/Users/jonat/workspace/anki-rwkv worktree add C:/Users/jonat/workspace/wt/eval-2225 -b eval-2225 origin/main`.
2. Labels (CPU, <= 6 processes, below-normal priority via `set_low_priority()`, not during breaks): a copy of `scripts/compress_labels.py` with `DB = artifacts/eval/2225-labels/label-db`, the user list = the 1,493 extra users of `eval-users-2225.json`, `--processes 6`. Build all 1,493 fresh in one DB (one provenance; about 105.6M reviews, about 7,300 process-seconds, 15-30 min wall at 6 processes) rather than merging `label-db-b`. Resumable: `process()` skips users already present.
3. Proof before the build: write 5 validation users (e.g. small, medium, long) into a scratch DB with the same toml and compare both keys byte for byte with the run's `label-db`; also compare 5 of the 787 users against `label-db-b`. `rmse_bins` indices come from a `set` of float tuples, which iterates deterministically (float hashes do not depend on `PYTHONHASHSEED`), so bitwise equality is expected.
4. Scorer: give `small_model_heldout.py` (on the scorer's branch, not main: main has no held-out scorer) an extra label DB (`--label-db-extra PATH`, looked up per user when the run's DB has no key) and record the DB path in each row's provenance. The scored rows of users already scored stay comparable because their labels come from the run's DB unchanged.
5. Score (GPU queue, cap 0.8, 0.92 for long users): `POOL=2225 ONLY=<gate label> bash wt/small-model/artifacts/small-model/runbook-eval-pools.sh` with the extra DB; about 45 min per w64 gate checkpoint and 114 min for the fork model (`HANDOFF-eval-pools.md`), ticket purposes starting with "benchmark labels for the 2,225-user set (EVAL): ".
