# HANDOFF: bigger evaluation pools (732 / 2225 users)

Pools (scripts/eval_pools.py, artifacts/small-model/eval-users-{732,2225}.json; `python scripts/check_eval_pools.py` prints ALL OK):
- 464: historical default, unchanged order. 31.87M reviews.
- 732: every validation user of the 8k split with id < 5000 (new standard). 49.23M reviews; 268 users / 17.36M reviews new.
- 2225: 732 + 1,493 id < 5000 train users not in the gate training tier. 154.84M reviews; 1,761 users / 122.97M new.
  INVALID for any model trained on the final tier (all 8,000 train users). Sealed final_test_users are never included.

`small_model_heldout.py --pool {464,732,2225}` (default 464); `--users-file` still filters on top. The scorer caps GPU memory at
HELDOUT_GPU_MEMORY_FRACTION (default 0.8): do not run it next to a training job.

## Run (POOL=732 first; only users without a row yet are scored; rows -> artifacts/small-model/heldout-pools.jsonl)
```
DRYRUN=1 POOL=732 bash C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-eval-pools.sh   # plan, ends DRYRUN OK
nohup bash C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-eval-pools.sh > artifacts/small-model/runbook-eval-pools-732.log 2>&1 &   # with POOL=732 exported
POOL=2225 ...   # later
```
Resumable (rerun). ONLY=<label substring> restricts labels. The lanes label (sm21v2) is skipped until its dec12 checkpoint exists;
rerun then. One queue ticket per label and pass, <= 60 min each.

## Expected GPU minutes (tickets incl. 2 min fixed)
| label | 732 (268 users) | 2225 (1,761 users) |
|---|---|---|
| fork/w8+s8 | 18 | 114 (2 tickets) |
| s-w64kd-gate-ctx32768c-dec10 | 8 | 45 |
| sc-carry-dec12 | 8 | 45 |
| sc-zero-dec12 | 8 | 45 |
| sc-lanes-dec12 (when it exists) | 8 | 45 |
Total 732: 42 min (50 with lanes); 2225: 249 min (294 with lanes). Basis: w64 11 min per 31.87M reviews; fork 29 min per 31.87M
(measured: model 26.8 min on the 464 users).

## Readout
```
python -X utf8 scripts/eval_pools_report.py --pool 732 --files artifacts/small-model/heldout.jsonl artifacts/small-model/heldout-pools.jsonl \
  C:/Users/jonat/workspace/wt/state-carry/artifacts/small-model/heldout-sm21.jsonl --labels fork/w8+s8 s-w64kd-gate-ctx32768c-dec10/w8+s8 \
  s-w64kd-gate-sc-carry-dec12/w8+s8 s-w64kd-gate-sc-zero-dec12/w8+s8
```
(first label is the paired reference; add heldout-sm21v2.jsonl and the lanes label when scored.) It reports users missing per label.
