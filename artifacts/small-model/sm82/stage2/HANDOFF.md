# Response-time prediction (SM82) stage 2: handoff

Branch `sm82s2` (from `sm82c`), worktree `C:/Users/jonat/workspace/wt/sm82s2`. Launched Sunday 11-01 08:56 via WMI (hidden), launcher `artifacts/small-model/sm82/stage2/launch-sm82s2.sh` at commit c4ddf3a8.

## What runs

The frozen released small model + the answer-time head on richer frozen features (SM82c), queried "what if this card were reviewed after t days" (`scripts/sm82s2_curves.py query`): 150 users of the 732 set ([users.json](users.json), seeded), every test review on a later day than the card's previous one at its real interval (identity point = SM82c's scoring), and up to 3,000 random test reviews per user on 15 elapsed-time grid points (0.25x, 0.5x, 0.75x, 1x, 1.5x, 2x, 3x, 4x the real interval; 1, 3, 7, 14, 30, 90, 365 days). Only the review's instant query row changes (interval, cumulative interval, review-day encodings, day of week, the head's elapsed-days feature). CPU check PASS ([gpu/check-cpu.json](gpu/check-cpu.json)): C1 identity = SM82c's recorded row (LogLoss gap 1e-8 relative), C2 query rows independent (bitwise), C3 zero shift bitwise / shifted interval = the scaler's value.

Ticket purposes, in order (queue owner small-model):

1. `response-time prediction (SM82) stage 2: answer-time curve queries on the frozen released small model + the answer-time head on richer frozen features (SM82c): 150 users of the 732 set, 15 elapsed-time grid points (pass 1)`: queue 000822, started 08:56; the first attempt (000821, 08:54) failed in 3 s on a loader pickling error, fixed in c4ddf3a8. 80/150 users at 733 s; expected end ~09:25 (worst ~09:45).
2. the same with `(pass 2)` only if users are missing.

Logs: `C:/Users/jonat/workspace/wt/sm82s2/artifacts/small-model/sm82/stage2/gpu/launch-sm82s2.log` (last lines `queried users: N of 150` then `done (rc 0)`), `gpu/query.out`, `gpu/launch-sm82s2.nohup`. Outputs: `work/curves/query.jsonl` (one line per user: identity sums, timings) and `work/curves/users/<user>.npz` (~300 MB in all, not committed).

## Follow-up (CPU, not during 16:00-01:00; Sonnet medium is enough)

The analysis (`sm82s2_analyze.py`, tested on 15 finished users) sits in this folder because `scripts/` must not change while the ticket runs. After `done (rc 0)`:

```
cd C:/Users/jonat/workspace/wt/sm82s2 && git fetch && git merge --ff-only
git mv artifacts/small-model/sm82/stage2/sm82s2_analyze.py scripts/sm82s2_analyze.py   # commit + push (author Jonathan Schoreels)
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; S=artifacts/small-model/sm82/stage2
$PY -X utf8 scripts/sm82s2_curves.py analyze --out-dir $S/work/curves --result-dir $S --procs 6    # ~5-8 min, below-normal priority
```

It writes `stage2-result.md`, `stage2-result.json`, `stage2-workload-users.json`. Check the header line "Identity point (x1) vs the recorded time rows ...": max relative gap should be <= 1e-5 (GPU vs GPU, same code path) and same review counts True. Then replace the `TAKEAWAYS` placeholder in `stage2-result.md` with plain-language takeaways (what a scheduler could do with this): how much expected time and P(fast Good) change from 0.25x to 4x the interval; time per 0.1 recall drop; whether calibration holds in the high interval-growth buckets (the extrapolation check); the strongest fast-Good patterns; the workload B / A cost ratio by desired retention and whether the optimum moved (with the time limit, B vs A (head)). Commit `stage2-result.{md,json}`, `stage2-workload-users.json`, `gpu/*.log`, `gpu/*.out`, `gpu/*.json`, `work/curves/query.jsonl` (`git add -f`; not the `.npz` files), push. The main session updates the Factory page.

Preliminary, 15 users (not the result): recall and time calibrated at the observed intervals in every interval-growth bucket (recall within 0.005, time within ~1 s); E time | pass rises ~13% from 1x to 4x the interval; head cost per Good falls from ~17 s at recall 0.70 to ~10 s at 0.98; with no time limit, time per memorized card under recall-dependent costs is ~+9-11% vs fixed costs at desired retention 0.70-0.85 and ~-3% at 0.95; optima sit at the bottom of the range for most users under every cost model.

## Notes

- Fast = under 3.9 s (bins 0-6): the response-time prediction (SM82) and answer-time head on richer frozen features (SM82c) tables label that quantity "under 5.5 s", but bin 6 ends at 3,891 ms. A 5.5 s model curve needs a re-query storing bins 0-7 (~25 min GPU).
- Simulator: FSRS-6 with each user's own fit from `C:/Users/jonat/workspace/srs-benchmark/result/FSRS-6-recency.jsonl` (read only), costs from the head as a per-user per-grade cost curve over recall (the model itself is not inside the simulator).
- Stop between tickets: `touch .../stage2/gpu/STOP`. Relaunch (resumes, skips queried users): the WMI command of the SM82c handoff with `sm82s2/artifacts/small-model/sm82/stage2/launch-sm82s2.sh` and `.../stage2/gpu/launch-sm82s2.nohup`. Do not edit `scripts/`, `expertium/` or `rwkv/` of this worktree while it runs.
