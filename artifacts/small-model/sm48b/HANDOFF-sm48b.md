# SM48b handoff

Branch `sm48b` (from `small-model`), worktree `C:/Users/jonat/workspace/wt/sm48b`. **Do not edit source in this worktree while ticket 000592 runs** (the job runs from this checkout).

## Done
- Lane-safe int variants: `scripts/state_vq.py` `SCORE` is per thread (`_ThreadSlots`); `small_model_heldout.py` no longer locks IntState variants under `--lanes` (`HELDOUT_INT_LANE_LOCK=1` restores the lock for an A/B). Scorer prints peak torch allocated / reserved at the end.
- `--exclude-users-file` (JSON with `users`), `artifacts/small-model/never-fit-users.json`, docs in `docs/SMALL_MODEL.md` (SM48b section).
- CPU checks (all bitwise): lanes 2 vs 1 with w8+s8hx8 / w8+s4c8nhx8 on the emulated GPU path (`check-cpu-int-lanes1-vs-lanes2.json`, 8/8); exclude file leaves the other 3 users' rows identical (`cpu-exclude.jsonl` vs SM48's lanes-1 rows). The CPU checks ran just before two env-gated / print-only edits (lock A/B, peak-memory print); they do not touch the arithmetic.

## Queued
- Ticket 000592 (small-model, gpu, --timing, 35 min), launcher `launch-sm48b.sh`, log `launch-sm48b.log`; job `job-sm48b.sh` (7 runs: 1-l1, 2-l2, 3-l4, 4-l4b250, 5-l4b150, 6-l4lock, 7-l1; 40 SM48 timing users, SM36, 4 variants w8+s8, w8+s4c8n, w8+s8hx8, w8+s4c8nhx8; every run compared with run 1).
- Done when `launch-sm48b.log` shows `[sm48b ...] done` then `queue exit`. `FAIL:` lines = a compare outside tolerance (job continues); `STOP` = a run failed.

## Follow-up (fresh Sonnet medium agent)
```bash
cd C:/Users/jonat/workspace/wt/sm48b
grep -aE '^\[sm48b|\[queue\]' artifacts/small-model/sm48b/launch-sm48b.log | tail -20
cat artifacts/small-model/sm48b/timing/summary.md
git add -f artifacts/small-model/sm48b/launch-sm48b.log artifacts/small-model/sm48b/timing/{summary.md,summary.json,runs.jsonl,compare-*.json,run-*.log}
# fill the measured numbers into docs/SMALL_MODEL.md (SM48b) and docs/GPU_OPTIMIZATIONS.md, commit, push
```
Pick the recommended `--lanes` / `--lane-review-budget`: fastest run with 0 error rows, bitwise compare and nvidia-smi max <= run 1-l1's. The 0.8 cap is enforced by torch (13.04 GB), so a lane run that exceeds it shows as OOM error rows; nvidia-smi adds context and per-stream workspaces on top (SM48: lanes 14.5 vs 13.1 GiB).
