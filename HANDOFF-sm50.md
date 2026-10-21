# SM50 handoff (2026-10-21 ~21:00)

Head-only fine-tunes of SM36 (curve head / instant head), 0.22 epoch (81.3M reviews, 1,172 users), LR 1e-4 -> 0 one_minus_sin, Muon state continued.
Scoring: 732 pool minus 9 never-fit (users-732-fit.json), --lanes 1 (runbook), w8+s8.

- Prepared: artifacts/small-model/sm50/head-curve, head-instant (prepare-sm50.sh, OK lines checked).
- CPU checks: A, B (curve), C, B (instant) PASS (check-head-only.json, 'ALL OK'). Check D (option absent == old code, loaded-Muon LR fix)
  NOT verified: its 4 parallel probes ran out of memory; a sequential rerun (HEAD_ONLY_CHECK_SEQUENTIAL=1) was killed after ~70 min unfinished at 21:00.
  Rerun later: `HEAD_ONLY_CHECK_SEQUENTIAL=1 python -X utf8 scripts/check_head_only.py --decay artifacts/small-model/sm50/probe-curve --out artifacts/small-model/sm50/check-head-only-d.json --parts D` (CPU, not during a break, not during a GPU ticket's CPU-sensitive timing).
- Launcher (detached, waits for 01:00 itself, then curve arm then instant arm, 110 min expected each):
  `MIN_CURVE=110 MIN_INSTANT=110 nohup bash artifacts/small-model/sm50/launch-sm50.sh > artifacts/small-model/sm50/launch-sm50.log 2>&1 & disown`
- Logs: artifacts/small-model/sm50/launch-sm50.log, runbook-sm50-curve.log, runbook-sm50-instant.log.
- Done lines: `[sm50-curve ...] done` / `[sm50-instant ...] done`, final `[sm50-launch ...] all arms finished`; STOP/`ended with exit` = failure.
- Results: summary-curve.{json,md}, summary-instant.{json,md} (arm minus SM36, paired bootstrap; 732 and 464 pools).
- Resume: rerun launcher (finished fine-tunes skipped, scoring resumes). Do not edit scripts/ expertium/ rwkv/ in wt/sm50 while it runs.
