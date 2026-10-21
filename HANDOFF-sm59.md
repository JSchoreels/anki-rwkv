# SM59 handoff (2026-10-22 ~02:00): better distillation, step 1

Question: does a larger teacher weight in the KD mix (target = w * teacher + (1 - w) * label) help the small model?
Setup = SM50's (fine-tune of SM36, 0.22 epoch = 81.3M reviews, 1,172 users, LR 1e-4 -> 0 one_minus_sin, Muon state continued,
score 732 pool minus 9 never-fit, w8+s8) but BOTH heads trained (no head_only). Code = branch sm50 (identical), worktree wt/sm59, branch sm59.
- Effective weight: `decay_alpha` (the LR schedule decays from review 0, so `Teacher.targets(decaying=True)` always uses decay_alpha; `alpha` 0.9 is unused here).
  Arms: kd-control decay_alpha 0.5 (as SM50/SM36), kd-090 0.9, kd-100 1.0. decay.json diffs vs SM50 head-curve: only head_only removed + decay_alpha (+created_utc); protocol.json identical.
- 1.0 is valid: replay targets exist for every labelled row (a labelled row without a stored target raises RuntimeError); alpha in [0,1].
- Prepared with `DECAY_ALPHA=<w> FRACTION=0.22 NAME=<arm> bash artifacts/small-model/sm59/prepare-sm59.sh`. Dry run of runbook-sm59.sh for all 3 arms OK (runbook-sm59-dryrun-*.log).
- Check D (option absent == old code): NOT run yet. My attempt (HEAD_ONLY_CHECK_SEQUENTIAL=1, 01:30) needed ~17 GB RAM while SM55 ran (free 12 -> 3.5 GB), so I killed my own check processes (only mine). The launcher runs it itself after SM50 (needs >= 12 GB free RAM, no quiet window; ~70+ min), then the arms; if it fails no arm is queued.
  Optional: run it earlier in the quiet gap after SM55 ends (~07:05) and before SM50 starts (08:30), from wt/sm59:
  `HEAD_ONLY_CHECK_SEQUENTIAL=1 C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 scripts/check_head_only.py --decay artifacts/small-model/sm59/probe-curve --out artifacts/small-model/sm59/check-head-only-d.json --parts D`
  (the launcher then skips it when check-head-only-d.json has passed).
- Launcher (detached, started 01:5x): `artifacts/small-model/sm59/launch-sm59.sh`, log `launch-sm59.log`. It waits for `[sm50-launch ...] all arms finished` in wt/sm50 launch-sm50.log (or STOP / 'ended with exit' there),
  then check D, then kd-control, kd-090, kd-100 (110 min each, queue tickets, quiet_wait.py before each). Logs: runbook-sm59-<arm>.log.
- Done lines: `[sm59-<arm> ...] done` per arm (runbook log), `[sm59-launch ...] <arm> done` and `all arms finished` in launch-sm59.log. Failure: `[sm59-launch ...] STOP:`; `deferred: <arm>` = would hit a quiet window, rerun launcher later.
- Results: artifacts/small-model/sm59/summary-<arm>.{json,md} (arm - SM36, paired bootstrap, 732 and 464 pools), summary-table.md (3 arms + SM50 arms vs SM36; written at the end by compare-sm59.py; `python compare-sm59.py` reruns it).
- Resume: `cd C:/Users/jonat/workspace/wt/sm59/artifacts/small-model/sm59 && nohup bash launch-sm59.sh > launch-sm59.log 2>&1 & disown` (finished fine-tunes skipped, scoring resumes). Stop it with a file `STOP` in that folder. Don't edit scripts/ expertium/ rwkv/ in wt/sm59 while running.
