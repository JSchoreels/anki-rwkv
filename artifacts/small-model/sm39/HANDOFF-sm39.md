# HANDOFF: SM39 arm 1 (Muon LR x0.5 decay) + SM51 (snapshot averaging), launched 2026-10-21 08:58, waiting for SM55

Worktree `C:/Users/jonat/workspace/wt/sm39` (branch `sm39`, from `small-model`). Code commit f8005ce5. Do not edit `scripts/ expertium/ rwkv/` here until the runbook is done (SourceWatch; the runbook also refuses a dirty tree).

## What runs
- Launcher `launch-sm39-after-sm55.sh` (detached, log `artifacts/small-model/sm39/launch-sm39.log`) waits until SM55's runbook log (`wt/sm55/artifacts/small-model/runbook-2026-10-21-sm55.log`) ends with its final `[sm55 ...] done` or a `STOP` / `ended with exit` line, idle 5 min, with no live SM55 queue ticket (or `wt/sm55` gone). Then it runs `runbook-sm39.sh` (log `artifacts/small-model/sm39/runbook-sm39.log`).
- Overrides: `touch artifacts/small-model/sm39/GO` = start now; `touch artifacts/small-model/sm39/STOP` = exit, nothing enqueued.
- The runbook waits out quiet windows itself before every ticket (`quiet_wait.py`: queue rule; if quiet.json cannot be parsed it falls back to daily 21:00-01:00 and Sunday 10-25 16:00-01:00). Breaks: if SM55 finishes ~19:20 today, the decay waits to 01:01 and runs ~01:01-06:10.

| Step | Ticket (owner small-model, claim gpu) | Expected min | Output |
|---|---|---:|---|
| CPU: prepare + `check_decay_snapshots.py cpu` | none (~2 min CPU) | — | `scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon0p5/`, `sm39/check-decay-snapshots-cpu-run.json` |
| GPU snapshot smokes (off, off, on bitwise) + decay | "SM39 arm 1: SM36 Muon decay with Muon LR x0.5 + 4 weight snapshots" | 335 (SM36: 306 measured) | `sm39/check-decay-snapshots-gpu.json`, `.../expertium/E89-F65e-F77a/master.pth` + `snapshots/users-01250, -02500, -03750` |
| CPU: SM51 averages | none (seconds) | — | `sm39/avg/avg2, avg3, avg4/master.pth` (+ `average.json`) |
| Scoring, final + avg2/3/4, 732 pool minus 9 never-fit (723 users), w8+s8 | "SM39 scoring arm 1 final + avg2/3/4 (SM51) ... (pass 1)" | 70 | `artifacts/small-model/heldout-sm39-pools.jsonl` |
| Scoring resume (only if users missing) | "... (pass 2)" | 25 | same |
| CPU: summary | none | — | `sm39/sm39-summary.md`, `sm39-summary.json` |

Ticket IDs are assigned at enqueue time; read them from `runbook-sm39.log` (`[queue] NNNNNN-small-model registered`).

## Done / failure lines
- Done: `[sm39 MM-DD HH:MM:SS] done` in `artifacts/small-model/sm39/runbook-sm39.log` (after the `[sm39-summary]` table); the launcher then logs `runbook exit 0`.
- Failure: `[sm39 ...] STOP: ...` or `[sm39 ...] ended with exit N` (e.g. GPU snapshot check failed: see `sm39/check-decay-snapshots-gpu.json`; an off-vs-off mismatch there means trainer non-determinism, not snapshots).
- Resume: `nohup bash C:/Users/jonat/workspace/wt/sm39/artifacts/small-model/sm39/runbook-sm39.sh >> C:/Users/jonat/workspace/wt/sm39/artifacts/small-model/sm39/runbook-sm39.log 2>&1 & disown` (finished decay, averages and scored users are skipped). Summary only: `SUMMARY_ONLY=1 bash .../runbook-sm39.sh`.

## Follow-up (fresh Sonnet medium agent)
Read `sm39/sm39-summary.md`: rows final / avg2 / avg3 / avg4 x pools 732 (723 users) and 464 subset (458 users), each with curve and instant LogLoss, % vs fork w8+s8, delta vs SM36 w8+s8 with user-bootstrap 95% CI. Reference on the same users: SM36 +0.64% / +0.63% (723), +0.74% / +0.64% (458). Report decay reviews/s and training seconds from `.../expertium/E89-F65e-F77a/result.json` (`reviews_per_second`, `training_seconds`). Commit `heldout-sm39-pools.jsonl` with the daily heldout commit, the summary, logs and check JSONs (`git add -f`), push.

## Design notes
- Optimizer: `{"kind": "muon", "lr_scale": 1.7857142857142858, ...}` = `expertium-iter53` with lr_scale x0.5 (CPU check: the dict at SM36's scale builds exactly iter53). AdamW is not tied to lr_scale. The Muon weight decay is stored as wd / lr_scale, so the per-step decay is unchanged (Expertium's AdamW-rate decay); only the Muon step halves.
- Snapshot points are user boundaries (1,250 / 2,500 / 3,750 of 5,000 users = 26.2 / 50.3 / 77.2% of the reviews), the trainer's existing `snapshot_users` mechanism; no trainer change.
- The 464 readout comes from the 732 rows: the 464 pool is inside the 732 pool and SM36's 464-file and pool-file rows are identical on 458 common users (max diff 0).
