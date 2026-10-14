# Accuracy loop handoff (2026-10-14, ~17:00)

Branch `accuracy-loop` (merged into main at 26f2cb7). Rules: [docs/ACCURACY_LOOP.md](../../docs/ACCURACY_LOOP.md); results: [docs/ACCURACY_LOG.md](../../docs/ACCURACY_LOG.md); ideas: [docs/ACCURACY_IDEAS.md](../../docs/ACCURACY_IDEAS.md). Done and logged: screen baseline seeds 2025/2026, `fullrun-e10`, two-seed noise band, tier costs.

## Running detached (no session attached; the shared queue runs them in turn)

| Ticket | What | Runbook / log | Outputs |
|---|---|---|---|
| 000316 | A15: train `screen-w32-s2025-o1` and `-o2` (seed 2025, order seeds 1 and 2), then registers the third-seed training `screen-w32-s2027`, then one fast eval of all three, collections, reports, two `noise` runs | `runbook-2026-10-14b.sh` → `runbook-2026-10-14b-detached.log` | `experiments/screen-w32-s2025-o1`, `-o2`, `screen-w32-s2027`; `noise/screen-w32-s2025--screen-w32-s2026--screen-w32-s2027.json`, `noise/screen-w32-s2025--screen-w32-s2025-o1--screen-w32-s2025-o2.json` |
| 000317 | A08: fast eval of expertium-catch-up's epoch-10 decay checkpoint (`wt/expertium-catch-up/artifacts/expertium-catch-up/decay-8k-e10-q25/.../master.pth`, read-only) on the 1k screen users, then collection, report, compare vs `fullrun-e10` | `runbook-2026-10-14c.sh` → `runbook-2026-10-14c-detached.log` | `experiments/decay-e10-q25`; `comparisons/decay-e10-q25--vs--fullrun-e10.json` |

Earlier tickets 000307/000308 were cancelled before running and re-registered as 000316/000317 when the runbooks were detached. Check with `scripts/r01_queue.py status` and `tail` of the logs. If a runbook died, rerun only the missing steps below. Every step is resumable or idempotent.

## Once they finish (from the worktree or main; PY = main checkout's .venv python)

```
AL=scripts/accuracy_loop.py
$PY $AL status
# only if a runbook stopped early:
$PY $AL train screen-w32-s2027
$PY $AL evaluate screen-w32-s2025-o1 screen-w32-s2025-o2 screen-w32-s2027 decay-e10-q25
for n in screen-w32-s2025-o1 screen-w32-s2025-o2 screen-w32-s2027 decay-e10-q25; do $PY $AL collection $n --threads 2; $PY $AL report $n; done
# always:
$PY $AL noise screen-w32-s2025 screen-w32-s2026 screen-w32-s2027          # seed component (now with SD)
$PY $AL noise screen-w32-s2025 screen-w32-s2025-o1 screen-w32-s2025-o2    # order component (same training seed)
$PY $AL compare fullrun-e10 decay-e10-q25                                 # A08 paired delta, 1k users
```

Log update: paste each `report` row into ACCURACY_LOG "Runs"; add seed and order rows to "Noise bands" (values, range, SD and SD×√2 from the noise JSONs). Then replace the provisional 0.0004 decision band in ACCURACY_LOOP.md "Seed-noise band" with 2 × max(seed SD×√2, order SD×√2), and say which component dominates. Record the A08 row: decay vs `fullrun-e10` (curve and instant, weighted and user-avg, 95% bootstrap), collection Δ% vs fork. The decay job's own quick-300 result was −0.0018 curve and −0.0013 instant. Update the A08/A15 statuses in ACCURACY_IDEAS.md. Commit the small records with `git add -f` (experiment.json, result.json, phases.jsonl, collection.json, eval/per_user.jsonl, epoch result.json, noise/, comparisons/, logs, results.jsonl), never checkpoints.

## Open items

- Untested hooks: `env`, `init`, `checkpoint_transform`, `evaluator`, `epoch_runner`, `--epochs-per-job`. `--order-seed` is exercised by 000316.
- After expertium-catch-up lands: re-baseline the screen with WSD decay (A03), then screen and gate s64h (A17) and width 80 (A01).
- No fork-model held-out scores in this harness yet. The default scorer builds the expertium model; compression's held-out eval B covers the fork. Fork comparisons must use `fork_unseen` (id < 5000).
- Gate and final runs, and any job over 60 minutes, need an explicit human request (ACCURACY_LOOP.md rule 6).
