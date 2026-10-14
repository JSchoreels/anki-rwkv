# Accuracy loop handoff (2026-10-14, ~17:00)

Branch `accuracy-loop` (merged into main at 26f2cb7). Rules: [docs/ACCURACY_LOOP.md](../../docs/ACCURACY_LOOP.md); results: [docs/ACCURACY_LOG.md](../../docs/ACCURACY_LOG.md); ideas: [docs/ACCURACY_IDEAS.md](../../docs/ACCURACY_IDEAS.md). Done and logged: screen baseline seeds 2025/2026, `fullrun-e10`, two-seed noise band, tier costs.

## Running detached (no session attached; the shared queue runs them in turn)

| Ticket | What | Runbook / log | Outputs |
|---|---|---|---|
| 000316 | A15: train `screen-w32-s2025-o1` and `-o2` (seed 2025, order seeds 1 and 2), then registers the third-seed training `screen-w32-s2027`, then one fast eval of all three, collections, reports, two `noise` runs | `runbook-2026-10-14b.sh` → `runbook-2026-10-14b-detached.log` | `experiments/screen-w32-s2025-o1`, `-o2`, `screen-w32-s2027`; `noise/screen-w32-s2025--screen-w32-s2026--screen-w32-s2027.json`, `noise/screen-w32-s2025--screen-w32-s2025-o1--screen-w32-s2025-o2.json` |
| 000317 | A08: fast eval of expertium-catch-up's epoch-10 decay checkpoint (`wt/expertium-catch-up/artifacts/expertium-catch-up/decay-8k-e10-q25/.../master.pth`, read-only) on the 1k screen users, then collection, report, compare vs `fullrun-e10` | `runbook-2026-10-14c.sh` → `runbook-2026-10-14c-detached.log` | `experiments/decay-e10-q25`; `comparisons/decay-e10-q25--vs--fullrun-e10.json` |
| 000324 | A15 step 2, registered by runbook 14b: train `screen-w32-s2027` (third seed). Finished exit 0 after 13.2 min | `runbook-2026-10-14b-detached.log` | `experiments/screen-w32-s2027` |
| 000328 | A15 step 3, registered by runbook 14b: one fast eval of `-o1`, `-o2`, `s2027` on the 1k screen users (~19 min). Then the runbook runs the CPU collections, reports and both `noise` runs | `runbook-2026-10-14b-detached.log` | `noise/*.json`, per-experiment eval |

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

## Update after 000317 / A08 (2026-10-14)

000317 and `runbook-2026-10-14c.sh` finished (exit 0, 6.1 min eval). `decay-e10-q25` (epoch-10 checkpoint plus the 0.25-epoch WSD decay from expertium-catch-up) against `fullrun-e10`, 1,000 screen users, review-weighted with 95% user bootstrap:

| Subset | Curve Δ LogLoss | Instant Δ LogLoss | Better users (curve / instant) |
|---|---|---|---|
| all (1000) | −0.00136 [−0.00192, −0.00081] | −0.00103 [−0.00129, −0.00080] | 676 / 747 |
| fork_unseen (464) | −0.00152 [−0.00268, −0.00037] | −0.00116 [−0.00163, −0.00073] | 322 / 347 |
| fork_seen (536) | −0.00122 [−0.00162, −0.00088] | −0.00093 [−0.00119, −0.00070] | 354 / 400 |

Own collection, full layout (173,690 reviews), Δ% vs fork RWKV (curve 0.4417, instant 0.4366):

| Run | Curve LL (Δ%) | Instant LL (Δ%) |
|---|---|---|
| `fullrun-e10` | 0.4535 (+2.68%) | 0.4525 (+3.64%) |
| `decay-e10-q25` | 0.4536 (+2.70%) | 0.4518 (+3.49%) |

**Noise verdict.** Order noise (000324, the A15 `-o1`/`-o2` evaluation) is still pending: it waits for the GPU behind 000323-wide-kernels, so only the two-seed band exists (0.0001 to 0.0002; decision band provisionally 0.0004). Against that provisional band the gain is outside it: 3.4x on curve, 2.6x on instant, with consistent sign on both fork subsets and the quick-300 result (−0.0018 / −0.0013). Caveats: the bootstrap covers user sampling only, not training order; the full run's consecutive epochs moved 0.002 to 0.003 on quick-300, so an order band near that size would swallow the curve gain. The decay starts from the same weights and adds only the cool-down, which makes a pure order effect less likely but unproven. Collection is flat on curve (+0.0001) and 0.0007 better on instant, one user, not evidence either way. Verdict: outside the seed band, order band not yet measured, so "likely real, small".

**Recommendation (user decides).** Promote A08 to a full run: the cool-down is cheap (a fraction of an epoch from an existing checkpoint) and low-risk. Do not claim the gain as established until 000324 gives the order SD; if 2 x order SD x sqrt(2) exceeds 0.0014, treat it as unproven.

**Next steps.**
1. When 000324 lands: run the `noise` commands above, fill the Noise bands rows, replace the provisional 0.0004 band in ACCURACY_LOOP.md, and re-judge the A08 gain against it.
2. Re-baseline the screen with WSD decay (A03) before screening A17 and A01.
3. `per_user.jsonl` of `decay-e10-q25` stays uncommitted (per-user data); the aggregate comparison JSON is committed.


## Note (2026-10-14 20:45)
The runbook's `q()` does not pass `--expected-minutes`, so its tickets get the default 60 min. Add it in the next runbook. Don't edit runbook-2026-10-14b.sh while it runs, because bash reads scripts as it goes.

## Update: A08 promoted (2026-10-14)

User approved promoting A08. Main now has `final_decay_epochs` (default 0.25) in `scripts/run_full_training.py` (merge 1017383; `scripts/check_full_run_decay.py` is the CPU check). Status: promoted, provisional until A15 order band (000328). When 000328 lands, re-judge against the order SD; if the gain is inside the band, set the default to 0 (`FINAL_DECAY_EPOCHS`) and mark A08 accordingly. Next screens should use the cool-down as the baseline (A03 re-baseline).
