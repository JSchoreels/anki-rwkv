# EV01 fast evaluation: handoff (branch `eval-speed`)

## Update after 000312 (2026-10-14)

Ticket 000312-eval-speed ran `artifacts/eval-speed/gpu/job5.sh` (EV01 promotion P3 retry): two full `run_full_training.py run` smokes (50 train users, 1 epoch, 100 eval users), `fast_evaluation` false then true. Exit 0 after 328 s; both runs complete (`status: complete`). Results committed in e458927.

Campaign status: P1 (epochs 01-09, 300 users), P2 (upstream 2.76M baseline, 100 fork-unseen users) and P3 (full-run smoke) all pass the proposed tolerance.

P3 fast vs reference (aggregates only):
- Training bit-identical: same trained and initial parameter hashes, 63 updates, 2,044,361 review presentations.
- Same `validation_input_sha256`, identical label counts, benchmark sizes and coverage; result schema differs only by the fast timing fields.
- LogLoss and RMSE(bins) (both heads): max |delta| 0; RMSE 0; per-user AUC max 1e-6 (ahead) and 2e-6 (instant); R01 per-user losses max 4.2e-8; aggregate max 1.0e-7 (limit 1e-6); per-user AUC limit 1e-4. `tolerance.passes` is true. (Top-level `passed: false` in compare-eval.json only means "not bit-exact".)
- Eval wall time 232.3 s (reference) vs 37.2 s (fast), 100 users. INDICATIVE ONLY: not a `--timing` job, other GPU/CPU work could run alongside. Not a speed result; the speed evidence is the 300/2,000-user runs in docs/EVAL_SPEED.md.

## Pending user decisions (nothing flipped)
1. Adopt or reject `docs/proposals/eval-tolerance-rule.diff` (AGENT.md and GPU_RETRAIN_LOOP.md wording). If approved: `git apply docs/proposals/eval-tolerance-rule.diff` from the repo root, then remove the PROPOSED labels.
2. Whether full runs may default `fast_evaluation` to true (today opt-in via `run.json`). Training parity stays bit-identical; reference mode and `--batch-chunks 8` stay for bit-exact claims.

## Left before fast evaluation can become the default
- The two decisions above.
- EV08: width-80 Expertium variant (GRU curve head, branch `expertium-catch-up`) is unsupported; `evaluate_fast` refuses it and full runs fall back to the reference. Needs a variant `loss_terms`, then the tolerance check against `--mode reference`. Until then a default flip leaves those models on the slow path.
- Optional: a `--timing` job (quiet window) if a citable P3 eval-time figure is wanted. Main session command, only if desired: queue `artifacts/eval-speed/gpu/job5.sh` with the timing flag per the GPU queue policy.

## Evidence paths
- `artifacts/eval-speed/promotion/p3/{compare-runs.json,compare-eval.json,compare_runs.py}`, run dirs `run-reference/`, `run-fast/`, logs `*-retry.log`
- `artifacts/eval-speed/gpu/job5.sh`, `job5-queue.log`
- `artifacts/eval-speed/promotion/p1/`, `p2/`
- `docs/EVAL_SPEED.md` (campaign table, proposed rule), `docs/GPU_OPTIMIZATIONS.md` (EV01, EV07, EV08)

## Update 2026-10-14: decisions made
User approved both: tolerance rule applied (AGENT.md, GPU_RETRAIN_LOOP.md, PROPOSED labels removed) and `fast_evaluation` defaults to true in `write_protocol` (`scripts/run_full_training.py`). Reference mode stays via `fast_evaluation: false`; EV08 models fall back to reference. Only EV08 and an optional `--timing` job remain.
