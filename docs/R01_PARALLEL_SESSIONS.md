# Parallel R01 preparation sessions

Several agent sessions may work in this same checkout at once (no worktrees: data, checkpoints, artifacts and `.venv` live here). Read this before editing anything, and follow it together with [AGENT.md](../AGENT.md). The work comes from the [proposed loop changes](R01_NEXT_ROUND_PLAN.md#proposed-loop-changes-2026-10-13-pending-human-approval).

## Sessions and owned files (2026-10-13)

| Session | Work | Owns (may edit freely) |
|---|---|---|
| `r01-turnaround` | P-F (F74): shortest possible iterations by reusing stored results (evaluation reuse, result store, merged full run), plus P-D variant cap and GPU-time accounting; measurement-design changes stay opt-in until approved | `scripts/r01_iterate.py`, `scripts/r01_replicates.py`, `scripts/r01_replicate_stats.py`, `scripts/r01_close_round.py`, `scripts/r01_paired_timing.py`, `scripts/r01_metrics.py`, `scripts/r01_preflight.py`, `scripts/r01_evaluation_reuse.py`, new `scripts/r01_result_store.py`, their `scripts/check_r01_*` tests, `artifacts/retraining/gpu-retrain-r01/result-store/`, `artifacts/diagnostics/r01-turnaround-*` |
| `r01-harness` | P-A exploration-lane records and P-B multi-seed non-inferiority tooling; every new rule is opt-in and inactive until the human approves it | new `scripts/r01_seed_gate.py` and `scripts/r01_exploration.py`, their `scripts/check_r01_*` tests, `artifacts/retraining/gpu-retrain-r01/exploration/`, `artifacts/diagnostics/r01-harness-*` |
| `r01-structural` (F05 part) | P-C: F05 CUDA Graphs on the R01 path, exactness checks and a ready-to-run candidate | `scripts/gpu_shape_buckets.py`, `scripts/check_gpu_shape_buckets.py`, `scripts/check_gpu_bucket_admission.py`, new `scripts/gpu_r01_graphs.py` and `scripts/check_r01_cuda_graphs.py`, `artifacts/diagnostics/r01-f05-*` |
| `r01-structural` (F69 part) | F69: chunked (parallel-scan) WKV7 recurrence prototype, numerical-difference and microbenchmark evidence | new `rwkv/model/chunked_wkv*.py`, new `scripts/kernels/*chunked*`, new `scripts/check_r01_chunked_wkv.py` and `scripts/bench_r01_chunked_wkv.py`, `artifacts/diagnostics/r01-f69-*` |

Not owned by any of these sessions (read only unless the human says otherwise): `AGENT.md`, `AGENTS.md`, the baseline-drift folder `artifacts/diagnostics/r01-native-toolchain/baseline-drift/` ledger tracks F70–F73 and iteration folders from 0074 on (the separate `baseline-drift-f71` chat, which follows these claims), recorded iteration folders, checkpoints and `data/`. Another session's owned files are also read only; send that session a message instead of editing them. The queue and claim helpers (`scripts/r01_queue.py`, `scripts/r01_claim.py`, `scripts/check_r01_queue.py`) are shared: change them only under the `plan` claim and rerun `scripts/check_r01_queue.py --output artifacts/diagnostics/<new dir>` before committing.

## Shared files need a claim

Claims are atomic directories managed by `scripts/r01_claim.py`; `status` shows who holds what. Hold a claim only for the edit-and-commit (or the GPU job) it covers, then release it.

| Claim | Covers |
|---|---|
| `training-source` | Every file in `R01SourceWatch` in `scripts/run_gpu_r01.py` (including `scripts/gpu_r01_training.py`, `scripts/gpu_r01_prefetch.py` and its option allowlist, `scripts/gpu_r01_policy.py`, `scripts/gpu_optimizations.py`, `scripts/run_gpu_r01.py` itself), `scripts/report_gpu_r01_next.py` (`describe()` names), `scripts/check_gpu_r01_policy.py`, and `docs/GPU_RETRAIN_LOOP.md` (the runners' live LOOP guard reads it, so any uncommitted edit there blocks launches) |
| `gpu` | Any process that uses the GPU: checks, kernel builds and tests, microbenchmarks, smokes, timing runs |
| `ledger` | `docs/GPU_OPTIMIZATIONS.md`, `docs/GPU_OPTIMIZATIONS_LOG.md`, `docs/R01_HOST_PATH.md`; edit only your own track rows or lines |
| `plan` | `docs/R01_NEXT_ROUND_PLAN.md`, this file |
| `git` | Staging and committing |

- Run every GPU job as one background job through the queue: `.venv/Scripts/python.exe scripts/r01_queue.py run --owner <session> --purpose "<what>" --claim gpu --claim training-source -- <command>`. It registers a ticket, blocks on a directory-change notification (no polling) until every earlier ticket that needs one of the same claims has finished and the claims are free, prints one progress line per minute while waiting, runs the command and releases the claims when it exits; the session is notified when the background job ends, so do not poll `status` meanwhile. `--max-wait-minutes N` gives up with exit code 75; `enqueue` reserves a place now for a command attached later with `run --ticket <id>` (reservations expire after `--hold-minutes`, default 30); `status` shows the running ticket, the waiting order and claims held outside the queue; `cancel` drops your own ticket; a wrapper still waiting on it exits with code 76 without running its command (and is terminated if still alive after 5 s), while a ticket that is already running keeps its command, so stop that background job itself. A wrapper that dies takes its command with it and its ticket is removed by the next waiter, so a crashed session cannot block the queue; `artifacts/coordination/queue/history.jsonl` records every finished ticket. `scripts/r01_claim.py run` still works and waits in the same queue. Holding `training-source` keeps the live source stable for checks that reject source changes mid-run; the claim is released when the command exits, even on failure. GPU quiet windows: while `artifacts/coordination/queue/quiet.json` lists a window (`[{"start": "2026-10-14T21:00", "end": "2026-10-15T01:00", "reason": "..."}]`, local time), a `gpu` ticket does not start inside it, nor when its `--expected-minutes` (default 60) would reach into it. Running jobs are never interrupted, `status` prints the window, and deleting the entry lifts it. Give long jobs a real `--expected-minutes` so they wait for the window to end instead of being held back early. Timing measured while any other GPU or heavy CPU process runs is invalid: before timing, also check `nvidia-smi` for jobs from chats that do not follow this file.
- Edit shared files with `scripts/r01_claim.py acquire` (with `--wait-minutes` it waits in the same queue; without it, it fails at once when the claim is busy), make the edit, commit, then `release`. Claims are taken in a fixed order by the helper, so pass every claim you need in one call. Never break someone else's claim; only the human may use `--force`.
- Before any GPU job, look for a recorded result you can reuse (the result store once `r01-turnaround` has built it, otherwise compatible recorded checks and runs as AGENT.md requires). Record every new result so that later work can reuse it instead of repeating it.
- Work on your own new files while you wait. Wire new code into shared training source only once it passes its standalone checks.

## Git

- Commit only your own paths: `r01_claim.py run --claim git --owner <session> --purpose commit -- git commit -m "<message>" -- <paths>`. Never `git add -A`, `git add .`, `git commit -a`, `git stash`, `git reset`, `git checkout -- <file>`, rebase or amend, and never push.
- Before committing a shared file, run `git diff -- <file>`; if it contains hunks you did not write, do not commit that file. Leave your edit uncommitted and report it instead.
- Use `git -c user.name=<session> -c user.email=noreply@anthropic.com` and the attribution line required by your instructions.

## Scope for these sessions

Allowed: implementation, CPU checks, exactness and kernel checks, microbenchmarks and smokes under the claims above, ledger rows for your own tracks, and proposals. Not allowed without a new explicit human request: counted R01 iterations or rounds, replicate-block campaigns, broader validation, administrator prompts, system or driver setting changes, and activating any pending rule from the proposal. Finish with a short report: what was built, evidence paths, commits, and the exact next command a counted iteration would run.
