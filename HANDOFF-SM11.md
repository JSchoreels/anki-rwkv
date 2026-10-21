# Handoff: SM11 full-history replay targets (2026-10-18)

The code is CPU-checked (`scripts/check_sm11_replay_targets.py`: ALL OK, 18 checks). Nothing is enqueued. Details are in `docs/SMALL_MODEL.md`, section "SM11 full-history replay targets".

## GPU ticket (not launched)

Run it from any directory, detached. It enqueues itself through `r01_queue.py` (owner small-model, claim gpu):

```
nohup bash C:/Users/jonat/workspace/wt/sm11/artifacts/small-model/runbook-sm11-replay.sh > C:/Users/jonat/workspace/wt/sm11/artifacts/small-model/runbook-sm11-replay.log 2>&1 & disown
```

- First, run `DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm11/artifacts/small-model/runbook-sm11-replay.sh`. It should log "DRYRUN OK" (`artifacts/small-model/runbook-sm11-replay-dryrun.log`).
- The runbook runs the CPU check (about 100 s; it needs ALL OK). It then loops queue tickets of `--expected-minutes 95`, each with this command:
  `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm11/scripts/sm11_replay_export.py export --out C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-gate5k --run C:/Users/jonat/workspace/wt/sm11/artifacts/accuracy/experiments/scale-w64kd-gate/run/run.json --device cuda --time-limit 5100`
  The loop ends when `status` exits 0 (all 5,000 gate training users written), and the runbook then runs `verify`.
- Expected: about 386M replayed reviews (369.5M plus 16.4M of burn-in for the 38 users above 600k reviews) at the scorer's 18.4–19.8k reviews/s. That is about 5.4–5.8 h of GPU, so 4–5 tickets of about 95 min. The export resumes per user, and between tickets the GPU is free for other queue work.
- To stop between tickets: `touch C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-gate5k/STOP`. Cancelling a running ticket loses only the user in progress.
- Output: `C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-gate5k`, which holds `manifest.json`, `users/<user>.npz` (float16 [reviews, 5]), `progress.jsonl` (one row per user, with errors and OOM retries), `status.json` and `verify.json`. Expected size: 3.70 GB.

## Verify after the run (CPU, Sonnet medium is enough)

1. `python -X utf8 scripts/sm11_replay_export.py status --out <OUT> --run <run.json>`: complete true, 5,000 users, `bytes_per_review` ≈ 10.0–10.3, 0 errors.
2. `verify.json`: ok true. It recomputes 8 sampled users of up to 20k reviews on the CPU in float32. The GPU float16 targets should match within 2.2e-4 on the curve and 3.5e-4 on the ratings, with the same NaN pattern. Run more users with `verify --sample 32` if needed.
3. In `progress.jsonl`: look at the rows with `"oom"` (passes halved) and the users with more than one pass. Quote the rate as reviews/s: the sum of `reviews` over the sum of `model_seconds`, and the wall time of the tickets.
4. Record the rate, size, OOM count and ticket times in the SM11 section of `docs/SMALL_MODEL.md` and in the backlog row.

## Using the targets in training

- Distillation spec: `{"teacher": "C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth", "alpha": 0.9, "decay_alpha": 0.5, "kd_source": "replay", "replay": "C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-gate5k"}`. Do not add "cache": the combination is refused.
- Training needs the sm11 code (`scripts/expertium_options.py` `ReplayTargets`) in the worktree that trains. Merge `sm11` into `small-model` only between GPU tickets, never while an epoch runs, because SourceWatch would kill it.
- Single-user batches only (no F29 packing). Users that are not in the export raise an error, so the 8k final run needs its own export (`--run` with that run's run.json, about 5.8 GB and about 9 h).
- First GPU use: a decay smoke (`expertium_decay.py smoke`) with the replay spec, then the SM21 v1 experiment, or an SM11-only decay screen paired with the same decay on window targets.

## Open points

- The GPU path is untested: the any-length recurrence on 600k-review training-user passes (the scorer did 599,569 on GPU), the worker RAM for the largest users (the 3.9M-review user's frame is held in the main process), and the real rate.
- Replay targets come from a float32 teacher, while the window teacher was bfloat16. Both changes are part of the recipe change.
- The replay mode adds one device-to-host copy of `batch.label_review_th` per update (a sync). It is small, but it has not been measured on GPU.
