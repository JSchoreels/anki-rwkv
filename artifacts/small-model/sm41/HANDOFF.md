# The small model in the review simulator (SM41): handoff after stage 1

Branch `sm41`, worktree `C:/Users/jonat/workspace/wt/sm41` (cut from `sm82s2`). Stage 1 is done; nothing is running.
Result: [prototype-result.md](prototype-result.md); design: [DESIGN.md](DESIGN.md).

## Files

- `scripts/sm41_sim.py`: `check` (CPU proof), `run` (one user, every lane), `bench` (batched-step throughput), `report`.
- `prototype.jsonl` (5 users x 15 lanes), `prototype-tables.md`, `prototype-result.json`, `check-cpu.json`, `bench.json`,
  `run-<user>.log`.
- `work/prototype-v0-pool-replay.jsonl` and `work/prototype-v1*.jsonl`: earlier runs of the same lanes. The lanes they
  share with `prototype.jsonl` gave the same numbers.
- The learner's weights (read only, not committed): `C:/Users/jonat/workspace/wt/sm82c/artifacts/small-model/sm82c-head-released/master.pth`.

## Re-run

```
bash C:/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/run-prototype.sh     # ~20 min, 4 CPU processes
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
$PY -X utf8 C:/Users/jonat/workspace/wt/sm41/scripts/sm41_sim.py check --user 722 --rows 3000 \
  --out C:/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/check-cpu.json
```

## Open follow-ups

1. GPU batched-step benchmark (not run; the queue was full). One short ticket, launched hidden via WMI:
   `bash C:/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/gpu/bench-gpu.sh > C:/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/gpu/bench-gpu.out 2>&1`
   (ticket purpose "the small model in the review simulator (SM41): batched recurrent-step throughput benchmark ...",
   cap 0.8). It adds a `cuda-threads1` entry to `bench.json`; then run `sm41_sim.py report` and add the GPU line to
   prototype-result.md.
2. Stage 2 (see prototype-result.md "Full run"):
   - the drift study (50 users x 2-3 split points, first-touch check, mitigations);
   - intervals chosen with the instant head (K candidate query rows), plus the time-aware policy;
   - the vectorised lane engine with its batched == unbatched proof;
   - the full run.
