# SM48 handoff: faster held-out scoring

Branch `sm48` (from `sm46`) in worktree `C:/Users/jonat/workspace/wt/sm48`, code commit be20647f. **Do not edit source in this worktree until the job has finished**: the queued job runs from this checkout (8 scorer processes, each reads the scripts at start).

## What was done

- Profile: [profile.md](profile.md). Only 51% of the scorer's wall time on SM45's 732-pool pass was the model call. The other 49%, ~10.8 µs per review-evaluation, was Python loops over labelled reviews (dict path: 8.70 µs per review and variant on CPU). The model call is bound by serial preset/user/deck WKV chains on 4 blocks of 16 threads.
- `scripts/heldout_fast_metrics.py`: numpy prediction collection and benchmark metrics, the scorer's new default (`--legacy-metrics` = old path). It costs 0.85 µs per review and variant. Bitwise checks:
  - 7 users' real labels (`check-fast-metrics.json`);
  - scorer vs the pre-SM48 scorer, 4 users x (w8+s8, w8+s4c8n), 8 of 8 rows (`check-cpu-orig-vs-array.json`, `check-cpu-orig-vs-dict.json`).
- `small_model_heldout.py --lanes N --lane-review-budget R` (CUDA): N users at once, threads with their own model copies and CUDA streams. Users start only while the reviews in flight stay <= R (default 400k, ~25 KB GPU memory per review). +vq variants run one at a time. The CPU scheduler check (fp32/w8) is bitwise (`check-cpu-lanes1-vs-lanes2.json`). Not yet run on the GPU.
- Files touched (for the merge with `sm47`, which also edits `small_model_heldout.py` / `state_vq.py`):
  - `scripts/small_model_heldout.py`: main loop refactored into `make_model` / `score_user` / lane scheduler; `LongWKV` bits now thread-local; 3 new flags.
  - new: `scripts/heldout_fast_metrics.py`, `scripts/check_heldout_fast_metrics.py`, `scripts/compare_heldout_rows.py`.
  - `state_vq.py` is unchanged.
  - Rows gain a `scorer` field (`{"metrics": "array"|"dict", "lanes": N}`).

## What is queued

- Launcher (detached, started 2026-10-20 19:31): `artifacts/small-model/sm48/launch-sm48.sh`, log `launch-sm48.log`.
  1. It waits for the SM47 launcher's ticket to end: the line `[sm47 ...] queue exit` in `C:/Users/jonat/workspace/wt/sm45/artifacts/small-model/sm47/launch-sm47.log`. SM47 itself waits for SM46, which waits for SM44.
  2. Then it waits while `C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/HOLD-chain` exists.
  3. Then it enqueues ONE ticket: owner small-model, claim gpu, `--timing`, `--expected-minutes 30`, launched by "detached launcher sm48". The queue holds it in quiet windows (break 2026-10-20 22:00 to 2026-10-21 01:00).
- Job `job-sm48.sh`:
  - Users: 40 users of the 732 pool (`timing-users.json`: evenly spaced over sizes, users <= 450k reviews so none OOM; 2,255,532 reviews). Model: SM36, w8+s8 + w8+s4c8n. Default memory cap 0.8, 4 prep workers.
  - Arms: `old` (`--legacy-metrics`, the pre-SM48 path), `arr` (numpy metrics), `l2` / `l4` (numpy + `--lanes 2` / `4`). Order: old arr l2 l4 l4 l2 arr old.
  - nvidia-smi samples GPU utilisation and memory every 0.5 s for each run.
  - Each run is compared with run 1-old (`compare_heldout_rows.py`, AGENT.md tolerance; bitwise pair counts recorded). The job logs `STOP` on a failed run or compare.
  - At the end, `sm48_summary.py` writes `timing/summary.md` / `summary.json`, then the job logs `[sm48 ...] done`.
- Estimate: old ~100 s scoring per run, faster arms less, plus ~30 s startup per run, so ~12-20 min. The ticket asks for 30.
- Not done: no multi-rank (SM46-type) codec kernel was built, so the SM46 timing part was skipped. See profile item 4 / track SM46-vq.

## Follow-up (fresh agent, Sonnet medium is enough, when `launch-sm48.log` shows `[sm48 ...] done`, `STOP` or `queue exit`)

```bash
cd C:/Users/jonat/workspace/wt/sm48
grep -aE '^\[sm48|\[queue\]' artifacts/small-model/sm48/launch-sm48.log | tail -25
cat artifacts/small-model/sm48/timing/summary.md
# re-run the summary by hand if needed (CPU, seconds):
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm48/sm48_summary.py
git add -f artifacts/small-model/sm48/launch-sm48.log artifacts/small-model/sm48/timing/{summary.md,summary.json,runs.jsonl,compare-*.json,run-*.log}
git -c user.name="Jonathan Schoreels" -c user.email=jonathan.schoreels@gmail.com commit   # message ends with a blank line + Co-Authored-By line
git push
```

Then:
- Fill in the measured review-evals/s and speedups in `docs/GPU_OPTIMIZATIONS.md` (rows SM48a, SM48b) and `docs/SMALL_MODEL.md` (section SM48). Add the implementation commit be20647f to the "Implementation commits" list.
- Report reviews/s per arm. The 732-pool ETA = 98.4M review-evals for 2 variants / rate.
- If lanes are exact (compare passed, ideally bitwise) and faster, propose `--lanes` for pool scoring and merge `sm48` into the small-model line after SM47 (expect conflicts in `small_model_heldout.py` main()).
- On `STOP: ... rows differ`, read `timing/compare-<run>.json` (`failures`, `max_abs_delta`):
  - `arr` differs: the numpy metrics differ on GPU predictions (unexpected; CPU was bitwise);
  - only `l2` / `l4` differ: a cross-thread race (LongWKV bits, state_vq globals) or a non-deterministic kernel. Compare run 8-old vs 1-old first to see whether the old path is itself deterministic.
- If lanes OOM (error rows, `only_a` > 0 in a compare): lower `--lane-review-budget`.
