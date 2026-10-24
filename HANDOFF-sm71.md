# HANDOFF: deleted-card leak check (SM71)

Branch `sm71` (worktree `C:/Users/jonat/workspace/wt/sm71`). Write-up: `artifacts/small-model/sm71/README.md`.

## Queued (detached runbook, GPU queue owner `small-model`)
- Runbook: `artifacts/small-model/sm71/gpu/run-sm71.sh`, log `artifacts/small-model/sm71/gpu/run-sm71.log` (scorer output in `run-sm71.out`). Launched 10-24 18:41 hidden via WMI (`Win32_Process.Create`, ShowWindow=0, Git bash pid 53540; stdout in `gpu/run-sm71.nohup`); rows record commit 490a66d5 or later (scorer source unchanged since dc13fdbf). Stop it with `touch artifacts/small-model/sm71/gpu/CANCEL` (takes effect before the next ticket).
- It waits for the SM56 launcher's epoch-14 scoring (`exit N: SM56 scoring s-w64kd-final-sc-lanes-e14` in `wt/sm56/artifacts/small-model/sm56/launch-sm56.log`, or a `STOP:` line, or 20:30), then enqueues one ticket per model and input mode, checking `quiet.json` before each enqueue (no ticket into the 21:00-01:30 break; the queue also holds tickets whose expected end reaches a window).
- 464 set (458 users), in order, expected minutes: big keep 20, small keep 10, big neighbour 20, small neighbour 10, big flags 20, small flags 10, big reseed 20, small reseed 10, small8k keep 10, small8k neighbour 10 (about 140 min). Then the 732 pool (265 more fit users + the 3 long users at the 0.92 cap) for big/small keep and neighbour (about 70 min).
- Every ticket logs `exit <rc>: SM71 deleted-card leak: <model> ... inputs=<mode>, <pool> users ...`. Lines to watch: `[sm71 ...] 464 set done` (first read-out), then `[sm71 ...] done`.
- Outputs (new files only): `artifacts/small-model/sm71/gpu/{big,small,small8k}-{keep,neighbour,flags,reseed}.jsonl` (rows with `benchmark`, `benchmark_existing`, `benchmark_deleted`; `deleted_cards` / `seed_offset` fields mark the mode).
- If a ticket exits non-zero: read the tail of `run-sm71.out`; rerunning the same command resumes (the scorer skips done users).

## Read-out (after `464 set done`; CPU only, about 1 min)
```
cd C:/Users/jonat/workspace/wt/sm71
G=artifacts/small-model/sm71/gpu
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 scripts/sm71_report.py --title "SM71 GPU, 464 set" \
  --users-file C:/Users/jonat/workspace/wt/sm56/artifacts/small-model/sm56/eval-users-464-fit.json \
  --series "big/keep=$G/big-keep.jsonl::fork/w8+s8" --series "small/keep=$G/small-keep.jsonl::sm56/w8+s8" \
  --series "big/neighbour=$G/big-neighbour.jsonl::fork/w8+s8" --series "small/neighbour=$G/small-neighbour.jsonl::sm56/w8+s8" \
  --series "big/flags=$G/big-flags.jsonl::fork/w8+s8" --series "small/flags=$G/small-flags.jsonl::sm56/w8+s8" \
  --series "big/reseed=$G/big-reseed.jsonl::fork/w8+s8" --series "small/reseed=$G/small-reseed.jsonl::sm56/w8+s8" \
  --out $G/report-464.json
```
- 8k model (SM55): a separate call with `--series "big/keep=..."`, `--series "big/neighbour=..."`, `--series "small8k/keep=$G/small8k-keep.jsonl::sm55/w8+s8"`, `--series "small8k/neighbour=$G/small8k-neighbour.jsonl::sm55/w8+s8"`, `--out $G/report-464-sm55.json`.
- 732 pool (726 users): the keep and neighbour series only, without `--users-file`, `--out $G/report-732.json` (the report uses users common to all series).
- Check first: `big-keep.jsonl` per-user `benchmark` equals the big model's rows in `wt/small-model/artifacts/small-model/heldout.jsonl` (label `fork/w8+s8`) within the GPU tolerance (default inputs, unchanged code path), and `small-keep.jsonl` equals `wt/sm56/artifacts/small-model/heldout-sm56-pools.jsonl` (label `s-w64kd-final-sc-lanes-dec10-1ep32k-muon/w8+s8`).
- Views: view 1 = `neighbour` (and `flags`) rows, subset `all`; view 2 = `keep` rows, subset `existing`. Read the neighbour change against the reseed change (noise floor of re-drawn id codes).
- Then add the tables to the README "GPU results" section, commit (explicit paths, `git add -f` for artifacts), push.
