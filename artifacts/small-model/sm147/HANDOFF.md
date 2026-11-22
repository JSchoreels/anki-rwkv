# Low-rank structure of the card and note states (SM147): handoff

CPU-only probe, no GPU ticket, no training. Design: `design.md` (written before any number). Source commit of the run: 0a128ce9 (scripts clean, `dirty=0` in `cpu/run.out`).

## What is running

- Launcher `run-sm147.sh`, hidden via WMI (`Win32_Process.Create`, ShowWindow 0), relaunched 11-23 00:15 (machine clock): WMI bash PID 4484; scorer python PIDs 33232 / 3048 (step 1, smoke proof rows); memory guard powershell PID 10820 (5.9 GB per process). The first launch (00:12, a child of the agent shell) was stopped and relaunched; the scorer resumed from its rows file (2 of 3 smoke users were done).
- Step 1: 3 smoke users x 6 models x (`w8`, `w8+lrsv`, `w8+s8hx8`) -> `cpu/smoke-rows.jsonl` (log `cpu/smoke.out`), a few minutes.
- Step 2: 8 users (`users.json`) x 6 models x 5 labels (`w8+lrsv` F32 with spectrum, `w8+s8hx8` REF, `w8+lr1`, `w8+lr2`, `w8+lr4`), two processes: `cpu/rows-big.jsonl` (log `cpu/rows-big.out`) and `cpu/rows-small-tiny.jsonl` (log `cpu/rows-small-tiny.out`); then concatenated into `cpu/sm147-rows.jsonl`.
- Launcher log: `cpu/run.out`. **Done pattern:** last line `SM147 RUN DONE`. **Failure pattern:** `SM147 RUN FAILED` (or a file `cpu/memguard-kill`). Expected end: about 01:20 to 01:40 (machine clock of 11-23), i.e. 60 to 80 minutes after the relaunch.
- Alive check (PowerShell): `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*sm147*' } | Select ProcessId, Name`.
- If it died without either pattern (session restart, power event): relaunch with the same command, the scorer resumes per user from the rows files:

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; cd C:/Users/jonat/workspace/wt/sm147/artifacts/small-model/sm147 && bash run-sm147.sh >> cpu/run.out 2>&1"'; ProcessStartupInformation=$si}
```

## Follow-up for a fresh agent (Sonnet, medium effort)

Work in `C:/Users/jonat/workspace/wt/sm147` (branch `sm147`); `git fetch` first. Python: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8`.

1. Confirm `cpu/run.out` ends with `SM147 RUN DONE` and no `cpu/memguard-kill` exists.
2. Checks: `cd artifacts/small-model/sm147 && <python> sm147_check.py` -> `check-sm147.json`; expect `all_ok: true` (C1 selftest, C2 F32 == plain w8 on 3 users x 6 models, C3 REF == SM145's committed CPU smoke rows, C4 240 finite rows with consistent spectrum counts). If a check fails, report it and stop (do not write a reading).
3. Tables: `<python> sm147_report.py` -> `sm147-result.json`, `sm147-result.md` (energy kept by rank, by answer-count bucket, per layer and head; benchmark LogLoss of R1 / R2 / R4 vs REF and vs F32 with bootstrap CI; all-scope LogLoss by the card's earlier answers; bytes for scale).
4. Write `RESULT.md` (short, aggregates only, no card ids, no review rows): the run facts (users, commit, wall time from `cpu/run.out`, CPU review-evaluations per second from the `.out` logs), the energy headline per size, the rank-r LogLoss table, the bucket pattern, the check summary, and a short reading: would a rank-r state be worth a proper GPU test, with or without a cool-down? It is a pointer, not a gate: no verdict rules. Naming rules: every model as "plain name (ID)" in every sentence (e.g. "Big two-head (SM91s2)", "six-layer Tiny Instant-Only (SM130)"); say "outside research", never the researcher's name; "%" never "points". Remember: an entity with n answers has a WKV state of rank at most n (each write adds a rank-1 term), so the young buckets are exactly low-rank by construction; the interesting buckets are 5-8 and above.
5. Commit (never `.pth`, `.npz`, `work/`, `quiet.json`, `docs/GPU_OPTIMIZATIONS_LOG.md`, `.claude/`; `git add -f` for artifacts/): `git add -f artifacts/small-model/sm147/{RESULT.md,check-sm147.json,sm147-result.json,sm147-result.md,cpu/*.jsonl,cpu/*.out}`, then `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit` with a message ending in a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; `git fetch origin && git merge origin/sm147` (if it exists), then `git push origin sm147`. Never force-push, never bypass hooks. Do not edit the factory page; give the coordinator a one-paragraph results text instead.

## Interim (2 of the 3 smoke users, F32 spectrum, not a result)

On the smallest smoke user (2388, mostly young cards), the mean energy share kept by rank 1 / 2 / 4 over all writes: card about 84% to 87% / 97% / 99.9%, note about 90% to 93% / 98.4% to 99.0% / 99.95%, for every model; rank 2 keeps the whole energy for at least half of the writes (the young states have rank at most their answer count). Rank-r benchmark LogLoss on that user only (one-user trial before the run, not a result), vs the int8 reference: rank 1 instant +0.14% for Big two-head (SM91s2) and +0.28% for six-layer Tiny two-head (SM130), curve -0.25% / -0.04%; rank 4 instant -0.05% / -0.03%.
