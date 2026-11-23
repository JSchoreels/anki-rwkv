# Rank-1 int8 states (SM150): handoff

Design and rules: `design.md` (pre-registered before any scoring of the new labels on the 732 set or the long-card users). The agent that launched it has stopped; a fresh agent (Sonnet, medium effort) processes the result. A narrow copy of the low-rank int8 states (SM148) tree: new labels rank 1 (R1, `w8+lr1q8x8`) and rank 1 then 2 from 16 answers (R12, `w8+lr1.16.2q8x8`), int8 factors; REF, Q4 and Q2 are SM148's rows (REF rescored on the 3 smoke users only).

## Launch

Status 11-23 18:20 (machine clock): **running**. Launched **18:12:16 Mon 11-23** from branch `sm150` at 589dfaaf (source clean; no change to `scripts/`, `rwkv/`, `expertium/`).

- Launcher: `launch-sm150.sh`, hidden via WMI (`Win32_Process.Create`, ShowWindow 0): WMI PID 36012, launcher bash Windows PID 6396 (memory guard parent).
- Proofs: CPU proofs ALL OK (`check-cpu-sm150.json`, 13 checks; REF rescored on the 3 smoke users equals SM148's rows within 1e-6; F1X0 equals SM147's rank-1 truncation (float64 SVD) within 2e-5). GPU smoke ticket passed 18:13 (36 checks; REF max GPU vs CPU 1.0e-5, low-rank labels max 1.4e-5). Tiny scoring ticket running since 18:13.
- **GPU ETA** (queue, owner sm150, cap 0.8, `PYTORCH_JIT=0`, 2 models x {R12, R1}, 723 users): Tiny about 40 to 45 min (done about 18:55 to 19:00), Small about 60 to 65 min (done about 20:00), Big about 100 to 110 min (done about 21:45; range 21:15 to 22:30; Big expects 718 users, the 5 largest never fit). GPU chain about 3.5 h in all (SM148 took 69 / 100 / 167 min for 4 labels). FIFO, other owners' tickets can slip in.
- CPU chain (parallel): 50 long-card users x {Tiny, Small} x 2 models x {R12, R1}, 2 processes at a time while the GPU runs, 3 after; about 2 to 3 h. **Big long-card is not run** (13 of its users exceed the 6 GB limit; warning only, design.md). Memory guard `-Match "small_model_heldout.py&--device cpu&sm150"` (`cpu/memguard.log`, kills in `cpu/memguard-kill`, killed users in `cpu/longcard/<size>/killed.txt`), dumps into `work/sm150-dump/` (never committed).
- Then SM148's REF dumps (`wt/sm148/work/sm148-dump`) are copied into `work/sm150-dump`, `sm150_dr.py aggregate --sizes Small,Tiny` -> `sm150-dr.json`, `sm150_report.py report` -> `sm150-result.{json,md}`.
- ETA of `done`: about 21:45 to 22:30 Mon 11-23 (the GPU chain is the long pole).

## Patterns

- Log: `C:/Users/jonat/workspace/wt/sm150/artifacts/small-model/sm150/gpu/launch-sm150.log` (raw stdout `gpu/launch-sm150.nohup`, scorer output `gpu/score.out`, tickets `gpu/tickets.jsonl`).
- Done: `grep -E '\] done \(status' gpu/launch-sm150.log` (the status line names the go pick per size).
- Pending: `report` exit 2 (fewer than 715 GPU users for a label or its comparison REF); the main session decides.
- Failure / STOP: `grep -E '\] STOP' gpu/launch-sm150.log` (the last is `STOP: ended with exit 1`; the one before names the step). Smoke failure files: `gpu/smoke-check.txt`, `gpu/smoke-check.json`, `gpu/gpu-selftest.json`. Do not relaunch blindly.
- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-sm150*' } | Select ProcessId, CommandLine`; queue: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`.

## Stop / resume

- Stop both chains between steps: `touch C:/Users/jonat/workspace/wt/sm150/artifacts/small-model/sm150/gpu/STOP` (a running ticket finishes; running CPU users finish). A running ticket can be cancelled with `r01_queue.py cancel <ticket>`.
- Resume: remove `gpu/STOP` (and `cpu/STOP` if the memory guard wrote one), then the launch command again; finished users (GPU rows, `cpu/longcard/<size>/u-<user>.jsonl` with their dumps) and `gpu/smoke-ok` are skipped.

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm150/artifacts/small-model/sm150/launch-sm150.sh >> C:/Users/jonat/workspace/wt/sm150/artifacts/small-model/sm150/gpu/launch-sm150.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- Do not edit `scripts/`, `expertium/`, `rwkv/` of the sm150 worktree, and do not merge into it, while it runs (the launcher refuses uncommitted source at start). Never edit `launch-sm150.sh` while it runs. The `.sh` files are LF in the working copy; do not re-check them out (autocrlf would make them CRLF). The `work/` folder is not git-ignored: never `git add` under it.
- After a session restart or power event: check the launcher and the queue (detached launchers have died at power events); relaunch with the command above.

## Follow-up after `done` (fresh agent, Sonnet medium is enough)

```bash
cd C:/Users/jonat/workspace/wt/sm150 && git fetch origin
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; D=artifacts/small-model/sm150
grep -E 'smoke|done|STOP|report|dR read-out|killed|failed' $D/gpu/launch-sm150.log | cut -c1-300
cat $D/sm150-result.md
$PY -X utf8 $D/sm150_report.py report   # re-run only if needed (exit 0 decided, 2 pending, 3 bucket mismatch)
```

1. **Report** to the main session: the go pick per size with bytes vs int8 (R1 -88.1 / -78.4 / -76.6%, R12 -87.6 / -77.4 / -75.6%, Q2 -82.8 / -68.2 / -66.7%, Q4 -74.4 / -52.3 / -51.1% for Big / Small / Tiny); per model and label the cost vs own int8 and, for Small and Tiny, vs the matching Big model at int8 (CI upper bound per head against +0.15% for Big, +0.5% for Small, +1.0% for Tiny); the starting point (Small and Tiny int8 vs Big int8); which labels pass for both models of each size; the warnings in plain words (long cards 128+ for Small and Tiny, per-card |dR|; Big not run); the bucket table (where the cost sits by answer count); coverage (users per label; killed or failed long-card users); the integrity line (REF rescored on the 3 smoke users equals SM148's rows within 1e-6; `check-cpu-sm150.json`); speed (`report_only.speed` in `sm150-result.json`; quote reviews/s); GPU minutes per ticket (`gpu/tickets.jsonl`), CPU process-minutes. Explain in full plain sentences.
2. **Write-up** `RESULT.md` (short: question, result per size, numbers, the warnings, what it means for the fork), aggregates only. Naming rules: every model as "plain name (ID)" in every sentence (e.g. "Big two-head (SM91s2)", "six-layer Tiny Instant-Only (SM130)"), never a bare ID; say "outside research", never a researcher's name; "%" never "points".
3. **Commit** on branch `sm150`: `git add -f artifacts/small-model/sm150` (never `work/`, `.npz`, `.pth`, `quiet.json`, `docs/GPU_OPTIMIZATIONS_LOG.md`, `.claude/`), `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit` with a message ending in a blank line and `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; `git fetch origin && git merge origin/sm150` (if it moved), then `git push origin sm150`. Never force-push, never bypass hooks.
4. **Factory page** (main repo `C:/Users/jonat/workspace/anki-rwkv`, `git fetch` + fast-forward first): close the SM150 plan row (the main session adds it; match prefix "Rank-1 int8 states (SM150): "; `when`: done time and verdict per size; `state` done; `ended_at`) and add a results row in the format of the SM148 results row (per model and label table, times, structured detail), by targeted byte insertion into `pages/data/factory-curated.json` (CRLF, HTML entities; never json.dumps the whole file), then `publish_factory_page.py` and `--check` (SYNC 0); commit factory-curated.json + factory.json only; fetch + merge origin, push main. No "Expertium"; "plain name (ID)" in every sentence; "%" not "points".
5. If the report is pending, a label fails only on one head or one model, or the Big result sits between R1 and R12, the decision is the user's (via the main session).
6. After the write-up is pushed, the dumps in `work/sm150-dump/` can stay for a later look; deleting them is the user's call.
