# Earlier int8 switch (SM144): handoff

Status 11-22 01:15: **running** (launched 01:10:22 Sun 11-22 from branch `sm144` at fc5b437f). Design and rules: `design.md` (pre-registered before any scoring). The agent that launched it has stopped; a fresh agent processes the result.

## Launch

- Launcher: `launch-sm144.sh`, hidden via WMI (`Win32_Process.Create`, ShowWindow 0): WMI PID 29824, launcher bash Windows PID 26920 (memory guard parent). Log `gpu/launch-sm144.log`, raw stdout `gpu/launch-sm144.nohup`, scorer output `gpu/score.out`, tickets `gpu/tickets.jsonl`.
- GPU chain (queue, owner sm144, cap 0.8): smoke ticket **001466** passed at 01:11 (75 checks OK), then one ticket per size on the 732 set: Tiny **001467** (running since 01:11), then Small and Big (ids assigned when enqueued; expected 65 / 120 / 150 min; FIFO, other owners' tickets can slip in, the per-user adaptation pilot (SM143) included). GPU done about 07:00 to 09:00 Sun.
- CPU chain (started 01:10, in parallel): the 50 long-card users x Tiny, Small, Big x every label (REF, A, A32, A16, AN32, AN16, scored directly), **2 processes at a time**, memory guard 5.9 GB (`cpu/memguard.log`, kills in `cpu/memguard-kill`, killed users in `cpu/longcard/<size>/killed.txt`).
- ETA of `done`: about 14:00 to 18:00 Sun 11-22 (the CPU chain is the long pole, about 12-16 h).

## Patterns

- Done: `grep -E '\] done \(status' gpu/launch-sm144.log` (the status line names the pick per size).
- Pending: `report pending (coverage)` (exit 2 of the report: fewer than 720 GPU users or 48 long-card users for a label); the main session decides.
- Failure: `grep -E '\] STOP' gpu/launch-sm144.log` (the last is `STOP: ended with exit 1`; the one before names the step). Smoke failure: `gpu/smoke-check.txt`, `gpu/smoke-check.json`, `gpu/gpu-selftest.json`. Do not relaunch blindly.
- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-sm144*' } | Select ProcessId, CommandLine`; queue: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`.

## Stop / resume

- Stop both chains between steps: `touch C:/Users/jonat/workspace/wt/sm144/artifacts/small-model/sm144/gpu/STOP` (a running ticket finishes; running CPU users finish). A running ticket can be cancelled with `r01_queue.py cancel <ticket>`.
- Resume: remove `gpu/STOP`, then the launch command again; finished users (GPU rows, `cpu/longcard/<size>/u-<user>.jsonl`) and `gpu/smoke-ok` are skipped.

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm144/artifacts/small-model/sm144/launch-sm144.sh >> C:/Users/jonat/workspace/wt/sm144/artifacts/small-model/sm144/gpu/launch-sm144.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- Do not edit `scripts/`, `expertium/`, `rwkv/` of the sm144 worktree, and do not merge into it, while it runs (the launcher refuses uncommitted source). The `.sh` files are LF in the working copy; do not re-check them out (autocrlf would make them CRLF).
- After a session restart or power event: check the launcher and the queue (detached launchers have died at power events); relaunch with the command above.

## Follow-up after `done` (fresh agent, Sonnet medium is enough)

```bash
cd C:/Users/jonat/workspace/wt/sm144
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; D=artifacts/small-model/sm144
grep -E 'smoke|done|STOP|report|killed|failed' $D/gpu/launch-sm144.log | cut -c1-300
cat $D/sm144-result.md
$PY -X utf8 $D/sm144_report.py report   # re-run only if needed (exit 0 decided, 2 pending, 3 bucket mismatch)
```

1. **Report** to the main session: the go per size (pick and bytes vs int8: A -27.7 / -27.6 / -20.6%, A32 -27.2 / -27.0 / -20.2%, A16 -25.5 / -25.4 / -19.0%, AN32 -43.2 / -38.9 / -38.0%, AN16 -40.3 / -36.3 / -35.3% for Big / Small / Tiny), and per model and label R2 (dataset-wide), R3 (cards with 128+ answers, every label scored directly on the CPU), the cost per answer-count bucket (GPU all users and CPU long-card users; the point of the experiment: is the cost gone when the switch is at 16 or 32 answers, and in which bucket does it sit), non-finite users, coverage (killed or failed long-card users), REF vs SM139's GPU rows (about 0) and GPU vs CPU on the long-card bucket. Quote GPU minutes per ticket (`gpu/tickets.jsonl`) and CPU minutes.
2. **Write-up** `RESULT.md` (short: question, result per size, numbers, what it means for the fork), commit with `git add -f artifacts/small-model/sm144`, fetch + merge, push `sm144`.
3. **Factory page** (main repo): update the SM144 plan row (`when`: done time and verdict; `state` done; `ended_at`) and add a results row (format of the SM141 results row: per model and label table), by targeted byte insertion into `pages/data/factory-curated.json` (CRLF, HTML entities; never json.dumps the whole file), then `publish_factory_page.py` and `--check` (SYNC 0); commit factory-curated.json + factory.json only; push main. No "Expertium"; "plain name (ID)" in every sentence; "%" not "points".
4. If the report is pending, or a scheme fails only on one head or one model, the decision is the user's (via the main session).
