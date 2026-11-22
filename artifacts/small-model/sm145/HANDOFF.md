# Int3 young tier on the early switch (SM145): handoff

Status 11-22 13:00: **running** (launched 12:56:41 Sun 11-22 from branch `sm145` at e178fa02). Design and rules: `design.md` (pre-registered before any scoring). The agent that launched it has stopped; a fresh agent (Sonnet, medium effort) processes the result.

## Launch

- Launcher: `launch-sm145.sh`, hidden via WMI (`Win32_Process.Create`, ShowWindow 0): WMI PID 24768, launcher bash Windows PID 34824 (memory guard parent). Log `gpu/launch-sm145.log`, raw stdout `gpu/launch-sm145.nohup`, scorer output `gpu/score.out`, tickets `gpu/tickets.jsonl`.
- GPU chain (queue, owner sm145, cap 0.8): smoke ticket **001474** passed at 12:57 (66 checks OK; int8 labels GPU vs CPU max 1.0e-5, tiered labels max 1.3e-5), then one ticket per size on the 732 set: Tiny **001475** (running since 12:58, expected about 41 min), then Small and Big (ids assigned when enqueued; expected about 73 / 110 min; FIFO, other owners' tickets can slip in). GPU done about 17:00 to 17:30 Sun.
- CPU chain (started 12:57, in parallel): the 50 long-card users x Tiny, Small, Big x REF, AN16, Y2N, Y4C, Y4N (every label scored directly), **2 processes at a time**, memory guard 5.9 GB (`cpu/memguard.log`, kills in `cpu/memguard-kill`, killed users in `cpu/longcard/<size>/killed.txt`), each with `--dump-scope all` into `work/sm145-dump/` (per-review values, never committed; a killed or failed user's dumps are deleted).
- Then `sm145_dr.py aggregate` -> `sm145-dr.json` (per-review / per-card |dR| vs int8, aggregates only), then `sm145_report.py report` -> `sm145-result.{json,md}`.
- ETA of `done`: about 22:00 to 00:00 Sun 11-22 (the CPU chain is the long pole; SM144 needed 11.4 h for 6 labels, here 5 labels).

## Patterns

- Done: `grep -E '\] done \(status' gpu/launch-sm145.log` (the status line names the go pick per size).
- Pending: `report pending` (exit 2 of the report: fewer than 720 GPU users or 48 long-card users for a label, or an incomplete dR read-out); the main session decides.
- Failure: `grep -E '\] STOP' gpu/launch-sm145.log` (the last is `STOP: ended with exit 1`; the one before names the step). Smoke failure files: `gpu/smoke-check.txt`, `gpu/smoke-check.json`, `gpu/gpu-selftest.json`. Do not relaunch blindly.
- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-sm145*' } | Select ProcessId, CommandLine`; queue: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`.

## Stop / resume

- Stop both chains between steps: `touch C:/Users/jonat/workspace/wt/sm145/artifacts/small-model/sm145/gpu/STOP` (a running ticket finishes; running CPU users finish). A running ticket can be cancelled with `r01_queue.py cancel <ticket>`.
- Resume: remove `gpu/STOP`, then the launch command again; finished users (GPU rows, `cpu/longcard/<size>/u-<user>.jsonl` with their dumps) and `gpu/smoke-ok` are skipped.

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm145/artifacts/small-model/sm145/launch-sm145.sh >> C:/Users/jonat/workspace/wt/sm145/artifacts/small-model/sm145/gpu/launch-sm145.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- Do not edit `scripts/`, `expertium/`, `rwkv/` of the sm145 worktree, and do not merge into it, while it runs (the launcher refuses uncommitted source at start; a mid-run change would not be in the process that already loaded the code). Never edit `launch-sm145.sh` while it runs. The `.sh` files are LF in the working copy; do not re-check them out (autocrlf would make them CRLF).
- After a session restart or power event: check the launcher and the queue (detached launchers have died at power events); relaunch with the command above.

## Follow-up after `done` (fresh agent, Sonnet medium is enough)

```bash
cd C:/Users/jonat/workspace/wt/sm145
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; D=artifacts/small-model/sm145
grep -E 'smoke|done|STOP|report|dR read-out|killed|failed' $D/gpu/launch-sm145.log | cut -c1-300
cat $D/sm145-result.md
$PY -X utf8 $D/sm145_report.py report   # re-run only if needed (exit 0 decided, 2 pending, 3 bucket mismatch)
```

1. **Report** to the main session: the go pick and the recommended label per size with bytes vs int8 (AN16 -40.25 / -36.32 / -35.34%, Y2N -42.75 / -38.57 / -37.53%, Y4C -43.48 / -39.53 / -37.74%, Y4N -45.34 / -40.91 / -39.80% for Big / Small / Tiny); per model and label R2 (dataset-wide), R3 (cards with 128+ answers, CPU, point <= +0.5% and upper bound <= +1.0%), R4 (young cards 1-M); the extra cost of each int3 label vs AN16 (paired) and the worth-its-bytes table; the per-review and per-card |dR| tables (P50 / P90 / P99 / max, shares above 0.05 and 0.10, by answer-count bucket: does any card lose a real retention prediction?); coverage (killed or failed long-card users), integrity vs SM144's rows (REF and AN16 should equal on the CPU). Quote GPU minutes per ticket (`gpu/tickets.jsonl`), reviews/s and CPU minutes.
2. **Write-up** `RESULT.md` (short: question, result per size, numbers, the per-card read-out in plain words, what it means for the fork), commit with `git add -f artifacts/small-model/sm145` (never `work/`, `.npz`, `quiet.json`, `docs/GPU_OPTIMIZATIONS_LOG.md`, `.claude/`), commit as `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit` with the Co-Authored-By line, fetch + merge origin, push `sm145`.
3. **Factory page** (main repo): close the SM145 plan row (match prefix "Int3 young tier on the early switch (SM145): "; `when`: done time and verdict; `state` done; `ended_at`) and add a results row (format of the SM144 results row: per model and label table, times, structured detail), by targeted byte insertion into `pages/data/factory-curated.json` (CRLF, HTML entities; never json.dumps the whole file), then `publish_factory_page.py` and `--check` (SYNC 0); commit factory-curated.json + factory.json only; push main. No "Expertium"; "plain name (ID)" in every sentence; "%" not "points".
4. If the report is pending, or a label fails only on one head or one model, or the go pick and the recommendation differ, the decision is the user's (via the main session).
5. After the write-up is pushed, the dumps in `work/sm145-dump/` (about 1 to 2 GB) can be kept for a later look; deleting them is the user's call.
