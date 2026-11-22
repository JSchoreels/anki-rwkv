# Aggressive state tiers for Tiny (SM146): handoff

Status 11-22 22:45: **running** (launched 22:41:30 Sun 11-22 from branch `sm146` at 5e0d246e). Design and rules: `design.md` (pre-registered before any scoring of the new labels). The agent that launched it has stopped; a fresh agent (Sonnet, medium effort) processes the result.

## Launch

- Launcher: `launch-sm146.sh`, hidden via WMI (`Win32_Process.Create`, ShowWindow 0): WMI PID 33220, launcher bash Windows PID 22832 (memory guard parent). Log `gpu/launch-sm146.log`, raw stdout `gpu/launch-sm146.nohup`, scorer output `gpu/score.out`, tickets `gpu/tickets.jsonl`.
- GPU chain (queue, owner sm146, cap 0.8): smoke passed 22:42 (23 checks OK; int8 labels GPU vs CPU max 1.0e-5, tiered labels max 4.0e-6), then scoring ticket **001483** (Tiny, 723 users x 2 models x 6 labels REF / AN16 / I3a / I3b / I2a / I2b, `--prior-buckets`), started 22:42, expected about 50 min (retry passes of 1 to 2 min possible). GPU done about 23:35.
- CPU chain (started 22:41, in parallel): the 50 long-card users x the 2 Tiny models x the 6 labels, every label scored directly, up to 3 processes at a time and fewer than 3 CPU scoring processes on the whole machine (`cpu-count.ps1`), memory guard 5.9 GB (`cpu/memguard.log`, kills in `cpu/memguard-kill`, killed users in `cpu/longcard/tiny/killed.txt`), each with `--dump-scope all` into `work/sm146-dump/` (per-review values, never committed; a killed or failed user's dumps are deleted).
- Then `sm146_dr.py aggregate` -> `sm146-dr.json` (per-review / per-card |dR| vs int8, aggregates only), then `sm146_report.py report` -> `sm146-result.{json,md}`.
- ETA of `done`: about 00:00 to 00:30 Mon 11-23 (CPU about 3.2 process-hours over 3 processes; SM145's Tiny long cards took 2.65 process-hours for 5 labels).
- Small int8 reference rows for R2: `gpu/small-ref-rows.jsonl` (SM145's, identical to SM144's; sha256 pinned in the report). Nothing to score for Small.

## Patterns

- Done: `grep -E '\] done \(status' gpu/launch-sm146.log` (the status line names the go pick).
- Pending: `report pending` (exit 2 of the report: fewer than 720 GPU users or 48 long-card users for a label, or an incomplete dR read-out); the main session decides.
- Failure: `grep -E '\] STOP' gpu/launch-sm146.log` (the last is `STOP: ended with exit 1`; the one before names the step). Smoke failure files: `gpu/smoke-check.txt`, `gpu/smoke-check.json`, `gpu/gpu-selftest.json`. Do not relaunch blindly.
- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-sm146*' } | Select ProcessId, CommandLine`; queue: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`.

## Stop / resume

- Stop both chains between steps: `touch C:/Users/jonat/workspace/wt/sm146/artifacts/small-model/sm146/gpu/STOP` (a running ticket finishes; running CPU users finish). A running ticket can be cancelled with `r01_queue.py cancel <ticket>`.
- Resume: remove `gpu/STOP`, then the launch command again; finished users (GPU rows, `cpu/longcard/tiny/u-<user>.jsonl` with their dumps) and `gpu/smoke-ok` are skipped.

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm146/artifacts/small-model/sm146/launch-sm146.sh >> C:/Users/jonat/workspace/wt/sm146/artifacts/small-model/sm146/gpu/launch-sm146.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- Do not edit `scripts/`, `expertium/`, `rwkv/` of the sm146 worktree, and do not merge into it, while it runs. Never edit `launch-sm146.sh` while it runs. The `.sh` files are LF in the working copy; do not re-check them out (autocrlf would make them CRLF).
- After a session restart or power event: check the launcher and the queue (detached launchers have died at power events); relaunch with the command above.

## Follow-up after `done` (fresh agent, Sonnet medium is enough)

```bash
cd C:/Users/jonat/workspace/wt/sm146
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; D=artifacts/small-model/sm146
grep -E 'smoke|done|STOP|report|dR read-out|killed|failed' $D/gpu/launch-sm146.log | cut -c1-300
cat $D/sm146-result.md
$PY -X utf8 $D/sm146_report.py report   # re-run only if needed (exit 0 decided, 2 pending, 3 bucket mismatch)
```

1. **Report** to the main session: the go pick and the bytes vs int8 (AN16 -35.34%, I3a -39.80%, I3b -42.71%, I2a -44.27%, I2b -48.64%); per model, label and head R2 (vs Small int8, upper bound <= +1.00%), the cost vs Tiny int8 (dataset-wide and per answer-count bucket), R3 (cards with 128+ answers, CPU, point <= +0.5% and upper bound <= +1.0%), G (share of long-card-user cards whose worst review has |dR| > 0.10, at most 1%), plus the reported-only shares of reviews and cards above 0.05 and 0.10; coverage (killed or failed long-card users), integrity vs SM144 and SM145 (REF and AN16 should equal, 0.0). Quote GPU minutes per ticket (`gpu/tickets.jsonl`), reviews/s and CPU minutes (`grep 'long-card tiny user' gpu/launch-sm146.log`).
2. **Write-up** `RESULT.md` (short: question, result, numbers, the per-card read-out in plain words, what it means for the fork), commit with `git add -f artifacts/small-model/sm146` (never `work/`, `.npz`, `quiet.json`, `docs/GPU_OPTIMIZATIONS_LOG.md`, `.claude/`), commit as `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit` with the Co-Authored-By line (Claude Sonnet 5.5), fetch + merge origin, push `sm146`.
3. **Factory page** (main repo): close the SM146 plan row (match prefix "Aggressive state tiers for Tiny (SM146): "; `when`: done time and verdict; `state` done; `ended_at`) and add a results row (format of the SM144 / SM145 results rows: per model and label table, times, structured detail), by targeted byte insertion into `pages/data/factory-curated.json` (CRLF, HTML entities; never json.dumps the whole file), then `publish_factory_page.py` and `--check` (SYNC 0); commit factory-curated.json + factory.json only; push main. No "Expertium"; "plain name (ID)" in every sentence; "%" not "points".
4. If the report is pending, or a label fails only on one head or one model, the decision is the user's (via the main session).
5. After the write-up is pushed, the dumps in `work/sm146-dump/` can be kept for a later look; deleting them is the user's call.

## Note on SM145's report script

`sm145_report.py` (SM145 worktree, not touched here) stopped at 22:31:55 with `KeyError: 'ahead'` in `dataset_cmp` (it reads `c[KEY[h]]`; `showtime_report.compare` returns the keys `curve` and `instant`, so it must be `c[h]`). `sm146_report.py` has the fix (`c[h]`); the same one-line fix applies to SM145 when it is its turn (all its scoring and the dR read-out are finished).
