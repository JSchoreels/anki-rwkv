# Handoff: release-candidate checks for the family (SM139) -- CLOSED 11-21 (result status pass, written up and pushed)

Design: `design.md` (pre-registered 11-21 before any GPU ticket). Models (sha256 pinned in `models.sh`): Big the show-time big model (SM91s2) + the show-time instant-only big model (SM99) (both unchanged since 11-13); Small the show-time two-head cool-down for Small (SM118) + the longer Small Instant-Only model (SM113); Tiny the six-layer Tiny two-head + Instant-Only models (SM130). Release candidates at `w8+s8` and `w8+s8hx8`; the published 11-13 Small / Tiny models (SM92, SM98, SM106, SM101) at `w8+s8`. Users: the 732 set (723 scored, 722 rule users) and the extension (`users-sm93-score.json`, 771 = 765 outside the 732 set + 6 check users, `--pool 1500`). Open questions in `design.md` (which Tiny is "published": SM106 / SM101, not SM124 / SM115; rolling recalibration; fork mapping).

## The running launcher

- Launched 11:45 Sat 11-21 from `e4bbd619` (branch `sm139`), hidden via WMI: wrapper bash PID 24884 (`bash .../launch-sm139.sh`).
- Ticket **001420** = GPU smoke, exit 0 at 11:46: `smoke OK`, 64 values, max |GPU - CPU| 3.0e-06 (tol 1e-5). **001421** = Tiny on the 732 set, started 11:46.
- Log `C:/Users/jonat/workspace/wt/sm139/artifacts/small-model/sm139/gpu/launch-sm139.log` (lines `[sm139 MM-DD HH:MM:SS]`); scorer output `gpu/score.out`; stdout/stderr `gpu/launch-sm139.nohup`. Queue owner `sm139`, purposes start with `Release-candidate checks for the family (SM139): `. Only quiet.json windows (scope all / gpu) count. GPU cap 0.8, 0.92 after a pass without progress.

## Steps and ETA (Sat 11-21; design.md "GPU time")

| step | estimate | ETA end |
|---|---|---|
| smoke | done 11:46 | |
| Tiny, 732 set (6 labels) | 25-30 min (+ retries) | ~12:20 |
| Small, 732 set (6 labels) | 40-50 min | ~13:15 |
| Big, 732 set (4 labels) | 70-80 min | ~14:40 |
| Tiny, extension | 35-40 min | ~15:25 |
| Small, extension | 55-65 min | ~16:35 |
| Big, extension | 90-105 min | ~18:25 |
| `sm139_report.py report` (CPU, seconds), final file check | | **result ~18:30-19:30** |

- Done: `grep -E '\] done \(status' .../gpu/launch-sm139.log` (status pass / incomplete / fail: C1 finite, C2 format cost, C3 coverage).
- Failure: `grep -E '\] STOP' .../gpu/launch-sm139.log` (the last is `STOP: ended with exit 1`; the one before names the step). Report; do not relaunch blindly.
- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-sm139*' } | Select ProcessId, CommandLine`

## Stop / relaunch

- Stop between tickets: `touch C:/Users/jonat/workspace/wt/sm139/artifacts/small-model/sm139/gpu/STOP`. A running scoring ticket: cancel it (`C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py cancel <ticket>`) and kill the launcher; scored users are kept (resumable rows).
- Relaunch (remove STOP first; the smoke and finished users are skipped):

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm139/artifacts/small-model/sm139/launch-sm139.sh >> C:/Users/jonat/workspace/wt/sm139/artifacts/small-model/sm139/gpu/launch-sm139.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- Do not edit `scripts/`, `expertium/`, `rwkv/` of the sm139 worktree while it runs (the launcher refuses uncommitted source at start). Edits under `artifacts/` are fine. `launch-sm139.sh` and `models.sh` are LF in the working copy; do not re-check them out (autocrlf would make them CRLF).

## Follow-up after `done` (CPU, fresh agent; Sonnet medium is enough)

```bash
cd C:/Users/jonat/workspace/wt/sm139
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; D=artifacts/small-model/sm139
grep -E 'smoke|scored for every label|no progress|report exit|done|STOP' $D/gpu/launch-sm139.log | cut -c1-400
$PY -X utf8 $D/sm139_report.py report        # re-print only if needed (the launcher ran it): sm139-result.{json,md}
$PY C:/Users/jonat/workspace/anki-rwkv/scripts/r01_inspect.py $D/sm139-result.md --lines 60
```

1. Write-up: add a short paragraph at the top of `sm139-result.md`: the status and, per release candidate, C1 / C2 (the format cost per head on the 732 set and the extension, vs the +0.05% ceiling); then the update table in plain sentences per size ("what users get": candidate at the release format vs the published 11-13 model; for Big it is the format only, and Big's extension users are its training users). Coverage and any non-finite or missing users, the row checks (check users and recorded rows; above 1e-5 = say so). Time heads: not rescored (design.md "Time heads"). Models always as "plain name (ID)"; "%" never "points"; no go / no-go wording for the update part. Then mark this HANDOFF closed.
2. Commit on `sm139` with `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit` and `git add -f`: `sm139-result.{md,json}`, `HANDOFF.md`, `gpu/*.log`, `gpu/smoke-check.txt`, `gpu/smoke-gpu-rows.jsonl`, `gpu/sm139-rows-732.jsonl`, `gpu/sm139-rows-ext.jsonl`, `gpu/tickets.jsonl` (per-user scores only, no raw review rows); not `score.out` if it is large (> 5 MB: commit its last 200 lines as `gpu/score-tail.out`), never `.claude/`. Message ends with a blank line + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. `git fetch` + merge `origin/sm139`, push `sm139`; never force-push.
3. Factory page (main repo `C:/Users/jonat/workspace/anki-rwkv`): `git fetch` + `git merge --ff-only origin/main`; in `pages/data/factory-curated.json` set the SM139 plan row to `"state": "done"` with `ended_at` and a `when` with the status and the headline numbers; add one results row (date, kind `gpu-eval` or as earlier eval rows, name, eval, fork "not used: release packaging is the fork agent's step", takeaway with the format costs and the update per size, inputs `show-time`, started/ended, gpu_minutes from `gpu/tickets.jsonl`, tickets, detail.verdict + detail.table). Targeted byte insertion only (the file is LF in the working copy; HTML entities; never json.dumps the whole file). Then `.venv/Scripts/python.exe scripts/publish_factory_page.py` and `--check` (must print `SYNC 0`); commit only `pages/data/factory-curated.json` + `pages/data/factory.json`; fetch + merge, push main. Never write "Expertium" in page text ("outside research").
4. Ideas recorded, not run: P(recall within S) of the answer-time heads rescored with the release-candidate state codec inside the head's feature capture (`sm117_head.py` scores fp32 state today); a recalibration decision for this release (open question 2).
