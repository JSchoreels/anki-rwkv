# Handoff: answer-time heads for the six-layer Tiny (SM138)

Design: `design.md` (pre-registered 11-21 before any GPU ticket; exactly the rules of the time heads for the new candidates (SM119); models: the six-layer Tiny two-head model (SM130), key `sm130th`, and the six-layer Tiny Instant-Only model (SM130), key `sm130io`; 346 head inputs). CPU proof PASS (16 checks, `check-cpu-sm138.json`); dry run OK (`gpu/launch-sm138-dryrun.log`).

## The running launcher

- Launched 09:34 Sat 11-21 from `da00e745` (branch `sm138`), hidden via WMI: wrapper bash PID 20336, launcher bash PID 37736 (`bash .../launch-sm138.sh`).
- First queue ticket **001412** (smoke of the two-head model, exit 0 at 09:35: first loss ln 16, grade bitwise True, equal to SM130's recorded show-time rows); **001413** = its head training, started 09:35.
- Log: `C:/Users/jonat/workspace/wt/sm138/artifacts/small-model/sm138/gpu/launch-sm138.log` (lines `[sm138 MM-DD HH:MM:SS]`); stdout/stderr `gpu/launch-sm138.nohup`. Queue owner `sm138`, purposes start with `Answer-time heads for the six-layer Tiny (SM138): `. Only quiet.json windows (scope all/gpu) count; no daily or Sunday break. GPU cap 0.8.

## Steps and ETA (Sat 11-21; SM119's Tiny took 1 min smoke, 42 min training, 0.5 min grade check, about 15 min scoring)

| step | ETA end |
|---|---|
| two-head: smoke (done 09:35), head training (~40 min), grade check | ~10:20 |
| Instant-Only: smoke, head training (~40 min), grade check | ~11:05 |
| scoring two-head, then Instant-Only (~15 min each; pass 2 at cap 0.92 only for missing users) | ~11:35 |
| `decide` per model + `report` (CPU, a few minutes) | **gate results ~11:45** |

- Outputs: heads `artifacts/small-model/sm138-<key>-head/` (`head.pth`, `ablation-head.pth`, `master.pth`, `train.json`); markers `gpu/smoke-ok-sm138-<key>`, `gpu/grade-ok-sm138-<key>`; grade check `gpu/grade-check-<key>.json`; scores `gpu/sm138-<key>-rows.jsonl` (+ `.hist/`); logs `gpu/{check,train,grade,score}-<key>.out`; gates `<key>-gate.json`, result `sm138-result.{md,json}` in this folder.
- Done: `grep -E '\] done$' .../gpu/launch-sm138.log`. Per model: `decide <key> exit 0|3` (3 = gate fail, not a launcher failure), then `report exit 0`.
- Failure: `grep -E '\] STOP' .../gpu/launch-sm138.log` (the last is `STOP: ended with exit 1`; the one before it names the step and file). Report; do not relaunch blindly.
- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-sm138*' } | Select ProcessId, CommandLine`

## Stop / relaunch

- Stop between tickets: `touch C:/Users/jonat/workspace/wt/sm138/artifacts/small-model/sm138/gpu/STOP`. A running head training: cancel its ticket (`C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py cancel <ticket>`) and kill the launcher; a relaunch resumes from `resume.pt`.
- Relaunch (remove STOP first; finished steps are skipped):

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm138/artifacts/small-model/sm138/launch-sm138.sh >> C:/Users/jonat/workspace/wt/sm138/artifacts/small-model/sm138/gpu/launch-sm138.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- Do not edit `scripts/`, `expertium/`, `rwkv/` of the sm138 worktree, and do not merge into it, while it runs (SourceWatch; the launcher also refuses uncommitted source). Edits under `artifacts/` are fine. The launcher file itself is LF in the working copy; do not re-check it out (autocrlf would make it CRLF).

## Follow-up after `done` (CPU, fresh agent; Sonnet medium is enough)

```bash
cd C:/Users/jonat/workspace/wt/sm138
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; D=artifacts/small-model/sm138; H="$PY -X utf8 scripts/sm117_head.py"
grep -E 'smoke:|head training:|bitwise|scored users|decide|report|done|STOP' $D/gpu/launch-sm138.log | cut -c1-400
# re-print (only if needed; the launcher already ran them):
$H decide --model sm130th --rows $D/gpu/sm138-sm130th-rows.jsonl --grade-check $D/gpu/grade-check-sm130th.json --train-json artifacts/small-model/sm138-sm130th-head/train.json
$H decide --model sm130io --rows $D/gpu/sm138-sm130io-rows.jsonl --grade-check $D/gpu/grade-check-sm130io.json --train-json artifacts/small-model/sm138-sm130io-head/train.json
$H report      # writes $D/sm138-result.{md,json}
$PY C:/Users/jonat/workspace/anki-rwkv/scripts/r01_inspect.py $D/sm138-result.md --lines 40
```

1. Write-up: edit `sm138-result.md` (the report text) with a short paragraph at the top: the verdict per head; the answer-time definition (front + back, question shown to grade pressed, capped at the maximum answer seconds); "recall within S s = recalled (Hard, Good or Easy) and answered within S s, reading the back included"; the Show-answer moment is not stored. Compare with the four-round Tiny heads of the time heads for the new candidates (SM119) (`wt/sm119/artifacts/small-model/sm119/sm119-result.md`, the longer cool-down for Tiny (SM115)) as information. Models always as "plain name (ID)", "%" never "points". Then mark this HANDOFF closed.
2. Commit on `sm138` with `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit` and `git add -f`: result files, `*-gate.json`, `gpu/*.log`, `gpu/*.out`, `gpu/*.json`, `gpu/sm138-*-rows.jsonl`, `gpu/grade-rows-*.jsonl`, `gpu/tickets.jsonl` if present, `../sm138-*-head/train.json`; never `.pth`, `resume.pt`, `grade-dump-*`, `*.hist`, `work/`. Message ends with a blank line + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. `git fetch` + merge `origin/sm138`, push `sm138`; never force-push.
3. Factory page (main repo `C:/Users/jonat/workspace/anki-rwkv`): `git fetch` + `git merge --ff-only origin/main`; in `pages/data/factory-curated.json` set the SM138 plan row to `"state": "done"` with `ended_at` and a `when` with the verdicts, and add a results row per model (like the SM137 rows of 11-21: date, kind `gpu-train`, name, eval, fork, takeaway with the gate numbers and P(recall within S seconds) reliability, inputs `show-time`, started/ended, gpu_minutes, tickets, detail.verdict + detail.table). Targeted byte insertion only (CRLF + HTML entities; never json.dumps the whole file). Then `.venv/Scripts/python.exe scripts/publish_factory_page.py` and `--check` (must print `SYNC 0`); commit only `pages/data/factory-curated.json` + `pages/data/factory.json`; fetch + merge, push main. Never write "Expertium" in page text ("outside research").
4. Fork step (another agent; mention only): package both heads for the family release (fork branch `rwkv-time-heads-family`, worktree `wt/fork-time-heads`, as done for SM113 / SM115). The fork's head runtime must accept a 346-input head for the six-layer Tiny (the four-round Tiny heads had 474).
