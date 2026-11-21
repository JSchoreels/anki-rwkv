# Handoff: time heads for the new candidates (SM119)

Design: `design.md` (same recipe and rules as the time heads for the family (SM117); models: the longer Small Instant-Only model (SM113), the longer cool-down for Tiny (SM115), and the show-time two-head cool-down for Small (SM118) only on a go). Launcher `launch-sm119.sh` (dry run OK: `gpu/launch-sm119-dryrun.log`; `bash -n` OK).

## The running launcher

- Launched 17:45 Mon 11-16 from `17e0ea19` (branch `sm119`), hidden via WMI: wrapper bash pid 21648, launcher bash pid 13372 (`bash .../launch-sm119.sh`; check both with the alive command below).
- It enqueues **nothing** until the launcher log of SM118 (`wt/sm118/artifacts/small-model/sm118/gpu/launch-sm118.log`, read only) ends with a `done` or `STOP` line (expected about 03:30 to 12:30 Tue 11-17); it checks every 5 min, then reads `sm118-decision.json` (verdict go adds SM118's model; the sha256 of its final master is taken then).
- Log: `C:/Users/jonat/workspace/wt/sm119/artifacts/small-model/sm119/gpu/launch-sm119.log` (lines `[sm119 MM-DD HH:MM:SS]`); stdout/stderr `gpu/launch-sm119.nohup`. Queue owner `small-model`, purposes start with `time heads for the family (SM119): `. Breaks (quiet.json, daily 20:00-01:00, Sunday from 16:00) are checked by the launcher before every ticket; GPU cap 0.8.

## Steps and outputs (order: the longer Small Instant-Only model (SM113), then the longer cool-down for Tiny (SM115), then SM118 on a go)

Per model: smoke (3 / 12 min budget), head training (55 / 90), grade check (7 / 10); then scoring of all models (18 / 30 each, pass 2 at cap 0.92 for missing users), then `decide` per model and `report` on CPU. About 1.5 GPU h for two models (SM117's three small heads took about 2.3 h), plus about 0.8 h for SM118 on a go.

- heads `artifacts/small-model/sm119-<key>-head/` (`head.pth`, `ablation-head.pth`, `master.pth`, `train.json`; keys sm113, sm115, sm118); markers `gpu/smoke-ok-sm119-<key>`, `gpu/grade-ok-sm119-<key>`; grade check `gpu/grade-check-<key>.json`; scores `gpu/sm119-<key>-rows.jsonl` (+ `.hist/`); logs `gpu/{check,train,grade,score}-<key>.out`.
- Gate and result: `<key>-gate.json`, `sm119-result.md`, `sm119-result.json` in this folder (written by the launcher's last step).
- Log lines to watch: `<key> smoke: ...`, `<key> head training: ... complete`, `[sm117-grade] ... bitwise True`, `<key> scored users: N of 723`, `decide <key> exit 0|3` (3 = gate fail, not a launcher failure), `report exit 0`, `done`. Any `STOP:` line names the step and file.

## Follow-up after `done`

1. Write `sm119-result.md` as `../sm117/sm117-result.md` (the launcher's report text plus a short paragraph at the top with the answer-time definition: front + back, question shown to grade pressed, capped at the maximum answer seconds; recall within S s = recalled (Hard, Good, Easy) and answered within S s, reading the back included; the Show-answer moment is not stored). Commit on `sm119` with `git add -f` (result files, `*-gate.json`, `gpu/*.log`, `gpu/*.out`, `gpu/*.json`, `gpu/sm119-*-rows.jsonl`, `gpu/grade-rows-*.jsonl`, `../sm119-*-head/train.json`; not `.pth`, `resume.pt`, `grade-dump-*`, `*.hist`, `work/`), fetch, merge, push.
2. Factory page (main, `C:/Users/jonat/workspace/anki-rwkv`): close the SM119 plan row, add a results row per model with times from `gpu/launch-sm119.log` (detail.notes as a JSON list of strings); `python "$TEMP/bare.py"` must print 0, then `scripts/publish_factory_page.py` and `--check` (SYNC 0), commit, fetch + merge, push.
3. Fork step: package the time head files for the longer Small Instant-Only model (SM113) and the longer cool-down for Tiny (SM115) (and SM118 on a go) into the fork branch `rwkv-time-heads-family` in worktree `C:/Users/jonat/workspace/wt/fork-time-heads` (another agent builds it), as done for the six family models.

## Stop / relaunch / alive check

- Stop between tickets: `touch .../sm119/gpu/STOP`. A running head training: cancel its queue ticket (`scripts/r01_queue.py cancel <ticket>`) and kill the launcher; relaunching resumes from `resume.pt`.
- Relaunch (finished steps are skipped):

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm119/artifacts/small-model/sm119/launch-sm119.sh >> C:/Users/jonat/workspace/wt/sm119/artifacts/small-model/sm119/gpu/launch-sm119.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- Alive: `tail -5 .../gpu/launch-sm119.log`; `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | ? { $_.CommandLine -like '*launch-sm119*' }`. After a Claude app restart check it; the wait for SM118 logs once an hour. If SM118's launcher died without a `done` / `STOP` line and is not relaunched, relaunch with `SKIP_SM118_WAIT=1` only once SM118 is really finished (the verdict is then read from its decision file).
- Do not edit `scripts/`, `expertium/`, `rwkv/` in this worktree while it runs.
