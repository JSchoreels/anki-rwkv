# Handoff: time heads for the family (SM117)

Design and pre-registered rules: `design.md` (gate per head, what "reliable" means, S grid, baselines). Code: `scripts/sm117_head.py`, `scripts/sm117_check.py`, `scripts/sm117_timelib.py` (branch `sm117`, cut from `origin/sm115` because `main` has no small-model code). CPU proof: PASS, 22 checks, at `dfb04d14` (`check-cpu-sm117.json`, `check-cpu-ok`). Dry run OK (`gpu/launch-sm117-dryrun.log`).

## The running launcher

- Launched 01:11 Sun 11-15 from `4acaadb2`, hidden via WMI: wrapper bash pid 35984, launcher bash pid 9452 (`bash .../launch-sm117.sh`).
- It enqueues **nothing** until the longer cool-down for Tiny (SM115)'s launcher log (`wt/sm115/artifacts/small-model/sm115/gpu/launch-sm115.log`, read only) ends with a `done` or `STOP` line; it checks every 5 min. SM115 is expected to end ~14:15 Sun (or ~05:30 if its early stop fires).
- Main log: `C:/Users/jonat/workspace/wt/sm117/artifacts/small-model/sm117/gpu/launch-sm117.log` (lines `[sm117 MM-DD HH:MM:SS]`); stdout/stderr of the launcher: `gpu/launch-sm117.nohup`. Queue owner `small-model`, purposes start with `time heads for the family (SM117): `.
- Breaks: the launcher checks quiet.json, the daily 20:00-01:00 break and Sunday from 16:00 before every ticket; a ticket starts only when its budget fits before the next break.

## Steps, budgets, outputs (in order)

| step | expected / budget (min) | output |
|---|---|---|
| per model, order sm106, sm98, sm101, sm99, sm91s2, sm92: smoke | 3 / 12 (Big 5 / 20) | `gpu/check-<key>.out`, `gpu/smoke-grade-check-<key>.json`, marker `gpu/smoke-ok-sm117-<key>` |
| head training | 55 / 90 (Big 95 / 170) | `artifacts/small-model/sm117-<key>-head/` (`head.pth`, `ablation-head.pth`, `master.pth` = frozen trunk, `train.json`, `resume.pt`); log `gpu/train-<key>.out` |
| grade check (5 users, show-time w8+s8, release vs trunk) | 7 / 10 (Big 10 / 15) | `gpu/grade-rows-<key>.jsonl`, `gpu/grade-dump-<key>/`, `gpu/grade-check-<key>.json`, marker `gpu/grade-ok-sm117-<key>` |
| scoring, per model (723 users; pass 2 at cap 0.92 only for failed users) | 18 / 30 (Big 25 / 45) | `gpu/sm117-<key>-rows.jsonl` + histograms `gpu/sm117-<key>-rows.hist/<user>.npz`; log `gpu/score-<key>.out` |
| decide per model + report (CPU) | 1 / 10 | `<key>-gate.json`, `sm117-result.md`, `sm117-result.json` |

Keys: sm106 = the Tiny two-head model (SM106), sm98 = the Small Instant-Only model (SM98), sm101 = the Tiny Instant-Only model (SM101), sm99 = the show-time instant-only big model (SM99), sm91s2 = the show-time big model (SM91s2), sm92 = the show-time retrain (SM92) (also scores its shipped head as `shipped`). About 600 GPU min in total. ETA of `sm117-result.md`: ~11:30 Mon 11-16 if everything starts Mon 01:00 (~10:30 if the Tiny training fits Sun 14:15-16:00); later if other tickets share the queue.

## Log lines to watch

- Per model: `<key> smoke: [sm117-train] step 1: time loss 2.7726 ...`, `<key> head training: [sm117-train] complete: ...`, `[sm117-grade] <key> 5 users: released vs the trunk after head training bitwise True; ...`, `<key> scored users: N of 723`.
- End: `decide <key> exit 0|3: [sm117-rule] <key>: ... PASS|FAIL` (exit 3 = gate fail, not a launcher failure), `report exit 0: [sm117-report] ...`, then `done`.
- Failure: any `STOP:` line names the step and its output file. A Big smoke that skips the longest user (out of memory) only logs a `WARNING` (the training records skipped users in `train.json`; report them).

## Follow-up commands (CPU, after `done`)

```
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; WT=C:/Users/jonat/workspace/wt/sm117; S=$WT/artifacts/small-model/sm117; G=$S/gpu
# re-run the rule of one model, then the six-model report (writes $S/<key>-gate.json, $S/sm117-result.md / .json)
for key in sm106 sm98 sm101 sm99 sm91s2 sm92; do
  $PY -X utf8 $WT/scripts/sm117_head.py decide --model $key --rows $G/sm117-$key-rows.jsonl --grade-check $G/grade-check-$key.json \
    --train-json $WT/artifacts/small-model/sm117-$key-head/train.json
done
$PY -X utf8 $WT/scripts/sm117_head.py report
```

Then commit on `sm117` with `git add -f`: `sm117-result.md`, `sm117-result.json`, `*-gate.json`, `gpu/*.log`, `gpu/*.out`, `gpu/*.json`, `gpu/sm117-*-rows.jsonl`, `gpu/grade-rows-*.jsonl`, `../sm117-*-head/train.json` (not `.pth`, not `resume.pt`, not `gpu/grade-dump-*/`, not `gpu/*.hist/`, not `work/`), fetch + merge, push.

Factory page (main, `C:/Users/jonat/workspace/anki-rwkv`): close the seven SM117 plan rows (`state` done, `ended_at`) and add one results row per model (or one row for all six) from `sm117-result.json`: per model the gate (grade unchanged, time LogLoss head vs the user's running median with CI), and per S the head's LogLoss vs baselines (a) / (b) with CIs, ECE and AUC, and the "reliable" verdict; times from `gpu/launch-sm117.log`. Every model as "plain name (ID)", "%" not "points", no "Expertium". Then `python "$TEMP/bare.py"` (must print 0), `.venv/Scripts/python.exe scripts/publish_factory_page.py`, the same with `--check` (SYNC 0), commit + push main.

## Stop / relaunch

- Stop between tickets: `touch C:/Users/jonat/workspace/wt/sm117/artifacts/small-model/sm117/gpu/STOP` (remove it before relaunching). To stop a running head training, cancel its queue ticket (`scripts/r01_queue.py cancel <ticket>`) and kill the launcher; a relaunch resumes from `resume.pt` (written after every buffer pass).
- Relaunch (finished steps are skipped by their markers / `train.json` status / scored users):

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm117/artifacts/small-model/sm117/launch-sm117.sh >> C:/Users/jonat/workspace/wt/sm117/artifacts/small-model/sm117/gpu/launch-sm117.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- Alive check: `tail -5 .../gpu/launch-sm117.log`; `powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | ? { $_.CommandLine -like '*launch-sm117*' } | select ProcessId,CommandLine"`. After a Claude app restart, check it; if the log has been silent for more than 10 min outside a ticket and outside the SM115 wait (that one logs once an hour), relaunch.
- If SM115's launcher dies without a `done` / `STOP` line and is not relaunched, this launcher keeps waiting: relaunch with `SKIP_SM115_WAIT=1` in the bash command only once SM115 is really finished.

Do not edit `scripts/`, `expertium/` or `rwkv/` in this worktree while it runs (the launcher refuses uncommitted source changes at start).
