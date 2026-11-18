# Handoff: time head for the show-time two-head cool-down for Tiny (SM126)

Design and pre-registered rules: `design.md`. Part A = the recipe of the time heads for the new candidates (SM119) for the show-time two-head cool-down for Tiny (SM124) (key `sm124`, final master = its pass-2 folder, sha256 `c78b24d8...`). Part B = the recipe of the rolling recalibration on the 732 set (SM125) for the same model with its new head (its own SM120-code scoring first: the baselines depend on the model's grades). Branch `sm126` (from `origin/sm125` + `origin/sm119`), worktree `C:/Users/jonat/workspace/wt/sm126`. Code: `scripts/sm117_head.py`, `scripts/sm125_recal.py` (small edits, committed before the launch), `scripts/sm120_fast.py` (unchanged).

**Answer time** = front + back: question shown to grade pressed, capped at the preset's maximum answer seconds. Every write-up says "answered within S s, reading the back included".

## The running launcher

- Launched Wed 11-18 08:44 from `ff2d284b` (branch `sm126`), hidden via WMI: wrapper bash pid 20572, launcher bash pid 20312 (`bash .../launch-sm126.sh`). First ticket 001330 (part A smoke), running at 08:44. Dry run `gpu/launch-sm126-dryrun.log` DRYRUN OK; `bash -n` OK. No other launcher is waited for.
- Expected end: part A (head training, grade check, scoring, `decide`, `report`) about 09:50; part B GPU (SM120-code scoring, dump smoke, dump) about 10:30; recalibration about 1.2 h CPU after it; `sm126-recal-result.md` and `done` about 11:45 Wed 11-18 (later if a window intervenes; the next break is 20:00).

- Log: `C:/Users/jonat/workspace/wt/sm126/artifacts/small-model/sm126/gpu/launch-sm126.log` (lines `[sm126 MM-DD HH:MM:SS]`); stdout/stderr `gpu/launch-sm126.nohup`; ticket outputs `gpu/{check,train,grade,score}-sm124.out`, `gpu/sm120-score.out`, `gpu/smoke-b.out`, `gpu/dump-sm124.out`, `gpu/recal-sm124.out`.
- Queue: owner `small-model`, purposes start with `time head for the show-time two-head cool-down for Tiny (SM126): `; GPU cap 0.8 (pass 2 for missing users 0.92). The launcher checks the GPU break itself before every ticket (quiet.json scope all/gpu, daily 20:00-01:00, Sunday from 16:00); the CPU recalibration stops 5 min before a window.
- Steps and expected GPU minutes: part A smoke (about 3), head training (about 42, budget 90), grade check (about 1), scoring (about 15), `decide` + `report` on CPU; part B SM120-code scoring (about 17), 3-user dump smoke (about 3), dump (about 17), CPU recalibration (about 72 min, one process), report. About 1.6 GPU h plus 1.2 h CPU.
- Outputs: heads `artifacts/small-model/sm126-sm124-head/` (`head.pth`, `ablation-head.pth`, `master.pth`, `train.json`; do not commit `.pth`), `sm124-gate.json`, `sm126-result.{md,json}` (part A), `sm126-recal-result.{md,json}` (part B), `gpu/sm126-sm124-rows.jsonl`, `gpu/sm120-sm124-rows.jsonl`, `gpu/dump-sm124.jsonl`, `rows/sm125-sm124-recal.jsonl` (about 15 MB: gzip -k, add only the `.gz`), `work/` (dumps about 4.4 GB: never commit).

## Log lines to watch

- Done: the last line is `[sm126 ...] done`. Part A finished: `PART A done (time head of sm124 trained, grade-checked and scored)`; `decide sm124 exit 0|3` (3 = gate fail, not a launcher failure); `[sm117-grade] ... bitwise True`; `report (part A) exit 0`. Part B: `sm124 SM120-code scored users: 723 of 723`, `part B GPU smoke passed`, `sm124 dumped users: 723 of 723`, `dump vs head scoring (all users) exit 0`, `sm124 recalibrated`, `report (part B) exit 0: [sm125-report] ...`.
- STOP: any line `[sm126 ...] STOP: ...` (it names the step and file); the last line `STOP: ended with exit 1` or `STOP: STOP file present`.

## Stop / relaunch / alive check

- Stop between tickets: `touch C:/Users/jonat/workspace/wt/sm126/artifacts/small-model/sm126/gpu/STOP` (remove before relaunching). A running ticket: `scripts/r01_queue.py cancel <ticket>` and kill the launcher; the head training resumes from `resume.pt`, dumped / recalibrated users are skipped.
- Relaunch (finished steps are skipped), PowerShell:

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm126/artifacts/small-model/sm126/launch-sm126.sh >> C:/Users/jonat/workspace/wt/sm126/artifacts/small-model/sm126/gpu/launch-sm126.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- Alive: `tail -5 .../gpu/launch-sm126.log`; `Get-CimInstance Win32_Process | ? CommandLine -match 'launch-sm126'`; `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status` (from `C:/Users/jonat/workspace/anki-rwkv`). After a Claude app restart or a power event check it (detached jobs died then before); relaunch from here.
- Do not edit `scripts/`, `expertium/`, `rwkv/` in `wt/sm126` while it runs (the launcher refuses uncommitted source changes at start of part B). Never touch `wt/sm124`, `wt/sm125`, `wt/sm119` (read only).

## Follow-up after done (fresh agent, Sonnet medium is enough)

1. Read `sm126-result.md` (part A: gate, time LogLoss, reliable S) and `sm126-recal-result.md` (part B: verdict G1-G4, "Good or Easy within 4 s" raw vs recalibrated, reliable cells, column "dump equals SM120" must be yes). Check that both start with / contain the answer-time definition (front + back, answered within S s, reading the back included); add a short paragraph at the top if missing.
2. Commit on `sm126` with `git add -f`: `sm124-gate.json`, `sm126-result.{md,json}`, `sm126-recal-result.{md,json}`, `gpu/*.log`, `gpu/*.out`, `gpu/*.json`, `gpu/sm126-sm124-rows.jsonl`, `gpu/sm120-sm124-rows.jsonl`, `gpu/dump-sm124.jsonl`, `gpu/grade-rows-sm124.jsonl`, `rows/sm125-sm124-recal.jsonl.gz`, `../sm126-sm124-head/train.json` (not `work/`, no `.npz`, no `.pth`, no `resume.pt`, no `grade-dump-*`, no `*.hist`); `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; `git fetch origin`, merge `origin/sm126`, push `sm126`. Optionally delete `work/dump-sm124` (4.4 GB) afterwards.
3. Factory page (main, `C:/Users/jonat/workspace/anki-rwkv`; `git fetch origin && git merge --ff-only origin/main` first): close the SM126 plan row (`state` done, `ended_at`) and add a results row (times from `gpu/launch-sm126.log`; `detail.notes` as a JSON list of strings): part A gate and time LogLoss vs baselines, reliable S; part B go / no go, "Good or Easy within 4 s" ECE raw to recalibrated and LogLoss vs raw and vs (a) / (b) in %, reliable 2-10 s cells raw to recalibrated. Then the bare-ID check, `.venv/Scripts/python.exe scripts/publish_factory_page.py`, `--check`, commit the two `pages/data` json files only, fetch + merge, push.
4. A plain-language summary to the user: every model as "plain name (ID)", "%" not "points", "answered within S s, reading the back included", no "Expertium" (say "outside research").

## Plan row to insert now (by the agent that owns `pages/data/factory-curated.json`; this agent did not edit it)

```json
{
 "id": "SM126",
 "track": "small model",
 "since": "11-18T08:44",
 "kind": "gpu",
 "minutes": 100,
 "match": [
  "time head for the show-time two-head cool-down for Tiny (SM126): "
 ],
 "what": "Time head for the show-time two-head cool-down for Tiny (SM126): the show-time two-head cool-down for Tiny (SM124) got a go at 04:46 Wed 11-18, but answer-time heads are tied to their trunk, so it needs its own. SM126 first trains the answer-time head of the time heads for the new candidates (SM119) on the frozen model (about 43 min), checks that the model's grades stay bitwise unchanged, and scores P(recall within S seconds) (recalled and answered within S s, reading the back included) on the 732 set against both baselines. Then it runs the rolling recalibration on the 732 set (SM125) on this model with its new head: per-review outputs on the GPU, a per-user causal Platt scaling refitted every 7 days on the latest 30,000 earlier reviews on the CPU (about 1.2 h), and the same pre-registered go rule (every timed event at 2 to 10 s reliable, Good or Easy within 4 s better than raw, no cell worse, P(Good or Easy) and P(Easy) reliable)",
 "when": "Started 08:44 Wed 11-18; part A (head training, grade check, scoring) about 09:50; part B GPU (SM120-code scoring, dump) about 10:30; recalibration about 1.2 h of CPU after it; result about 11:45 Wed 11-18 (later if the 20:00 break intervenes)",
 "decides": "Whether the show-time two-head cool-down for Tiny (SM124) can ship an answer-time estimate like the other family models, and whether per-user recalibration of its timed Good/Easy predictions makes them reliable",
 "state": "running",
 "started_at": "2026-11-18T08:44+01:00"
}
```
