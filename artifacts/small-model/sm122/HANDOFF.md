# Handoff: retrieval-effort probe (SM122)

Design and pre-registered rules: `design.md` (committed before any result). Code: `scripts/sm122_effort.py` (imports read only from `wt/sm121`). CPU only, no GPU, no queue ticket. Cite the paper as Pyc & Rawson (2009).

**Answer time** = front + back: question shown to grade pressed, capped at the preset's maximum answer seconds. In the models it is feature 8 on the history rows.

## State at handoff

- Tue 11-17 08:12. Design pre-registered and committed (`d8f567e1`), one amendment after the CPU check of the show-time retrain (SM92) (K3 excluded the target's own query row; `d6568fae`). That check passed every other proof (K4: exact -1.15% of R against batched -1.12%, Pearson 0.9994): `run/check-sm92.json` (not committed; the full check writes `check-cpu.json`).
- **Two hidden launchers (WMI):**
  1. `MODELS="sm92 sm106" launch-sm122.sh` (started 08:07, wrapper pid 21888): the Small two-head and Tiny two-head passes on 200 users, about 124 s per 20,000-review user with all eight arms, about 6 h in total (expected end about 14:00-15:00), then `report`.
  2. `launch-sm122-big.sh` (started 08:08, wrapper pid 1648): the CPU check of all three models (`check-cpu.json`, about 50 min), then, only if it passes, `MODELS=sm91s2 launch-sm122.sh`: the show-time big model (SM91s2) on 60 users, four arms, about 2.5-3 h, then `report` again.
- Whichever launcher ends last writes the complete `sm122-result.*`; if the Big launcher ends first, the second report (from launcher 1) contains everything. Done = both "done" lines in `run/launch-sm122.log` and `work/{sm92,sm106,sm91s2}/DONE`. If the check failed (`check exit 1` in the log), read `run/check.out` / `check-cpu.json`: Big did not run; report Small and Tiny only (or fix, amend `design.md`, rerun the check and `MODELS=sm91s2 launch-sm122.sh`).
- If the report was written while one model was still running (partial users), rerun `report` once everything is DONE.

## The running launcher

- `artifacts/small-model/sm122/launch-sm122.sh`, hidden via WMI. It starts three python processes in parallel (`run --model sm92`, `sm106`, `sm91s2`), each checks quiet.json (scope all/cpu) and the daily 20:00-01:00 break before every user and sleeps through a window, then it runs `report` -> `sm122-result.md` + `sm122-result.json`, and logs `done`.
- Log: `run/launch-sm122.log`; per model `run/run-<key>.out` (one line per user with seconds and peak GB); report `run/report.out`. Per-user arrays: `work/<key>/<user>.npz` + `work/<key>/meta.json` + `work/<key>/DONE`. **Never commit `run/` or `work/`.**
- Alive: `tail -3 .../run/run-sm92.out`; `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ? { $_.CommandLine -like '*sm122_effort*' }`. Detached jobs died at Claude app self-updates before: check after a restart and relaunch (finished users are skipped).
- Stop: `touch .../sm122/run/STOP` (checked before every user; remove before relaunching).
- Relaunch:

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm122/artifacts/small-model/sm122/launch-sm122.sh >> C:/Users/jonat/workspace/wt/sm122/artifacts/small-model/sm122/run/launch-sm122.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- If only the report is needed again (e.g. after a code fix of the report): `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 scripts/sm122_effort.py report` in `wt/sm122` (CPU, a few minutes).

## Follow-up after `done` (fresh agent, Sonnet medium is enough)

1. Read `run/launch-sm122.log` (three DONE), `sm122-result.md`, the privacy scan line in `run/report.out` (must say clean). Add a short top paragraph to `sm122-result.md` in plain sentences: (1) the model's learned duration effect, sign and size, x2 arm, instant R at the next review and curve R at 1 w / 30 d, in % of R, for each model; (2) whether real outcomes show a residual duration effect (the cross-fitted LogLoss gain of the quintile offsets, within card and within user, against the 0.05% bar); (3) the extra same-day success, model and real data; (4) what it implies for scheduling and for the simulator (which feeds sampled answer times back as inputs: if the model's effect is real-sized, the simulator's sampled times move the predictions and must come from a realistic time model; if it is negligible, the times matter little for recall). Every model as "plain name (ID)", "%" never "points", never the word "Expertium".
2. Commit on `sm122` with `git add -f` only: `sm122-result.md`, `sm122-result.json` (gzip if over 5 MB), `check-cpu.json`, `launch-sm122.sh`, this file. Never `work/`, never `run/`, no `.npz`, no `.pth`. `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Fetch + merge origin, push `sm122` (never bypass the pre-push hook).
3. Factory page on main (`C:/Users/jonat/workspace/anki-rwkv`, `git fetch origin && git merge --ff-only origin/main` first): close the SM122 plan row (`state` "done", `ended_at`, `when` "Done HH:MM ...") and add a results row modelled on the SM123 row (targeted text insertion, keep CRLF, no json.dumps reformat; `detail.notes` MUST be a JSON list of strings; times from `run/launch-sm122.log`). Then `python "$TEMP/bare.py"` (prints 0), `.venv/Scripts/python.exe scripts/publish_factory_page.py`, `--check` (SYNC 0); commit only the two `pages/data` json files, fetch + merge, push main.
4. Summary to the user (under 300 words): learned effect (sign, size), residual effect in real outcomes, same-day result, implications for scheduling and the simulator, commits, CPU time (from `work/<key>/meta.json` seconds).
