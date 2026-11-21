# Handoff: Fast Good/Easy prediction (SM120)

Design and pre-registered bar: `design.md` (events, baselines, "reliable" = ECE < 0.01 and LogLoss below every baseline with the 95% CI below zero, 2,000 user bootstraps, seed 0). Code: `scripts/sm120_fast.py` (imports `scripts/sm117_head.py`, which is not edited). Scoring only, no training; grades unchanged (the trunk is checked equal to the release before scoring).

**Answer time** = front + back: question shown to grade pressed, capped at the preset's maximum answer seconds. Every write-up must say "answered within S s, reading the back included".

## CPU check: PASS (before any GPU run)

- `check-cpu.json`: the Tiny two-head model (SM106), 3 small users of the 732 pool (2253, 2388, 1952), CPU. The recalled event reproduces SM117's recorded per-user sums (row and event counts exactly; LogLoss / Brier / sum p within 2e-6 relative for the head, 3e-7 for baseline (a), 0 for baseline (b)); the running baselines equal `sm117_head.running_baselines`; the new priors equal SM117's `priors.json` on SM117's S values; event rates and mean p nest (Easy <= Good or Easy <= recalled) and grow with S; label counts match the grade counts.
- `check-cpu-sm91s2.json`: the same checks with the show-time big model (SM91s2): PASS.
- `priors.json`: pooled priors of the running baselines on SM117's 1,991 plan training users (CPU, took under 1 min).

## The running launcher

- Launched Mon 11-16 18:25 from `af54f645` (branch `sm120`), hidden via WMI: wrapper bash pid 21756, launcher bash pid 1988 (`bash .../launch-sm120.sh`).
- It enqueues **nothing** until the launcher log of the time heads for the new candidates (SM119) (`wt/sm119/artifacts/small-model/sm119/gpu/launch-sm119.log`, read only) ends with a `done` or `STOP` line; it checks every 5 min (logs once an hour). Then it adds the SM119 heads whose `train.json` says complete (the longer Small Instant-Only model (SM113), the longer cool-down for Tiny (SM115), and the show-time two-head cool-down for Small (SM118) if SM119 trained it; the SM118 master's sha256 is taken from that head's `train.json` and checked).
- Order: the Tiny two-head model (SM106), the Small Instant-Only model (SM98), the Tiny Instant-Only model (SM101), the show-time instant-only big model (SM99), the show-time big model (SM91s2), the show-time retrain (SM92), then the SM119 models. Per model one ticket at cap 0.8 (budget 30 min, Big 45), then a missing-users pass at 0.92 (15 min) only if fewer than 723 users scored. Queue owner `small-model`, purposes start with `Fast Good/Easy prediction (SM120): `. The launcher checks quiet.json, the daily 20:00-01:00 break and Sunday from 16:00 before every ticket.
- Log: `C:/Users/jonat/workspace/wt/sm120/artifacts/small-model/sm120/gpu/launch-sm120.log` (lines `[sm120 MM-DD HH:MM:SS]`); stdout/stderr `gpu/launch-sm120.nohup`.

## Expected timeline

SM119 is itself waiting for the show-time two-head cool-down for Small (SM118) (expected to end about 03:30 to 12:30 Tue 11-17), then needs about 1.5 to 2.3 GPU h, so SM119 ends roughly Tue 06:00-08:00 (later if SM118 ends late). SM120 then needs about 15 to 25 min per model (SM117's scoring times): about 2 to 2.5 h for six models, plus about 15 min per SM119 model. `sm120-result.md` roughly Tue 08:30-11:00 (into the afternoon if SM118 ends at 12:30).

## Outputs

- per model: `gpu/sm120-<key>-rows.jsonl` (one record per user), `gpu/sm120-<key>-rows.hist/<user>.npz` (fine histograms for ECE / AUC; do not commit), `gpu/score-<key>.out`.
- the launcher's last step (CPU): `sm120-result.md` and `sm120-result.json` in this folder: the reliable S per event and model, the "Good or Easy within 4 s" table, grade-only events, fast events per S, speed given the grade (Hard, Good, Easy) and the predicted median vs the user's running median, and the grade-weighted mixtures.
- Log lines to watch: `models: ...`, `<key> scored users: N of 723`, `report exit 0: [sm120-report] ...`, `done`. Any `STOP:` line names the step and file.

## Follow-up after `done` (fresh agent, Sonnet medium is enough)

1. Read `sm120-result.md`; check its top paragraph (answer-time definition) is there. Commit on `sm120` with `git add -f`: `sm120-result.md`, `sm120-result.json`, `gpu/*.log`, `gpu/*.out`, `gpu/sm120-*-rows.jsonl` (not `gpu/*.hist/`, not `work/`, no `.pth`); fetch + merge, push.
2. Factory page (main, `C:/Users/jonat/workspace/anki-rwkv`; `git fetch origin && git merge --ff-only origin/main` first): close the SM120 plan row (`state` done, `ended_at`) and add a results row (times from `gpu/launch-sm120.log`; `detail.notes` as a JSON list of strings): per model where "Good or Easy within S s" and "Easy within S s" are reliable, the 4 s numbers (LogLoss vs (a) and (b) in %, ECE, AUC), grade-only P(Good or Easy) / P(Easy), speed of a Good / Easy answer, and the mixture medians vs the running median. Then `python "$TEMP/bare.py"` (prints 0), `.venv/Scripts/python.exe scripts/publish_factory_page.py`, `--check` (SYNC 0), commit the two `pages/data` json files only, fetch + merge, push.
3. A plain-language summary to the user: every model as "plain name (ID)", "%" not "points", and "answered within S s, reading the back included".

## Stop / relaunch / alive check

- Stop between tickets: `touch .../sm120/gpu/STOP` (remove before relaunching). A running scoring ticket: `scripts/r01_queue.py cancel <ticket>` and kill the launcher; scored users are kept and skipped on relaunch.
- Relaunch:

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm120/artifacts/small-model/sm120/launch-sm120.sh >> C:/Users/jonat/workspace/wt/sm120/artifacts/small-model/sm120/gpu/launch-sm120.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- Alive: `tail -5 .../gpu/launch-sm120.log`; `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | ? { $_.CommandLine -like '*launch-sm120*' }`. After a Claude app restart check it. If SM119's launcher died without a `done` / `STOP` line and is not relaunched, relaunch SM120 with `export SKIP_SM119_WAIT=1;` in the bash command only once SM119 is really finished.
- Do not edit `scripts/`, `expertium/`, `rwkv/` in this worktree while it runs (the launcher refuses uncommitted source changes when it starts the scoring).
