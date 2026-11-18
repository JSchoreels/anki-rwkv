# Handoff: Rolling recalibration for the Big models (SM128)

Exact recipe of rolling recalibration on the 732 set (SM125) (`../sm125/design.md`), same pre-registered rules (copied into `design.md`), applied to the two Big models: the show-time big model (SM91s2) and the show-time instant-only big model (SM99), each with its head from the time heads for the family (SM117). Code: `scripts/sm125_recal.py` (only change: env `SM125_TAG=sm128`, `SM125_MODELS="sm91s2 sm99"` choose the output folder, file prefix, model list and report title; defaults are SM125's). Branch `sm128` (cut from `origin/sm125`).

**Answer time** = front + back: question shown to grade pressed, capped at the preset's maximum answer seconds. Every write-up says "answered within S s, reading the back included".

## The running launcher

- Launched 12:15 Wed 11-18 from `51834bc5` (smoke passed 12:15:57, equals SM120 within 3e-7). Job: `C:/Users/jonat/workspace/wt/sm128/artifacts/small-model/sm128/launch-sm128.sh`, launched hidden via WMI (see relaunch). No waiting on another launcher.
- Steps: GPU smoke (3 users 2388, 1952, 3922, show-time big model (SM91s2), equals SM120's recorded sums within 1e-5; 10 min budget) -> dump of the show-time big model (SM91s2) (45 min budget, about 20 expected; missing users at cap 0.92 like SM120) -> its CPU recalibration in the background -> dump of the show-time instant-only big model (SM99) -> its recalibration after the first one has ended (one at a time, 1 thread, under 6 GB RAM, about 1.2 h each; another CPU job (SM127) runs until about 14:00) -> `sm128-result.md` / `.json`.
- Queue: owner `small-model`, purposes start with `rolling recalibration for the Big models (SM128): `, cap 0.8. Breaks (quiet.json, daily 20:00-01:00, Sunday from 16:00; CPU quiet too) checked before every ticket and before CPU work; the recalibration stops itself 5 min before a break and resumes after it.
- Log: `C:/Users/jonat/workspace/wt/sm128/artifacts/small-model/sm128/gpu/launch-sm128.log` (lines `[sm128 MM-DD HH:MM:SS]`); stdout/stderr `gpu/launch-sm128.nohup`; per-step `gpu/smoke.out`, `gpu/dump-<key>.out`, `gpu/recal-<key>.out`.
- Patterns: `GPU smoke passed`; `<key> dumped users: N of 723`; `<key> recalibrated`; `report exit 0: [sm125-report] 2 models: ...`; final line `done` (success) or `STOP: <step and file>` (failure).
- Expected: GPU about 50 min (smoke 6 + two dumps about 22 each); first recalibration ends about 1.2 h after its dump, the second about 1.2 h later; report about 30 min; `sm128-result.md` about 15:45 Wed 11-18 (up to 16:30), before the 20:00 break.
- Outputs: `gpu/dump-<key>.jsonl` (commit), `work/dump-<key>/*.npz` (about 4.4 GB each, NEVER commit), `rows/sm128-<key>-recal.jsonl` (commit the gzipped `.jsonl.gz` via `gzip -k`), `work/recal-<key>.hist` (do not commit), `sm128-result.md`, `sm128-result.json`.

## Alive check / stop / relaunch

- Alive: `tail -5 C:/Users/jonat/workspace/wt/sm128/artifacts/small-model/sm128/gpu/launch-sm128.log`; `Get-CimInstance Win32_Process | ? CommandLine -match 'launch-sm128'`; `.venv/Scripts/python.exe scripts/r01_queue.py status` (in `C:/Users/jonat/workspace/anki-rwkv`). After a Claude app restart or a power event check it (detached jobs died before).
- Stop between steps: `touch C:/Users/jonat/workspace/wt/sm128/artifacts/small-model/sm128/gpu/STOP` (remove before relaunching). A running ticket: `scripts/r01_queue.py cancel <ticket>` and kill the launcher; dumped and recalibrated users are kept and skipped on relaunch.
- Relaunch (exact):

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm128/artifacts/small-model/sm128/launch-sm128.sh >> C:/Users/jonat/workspace/wt/sm128/artifacts/small-model/sm128/gpu/launch-sm128.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- Do not edit `scripts/`, `expertium/`, `rwkv/` in `wt/sm128` while it runs (the launcher refuses uncommitted source changes at start). Never touch other worktrees (read only: `wt/sm117`, `wt/sm125`).

## Follow-up after `done` (fresh agent, Sonnet medium is enough)

1. Read `sm128-result.md`; check the answer-time definition at the top and the "dump equals SM120" column (all yes). On `sm128`, commit with `git add -f`: `sm128-result.md`, `sm128-result.json`, `gpu/*.log`, `gpu/*.out`, `gpu/dump-*.jsonl`, `rows/*.jsonl.gz` (not `work/`, no `.npz`, no `.pth`); `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; fetch, merge `origin/sm128`, push.
2. Factory page (main, `C:/Users/jonat/workspace/anki-rwkv`; `git fetch origin && git merge --ff-only origin/main` first): close the SM128 plan row (`state` done, `ended_at`), add a results row (times from `gpu/launch-sm128.log`; `detail.notes` a JSON list of strings): per model go / no go, "Good or Easy within 4 s" ECE raw to recalibrated and LogLoss recalibrated vs raw and vs both baselines in %, reliable 2-10 s cells raw to recalibrated, share of rows recalibrated. Update `family.gen.rows` in `pages/data/factory-curated.json` for the two Big rows' `time_cur` text with the recalibrated outcome. Then the bare-ID check, `.venv/Scripts/python.exe scripts/publish_factory_page.py`, `--check`, commit the two `pages/data` json files only, fetch + merge, push.
3. Plain-language summary: every model as "plain name (ID)", "%" not "points", "answered within S s, reading the back included", no "Expertium" (say "outside research").

## Plan row to insert now (by the agent that owns `pages/data/factory-curated.json`; this agent did not edit it)

```json
{
 "id": "SM128",
 "track": "small model",
 "since": "11-18T12:15",
 "kind": "gpu",
 "minutes": 50,
 "match": [
  "rolling recalibration for the Big models (SM128): "
 ],
 "what": "Rolling recalibration for the Big models (SM128): the exact recipe of rolling recalibration on the 732 set (SM125), which got a go for all five release family candidates, applied to the show-time big model (SM91s2) and the show-time instant-only big model (SM99) with their time heads for the family (SM117). Per-review outputs on the GPU (fast Good/Easy prediction (SM120) already scored both, so the dump is checked against its sums), then a per-user causal Platt scaling refitted every 7 days on the latest 30,000 earlier reviews on the CPU (about 1.2 h per model, one model at a time), raw model before 1,000. Same pre-registered go rule (every timed event at 2 to 10 s reliable: ECE below 0.01 and LogLoss below both baselines, CI below 0 and at least 0.05%; Good or Easy answered within 4 s, reading the back included, better than raw; no cell worse; P(Good or Easy) and P(Easy) still reliable)",
 "when": "Started 12:15 Wed 11-18; GPU (smoke + two dumps) about 50 min, then about 1.2 h of CPU per model one after the other; result about 15:45 Wed 11-18 (up to 16:30)",
 "decides": "Whether the Big models can ship timed Good/Easy predictions that are reliable after per-user recalibration from the client's own history, like the five other family models",
 "state": "running",
 "started_at": "2026-11-18T12:15+01:00"
}
```
