# Handoff: Rolling recalibration on the 732 set (SM125)

Design and pre-registered rules: `design.md` (per-user causal rolling Platt, 7-day refits on the user's latest 30,000 earlier reviews, raw model before 1,000 earlier reviews; reliable = ECE < 0.01 and LogLoss below both SM120 baselines with the 95% CI from 2,000 user bootstraps below 0 and at least 0.05%; go = G1 all 21 cells of 2-10 s reliable after recalibration, G2 "Good or Easy within 4 s" better than raw, G3 no cell worse than raw, G4 P(Good or Easy) / P(Easy) reliable). Code: `scripts/sm125_recal.py` (imports `scripts/sm120_fast.py` and `scripts/sm117_head.py`, not edited). Branch `sm125`, cut from `origin/sm120` because the SM117 / SM120 code is not on main.

**Answer time** = front + back: question shown to grade pressed, capped at the preset's maximum answer seconds. Every write-up says "answered within S s, reading the back included".

## CPU proofs: PASS (`check-cpu.json`, before any GPU run)

4 small users (2388, 1952, 3922, 4563), the Tiny two-head model (SM106), CPU: the dump reproduces SM120's GPU sums (7e-6 relative), Platt equals scipy (2e-8), causality (later labels flipped: earlier predictions bitwise unchanged), identity without history, repair invariants, synthetic recovery (1.58, 0.40 vs 1.6, 0.4; ECE 0.064 to 0.005), end to end, CPU cost about 1.2 CPU hours per model.

## The running launcher

- Launched Tue 11-17 14:02 from `59df151d`, hidden via WMI: wrapper bash pid 13652, launcher bash pid 23148.
- It enqueues **nothing** until the last line of the launcher log of the show-time two-head cool-down for Tiny (SM124) (`wt/sm124/artifacts/small-model/sm124/gpu/launch-sm124.log`, read only) is a `done` or `STOP` line; checks every 5 min, logs once an hour.
- Then, through `scripts/r01_queue.py` (owner `small-model`, purposes start with `Rolling recalibration on the 732 set (SM125): `, cap 0.8): a 3-user GPU smoke of the Tiny two-head model (SM106) checked against SM120's sums within 1e-5 (6 min); one dump ticket per model (30 min budget, about 17 min expected) in the order the longer Small Instant-Only model (SM113), the longer cool-down for Tiny (SM115), the show-time two-head cool-down for Small (SM118), the Small Instant-Only model (SM98), the Tiny two-head model (SM106); a missing-users pass at 0.92 (15 min) only if fewer than 723 users dumped. After each dump, that model's CPU recalibration starts in the background (1 thread, under 1.5 GB RAM, about 1.2 h); it stops 5 min before every break and the launcher resumes it after. Then `sm125-result.md` / `.json`.
- Breaks: quiet.json (scope all/gpu), daily 20:00-01:00, Sunday from 16:00, checked before every ticket and by the CPU step.
- Log: `C:/Users/jonat/workspace/wt/sm125/artifacts/small-model/sm125/gpu/launch-sm125.log` (lines `[sm125 MM-DD HH:MM:SS]`); stdout/stderr `gpu/launch-sm125.nohup`.

## Expected timeline

SM124's verdict is expected about 05:00 to 09:40 Wed 11-18. Then GPU about 1.5 to 1.8 h (smoke 6 min + 5 x about 17 min; budget up to 2.6 h with retries); the last model's recalibration ends about 1.2 h after its dump; `sm125-result.md` roughly 08:00 to 12:45 Wed 11-18 (later if SM124 runs late or a 20:00 break intervenes).

## Outputs

- `gpu/dump-<key>.jsonl`: one index record per user (review sha256, cap, days, float64 raw sums): small, commit.
- `work/dump-<key>/<user>.npz`: per-review dump, about 4.4 GB per model, 22 GB for five. **Never commit** (`work/`, `.npz`). Delete after the result is committed if disk is needed (it can be re-dumped in about 1.5 GPU h).
- `rows/sm125-<key>-recal.jsonl`: per-user sums of every arm and event (about 15 MB per model): commit gzipped (`gzip -k`, add only the `.jsonl.gz`).
- `work/recal-<key>.hist/<user>.npz`: fine histograms for ECE / AUC, do not commit.
- `sm125-result.md`, `sm125-result.json`: verdict per model, "Good or Easy within 4 s" table, reliable S raw to recalibrated, ECE per S, grade-only events, coverage and fits.
- Log lines to watch: `GPU smoke passed`, `<key> dumped users: N of 723`, `<key> recalibrated`, `report exit 0: [sm125-report] ...`, `done`. Any `STOP:` line names the step and file.

## Follow-up after `done` (fresh agent, Sonnet medium is enough)

1. Read `sm125-result.md`; check its top paragraph (answer-time definition) and the "dump equals SM120" column (all yes). Commit on `sm125` with `git add -f`: `sm125-result.md`, `sm125-result.json`, `gpu/*.log`, `gpu/*.out`, `gpu/dump-*.jsonl`, `rows/*.jsonl.gz` (not `work/`, no `.npz`, no `.pth`); `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; fetch + merge `origin/sm125`, push.
2. Factory page (main, `C:/Users/jonat/workspace/anki-rwkv`; `git fetch origin && git merge --ff-only origin/main` first): close the SM125 plan row (`state` done, `ended_at`) and add a results row (times from `gpu/launch-sm125.log`; `detail.notes` as a JSON list of strings): per model go / no go, "Good or Easy within 4 s" ECE raw to recalibrated and LogLoss recalibrated vs raw and vs (a) / (b) in %, reliable 2-10 s cells raw to recalibrated, share of rows recalibrated. Then the bare-ID check, `.venv/Scripts/python.exe scripts/publish_factory_page.py`, `--check`, commit the two `pages/data` json files only, fetch + merge, push.
3. A plain-language summary to the user: every model as "plain name (ID)", "%" not "points", "answered within S s, reading the back included", no "Expertium" (say "outside research").

## Plan row to insert now (by the agent that owns `pages/data/factory-curated.json`; this agent did not edit it)

```json
{
 "id": "SM125",
 "track": "small model",
 "since": "11-17T14:02",
 "kind": "gpu",
 "minutes": 110,
 "match": [
  "Rolling recalibration on the 732 set (SM125): "
 ],
 "what": "Rolling recalibration on the 732 set (SM125): fast Good/Easy prediction (SM120) found the timed events (for example Good or Easy and answered within 4 s, reading the back included) beat both baselines by about 20% and 29% LogLoss but only one model passes the calibration bar (ECE below 0.01). SM125 re-dumps per-review outputs of five release family candidates (the longer Small Instant-Only model (SM113), the longer cool-down for Tiny (SM115), the show-time two-head cool-down for Small (SM118), the Small Instant-Only model (SM98), the Tiny two-head model (SM106)) on the GPU, then recalibrates each user causally on the CPU: Platt scaling refitted every 7 days on the user&rsquo;s latest 30,000 earlier reviews, raw model before 1,000 (what a client could do with its own history). Go per model = every timed event at 2 to 10 s reliable after recalibration (ECE below 0.01 and LogLoss below both baselines, CI below 0 and at least 0.05%), Good or Easy within 4 s better than raw, no cell worse, P(Good or Easy) and P(Easy) still reliable",
 "when": "Queued since 14:02 Tue 11-17, waits for the show-time two-head cool-down for Tiny (SM124) to end (about 05:00 to 09:40 Wed 11-18); then about 1.5 to 1.8 GPU h (smoke + 5 dumps) and about 1.2 h of CPU recalibration after the last dump; result about 08:00 to 12:45 Wed 11-18",
 "decides": "Whether the Anki client should recalibrate the timed Good/Easy predictions per user from its own history before showing them, for which release models, and whether that keeps their LogLoss lead over the baselines",
 "state": "queued",
 "started_at": "2026-11-17T14:02+01:00"
}
```

## Stop / relaunch / alive check

- Stop between steps: `touch .../sm125/gpu/STOP` (remove before relaunching). A running dump ticket: `scripts/r01_queue.py cancel <ticket>` and kill the launcher; dumped and recalibrated users are kept and skipped on relaunch.
- Relaunch:

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm125/artifacts/small-model/sm125/launch-sm125.sh >> C:/Users/jonat/workspace/wt/sm125/artifacts/small-model/sm125/gpu/launch-sm125.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- Alive: `tail -5 .../gpu/launch-sm125.log`; `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | ? { $_.CommandLine -like '*launch-sm125*' }`. After a Claude app restart check it (detached jobs died at app self-updates before). If SM124's launcher died without a `done` / `STOP` line and is not relaunched, relaunch SM125 with `export SKIP_SM124_WAIT=1;` in the bash command only once SM124 is really finished.
- Do not edit `scripts/`, `expertium/`, `rwkv/` in `wt/sm125` while it runs (the launcher refuses uncommitted source changes at start). Never touch `wt/sm124` or `wt/sm122`.
