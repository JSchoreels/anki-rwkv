# Handoff: Timed recall on your own collection (SM121)

Design and pre-registered bar: `design.md`. Code: `scripts/sm121_own.py` (imports `scripts/sm117_head.py` and `scripts/sm120_fast.py`, not edited). CPU only, no GPU, no queue ticket. Own collection: aggregate outputs only.

**Answer time** = front + back: question shown to grade pressed, capped at the preset's maximum answer seconds. Every write-up must say "answered within S s, reading the back included".

## State at handoff (Mon 11-16 22:05)

- Designed during the daily break (it was already 21:40 when the work started), so **no CPU check was run tonight**: the check is the launcher's first step after 01:05. A synthetic test of the evaluation, discrimination and report code (random data, under a second) ran clean; the dry run of the launcher is clean.
- The repro copy was made tonight: a plain file copy of the profile's `collection.anki2` (Anki closed, no `-wal`, last written 2026-11-03) to `work/repro/collection-copy/` (gitignored). Nothing else under AppData was read except sha256 hashes of the backups folder.

## The running launcher

- Launched Mon 11-16 22:01 from `8009cfc5` (branch `sm121`), hidden via WMI: wrapper bash pid 17604, launcher bash pid 12396 (`bash .../launch-sm121.sh`). It sleeps until Tue 01:05.
- Log: `C:/Users/jonat/workspace/wt/sm121/artifacts/small-model/sm121/run/launch-sm121.log` (lines `[sm121 MM-DD HH:MM:SS]`); per-step output `run/*.out`; stdout/stderr `run/launch-sm121.nohup`. The converter's own output (it prints review dates) goes to `work/convert.out`. **Do not commit `run/` or `work/`.**
- Steps: convert the repro copy -> CPU check (first 800 reviews; any failing check stops the launcher) -> main trunks fp32: the show-time retrain (SM92, both its SM117 head and the SM83 head), the Small Instant-Only model (SM98), the Tiny two-head model (SM106), then Big: the show-time instant-only big model (SM99), the show-time big model (SM91s2) (each killed and skipped after 3 h) -> the reproduction control trunks (SM92 + SM98, int4 card / int8 note state, repro copy) -> `sm121-repro.{md,json}` -> `sm121-result.{md,json}` -> waits (polls every 10 min, until Tue 18:30) for the SM119 heads of the longer Small Instant-Only model (SM113) and the longer cool-down for Tiny (SM115) (`wt/sm119/artifacts/small-model/sm119-sm11{3,5}-head`, `train.json` status complete) -> scores them -> rewrites `sm121-result.*` -> `done`.
- Every step checks quiet.json (scope all/cpu), the daily 20:00-01:00 break and Sunday from 16:00, and starts only if its budget fits before the next window.

## Expected timeline (Tue 11-17)

01:05 convert (1 min) + check (about 5-15 min); small trunks about 10-20 min each (the fork's big model took 6.5 min on 2 threads for a plain forward of this collection; the capture adds to it); Big about 20-60 min each; repro trunks about 20-40 min; repro report a few minutes; main report about 30-60 min (11 combinations x 6 row sets, card-block bootstraps). First `sm121-result.md` roughly 04:00-06:00. The SM119 heads were expected Tue morning (SM119 itself waits for the show-time two-head cool-down for Small (SM118)); then about 1 h more. If they are not there by 18:30, the launcher ends without them (the result lists them as missing).

## Follow-up after `done` (fresh agent, Sonnet medium is enough)

1. Read `check-cpu.json` (all K pass), `sm121-repro.md` and `sm121-result.md` (top paragraph present). Run `python scripts/sm121_own.py`'s privacy scan again if anything was edited by hand. Commit on `sm121` with `git add -f` only: `sm121-result.md`, `sm121-result.json`, `sm121-repro.md`, `sm121-repro.json`, `check-cpu.json` (aggregates). Never `work/`, never `run/`, no `.pth`, no `.npz`. Fetch + merge, push (the pre-push hook audits for real-user files: never bypass it).
2. Factory page (main, `C:/Users/jonat/workspace/anki-rwkv`; `git fetch origin && git merge --ff-only origin/main` first): close the SM121 plan row (`state` done, `ended_at`) and add a results row (times from `run/launch-sm121.log`; `detail.notes` as a JSON list of strings): where "recalled within S s" is reliable per combination on the benchmark rows, the 4 s numbers against the strongest baseline, the Mac-slice comparison, and the discrimination answer (resolution, AUC given Good, within-card correlation, head vs card history). Say "your own collection", never deck names. Then `python "$TEMP/bare.py"` (prints 0), `.venv/Scripts/python.exe scripts/publish_factory_page.py`, `--check` (SYNC 0), commit the two `pages/data` json files only, fetch + merge, push.
3. A plain-language summary to the user: every model as "plain name (ID)", "%" not "points", "answered within S s, reading the back included", timed outcomes kept apart from the untimed grade LogLoss.

## For the Mac (reproduction control)

- **Needed from the Mac:** the backup it used (sha256 `2515ef66b956ed9b28e065aa895a92db539fc2b126fabd5660217711cf36f66b`, 226,374 canonical reviews). No file here matches: the 7 backups of the profile (latest 2026-11-03), its `collection.anki2` (`c5380f78...`, used as the closest copy), the 2026-10-14 copy (`fb1183e1...`), `Downloads/collection-2026-10-11@15-47-24.colpkg` and an iCloud `collection-2026-11-07@06-02-25.colpkg` in a `SPW` folder (not used: not clearly this profile). With the backup: put it as `work/repro/collection-copy/collection.anki2` (a `.colpkg` must be unpacked first), delete `work/repro/copy-info.json`, `work/repro-trunk-*`, relaunch: the convert and repro steps rerun (or run `convert`, the two repro `trunk` commands and `repro` by hand on CPU).
- **Ask the Mac if the numbers disagree materially** (its scripts were in `/private/tmp`, not reachable): (a) "pooled success fraction" in grade x recent read as the smoothed (fast + 0.5) / (n + 1) of R x recent (not the raw ratio); (b) card answers counted with any grade (Again included in "the last three", only successes counted as successes); (c) whether its time-head CDF at T interpolates linearly in ln time inside the bin, as SM117 does; (d) int4 card / int8 note state here is the scorer's WKV-row rounding (`small_model_heldout` s4c8n), which may differ in detail from the fork runtime's; (e) the Yomitan subset via the card's current deck (filtered decks mapped to their home deck).

## Follow-up suggestion (not done, do not touch wt/sm120)

The card-history baselines are cheap on the 732-set scorer: the per-user revlog already carries `card_id`, `duration` and `rating`, so (4), (5) and the Mac's four are a few vectorised lines in `running()` of a new SM120-style scorer. Worth adding to the next 732-set time-head read-out.

## Stop / relaunch / alive check

- Stop between steps: `touch .../sm121/run/STOP` (remove before relaunching); a running step: kill its python (finished trunk files are kept and skipped on relaunch).
- Relaunch (set `START_AT` to an earlier time to skip the 01:05 gate):

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm121/artifacts/small-model/sm121/launch-sm121.sh >> C:/Users/jonat/workspace/wt/sm121/artifacts/small-model/sm121/run/launch-sm121.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- Alive: `tail -5 .../run/launch-sm121.log`; `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | ? { $_.CommandLine -like '*launch-sm121*' }`. After a Claude app restart check it (detached jobs died at app self-updates before).
- Do not edit `scripts/` in this worktree while it runs (the launcher refuses uncommitted script changes when it starts).
