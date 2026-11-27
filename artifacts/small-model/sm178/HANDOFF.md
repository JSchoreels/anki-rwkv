# Today's models on donor data (SM178): handoff

Status: DONE 11-27 16:48 (196 of 218 donors covered by all six models; 22 largest donors OOM). Read-out in RESULT.md, sm178-result.md/json.
ticket 001793 = Tiny pass 1, 116 users left). No training; scoring only.

Question: how do the six release-candidate models (Big two-head (SM91s2), Big Instant-Only (SM99), Small two-head (SM118), Small Instant-Only (SM113), six-layer Tiny two-head and Instant-Only (SM130)) score on the 218 donated users (`data/donations-20261126-answer-time`, 39.9M reviews), against the 732 set? Label REF = `w8+s8hx8` (int8 weights, int8 card and note rows), the label of SM148's 732-set rows.

## Done

- Worktree `C:/Users/jonat/workspace/wt/sm178`, branch `sm178` from origin/main (d605a8b2).
- Scorer: `scripts/small_model_heldout.py` of the sm164 worktree (commit c30a0070, clean, unchanged), wrapped by `sm178_score.py`, which points it at the donations dataset exactly as SM177's CPU smoke did (replaces `compress_common.run_info / data_path / RUN` and the pool). Same flags as SM164: `--inputs e2s-query --deleted-cards neighbour --split-existing --nonfinite-rows --prior-buckets ...`, `PYTORCH_JIT=0`. Benchmark test reviews: FSRS-6 label filter (`rwkv/find_equalize_test_reviews.process`, from the main checkout) built per donor into `work/sm178/run/label-db` (not committed).
- Model sha256 pins (`models.sh`, copy of SM164's) all OK.
- **GPU smoke OK** (`smoke.sh`, tickets 001788 and 001789, 1 min): six-layer Tiny two-head (SM130) at `w8+s8` on SM177's 20 smoke users gives curve 0.334454 / instant 0.323671 on 1,621,900 benchmark reviews, equal to SM177's CPU values to 6 decimals; the six models at REF on 3 donors give 18 finite rows (`gpu/smoke-rows.jsonl`, `gpu/smoke-check.json`, `gpu/smoke-ok`). GPU rate about 40-50k review-evaluations/s on the Tiny.
- `sm178_report.py report` tested on the smoke rows (works; exit 2 there only because 3 of 218 users are covered).

## 11-27 crash and fix

- First run: labels done 15:27; ticket 001790 (Tiny) stopped after 106 of 218 users (3.5 min) with a CUDA device-side
  assert (`gpu/score.out`). Cause: donor 1000029 has one review with a zeroed id (1970), so a 55-year interval, past
  the model's interval grid (exp(21) s, about 42 years): `srs_model.interp`'s searchsorted index runs off the grid.
  Donor 1000155 has 105 reviews from a clock set to May 2000 (within the grid, but impossible). Found with
  `CUDA_LAUNCH_BLOCKING=1` on that user (ticket 001791) and a scan of all users' intervals and dates.
- Fix: the converter (main a57b788f) drops reviews answered before 2006-01-01 UTC. The donation folders grew since
  11-26 (284 collections, 227 donors: donors renumbered), so SM178 keeps its 218-donor snapshot and applies the same
  rule in place: `sm178_timefix.py` -> `data/donations-20261126-timefix` (+ `-answer-time`). Check: rebuilding without
  dropping equals the 11-26 build bit for bit on all 218 users; with dropping, the two donors equal a fresh fixed
  conversion bit for bit. Dropped: 106 bad reviews + 274 reviews of 74 cards whose only Learning start was in 2000;
  39,891,421 -> 39,891,041 reviews. The other 216 users are unchanged, so their 106 scored rows and labels stand.
- `sm178_score.py` / `sm178_report.py` now point at the timefix folders; the two donors' labels were rebuilt in
  `work/sm178/run/label-db` (436 entries). GPU check (ticket 001792): donor 1000029 scores (curve 0.4083, instant 0.4041,
  Tiny two-head (SM130)); donor 1000155 (1.4M reviews) is OOM at cap 0.8, which the launcher's 0.92 pass handles.

## Running

- **Relaunch 11-27 15:56:55** (WMI PID 53796): Tiny ticket 001793 started at once (116 of 218 users left: 106 not
  reached + 4 OOM users and their rows from the first pass); donor 1000029 scored in it with the same values as the
  check ticket 001792. ETA `SM178 DONE` about 17:15 to 17:45 on 11-27 (Tiny ~12 min left + Small ~25 + Big ~35, plus
  0.92 passes for the 1M+ review donors). Everything below still applies.

- Launcher `launch-sm178.sh`, started hidden via WMI at 11-27 14:49:55 (WMI PID 18956, outer bash). Steps: (1) labels, CPU, 2 processes, largest users first (peaks seen 2.2 + 1.7 GB), expected about 40 min (SM177's 20 users took 2 min 17 s for 2.2M reviews), then `work/sm178/run/labels-done`; (2) checks `gpu/smoke-ok` (present); (3) one queue ticket per size, order Tiny, Small, Big (2 models each, all 218 users), passes at cap 0.8 then 0.92 after a pass without progress -> `gpu/sm178-rows.jsonl`. Expected GPU time about 20 + 25 + 35 min (SM164's REF tickets scaled to 39.9M reviews; the 8 donors with 840k to 1.5M reviews may need the 0.92 pass or fail on memory).
- ETA: labels done about 15:30, `SM178 DONE` about 16:30 to 17:00 on 11-27 (later if other tickets queue in between).
- Queue purpose prefix: `Today's models on donor data (SM178): `, owner `sm178`.
- Log: `gpu/launch-sm178.log` (labels `gpu/labels.out`, scorer `gpu/score.out`, tickets `gpu/tickets.jsonl`, raw stdout `gpu/launch-sm178.nohup`). Done line `SM178 DONE`; stopped or failed `SM178 STOP` (cause on an earlier `STOP:` line).
- Alive (PowerShell): `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*sm178*' } | Select ProcessId, Name`. Queue: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`.
- Wait without polling (Bash, run_in_background): `until grep -qE 'SM178 (DONE|STOP)' C:/Users/jonat/workspace/wt/sm178/artifacts/small-model/sm178/gpu/launch-sm178.log; do sleep 120; done; tail -5 C:/Users/jonat/workspace/wt/sm178/artifacts/small-model/sm178/gpu/launch-sm178.log`
- Stop: `touch C:/Users/jonat/workspace/wt/sm178/artifacts/small-model/sm178/gpu/STOP` (before the next ticket); for the running ticket also `r01_queue.py cancel` on its id.
- Resume after a stop or restart (labels already done and finished users are kept; remove `gpu/STOP`; check no sm178 process is alive; a half-written label-db is fine, `process()` skips users already present):
  `$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm178/artifacts/small-model/sm178/launch-sm178.sh >> C:/Users/jonat/workspace/wt/sm178/artifacts/small-model/sm178/gpu/launch-sm178.nohup 2>&1"'; ProcessStartupInformation=$si}`
- Do not change anything under `scripts/`, `expertium/`, `rwkv/` of `wt/sm164` while this runs (it is the scorer).

## Follow-up (fresh agent, Sonnet, medium effort)

1. `git -C C:/Users/jonat/workspace/wt/sm178 fetch` and merge `origin/sm178` if it moved. Wait for `SM178 DONE` / `SM178 STOP`. On a STOP: read the `STOP:` line and the tail of `gpu/score.out` / `gpu/labels.out`; fix nothing in source; report.
2. Read-out (CPU, seconds): `cd C:/Users/jonat/workspace/wt/sm178/artifacts/small-model/sm178 && C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 sm178_report.py report` -> `sm178-result.json`, `sm178-result.md`. Exit 0 = at least 95% of the 218 donors have a finite row for all six models; exit 2 = fewer (say how many are missing and why: OOM rows in `gpu/sm178-rows.jsonl` have an `error` field; report anyway, do not rerun more than once).
3. Write `RESULT.md`: status line; the per-model table (donors vs 732 set, curve / instant; Instant-Only models instant only, their curve head is untrained); the by-source table; donors' review-weighted LogLoss with the user-bootstrap 95% CI (already in the tables); the size ranking (Small and Tiny vs the matching Big model, paired, donors vs 732 set: does Big still beat Small and Tiny by the same margins?); the benchmark-set shape table; and a short note on why donors may score worse, using the shape table plus these aggregates (never name donors): longer histories (median benchmark reviews per user, users with 1M+ reviews); a Japanese-heavy subject mix (the inventory aggregate `pages/data/donations.json`: Japanese 16.8M of the reviews, 94 donors with Japanese as their main category; read only, do not edit `pages/`); different schedulers and habits (more manual reschedules / filtered decks, unknown add-ons; the dataset has no scheduler field, so say this is a hypothesis). Note that donors and the 732 set are different users, so "donors vs 732" is not a paired comparison; only the size comparisons are paired. Plain names with IDs in parentheses, "%" never "points".
4. Commit (`git add -f artifacts/small-model/sm178`, never `work/`, `data/`, .pth, .npz, quiet.json, .claude/), identity `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; `git fetch origin && git merge origin/main` (resolve here), then `git push origin sm178`.
