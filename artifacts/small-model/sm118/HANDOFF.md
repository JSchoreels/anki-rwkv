# Handoff: the show-time two-head cool-down for Small (SM118)

Design: `design.md` (the two-pass cool-down of the longer cool-down for Tiny (SM115) from the show-time retrain of the
release (SM92)'s epoch 26, w64, both heads trained, with the show-time big model (SM91s2)'s show-time curve + grade
targets for the 8,000 users, exported first by this run). Rules: `sm118_rule.py` (pre-registered; per head x = new
minus SM92, show-time inputs, 722 users of the 732 set; gain = CI below 0 and <= -0.05%; regression = >= +0.05% or CI
above 0; go = a gain on one head and a regression on neither; early stop after pass 1 against SM92's own snapshot after
4,056 users: continue if one head is below 0 and neither is +0.05% or more). Whatever the verdict, it is held for the
next release (user 11-14). Next in the two-week plan after this: the same recipe for the Tiny two-head model (SM106).

## The running job

- Launched 06:40 Mon 11-16 from branch `sm118` at 74c5771b (CPU proofs ALL OK at c82b4203, dry run OK, pre-launch GPU
  smoke of pass 1 with the 1,991-user targets of the same teacher passed at 06:17, queue ticket 001255), hidden via WMI.
  Windows pids: wrapper bash 32560, launcher bash 1868 (`bash .../launch-sm118.sh`), its GPU-watch subshell 7348.
- Main log: `artifacts/small-model/sm118/gpu/launch-sm118.log` (lines start `[sm118 MM-DD HH:MM:SS]`); nohup
  `gpu/launch-sm118.nohup`. Export `gpu/export.out`, trainer `gpu/train.out`, scorer `gpu/score.out`,
  `gpu/segments.jsonl`, `gpu/tickets.jsonl`. Queue owner `sm118`, purposes start with "the show-time two-head cool-down
  for Small (SM118): ". First ticket 001256 (the reference scoring).
- New targets folder: `C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/sm91s2-big-showtime-final8k-neighbour-e2s`.
- The factory publish loop had died 22:11 Sun; relaunched 06:42 Mon (pid in `anki-rwkv/artifacts/factory-page/loop.pid`).

## Expected timeline (breaks Mon 20:00 -> Tue 01:00, Tue 20:00 -> Wed 01:00)

| step | ends about |
|---|---|
| reference scoring: SM92's cool-down snapshot after 4,056 users, show-time, 723 users | 07:05 Mon |
| teacher targets, 8,000 users (~4.2 h at ~39k reviews/s) | ~11:35 Mon |
| CPU checks of the targets, pass 1 prepare + pair, smoke | ~11:55 Mon |
| pass 1 (~8 h; snapshot stop before 20:00 if not done, resumes 01:00) | ~19:55 Mon (or ~01:30 Tue) |
| early stop: pass 1 scored, `rule mid` | ~01:40 Tue |
| pass 2 (if continue) | ~09:50 Tue |
| final scoring (show-time + normal, 1,500 pool for the new model and SM92) + `rule decide` | **~12:30 Tue 11-17** |

About 23.5 GPU h (about 13 if the early stop fires). If anything slips into a break it pauses and resumes by itself.

## Log lines to watch

- Export: `export ticket: [sm11] pass finished: 8000 users, ...` then `export complete:` and `export verify OK:`; the
  targets check line `[sm118-check] targets: {... "ok": true}` (in launch-sm118.log), then `[sm118-pair] p1 OK:`.
- Pass 1: `pass 1 ... complete: ... reviews/s ...`, `lr check: pass 1 ended at 0.29... x peak`.
- Early stop: `rule mid (after pass 1 of 2) exit 0: [sm118-rule] mid continue|mid stop`.
- Verdict: `rule decide (final) exit 0: [sm118-rule] verdict go|no go (gain [...], regression [...])`, then `done (verdict ...)`.
- Failure: any `STOP:` line. `STOP: pass N ended at lr` = a broken schedule; `STOP: the new targets ...` = a bad export:
  report, don't relaunch blindly.

## Commands

- Verdict (CPU, seconds): `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm118/artifacts/small-model/sm118/sm118_rule.py decide`
  (early stop: `... sm118_rule.py mid`; exit 2 = rows still missing).
- Progress: `tail -5 C:/Users/jonat/workspace/wt/sm118/artifacts/small-model/sm118/gpu/launch-sm118.log`;
  `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`.
- Stop: `touch C:/Users/jonat/workspace/wt/sm118/artifacts/small-model/sm118/gpu/STOP` (between tickets; a running pass
  snapshot-stops at the next user boundary, a running export stops at its time limit). Remove the file before a relaunch.
- Relaunch (only if the launcher is gone and the log has no `done`; finished steps are skipped, the export skips
  finished users, a pass resumes from its last snapshot), PowerShell:
  `$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm118/artifacts/small-model/sm118/launch-sm118.sh >> C:/Users/jonat/workspace/wt/sm118/artifacts/small-model/sm118/gpu/launch-sm118.nohup 2>&1"'; ProcessStartupInformation=$si}`
- Never edit the running launcher or `scripts/ expertium/ rwkv/` of `wt/sm118` while it runs.

## After the verdict

1. Branch sm118 (`cd C:/Users/jonat/workspace/wt/sm118; git fetch origin sm118; git merge FETCH_HEAD`): commit with
   `git add -f` the decision files (`sm118-decision.{md,json}`, `sm118-mid.{md,json}`), `gpu/*.jsonl` row files, the logs
   (`launch-sm118.log`, `launch-sm118.nohup`, `export.out`, `score.out`, `check.out`, `prepare.out`, `trim.out`,
   `segments.jsonl`, `tickets.jsonl`), `gpu/check-*.json`, `gpu/replay-*.json`, a `gpu/train-tail.out` (last ~200 lines
   of train.out). Never the .pth / optimizer / target files. Identity `-c user.name="Jonathan Schoreels" -c
   user.email="jonathan.schoreels@gmail.com"`, message ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; push.
2. Main (`cd C:/Users/jonat/workspace/anki-rwkv; git fetch origin main; git merge --ff-only FETCH_HEAD`): in
   `pages/data/factory-curated.json` close the SM118 plan row (`state` done, `when` with the end time and verdict,
   `ended_at`), add a `results` row (started_at / ended_at / gpu_minutes / tickets from the queue history, absolute
   LogLoss curve / instant, 722 users of the 732 set, 30.13M reviews, `detail` with the verdict and the two-head table,
   vs SM92, vs the show-time big model (SM91s2), vs the release (SM63), gap closed per head); if go, note it as the
   Small two-head candidate in the `family` table (held, not released). Then `python "$TEMP/bare.py"` must print 0,
   `.venv/Scripts/python.exe scripts/publish_factory_page.py`, `... --check` must print `SYNC 0`, commit those two
   files only, push.
3. Report to the user in % (never "points"): both heads vs SM92 with CIs, gap closed, reviews/s of each pass, the
   8k / 2k 10-epoch ETA at the measured speed; then start the next plan step (the same recipe for the Tiny two-head
   model (SM106)).
