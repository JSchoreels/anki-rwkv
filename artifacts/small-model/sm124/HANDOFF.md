# Handoff: the show-time two-head cool-down for Tiny (SM124)

Design: `design.md` (the two-pass cool-down of the show-time two-head cool-down for Small (SM118) applied to the Tiny two-head model (SM106)'s constant-LR epoch 29, w64-c1, both heads trained, with the show-time big model (SM91s2)'s show-time curve + grade targets for the 8,000 users, the folder SM118 exported on 11-16, reused read only). Rules: `sm124_rule.py` (pre-registered; per head x = new minus the Tiny two-head model (SM106), show-time inputs, 722 users of the 732 set; gain = CI below 0 and <= -0.05%; regression = >= +0.05% or CI above 0; go = a gain on one head and a regression on neither; early stop after pass 1 against SM106's own cool-down snapshot after 4,000 users: continue if one head is below 0 and neither is +0.05% or more). Whatever the verdict, it is held for the 11-28 release decision.

## The running job

- Launched 11:05 Tue 11-17 from branch `sm124` at 785795d0 (CPU proofs `check-cpu-sm124.json` ALL OK, dry run `gpu/launch-sm124-dryrun.log` DRYRUN OK, pre-launch GPU smoke of pass 1 passed 11:00, queue ticket 001303), hidden via WMI. Windows pids: wrapper bash 20528 (the WMI process), launcher bash 33668 (`bash .../launch-sm124.sh`), its subshells 2172 and 4612 (GPU watch).
- Main log: `artifacts/small-model/sm124/gpu/launch-sm124.log` (lines start `[sm124 MM-DD HH:MM:SS]`); nohup `gpu/launch-sm124.nohup`. Trainer `gpu/train.out`, scorer `gpu/score.out`, smokes `gpu/check.out`, `gpu/segments.jsonl`, `gpu/tickets.jsonl`. Queue owner `sm124`, purposes start with "the show-time two-head cool-down for Tiny (SM124): ". First ticket 001304 (the early-stop reference scoring).
- Folders (in `wt/sm124/artifacts/small-model/`, not committed): pass 1 `sm124-c1-e29st-dec2ep32k-t91s2-p1` (prepared before the launch), pass 2 `sm124-c1-e29st-dec2ep32k-t91s2-p2` (prepared by the launcher after pass 1).

## Expected timeline (breaks Tue 20:00 -> Wed 01:00, Wed 20:00 -> Thu 01:00)

The speed is uncertain: the Tiny two-head model (SM106)'s cool-down and the longer cool-down for Tiny (SM115) ran at ~21k reviews/s, the show-time two-head cool-down for Small (SM118) at ~28.9k (11-16). One pass = 336 min at 29k, 463 min at 21k.

| step | ends about (29k / 21k reviews/s) |
|---|---|
| reference scoring: SM106's cool-down snapshot after 4,000 users, show-time, 723 users | ~11:20 Tue |
| CPU targets + pass-1 checks, smoke of pass 1 | ~11:30 Tue |
| pass 1 (snapshot stop before 20:00 if not done, resumes 01:00) | ~17:10 / ~19:15 Tue |
| early stop: pass 1 scored, `rule mid` | ~17:25 / ~19:30 Tue |
| pass 2 (if continue; at 29k it runs ~2.4 h before the break and the rest after 01:00; at 21k it starts at 01:00) | ~04:15 / ~08:45 Wed |
| final scoring (show-time + normal, 1,500 pool for the new model) + `rule decide` | **~05:05 / ~09:40 Wed 11-18** |

About 12.5 GPU h at 29k, 16.5 GPU h at 21k (6-8 GPU h if the early stop fires). Anything that slips into a break pauses and resumes by itself.

## Log lines to watch

- Start: `[sm124-pair] recipe OK`, `[sm124-rule] selftest: ... -> OK`, then the reference scoring `sm106-dec-u4000/w8+s8@e2s-query: 723 users scored`.
- Pass-1 checks: `targets: status complete, manifest == SM118's`, `[sm124-muon] ... PASS lines`, `[sm124-pair] p1 OK:`; `smoke p1: {"smoke": ... "exit": 0 ...}`.
- Pass 1: `pass 1 ... complete: ... reviews/s ...`, `lr check: pass 1 ended at 0.29... x peak`.
- Early stop: `rule mid (after pass 1 of 2) exit 0: [sm124-rule] mid continue|mid stop`.
- Verdict: `rule decide (final) exit 0: [sm124-rule] verdict go|no go (gain [...], regression [...])`, then `done (verdict ...)`.
- Failure: any `STOP:` line. `STOP: pass N ended at lr` = a broken schedule; `STOP: the reused targets ...` / `the targets' manifest ...` = the targets changed: report, don't relaunch blindly.

## Commands

- Verdict (CPU, seconds): `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm124/artifacts/small-model/sm124/sm124_rule.py decide` (early stop: `... sm124_rule.py mid`; exit 2 = rows still missing).
- Progress: `tail -5 C:/Users/jonat/workspace/wt/sm124/artifacts/small-model/sm124/gpu/launch-sm124.log`; `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`.
- Stop: `touch C:/Users/jonat/workspace/wt/sm124/artifacts/small-model/sm124/gpu/STOP` (between tickets; a running pass snapshot-stops at the next user boundary). Remove the file before a relaunch.
- Relaunch (only if the launcher is gone and the log has no `done`; finished steps are skipped, a pass resumes from its last snapshot), PowerShell:
  `$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm124/artifacts/small-model/sm124/launch-sm124.sh >> C:/Users/jonat/workspace/wt/sm124/artifacts/small-model/sm124/gpu/launch-sm124.nohup 2>&1"'; ProcessStartupInformation=$si}`
- Never edit the running launcher or `scripts/ expertium/ rwkv/` of `wt/sm124` while it runs. Read-only inputs that must not be cleaned up meanwhile: `wt/sm106`, `wt/sm91s2`, `wt/sm118` (its pass-1 folder), `wt/sm55`, `wt/sm72s`, `wt/sm57`, `anki-rwkv/artifacts/cache/kd-replay/sm91s2-big-showtime-final8k-neighbour-e2s`.

## After the verdict

1. Branch sm124 (`cd C:/Users/jonat/workspace/wt/sm124; git fetch origin sm124; git merge FETCH_HEAD`): commit with `git add -f` the decision files (`sm124-decision.{md,json}`, `sm124-mid.{md,json}`), `gpu/*.jsonl` row files, the logs (`launch-sm124.log`, `launch-sm124.nohup`, `score.out`, `check.out`, `prepare.out`, `trim.out`, `export-status.out`, `segments.jsonl`, `tickets.jsonl`), `gpu/check-*.json`, a `gpu/train-tail.out` (last ~200 lines of train.out), and a short `RESULT.md` (as SM118's). Never the .pth / optimizer / target files, never `work/`. Identity `-c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com"`, message ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; push.
2. Main (`cd C:/Users/jonat/workspace/anki-rwkv; git fetch origin main; git merge --ff-only FETCH_HEAD`): in `pages/data/factory-curated.json` close the SM124 plan row (`state` done, `when` with the end time and verdict, `ended_at`), add a `results` row (started_at / ended_at / gpu_minutes / tickets from the queue history, absolute LogLoss curve / instant, 722 users of the 732 set, 30.13M reviews, `detail` with the verdict and the two-head table vs SM106, vs the show-time big model (SM91s2), vs SM118, vs the release (SM63), gap closed per head); if go, note it as the Tiny two-head candidate in the `family` table (held, not released). Then `python "$TEMP/bare.py"` must print 0, `.venv/Scripts/python.exe scripts/publish_factory_page.py`, `... --check` must print `SYNC 0`, commit those two files only, push.
3. Report to the user in % (never "points"): both heads vs SM106 with CIs, gap closed, reviews/s of each pass, the 8k / 2k 10-epoch ETA at the measured speed.

## Plan row to insert into `pages/data/factory-curated.json` (not inserted by the design agent: another agent was editing that file)

Insert into `plan` (next to the SM118 row) when the file is free:

```json
{
 "id": "SM124",
 "track": "small model",
 "since": "11-17T11:05",
 "kind": "gpu",
 "minutes": 900,
 "match": [
  "the show-time two-head cool-down for Tiny (SM124): "
 ],
 "what": "The show-time two-head cool-down for Tiny (SM124): the recipe of the show-time two-head cool-down for Small (SM118) (one learning-rate decay stretched over two passes of the 8,000 users) applied to the Tiny two-head model (SM106), from its constant-LR epoch 29 (the checkpoint its own cool-down started from), both heads trained, with the curve and grade targets of the show-time big model (SM91s2) on show-time inputs (the targets the show-time two-head cool-down for Small (SM118) exported, reused: no export). Early stop after pass 1 unless a head beats the Tiny two-head model (SM106)&rsquo;s own snapshot at about the same learning rate and neither is +0.05% worse; go = new &minus; the Tiny two-head model (SM106) on show-time inputs: one head better (CI below 0 and at least 0.05%) and neither head worse (+0.05% or more, or CI above 0)",
 "when": "Running since 11:05 Tue 11-17: reference scoring, pass 1 (5.6 to 7.7 h), early stop, pass 2 across the 20:00-01:00 break; verdict expected about 05:00 to 09:40 Wed 11-18",
 "decides": "Whether the longer cool-down with the big two-head teacher lifts the Tiny two-head model on both heads (the show-time big model (SM91s2) is &minus;0.422% curve / &minus;0.430% instant better than the Tiny two-head model (SM106)); a go is held for the 11-28 release decision",
 "state": "running",
 "started_at": "2026-11-17T11:05+01:00"
}
```
