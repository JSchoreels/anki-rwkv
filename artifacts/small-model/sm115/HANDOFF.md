# Handoff: the longer cool-down for Tiny (SM115)

Design: `design.md` (SM113's two-pass cool-down with the show-time instant-only big model (SM99)'s show-time targets, from the Tiny Instant-Only model (SM101)'s epoch 38, w64-c1). Rules: `sm115_rule.py` (pre-registered; go = new minus SM101, show-time instant, 722 users of the 732 set, 95% CI below 0 and at least 0.05%; early stop after pass 1 against SM101's own snapshot at the matching learning rate, continue if below 0). Whatever the verdict, it is held for the next release (user 11-14).

## The running job

- Launched 15:06 Sat 11-14 from branch `sm115` at 112c4082, hidden via WMI. Launcher bash pids 16260 / 1376 / 11104 (`bash .../launch-sm115.sh`), wrapper 10732.
- Main log: `artifacts/small-model/sm115/gpu/launch-sm115.log` (lines start `[sm115 MM-DD HH:MM:SS]`). Trainer `gpu/train.out`, scorer `gpu/score.out`, `gpu/segments.jsonl`, `gpu/tickets.jsonl`. Queue owner `sm115`, purposes start with "the longer cool-down for Tiny (SM115): ".
- First step at launch: scoring the early-stop reference (SM101's cool-down snapshot after 4,094 users) on show-time inputs, ~25 min, then pass 1.

## Expected timeline (breaks Sat 20:00 -> Sun 01:00, Sun 16:00 -> Mon 01:00)

| step | ends about |
|---|---|
| reference scoring, pass 1 first segment; snapshot stop before 20:00 | 19:58 Sat |
| pass 1 resumes from its snapshot | ~05:15 Sun |
| early stop: pass 1 scored, `rule mid` | ~05:30 Sun |
| pass 2 (if continue) | ~13:25 Sun |
| final scoring + `rule decide` | **~14:15 Sun** |

About 16.9 GPU h (8.3 if the early stop fires). If anything slips past Sun 16:00 it pauses and resumes Mon 01:00 by itself.

## Log lines to watch

- Early stop: `rule mid (after pass 1 of 2) exit 0: [sm115-rule] mid continue|mid stop`.
- Verdict: `rule decide ... verdict go|no go`, then `done (verdict ...)`.
- Failure: any `STOP:` line. A `STOP: pass N ended at lr` line means a broken schedule: report, don't relaunch.

## Commands

- Verdict (CPU, seconds): `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm115/artifacts/small-model/sm115/sm115_rule.py decide` (early stop: `... mid`).
- Stop: `touch .../sm115/gpu/STOP`. Relaunch (only if the process is gone and the log has no `done`): the WMI command of SM113's HANDOFF.md with `sm113` replaced by `sm115`; finished steps are skipped and a pass resumes from its last snapshot.

## After the verdict

Commit the decision files, logs and row files with `git add -f` (not checkpoints), add a factory results row and close the SM115 plan row on main (bare-ID check, publish, `--check`), report % / reviews/s / ETA.
