# HANDOFF: smaller model (SM73) stage 1, prepared 2026-10-29 ~09:30 (NOT launched)

Design: `C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/sm73/DESIGN.md` (main). Bar (user 10-29):
<= +0.50% curve AND <= +0.40% instant vs the big model (1 leak-free epoch), 732 set (722 users).

Worktree `C:/Users/jonat/workspace/wt/sm73` (branch `sm73`, cut from `sm72s` at 85da58b9; no source changes). Reads
from wt/sm72s (checkpoint of the cool-down from epoch 20, its int8 rows, the 723-user list) and wt/sm72 (big model
(1 leak-free epoch) rows), read only.

## What runs
`artifacts/small-model/sm73/launch-sm73.sh`: the leak-free small model (SM72s, cool-down from epoch 20), checkpoint
sha 3836b883e398, scored on the 723 fit users of the 732 set with leak-free scoring (the same flags as its readout):
1. `w8+s4` (int4 card + note state, the fork's int4 format, 3,018 B per card + note, -34%): 1 ticket ~12-15 min
   (+ a short second pass if users are missing), then `sm73_rule.py decide` (CPU, seconds).
2. Only if 1 fails: `w8+s4c8n` (int4 card, int8 note, 3,530 B, -22%), same.
3. `sm73_rule.py summary` -> `sm73/sm73-formats.md`.
Breaks: quiet.json only (scope all/gpu), re-read every <= 10 min; a ticket starts only if it ends before the next
window. Expected GPU: 15 min (A1 passes) to 30 min. ETA: ~20-40 min after launch if the queue is free.

Dry run 10-29 09:19: `DRYRUN OK` (`sm73/launch-sm73-dryrun.log`). Rule self-test on the int8 rows reproduces the
readout (722 users, 0.30019 / 0.27664, +0.41% / +0.32%). CPU smoke of the int4 path with the leak-free flags on user
2388: int4 vs int8 +0.00006 / +0.00002.

## 10-29 ~10:00: int4 (w8+s4) diverged, relaunched for w8+s4c8n
Ticket 000754 (w8+s4) stopped after 37 users: `Non-finite evaluation loss` on user 3876. Not a quantizer bug (scales
are clamped to 1 when 0/NaN): the int4 note state grows geometrically on a 9,539-step note sequence (WKV output ~1e32
by step 6000, inf at 6843; float 5, int8 6.5). With int4 card + int8 note (w8+s4c8n) the user is finite. Recorded as
FAIL with `sm73_rule.py diverged` (line in `sm73-formats.jsonl`, summary row "diverged on user 3876"); the launcher
skips w8+s4 as decided and scores w8+s4c8n. The 37 partial w8+s4 rows stay in `gpu/sm73-formats-732.jsonl` (own label,
not counted for w8+s4c8n).

## Launch (PowerShell, hidden WMI process)
`$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm73/artifacts/small-model/sm73/launch-sm73.sh >> C:/Users/jonat/workspace/wt/sm73/artifacts/small-model/sm73/launch-sm73.nohup 2>&1"'; ProcessStartupInformation=$si}`

## Files
- Log `artifacts/small-model/sm73/launch-sm73.log` (stdout `launch-sm73.nohup`); ticket output `sm73/gpu/score.out`.
- Rows `sm73/gpu/sm73-formats-732.jsonl` (labels `s-w64kd-final-sc-lanes-leakfree-dec20-1ep32k-muon/w8+s4`, `.../w8+s4c8n`).
- Decisions `sm73/sm73-formats.jsonl` (one line per format: LogLoss, % vs big, paired delta vs int8 with CI, pass),
  table `sm73/sm73-formats.md`. The log line per format ends in `bar +0.50% / +0.40%: PASS|FAIL`.

## Commands
- Status: `tail -5 C:/Users/jonat/workspace/wt/sm73/artifacts/small-model/sm73/launch-sm73.log`
- Stop between tickets: `touch C:/Users/jonat/workspace/wt/sm73/artifacts/small-model/sm73/STOP`
- Resume: the launch command again (scored users and decided formats are skipped). Remove STOP first.
- Decide again (CPU): `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm73/artifacts/small-model/sm73/sm73_rule.py decide s-w64kd-final-sc-lanes-leakfree-dec20-1ep32k-muon/w8+s4`

## Follow-ups (fresh Sonnet medium agent)
- On `done`: report the `sm73-formats.md` table (bytes per card + note, % vs big on both heads, delta vs int8).
  Factory: results row for the smaller model (SM73) stage 1 (plain names, ID in parentheses, "(leak-free)" tag rules),
  plan row -> done. Commit (git add -f) `sm73/gpu/*.jsonl` (benchmark users only), `sm73/sm73-formats.*`, the log; push.
- If a format passes: the release side needs the fork's state precision set to int4 (A1: existing option) or per
  entity kind (A2: fork change), plus an INTEGRATION.md 11.1 row "as scored". Tell the coordinator; do not edit
  artifacts/models/ here.
- Stage 2 (card stream 1 layer) is NOT prepared: DESIGN.md section 4 gates it on a buy-back lever (better teacher (SM65)).
