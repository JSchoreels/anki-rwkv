# Handoff: compact state for Small and Big (SM131)

Design: `design.md` (pre-registered 11-19 ~14:00 before any GPU ticket). Training-free: the compact integer state of the compact int state test (SM47) (`w8+s4c8nhx8`) against the fork default (`w8+s4c8n`, int4 cards + int8 notes) for the show-time two-head cool-down for Small (SM118), the longer Small Instant-Only model (SM113), the show-time big model (SM91s2) and the show-time instant-only big model (SM99); card-only (`w8+s4hgx4c8hx8n`) as a report-only arm. Rule (`sm131_rule.py decide`): compact is free for a model iff every head's 95% CI upper bound of 100 x (compact / fork default - 1) < +0.05%; per size free iff both models are free; at least 718 of the 722 rule users in both formats, otherwise exit 2 (pending).

Bytes (fork layout, `sm131-bytes.json`): Small 1,963 / 1,567 B (card / note, fork default) -> 1,521 / 1,315 B compact (card + note -19.7%); Big 9,271 / 10,283 B -> 7,747 / 9,267 B (card + note -13.0%).

## The running job

- **Relaunched 15:21 Thu 11-19** from branch `sm131` at 65b073a9, hidden via WMI. Windows PIDs: wrapper bash 23872 (the WMI process), launcher bash 9592 (`bash .../launch-sm131.sh`; its children are its subshells). CPU proofs passed again at 15:22:13 (`CPU proofs ALL OK`, new check code). Queue ticket `001378-sm131` (owner `sm131`), waiting behind the six-layer Tiny (SM130)'s Instant-Only epoch 40 (`001376-sm130`, started ~14:55, ~110 min).
- First launch (14:04, 078d3f37): its ticket (14:43-14:46) passed the GPU smoke, scored 37 users in the Big step (kept: the fix changes no finite number) and stopped at user 3876 with `Non-finite evaluation loss`. Cause (design.md, amendment 11-19): the fork default `w8+s4c8n` (int4 card rows, float32 scales, as the fork's precision.rs) overflows float32 for 3876 **on CPU and GPU alike**, both Big models; a card-module row of 9,465 steps (4,733 updates) grows the int4 row scale from 0.04 to 4e33. Int8 rows stay small; the float16-scale formats (compact, card-only) clamp and stay finite. Not a GPU or kernel issue (user 2450, first suspected, had finished and matches the CPU within 7e-6).
- Fix: scorer `--nonfinite-rows` (a `nonfinite` row, no `benchmark`, done and not retried); ticket `missing` and launcher `min_users` count it as scored; the rule (finite rows in both formats, floor 718 of 722) lists `nonfinite_rows`, so 3876 is one of the four allowed missing Big users. Smoke gate re-runs (`smoke-ok` removed) with users 2450 and 3876 added (3876: fork default non-finite on the GPU too, w8+s8 vs published 1e-5, compact / card-only finite with the difference reported only).
- ONE chained ticket (`sm131_ticket.sh 0.8 smoke big small`): GPU smoke (2450, 3876 new; 2388, 4676 already in `gpu/smoke-gpu-rows.jsonl`), then Big (686 users left, both models x 3 formats), then Small (both models x 2 formats); in-ticket retries for missing users at cap 0.8, then 0.92. Further tickets only if users are still missing (each waits behind one SM130 ticket).
- Quiet windows: quiet.json scope all / gpu only (`sm131_block.py`), re-read before every ticket. No daily or Sunday break. A running ticket is not interrupted; to pause, add a quiet.json window (the next ticket waits) or touch STOP.
- Logs: main `artifacts/small-model/sm131/gpu/launch-sm131.log` (lines `[sm131 MM-DD HH:MM:SS] ...`); nohup `gpu/launch-sm131.nohup`; scorer and ticket steps `gpu/score.out` (lines `[sm131-ticket ...] step ...`, `[small-model] N/723 user ...`, `[small-model] user U LABEL: non-finite evaluation, row 'nonfinite' written`); CPU proofs `gpu/check.out` + `check-cpu-sm131.json`; GPU smoke `gpu/smoke-gpu-rows.jsonl` + `gpu/smoke-check.json` (marker `gpu/smoke-ok`); memory guard `gpu/memguard.log`; `gpu/tickets.jsonl`; rows `gpu/sm131-rows.jsonl`.
- Diagnosis files (scratchpad, not committed): `C:/Users/jonat/AppData/Local/Temp/claude/eefeeb6d2df69e2b/41a2947b-8da0-4ee6-a033-bc96eb42a697/scratchpad/sm131diag3876/` (`cpu_locate.py`, `replay.py`, CPU rows of 3876).

## Expected timeline (no pause tonight)

| step | ends about |
|---|---|
| queue wait behind SM130's epoch 40 | ~16:45 Thu 11-19 |
| GPU smoke (~5 min), Big (686 users, ~75-80 min), Small (~25-30 min), in-ticket retries | ~18:45 Thu |
| `rule decide` -> `done (verdict Small ..., Big ...)` | **~18:45 Thu 19 Nov** (18:15-20:30; +~2 h if a retry ticket is needed) |

About 1.9 GPU h. If more than four Big users miss (3876 non-finite plus the four SM128 out-of-memory users 481, 2267, 3735, 4852), `decide` exits 2 (`STOP: the rule is pending`) and the main session decides.

## Log lines to watch

- Done: `grep -E '\] done \(verdict' C:/Users/jonat/workspace/wt/sm131/artifacts/small-model/sm131/gpu/launch-sm131.log`
- Failure: `grep -E '\] STOP:' C:/Users/jonat/workspace/wt/sm131/artifacts/small-model/sm131/gpu/launch-sm131.log`
- Milestones: `exit 0: scoring pass 1:smoke big small`, `all labels scored: Big 723, Small 723 of 723 users`, `rule decide exit 0: [sm131-rule] verdict Small free|not free (...), Big ...`; in `score.out`: `[sm131-smoke] PASS`, `[sm131-ticket ...] step big`, `step small`, `ticket done`.

## Commands

- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-sm131.sh*' } | Select ProcessId, ParentProcessId, CommandLine`
- Progress: `tail -5 C:/Users/jonat/workspace/wt/sm131/artifacts/small-model/sm131/gpu/launch-sm131.log`; `grep -a 'small-model\] [0-9]*/' C:/Users/jonat/workspace/wt/sm131/artifacts/small-model/sm131/gpu/score.out | tail -2`; `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`
- Verdict by hand (CPU, seconds): `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm131/artifacts/small-model/sm131/sm131_rule.py decide` (exit 2 = fewer than 718 users in both formats for some model)
- Stop: `touch C:/Users/jonat/workspace/wt/sm131/artifacts/small-model/sm131/gpu/STOP` (between tickets; a running ticket ends normally; to end a running ticket now, cancel it through `r01_queue.py`). Remove STOP before a relaunch.
- Relaunch (only if the launcher is gone and the log has neither `done (` nor a final `STOP:`; scored users and a passed smoke are skipped), PowerShell:
  `$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm131/artifacts/small-model/sm131/launch-sm131.sh >> C:/Users/jonat/workspace/wt/sm131/artifacts/small-model/sm131/gpu/launch-sm131.nohup 2>&1"'; ProcessStartupInformation=$si}`
- Never edit the running launcher or `scripts/` of `wt/sm131` while a ticket runs. Read-only inputs: `wt/sm91s2`, `wt/sm99`, `wt/sm118`, `wt/sm113` (models, rows), `wt/sm72s` (users file). `wt/sm130` is never touched.
- If `STOP: GPU smoke mismatch`: read `gpu/smoke-check.json` (per user and format, GPU vs CPU max LogLoss difference; tolerance 1e-4, w8+s8 1e-5 for 2388 / 4676, vs published 1e-5 for all; 3876's compact / card-only are report only, its fork default must be `nonfinite` on both); do not relax the tolerance without the user.
- If `STOP: the rule is pending`: `sm131_rule.py decide` prints which model has fewer than 718 users; check `score.out` for `OOM` rows; a relaunch retries them (cap 0.8, then 0.92).

## After `done` (a fresh Sonnet medium agent is enough)

1. Branch sm131 (`cd C:/Users/jonat/workspace/wt/sm131; git fetch origin; git merge origin/sm131`): `git add -f` `sm131-decision.{md,json}`, `check-cpu-sm131.json`, `gpu/sm131-rows.jsonl`, `gpu/smoke-gpu-rows.jsonl`, `gpu/smoke-check.json`, `gpu/launch-sm131.log`, `gpu/launch-sm131.nohup`, `gpu/score.out`, `gpu/check.out`, `gpu/tickets.jsonl`, `gpu/memguard.log` (if any), and a short `RESULT.md` (verdict per size and model with CIs, the tables of `sm131-decision.md`, bytes per card / note and saving, users covered, unparseable lines, GPU minutes from `tickets.jsonl`). Never `.pth`, `.npz`. Identity `-c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com"`, message ending with a blank line + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; fetch + merge, push.
2. Main session: close the SM131 plan row in `pages/data/factory-curated.json` (`state` done, `when` with the end time and verdict, `ended_at`) and add a results row, then the usual publish + `--check` (SYNC 0). Template (fill the `<...>`; "&minus;" for negative numbers; never a bare ID):

```json
{
 "date": "11-19",
 "kind": "gpu-score",
 "id": "SM131",
 "track": "small model",
 "name": "Compact state for Small and Big (SM131): the show-time two-head cool-down for Small (SM118), the longer Small Instant-Only model (SM113), the show-time big model (SM91s2) and the show-time instant-only big model (SM99) scored with the compact integer state of the compact int state test (SM47) (int4 cards + int8 notes, float16 row scales, int8 token-shift vectors) against the fork&rsquo;s default stored state (int4 cards + int8 notes), no training",
 "eval": "Leak-free, <n> users of the 732 set (722 rule users), show-time inputs, w8 weights; % = 100 &times; (compact / fork default &minus; 1) of the review-weighted LogLoss, paired user bootstrap 95% CI. Free per model = every head&rsquo;s CI upper bound &lt; +0.05%; free per size = both models free",
 "fork": "Not used: the fork stores int4 cards + int8 notes today (Small 1,963 / 1,567 B, Big 9,271 / 10,283 B per card / note); the compact format needs a fork state-format change (not part of this run)",
 "takeaway": "<strong>Small: <verdict>; Big: <verdict></strong>. Compact vs fork default: Small two-head <c>% [<lo>, <hi>] curve, <i>% [...] instant, Small Instant-Only <io>% [...]; Big two-head <c>% [...] curve, <i>% [...] instant, Big Instant-Only <io>% [...]. State per card / note: Small 1,521 / 1,315 B (card + note &minus;19.7%), Big 7,747 / 9,267 B (&minus;13.0%). Card-only format (report only): <...>.",
 "inputs": "show-time",
 "started_at": "2026-11-19T14:04+01:00",
 "ended_at": "<from the done line>",
 "gpu_minutes": <sum of tickets.jsonl minutes>,
 "tickets": <count of tickets.jsonl lines>,
 "detail": {
  "verdict": "<verdicts>; branch sm131 <commit> (sm131-decision.md, RESULT.md)",
  "table": {
   "head": ["Model", "Head", "Card / note bytes (compact vs fork default)", "Compact vs fork default %", "95% CI", "Free", "Compact vs published w8+s8 %", "Card-only vs fork default % (report only)"],
   "rows": [
    ["the show-time two-head cool-down for Small (SM118)", "curve", "1,521 / 1,315 vs 1,963 / 1,567", "...", "...", "...", "...", "..."],
    ["the show-time two-head cool-down for Small (SM118)", "instant", "", "...", "...", "", "...", "..."],
    ["the longer Small Instant-Only model (SM113)", "instant", "1,521 / 1,315 vs 1,963 / 1,567", "...", "...", "...", "...", "..."],
    ["the show-time big model (SM91s2)", "curve", "7,747 / 9,267 vs 9,271 / 10,283", "...", "...", "...", "...", "..."],
    ["the show-time big model (SM91s2)", "instant", "", "...", "...", "", "...", "..."],
    ["the show-time instant-only big model (SM99)", "instant", "7,747 / 9,267 vs 9,271 / 10,283", "...", "...", "...", "...", "..."]
   ]
  }
 }
}
```

3. Report to the user in % (never "points"), every model as "plain name (ID)": the verdict per size and model with CIs, the bytes and saving per card / note, the users covered (and any of 481, 2267, 3735, 4852 missing), the non-finite fork default rows (`nonfinite_rows`, user 3876 at least: the int4 card state overflow on Big, a finding for the fork's default card state), and the card-only numbers as report only. Note: Small's fork-default shift vectors are float in the scorer (expertium path), Big's are float16 (fork path); each comparison is internally consistent.
