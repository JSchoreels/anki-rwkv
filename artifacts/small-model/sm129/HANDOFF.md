# Handoff: compact state for Tiny (SM129)

Design: `design.md` (pre-registered 11-18 before any GPU ticket). Training-free: the show-time two-head cool-down for Tiny (SM124) and the longer cool-down for Tiny (SM115) scored with the compact int state of the compact int state test (SM47) (`w8+s4c8nhx8`, 803 B per card, 1,315 B per note), the card-only arm of the second integer-state test (SM49) (`w8+s4hgx4c8hx8n`, 619 B per card) and the fork default (`w8+s4c8n`, report only); then the Small references at `w8+s4c8n` (report only). Rule (`sm129_rule.py decide`): per arm and model, every head's 95% CI upper bound of 100 x (Tiny arm / Small - 1) <= +1.00% (two-head vs the show-time two-head cool-down for Small (SM118), Instant-Only vs the longer Small Instant-Only model (SM113)); go = both models pass.

## The running job

- Launched 22:57 Wed 11-18 from branch `sm129` at 52c24030, hidden via WMI. Windows PIDs: wrapper bash 34248 (the WMI process), launcher bash 24836 (`bash .../launch-sm129.sh`). It waits for the break to end (01:00 Thu 11-19) by itself.
- Main log `artifacts/small-model/sm129/gpu/launch-sm129.log` (lines `[sm129 MM-DD HH:MM:SS] ...`); nohup `gpu/launch-sm129.nohup`; scorer `gpu/score.out`; CPU proofs `gpu/check.out` + `check-cpu-sm129.json`; memory guard `gpu/memguard.log` (kill marker `gpu/memguard-kill`); `gpu/tickets.jsonl`; rows `gpu/sm129-rows.jsonl`. Queue owner `sm129`, purposes start with `Compact state for Tiny (SM129): `.
- The six-layer Tiny (SM130) launcher waits for this log's final line before its first GPU ticket.

## Expected timeline

| step | ends about |
|---|---|
| CPU proofs (bytes, parity with SM124's tree, GPU path on CPU, rule selftest) | 01:08 Thu 11-19 |
| SM124 in 3 formats, then SM115 in 3 formats (one ticket each, 723 users) | ~01:45, ~02:20 Thu |
| SM118 and SM113 at w8+s4c8n (report only) | ~02:45 Thu |
| `rule decide` -> `done (verdict compact ..., card-only ...)` | **~02:45 Thu 19 Nov** (02:15-03:15) |

About 1.2-1.6 GPU h.

## Log lines to watch

- Done: `grep -E '\] done \(verdict' C:/Users/jonat/workspace/wt/sm129/artifacts/small-model/sm129/gpu/launch-sm129.log`
- Failure: `grep -E '\] STOP:' C:/Users/jonat/workspace/wt/sm129/artifacts/small-model/sm129/gpu/launch-sm129.log`
- Milestones: `CPU proofs ALL OK`, `sm124-c1-e29st-dec2ep32k-t91s2: 723 users scored for each of ...`, `rule decide exit 0: [sm129-rule] verdict compact go|no go, card-only ...`.

## Commands

- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-sm129.sh*' } | Select ProcessId, CommandLine`
- Progress: `tail -5 C:/Users/jonat/workspace/wt/sm129/artifacts/small-model/sm129/gpu/launch-sm129.log`; `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`
- Verdict by hand (CPU, seconds): `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm129/artifacts/small-model/sm129/sm129_rule.py decide` (exit 2 = rows missing)
- Stop: `touch C:/Users/jonat/workspace/wt/sm129/artifacts/small-model/sm129/gpu/STOP` (between tickets; a running ticket ends normally). Remove it before a relaunch.
- Relaunch (only if the launcher is gone and the log has neither `done (` nor a final `STOP:`; scored users are skipped), PowerShell:
  `$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm129/artifacts/small-model/sm129/launch-sm129.sh >> C:/Users/jonat/workspace/wt/sm129/artifacts/small-model/sm129/gpu/launch-sm129.nohup 2>&1"'; ProcessStartupInformation=$si}`
- Never edit the running launcher. Read-only inputs: `wt/sm124`, `wt/sm115`, `wt/sm118`, `wt/sm113` (final models, rows), `wt/sm72s` (users file).

## After `done` (a fresh Sonnet medium agent is enough)

1. Branch sm129 (`cd C:/Users/jonat/workspace/wt/sm129; git fetch origin; git merge origin/sm129`): `git add -f` `sm129-decision.{md,json}`, `check-cpu-sm129.json`, `gpu/sm129-rows.jsonl`, `gpu/launch-sm129.log`, `gpu/launch-sm129.nohup`, `gpu/score.out`, `gpu/check.out`, `gpu/tickets.jsonl`, `gpu/memguard.log` (if any), and a short `RESULT.md` (verdict per arm and model, the tables of `sm129-decision.md`, bytes per card / note, GPU minutes from `tickets.jsonl`). Identity `-c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com"`, message ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; fetch + merge, push.
2. Main session: close the SM129 plan row in `pages/data/factory-curated.json` (`state` done, `when` with the end time and verdict, `ended_at`) and add a results row, then the usual publish + `--check` (SYNC 0). Template (fill the `<...>`; "&minus;" for negative numbers; never a bare ID):

```json
{
 "date": "11-19",
 "kind": "gpu-score",
 "id": "SM129",
 "track": "small model",
 "name": "Compact state for Tiny (SM129): the current Tiny candidates, the show-time two-head cool-down for Tiny (SM124) and the longer cool-down for Tiny (SM115), scored with the compact integer state of the compact int state test (SM47) (int4 cards + int8 notes, float16 row scales, int8 token-shift vectors) and with the card-only format of the second integer-state test (SM49), no training",
 "eval": "Leak-free, 722 users of the 732 set, show-time inputs, w8 weights; % = 100 &times; (Tiny / Small &minus; 1) of the review-weighted LogLoss, paired user bootstrap 95% CI. Go per format = every head&rsquo;s CI upper bound &le; +1.00% vs the show-time two-head cool-down for Small (SM118) (two-head) and vs the longer Small Instant-Only model (SM113) (Instant-Only), both models",
 "fork": "Not used: the fork stores int4 cards + int8 notes today (1,055 / 1,567 B for Tiny); the compact format needs a fork state-format change (not part of this run)",
 "takeaway": "<strong><verdict compact></strong> for the compact state: two-head <c>% [<lo>, <hi>] curve and <i>% [...] instant vs Small, Instant-Only <io>% [...]; it costs <x>% / <y>% / <z>% vs the unmodified Tiny. State 803 B per card and 1,315 B per note (fork default 1,055 / 1,567 B: &minus;24% per card). Card-only format (619 B per card): <verdict card-only> (...).",
 "inputs": "show-time",
 "started_at": "2026-11-19T01:00+01:00",
 "ended_at": "<from the done line>",
 "gpu_minutes": <sum of tickets.jsonl minutes>,
 "tickets": <count of tickets.jsonl lines>,
 "detail": {
  "verdict": "<verdicts>; branch sm129 <commit> (sm129-decision.md, RESULT.md)",
  "table": {
   "head": ["Format (Tiny model)", "Card / note bytes", "Curve % vs Small", "Curve 95% CI", "Instant % vs Small", "Instant 95% CI", "Cost vs Tiny published (curve / instant)"],
   "rows": [
    ["published w8+s8 (the show-time two-head cool-down for Tiny (SM124))", "1,567 / 1,567", "+0.118", "[+0.057, +0.174]", "+0.183", "[+0.164, +0.204]", "0 / 0"],
    ["fork default w8+s4c8n (SM124)", "1,055 / 1,567", "...", "...", "...", "...", "..."],
    ["compact w8+s4c8nhx8 (SM124)", "803 / 1,315", "...", "...", "...", "...", "..."],
    ["card-only w8+s4hgx4c8hx8n (SM124)", "619 / 1,315", "...", "...", "...", "...", "..."],
    ["published w8+s8 (the longer cool-down for Tiny (SM115), instant only)", "1,567 / 1,567", "", "", "+0.186", "[+0.166, +0.208]", "0"],
    ["compact w8+s4c8nhx8 (SM115)", "803 / 1,315", "", "", "...", "...", "..."],
    ["card-only w8+s4hgx4c8hx8n (SM115)", "619 / 1,315", "", "", "...", "...", "..."]
   ]
  }
 }
}
```

3. Report to the user in % (never "points"), every model as "plain name (ID)": the verdict per format and model with CIs, the bytes, and the cost vs the unmodified Tiny; note that the six-layer Tiny (SM130) will be scored in the compact format if the compact arm went go.
