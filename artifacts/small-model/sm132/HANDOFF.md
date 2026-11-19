# Handoff: Again pressed as Hard (SM132)

Design: `design.md` (pre-registered 11-19 ~16:00, before any GPU ticket). Training-free: every Again review is recorded as Hard in the model inputs with probability p (seed `[132, user]`, nested in p; the grade one-hot `rating_1` -> `rating_2` of the review's update row, and the card state Anki would record: a flipped review-state Again does not lapse the card, so the relearning rows after it read review state). Targets stay the true outcomes; intervals stay real; query rows untouched. Scorer flag `--again-as-hard P` (`--again-as-hard-state keep` = grade only), code in `scripts/sm31_inputs.py` (`again_as_hard`) and `scripts/small_model_heldout.py`. Without the flag nothing changes (CPU proofs K2 / K4).

- The show-time two-head cool-down for Small (SM118) at p = 0 (reference), 5, 10, 20, 30, 50%, plus grade-only at 20%; the show-time big model (SM91s2) at 10% and 20% (report only, reference = its published rows). Published format w8+s8, show-time inputs, leak-free (neighbour), 723 users of the 732 set (722 rule users).
- Readout (`sm132_rule.py decide`): the user's bet holds for a head iff the 95% CI upper bound of 100 x (corrupted / clean - 1) is <= +2.0% at every p in {5, 10, 20}%; verdict "holds" iff both heads (curve, instant) of SM118 hold. At least 718 of 722 users at p = 0 and every p, otherwise exit 2 (pending).
- Again is 15.91% of all reviews (723 users; median user 14.59%), so p = 20% changes 3.18% of all reviews (`again-share.json`).

## The running job

- **Launched 16:01 Thu 11-19** from branch `sm132` at 195687a3, hidden via WMI. Windows PIDs: wrapper bash 5944 (the WMI process), launcher bash 26076 (`bash .../launch-sm132.sh`; 17732 and later PIDs are its subshells).
- Steps: CPU proofs again under the memory guard (passed 16:02:24, `CPU proofs ALL OK`, 30 s), then ONE queue ticket (`001380-sm132`, enqueued 16:02:25, owner `sm132`, purpose prefix `Again pressed as Hard (SM132): scoring pass 1:smoke s0 s5 s10 s20 s30 s50 s20g b10 b20`, expected 121 min): GPU smoke (2 users; mismatch = `STOP`), then one scorer process per p in that order, each with in-ticket retries (cap 0.8, then 0.92). Retry tickets only if users are still missing.
- Queue position at launch (16:03): behind the compact state for Small and Big (SM131)'s chained ticket (`001378-sm131`, running since ~15:52, expected 153 min, ends ~18:25) and the six-layer Tiny (SM130)'s Instant-Only epoch 41 (`001379-sm130`, waiting, 110 min, ends ~20:15).
- Quiet windows: quiet.json scope all / gpu only (`sm132_block.py`), re-read before every ticket. No daily break. A running ticket is not interrupted; to pause, add a quiet.json window (the next ticket waits) or touch STOP.
- Logs: main `artifacts/small-model/sm132/gpu/launch-sm132.log` (lines `[sm132 MM-DD HH:MM:SS] ...`); nohup `gpu/launch-sm132.nohup`; scorer and ticket steps `gpu/score.out` (lines `[sm132-ticket ...] step sN`, `[small-model] N/723 user ...`); CPU proofs `gpu/check.out` + `check-cpu-sm132.json`; GPU smoke `gpu/smoke-gpu-rows.jsonl` + `gpu/smoke-check.json` (marker `gpu/smoke-ok`); memory guard `gpu/memguard.log`; `gpu/tickets.jsonl`; rows `gpu/sm132-rows.jsonl`.

## Expected timeline

| step | ends about |
|---|---|
| CPU proofs, enqueue | 16:03 Thu 11-19 |
| queue wait behind the SM131 ticket and SM130 epoch 41 | ~20:15 Thu |
| GPU smoke (~3 min), Small 7 x ~11 min, Big 2 x ~15 min | ~22:15 Thu |
| `rule decide` -> `done (verdict the bet holds / does not hold (...))` | **~22:15 Thu 19 Nov** (21:30-23:30; +~1 h if a retry ticket is needed or a ticket ahead runs long) |

About 2.0 GPU h (121 min expected).

## Log lines to watch

- Done: `grep -E '\] done \(verdict' C:/Users/jonat/workspace/wt/sm132/artifacts/small-model/sm132/gpu/launch-sm132.log`
- Failure: `grep -E '\] STOP:' C:/Users/jonat/workspace/wt/sm132/artifacts/small-model/sm132/gpu/launch-sm132.log`
- Milestones: `CPU proofs ALL OK`, `enqueue: scoring pass 1:smoke ...`, `exit 0: scoring pass 1:...`, `all steps scored (723/723/.../ of 723 users)`, `rule decide exit 0: [sm132-rule] verdict: the bet holds|does not hold (...) ...`; in `score.out`: `[sm132-smoke] PASS`, `[sm132-ticket ...] step s0` ... `step b20`, `ticket done`.

## Commands

- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-sm132.sh*' } | Select ProcessId, ParentProcessId, CommandLine`
- Progress: `tail -5 C:/Users/jonat/workspace/wt/sm132/artifacts/small-model/sm132/gpu/launch-sm132.log`; `grep -a 'sm132-ticket\|small-model\] [0-9]*/' C:/Users/jonat/workspace/wt/sm132/artifacts/small-model/sm132/gpu/score.out | tail -3`; `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`
- Readout by hand (CPU, seconds): `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm132/artifacts/small-model/sm132/sm132_rule.py decide` (exit 2 = fewer than 718 users at p = 0 and every p)
- Stop: `touch C:/Users/jonat/workspace/wt/sm132/artifacts/small-model/sm132/gpu/STOP` (between tickets; a running ticket ends normally; to end a running ticket now, cancel it through `r01_queue.py`). Remove STOP before a relaunch.
- Relaunch (only if the launcher is gone and the log has neither `done (` nor a final `STOP:`; scored users, the CPU proofs (`gpu/cpu-ok`) and a passed smoke (`gpu/smoke-ok`) are skipped), PowerShell:
  `$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm132/artifacts/small-model/sm132/launch-sm132.sh >> C:/Users/jonat/workspace/wt/sm132/artifacts/small-model/sm132/gpu/launch-sm132.nohup 2>&1"'; ProcessStartupInformation=$si}`
- Never edit the running launcher or `scripts/` of `wt/sm132` while a ticket runs. Read-only inputs: `wt/sm118`, `wt/sm91s2` (models, rows), `wt/sm72s` (users file). `wt/sm130` and `wt/sm131` are never touched.
- If `STOP: GPU smoke mismatch`: read `gpu/smoke-check.json` (per user and label, GPU vs CPU max LogLoss difference, tolerance 1e-5; corruption counts must be identical; p = 0 vs published 1e-5); do not relax the tolerance without the user.
- If `STOP: the rule is pending`: `sm132_rule.py decide` prints the user count; check `score.out` for `OOM` rows; a relaunch retries them (cap 0.8, then 0.92).

## After `done` (a fresh Sonnet medium agent is enough)

1. Branch sm132 (`cd C:/Users/jonat/workspace/wt/sm132; git fetch origin; git merge origin/sm132`): `git add -f` `sm132-decision.{md,json}`, `check-cpu-sm132.json`, `gpu/sm132-rows.jsonl`, `gpu/smoke-gpu-rows.jsonl`, `gpu/smoke-check.json`, `gpu/launch-sm132.log`, `gpu/launch-sm132.nohup`, `gpu/score.out`, `gpu/check.out`, `gpu/tickets.jsonl`, `gpu/memguard.log` (if any), and a short `RESULT.md` (verdict per head with CIs at every p, the tables of `sm132-decision.md`, the Again share, the grade-only arm, Big vs Small, the corrupted-label LogLoss labelled as label noise, the timeline limitation from design.md, users covered, unparseable lines, GPU minutes from `tickets.jsonl`). Never `.pth`, `.npz`. Identity `-c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com"`, message ending with a blank line + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; fetch + merge, push.
2. Main session: close the SM132 plan row in `pages/data/factory-curated.json` (`state` done, `when` with the end time and verdict, `ended_at`) and add a results row, then the usual publish + `--check` (SYNC 0). Template (fill the `<...>`; "&minus;" for negative numbers; never a bare ID):

```json
{
 "date": "11-19",
 "kind": "gpu-score",
 "id": "SM132",
 "track": "small model",
 "name": "Again pressed as Hard (SM132): the show-time two-head cool-down for Small (SM118) scored on histories where each Again was recorded as Hard with probability p (5 to 50%; the grade and the card state Anki would record; targets true), plus the show-time big model (SM91s2) at 10% and 20%, no training",
 "eval": "Leak-free, <n> users of the 732 set (722 rule users), show-time inputs, published format; % = 100 &times; (corrupted / clean &minus; 1) of the review-weighted LogLoss on the true outcomes, paired user bootstrap 95% CI. The user&rsquo;s bet holds per head iff the CI upper bound is &le; +2.0% at p = 5, 10 and 20%",
 "fork": "Not used (robustness of the trained models to how people grade; no model change)",
 "takeaway": "<strong>The bet <holds / does not hold (...)></strong>: at p = 20% (3.2% of all reviews changed; Again is 15.9% of reviews) curve <c>% [<lo>, <hi>], instant <i>% [...]; 5%: ...; 10%: ...; 30%: ...; 50%: .... Grade only (card states kept) at 20%: .... The show-time big model (SM91s2) at 20%: curve ..., instant ... (Big minus Small ...). Against the corrupted labels (what Anki&rsquo;s own metrics would show, dominated by label noise): ... at 20%.",
 "inputs": "show-time",
 "started_at": "2026-11-19T16:01+01:00",
 "ended_at": "<from the done line>",
 "gpu_minutes": <sum of tickets.jsonl minutes>,
 "tickets": <count of tickets.jsonl lines>,
 "detail": {
  "verdict": "<verdict>; branch sm132 <commit> (sm132-decision.md, RESULT.md)",
  "table": {
   "head": ["p", "Share of all reviews changed", "Curve % vs clean", "95% CI", "Instant % vs clean", "95% CI", "Within the bet", "Corrupted labels: curve / instant % (label noise)"],
   "rows": [
    ["5%", "0.80%", "...", "...", "...", "...", "...", "..."],
    ["10%", "1.59%", "...", "...", "...", "...", "...", "..."],
    ["20%", "3.18%", "...", "...", "...", "...", "...", "..."],
    ["30%", "4.77%", "...", "...", "...", "...", "(not in the bet)", "..."],
    ["50%", "7.95%", "...", "...", "...", "...", "(not in the bet)", "..."],
    ["20%, grade only (report only)", "3.18%", "...", "...", "...", "...", "", ""],
    ["the show-time big model (SM91s2), 10% (report only)", "1.59%", "...", "...", "...", "...", "...", "..."],
    ["the show-time big model (SM91s2), 20% (report only)", "3.18%", "...", "...", "...", "...", "...", "..."]
   ]
  }
 }
}
```

3. Report to the user in % (never "points"), every model as "plain name (ID)": whether the bet holds per head (CI upper bounds at 5, 10, 20%), the curve over p, the Again share (15.9% of reviews, so p = 20% = 3.2% of reviews changed), the grade-only arm, Big vs Small, the corrupted-label LogLoss as Anki's view (dominated by label noise), and the limitation that the real timeline (short relearning intervals after a hidden lapse) stays in the history.
