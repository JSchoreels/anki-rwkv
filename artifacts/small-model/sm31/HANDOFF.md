# Duration-leak scoring (SM31) and robustness to other apps (SM29): handoff

Branch `sm31` (from `sm81`, the newest scorer: `--start-of-day` + leak-free options), worktree `C:/Users/jonat/workspace/wt/sm31`. Approved by the user Sunday 2026-11-01 ~10:45. Grade heads only (curve, instant); the answer-time head is not evaluated. No retraining: the released small model (`s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon`) and the big model (1 leak-free epoch) (`big-leakfree-1ep`) as they are, the 732 set, leak-free (deleted cards: neighbour), w8+s8.

## What the variants are (`scripts/sm31_inputs.py`, scorer `--inputs V`, rows labelled `LABEL@V`)

- Normal: `elapsed_seconds` runs from the card's previous answer to this answer, so it holds the review's own answer time.
- `e2s-query`, show-time intervals (the deployment-realistic read-out): everything asked before the answer uses the interval from the card's previous answer to the moment the card is shown, `floor(max(elapsed_seconds - duration/1000, 0))` (first-review sentinel -1 kept): the instant query row's interval and its sin / cos, the cumulative interval (the card's earlier answer-to-answer intervals + the show-time one) and its sin / cos, and the curve query time (`label_elapsed_seconds`). Update rows (after the answer, when the full interval and the time are known) keep their inputs.
- `e2s-all`, the outside-research definition (`expertium/intervals.end_to_start_published`): the same subtraction on every row (update rows and the cumulative interval too).
- `passfail`: Hard and Easy get the Good rating one-hot in the inputs (labels untouched). `notime`: `scaled_duration` of a missing time (duration 0) on every row. `passfail-notime`: both.

## CPU proof (commit cd82f372, all on CPU)

- `check-cpu-sm31-inputs.json` (4 users, ok): `normal` through the new module is bit-identical to the scorer's path (every sample tensor); each variant moves only the expected columns: `e2s-query` columns 2-7 on query rows only + the curve query time of update rows (equal to `e2s-all`'s), `e2s-all` columns 2-7 on all rows, `passfail` the rating one-hot on exactly the Hard / Easy update rows, `notime` column 8 on update rows; ids, permutations, review_ths, every other label identical.
- `cpu-proof/compare.json` (ok): default scorer rows of the new commit identical to the previous scorer (wt/sm81 efd50462) on 2 users x 2 models (every field except timing / provenance); variant rows score the same test reviews.

## What runs

Launcher `artifacts/small-model/sm31/launch-sm31.sh` started 2026-11-01 10:35 (WMI hidden, pid 76352), log `artifacts/small-model/sm31/gpu/launch-sm31.log`, scorer output `gpu/score.out`, rows `gpu/sm31-inputs.jsonl`. It respects quiet.json and the daily break itself (Sunday from 16:00; daily 20:00-01:00); a ticket starts only when it fits before the next window.

Ticket purposes in order (each one scoring pass, 723 users at the 0.8 cap; a "pass 2" ticket only if a time limit left users out):
1. duration-leak scoring (SM31): show-time intervals (e2s-query): scoring the released small model on the 723 users of the 732 set (pass 1) (~20 min)
2. duration-leak scoring (SM31): show-time intervals (e2s-query): scoring the big model (1 leak-free epoch) on the 723 users of the 732 set (pass 1) (~25 min)
3. duration-leak scoring (SM31): end-to-start intervals on every row (e2s-all): small, then 4. big
5. robustness to other apps (SM29): pass / fail ratings only (passfail): small, then 6. big
7. robustness to other apps (SM29): no answer times (notime): small, then 8. big
9. robustness to other apps (SM29): pass / fail ratings and no answer times (passfail-notime): small, then 10. big
11. CPU table on the 722-user rule pool -> `sm31-result.{md,json}`, `../sm29/sm29-result.{md,json}`
12. the same 5 variants x 2 models on the 3 long users of the 732 set (cap 0.92; small ~20 min, big ~10 min each), then the table again (adds the 732+long pool).

Expected end: the first ticket queued behind curve consistency on rapid Good pairs (ST01a) and grade-order sanity (SM33) (~40 min); the 10 main tickets need ~3.7 h of GPU, so most land before the Sunday 16:00 break and the rest after 01:00 Monday; the long users ~2.5 h more, final table around 03:00-04:00 Monday.

## Follow-up (fresh agent, Sonnet medium is enough)

1. `git -C C:/Users/jonat/workspace/wt/sm31 fetch` + `merge --ff-only`; check the log ends with `done` (`tail -3 artifacts/small-model/sm31/gpu/launch-sm31.log`). If the launcher died (app restart) and the log is silent > 10 min outside a ticket, relaunch (it resumes; remove `gpu/STOP` first if present):
   ```powershell
   $si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm31/artifacts/small-model/sm31/launch-sm31.sh >> C:/Users/jonat/workspace/wt/sm31/artifacts/small-model/sm31/gpu/launch-sm31.nohup 2>&1"'; ProcessStartupInformation=$si}
   ```
2. Table (CPU, seconds; the launcher runs it too): `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 scripts/sm31_table.py --rows artifacts/small-model/sm31/gpu/sm31-inputs.jsonl` (exit 2 = some users missing, tables marked partial).
3. Commit with `git add -f` the rows `artifacts/small-model/sm31/gpu/sm31-inputs.jsonl`, `gpu/launch-sm31.log`, `sm31-result.md/.json`, `artifacts/small-model/sm29/sm29-result.md/.json`; push.
4. Report: each model normal vs show-time intervals (and e2s-all), curve and instant, the small-vs-big gap under both; for robustness to other apps (SM29) each model under each variant vs normal. Points = % of the big model (1 leak-free epoch) with normal inputs.

Stop between tickets: `touch C:/Users/jonat/workspace/wt/sm31/artifacts/small-model/sm31/gpu/STOP`. Do not edit `scripts/`, `expertium/` or `rwkv/` in this worktree while it runs.
