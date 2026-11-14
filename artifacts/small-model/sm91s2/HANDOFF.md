# Show-time interval retrain (SM91) step 2: handoff

Branch `sm91s2` (cut from `sm91` at `24b3f154`, then merged `sm65` at `0e78817a` for the big-epoch `--init / --track` tooling), worktree `C:/Users/jonat/workspace/wt/sm91s2`. Design, rule, proofs and deviations: `DESIGN.md` (this folder).

## What it asks

Does one added big-model epoch on show-time intervals win back the big model's show-time cost? New model = the big model (1 leak-free epoch) + one epoch with the leak-free retrain (SM72) step 1 recipe (`scripts/sm72_big_epoch.py`: users 5000-10000, shuffle seed 1, fresh AdamW, warmup 200, 3.5e-4 -> 0 one_minus_sin, 2 strided lanes x 8,192 carried state, cap 0.8, deleted-card inputs removed), the only change being `"show_time_intervals": true` (the step 1 transform in the same trainer, no port). Reference = the big model (1 leak-free epoch) on show-time inputs, rows from the duration-leak scoring (SM31), reused: its cost there is +0.205 curve / +0.508 instant points.

**Rule** (`sm91s2_rule.py`, fixed before launch): 722 users of the 732 set, new - reference, both scored with `--inputs e2s-query`, review-weighted LogLoss, points = % of the big model (1 leak-free epoch) normal-input LogLoss, paired user bootstrap 2,000, seed 0, 95% CI. **go to step 3** if one head gains >= 0.10 points with its CI entirely below zero; else **stop**. Also reported: the gain as a share of SM31's +0.205 / +0.508, the cost left, the new model on normal inputs vs the reference (the added-epoch confound check), its own show-time cost, the 732+long pool.

## Done before launch (no CPU proof run yet: the user's break)

- `scripts/sm72_big_epoch.py prepare --extra-options` (absent = unchanged), `scripts/sm91s2.py cpu-check` (T1-T6), `sm91s2_rule.py`, `launch-sm91s2.sh` (dry run OK: `gpu/launch-sm91s2-dryrun.log`).
- Epoch folder `artifacts/small-model/sm91s2-big-showtime` prepared (not committed): `env RWKV_DELETED_CARDS=neighbour python scripts/sm72_big_epoch.py prepare --out artifacts/small-model/sm91s2-big-showtime --init <big model (1 leak-free epoch) master.pth> --track big-showtime --chunk-reviews 8192 --batch-chunks 2 --snapshot-users 50 --extra-options '{"show_time_intervals": true}'`. A quick JSON diff vs SM72's folder: only `show_time_intervals`, start weights / sha256, track, control dir, `snapshot_users` differ; rows equal `protocol.untrimmed.json` (the trims come with the launcher's label-window pre-flight; T4 then checks them against SM72's).

## What runs (detached, WMI hidden)

Launcher `launch-sm91s2.sh`, log `gpu/launch-sm91s2.log`, nohup `gpu/launch-sm91s2.nohup`. Launched 2026-11-02 20:11 (WMI pid **72836**, inner bash 76208, commit `70edc9f5`). It sleeps through the break (logs about once an hour), then:

| # | step (ticket purposes start with "show-time interval retrain (SM91) step 2: ") | expected min | output |
|---|---|---|---|
| - | CPU after 01:00: label-window pre-flight (`sm72_label_windows.py --trim`), then `scripts/sm91s2.py cpu-check`; log `CPU proofs exit 0: [sm91s2-check] ALL OK ...` or `STOP: CPU proofs not ALL OK` | ~2 + ~10-20 | `check-cpu-sm91s2.json`, `gpu/check.out`, `gpu/check-label-windows-sm91s2.json` |
| - | wait until `wt/sm91/.../gpu/launch-sm91.log` ends with `done` / `STOP: ended` (or its launcher has been gone 15 min, or 6 h) | SM91's normal-input scoring + rule, ~1 h after 01:00 | `gpu/sm91-waited` |
| 1 | trainer smoke of the show-time big-model epoch (fork RWKV, show-time inputs, 2 users x 512 reviews, scratch copy) | 10 (SM65: 12 s) | `gpu/smoke.out`, `gpu/smoke-ok` |
| 2 | the big model (fork RWKV) one added epoch on show-time intervals from the big model (1 leak-free epoch) (users 5000-10000, carried state 2 lanes x 8,192, AdamW 3.5e-4 -> 0, deleted-card inputs removed), NM of 370M reviews left (segment to the next window or the end) | ~573 (370.3M at ~11,000-11,450 reviews/s) | `gpu/train.out`, `artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth` |
| 3-4 | scoring the show-time big model (big model (1 leak-free epoch) + 1 show-time epoch) on e2s-query inputs, the 732 set minus 9 never-fit (723 users) (pass 1, cap 0.8), then the 3 long users (cap 0.92) | 50 (+15 per retry pass) + 20 | `gpu/sm91s2-inputs.jsonl`, `gpu/score.out` |
| 5-6 | the same on normal inputs | 30 (+15) + 20 | same |
| - | CPU rule -> `sm91s2-decision.{json,md}`; log `rule exit 0: [sm91s2-rule] ...`, then `done` | 1 | |

Labels: `sm91s2-big-showtime/w8+s8@e2s-query`, `sm91s2-big-showtime/w8+s8` (normal). GPU caps: trainer 0.8 (options), scorer 0.8, long users 0.92. Timeline if the queue stays free: CPU proofs 01:00-01:20, SM91 step 1 done ~02:00-02:30, smoke right after, epoch ~02:30 -> ~12:00, scoring ~12:00 -> ~14:00, rule ~14:00 on 2026-11-03 (well before the 20:00 break). A stop or a window cuts the epoch with a snapshot stop at the next user boundary; the next segment resumes from it.

## Follow-up (fresh agent, Sonnet medium is enough)

```bash
cd C:/Users/jonat/workspace/wt/sm91s2 && git fetch && git merge --ff-only
tail -5 artifacts/small-model/sm91s2/gpu/launch-sm91s2.log
powershell -NoProfile -c "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | ? { \$_.CommandLine -match 'launch-sm91s2\.sh' } | select ProcessId"   # alive? (72836 / 76208)
cat artifacts/small-model/sm91s2/sm91s2-decision.md      # when the log says "done"
# re-run the rule by hand (CPU, seconds):
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm91s2/sm91s2_rule.py decide
```

1. Check done: the log ends with `done`. On `STOP: ...` read the line before it and the tails of `gpu/check.out` (CPU proofs), `gpu/smoke.out`, `gpu/train.out`, `gpu/score.out`. A CPU-proof failure: read `check-cpu-sm91s2.json` (which check, its `problems` / `differing`), fix, delete the json, commit, relaunch (do not relaunch over a failure without understanding it).
2. Run the rule (above; exit 2 = rule rows missing).
3. Commit with `git add -f`, then push: `gpu/launch-sm91s2.log`, `gpu/*.out` (if < 20 MB), `gpu/check-label-windows-sm91s2.json`, `check-cpu-sm91s2.json`, `gpu/sm91s2-inputs.jsonl`, `sm91s2-decision.{json,md}`, plus `artifacts/small-model/sm91s2-big-showtime/decay.json` and `protocol.untrimmed.json`. Do not commit checkpoints or snapshots. Commit as Jonathan Schoreels, message ending with the Co-Authored-By line. Do not edit `pages/data/factory-curated.json` (the main session does the Factory page).
4. Report: the verdict line; both heads new - reference on e2s-query (points with CI); the win-back share of SM31's +0.205 / +0.508; the cost left; new normal - reference normal (if it shows a clear gain, part of the e2s gain is the added epoch, say so); epoch reviews/s and minutes (the launcher's `epoch complete:` line covers the last attempt only: for a resumed epoch add the segments from `gpu/train.out`) and CPU proofs ALL OK.
5. If go: step 3 (new teacher targets from `sm91s2-big-showtime/expertium/big-showtime/master.pth` on show-time inputs + small-model cool-down) needs its own design; the main session decides.

## Relaunch (resumes: finished steps are skipped, an interrupted epoch resumes from its snapshot)

Only if the launcher processes are gone and the log is not at `done` (the waits are mostly silent: about one log line per hour in a break, none while waiting for SM91, so a silent log alone does not mean it died):

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm91s2/artifacts/small-model/sm91s2/launch-sm91s2.sh >> C:/Users/jonat/workspace/wt/sm91s2/artifacts/small-model/sm91s2/gpu/launch-sm91s2.nohup 2>&1"'; ProcessStartupInformation=$si}
```

To stop between tickets (and snapshot-stop a running epoch at the next user boundary, ~1 min): `touch C:/Users/jonat/workspace/wt/sm91s2/artifacts/small-model/sm91s2/gpu/STOP` (remove it before relaunching). A quiet.json window starting now does the same within ~1 min. A hard kill loses at most the last 50 users (~5-6 min of GPU). Do not edit `scripts/`, `expertium/` or `rwkv/` in this worktree while it runs (SourceWatch stops the training; the launcher also refuses to start with uncommitted changes there).
