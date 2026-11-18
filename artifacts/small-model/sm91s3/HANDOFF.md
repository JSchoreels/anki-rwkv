# Show-time interval retrain (SM91) step 3: handoff

Branch `sm91s3` (cut from `sm91s2` at `514e2016`, `origin/sm91` merged for step 1's final rows), worktree `C:/Users/jonat/workspace/wt/sm91s3`. Design, confound, rule, proofs: `DESIGN.md` (this folder).

## What it asks

Does the cool-down with KD targets from the show-time big model on show-time inputs beat step 1's arm (the same epoch 21 + the same cool-down with the old targets) on `e2s-query`? Arms differ only in the cool-down's `distillation.teacher` + `distillation.replay` (`sm91s3_pair.py`). Confound: epoch 21 (shared) was trained with the old targets, so this measures the teacher swap over the last quarter-epoch only.

**Rule** (`sm91s3_rule.py`, fixed before launch): 722 users of the 732 set, new - step 1's arm, both on `e2s-query`, review-weighted LogLoss, points = % of the big model (1 leak-free epoch) normal LogLoss, paired user bootstrap 2,000, seed 0, 95% CI. **adopt the show-time targets** if one head gains >= 0.05 points with its CI below zero and no head is worse by >= 0.05 with its CI above zero; else **no gain**. Also reported: vs the big model (1 leak-free epoch) e2s, vs the show-time big model e2s, vs the released small model e2s, cost left, the smaller-model bar three ways (a normal vs normal big, b e2s vs big e2s, c e2s vs show-time big e2s).

## What runs (detached, WMI hidden)

Launcher `launch-sm91s3.sh`, log `gpu/launch-sm91s3.log`, nohup `gpu/launch-sm91s3.nohup`. Launched 2026-11-03 11:00:58, WMI pid **73420**, commit `185477e1`. Ticket purposes start with "show-time interval retrain (SM91) step 3: ".

| # | step | expected min | clock if on time | output |
|---|---|---|---|---|
| - | CPU proofs (`scripts/sm91s3.py cpu-check`) + pair check | 2 | 11:01-11:03 | `check-cpu-sm91s3.json`, `gpu/check.out` |
| 1 | teacher targets from the show-time big model on show-time inputs, the 1,991 users of the cool-down (145.6M reviews) | 60-75 (running at ~39,000 reviews/s: ~62 min) | 11:03-~12:07 | `anki-rwkv/artifacts/cache/kd-replay/big-showtime-cool1991-neighbour-e2s`, `gpu/export.out` |
| - | CPU: export status + verify (8 users) + label windows + Muon fresh check + pair `--targets` | 3 | ~12:10 | `gpu/after-export-ok`, `gpu/check-*.json` |
| 2 | trainer smoke of the cool-down with the show-time targets | 1-2 | ~12:12 | `gpu/smoke-ok` |
| 3 | quarter-epoch 32k Muon cool-down with the show-time targets from step 1's arm epoch 21 | ~120 | ~12:12-14:15 | `artifacts/small-model/sm91s3-arm-e21-dec-q32k-muon-stt`, `gpu/train.out` |
| 4-5 | scoring on e2s-query, 723 users (cap 0.8) + 3 long users (0.92) | ~15 | 14:15-14:30 | `gpu/sm91s3-inputs.jsonl`, `gpu/score.out` |
| - | rule -> `sm91s3-decision.{json,md}` | 1 | 14:30 | |
| 6-7 | scoring on normal inputs (information) | ~15 | 14:30-14:45 | same rows file |
| - | rule again (adds the normal-input rows and bar (a)), then `done` | 1 | ~14:45 | |

**16:00 user stop.** The launcher never waits through the 16:00 window: if a step does not fit before it, a running cool-down gets a snapshot stop, the log says `STOP: user stop 16:00, resume needs the user's go` and it exits (code 3). Nothing restarts by itself. After the user's go: relaunch (below); finished steps are skipped, the export skips finished users, the cool-down resumes from its snapshot.

## Follow-up (fresh agent, Sonnet medium is enough)

```bash
cd C:/Users/jonat/workspace/wt/sm91s3 && git fetch && git merge --ff-only
tail -5 artifacts/small-model/sm91s3/gpu/launch-sm91s3.log
powershell -NoProfile -c "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | ? { \$_.CommandLine -match 'launch-sm91s3\.sh' } | select ProcessId"   # alive?
cat artifacts/small-model/sm91s3/sm91s3-decision.md       # when the log says "done" (a first version exists after the e2s-query rows)
# re-run the rule by hand (CPU, seconds):
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm91s3/sm91s3_rule.py decide
```

1. Check done: the log ends with `done`. On `STOP: user stop 16:00 ...`: report what is left, do not relaunch without the user's go. On another `STOP: ...`: read the line before it and the tails of `gpu/check.out`, `gpu/export.out`, `gpu/train.out`, `gpu/score.out`.
2. Run the rule (above; exit 2 = rule rows missing).
3. Commit with `git add -f`, then `git push`: `gpu/launch-sm91s3.log`, `gpu/*.out` (if < 20 MB), `gpu/check-*.json`, `check-cpu-sm91s3.json`, `gpu/sm91s3-inputs.jsonl`, `sm91s3-decision.{json,md}`, `artifacts/small-model/sm91s3-arm-e21-dec-q32k-muon-stt/protocol.prepared.json` if present, and `anki-rwkv/artifacts/cache/kd-replay/big-showtime-cool1991-neighbour-e2s/{manifest,status,verify}.json` copied into `gpu/replay-*.json` (the cache itself is not committed). Never commit checkpoints, snapshots, target caches, `docs/GPU_OPTIMIZATIONS_LOG.md` or `.claude/`. Commit as `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do not edit `pages/data/factory-curated.json`.
4. Report: the verdict line; both heads new - step 1's arm on e2s-query (points with CI); new vs the big model (1 leak-free epoch) e2s, vs the show-time big model e2s, vs the released small model e2s; the bar table (a)/(b)/(c); export reviews/s and minutes (`gpu/export.out` "pass finished" lines), cool-down reviews/s and minutes (the launcher's `complete:` line, add segments if resumed) and the 8k / 2k 10-epoch ETA; CPU proofs ALL OK.
5. If adopt: the full version (new targets for all 8,000 users ~4.8 h, then epoch 21 + cool-down from SM72s epoch 20 with them) needs the main session's decision.

## Relaunch (only if the launcher is gone, the log is not at `done`, and, after 16:00 today, only with the user's go)

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm91s3/artifacts/small-model/sm91s3/launch-sm91s3.sh >> C:/Users/jonat/workspace/wt/sm91s3/artifacts/small-model/sm91s3/gpu/launch-sm91s3.nohup 2>&1"'; ProcessStartupInformation=$si}
```

To stop: `touch C:/Users/jonat/workspace/wt/sm91s3/artifacts/small-model/sm91s3/gpu/STOP` (between tickets; a running cool-down gets a snapshot stop at the next user boundary; a running export ticket finishes its current user only at its time limit, so cancel its queue ticket to stop it now: it resumes per user). Remove `gpu/STOP` before relaunching. Do not edit `scripts/`, `expertium/` or `rwkv/` in this worktree while it runs; never edit the running launcher.
