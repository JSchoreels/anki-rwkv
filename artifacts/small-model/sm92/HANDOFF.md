# Show-time retrain of the release (SM92): handoff

Branch `sm92` (cut from `sm91s3` at `6cd468df`: show-time interval retrain (SM91) steps 1-3 tooling + the resumable decay (SM69), `decay-resume` already merged), worktree `C:/Users/jonat/workspace/wt/sm92`. Design, recipe and rule: `DESIGN.md` (this folder).

## What it asks

The leak-free small model (SM72s) epoch 25 (the release's start) + epoch 26 on show-time inputs (the release's epoch 21-25 recipe + `"show_time_intervals": true`) + the release's exact cool-down (one epoch, 32k windows, fresh Muon, LR 7e-4 -> 0, same users / order / seeds / KD replay targets) on show-time inputs. Is it a ship candidate against the released small model (cool-down from epoch 25, SM63) when both are scored on show-time inputs? Confound: one more epoch than the release (the normal-vs-normal row shows its size).

**Rule** (`sm92_rule.py`, fixed before launch): 722 users of the 732 set, new - release both on `e2s-query`, review-weighted LogLoss, points = % of the big model (1 leak-free epoch) normal-input LogLoss, paired user bootstrap 2,000, seed 0, 95% CI. **ship candidate** if one head gains >= 0.05 points with its CI below zero and no head is worse by >= 0.05 points with its CI above zero; else **no ship**.

## Done before launch (CPU)

- Epoch folder `artifacts/small-model/sm92-e26` prepared (not committed): 8,000 users, 581,942,724 reviews, start SM72s epoch 25 (master + AdamW), update offset 551,950.
- CPU proofs `check-cpu-sm92.json` on `f8a19500`: **ALL OK** (C1 epochs 21-25 rebuilt == disk; C2 epoch 26 == epoch 25's recipe except flag / start / index / order / paths; C3 trainer + scorer files == `cc7f5358` where SM91's T1-T3 passed, T1 + T2 re-run on epoch-26 rows: trainer rows == scorer `e2s-query` rows bitwise; C4 a flag cool-down from epoch 25 == the release's cool-down except flag and names; C5 rule self-test reproduces step 3's +0.146 [+0.066, +0.230] / -0.062 [-0.161, +0.016]).

## What runs (detached, WMI hidden)

Launcher `launch-sm92.sh` (commit `764bac35`), WMI pid **76972**, log `gpu/launch-sm92.log`, nohup `gpu/launch-sm92.nohup`, launched 2026-11-03 15:55. Ticket purposes, each prefixed "show-time retrain of the release (SM92): ":

| # | purpose (after the prefix) | expected min | clock (if the queue stays free) |
|---|---|---|---|
| 1 | trainer smoke of epoch 26 (w64, show-time inputs, 2 users x 512 reviews) | 15 | **passed** 15:55:44-15:55:58 (exit 0, 2 updates, 1,024 reviews) |
| 2 | epoch 26 on show-time inputs (w64 from SM72s epoch 25, the release's epoch 21-25 recipe ...) (segment to the next window or the end) | ~111 training | **running since 15:56** (ticket expected 89 min); snapshot stop ~17:28 (~80% done), **pause 17:30-19:30**, resume 19:31 -> ~19:55 |
| - | CPU: cool-down prepare (`sm77_warm.py cooldown --epoch 25 --fraction 1.0`) + label-window trim/check + Muon fresh check + `sm92_pair.py decay` vs the release (log `[sm92-pair] decay OK`) | ~5-15 | ~19:55-20:10 |
| 3 | the release's one-epoch 32k Muon cool-down on show-time inputs from epoch 26 (segment to the next window or the end) | 560 (~520 training) | ~20:10 -> ~04:50 on 11-04 |
| 4-5 | scoring on e2s-query inputs, 723 users (cap 0.8), then the 3 long users (cap 0.92) | 32 + 25 (SM91: ~12 + ~1) | ~04:50-05:20 |
| 6-7 | scoring on normal inputs, 723 users, then the 3 long users | 32 + 25 | ~05:20-05:50 |
| - | CPU rule -> `sm92-decision.{json,md}`; log `rule exit 0: [sm92-rule] ...`, then `done` | 1 | ~05:50 (at the latest ~06:45) |

Label `sm92-e26-dec1ep32k/w8+s8` (`@e2s-query` for show-time), rows `gpu/sm92-inputs.jsonl`. Folders: epoch `artifacts/small-model/sm92-e26`, cool-down `artifacts/small-model/sm92-e26-dec-1ep32k-muon`. Windows: quiet.json (17:30-19:30 tonight) + the daily 20:00-01:00 break, **except the 2026-11-03 20:00 one** (date fixed in `block()`); from 11-04 the break applies again (next: 11-04 20:00, far after the expected end). The epoch and the cool-down get a snapshot stop <= 2 min before a window and resume after it.

## Follow-up (fresh agent, Sonnet medium is enough)

```bash
cd C:/Users/jonat/workspace/wt/sm92 && git fetch && git merge --ff-only
tail -5 artifacts/small-model/sm92/gpu/launch-sm92.log
tasklist //FI "PID eq 76972"                                  # launcher alive?
cat artifacts/small-model/sm92/sm92-decision.md               # when the log says "done"
# re-run the rule by hand (CPU, seconds):
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm92/sm92_rule.py decide
```

1. Check done: the log ends with `done` (or `STOP: ...`: read the line before it, `gpu/train.out` / `gpu/score.out` tails).
2. Run the rule (above; exit 2 = rule rows missing).
3. Commit with `git add -f` (commit as `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ends with a blank line + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`): `gpu/launch-sm92.log`, `gpu/launch-sm92.nohup`, `gpu/*.out` (if < 20 MB), `gpu/check-*.json`, `gpu/sm92-inputs.jsonl`, `sm92-decision.{json,md}`; then `git push`. Do not commit checkpoints, snapshots or caches. Do not edit `pages/data/factory-curated.json` (the main session does the Factory page).
4. Report: the verdict line; both heads new - release on e2s-query (points with CI); new normal - release normal (the extra-epoch confound); the new model's own show-time cost; vs the big model (1 leak-free epoch) normal and e2s-query and vs the show-time big model; the smaller-model bar three ways; 732+long; epoch and cool-down reviews/s and minutes (the launcher's "complete:" lines; for a resumed run they cover the last segment only, so also give the wall clock from `gpu/launch-sm92.log`) and the 8k / 2k 10-epoch ETA from the epoch speed.

## Relaunch (resumes: finished steps are skipped, an interrupted epoch / cool-down resumes from its snapshot)

Only if pid 76972 is gone and the log is not at `done`:

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm92/artifacts/small-model/sm92/launch-sm92.sh >> C:/Users/jonat/workspace/wt/sm92/artifacts/small-model/sm92/gpu/launch-sm92.nohup 2>&1"'; ProcessStartupInformation=$si}
```

To stop between tickets (and snapshot-stop a running epoch / cool-down at the next user boundary): `touch C:/Users/jonat/workspace/wt/sm92/artifacts/small-model/sm92/gpu/STOP` (remove it before relaunching). Do not edit `scripts/`, `expertium/` or `rwkv/` in this worktree while it runs (SourceWatch stops the training), and do not edit the running `launch-sm92.sh`.
