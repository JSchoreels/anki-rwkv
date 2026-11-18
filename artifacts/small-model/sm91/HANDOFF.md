# Show-time interval retrain (SM91) step 1: handoff

Branch `sm91` (cut from `sm73s2` at `518b918b`, then merged `sm31`: the smaller model, card stream 1 layer (SM73 stage 2) warm-start / cool-down tooling + the duration-leak scoring (SM31) inputs module and scorer), worktree `C:/Users/jonat/workspace/wt/sm91`. Design, KD-target decision and deviations: `DESIGN.md` (this folder).

## What it asks

Does training on show-time intervals win back part of SM31's cost (+0.13 curve / +0.43 instant points for the released small model scored on show-time inputs without retraining)? Arm = the leak-free small model (SM72s) epoch 20 (w64) + epoch 21 (SM72s's protocol) + quarter-epoch 32k Muon cool-down, all on show-time inputs (track option `"show_time_intervals": true` = the scorer's `e2s-query` transform); reference = SM72s epoch 21 + the same cool-down on normal inputs (the tiny architecture additions (SM77) reference, re-scored). KD replay targets unchanged in both arms (DESIGN.md says why).

**Rule** (`sm91_rule.py`, fixed before launch): 722 users of the 732 set, arm - reference, both scored with `--inputs e2s-query`, review-weighted LogLoss, points = % of the big model (1 leak-free epoch) normal-input LogLoss, paired user bootstrap 2,000, seed 0, 95% CI. **go to step 2** if one head gains >= 0.10 points (curve ~0.00030 / instant ~0.00028 LogLoss) with its CI entirely below zero; else **stop**. Also reported: the reference's own show-time cost, the share of it (and of SM31's +0.13 / +0.43) the arm wins back, the arm on normal inputs vs the reference, the 732+long pool.

## Done before launch (CPU)

- `scripts/gpu_r01_samples.py` (`show_time_query_rows`, `show_time_frame`), `scripts/gpu_r01_prefetch.py` (option whitelist + frame hook, refuses unsupported paths), `scripts/sm77_warm.py epoch --extra-options`; `scripts/sm91.py cpu-check`.
- Arm epoch folder `artifacts/small-model/sm91-arm-e21` prepared (not committed): `sm91_pair.py epoch` OK (8,000 users, update offset 441,560, start SM72s epoch 20).
- CPU proofs `check-cpu-sm91.json` on commit `df9fee59`: **ALL OK** (T1 flag moves only interval columns 2-7 on query rows + the curve query time on update rows; T2 trainer rows == scorer `e2s-query` rows bitwise on 3 users / 93,515 rows; T3 flag off == the current trainer (sm73s2 `518b918b` tree) bitwise on full training batches; T4 pair config epoch + cool-down, reference-vs-itself refused; T5 rule reproduces SM31's +0.128 [+0.106, +0.160] / +0.428 [+0.337, +0.546]).

## What runs (detached, WMI hidden)

Launcher `launch-sm91.sh`, log `gpu/launch-sm91.log`, nohup `gpu/launch-sm91.nohup`. Launched 2026-11-02 11:03 (WMI pid **61404**, commit `8cb60579`). It first waits, **silently**, until the smaller model, card stream 1 layer (SM73 stage 2) launch log (`wt/sm73s2/.../gpu/launch-sm73s2.log`) ends with `done` or `STOP: ended` (at most 5 h, then goes ahead), so SM73 stage 2's scoring is not overtaken; expected ~13:30-14:15. Ticket purposes in order, each prefixed "show-time interval retrain (SM91): ":

| # | purpose (after the prefix) | expected min | output |
|---|---|---|---|
| 1 | scoring the tiny architecture additions (SM77) reference cool-down (the leak-free small model (SM72s) epoch 21 + quarter-epoch 32k Muon, normal-trained) on e2s-query inputs, the 732 set minus 9 never-fit (723 users) (pass 1, 0 done, cap 0.8) | 32 (retry passes 20) | `gpu/sm91-inputs.jsonl`, `gpu/score.out` |
| 2 | the same, the 3 long users of the 732 set (pass 1, cap 0.92) | 25 | same |
| 3 | trainer smoke of the arm epoch (w64, show-time inputs, 2 users x 512 reviews) | 15 | `gpu/check.out`, `gpu/smoke-ok` |
| 4 | arm epoch 21 on show-time inputs (w64 from the leak-free small model (SM72s) epoch 20, its epoch-21 protocol: ...) (segment to the next window or the end) | 125 (must fit whole) | `gpu/train.out`, `artifacts/small-model/sm91-arm-e21` |
| - | CPU: cool-down prepare + label windows + Muon fresh check + `sm91_pair.py decay` (log line `[sm91-pair] decay OK`) | ~2 | `gpu/check-*.json`, `gpu/prepare.out` |
| 5 | arm quarter-epoch 32k Muon cool-down on show-time inputs from arm epoch 21 (w64, the reference's options and order) (segment to the next window or the end) | 150 | `artifacts/small-model/sm91-arm-e21-dec-q32k-muon` |
| 6-7 | scoring the show-time arm cool-down on e2s-query inputs, 723 users (cap 0.8), then the 3 long users (cap 0.92) | 32 + 25 | `gpu/sm91-inputs.jsonl` |
| 8-9 | scoring the show-time arm cool-down (information) on normal inputs, 723 users, then the 3 long users | 32 + 25 | same |
| - | CPU rule -> `sm91-decision.{json,md}`; log `rule exit 0: [sm91-rule] ...`, then `done` | 1 | |

Labels: `sm77-ref-e21-q32k/w8+s8@e2s-query`, `sm91-arm-e21-q32k/w8+s8@e2s-query`, `sm91-arm-e21-q32k/w8+s8` (normal). GPU caps: trainer 0.8 (options), scorer 0.8, long users 0.92. The launcher respects quiet.json and the daily 20:00-01:00 break itself; the epoch only starts when 125 min fit before the next window; the cool-down is cut by a snapshot stop <= 2 min before a window and resumes after it (01:00). Expected: epoch ~14:45-16:50, cool-down ~16:55-19:58 (rest after 01:00), arm scoring after 01:00, rule ~02:30 on 2026-11-03 if the queue stays free.

## Follow-up (fresh agent, Sonnet medium is enough)

```bash
cd C:/Users/jonat/workspace/wt/sm91 && git fetch && git merge --ff-only
tail -5 artifacts/small-model/sm91/gpu/launch-sm91.log
tasklist //FI "PID eq 61404"                                  # launcher alive?
cat artifacts/small-model/sm91/sm91-decision.md               # when the log says "done"
# re-run the rule by hand (CPU, seconds):
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm91/sm91_rule.py decide
```

1. Check done: the log ends with `done` (or `STOP: ...`: read the line before it, `gpu/train.out` / `gpu/score.out` tails).
2. Run the rule (above; exit 2 = rule rows missing).
3. Commit with `git add -f`: `gpu/launch-sm91.log`, `gpu/*.out` (if < 20 MB), `gpu/check-*.json`, `gpu/sm91-inputs.jsonl`, `sm91-decision.{json,md}`; push. Do not commit checkpoints. Do not edit `pages/data/factory-curated.json` (the main session does the Factory page).
4. Report: the verdict line; both heads arm - reference on e2s-query (points with CI); the reference's own show-time cost and the win-back shares (of it and of SM31's +0.13 / +0.43); the arm on normal inputs vs the reference; epoch and cool-down reviews/s and minutes (the launcher's "complete:" lines) and the 8k / 2k 10-epoch ETA from them.

## Relaunch (resumes: finished steps are skipped, an interrupted epoch / cool-down resumes from its snapshot)

Only if pid 61404 is gone and the log is not at `done` (the first wait is silent, so a silent log alone does not mean it died):

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm91/artifacts/small-model/sm91/launch-sm91.sh >> C:/Users/jonat/workspace/wt/sm91/artifacts/small-model/sm91/gpu/launch-sm91.nohup 2>&1"'; ProcessStartupInformation=$si}
```

To stop between tickets (and snapshot-stop a running epoch / cool-down at the next user boundary): `touch C:/Users/jonat/workspace/wt/sm91/artifacts/small-model/sm91/gpu/STOP` (remove it before relaunching). Do not edit `scripts/`, `expertium/` or `rwkv/` in this worktree while it runs (SourceWatch stops the training).
