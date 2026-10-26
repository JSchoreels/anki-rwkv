# HANDOFF: SM72 step 3, the small model retrained leak-free (sm72s), prepared 2026-10-26 ~08:15

Worktree `C:/Users/jonat/workspace/wt/sm72s` (branch `sm72s`, cut from `sm72` at 0f817fc4). Launcher started
detached (hidden WMI process) at ~08:20; it waits for step 2 (the leak-free teacher export run by wt/sm72's launcher).

## What runs
`artifacts/small-model/sm72s/launch-sm72s.sh` = the full retrain (SM56, `wt/sm56/.../sm56/launch-sm56.sh`) with:
- every training process `RWKV_DELETED_CARDS=neighbour` (also the experiment's `env`), the SM71 transform in
  get_rwkv_data;
- distillation teacher = `wt/sm72/artifacts/small-model/sm72-big-leakfree-1ep/expertium/big-leakfree-1ep/master.pth`
  (step 1), replay `anki-rwkv/artifacts/cache/kd-replay/big-leakfree-1ep-final8k-neighbour` (step 2; manifest
  `deleted_cards: neighbour`, teacher sha ac7f0d8a96cb); the trainer refuses a teacher or deleted_cards mismatch;
- 15 fixed epochs (no plateau rule), then SM56's 1-epoch 32,768 x 1 Muon cool-down from e15 (main result), then the
  same cool-down from e10 (side branch, comparison with the released version; skip with `touch .../sm72s/NO-DEC10`);
- scoring leak-free (`--deleted-cards neighbour --split-existing`, scorer without the env var).
Experiment `artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree`; checked key by key against SM56's
experiment (only distillation teacher/replay and env differ); cool-downs checked against SM56's e15 cool-down
(only teacher/replay and the source epoch differ; checked on a scratch prepare 10-26: options and protocol full_run
identical, the remaining differences were the skipped trim pass and the source run, as expected).
Breaks: quiet.json ONLY (scope all/gpu), re-read every <= 10 min; no hard-coded daily break. Epochs (not resumable)
start only if they end before the next window; cool-downs run in segments with a snapshot stop 2 min before a window
(SM69 resumable decay) and resume after it.

## Order and estimates (daily window 21:00-02:00 from quiet.json)
| Step | Expected | ETA |
|---|---|---|
| wait for step 2: wt/sm72 log `verify exit 0;` + `done` | export ~09:40 | ~09:50 |
| CPU checks: replay manifest, experiment, label windows (trims == SM56's: 5599), smoke inputs | ~8 min | ~10:00 |
| reference row: SM56 cool-down from e15 (leak inputs) scored leak-free, 723 + 3 long users | 12 + 6 min | ~10:20 |
| epochs 1-15 (one ticket each, ~110-118 min wall at 82-93k reviews/s; epoch 1 smoke first) + 464 scoring (~10 min) | ~122 min / epoch | e5 Mon ~20:20, e14 Tue ~20:10, e15 Wed ~03:55 |
| cool-down from e15 (~505 min, segmented) + 732 scoring (723 + 3 long) + summary | ~8.7 h | Wed ~12:50 |
| side: cool-down from e10 + scoring + summary | ~8.7 h (+ the 21:00 window) | Thu ~02:40 |

## Files
- Launcher log: `artifacts/small-model/sm72s/launch-sm72s.log` (stdout: `launch-sm72s.nohup`); dry run 10-26 08:08
  `launch-sm72s-dryrun.log` (DRYRUN OK; label-window check skipped there, export incomplete).
- Per-epoch lines: `[sm72s ...] epoch k/15: curve X instant Y; gain vs e1 ...; vs big model leak-free (458 users) +a% / +b%`
  (also `sm72s/epochs.jsonl`). Scores: `sm72s/gpu/small-lf-epochs-464.jsonl` (labels `s-w64kd-final-sc-lanes-leakfree-eK/w8+s8`),
  `sm72s/gpu/small-lf-neighbour.jsonl` (`...-leakfree-dec15-1ep32k-muon/w8+s8`, `...-dec10-...`),
  `sm72s/gpu/small-dec15-neighbour.jsonl` (`sm56dec15/w8+s8`). Training: `.../scale-w64kd-final-sc-lanes-leakfree/phases.jsonl`,
  cool-downs `artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec{15,10}-1ep32k-muon/`; ticket output `sm72s/gpu/{score,decay}.out`.
- Summary (after each cool-down): `artifacts/small-model/sm72s/sm72s-summary.{md,json}`: 464 set (458), 732 (723 fit users,
  = the SM71/SM72 readouts), 732+long (726); every model vs the big model leak-free (% with CI), all / existing / deleted
  reviews, paired deltas leak-free vs the leak-input small models, the epoch table, training times.

## Commands
- Status: `tail -5 C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/launch-sm72s.log`
- Stop between tickets (a running cool-down segment gets a snapshot stop): `touch C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/STOP`
- Resume / relaunch after a restart (skips finished epochs, scored users, finished cool-downs; a cool-down resumes from
  its snapshot; an interrupted epoch restarts that epoch). Remove STOP first. PowerShell:
  `$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/launch-sm72s.sh >> C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/launch-sm72s.nohup 2>&1"'; ProcessStartupInformation=$si}`
- Summary again (CPU): `EXP=C:/Users/jonat/workspace/wt/sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/sm72s_rule.py summary`
- Do not edit `scripts/ expertium/ rwkv/` here while it runs (SourceWatch). Never edit the running launch-sm72s.sh.

## Follow-ups (fresh Sonnet medium agent each)
- Each epoch line: repost the epoch table (gain vs e1 and vs the big model leak-free, curve + instant); Factory row.
- `done` or a summary line: report from `sm72s-summary.md`; Factory results row tagged "(leak-free)"; commit (git add -f)
  `sm72s/gpu/*.jsonl` (benchmark users only), `sm72s/epochs.jsonl`, the summary, the log; push.
