# The instant-only small model (SM57) stage 4 (a better dedicated instant model): handoff (2026-11-07) -- DONE (2026-11-08, see sm57-stage4-result.md)

Branch `sm57`, worktree `C:/Users/jonat/workspace/wt/sm57`. Plan and pre-registered rule: `DESIGN.md`, section "Stage 4 plan" (fixed 11-07 ~12:20, before any stage-4 GPU run; commit `0cc42e29`).

## What it asks

Does more instant-only training before the cool-down make a clearly better instant model? Candidate = stage 1's instant-only epoch 26 (`artifacts/small-model/sm57-inst-e26`, reused) + instant-only epochs 27-30 at the release's epoch recipe (8k final tier, the run's epoch-N order, AdamW 7e-4 constant, `curve_loss_weight` 0.0) + the release's one-epoch 32k Muon cool-down from epoch 30, instant-only. Reference = stage 1's instant-only arm (the ship candidate, label `sm57-inst-dec1ep32k/w8+s8`).

**Rule** (`sm57_stage4.py rule`): candidate - stage 1's instant-only arm, **instant** head, 722 users of the 732 set, points = % of the big model (1 leak-free epoch) LogLoss, paired user bootstrap 2,000, seed 0: **go** if the 95% CI is entirely below zero and the gain is at least 0.05 points; else **stop**. On go the launcher scores the candidate on the 1,500 validation users and `sm57_stage4.py pool1500` reports candidate - the release (SM63) (% of the release's LogLoss, the SM93 bar: CI below 0 and >= 0.05) and candidate - stage 1.

## Done before launch

- CPU proofs `check-cpu-sm57-stage4.json`: **ALL OK** (V1-V7, 12 s): the epoch builder reproduces stage 1's epoch 26 and SM72s epochs 22-25; epoch 27 passes `pair` with 5 refusals; orders 21-30 distinct; only the 1,500-pool scorer files changed (outside SourceWatch), pools equal their files; a cool-down prepared from epoch 26 by this tree == stage 1's instant-only cool-down and == the release + the flag; the rule code reproduces stage 1's -0.162 [-0.176, -0.148] and SM93's 1,487-user -0.164 [-0.179, -0.152]; break logic (Saturday 20:00, Sunday 16:00, moved nights).
- The 1,500-user pool option of SM93 (`00072cac`) checked out into `sm57` (scripts/eval_pools.py, scripts/small_model_heldout.py, eval-users-1500.json); SM93's rows copied to `stage4/`.
- Epoch-27 folder prepared (`artifacts/small-model/sm57s4-inst-e27`, not committed).

## What runs (detached, WMI hidden)

Launcher `launch-sm57-stage4.sh` (commit `0cc42e29`), WMI pid **67476** (bash.exe), launched 2026-11-07 12:17:41 (held at its RAM gate until 12:22: 2.9 GB free < 3 GB; the changing-learner (SM41g) CPU run in `wt/sm41` uses most of the RAM). Log `gpu/launch-sm57-stage4.log`, nohup `gpu/launch-sm57-stage4.nohup`, trainer output `gpu/train-stage4.out`, scorer `gpu/score-stage4.out`, GPU check and smokes `gpu/check-stage4.out`, prepare `gpu/prepare-stage4.out`, per segment `gpu/segments-stage4.jsonl`, per ticket `gpu/tickets-stage4.jsonl`. Queue owner `sm57`; every ticket purpose starts with "the instant-only small model (SM57) stage 4: ".

First minutes of epoch 27 (ticket after 000991, enqueued 12:23:22): 77.3k reviews/s average incl. start-up, 85-89k reviews/s now (stage 1's epoch 26: 87.7k average); nvidia-smi 14,097 MiB total, baseline ~1,340 MiB (~12.8 GB training, cap 0.8 in the options); 3.9 GB RAM free; trainer ETA ~7,100 s. 10-epoch ETA (training only): at 87.6k reviews/s 2k = 10 x 146.65M / 87.6k = ~4.65 h, 8k = 10 x 581.94M / 87.6k = ~18.5 h; at the 77.3k start-up average ~5.3 h / ~20.9 h.

| step | expected (estimate, if the queue stays free) |
|---|---|
| GPU check (ticket 000991) | **passed** 12:22:42-12:23:21: ALL OK (H1 stage 1's epoch-26 smoke == stage 1's GPU check bit for bit; H2 epoch 27 trains with curve loss weight 0 from epoch 26's trained parameters); peak allocated 143 / 137 MiB per smoke |
| epoch 27 | 12:23 -> ~14:20 |
| epoch 28 | ~14:20 -> ~16:15 |
| constant-LR scoring of epochs 26 and 28 (723 users each, report only) | ~16:15 -> ~16:45 |
| epoch 29 | ~16:45 -> ~18:40 |
| epoch 30 | ~18:40 -> snapshot stop ~19:58; **break 20:00-01:00**; resume 01:00 -> ~01:40 on 11-08 |
| cool-down prepare (CPU) + pair checks + smoke | ~01:40 -> ~02:00 |
| one-epoch 32k Muon cool-down (stage 1: 477 min) | ~02:00 -> ~10:00 |
| scoring the candidate (723 users) + rule -> `sm57-stage4-decision.{json,md}`, log `rule (final) exit 0` | ~10:00 -> ~10:20 |
| the 3 long users + rule again | ~10:20 -> ~10:45 |
| on go: the 1,500 validation users (771 users) + `pool1500` -> `sm57-stage4-pool1500.{json,md}` | ~10:45 -> ~11:35 |
| constant-LR epoch 30 scoring (report only) + rule again, then `done (verdict ...)` | ~11:35 -> **~11:50 on 11-08** (Sunday break from 16:00) |

Folders (not committed): `artifacts/small-model/sm57s4-inst-e27` .. `e30`, `sm57s4-inst-e30-dec-1ep32k-muon`. Labels `sm57s4-inst-e30-dec1ep32k/w8+s8` (candidate), `sm57s4-inst-e26/w8+s8`, `sm57s4-inst-e28/w8+s8`, `sm57s4-inst-e30/w8+s8` (constant-LR, report only); rows `gpu/sm57s4-rows.jsonl`, the 1,500-user extra rows `gpu/sm57s4-rows-1500.jsonl`.

## Follow-up (fresh agent, Sonnet medium is enough)

Do not wait on the GPU: check, act, stop. Always fetch first.

```bash
cd C:/Users/jonat/workspace/wt/sm57 && git fetch && git merge --ff-only
tail -5 artifacts/small-model/sm57/gpu/launch-sm57-stage4.log
tasklist //FI "PID eq 67476"                           # launcher alive?
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status
# re-run the rule by hand (CPU, seconds; exit 2 = the candidate's rows are not there yet):
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm57/sm57_stage4.py rule
cat artifacts/small-model/sm57/sm57-stage4-decision.md
cat artifacts/small-model/sm57/sm57-stage4-pool1500.md       # on go, after the 1,500-user scoring
```

1. **Mid-run (optional, after the log line `sm57s4-inst-e28/w8+s8: 723 users scored`, ~17:00 on 11-07)**: run the rule; report the constant-LR reading e28 - e26 on instant (points, CI, users better / worse). Report only, no decision.
2. **When the log ends with `done (verdict ...)`** (or `STOP: ...`: read the line before it and the tails of `gpu/train-stage4.out` / `gpu/score-stage4.out`): run the rule (and `pool1500` if go). Relaunch only if the launcher pid is gone and the log is not at `done` (below).
3. Commit with `git add -f` (as `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`): `gpu/launch-sm57-stage4.log`, `gpu/launch-sm57-stage4.nohup`, `gpu/segments-stage4.jsonl`, `gpu/tickets-stage4.jsonl`, `gpu/check-gpu-sm57-stage4.json`, `gpu/check-stage4.out`, `gpu/prepare-stage4.out`, `gpu/score-stage4.out`, `gpu/train-stage4.out` (if < 20 MB), `gpu/check-replay-label-windows-sm57s4-inst-e30-dec-1ep32k-muon.json`, `gpu/check-muon-fresh-sm57s4-inst-e30-dec-1ep32k-muon.json`, `gpu/sm57s4-rows.jsonl`, `gpu/sm57s4-rows-1500.jsonl` (on go), `sm57-stage4-decision.{json,md}`, `sm57-stage4-pool1500.{json,md}` (on go); write `sm57-stage4-result.md` (verdict, tables, speeds) and add it; mark this file DONE; then `git push` (HTTPS fallback: `git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/JSchoreels/anki-rwkv.git sm57`). Never commit checkpoints, snapshots, `.claude/`, `pages/`, `docs/GPU_OPTIMIZATIONS_LOG.md` or quiet.json. Do not edit `pages/data/factory-curated.json` (the main session does the Factory page).
4. Report (plain names with IDs in parentheses, never a bare ID; never name outside research sources; always state the unit): the verdict line; candidate - stage 1's instant-only arm on instant (points, CI, users better / worse) and whether >= 0.05 points; candidate - the release (SM63) and - the big model (1 leak-free epoch) on instant, the smaller-model bar; 732+long; the constant-LR trajectory (e28 - e26, e30 - e26) and the cool-down effect from e26 vs e30; on go the 1,500-user table (candidate - the release, candidate - stage 1, by part, row check); per epoch and the cool-down: wall minutes over all segments (`gpu/segments-stage4.jsonl`), **reviews/s** (result.json; the launcher's `complete:` lines cover the last segment only), peak GPU memory (allocated / reserved, nvidia-smi max minus baseline); the chain check; **10-epoch ETA** from the measured epoch reviews/s: 2k tier = 10 x 146.65M / (reviews/s), 8k tier = 10 x 581.94M / (reviews/s) (training only, plus ~1-2 min setup per epoch; at 87.6k reviews/s: ~4.65 h and ~18.5 h). Put measured speeds in the reply for the main session's `docs/GPU_OPTIMIZATIONS.md` (do not edit it yourself). If go: the candidate replaces stage 1's arm as the instant ship candidate only after the user decides (the SM93 agent integrates; do not touch `wt/sm93` or the fork).

## Relaunch (resumes: finished steps are skipped, an interrupted epoch / cool-down resumes from its snapshot)

Only if the launcher pid is gone and the log is not at `done`:

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/launch-sm57-stage4.sh >> C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/gpu/launch-sm57-stage4.nohup 2>&1"'; ProcessStartupInformation=$si}
```

To stop between tickets (and snapshot-stop a running epoch / cool-down at the next user boundary): `touch C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/gpu/STOP4` (remove it before relaunching). Do not edit `scripts/`, `expertium/` or `rwkv/` in this worktree while it runs (SourceWatch stops the training), and do not edit the running `launch-sm57-stage4.sh`. Never touch `C:/Users/jonat/workspace/wt/sm41` (a CPU run is going there) or `wt/sm93` / the Anki fork.
