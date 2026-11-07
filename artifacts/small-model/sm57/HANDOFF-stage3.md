# DONE - The instant-only small model (SM57) stage 3 (curve-only, the mirror arm): handoff (2026-11-07 08:19, filled 08:30 by the main session)

Branch `sm57`, worktree `C:/Users/jonat/workspace/wt/sm57`. Plan and pre-registered rule: `DESIGN.md`, section "Stage 3 plan" (fixed 11-07 ~08:20, before any stage-3 GPU run). Stage 2 (go, instant-only - joint on instant at epoch 5: -0.773 points [-0.861, -0.685]): `sm57-stage2-result.md`, `sm57-stage2-decision.md`.

## What it asks

One new arm, curve-only = stage 2's joint arm + `{"instant_loss_weight": 0.0}` (SM26's option, no code change: the instant term of the loss, which holds the instant KD targets, gets weight 0). Same fresh init, the 2k tier (2,000 users, 146,654,211 reviews/epoch), the leak-free small model (SM72s) recipe, same users / order / seeds / teacher targets as stage 2's joint arm, which is the comparison (not retrained). 5 epochs, scored at epochs 1, 3, 5 (723 users of the 732 set, leak-free, normal inputs, w8+s8).

**Rule** (`sm57_stage3.py rule`): curve-only - joint, **curve** head, epoch 5, 722 users, points = % of the big model (1 leak-free epoch) LogLoss, paired user bootstrap 2,000, seed 0: **go** if the 95% CI is entirely below zero and the gain is at least 0.05 points; else **stop**. Reported without a rule: curve-only's instant head (expected untrained), and the two-model comparison (instant-only instant, curve-only curve) vs joint per scored epoch (does the pair of dedicated models, the model-per-scheduler idea (SM93), beat the shared model on both heads).

## Done before launch

- CPU proofs `check-cpu-sm57-stage3.json` (log `check-cpu-sm57-stage3.out`): **ALL OK** (T1-T6): same fresh init hash (curve-only options == joint == stage 2's e01 results == SM72s epoch 1's `47a33c78...`); the epoch-1 folder differs from stage 2's joint e01 only by the flag / folder names / stage name, refusals work; the option's source path (applied at 0.0, teacher-blind, batch-blind, only `IMMEDIATE_SCALE` in the loss, the instant KD inside that term); a CPU trainer slice from the fresh init with the run's real replay KD: same batches, same first forward, curve-only's loss == the curve terms of the loss bit for bit (first update: joint 1.60165 = curve terms 0.53438 + instant 1.06726; curve-only 0.53438), reversing the teacher's rating targets leaves curve-only bit for bit unchanged (changes joint), flipping the curve targets changes curve-only, only `head_p` and `p_linear` (the instant head) lose their gradient (instant-only zeroes `head_w` and the GRU curve head instead); the rule code reproduces stage 2's epoch-5 pair; break logic.
- GPU check `gpu/check-gpu-sm57-stage3.json`: **ALL OK** (ticket 08:24:39-08:25:04; peak allocated 147 MiB per smoke, nvidia-smi max 1,450 MiB, baseline 939)
- Commits: `7362b5d7` (plan, scripts, launcher, CPU proofs); this handoff in the next commit

## What runs (detached, WMI hidden)

Launcher `launch-sm57-stage3.sh`, WMI pid **61452** (bash.exe), launched 2026-11-07 08:19:37 (held at its RAM gate until 08:24: the main session stopped the 2 extra changing-learner (SM41g) workers at 08:20); log `gpu/launch-sm57-stage3.log`, nohup `gpu/launch-sm57-stage3.nohup`, trainer output `gpu/train-stage3.out`, scorer `gpu/score-stage3.out`, GPU check `gpu/check-stage3.out`. Queue owner `sm57`; ticket purposes start with "the instant-only small model (SM57) stage 3: ". Curve-only epoch 1 enqueued 08:25:05; first minutes ~77k reviews/s average (~84-90k now) incl. start-up, 11.0 GB nvidia-smi total.

| step | expected (estimate, if the queue stays free) |
|---|---|
| GPU check | **passed** 08:24:39-08:25:04 (ALL OK: G1 joint smoke == stage 2's joint smoke bitwise, G2 curve-only trains with instant weight 0, G3 same fresh init) |
| curve-only e01 | 08:25 -> ~08:57 |
| scoring e01 (~14 min), partial rule | ~08:57 -> ~09:12 |
| e02, e03 + scoring e03 | ~09:12 -> ~10:30 |
| e04, e05 + scoring e05 + final rule -> `sm57-stage3-decision.{json,md}`, log `rule (final) exit 0`, then `done` | ~10:30 -> **~11:45** |

Folders `artifacts/small-model/sm57s3-curve-e0N` (not committed). Rows `gpu/sm57s3-rows.jsonl`, labels `sm57s3-curve-e0N/w8+s8`; the joint and instant-only rows are stage 2's `gpu/sm57s2-rows.jsonl`. Per segment `gpu/segments-stage3.jsonl` (wall minutes, nvidia-smi baseline / max), per ticket `gpu/tickets-stage3.jsonl`. Breaks: quiet.json all/gpu + the nightly 20:00-01:00 unless quiet.json has a window overlapping that night (then that window instead); epochs snapshot-stop <= 2 min before a break and resume after.

## Follow-up (fresh agent, Sonnet medium is enough)

Do not wait on the GPU: check, act, stop. Always fetch first.

```bash
cd C:/Users/jonat/workspace/wt/sm57 && git fetch && git merge --ff-only
tail -5 artifacts/small-model/sm57/gpu/launch-sm57-stage3.log
tasklist //FI "PID eq 61452"                                  # launcher alive?
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status
# re-run the rule by hand (CPU, seconds; exit 2 = epoch-5 rows not there yet, the file still has the epochs scored so far):
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm57/sm57_stage3.py rule
cat artifacts/small-model/sm57/sm57-stage3-decision.md
```

1. **After epoch 1 is scored** (log `rule (partial) exit 2: ...`): optional preliminary report: the epoch-1 pair (curve-only - joint on curve, points + CI, users better / worse) and curve-only's instant.
2. **When the log ends with `done`** (or `STOP: ...`: read the line before it and the tails of `gpu/train-stage3.out` / `gpu/score-stage3.out`): run the rule (above). Relaunch only if the launcher pid is gone and the log is not at `done` (below).
3. Commit with `git add -f` (as `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`): `gpu/launch-sm57-stage3.log`, `gpu/launch-sm57-stage3.nohup`, `gpu/segments-stage3.jsonl`, `gpu/tickets-stage3.jsonl`, `gpu/check-gpu-sm57-stage3.json`, `gpu/check-stage3.out`, `gpu/prepare-stage3.out`, `gpu/score-stage3.out`, `gpu/train-stage3.out` (if < 20 MB), `gpu/sm57s3-rows.jsonl`, `sm57-stage3-decision.{json,md}`; write `sm57-stage3-result.md` (verdict, tables, speeds) and add it; mark this file DONE; then `git push` (HTTPS fallback: `git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/JSchoreels/anki-rwkv.git sm57`). Never commit checkpoints, snapshots, `.claude/`, `pages/`, `docs/GPU_OPTIMIZATIONS_LOG.md` or quiet.json. Do not edit `pages/data/factory-curated.json` (the main session does the Factory page).
4. Report (plain names with IDs in parentheses, never a bare ID; never name outside research sources): the verdict line; the **trajectory table** from `sm57-stage3-decision.md` (per scored epoch: curve-only - joint on curve with CI and users better / worse, on instant; curve-only and joint vs the big model (1 leak-free epoch)); the **two-model table** (instant-only instant, curve-only curve vs joint) and whether both rules are go at epoch 5 (two dedicated models beat the shared one on both heads); same init confirmed; per epoch: training minutes, **reviews/s** (result.json), peak GPU memory (allocated / reserved, nvidia-smi max minus baseline), next to stage 2's joint arm; **ETA for 10 epochs** from the measured reviews/s: 2k tier = 10 x 146.65M / (reviews/s), 8k tier = 10 x 581.94M / (reviews/s) (training only, plus ~1-2 min setup per epoch; at ~83k reviews/s: ~4.9 h and ~19.5 h). Put measured speeds in the reply for the main session's `docs/GPU_OPTIMIZATIONS.md` (do not edit it yourself).

## Relaunch (resumes: finished steps are skipped, an interrupted epoch resumes from its snapshot)

Only if pid 61452 is gone and the log is not at `done`:

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/launch-sm57-stage3.sh >> C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/gpu/launch-sm57-stage3.nohup 2>&1"'; ProcessStartupInformation=$si}
```

To stop between tickets (and snapshot-stop a running epoch at the next user boundary): `touch C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/gpu/STOP3` (remove it before relaunching). Do not edit `scripts/`, `expertium/` or `rwkv/` in this worktree while it runs (SourceWatch stops the training), and do not edit the running `launch-sm57-stage3.sh`. Never touch `C:/Users/jonat/workspace/wt/sm41` (a CPU run is going there; watch free RAM: ~2.8 GB free at launch).
