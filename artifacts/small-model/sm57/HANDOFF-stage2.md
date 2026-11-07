# RUNNING - The instant-only small model (SM57) stage 2: handoff (2026-11-07 01:42)

Branch `sm57`, worktree `C:/Users/jonat/workspace/wt/sm57`. Plan and pre-registered rule: `DESIGN.md`, section "Stage 2 plan" (fixed 11-07 ~01:35, before any run). Stage 1 (go, instant -0.147 points vs joint): `sm57-result.md`.

## What it asks

From the same fresh init, on the factory's 2k tier (2,000 users, 146,654,211 reviews/epoch) with the leak-free small model (SM72s)'s own from-scratch recipe (2,048-review windows x 16 lanes, leak-free replay KD, AdamW 7e-4 constant, training_seed 2025), does instant-only (`"curve_loss_weight": 0.0`) lead joint on instant from epoch 1, or does the curve task help early? Up to 5 epochs per arm (F, cut by the time plan after epoch 1 if > 7 h), interleaved; scored at epochs 1, 3 and F (723 users of the 732 set, leak-free, normal inputs, w8+s8). No cool-down.

**Rule** (`sm57_stage2.py rule`): instant-only - joint, instant head, final epoch F, 722 users, points = % of the big model (1 leak-free epoch) LogLoss, paired user bootstrap 2,000, seed 0: **go** if the 95% CI is entirely below zero and the gain is at least 0.05 points; else **stop**.

## Done before launch

- CPU proofs `check-cpu-sm57-stage2.json`: **ALL OK** (S1-S7: tier facts; same fresh init hash in both arms == SM72s epoch 1's `47a33c78...`; epoch-1 folders differ only by the flag, refusals work; distinct epoch orders; CPU trainer slice from the fresh init: `wt/sm92` tree == this tree bitwise for joint, instant-only same batches / same first predictions, loss == instant average; rule code reproduces stage 1's -0.147 [-0.162, -0.133]; break logic).
- GPU check `gpu/check-gpu-sm57-stage2.json` (ticket 000952, 01:35:47-01:36:27): **ALL OK** (joint smoke base tree == this tree bitwise; instant-only trains with weight 0; all three smokes from the same init == SM72s epoch 1's).
- Commit `9d957f22` (plan, scripts, launcher, proofs).

## What runs (detached, WMI hidden)

Launcher `launch-sm57-stage2.sh`, WMI pid **34176** (bash.exe), launched 2026-11-07 01:35:46; log `gpu/launch-sm57-stage2.log`, nohup `gpu/launch-sm57-stage2.nohup`, trainer output `gpu/train-stage2.out`, scorer `gpu/score-stage2.out`. Queue owner `sm57`; ticket purposes start with "the instant-only small model (SM57) stage 2: ". Instant-only epoch 1 started 01:36:28 (ticket 000953); first minutes ~72k reviews/s incl. start-up, 12.7 GB nvidia-smi total (baseline ~0.6 GB).

| step | expected (estimate, if the queue stays free) |
|---|---|
| GPU check (000952) | **passed** 01:36 |
| instant-only e01, joint e01 | 01:36 -> ~02:40 |
| scoring both e01 (~13 min each), plan (`gpu/plan-stage2.json`: F), partial rule | ~02:40 -> ~03:10 |
| e02, e03 pairs + scoring e03 | ~03:10 -> ~05:40 |
| e04, e05 pairs + scoring e05 + final rule -> `sm57-stage2-decision.{json,md}`, log `rule (final) exit 0`, then `done` | ~05:40 -> **~08:15** |

Folders `artifacts/small-model/sm57s2-{inst,joint}-e0N` (not committed). Rows `gpu/sm57s2-rows.jsonl`, labels `sm57s2-{inst,joint}-e0N/w8+s8`. Per segment `gpu/segments-stage2.jsonl` (wall minutes, nvidia-smi baseline / max), per ticket `gpu/tickets-stage2.jsonl`. Breaks: quiet.json all/gpu + the nightly 20:00-01:00 unless quiet.json has a window overlapping that night (then that window instead); epochs snapshot-stop <= 2 min before a break and resume after.

## Follow-up (fresh agent, Sonnet medium is enough)

Do not wait on the GPU: check, act, stop. Always fetch first.

```bash
cd C:/Users/jonat/workspace/wt/sm57 && git fetch && git merge --ff-only
tail -5 artifacts/small-model/sm57/gpu/launch-sm57-stage2.log
tasklist //FI "PID eq 34176"                                    # launcher alive?
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status
cat artifacts/small-model/sm57/gpu/plan-stage2.json              # F and measured epoch / scoring minutes (after epoch 1)
# re-run the rule by hand (CPU, seconds; exit 2 = final-epoch rows not there yet, the file still has the epochs scored so far):
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm57/sm57_stage2.py rule
cat artifacts/small-model/sm57/sm57-stage2-decision.md
```

1. **After epoch 1 is scored** (log `plan: {...}` then `rule (partial) exit 2: ...`): optional preliminary report: the epoch-1 pair (instant-only - joint, instant, points + CI, users better / worse) and the epoch-1 reading; F from the plan.
2. **When the log ends with `done`** (or `STOP: ...`: read the line before it and the tails of `gpu/train-stage2.out` / `gpu/score-stage2.out`): run the rule (above).
3. Commit with `git add -f` (as `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`): `gpu/launch-sm57-stage2.log`, `gpu/launch-sm57-stage2.nohup`, `gpu/segments-stage2.jsonl`, `gpu/tickets-stage2.jsonl`, `gpu/plan-stage2.json`, `gpu/check-gpu-sm57-stage2.json`, `gpu/check-stage2.out`, `gpu/prepare-stage2.out`, `gpu/score-stage2.out`, `gpu/train-stage2.out` (if < 20 MB), `gpu/sm57s2-rows.jsonl`, `sm57-stage2-decision.{json,md}`; mark this file DONE; then `git push` (HTTPS fallback: `git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/JSchoreels/anki-rwkv.git sm57`). Never commit checkpoints, snapshots, `.claude/`, `pages/`, `docs/GPU_OPTIMIZATIONS_LOG.md` or quiet.json. Do not edit `pages/data/factory-curated.json` (the main session does the Factory page).
4. Report (plain names with IDs in parentheses, never a bare ID; never name outside research sources): the verdict line; the **trajectory table** from `sm57-stage2-decision.md` (per scored epoch: instant-only - joint on instant with CI and users better / worse; each arm's instant and the joint arm's curve vs the big model (1 leak-free epoch)); the epoch-1 reading (does instant-only lead from the start); same init confirmed; F and whether the time plan cut epochs; per arm and epoch: training minutes, **reviews/s** (result.json), peak GPU memory (allocated / reserved, nvidia-smi max minus baseline); **ETA for 10 epochs** from the measured reviews/s: 2k tier = 10 x 146.65M / (reviews/s), 8k tier = 10 x 581.94M / (reviews/s) (training only, plus ~1-2 min setup per epoch). Put measured speeds in the reply for the main session's `docs/GPU_OPTIMIZATIONS.md` (do not edit it yourself).

## Relaunch (resumes: finished steps are skipped, an interrupted epoch resumes from its snapshot)

Only if pid 34176 is gone and the log is not at `done`:

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/launch-sm57-stage2.sh >> C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/gpu/launch-sm57-stage2.nohup 2>&1"'; ProcessStartupInformation=$si}
```

To stop between tickets (and snapshot-stop a running epoch at the next user boundary): `touch C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/gpu/STOP2` (remove it before relaunching). Do not edit `scripts/`, `expertium/` or `rwkv/` in this worktree while it runs (SourceWatch stops the training), and do not edit the running `launch-sm57-stage2.sh`.
