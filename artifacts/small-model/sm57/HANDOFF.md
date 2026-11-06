# RUNNING - The instant-only small model (SM57): handoff (2026-11-06 01:56)

Branch `sm57` (cut from `sm92` at `29727c6b`: main has no small-model training or scoring code), worktree `C:/Users/jonat/workspace/wt/sm57`. Design, option and pre-registered rule: `DESIGN.md` (this folder).

## What it asks

Does giving the whole small model to the instant task beat sharing it with the curve? Two arms from the leak-free small model (SM72s) epoch 25 (the release's start), identical except the loss: epoch 26 (the release's epoch 21-25 recipe) + the release's exact one-epoch 32k Muon cool-down (same users / order / seeds / KD replay targets), normal inputs.

- **instant-only**: track option `"curve_loss_weight": 0.0` (every curve loss term, curve KD included, at weight 0) in the epoch and the cool-down.
- **joint**: no extra option (the normal loss). Its epoch 26 is also epoch 26 of the "more epochs" run of the release from epoch 25 (SM63): keep `artifacts/small-model/sm57-joint-e26/expertium/E89-F65e-F77a/{master,optimizer}.pth`.

**Rule** (`sm57_rule.py`, fixed before launch): 722 users of the 732 set, normal inputs, leak-free, instant-only - joint on the instant head, points = % of the big model (1 leak-free epoch) LogLoss, paired user bootstrap 2,000, seed 0, 95% CI. **go** if the CI is entirely below zero and the gain is at least 0.05 points (minimum added by the user 11-07 ~00:40, before the joint arm was scored; self-test re-run OK) (then propose stage 2, do not run it); **stop** otherwise.

## Done before launch

- Option `curve_loss_weight` (default 1.0) in `expertium/model/variant_model.py`, wired in `scripts/gpu_r01_training.py` (recorded in result.json), allowlisted in `scripts/gpu_r01_prefetch.py`, `TEACHER_BLIND` in `scripts/expertium_options.py` (commit `fd43ae79`).
- CPU proofs `check-cpu-sm57.json` on `fd43ae79`: **ALL OK** (33 checks, 194 s). Default training is bit-identical: the loss code before SM57 == option untouched == weight 1.0 (loss, per-review losses, every gradient; eager and TorchScript; with and without KD + sync-free loss), and a CPU trainer slice (8 updates, batch path with state-carry lanes, clip, AdamW and fresh Muon, TorchScript) gives bit-identical per-update losses, gradients and final weights in the `wt/sm92` tree and this tree. Weight 0.0: loss == instant average exactly; only `head_w.*` and `gru_*` (the curve head) get newly zero gradients.
- GPU check `gpu/check-gpu-sm57.json` (ticket 000937, 01:52:59-01:54:17): **ALL OK**: the joint epoch-26 trainer smoke run by the `wt/sm92` tree and by this tree gives the same weights bit for bit on GPU; the instant-only smoke trains with `curve_loss_weight` 0.0 recorded.
- Epoch folders prepared (not committed): `artifacts/small-model/sm57-inst-e26`, `artifacts/small-model/sm57-joint-e26` (8,000 users, 581,942,724 reviews, update offset 551,950).

## What runs (detached, WMI hidden)

Launcher `launch-sm57.sh` (commit `163c66d6`), WMI pid **21128** (bash.exe), log `gpu/launch-sm57.log`, nohup `gpu/launch-sm57.nohup`, launched 2026-11-06 01:52. Queue owner `sm57`; every ticket purpose starts with "the instant-only small model (SM57): ". GPU memory seen at the start of the instant-only epoch: 14,640 MiB nvidia-smi total with a ~2,600 MiB desktop baseline (~12 GB training, cap 0.8 in the options); ~80-100k reviews/s in the first minutes.

| step | ticket | expected (if the queue stays free) |
|---|---|---|
| GPU check | 000937 | **passed** 01:53-01:54 |
| instant-only epoch 26 | 000938 (running) | 01:54 -> ~03:50 |
| joint epoch 26 | next | ~03:50 -> ~05:45 |
| CPU: both cool-downs prepared + label windows + Muon fresh checks + `sm57_pair.py decay` (each vs the release) + `arms` | - | ~05:45-06:00 |
| instant-only cool-down smoke (2 users x 512 reviews) | next | ~06:00 |
| instant-only cool-down (~480 min wall in SM92) | next | ~06:00 -> ~14:00 |
| scoring the instant-only model (723 users cap 0.8 + 3 long users cap 0.92), then a partial rule run (instant-only vs the release) | next | ~14:00-14:20 |
| joint cool-down | next | ~14:20 -> **snapshot stop ~19:58**, break 20:00-01:00, resume 01:00 11-07 -> ~03:20 |
| scoring the joint model, then the rule -> `sm57-decision.{json,md}`, log `rule (final) exit 0: [sm57-rule] ...`, then `done` | - | ~03:20-03:45 on 11-07 |

Folders: `artifacts/small-model/sm57-{inst,joint}-e26` (epochs), `artifacts/small-model/sm57-{inst,joint}-e26-dec-1ep32k-muon` (cool-downs). Labels `sm57-inst-dec1ep32k/w8+s8`, `sm57-joint-dec1ep32k/w8+s8`, rows `gpu/sm57-rows.jsonl`. Per segment: `gpu/segments.jsonl` (wall minutes, exit, nvidia-smi baseline / max MiB). Breaks: quiet.json (scope all/gpu) + the daily 20:00-01:00, re-read at every check; epochs and cool-downs snapshot-stop <= 2 min before a window and resume after it.

## Follow-up (fresh agent, Sonnet medium is enough)

Do not wait on the GPU: check, act, stop. Always `git fetch && git merge --ff-only` first.

```bash
cd C:/Users/jonat/workspace/wt/sm57 && git fetch && git merge --ff-only
tail -5 artifacts/small-model/sm57/gpu/launch-sm57.log
tasklist //FI "PID eq 21128"                                   # launcher alive?
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status
cat artifacts/small-model/sm57/sm57-decision.md                # after "rule (partial)" or "done"
# re-run the rule by hand (CPU, seconds; exit 2 = joint rows not there yet, the file still has the partial tables):
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm57/sm57_rule.py decide
```

1. **After the instant-only scoring** (log line `rule (partial) exit 2: ...`): optional preliminary report of instant-only - the release (instant, points with CI). Not the rule.
2. **When the log ends with `done`** (or `STOP: ...`: read the line before it, `gpu/train.out` / `gpu/score.out` tails): run the rule (above).
3. Commit with `git add -f` (as `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`): `gpu/launch-sm57.log`, `gpu/launch-sm57.nohup`, `gpu/segments.jsonl`, `gpu/check-*.json`, `gpu/check.out`, `gpu/prepare.out`, `gpu/score.out`, `gpu/train.out` (if < 20 MB), `gpu/sm57-rows.jsonl`, `sm57-decision.{json,md}`; mark this file DONE; then `git push`. Never commit checkpoints, snapshots, caches, `.claude/`, `pages/`, `docs/GPU_OPTIMIZATIONS_LOG.md` or quiet.json. Do not edit `pages/data/factory-curated.json` (the main session does the Factory page).
4. Report (plain names with IDs in parentheses, never a bare ID): the verdict line; instant-only - joint on instant (points, CI, users better / worse) and whether >= 0.05 points; instant-only - the release (instant); joint - the release on both heads (the one-more-epoch confound) and the joint arm's curve; both vs the big model (1 leak-free epoch) and the smaller-model bar; 732+long; per arm and stage: training minutes, reviews/s (the launcher's `complete:` lines cover the last segment only, so also give the wall clock from `gpu/segments.jsonl`), peak GPU memory (result.json peak allocated / reserved, nvidia-smi max minus baseline); the 8k / 2k 10-epoch ETA from each arm's epoch speed. If **go**: propose stage 2 (from-scratch instant-only vs joint at small scale, e.g. a few epochs at 2,048 windows, same seeds; not run). Either way the joint epoch-26 checkpoint is ready for the "more epochs" run of the release from epoch 25 (SM63).

## Relaunch (resumes: finished steps are skipped, an interrupted epoch / cool-down resumes from its snapshot)

Only if pid 21128 is gone and the log is not at `done`:

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/launch-sm57.sh >> C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/gpu/launch-sm57.nohup 2>&1"'; ProcessStartupInformation=$si}
```

To stop between tickets (and snapshot-stop a running epoch / cool-down at the next user boundary): `touch C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/gpu/STOP` (remove it before relaunching). Do not edit `scripts/`, `expertium/` or `rwkv/` in this worktree while it runs (SourceWatch stops the training), and do not edit the running `launch-sm57.sh`.
