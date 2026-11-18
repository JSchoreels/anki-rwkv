# The Tiny two-head model (SM106): handoff (2026-11-11, prepared, NOT launched)

Design: `design.md`, in this folder. In short: the show-time retrain (SM92)'s constant-LR epoch 26, card stream cut to 1 layer (w64-c1), 3 two-head constant-LR epochs (27-29) on show-time inputs with SM92 epoch 26's recipe, then the release's one-epoch 32k Muon cool-down on show-time inputs (SM92's cool-down except the start and the architecture). Early stop after epoch 27. About 15 GPU h; ~2.3 GPU h if the early check stops it.

**Rule A is on point estimates** (changed before launch 11-11 19:25 by the main session, to match the 10-29 pruning bar): curve <= +0.5%, instant <= +0.4% vs the big model (1 leak-free epoch), CIs reported. The show-time retrain (SM92) sits at +0.259% curve, so with the cut cost seen before (~+0.16%) the curve should pass narrowly (~+0.42%).

## Precondition (check by hand first)

Start only after the show-time instant-only big model (SM99) has finished training AND its scoring (expected Thursday 2026-11-12 ~07:00-08:00):
- `tail -3 C:/Users/jonat/workspace/wt/sm99/artifacts/small-model/sm99/gpu/launch-sm99.log` ends `done (verdict ...)`;
- no GPU job runs: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` is empty, and `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status` shows no running ticket;
- not inside a break (20:00-01:00 daily, Sunday from 16:00, `artifacts/coordination/queue/quiet.json`); the launcher waits for the break itself anyway.

The launcher also refuses to start (log `STOP: the show-time instant-only big model (SM99) has not logged ...`) unless SM99's log has a `done (verdict` line. Override only after checking by hand: prefix the command with `SKIP_SM99_GATE=1`.

## Launch (the main session does this)

From `C:/Users/jonat/workspace/wt/sm106`, at the pushed commit of branch `sm106` (`git -C C:/Users/jonat/workspace/wt/sm106 fetch && git -C C:/Users/jonat/workspace/wt/sm106 status` clean, up to date), hidden via WMI:

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash /c/Users/jonat/workspace/wt/sm106/artifacts/small-model/sm106/launch-sm106.sh >> /c/Users/jonat/workspace/wt/sm106/artifacts/small-model/sm106/gpu/launch-sm106.nohup 2>&1"'; ProcessStartupInformation=$si}
```

After launch, record two pids here: the returned ProcessId (the wrapper) and the launcher's bash pid.

- **Alive check.** `tail -5 C:/Users/jonat/workspace/wt/sm106/artifacts/small-model/sm106/gpu/launch-sm106.log`; `powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | ? { $_.CommandLine -like '*launch-sm106*' } | select ProcessId,CommandLine"`
- **Stop.** `touch C:/Users/jonat/workspace/wt/sm106/artifacts/small-model/sm106/gpu/STOP` (between tickets; a running epoch or cool-down gets a snapshot stop).
- **Relaunch.** Only if the process is gone and the log has no `done`. If the stop was deliberate, remove `gpu/STOP` first. Then the same WMI command: finished steps are skipped, an epoch or the cool-down resumes from its snapshot.
- **Queue and GPU.** Queue owner `sm106` (the main checkout's `scripts/r01_queue.py`), purposes start "the Tiny two-head model (SM106): ". GPU cap 0.8 (trainer options; scorer `HELDOUT_GPU_MEMORY_FRACTION=0.8`); 0.92 only for the 3 long users or a retry pass that made no progress.
- **Breaks.** `sm106_block.py`: quiet.json (scope all / gpu) + the daily 20:00-01:00 (Sunday from 16:00). Epochs and the cool-down start when 30 min fit, snapshot-stop <= 2 min before a break and resume after it; scoring tickets must end before the next break.
- **Do not edit** `scripts/`, `expertium/` or `rwkv/` in `wt/sm106` while it runs (SourceWatch). Never touch `wt/sm92` or `wt/sm101` (the run reads SM92's epoch 26 and cool-down from `wt/sm92`, and SM101's recipe proof from `wt/sm101` in the checks).

## Output paths

- Main log `gpu/launch-sm106.log`: ends `done (verdict go|no go)`, `done (verdict early stop)` or a `STOP:` line.
- Trainer `gpu/train.out`; scorer `gpu/score.out`; smokes `gpu/check.out`; cool-down prepare `gpu/prepare.out`; segments `gpu/segments.jsonl`; tickets `gpu/tickets.jsonl`.
- Rows: early `gpu/sm106-early-rows.jsonl` (labels `sm92-e26-const/w8+s8@e2s-query`, `sm106-c1-e27-st-const/w8+s8@e2s-query`); final `gpu/sm106-rows.jsonl` and `gpu/sm106-rows-1500.jsonl` (label `sm106-c1-e29st-st-dec1ep32k/w8+s8@e2s-query`).
- Rules: `sm106-early.json` / `.md`, `sm106-decision.json` / `.md`; cool-down proof `check-cpu-sm106-cooldown.json`.
- Local folders (this computer only, never committed): `artifacts/small-model/sm106-cut-e26` (the cut), `sm106-c1-e27-st` (epoch 27, prepared), `sm106-c1-e28-st`, `sm106-c1-e29-st`, `sm106-c1-e29st-st-dec-1ep32k-muon` (the cool-down; its master is the model), `sm106-recipe-start-sim` + `sm106-recipe-sim-dec-1ep32k-muon` (the recipe proof, never trained). To recreate: the cut `scripts/sm73s2.py cut --master <wt/sm92 .../sm92-e26/expertium/E89-F65e-F77a/master.pth> --optimizer <its optimizer.pth> --arch w64-c1 --out artifacts/small-model/sm106-cut-e26`; epoch 27 `sm106_epoch.py prepare --epoch 27`; the recipe proof `PREPARE_RECIPE=1 bash launch-sm106.sh`; then `sm106_check.py`.

## Steps and expected times (if launched ~08:00 Thu 2026-11-12, queue empty)

| step | expected | ends about |
|---|---|---|
| preconditions + scoring of SM92's constant-LR epoch 26 on show-time inputs (723 users) | ~15 min | 08:15 |
| epoch 27 pair + trainer smoke | 3 min | 08:18 |
| **epoch 27** (from the cut), 581.9M reviews at ~88-92k reviews/s | ~108 min | ~10:06 |
| scoring of epoch 27 on show-time inputs + **early rule** | ~15 min | **~10:21** (early check ~2.3 GPU h) |
| epochs 28, 29 (each prepared + pair-checked first) | ~2 x 110 min | ~14:01 |
| cool-down prepare + label windows + Muon fresh + `sm106_check.py cooldown` + smoke | ~5 min | ~14:06 |
| **cool-down** segment 1 (snapshot stop ~19:58) | ~352 min | 19:58 |
| break 20:00-01:00 | | 01:00 Fri |
| **cool-down** resume (~110-125 min left at ~20.8k reviews/s + ~5 min restart) | ~115-130 min | ~03:00 |
| scoring: show-time 732 (+3 long), rule, the 1,500 pool, final rule | ~70 min | **~04:10 Fri 2026-11-13** |

About 15 GPU h in all. A later launch moves the break into an epoch or into the cool-down; the total stays the same.

## Follow-up (fresh Sonnet agent, medium effort; do not wait on the GPU)

1. `git -C C:/Users/jonat/workspace/wt/sm106 fetch` and fast-forward `sm106`; `tail -8 artifacts/small-model/sm106/gpu/launch-sm106.log`; mid-run `grep -E "complete:|EARLY|STOP|break|rule" artifacts/small-model/sm106/gpu/launch-sm106.log | tail`.
2. **`done (verdict early stop)`:** write `sm106-result.md` with the early numbers (`sm106-early.md`, both heads), epoch 27's wall minutes and reviews/s (its `complete:` line and `gpu/segments.jsonl`), nvidia-smi max, and the 10-epoch ETA (8k tier 581.94M reviews per epoch, 2k tier 147.6M). The prepared cool-down never runs. Go to step 5.
3. **`STOP:`:** read that line and the tails of `gpu/train.out`, `gpu/score.out`, `check-cpu-sm106-cooldown.json`. Relaunch (WMI command above) only for a clear transient; report anything else.
4. **`done (verdict go|no go)`:** if the rule did not run, run it (CPU, seconds; exit 2 = rows missing): `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm106/artifacts/small-model/sm106/sm106_rule.py decide`. Write `sm106-result.md` in the layout of `artifacts/small-model/sm101/sm101-result.md`: the verdict with A on both heads and their CIs; the B table (vs SM92 both heads, vs SM101 instant, vs the release); the 1,500 pool; the early check; each epoch's and the cool-down's wall minutes and reviews/s; nvidia-smi max; total GPU minutes from the queue history; the 10-epoch ETA (8k / 2k tier).
5. Commit with explicit paths and `git add -f` (only those that exist): `sm106-early.json`, `sm106-early.md`, `sm106-decision.json`, `sm106-decision.md`, `sm106-result.md`, `check-cpu-sm106-cooldown.json`, `gpu/sm106-early-rows.jsonl`, `gpu/sm106-rows.jsonl`, `gpu/sm106-rows-1500.jsonl`, `gpu/launch-sm106.log`, `gpu/launch-sm106.nohup`, `gpu/check.out`, `gpu/prepare.out`, `gpu/segments.jsonl`, `gpu/tickets.jsonl`, `gpu/check-muon-fresh-sm106-c1-e29st-st-dec-1ep32k-muon.json`, `gpu/check-replay-label-windows-sm106-c1-e29st-st-dec-1ep32k-muon.json`. Never commit checkpoints, snapshots, epoch or cool-down folders, `.claude/`, `docs/GPU_OPTIMIZATIONS_LOG.md` or quiet.json. Commit as `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line + `Co-Authored-By: <model> <noreply@anthropic.com>`. Right before the push: `git fetch`, merge `origin/sm106`, then push `sm106`. Do not merge into main.
6. Report in plain names with IDs in parentheses in every sentence (e.g. "the Tiny two-head model (SM106)"), results in %, never "points".

## Manual scoring commands (both heads on show-time inputs; the launcher runs these itself)

Each scores both heads (the rows hold `benchmark.ahead` = curve and `benchmark.instant`). Run through the queue from `C:/Users/jonat/workspace/wt/sm106`, with `PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe`, `S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s`, `WT=C:/Users/jonat/workspace/wt/sm106`, `G=$WT/artifacts/small-model/sm106/gpu`, `M=$WT/artifacts/small-model/sm106-c1-e29st-st-dec-1ep32k-muon/expertium/E89-F65e-F77a/master.pth` (absolute paths only: the queue refuses relative path arguments queued from a worktree):

```bash
# the 732 set (723 users), cap 0.8
"$PY" C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py run --owner sm106 --purpose "the Tiny two-head model (SM106): scoring on show-time inputs, 723 users" --claim gpu --expected-minutes 32 -- \
  env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START HELDOUT_GPU_MEMORY_FRACTION=0.8 "$PY" -X utf8 $WT/scripts/small_model_heldout.py --pool 732 \
  --users-file $S72/eval-users-732-fit.json --model sm106-c1-e29st-st-dec1ep32k=expertium:w64-c1:$M --only sm106-c1-e29st-st-dec1ep32k/w8+s8 \
  --inputs e2s-query --time-limit 1800 --split-existing --deleted-cards neighbour --out $G/sm106-rows.jsonl
# the 3 long users: the same with --users-file $S72/users-732-long.json --time-limit 1380 and HELDOUT_GPU_MEMORY_FRACTION=0.92
# the 1,500 pool: the same with --pool 1500 --users-file C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/stage4/users-sm93-score.json --time-limit 2400 --out $G/sm106-rows-1500.jsonl
# then: "$PY" -X utf8 $WT/artifacts/small-model/sm106/sm106_rule.py decide
```

The scorer skips users whose label is already in the `--out` file (error rows are retried), so a pass cut by its time limit is simply run again. `--split-existing --deleted-cards neighbour` is the leak-free scoring (rows restricted to existing / deleted cards are stored too).
