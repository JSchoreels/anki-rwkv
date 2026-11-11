# The show-time instant-only big model (SM99): handoff (2026-11-11, launched)

Design: `DESIGN.md`, in this folder. In short:
- Start from the show-time big model (SM91 step 2) (read only, in `wt/sm91s2`).
- Train big-model epoch 1 on show-time inputs, instant-only (curve loss weight 0), with the big model's own epoch
  recipe: users 5000-10000, fresh AdamW, warmup 200, LR 3.5e-4 -> 0 over the epoch, 2 x 8,192 carried lanes, no
  teacher, shuffle seed 2. The epoch ends at LR 0, so it is its own cool-down.
- Early rule after epoch 1: continue to epoch 2 (same recipe from epoch 1, seed 3) only if epoch 1 minus the show-time
  instant-only model, stage 2 (SM98) is below 0 on show-time instant. Otherwise the final read-out runs on epoch 1.
- The verdict is a measurement: "ceiling above SM98" (<= -0.05% with the CI below 0, show-time instant, 722 users) means
  a teacher experiment is worth running next; otherwise the small instant-only model is at the big-model ceiling.

## Launch

- Commit 151dd8a2 on branch `sm99`, started 07:04 Wed 2026-11-11 hidden via WMI.
- Pids: the WMI wrapper (returned ProcessId) **33800**; the launcher's bash **24008 (its subshell 32484)**.
- First ticket: the epoch-1 trainer smoke passed 07:04:23 (2 updates, 1,024 reviews, 9.0 s); epoch 1 is queue ticket 001132-sm99, started 07:04:23, first progress lines at ~11.2k reviews/s, ETA ~9.2 h.

Launch / relaunch command (from `C:/Users/jonat/workspace/wt/sm99`, at the pushed commit of `sm99`):

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash /c/Users/jonat/workspace/wt/sm99/artifacts/small-model/sm99/launch-sm99.sh >> /c/Users/jonat/workspace/wt/sm99/artifacts/small-model/sm99/gpu/launch-sm99.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- **Alive check.**
  - `tail -5 C:/Users/jonat/workspace/wt/sm99/artifacts/small-model/sm99/gpu/launch-sm99.log`
  - `powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | ? { $_.CommandLine -like '*launch-sm99*' } | select ProcessId,CommandLine"`
  - Epoch progress: `tail -3 C:/Users/jonat/workspace/wt/sm99/artifacts/small-model/sm99/gpu/train.out`, and the queue:
    `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`
- **Stop.** `touch C:/Users/jonat/workspace/wt/sm99/artifacts/small-model/sm99/gpu/STOP`. This acts between tickets;
  a running epoch gets a snapshot stop at the next user boundary (~1 min). A hard kill loses <= 50 users (~5 min).
- **Relaunch.** Only if the process is gone and the log has no `done`. If the stop was deliberate, remove `gpu/STOP`
  first. Then run the WMI command above. Finished steps are skipped; an epoch resumes from its snapshot (the user in
  flight is retrained from its start). Detached launchers die when the Claude app restarts: check after a restart.
- **Where the output goes** (all under `artifacts/small-model/sm99/`):
  - main log `gpu/launch-sm99.log` (it ends `done (verdict ... / <final label>)` or a `STOP:` line);
  - trainer `gpu/train.out`; scorer `gpu/score.out`; smokes `gpu/check.out`; prepares `gpu/prepare.out`;
  - segments `gpu/segments.jsonl`; tickets `gpu/tickets.jsonl`; pair checks `gpu/pair-e1.json`, `gpu/pair-e2.json`;
  - rows `gpu/sm99-rows.jsonl` (labels `sm99-big-inst-e1/w8+s8[@e2s-query]`, `sm99-big-inst-e2/...`) and
    `gpu/sm99-rows-1500.jsonl` (the final model and `sm91s2-big-showtime/w8+s8@e2s-query`);
  - rules `sm99-early.json` / `.md`, `sm99-decision.json` / `.md`;
  - epoch folders `artifacts/small-model/sm99-big-inst-e1`, `sm99-big-inst-e2` (checkpoints: never committed).
- **Queue and GPU.** Owner `sm99`; every purpose starts "the show-time instant-only big model (SM99): ". GPU cap 0.8
  (trainer options; scorer `HELDOUT_GPU_MEMORY_FRACTION=0.8`); 0.92 only for the 3 long users or a retry pass that made
  no progress.
- **Breaks.** `sm99_block.py` (quiet.json + the daily 20:00-01:00, Sunday from 16:00). The epoch snapshot-stops <= 2 min
  before a break and resumes after it; scoring tickets wait for a slot that ends before the break.
- **Do not edit** while it runs: `scripts/`, `expertium/` or `rwkv/` in `wt/sm99` (SourceWatch), or anything in
  `wt/sm91s2` (the start checkpoint and the reference folder).

## Checks done before launch

- CPU proofs `check-cpu-sm99.json` (+ `.out`): ALL OK (K1 code trees, K2 epoch-1 pair + 11 negatives, K3a/b/c trainer slice, K4 show-time samples, K5 scorer, K6 rule selftest, K7 breaks).
- GPU pair check `gpu/check-gpu-sm99.json` (+ `gpu/check-gpu.out`): ALL OK on run 2 (G1 SM91 step 2 epoch smoke bitwise on both trees, G2 weight 0 recorded and effective, G3 GPU scorer rows; run 1 failed only G3, whose test then compared raw float means exactly: one user differed by 1.5e-8 from GPU reduction order, benchmark blocks identical; kept as `gpu/check-gpu-sm99-run1.json`).
- Label windows: `gpu/check-label-windows-sm99-e1.json` (27 tails trimmed, pass). Pair: `gpu/pair-e1.json` OK.
- Dry run: `gpu/launch-sm99-dryrun.log` ends `DRYRUN OK`.

## Timeline (launched 07:04 Wed 2026-11-11, queue empty)

| step | expected | ends about |
|---|---|---|
| preconditions (CPU, rule selftest) + epoch 1 smoke | ~5 min | 07:05 |
| **epoch 1**, 370.3M reviews at ~11.4k reviews/s | ~545 min | ~16:20 Wed |
| epoch 1 show-time scoring (723 users, ~17 min) + **early rule** | ~20 min | ~16:40 Wed |
| if continue: epoch 2 prepare + pair + smoke, first segment to the 19:58 snapshot stop | | 19:58 Wed |
| break 20:00-01:00 | | 01:00 Thu |
| epoch 2 resumed (~10 min restart + the rest) | | ~07:05 Thu |
| final scoring: show-time + normal (723 + 3 long), rule, the 1,500 pool for the final model and for the show-time big model (SM91 step 2), final rule | ~80 min | **~08:30 Thu 2026-11-12** |
| if the early rule stops: epoch 1's final scoring right after the early rule | ~80 min | ~18:00 Wed |

About 20.3 GPU h in all (10.6 h if the early rule stops).

## Follow-up (fresh Sonnet agent, medium effort; do not wait on the GPU)

1. Check where the launcher is.
   - `git -C C:/Users/jonat/workspace/wt/sm99 fetch` and fast-forward `sm99`.
   - `tail -8 C:/Users/jonat/workspace/wt/sm99/artifacts/small-model/sm99/gpu/launch-sm99.log`
   - Mid-run: `grep -E "complete:|EARLY|STOP|break|rule|segment" gpu/launch-sm99.log | tail`
2. **If the log ends `STOP:`,** read that line and the tails of `gpu/train.out`, `gpu/score.out`, `gpu/check.out`.
   Fix only what is clearly a transient (a reboot, a killed process) by relaunching with the WMI command above. Report
   anything else.
3. **If the log ends `done (verdict ...)`:**
   - If a rule did not run, run it yourself (CPU, seconds; exit 2 = rows missing):
     `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm99/sm99_rule.py decide`
   - Write `artifacts/small-model/sm99/sm99-result.md` in the layout of
     `artifacts/small-model/sm101/sm101-result.md`, with:
     - the verdict and the primary (the show-time instant-only big model (SM99) minus the show-time instant-only
       model, stage 2 (SM98), show-time instant) with its CI, and right after it the show-time instant-only big model
       (SM99) minus the show-time big model (SM91 step 2);
     - the read-out table and the 1,500 pool, from `sm99-decision.md`;
     - the early rule's numbers (`sm99-early.md`) and which epoch is the final model;
     - each epoch's wall minutes (from `gpu/segments.jsonl`, all segments added) and reviews/s (from its `complete:`
       line), the process-tree RSS max, and nvidia-smi max;
     - the 10-epoch ETA at the measured speed (8k tier = 581.94M reviews per epoch, 2k tier = 147.6M);
     - the total GPU time (`gpu/tickets.jsonl` + `gpu/segments.jsonl`).
4. Commit with explicit paths and `git add -f`. What to commit (only those that exist):
   - `sm99-early.json`, `sm99-early.md`, `sm99-decision.json`, `sm99-decision.md`, `sm99-result.md`;
   - `gpu/sm99-rows.jsonl`, `gpu/sm99-rows-1500.jsonl`, `gpu/pair-e2.json`, `gpu/check-label-windows-sm99-e2.json`;
   - `gpu/launch-sm99.log`, `gpu/launch-sm99.nohup`, `gpu/check.out`, `gpu/prepare.out`, `gpu/segments.jsonl`,
     `gpu/tickets.jsonl`, `gpu/score.out`.

   Never commit checkpoints, snapshots, the epoch folders, `sm99/work/`, `artifacts/small-model/smoke/`, `.claude/`,
   `docs/GPU_OPTIMIZATIONS_LOG.md` or quiet.json. `gpu/train.out` is large: commit only its last 200 lines as
   `gpu/train-tail.out`.

   Commit as `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`. End the message
   with a blank line + `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

   Right before the push: `git fetch`, merge `origin/sm99`, then push `sm99`. Do not merge into main. Do not touch the
   Anki fork, the workload forecast, or `pages/data/factory-curated.json` (the user updates it).
5. Report in plain names with IDs in parentheses, in every sentence (e.g. "the show-time instant-only big model
   (SM99)"), results as %, never "points". Quote reviews/s per epoch. Never write the outside research group's name.
