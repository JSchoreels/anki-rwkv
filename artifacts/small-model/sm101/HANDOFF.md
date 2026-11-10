# The smaller instant-only model (SM101): handoff (2026-11-10, prepared, NOT launched)

Design: `DESIGN.md`, in this folder. In short:
- Start from the constant-LR epoch 35 of the show-time instant-only model, stage 2 (SM98), with the card stream cut to
  1 layer (w64-c1).
- Train 3 instant-only epochs (36-38) on show-time inputs, with the recipe of the show-time instant-only model, stage 2
  (SM98).
- Then run the release's one-epoch 32k Muon cool-down on show-time inputs.
- There is an early stop after epoch 36.

Bytes per card + note:
- **Alone:** 4,554 -> 3,134 B (int8). The saving counts only where RWKV-Instant runs alone.
- **Paired with the release (SM63):** no saving today (see `DESIGN.md`).

## Files

- **Scripts.** `launch-sm101.sh` (launcher), `sm101_epoch.py` (epoch prepare / pair), `sm101_pair.py` (cool-down pair),
  `sm101_rule.py` (early + final rule, selftest), `sm101_check.py` (CPU proofs), `sm101_block.py` (breaks),
  `extra-options.json` (the cool-down's; it equals the files of the show-time instant-only model, stage 2 (SM98) and
  the show-time instant-only small model (SM96)).
- **CPU proofs.**
  - `check-cpu-sm101.json` (+ `.out`) is ALL OK:
    - K1 code trees == 1e22b62c;
    - K2 options;
    - K3 epoch pair + simulated chain + 6 negative controls;
    - K4 the cut;
    - K5 the cool-down recipe == the cool-down of the show-time instant-only model, stage 2 (SM98), except the start and
      the architecture;
    - K6 rule selftest + end-to-end runs;
    - K7 breaks;
    - K8 bytes.
  - `check-cut-equivalence.json` (+ `.out`) is the cut tool's generic proofs re-run on this code, ALL OK.
- **Logs.** `gpu/launch-sm101-dryrun.log` ends `DRYRUN OK`. `gpu/launch-sm101-recipe.log` is the recipe proof.
  `gpu/check.out` holds the trainer smoke of epoch 36, which passed (2 updates, 1,024 reviews, 10.5 s, queue ticket of
  11:13).
  - A scorer smoke also passed on 2 users with show-time inputs: the cut at w8+s8 and the uncut epoch 35. It ran from the
    scratchpad and is not committed.
- **Local folders (this computer only, not committed: checkpoints and folders).**
  - `artifacts/small-model/sm101-cut-e35` holds the cut, its AdamW state and `extend.json` (master sha256 716edd0fe143).
  - `sm101-c1-e36-st` is epoch 36, prepared.
  - `sm101-recipe-start-sim` and `sm101-recipe-sim-dec-1ep32k-muon` are the recipe proof (never trained).
  - `sm101/work/` holds the equivalence proof's scratch.
  - If any of these is missing, the launcher stops at its preconditions. To recreate them:
    - the cut: `scripts/sm73s2.py cut --master <the epoch-35 master of the show-time instant-only model, stage 2 (SM98)> --optimizer <its optimizer.pth> --arch w64-c1 --out artifacts/small-model/sm101-cut-e35`;
    - epoch 36: `sm101_epoch.py prepare --epoch 36`;
    - the recipe proof: `PREPARE_RECIPE=1 bash launch-sm101.sh`.

## Launch (NOT done; the caller launches after telling the user)

Run it from `C:/Users/jonat/workspace/wt/sm101`, at the pushed commit of branch `sm101`, hidden via WMI:

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash /c/Users/jonat/workspace/wt/sm101/artifacts/small-model/sm101/launch-sm101.sh >> /c/Users/jonat/workspace/wt/sm101/artifacts/small-model/sm101/gpu/launch-sm101.nohup 2>&1"'; ProcessStartupInformation=$si}
```

After launch, record two pids here: the returned ProcessId (the wrapper) and the launcher's bash pid.

- **Alive check.**
  - `tail -5 artifacts/small-model/sm101/gpu/launch-sm101.log`
  - `powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | ? { $_.CommandLine -like '*launch-sm101*' } | select ProcessId,CommandLine"`
- **Stop.** `touch C:/Users/jonat/workspace/wt/sm101/artifacts/small-model/sm101/gpu/STOP`. This acts between tickets;
  a running epoch or cool-down gets a snapshot stop.
- **Relaunch.** Do this only if the process is gone and the log has no `done`. If the stop was deliberate, remove
  `gpu/STOP` first. Then run the same WMI command. Finished steps are skipped, and an epoch or the cool-down resumes
  from its snapshot.
- **Where the output goes.**
  - main log `gpu/launch-sm101.log` (it ends `done (verdict ...)`, `done (verdict early stop)` or a `STOP:` line);
  - trainer `gpu/train.out`; scorer `gpu/score.out`; smokes `gpu/check.out`; cool-down prepare `gpu/prepare.out`;
  - segments `gpu/segments.jsonl`; tickets `gpu/tickets.jsonl`;
  - early rows `gpu/sm101-early-rows.jsonl`; final rows `gpu/sm101-rows.jsonl` and `gpu/sm101-rows-1500.jsonl`.
- **Queue and GPU.** The queue owner is `sm101`, and purposes start "the smaller instant-only model (SM101): ". The GPU
  cap is 0.8 (trainer options; scorer `HELDOUT_GPU_MEMORY_FRACTION=0.8`). 0.92 is used only for the 3 long users or a
  retry pass that made no progress.
- **Do not edit** while it runs:
  - `scripts/`, `expertium/` or `rwkv/` in `wt/sm101` (SourceWatch);
  - `wt/sm98`, `wt/sm96` or `wt/sm57` at any time. The run reads the epoch 35 and cool-down of the show-time
    instant-only model, stage 2 (SM98) from `wt/sm98`.

## Steps and expected times (if launched ~12:00 Tue 2026-11-10, queue empty)

| step | expected | ends about |
|---|---|---|
| preconditions (CPU) + scoring of the show-time instant-only model, stage 2 (SM98)'s constant-LR epoch 35 on show-time inputs (723 users) | ~14 min | 12:14 |
| epoch 36 pair + trainer smoke | 3 min | 12:17 |
| **epoch 36** (from the cut), 581.9M reviews at ~88-93k reviews/s | ~110 min | ~14:07 |
| scoring of epoch 36 on show-time inputs + **early rule** | ~14 min | **~14:21** (early check ~2.3 GPU h) |
| epochs 37, 38 (each prepared + pair-checked first) | ~2 x 110 min | ~18:01 |
| cool-down prepare + label windows + Muon fresh + `sm101_check.py cooldown` + smoke | ~5 min | ~18:06 |
| **cool-down** segment 1 (snapshot stop ~19:58) | ~112 min | 19:58 |
| break 20:00-01:00 | | 01:00 Wed |
| **cool-down** resume (~320-370 min left at 20-22.5k reviews/s + ~5 min restart) | ~330-375 min | ~06:30-07:15 |
| scoring: show-time 732 (+3 long), normal 732 (+3 long), rule, the 1,500 pool, final rule | ~55 min | **~07:30-08:10 Wed 2026-11-11** |

That is ~14.5 GPU h in all, or about 2.3 h if the early check stops it. A later launch moves the break into the cool-down
or into epoch 37/38 instead; the total stays the same.

## Follow-up (fresh Sonnet agent, medium effort; do not wait on the GPU)

1. Check where the launcher is.
   - `git -C C:/Users/jonat/workspace/wt/sm101 fetch` and fast-forward `sm101`.
   - `tail -8 artifacts/small-model/sm101/gpu/launch-sm101.log`
   - Mid-run progress: `grep -E "complete:|EARLY|STOP|break|rule" gpu/launch-sm101.log | tail`
2. **If the log ends `done (verdict early stop)`,** the early rule said stop (`sm101-early.json` / `.md`). Write
   `sm101-result.md` with:
   - the early numbers;
   - epoch 36's wall minutes and reviews/s (from its `complete:` line and `gpu/segments.jsonl`);
   - the 10-epoch ETA at that speed (8k tier = 581.94M reviews per epoch, 2k tier = 147.6M);
   - nvidia-smi max.

   Then go to step 5. The prepared cool-down is never run.
3. **If the log ends `STOP:`,** read that line and the tails of `gpu/train.out`, `gpu/score.out` and
   `check-cpu-sm101-cooldown.json`. Fix only what is clearly a transient, by relaunching with the WMI command above.
   Report anything else.
4. **If the log ends `done (verdict go|no go)`:**
   - If the rule did not run, run it yourself:
     `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm101/sm101_rule.py decide`
     (CPU, seconds; exit 2 = rows missing).
   - Write `sm101-result.md` in the layout of `artifacts/small-model/sm98/sm98-result.md`, with:
     - the verdict, and A and B with their CIs;
     - the secondary table;
     - the 1,500 pool;
     - the early check;
     - each epoch's and the cool-down's wall minutes and reviews/s;
     - nvidia-smi max;
     - the 10-epoch ETA (8k / 2k tier);
     - bytes alone 4,554 -> 3,134 and the paired case (no saving today).
5. Commit with explicit paths and `git add -f`. What to commit (only those that exist):
   - `sm101-early.json`, `sm101-early.md`, `sm101-decision.json`, `sm101-decision.md`, `sm101-result.md`,
     `check-cpu-sm101-cooldown.json`;
   - `gpu/sm101-early-rows.jsonl`, `gpu/sm101-rows.jsonl`, `gpu/sm101-rows-1500.jsonl`;
   - `gpu/launch-sm101.log`, `gpu/launch-sm101.nohup`, `gpu/check.out`, `gpu/prepare.out`, `gpu/segments.jsonl`,
     `gpu/tickets.jsonl`, `gpu/check-muon-fresh-sm101-c1-e38st-st-dec-1ep32k-muon.json`,
     `gpu/check-replay-label-windows-sm101-c1-e38st-st-dec-1ep32k-muon.json`.

   Never commit checkpoints, snapshots, epoch or cool-down folders, `sm101/work/`, `.claude/`,
   `docs/GPU_OPTIMIZATIONS_LOG.md` or quiet.json.

   Commit as `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`. End the message
   with a blank line + `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

   Right before the push: `git fetch`, merge `origin/sm101`, then push `sm101`. Do not merge into main. Do not touch the
   Anki fork or the workload forecast.
6. Report in plain names with IDs in parentheses, in every sentence (e.g. "the smaller instant-only model (SM101)"), and
   state the units. Never write the outside research group's name.
