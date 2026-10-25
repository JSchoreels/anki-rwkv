# HANDOFF: leak-free retrain (SM72)

Branch `sm72` (worktree `C:/Users/jonat/workspace/wt/sm72`), cut from `sm71`. Write-up with the check, the estimates, the
big-model facts and the open decisions: `artifacts/small-model/sm72/README.md`. No GPU job was launched; nothing queued.

## State 2026-10-25 ~10:50: step 1 switched to carried state, LR 3.5e-4; ready, NOT launched

User 10-25 ~10:05: peak LR 3.5e-4 (not 7e-4), and "are we not doing a state save/load between those windows?" -> the
epoch now trains with the full retrain's carried state (strided lanes, `"state_carry": {"lanes": "strided"}`), so a
window's start continues its lane's previous window instead of starting fresh.

- Code: `expertium/variants.py` preset `"fork"` = the fork's `rwkv.architecture` config in the variant class, with the
  new `VariantConfig.stream_major` (default False = unchanged): the sequential (stream-major) layer order run through
  the canonical gather/scatter path `_interleaved_streams`, where `scripts/state_carry_train.StateCarry` hooks in.
  `state_carry_train._identity` falls back to `features2card.0.weight` through the C++ module when a scripted wrapper
  lists no root-level parameter (the fork has none; w64's GRU head has some, so w64 is unchanged).
  `scripts/sm72_big_epoch.py prepare` defaults: carried state, `"architecture": "fork"`, SM56's decay R01 options
  (minus teacher and Muon; width-32-only switches dropped by `compatible_options`, none in that set), `--peak 3.5e-4`;
  `--fresh-state` = the earlier recipe (baseline class, fresh sections).
- Checks (CPU): scorer rows for `fork::<shipped>` and `expertium:fork:<shipped>` identical on users 4676 and 516 (fp32;
  `sm72/carry-check/fork-vs-variant.jsonl`). `scripts/check_state_carry_lanes.py --skip-base --architecture fork
  --checkpoint <shipped>` ALL OK (`sm72/check-state-carry-lanes-fork.json`): L layout, B first update = stock bitwise,
  D exact carry (seeded lane starts vs full history: max |dp| 8e-7, |dw| 7e-7; lane 0 with zero starts <= 5e-7),
  F user boundary, G scripted model (after the `_identity` fix; w64's G re-run: same numbers as before). The check
  script's `clone_slots` / `seed_lane_starts` now skip SM32c's non-tensor `rank1` slot entry (it failed for any model).
- GPU probe (`sm72/probe-big-carry-sm72.json`, 4 longest users x 131,072 reviews, 32 updates each):

  | Layout (carried) | Reviews/s | Peak alloc | Mean loss | Epoch |
  |---|---|---|---|---|
  | 1 lane x 16,384 | 7,287 | 10.1 GB | 0.6283 | 14.1 h |
  | 2 lanes x 8,192 (chosen) | 10,821 | 10.2 GB | 0.6278 | 9.5 h |
  | 4 lanes x 4,096 | 14,121 | 11.1 GB | 0.6302 | 7.3 h |
  | 8 lanes x 2,048 | 16,796 | 11.7 GB | 0.6358 | 6.1 h |
  | 16 lanes x 2,048 | out of memory (0.8 cap) | | | |

  Lane starts begin from zero state, so L lanes = L fresh starts per user: the loss rises with the lane count (8 lanes
  is as bad as the fresh-state 1 x 16,384 run, 0.6377). 2 x 8,192 matches the whole-history single lane at 2/3 of its
  time. Faster alternative: 4 x 4,096 (7.3 h, +0.002 loss in the probe).
- Prepared run (re-prepared; the earlier folder had never run): `artifacts/small-model/sm72-big-leakfree-1ep`,
  `"model": "expertium"`, 2 x 8,192 carried, 3.5e-4 -> 0, warmup 200 updates (~22,600 updates), 5,001 users, 370.3M reviews.
  Checkpoint: `sm72-big-leakfree-1ep/expertium/big-leakfree-1ep/master.pth` (same state-dict keys and dtypes as the
  shipped .pth, checked on a probe checkpoint), so the scorer (`fork::`) and the exporter (`--teacher`) read it as before.
  To switch to 4 x 4,096: `rm -rf` that folder (only if it never ran) and
  `RWKV_DELETED_CARDS=neighbour $PY -X utf8 scripts/sm72_big_epoch.py prepare --out artifacts/small-model/sm72-big-leakfree-1ep --chunk-reviews 4096 --batch-chunks 4`,
  then `RATE=14000` in the launcher. Fallback if carry misbehaves: `... prepare --out ... --fresh-state --chunk-reviews 16384 --batch-chunks 1` (13.6 h, `RATE=7500`).
- Launcher: `RATE=10800`; the checkpoint path follows decay.json's model. Dry run 10:46: first training segment expected
  188 min, ending before the Sunday quiet window (14:00 -> Mon 02:00, quiet.json); a snapshot stop is requested 2 min
  before it (at the next user boundary), the next segment resumes at 02:00.
- Launch command: unchanged (below).
- ETA if SM71 ends ~11:30: train 11:30-14:00 (~95M reviews), 02:00 -> Mon ~09:15 (~275M), scoring ~1 h -> ~10:15,
  teacher export ~4.8 h -> Mon ~15:15.

## Earlier state 2026-10-25 ~10:00 (superseded: fresh-state sections, 7e-4; launch command still valid)

Plan (user 10-24 ~20:40): (1) the big model gets ONE leak-free epoch from its shipped weights, (2) leak-free teacher
targets are exported once from that model, (3) the small model is retrained leak-free on them (sections below).

- `scripts/sm72_big_epoch.py` (4c0c63f4): `check` / `prepare` / `run` / `smoke` / `stop` / `save`. The fork's
  architecture is the R01 trainer's "baseline" model (`SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)`, the scorer's fork class); the
  run folder is a `scripts/expertium_decay.py` folder with `"model": "baseline"` (that script now reads the model from
  decay.json; every earlier decay says expertium), so the epoch is resumable from snapshots every 400 users.
- CPU check PASS (`artifacts/small-model/sm72/check-big-load-sm72.json`): 2,762,884 parameters, the shipped state dict
  loads into the trainer's model bit for bit, and the scorer gives identical rows (fp32 and w8+s8, users 2388 and 4676)
  for the shipped file and the trainer's re-saved copy.
- Prepared run: `artifacts/small-model/sm72-big-leakfree-1ep` (users 5000-10000 shuffled seed 1, 5,001 users, 370,299,909
  reviews, fresh AdamW, LR 7e-4 x linear warmup 200 updates x one_minus_sin (upstream TRAIN_MODE "D") to 0 over the
  epoch's reviews, 1 x 16,384 reviews per update = 25,065 updates, cap 0.8, RWKV_DELETED_CARDS=neighbour).
- GPU probes (`artifacts/small-model/sm72/probe-big-sm72.json`; the 4 longest users x 131,072 reviews): 2 x 32,768 and
  1 x 32,768 per update run out of memory at the 0.8 cap (12.7 GB); 1 x 16,384: 7,572 reviews/s, 9.0 GB, mean loss
  0.6377; 2 x 8,192: 12,248/s, 0.6438; 4 x 4,096: 16,051/s, 0.6508; 8 x 2,048: 17,781/s, 0.6604. Each window is one
  sequence from a fresh state and the recurrence is sequential in the window length: longer windows are slower but
  closer to upstream (32,768-review sections) and to scoring (whole history); the loss rises as windows shorten.
  At 1 x 16,384 the epoch is **~13.6 h GPU**, not the 3.6 h estimated from R01's 128-review windows. Faster choice:
  `--batch-chunks 2 --chunk-reviews 8192` (~8.4 h). To switch: delete the prepared folder, re-run `prepare`
  (with `RWKV_DELETED_CARDS=neighbour`) with the new values, and set `RATE` in the launcher.
- Launcher `artifacts/small-model/sm72/launch-sm72-big.sh` (DRYRUN=1 checked): waits for SM71's runbook "done" (or its
  process gone); trains in segments, each queue ticket ending at the next quiet window (a watcher requests a snapshot
  stop 2 min before or inside a window; the next segment resumes after it); then scores `big-leakfree-1ep/w8+s8` with
  `--split-existing`, keep and neighbour inputs, 464 set, 732 pool, 3 long users (0.92 cap), into
  `artifacts/small-model/sm72/gpu-big/bigleak-{keep,neighbour}.jsonl` (scorers run with `env -u RWKV_DELETED_CARDS`);
  then exports the teacher targets from the new `master.pth` (`--teacher`) into the NEW folder
  `C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/big-leakfree-1ep-final8k-neighbour` (the 8,000
  final-tier users of `sm55-final8k-run.json`, ~4.8 h, tickets cut at windows with `--time-limit`), status, CPU verify.
  Log `artifacts/small-model/sm72/gpu-big/launch-sm72-big.log`; stop: `touch artifacts/small-model/sm72/gpu-big/STOP`.
- Launch (hidden, survives session restarts), PowerShell:
  ```
  $si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
  Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/launch-sm72-big.sh >> C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/launch-sm72-big.nohup 2>&1"'; ProcessStartupInformation=$si}
  ```
- ETA at 1 x 16,384 if SM71 ends ~11:30: training 11:30-16:00 (~4.5 h), break, 02:00-~11:10 Mon; scoring ~1 h;
  export ~4.8 h -> Mon ~17:00. Step 3 (small model) then needs the distillation `teacher` set to the new master.pth
  (ReplayTargets checks the teacher sha256 against the manifest) and `replay` to the new folder.
- Checks after training: its all-card score should meet its existing-card score (neighbour inputs);
  `scripts/sm71_report.py` against SM71's `fork` rows.

## State (2026-10-24 ~20:40)
- Code done, default off: `RWKV_DELETED_CARDS=neighbour|flags` in the environment makes `rwkv/data_processing.get_rwkv_data`
  apply `scripts/sm71_leak.neutralise` to every frame (trainer, its workers, replay exporter, upstream LMDB builder).
  Replay manifests record it; the trainer refuses replay targets whose `deleted_cards` differs from its environment.
- CPU check `scripts/sm72_check.py` on users 4676, 516, 4789, 1058: ALL PASS (`artifacts/small-model/sm72/check-inputs-sm72.json`).
- There is no training LMDB in the SM56 recipe (inputs are built on the fly); nothing to rebuild for inputs.
- Waiting on: the user's choice of teacher targets (README "Open decisions" 1). Default plan below = option (a).

## Next steps (in order; each GPU step is one queue ticket, owner `small-model`, respect quiet.json and the 21:00-01:30 break)
Common: `PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe`, `WT=C:/Users/jonat/workspace/wt/sm72`,
`RUN8K=C:/Users/jonat/workspace/wt/sm55/artifacts/small-model/sm55-final8k-run.json` (the 8,000 final-tier users, the run
used for `fork-fp32-final8k`), `R8N=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-final8k-neighbour`
(NEW folder; never write into `fork-fp32-final8k`).

1. Leak-free teacher targets (GPU ~4.8 h at the measured 34k reviews/s; resumable per user, use `--time-limit` per ticket):
   ```
   cd $WT && RWKV_DELETED_CARDS=neighbour "$PY" -X utf8 scripts/sm11_replay_export.py export --out "$R8N" --run "$RUN8K" --device cuda --time-limit 14400
   RWKV_DELETED_CARDS=neighbour "$PY" -X utf8 scripts/sm11_replay_export.py status --out "$R8N" --run "$RUN8K"   # exit 0 = complete
   RWKV_DELETED_CARDS=neighbour "$PY" -X utf8 scripts/sm11_replay_export.py verify --out "$R8N" --run "$RUN8K" --sample 8   # CPU
   ```
   Check `$R8N/manifest.json` has `"deleted_cards": "neighbour"`. Untested saving: users with no deleted card have identical
   frames, so their `.npz` could be copied from `fork-fp32-final8k` instead of recomputed (needs a byte-equality spot check).
2. Runbook to write: `artifacts/small-model/sm72/launch-sm72.sh`, a copy of `wt/sm56/artifacts/small-model/sm56/launch-sm56.sh`
   (read only; copy, never edit the sm56 one) with:
   - `export RWKV_DELETED_CARDS=neighbour` at the top for training/decay/export/check processes, and `env -u RWKV_DELETED_CARDS`
     for every scorer call;
   - `WT=$WT`, `S=$SM/sm72`, `NAME=scale-w64kd-final-sc-lanes-leakfree`, `REPLAY8K=$R8N`, `LBL=s-w64kd-final-sc-lanes-leakfree`,
     outputs `heldout-sm72.jsonl` / `heldout-sm72-pools.jsonl`, `MAX_EPOCHS=10`, fixed 10 epochs (no plateau stop; keep the
     per-epoch 464 readout and the rule output for the record);
   - every scorer call (`scripts/small_model_heldout.py`) with `--deleted-cards neighbour` (and `--split-existing`), as SM71;
   - the same `accuracy_loop prepare` (options = sc-lanes-cont's overrides with `distillation.replay` -> `$R8N`, tier final, seed
     2025, order seed 20261013, `--epoch-runner scripts.sm56_epochs:run_epoch`), then `scripts/sm56_epochs.py check --experiment
     $EXP` (label-window trims against the new replay; CPU 10-20 min);
   - decay from e10: `scripts/expertium_decay.py prepare --source-run $EXP/run --epoch 10 ... --extra-options "$EXTRA"` with
     `$EXTRA` = SM56's (replay -> `$R8N`), the label-window and Muon fresh-state checks, then `smoke` + `run` (resumable decay:
     snapshots every 400 users; pause/resume runbooks in sm56).
   - Dry run first: `DRYRUN=1 bash $WT/artifacts/small-model/sm72/launch-sm72.sh`; start hidden via WMI Win32_Process.Create
     (ShowWindow=0), not nohup (see memory: detached jobs die on session restart).
3. GPU budget: export 4.8 h + 10 x 118 min + 10 x 12 min scorings + decay 8.5 h + 732-pool scorings ~0.4 h = about 35 h.
4. Comparison: the big model leak-free rows come from SM71's runbook (`wt/sm71/artifacts/small-model/sm71/gpu/big-neighbour.jsonl`,
   464 set and 732 pool). Read-out with `scripts/sm71_report.py --series "big/neighbour=...::fork/w8+s8" --series
   "sm72/neighbour=<heldout-sm72-pools.jsonl>::<LBL>-dec10.../w8+s8"` and SM56 neighbour (`small-neighbour.jsonl`) as the leaky-trained control.

## Do not
- Touch `wt/sm56` (training running) or the fork repo `C:/Users/jonat/workspace/anki` (read only).
- Set `RWKV_DELETED_CARDS` in scorer processes, or point a leak-free run at `fork-fp32-final8k` (the trainer refuses it).
