# HANDOFF: leak-free retrain (SM72)

Branch `sm72` (worktree `C:/Users/jonat/workspace/wt/sm72`), cut from `sm71`. Write-up with the check, the estimates, the
big-model facts and the open decisions: `artifacts/small-model/sm72/README.md`. No GPU job was launched; nothing queued.

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
