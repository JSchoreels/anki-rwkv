# HANDOFF: SM56 (full retrain on 8,000 users from scratch, adaptive epochs), prepared 2026-10-22

Worktree `C:/Users/jonat/workspace/wt/sm56` (branch `sm56`, cut from `sm55`). Not launched by the preparing agent (coordinator, 10-22): the main session starts it.

## Launch (main session)
- Command: `nohup bash C:/Users/jonat/workspace/wt/sm56/artifacts/small-model/sm56/launch-sm56.sh >> C:/Users/jonat/workspace/wt/sm56/artifacts/small-model/sm56/launch-sm56.log 2>&1 & disown`
- Bash: `C:/Program Files/Git/usr/bin/bash.exe`. Log: `C:/Users/jonat/workspace/wt/sm56/artifacts/small-model/sm56/launch-sm56.log`.
- Dry run: `DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm56/artifacts/small-model/sm56/launch-sm56.sh` -> `sm56/launch-sm56-dryrun.log`, ends `DRYRUN OK`.
- Resume after a stop or crash: the same launch command (finished/scored epochs, the finished decay and scored users are skipped; decisions are recomputed from `sm56/epochs.jsonl`). An epoch interrupted mid-way restarts that epoch (accuracy_loop moves the partial folder to `epoch-KK-attempt-N`).
- Stop between tickets: `touch C:/Users/jonat/workspace/wt/sm56/artifacts/small-model/sm56/STOP`. Do not edit `scripts/ expertium/ rwkv/` in wt/sm56 while it runs.
- Summary again: `SUMMARY_ONLY=1 bash C:/Users/jonat/workspace/wt/sm56/artifacts/small-model/sm56/launch-sm56.sh`.

## Recipe and lineage
SM55's lineage (5,000 gate users, seed 2025, order seed 20261013, AdamW R01 schedule peak 7e-4 / 5-update warmup / constant, w64, KD alpha 0.9 decay_alpha 0.5):

| Epochs | Experiment (worktree) | Updates | Carry | KD targets | 5k epoch time |
|---|---|---|---|---|---|
| 1-6 | scale-w64kd-gate-cached-ctx2048 (small-model) | 2,048 x 16 | none | cached bf16 teacher (`kd-teacher/fork-bf16-s2025`) | 3,205-4,570 s |
| 7-8 | scale-w64kd-gate-cached-ctx8192-cont | 8,192 x 4 | none | same cache | 7,351-11,359 s |
| 9 | ...-ctx16384-cont | 16,384 x 2 | none | same cache | 16,025 s |
| 10 | ...-ctx32768-cont | 32,768 x 1 | none | same cache | 25,135 s |
| 11-12 | scale-w64kd-gate-sc-lanes-cont (sm21v2) | 2,048 x 16 | strided lanes | replay `fork-fp32-gate5k` | 4,166-4,176 s (88.5k reviews/s) |
| decay | SM55: 1 epoch 32,768 x 1, Muon fresh, 8k users, replay `fork-fp32-final8k` | | lanes | | 495 min (8k) |

SM56 (experiment `artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes`): epochs 11-12's recipe for EVERY epoch, from scratch, on the 8,000 final-tier users: options = sc-lanes-cont's options with `distillation.replay` -> `fork-fp32-final8k` (checked key by key), training seed 2025, order seed 20261013, AdamW 7e-4 constant (5-update warmup at epoch 1), 2,048 x 16, strided-lane carry, gpu_memory_fraction 0.8.
Differences from the lineage, chosen as the simplest faithful option (the user's "carried state at 2,048 every epoch"): (1) no growing windows (epochs 7-10 at 8k/16k/32k x 1 without carry are replaced by 2,048 x 16 with carry; the carry gives every window its full-history state, which is what the long windows approximated, at ~3-6x lower epoch cost); (2) carry from epoch 1 instead of epoch 11; (3) full-history replay KD targets from epoch 1 instead of the windowed bf16 cache (the cache is not built for the 3,000 extra users; replay covers all 8,000); (4) 8,000 users per epoch instead of 5,000. Nothing checked shows this cannot work: the largest users of the 8k tier are the same as in the 5k tier (6810: 3.9M reviews), which sc-lanes-cont trained at this layout under the 0.8 cap.
Label windows: in this layout user 5599 (32,782 reviews) has an unlabelled last update (its 14-review final window, first reviews only), as in SM55's decay. The epoch hook `scripts/sm56_epochs.py` (`--epoch-runner scripts.sm56_epochs:run_epoch`) cuts it from the train_range ([0, 32768]); recorded in `label-window-trims.json` and in every epoch's protocol.json (`label_window_trims`).

## Ticket plan (all queue tickets: owner small-model, claim gpu)
| Step | Expected min | Output |
|---|---:|---|
| Epoch k (k = 1..14), one ticket each; epoch 1: GPU smoke first (`sm56/smoke-prep`) | 120 for epoch 1 (sc-lanes-cont 4,184 s x 1.575 = 110 + smoke/setup); later: 1.05 x the slowest of the last 3 measured walls + 3 | `.../scale-w64kd-final-sc-lanes/run/epoch-KK/` (master.pth, optimizer.pth, result.json), `phases.jsonl` |
| 464-set scoring after each epoch (458 fit users, w8+s8, cap 0.8), pass 2 only if users are missing | 12 (+8) | `artifacts/small-model/heldout-sm56.jsonl`, label `s-w64kd-final-sc-lanes-eK/w8+s8`; losses `sm56/epochs.jsonl` |
| Decay from the stopping epoch K (SM55's setup exactly, smoke first) | 510 (SM55: 495 min training) | `artifacts/small-model/scale-w64kd-final-sc-lanes-decK-1ep32k-muon/` |
| 732 pool minus 9 never-fit (723; contains the 464 set) | 20 (+12) | `artifacts/small-model/heldout-sm56-pools.jsonl`, label `s-w64kd-final-sc-lanes-decK-1ep32k-muon/w8+s8` |
| Summary (CPU) vs SM55 and SM36, 464 and 732, paired user bootstrap 2,000 seed 0, vs fork | - | `artifacts/small-model/sm56-summary.{md,json}` |

Stopping rule (user 10-22, 2 consecutive plateau epochs; `sm56/sm56_rule.py`, self-tested at every start): per head gain_k = L_{k-1} - L_k, total_k = L_1 - L_k (458-user review-weighted LogLoss); stop after epoch k >= 5 when gain_k < 0.05 total_k AND gain_{k-1} < 0.05 total_{k-1} on BOTH heads; min 5, cap 14; the decay starts from epoch k.
Scheduling: epoch 1 waits until `wt/sm59/artifacts/small-model/sm59/launch-sm59.log` has a `[sm59-launch ...]` line with `check D passed`, `start arm `, `STOP` or `all arms finished` (check D needs ~17 GB RAM). If the SM59 launcher is not running (all detached launchers were killed at 10:18), SM56 waits (heartbeat line every 30 min): restart the SM59 launcher, or tell the user; a STOP line in that log also releases SM56. Every ticket checks quiet.json and the daily 21:00-01:00 break (an epoch never starts if it would end after 21:00; the decay needs 510 min before the next break, so typically 01:00 -> ~09:30). GPU budget: minimum 5 x (120 + 12) + 510 + 20 = ~19 h of GPU, cap 14 epochs = ~39 h.

## Checks (dry run 10-22 11:57, `sm56/launch-sm56-dryrun.log`, DRYRUN OK)
- Experiment = sc-lanes-cont's options except the replay folder; 8,000 final-tier users / 581,942,738 reviews per epoch, disjoint from the 464/732 pools and the sealed users; options allowlisted, carry options valid, replay spec valid.
- Label windows (`sm56/check-label-windows-sm56.json`): all 8,000 replay files present with rows == review counts, label counts == the export records (505,739,554 labelled), one trimmable tail (5599), 22,079 updates per epoch before the trim.
- Epoch build (CPU): epoch 1 protocol from scratch (no init, update_offset 0, evaluation off, seed 20261013x1000+1 order, trim applied); epoch 2 chains epoch 1's master + AdamW state.
- Smoke inputs = the experiment's epoch options. Decay: SM55's exact prepare command (from sc-lanes-cont e12, scratch) reproduces SM55's decay.json/protocol.json/trims; SM56 run options + the decay EXTRA == SM55's decay options; Muon fresh-state check 7 PASS (`sm56/check-muon-fresh-decay-dry.json`).
- Summary code reproduces SM55's own summary lines when fed SM55's rows.

## Log lines to watch (`sm56/launch-sm56.log`)
- Per epoch: `[sm56 MM-DD HH:MM:SS] epoch k: curve X instant Y gain curve [...] instant [...] -> continue|stop (reason)`.
- Decay: `[sm56 ...] decay start: ...`; done: `[sm56 ...] done` after the `[sm56-summary]` block.
- Failure: any `[sm56 ...] STOP: <reason>` (ends with `STOP: ended with exit N`).

## Follow-up (fresh Sonnet medium agent, when `done` appears)
Report from `sm56-summary.md`: SM56 vs SM55 and vs SM36 (delta, 95% CI) and vs the fork on 464 and 732 (723 users), the per-epoch curve/instant table and the stopping epoch, reviews/s and minutes per epoch and for the decay. Commit `sm56-summary.*`, `heldout-sm56*.jsonl` (benchmark users only), `sm56/epochs.jsonl`, the log; then the SM56 section of `docs/SMALL_MODEL.md`.
