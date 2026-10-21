# HANDOFF: SM42 (no-carry-source control) and SM36 (Muon decay), prepared 2026-10-19, CPU only, NOT started

Worktree `C:/Users/jonat/workspace/wt/sm42` (branch `sm42`, from `sm21v2`). Both arms are one change away from SM40v2
(one-epoch decay at 32,768 x 1, strided-lane carry, replay targets, gpu_memory_fraction 0.8, from v2 e12).

| Arm | Change vs SM40v2 | Label | Source |
|---|---|---|---|
| SM42 control "nocarry-src" | source = plain 32k run (no carry) epoch 10 instead of v2 epoch 12 | `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes` | `wt/small-model/.../scale-w64kd-gate-cached-ctx32768-cont` e10 |
| SM36 Muon decay | `"optimizer": "expertium-iter53"`, fresh optimizer state | `s-w64kd-gate-sc-lanes-dec12-1ep32k-muon` | `wt/sm21v2/.../scale-w64kd-gate-sc-lanes-cont` e12 |

Files (all in `C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/`): `runbook-2026-10-19-sm42-sm36.sh` (one runbook, `ARM=sm42|sm36`),
dry-run logs `runbook-2026-10-19-sm42-dryrun.log` / `-sm36-dryrun.log` (both end `DRYRUN OK`), `launch-sm42-sm36-after-prune.sh`,
`check-muon-fresh-decay.json` (CPU check, passed). Code: `scripts/expertium_decay.py prepare --fresh-optimizer`, `scripts/check_muon_fresh_decay.py`.

## Findings
- **SM42 source options.** The plain 32k run used cached window-teacher targets (`"cache": .../kd-teacher/fork-bf16-s2025`) and no
  `state_carry`. The decay passes the SM40v2 replay spec (whole `distillation` dict replaced via `--extra-options`, so no `cache` key) plus
  `state_carry {"lanes": "strided"}`; chunk 32768 x 1 and gpu_memory_fraction 0.8 were already the source's. Possible, no improvisation.
  The runbook compares the prepared `decay.json`/`protocol.json` with SM40v2's own (wt/sm21v2): options are IDENTICAL for SM42, identical
  except `optimizer` for SM36; same 5,000 users in the same order, 369,477,375 reviews, LR 7e-4 -> 0 one_minus_sin. Passed in both dry runs.
  The decay continues from e10's AdamW state (as dec10 and SM40v2 continue theirs). The model was never trained on carried state, so SM42
  starts out of distribution for the first updates; that is part of the question.
- **SM36 fresh state.** AdamW state cannot be loaded into Muon (the hybrid state_dict has keys `adamw`/`muon`; loading e12's file raises
  `KeyError: 'adamw'`). `prepare --fresh-optimizer` drops `init_optimizer` from the protocol (and marks `optimizer_state: fresh`), so the
  trainer builds `expertium_iter53` and never loads a file. CPU check (`check-muon-fresh-decay.json`, passed): 309,888 Muon / 21,623 AdamW
  parameters, empty state at start, one step finite with the written LR restored, state round-trips through state_dict.
  Not covered by CPU: the first GPU Muon run. The smoke in the decay ticket is its first test; if it fails, stop and report.
  Note: `update_offset` is kept, so there is no 5-update warmup (same as SM40v2); Muon starts at full LR with zero momentum.
- **Cache key.** `"optimizer"` does NOT change any KD key. With replay targets (`ReplayTargets`) there is no cache folder or digest at all,
  and the window-teacher cache key excludes `"optimizer"` (`TEACHER_BLIND`; `check_kd_cache.py`). No cold cache, no extra teacher
  recompute. (The older doc remark "optimizer is part of the cache key" in SMALL_MODEL.md predates SM20.)
- **732 readout.** The 732 pool = the 464 users + 268 extra. The SM40v2 runbook's pool pair used only the 268-user pool file (265 users);
  this runbook merges the 464 rows file and the pool file, so the 732 pairs cover 723 users (SM40v2 scored 723 of 732).

## GPU tickets (queue owner small-model, claim gpu, minutes = `--expected-minutes`)
| Ticket | SM42 | SM36 |
|---|---:|---:|
| smoke + 1-epoch decay (369.5 M reviews, ~24,500 reviews/s; SM40v2 took 290 min measured) | 330 | 360 (Muon step cost unmeasured) |
| score 464 users, pass 1 (SM40v2: 15 min) | 25 | 25 |
| score 464 users, resume pass 2 (only if users missing) | 15 | 15 |
| score 732 pool, pass 1 (SM40v2: 18 min) | 45 | 45 |
| score 732 pool, resume pass 2 (SM40v2: 1.4 min) | 20 | 20 |
| **Total nominal (pass 1 only / with resumes)** | **400 / 435** | **430 / 465** |

Expected actual: ~325 min (SM42), ~350 min (SM36). Both arms back to back: 830 nominal, ~11-14 h wall clock.

## Launch (do not start before the prune run is done; the launcher waits by itself)
```
nohup bash C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/launch-sm42-sm36-after-prune.sh > C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/launch-sm42-sm36-after-prune.log 2>&1 & disown
```
It waits for a line `^\[prune [0-9: -]+\] (done|ended with exit)` in `wt/prune/artifacts/small-model/launch-prune-after-chain.log`, waits while
`wt/small-model/artifacts/small-model/HOLD-chain` exists (checked before each arm), runs SM42 then SM36 (SM36 runs even if SM42 failed; logged).
Arm logs: `runbook-2026-10-19-sm42.log`, `runbook-2026-10-19-sm36.log`. Rerun the same launcher or `ARM=... bash runbook...` to resume
(finished decay / scored users are skipped). Do not edit scripts/ expertium/ rwkv/ in wt/sm42 while it runs (SourceWatch).

## Results (after the runs; fresh Sonnet medium agent)
`ARM=sm42 SUMMARY_ONLY=1 bash .../runbook-2026-10-19-sm42-sm36.sh` (and sm36) writes `sm42-summary.{json,md}` / `sm36-summary.{json,md}`:
SM42 pairs vs dec10 and vs SM40v2 (464, 732); SM36 pairs vs SM40v2 (464, 732); review-weighted, user bootstrap 2,000 seed 0.
Reading: SM42 ~ SM40v2 -> the long 32k decay alone gives the gain, carried-state epochs did not matter; SM42 clearly worse -> the 2 carried epochs matter.
