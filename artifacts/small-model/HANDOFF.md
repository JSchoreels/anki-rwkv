# Small-model handoff (2026-10-15 ~00:55)

Worktree `C:\Users\jonat\workspace\wt\small-model`, branch `small-model` (pushed). Context and frontier table: [docs/SMALL_MODEL.md](../../docs/SMALL_MODEL.md). The goal is a model clearly lighter than the fork's 2.76M that is close to fork int8 (int8 weights + int8 state) on held-out users.

## Running: night runbook (detached, launched 00:44 from the main checkout)

`artifacts/small-model/runbook-2026-10-15.sh`. Log: `artifacts/small-model/runbook-2026-10-15.log`; it prints each ticket id as the ticket registers. Steps run in order, one queue ticket each, all `--claim gpu`, no `--timing`:

| # | Ticket | Step | Expected min | Output |
|---|---|---|---:|---|
| 1 | 000332 | train `screen-w64-s2025` (order = w32-s2025) | 28 | `artifacts/accuracy/experiments/screen-w64-s2025/run/epoch-01/expertium/E89-F65e-F77a/master.pth`, `phases.jsonl` (reviews/s) |
| 2 | next | train `screen-w64kd-s2025` | 36 | same layout |
| 3 | next | score fork `fp32`, `w8+s8` | 50 (limit 2,700 s) | rows in `artifacts/small-model/heldout.jsonl` |
| 4 | next | score `decay` fp32/w8+s8, `fullrun-e10`, `w32-o0`, `w64-o0`, `w64kd-o0` (fp32) | 55 (limit 3,000 s) | same file |
| 5–8 | next | train `screen-w64-s2025-o1`, `screen-w64kd-s2025-o1`, `-o2`, `-o2` | 28 / 36 / 28 / 36 | as 1 |
| 9 | next | score `w32-o1/o2`, `w64-o1/o2`, `w64kd-o1/o2` (fp32) | 55 | same file |
| 10 | next | resume every label (users cut by time limits), plus `w64-o0/w8+s8`, `w64kd-o0/w8+s8` | 50 | same file |

The queue was 000330 (running) and 000331 (20 min) ahead at launch. Projected: start ≈ 01:15, end ≈ 08:00 if nothing else interleaves. Scoring skips checkpoints that are missing, so a failed training step does not kill the chain.

Untested on GPU at launch (the GPU was busy):
- the head-size-16 any-length WKV kernel. Each scoring ticket self-tests it against the float32 reference and aborts above 1e-4.
- w64 training through `accuracy_loop` with the wide kernels. The options were validated on CPU by `CatchUp`.
- KD at 256 chunks (memory unmeasured).

The scorer passed a CPU smoke on 1 user (fork, w32 iter36, random-init w64).

# INTERIM k=1 read, tickets 000341/000342 (2026-10-15, CPU report; accuracy lane, no speed figures)

Scored: fork fp32/w8+s8 on 360 users (time limit), decay fp32/w8+s8, fullrun-e10, w32-o0, w64-o0, w64kd-o0 on 410 of the 464 users (time limit, resume in step 10). Report on the 360 users every label has (`heldout-report.md`); screens paired on 410 users. o1/o2 and w8+s8 of w64/w64kd not scored yet.

| Label | Params | Curve LL | Δ% vs fork w8+s8 | Instant LL | Δ% | Curve RMSE(bins) Δ% | Instant RMSE(bins) Δ% |
|---|---:|---:|---:|---:|---:|---:|---:|
| fork/fp32 | 2,762,884 | 0.29870 | +0.00% | 0.27475 | -0.01% | +0.0% | -0.3% |
| fork/w8+s8 | 2,762,884 | 0.29869 | ref | 0.27478 | ref | ref | ref |
| decay/w8+s8 (w32) | 192,800 | 0.31414 | +5.17% | 0.29282 | +6.57% | +37.9% | +77.2% |
| fullrun-e10 (w32) | 192,800 | 0.31518 | +5.52% | 0.29382 | +6.93% | +42.2% | +79.4% |
| w32-o0 | 192,800 | 0.32870 | +10.05% | 0.31310 | +13.95% | +75.4% | +193.3% |
| w64-o0 | 331,511 | 0.32814 | +9.86% | 0.30539 | +11.14% | +75.6% | +140.6% |
| w64kd-o0 | 331,511 | 0.32494 | +8.79% | 0.29812 | +8.49% | +78.7% | +104.8% |

Cross-check: the 79 users shared with held-out B reproduce `heldout-b.jsonl` for fork/fp32 and fork/w8+s8 (w8-pc+s8) exactly (max |dLL| = 0.0 on both heads). Self-test (max abs vs float32): fork bits0 1.8e-7, K16 bits0 8.9e-8, K32 bits0 1.2e-7; int8-state rows bits8: fork 2.2e-4, K16 4.2e-4, K32 1.3e-3 (the scorer did not abort; int8-state rounding differs by order).

Paired k=1 (410 users; full-history band is nan at k=1, window band 0.00622 curve / 0.00272 instant): w64-w32 curve -0.00063, instant -0.00765; w64kd-w32 curve -0.00313, instant -0.01467; kd-vs-w64 curve -0.00250, instant -0.00703.

Threshold at k=1 (needs k=3, so NOT met / not decidable yet):
1. Paired k=3: only o0 exists. w64kd: both heads negative; instant beats both bands; curve -0.00313 is inside the 0.0036 band. w64: curve -0.00063 inside, instant -0.00765 beyond the 0.0016 band.
2. Gap closure (o0 only): w64kd 1.26 pp curve (>=1.0 ok), 5.46 pp instant (>=1.5 ok); w64 0.19 pp curve (fails), 2.81 pp instant (ok).
3. int8 free: not scored yet (`w64-o0/w8+s8`, `w64kd-o0/w8+s8` are in step 10).
Verdict: no recipe is "promising" yet; w64kd is the leader and would be R (kd-vs-w64 negative on both heads). Caveat: the o0 screens sit far above the full-run w32 (+10% vs +5%), so the absolute "good candidate" bar is only reachable by the scale-up.
Scale-up sizing (accuracy-lane estimate, not a benchmark): w64kd r=106,987/s gives EPOCH_MIN = 369477375/106987/60+4 = 61.6 (> 57: refused, so the w64kd fallback applies); w64 r=166,967/s gives 40.9, i.e. EPOCH_MIN=41 (4x41+135 = 299 min <= 360). Per the HANDOFF rule, w64kd that passes but has EPOCH_MIN > 57 falls back to w64 only if w64 also passes; at k=1 w64 does not pass curve gap closure.

## Follow-up (fresh agent, Sonnet medium)

Commands (run from the small-model worktree; PY is the main checkout's venv python):
```
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
cd C:/Users/jonat/workspace/wt/small-model
$PY C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status
grep -E "runbook|CUDA recurrences|time limit|Error|error" artifacts/small-model/runbook-2026-10-15.log | tail -40
tail -2 artifacts/accuracy/experiments/screen-w64*-s2025*/phases.jsonl      # reviews/s, exit status per run
$PY -X utf8 scripts/small_model_report.py \
  --labels fork/fp32,fork/w8+s8,decay/fp32,decay/w8+s8,fullrun-e10/fp32,w32-o0/fp32,w32-o1/fp32,w32-o2/fp32,w64-o0/fp32,w64-o1/fp32,w64-o2/fp32,w64kd-o0/fp32,w64kd-o1/fp32,w64kd-o2/fp32,w64-o0/w8+s8,w64kd-o0/w8+s8 \
  --screen "w64-vs-w32:w64-o0/fp32=w32-o0/fp32,w64-o1/fp32=w32-o1/fp32,w64-o2/fp32=w32-o2/fp32" \
  --screen "w64kd-vs-w32:w64kd-o0/fp32=w32-o0/fp32,w64kd-o1/fp32=w32-o1/fp32,w64kd-o2/fp32=w32-o2/fp32" \
  --screen "kd-vs-w64:w64kd-o0/fp32=w64-o0/fp32,w64kd-o1/fp32=w64-o1/fp32,w64kd-o2/fp32=w64-o2/fp32"
```

Steps:
1. Drop any label that is missing from `--labels`, because the frontier uses the users that every listed label has.
2. Cross-check: the 100 users shared with held-out B must reproduce `C:/Users/jonat/workspace/wt/compression/artifacts/compression/heldout-b.jsonl` for `fork/fp32` and `fork/w8+s8` (variant `w8-pc+s8` there) within about 2e-4 LogLoss per user.
3. Then:
   - fill the SMALL_MODEL.md frontier table and ledger rows S01/S02 (accuracy and the self-test errors; no speed columns: these are accuracy-lane runs, docs/FLOW_AUDIT.md item 1);
   - commit `heldout.jsonl` (public anonymized benchmark users, allowed) and `heldout-report.{md,json}`;
   - commit the experiments' `experiment.json`/`phases.jsonl`/`result.json` with `git add -f`, never checkpoints;
   - commit as Jonathan Schoreels with the Co-Authored-By trailer, then push.
4. Use each training run's logged reviews/s only to size the scale-up (`EPOCH_MIN`) and as a rough planning ETA, labelled "accuracy-lane estimate, not a benchmark". Never quote it as a speed result; a real speed figure needs a `--timing` ticket (docs/FLOW_AUDIT.md item 1).

If a step failed:
- Training: `$PY .../r01_queue.py run --owner small-model --purpose "small-model: retrain <name>" --claim gpu --expected-minutes 36 -- $PY -X utf8 C:/Users/jonat/workspace/wt/small-model/scripts/accuracy_loop.py train <name>`. If the log shows an out-of-memory error, prepare a new experiment with `"batch_chunks": 128`.
- Scoring: re-run the step's `small_model_heldout.py` command (copy it from the log `CMD`, or from the runbook's `score` function). It resumes per (user, label).
- Kernel self-test failure: score on the generic path is impossible for long histories. Fix `scripts/small_model_wkv.py` (`_source`: K=16 uses `__syncwarp(0xffff)`) and test it with `$PY -m scripts.small_model_wkv` through a short gpu ticket.

## Decision criteria

All figures are review-weighted benchmark LogLoss (and RMSE(bins)) in full history on the 464 fork-unseen screen users, Δ% against `fork/w8+s8` (fork int8 weights + int8 state). R = recipe `w64kd` (preferred) or `w64`.

**"Promising" (triggers the 6 h scale-up). All of these must hold for R:**
1. **Paired k = 3 vs the w32 baselines:** all three deltas (R-o_i − w32-o_i) are negative on both heads, and |mean| exceeds the larger of the full-history band printed by the report and the window-protocol band (0.0036 curve / 0.0016 instant).
2. **Gap closure:** mean over i of [Δ%(w32-o_i) − Δ%(R-o_i)] is at least **1.0 percentage point on curve and 1.5 pp on instant**. That is roughly a third of the full-run w32 gap (+2.7% / +3.5% on the collection, +4.7% / +7.2% held-out).
3. **int8 is free for R:** `R-o0/w8+s8` minus `R-o0/fp32` is ≤ +0.10% LogLoss on both heads.

Choosing R:
- Take `w64kd` if it passes and the KD screen (`kd-vs-w64`) has a negative mean on both heads.
- Otherwise take `w64` if it passes.
- If neither passes, do not scale up. Report, and propose w80 + KD (SM06) to the user instead.

**"Good candidate" (the user's goal; judged after the scale-up):**
- `scale-R-dec/w8+s8` vs `fork/w8+s8`: LogLoss ≤ **+1.0% curve and ≤ +1.5% instant**, with the 95% user-bootstrap upper bounds ≤ +2.0% / +2.5%;
- instant RMSE(bins) ≤ +15%;
- it beats `decay/w8+s8` (our current w32) on both heads.
- Size is satisfied by construction for w64: 331,511 params (8.3× fewer), int8 file 410,459 B (7.3× smaller than 2,995,488), int8 card state 3,072 B vs 15,360.
- Results between "promising" and "good" are reported as "closer, not yet close enough", with the remaining gap.

## Scale-up (only if "promising"; user budget 6 h; queued by the main session, not by this agent)

Runbook: `artifacts/small-model/runbook-scaleup.sh <w64|w64kd>` (committed, never launched).

What it trains:
- Gate tier: 5,000 training users, which contain the screen's 2,000; 369,477,375 reviews per epoch.
- 4 epochs at constant LR, one queue ticket each (`train --epochs-per-job 1`; exit 3 = more epochs remain).
- Then the A08 0.25-epoch decay (`expertium_decay.py`, `--evaluate none`). Its CPU `prepare` must succeed first, or the decay is skipped.
- Then full-history scoring of `scale-R-dec` (fp32 and w8+s8) and `scale-R-e4` (fp32) on the same 464 users, in two tickets (55 + 45 min), appended to `heldout.jsonl`.

1. Compute `EPOCH_MIN = 369477375 / r / 60 + 4`, where r = `reviews_per_second` of `screen-R-s2025` in its `phases.jsonl`. With W04's w64 speed (141k/s) that is about 48 min. It must be ≤ 57; the runbook refuses anything larger.
2. Budget: 4 × EPOCH_MIN + 35 + 55 + 45 ≤ 360 min.
3. Launch (from `C:\Users\jonat\workspace\anki-rwkv`):
   ```
   EPOCH_MIN=<n> nohup "C:/Program Files/Git/usr/bin/bash.exe" C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-scaleup.sh <w64|w64kd> > C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-scaleup.log 2>&1 & disown
   ```
4. Fallback if R = w64kd and EPOCH_MIN > 57 (the teacher forward is too slow for one ticket per epoch): use `w64` if it also met the threshold. Otherwise do not scale up, and report.
5. Evaluation afterwards: run `small_model_report.py` with the scale labels added to `--labels` (fork/w8+s8 reference). Apply the "good candidate" criteria, and also compare `scale-R-dec` with `decay` (w32 at 8k users × 10 epochs).
6. Not covered: the user's own collection and the window protocol. Both evaluators build only the w32 model (SMALL_MODEL SM07).
7. Re-launch semantics: with the same recipe the runbook resumes (finished epochs and a finished decay are skipped). `RUN_TAG=<tag>` starts a separate run: experiment `scale-R-gate-<tag>`, labels `scale-R-<tag>-{e4,dec}`. Checkpoints: every epoch in `artifacts/accuracy/experiments/scale-R-gate/run/epoch-0N/...`, the decayed final in `artifacts/small-model/scale-R-gate-decay/expertium/E89-F65e-F77a/master.pth`.

## Save and export a candidate (CPU only, about 1 min each; docs/SMALL_MODEL.md section 4)

Run this on every checkpoint that meets the "good candidate" bar (`scale-R-dec`). Also run it on the best screen checkpoint (`R-o0`) if the user asks to keep it. Use a new `--out` per checkpoint; `package` refuses an existing folder:
```
$PY -X utf8 scripts/small_model_export.py package \
  --checkpoint artifacts/small-model/scale-R-gate-decay/expertium/E89-F65e-F77a/master.pth \
  --arch w64 --label scale-R-dec --out C:/Users/jonat/workspace/anki-rwkv/artifacts/models/scale-R-dec
```
- It copies `master.pth` to the main checkout (it survives removal of the worktree). It writes `weights.f32.bin` (the fork's ARWKVWEIGHTS1 container), `weights.int8.bin`, `architecture.json`, `parity.json` and `manifest.json` (sha256, params, architecture, training config, held-out summary vs `fork/w8+s8`).
- Exit 0 means NumPy vs PyTorch max |Δp| < 1e-5 on 3 held-out users (fp32 and int8). Quote the parity figures and the manifest's `evaluation` Δ% in the report.
- If the held-out scoring finishes later, refresh only the manifest: `$PY -X utf8 scripts/small_model_export.py manifest --dir <out> --checkpoint <the original master.pth, so the training record is found> --arch w64 --label scale-R-dec`.
- Never commit the package (`artifacts/` is gitignored; no `.pth` in git).
- Anki cannot load it yet: the fork's Rust runtime is fixed to its 2.76M shapes. The changes it needs are listed in docs/SMALL_MODEL.md section 4 ("Fork changes needed", SM08) for the user or Codex. Do not edit the fork.

Caveats:
- The w64 screens use 256 chunks and the baselines 512, so twice the updates (disclosed).
- KD: 1,003 of the 2,000 screen training users (2,524 of the gate's 5,000) are teacher-training users.
- Screens choose a recipe; they validate nothing (AGENT.md). A broader validation needs the user's explicit request.

# FINAL k=3 verdict, tickets 000332-000348 processed (2026-10-15, CPU report; accuracy lane, no speed figures)

All 17 tickets exit 0. Report on the 430 fork-unseen users every label has (34 of 464 cut by time limits across all resume passes): `heldout-report.md/json`. heldout.jsonl has 6,880 rows (the main checkout has no copy; this worktree file is the complete one and is committed).

| Label | Mean Δ% curve vs fork w8+s8 | Mean Δ% instant | Paired mean Δ vs w32 curve (band) | instant (band) | Gap closure curve / instant (pp) |
|---|---:|---:|---|---|---|
| w32 o0-o2 | +12.19 | +16.47 | base | base | base |
| w64 o0-o2 | +11.68 | +12.13 | -0.00151 (0.00359, inside) | -0.01166 (0.00471, better) | 0.51 (fails 1.0) / 4.34 |
| w64kd o0-o2 | +9.79 | +10.25 | -0.00706 (0.00490, better) | -0.01673 (0.00488, better) | 2.40 / 6.22 |

Per-seed deltas vs w32 are all negative on both heads for w64kd (curve -0.00367, -0.01098, -0.00653; instant -0.01574, -0.01976, -0.01470); w64 curve -0.00016, -0.00313, -0.00123. kd-vs-w64: curve -0.00555, instant -0.00507, all seeds negative. Reference points: w32 decay/w8+s8 +6.69% / +7.30%, fullrun-e10 +6.95% / +7.53%, fork fp32 +0.01% / -0.01%.

int8 cost (R-o0 w8+s8 minus fp32): w64 curve +0.01 pp, instant +0.01 pp; w64kd curve +0.01 pp, instant 0.00 pp (both <= 0.10: free).

Cross-check: 93 users shared with held-out B reproduce `heldout-b.jsonl` for fork/fp32 (max |dLL| 3e-8 curve, 0 instant) and fork/w8+s8 = w8-pc+s8 (6e-8, 5e-9).

Self-test: `small_model_heldout.py` aborts only on keys ending in "0" (the bits0 / float-state recurrences, threshold 1e-4); the int8-state (bits8) errors up to 1.3e-3 are reported but exempt by design, because int8 state rounding legitimately differs by summation order.

Criteria: (1) w64kd passes (all negative, |mean| above both bands); w64 fails on curve. (2) w64kd passes (2.40 >= 1.0, 6.22 >= 1.5); w64 fails curve. (3) int8 free for both. So R = w64kd (kd-vs-w64 negative on both heads). Sizing (accuracy-lane estimate, not a benchmark): w64kd r=106,987/s gives EPOCH_MIN = 369477375/106987/60+4 = 61.6 > 57, refused; fallback to w64 needs w64 to pass, and it does not. VERDICT: w64kd is the promising recipe, but the HANDOFF rule does NOT permit the scale-up. Do not scale up; the rule says to report and propose w80 + KD (SM06) or a KD-speed fix (smaller teacher cost per epoch, or a relaxed EPOCH_MIN cap). For information only, w64 r=166,967/s would be EPOCH_MIN=41 (299 min), but it is not promising. If the user overrides the rule for w64kd, EPOCH_MIN=62 gives 4x62+135 = 383 min > 360 (does not fit; 3 epochs would). No good-candidate checkpoint exists, so nothing was packaged. The screens are above the full-run w32 (+6.7%/+7.3%).

## Scale-up launched (main session, 2026-10-15 ~07:05)

The k=3 verdict found w64kd promising; the HANDOFF rule blocked the scale-up only through the runbook's own one-ticket-per-epoch cap (EPOCH_MIN 61.6 > 57) and the 4-epoch budget (383 > 360 min). The user's instruction ("if that model seems promising, start training it on more users/epochs; budget 6h") governs, the queue has no ticket-length rule and quiet.json is empty, so the main session launched **w64kd with EPOCHS=3, EPOCH_MIN=62, MAX_EPOCH_MIN=66**: 3x62 + 35 + 55 + 45 = 321 min <= 360. Runbook change: `EPOCHS`/`MAX_EPOCH_MIN` env options (defaults unchanged: 4 / 57). Labels: `scale-w64kd-e3` (epoch 3) and `scale-w64kd-dec` (decayed). Log: `artifacts/small-model/runbook-scaleup.log` (worktree). Follow-up when its last ticket ends: steps 5-7 of "Scale-up" above (report with the scale labels vs fork/w8+s8 and vs `decay`; package `scale-w64kd-dec` if it meets the good-candidate bar, and package it anyway as "closer, not yet close enough" if it beats the w32 `decay` checkpoint, so the user has a usable file).

Scale-up tickets: 000349-000351 epochs 1-3 (exit 3, 3, 0). 000352 decay exited 1 after training completed: the decayed checkpoint `artifacts/small-model/scale-w64kd-gate-decay/expertium/E89-F65e-F77a/master.pth` is valid; only the post-run `compare` crashed (KeyError 'per_user' with --evaluate none, fixed in bcc8c9e). No re-queue needed: 000353 (scoring) and the resume ticket score `scale-w64kd-dec` fp32 + w8+s8 and `scale-w64kd-e3` fp32.

# Scale-up RESULT, tickets 000349-000354 all processed (2026-10-15, CPU report; accuracy lane, no speed figures)

Report on the 430 users every label has (`heldout-report.md/json`; scale rows were written to the main checkout's heldout.jsonl by the queue and merged into this worktree's file, 8,272 rows). Δ% vs fork w8+s8:

| Label | Curve LL / Δ% | Instant LL / Δ% | Instant RMSE(bins) Δ% |
|---|---|---|---:|
| scale-w64kd-dec w8+s8 | 0.30970 / +5.28% [+4.17, +6.82] | 0.28534 / +5.99% [+5.42, +6.61] | +85.0% |
| scale-w64kd-dec fp32 | 0.30970 / +5.28% | 0.28532 / +5.98% | +84.9% |
| scale-w64kd-e3 fp32 | 0.31037 / +5.51% | 0.28533 / +5.99% | +81.3% |
| w32 decay w8+s8 | 0.31385 / +6.69% | 0.28886 / +7.30% | +88.6% |
| w64kd screens o0-o2 (mean) | +9.79% | +10.25% | +115 to +152% |

Paired scale-w64kd-dec vs w32 decay (k = 1): curve -0.00415 (window band 0.00622, inside), instant -0.00348 (0.00272, marginally beyond); same for w8+s8 (-0.00416 / -0.00351). Beats `decay` on both heads. Fork fp32 for reference: +0.01% / -0.01%.

Good-candidate bar (<= +1.0% / +1.5%, bootstrap upper <= +2.0% / +2.5%, RMSE(bins) <= +15%): NOT met (+5.28% / +5.99%, upper +6.82% / +6.61%, RMSE +85%). It beats `decay` on both heads, so verdict "closer, not yet close enough" (remaining gap about 4.3 pp curve, 4.5 pp instant). int8 free. Packaged: `C:/Users/jonat/workspace/anki-rwkv/artifacts/models/scale-w64kd-dec/` (not in git): int8 file 436,391 B, 331,511 params, parity PASS (fp32 max |dp| 8.3e-7 curve / 1.4e-6 instant; w8 8.9e-7 / 1.3e-6). The 000352 compare crash was fixed in bcc8c9e. Next ideas: SM06 (w80 + KD, longer decay), more epochs/users.

# Next step designed, NOT launched (2026-10-15 ~12:00; design: docs/SMALL_MODEL.md section 5)

Nothing is queued. Every stage of `artifacts/small-model/runbook-next.sh` needs the user's go (heavy GPU).

- **KD cost.** Teacher-target cache implemented (`distillation.cache`; `scripts/check_kd_cache.py` gives bit-identical loss and gradients on a hit). Estimate: KD epochs drop from 57 to about 38 min at 5k users after the first epoch. 8k users × 6–10 epochs does not fit 6 h even without KD; 5k × 6 (4.7 h) or 8k × 4 (5.1 h) does.
- **Calibration** (CPU diagnostic, `scripts/small_model_calibration.py`, `artifacts/small-model/calibration-{full,window}.json`, 64 users):
  - in the 128-review training protocol the student matches the fork: instant LogLoss +0.1%, RMSE(bins) +1.7%;
  - over full histories it is +5.3% / +45%: the gap is long-context behaviour, not capacity and not a head temperature (Platt closes 8%).
- **Screen** (paired k = 3 decays from `scale-w64kd-gate` epoch 3):
  - `check` (≈ 10 min): GPU parity of the WKV recurrence beyond 383 steps + 32 × 1,024 update probe. Gate for `context`;
  - `control` (≈ 1.9 h; also completes the fork on all 464 users and scores epochs 1–2 for the slope);
  - arm B `context`, 1,024-review chunks (recommended; P1 implemented and CPU-checked 2026-10-15: `chunk_reviews`, `check_long_chunks.py`);
  - arm A `kd09` (optional, low value).
- **Smoke.** Every training ticket starts with `expertium_decay.py smoke` (60–120 s).
- **Epoch gate** (SMALL_MODEL.md 5.6): `scale` gates itself after epochs 1 and 2 on the fixed 100-user subset (`gate-users.json`). Decisions: `artifacts/small-model/gate-<run>-e<K>.json`. "stop" ends the launch resumably; `FORCE=1` continues.
- **Launch** (from `C:\Users\jonat\workspace\anki-rwkv`):
  ```
  nohup "C:/Program Files/Git/usr/bin/bash.exe" C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-next.sh check > C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-next-check.log 2>&1 & disown
  # then the same with `control`, then `context` (each after the previous launch returned)
  ```
- **Afterwards** (fresh Sonnet agent):
  1. rows land directly in this worktree's `artifacts/small-model/heldout.jsonl` (runbook-next writes there);
  2. run the `small_model_report.py --rmse-bins` command in SMALL_MODEL.md 5.4;
  3. apply the paired rule on LogLoss and RMSE(bins);
  4. check the `kd_cache` counters and reviews/s in each decay's `result.json`/log (accuracy lane, not a benchmark).

# LAUNCHED 2026-10-15 12:18-12:20 (user's go; check -> control -> context; detached, through the r01 queue)

The runbook stages run as separate detached processes. The queue serializes their tickets, so control and context interleave. Logs: `artifacts/small-model/runbook-next-{check,control,context}.log`. Don't edit `runbook-next.sh` while they run: bash reads the script as it executes. The uncommitted usage-comment line in it (mentions a chain mode that does not exist) is harmless; drop it after the runs.

| Ticket | Stage | Step | Status |
|---|---|---|---|
| 000355 | check | WKV parity beyond 383 steps + update probe | **done, 35/35 passed**, 38 s (`check-long-wkv.json`) |
| 000356 | control | fork w8+s8 on the 34 cut users + `scale-w64kd-e1/e2/dec` w8+s8 (464 users) | running at 12:19 |
| 000357 | context | decay `cal-ctx1024-s1` (smoke first) | queued behind 000356 |
| next | control / context | decays `cal-ctl-s2`, `cal-ctl-s3`, `cal-ctx1024-s2`, `cal-ctx1024-s3` (smoke first each), then scoring of each stage's decays + resume | each registers when the previous ticket of its stage ends |

check results:
- Parity: K = 16/32, T = 512/1,024/2,100. Generic NVRTC vs float64 reference: relative max error ≤ 1.9e-7 (forward and all gradients). The first 383 outputs of a long run are bit-identical to a 383-step run, and row-tiled exact equals generic bit for bit (BF16 and float32).
- Probe (one 32,768-review update, user 101, teacher + w64 student fwd/bwd, eager): 256 × 128 chunks 499 ms (row-tiled 482), 32 × 1,024 chunks 632 ms (row-tiled 652), so about 1.3× slower per update; peak 5.6 → 6.0 GB. Under the 2× re-size threshold, so arm B keeps 0.25-epoch decays (≈ 26–30 min each instead of ≈ 20). Probe timings are eager-mode step times, not training rates (accuracy lane, not a benchmark).

Expected end: about 4.5–5 h after 12:19, so ≈ 17:00–17:30.

Follow-up (fresh Sonnet agent, when both logs say `done:`):
1. `grep -E "exit|done|stop|smoke" artifacts/small-model/runbook-next-{control,context}.log`. Each decay ticket prints a smoke JSON line first. A failed smoke leaves no checkpoint and the stage logs "no checkpoint". Re-launch that stage the same way: it resumes.
2. Report (CPU), on all 464 users and again with `--min-user-id 101`:
   ```
   $PY -X utf8 scripts/small_model_report.py --rmse-bins \
     --labels fork/w8+s8,scale-w64kd-e1/w8+s8,scale-w64kd-e2/w8+s8,scale-w64kd-dec/w8+s8,cal-ctl-s2/w8+s8,cal-ctl-s3/w8+s8,cal-ctx1024-s1/w8+s8,cal-ctx1024-s2/w8+s8,cal-ctx1024-s3/w8+s8 \
     --screen "ctx1024-vs-ctl:cal-ctx1024-s1/w8+s8=scale-w64kd-dec/w8+s8,cal-ctx1024-s2/w8+s8=cal-ctl-s2/w8+s8,cal-ctx1024-s3/w8+s8=cal-ctl-s3/w8+s8"
   ```
3. Apply the paired k = 3 rule (SMALL_MODEL.md 5.3) on LogLoss and RMSE(bins). Also read each decay's `kd_cache` counters and reviews/s in its `result.json` (accuracy lane).
4. Optional: dump `cal-ctx1024-s1` on the 64 calibration users (CPU, `small_model_heldout.py --dump --max-reviews 80000 --count 64`) and re-run `small_model_calibration.py`.
5. Do NOT launch `scale` without the user: it needs the screen verdict (`EXTRA` with the long chunks if B wins) and the user's go.

# RESULT 2026-10-15 (calibration screen B: 1,024-review chunks vs 128, paired k = 3)

**Verdict: yes. 1,024-review training context beats 128 under the k = 3 rule on LogLoss (curve and instant) and on RMSE(bins) (curve and instant), all three seed deltas negative and well beyond both bands.**

Follow-up step 1: both logs end `done:`; every ticket through 000365 exited 0, all six smokes exited 0 (status complete), no "no checkpoint". Step 4 (calibration dump) skipped. `runbook-next.sh` stray usage comment dropped (checkout). Accuracy lane, no speed claim.

Held-out, full history, 464 fork-unseen users (min-user-id 101: 451 users, same picture; numbers in brackets). Δ% vs `fork/w8+s8` (int8, LogLoss 0.29766 curve / 0.27216 instant, RMSE(bins) 0.03867 / 0.02084):

| Label | Curve LL | Δ% | Instant LL | Δ% | Curve RMSE(bins) Δ% | Instant RMSE(bins) Δ% |
|---|---:|---:|---:|---:|---:|---:|
| scale-w64kd-e1 | 0.31803 | +6.85% | 0.29344 | +7.82% | +51.8% | +102.7% |
| scale-w64kd-e2 | 0.31469 | +5.72% | 0.28948 | +6.36% | +41.5% | +83.6% |
| scale-w64kd-dec (ctl s1) | 0.31374 | +5.40% | 0.28892 | +6.16% | +42.5% | +91.9% |
| cal-ctl-s2 | 0.31434 | +5.61% | 0.28876 | +6.10% | +44.6% | +89.4% |
| cal-ctl-s3 | 0.31425 | +5.58% | 0.28827 | +5.92% | +44.3% | +84.8% |
| cal-ctx1024-s1 | 0.30929 | +3.91% | 0.28341 | +4.13% | +27.3% | +45.0% |
| cal-ctx1024-s2 | 0.30963 | +4.02% | 0.28313 | +4.03% | +27.8% | +43.9% |
| cal-ctx1024-s3 | 0.30929 | +3.91% | 0.28318 | +4.05% | +26.4% | +45.5% |

Means: control curve +5.53% / instant +6.06%, RMSE(bins) +43.8% / +88.7%; ctx1024 curve +3.95% / instant +4.07%, RMSE(bins) +27.2% / +44.8%.

Paired k = 3 (cand = ctx1024 seed i, base = control seed i; s1 base is `scale-w64kd-dec`). Bands: full-history sigma (window sigma for LogLoss in brackets):

| Metric | Deltas | Mean | Band | Verdict |
|---|---|---:|---:|---|
| LogLoss curve | -0.00445, -0.00472, -0.00496 | -0.00471 | 0.00053 (window 0.00359) | better, beyond both |
| LogLoss instant | -0.00551, -0.00563, -0.00509 | -0.00541 | 0.00056 (window 0.00157) | better, beyond both |
| RMSE(bins) curve | -0.00587, -0.00650, -0.00692 | -0.00643 | 0.00073 | better |
| RMSE(bins) instant | -0.00978, -0.00950, -0.00819 | -0.00915 | 0.00123 | better |

451 users: LogLoss -0.00448 / -0.00531, RMSE(bins) -0.00611 / -0.00903, same verdicts. Effect size: the long context removes about 1.6 pp of the LogLoss gap to fork int8 (both heads) and cuts the RMSE(bins) gap roughly in half on the instant head (+89% to +45%).

Epoch slope of `scale-w64kd-gate` (LogLoss, curve / instant): e1 to e2 -0.00334 / -0.00396 (Δ% 6.85 to 5.72, 7.82 to 6.36); e2 to dec(128) only -0.00095 / -0.00056. The 128-chunk decay gives little; its RMSE(bins) instant even rises (+83.6% to +91.9%). The ctx1024 decay gives -0.0054 more on top, so the improvement is the context, not the extra decay steps.

Training speed (accuracy lane, `result.json` reviews_per_second, 0.25-epoch decays, ~92.4 M reviews LR schedule; the runs shared the box with CPU scoring jobs and the cache state differs, so only indicative):

| Decay | reviews/s | seconds | kd_cache hits / writes (online) |
|---|---:|---:|---|
| dec (ctl s1, 128) | 79,605 | 1,161 | no counters recorded |
| cal-ctl-s2 (128) | 60,205 | 1,536 | 0 / 3,534 |
| cal-ctl-s3 (128) | 63,397 | 1,457 | 833 / 2,631 |
| cal-ctx1024-s1 | 50,958 | 1,813 | 0 / 3,465 |
| cal-ctx1024-s2 | 54,300 | 1,703 | 979 / 2,555 |
| cal-ctx1024-s3 | 70,147 | 1,317 | 1,523 / 1,941 |

Cold-cache pairs (0 hits): ctx1024 50,958 vs ctl-s2 60,205 reviews/s, about 15% slower. Means 58,468 (ctx) vs 67,736 (ctl, three), about 14% slower; the check-stage probe predicted ~1.3x per update, so the cost is at most 1.15-1.3x. The KD cache is keyed by chunk option, so 1,024-chunk targets are separate; hits come from re-reading targets written earlier in the same seed family. Not a benchmark.

Recommendation: `scale` should use 1,024-review chunks (`EXTRA` with chunk_reviews 1024, batch_chunks 32), accepting about 15-30% lower training throughput. Still needs the user's go; nothing enqueued.

## LAUNCHED 2026-10-15 ~16:40: probe-scale (user go: "ok for your order, you can proceed and fill the queue")

Plan in docs/SMALL_MODEL.md 5.7; launched detached from the main session:
`nohup bash runbook-next.sh probe-scale > artifacts/small-model/runbook-next-probe-scale.log`.
1. `ctx2048`: decay `cal-ctx2048-s1` (2,048 x 16 chunks, ~40 min), then scoring on 464 users (~16 min + resume).
2. `scripts/small_model_chunk_pick.py` writes `artifacts/small-model/chunk-pick.json` (2,048 only if it beats `cal-ctx1024-s1` by at least 0.001 LogLoss on both heads, CI upper bound < 0, at most 1.5x slower; else 1,024).
3. `scale` TIER=gate EPOCHS=4 RUN_TAG=ctx<chunk>: run `scale-w64kd-gate-cached-ctx<chunk>`, labels `s-w64kd-gate-ctx<chunk>-e<K>/w8+s8` and `-dec`. Epoch gates after epochs 1 and 2 (`gate-scale-w64kd-gate-cached-ctx<chunk>-e<K>.json`); a "stop" ends the launch cleanly.
- Estimate: probe ~1 h, then about 5.5 h of training, gates, decay and scoring if both gates say go, or about 1.5 h if epoch 1 stops. Finish around 00:00-01:00.

Follow-up (fresh Sonnet medium agent, when the log's last line is `done: scale` or a gate stop):
1. grep the log for exit/done/stop/smoke/"chunk pick"; read chunk-pick.json and the gate JSONs.
2. If scale finished: `$PY -X utf8 scripts/small_model_report.py --rmse-bins --labels fork/w8+s8,scale-w64kd-dec/w8+s8,cal-ctx1024-s1/w8+s8,cal-ctx2048-s1/w8+s8,s-w64kd-gate-ctx<chunk>-e4/w8+s8,s-w64kd-gate-ctx<chunk>-dec/w8+s8` (all users and `--min-user-id 101`), restoring heldout-report.{md,json} afterwards, or writing them with `--out` into a scratch path. Compare against the good-candidate bar in this file. Quote reviews/s per epoch from the run's result files.
3. Write a "RESULT probe-scale" section here and in SMALL_MODEL.md, `git add -f` heldout.jsonl and the JSONs, commit and push. Launch nothing.
4. Then the main session starts the priority re-assessment (AGENT.md, Opus high).

### Override 2026-10-15 17:17: scale runs at 2,048-review chunks (user: "yeah go for 2048")

- **The pick** (`chunk-pick.json`) kept 1,024 on a near miss.
  - cal-ctx2048-s1 vs cal-ctx1024-s1 LogLoss: −0.00095 curve (95% upper bound −0.00077) and −0.00106 instant (−0.00093), against the −0.001 margin.
  - Both decays had a cold KD cache (0 hits), and the 2,048 decay trained faster: 62.4k vs 51.0k reviews/s.
  - The main session proposed the post-hoc override and the user approved it.
- **The 1,024 run was stopped** after about 20 min of epoch 1, before any checkpoint (ticket 000369 cancelled, process tree killed). `artifacts/accuracy/experiments/scale-w64kd-gate-cached-ctx1024` holds only its prepared config.
- **Relaunched detached** from the main checkout at 17:17, as ticket 000370 onwards:
  `TIER=gate EPOCHS=4 EPOCH1_MIN=69 EPOCH_MIN=59 RUN_TAG=ctx2048 EXTRA='{"chunk_reviews": 2048, "batch_chunks": 16}' runbook-next.sh scale`
  - Log: `artifacts/small-model/runbook-next-scale-ctx2048.log`.
  - Run: `scale-w64kd-gate-cached-ctx2048`; labels `s-w64kd-gate-ctx2048-e<K>/w8+s8` and `-dec`.
  - Epoch gates after epochs 1 and 2. Estimate: done around 22:30–23:30 if both gates say go.
- **Follow-up:** as in "LAUNCHED ~16:40" above, with `<chunk>` = 2048 and this log.

## PAUSED 2026-10-15 19:44: scale-w64kd-gate-cached-ctx2048 after epoch 2 (user cap ~20:00; epoch 3 needs an explicit go)

- **Status:** `scale-w64kd-gate-cached-ctx2048` (2,048-review chunks x 16, section "Override 2026-10-15 17:17", commit 09d0cb7) finished epochs 1 and 2. Both epoch gates said GO (`gate-scale-w64kd-gate-cached-ctx2048-e1.json`, `-e2.json`; 100-user subset, w8+s8).

| Checkpoint | LogLoss curve | vs fork 0.28665 | LogLoss instant | vs fork 0.26105 |
|---|---:|---:|---:|---:|
| e1 | 0.29771 | +3.86% | 0.27421 | +5.04% |
| e2 | 0.29595 | +3.24% | 0.27078 | +3.73% |
| e2 projected final | 0.29368 | +2.45% | 0.26707 | +2.31% |

  The previous scale run's best decay (same subset) was +4.98% / +6.23%.
- **Timings:** epoch 1 took 77 min (71 planned), epoch 2 took 61 min (61 planned); gates took 5.3 and 3.4 min.
- **What stopped it:** the user said "don't start epoch 3 before I tell you". The runbook had already enqueued epoch 3 as ticket 000374; it was cancelled about 10 s in (smoke step) and killed, so the log ends with "training failed (exit 1), stop". The epoch-2 checkpoint is intact (`artifacts/accuracy/experiments/scale-w64kd-gate-cached-ctx2048/run/epoch-02`); there is no `epoch-03` folder, so no `epoch-03-attempt-N` leftover.
- **Epoch 3 needs the user's explicit go.** Do not resume without it.

### Resume command (after the user's go)

Same form as the 17:17 relaunch (detached, from `C:\Users\jonat\workspace\anki-rwkv`), same parameters; no FORCE:

```
TIER=gate EPOCHS=4 EPOCH1_MIN=69 EPOCH_MIN=59 RUN_TAG=ctx2048 EXTRA='{"chunk_reviews": 2048, "batch_chunks": 16}' \
  nohup "C:/Program Files/Git/usr/bin/bash.exe" C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-next.sh scale \
  > C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-next-scale-ctx2048.log 2>&1 & disown
```

(Use `>>` instead of `>` to keep the old log. Epoch minutes are the planned values from the 17:17 launch; e2 actually took 61 min and e1 77, so EPOCH_MIN=59 is slightly low; it only sets the queue ticket's expected minutes.)

Verified by reading `runbook-next.sh` (`scale` stage) and `scripts/accuracy_loop.py`:
- NAME = `scale-w64kd-gate-cached-ctx2048` follows from TIER=gate and RUN_TAG=ctx2048; `experiment.json` exists, so it logs "resuming existing" and does not re-prepare.
- The loop over epochs skips any epoch whose `run/epoch-NN/<CKPT>/master.pth` exists, so epochs 1 and 2 are not retrained; `accuracy_loop.py train` also skips epochs whose `result.json` is complete. Training starts at epoch 3, one epoch per ticket (smoke first).
- FORCE=1 is only consulted when a gate says STOP; both gates said GO, so it is not needed. Do not set it.
- The script re-runs the epoch-1 and epoch-2 gates after the (skipped) training step. Rows already in `heldout.jsonl` are skipped, so each should be a short ticket (a few minutes of GPU queue, expected 15) and give the same GO. It also re-makes the CPU smoke-prep folder each launch.
- Epoch 4, the 0.25-epoch decay and scoring then follow in the same launch.
- NOT confirmed: that epoch 3 (the `_epoch` child, `run_full_training.py`) picks up the optimizer/model state from `epoch-02` with these options. I did not read that code or run anything; the epoch-3 smoke killed at ~10 s gave no evidence either way. Check the first lines of the new log / `phases.jsonl` for the epoch-3 start and that the loss continues from the epoch-2 level.
- NOT confirmed: the uncommitted `heldout.jsonl` change in the worktree is the gate rows from this run; it was left uncommitted here on purpose.

### Follow-up after the run ends

1. The "LAUNCHED ~16:40" follow-up steps (fresh Sonnet medium agent, when the log ends with `done: scale` or a gate stop), with `<chunk>` = 2048: grep the log, read the gate JSONs, run `small_model_report.py --rmse-bins` for the listed labels, write a "RESULT probe-scale" section here and in SMALL_MODEL.md, `git add -f` heldout.jsonl and the JSONs, commit and push. Launch nothing.
2. Then the priority re-assessment per anki-rwkv `AGENT.md` (main session, Opus high).

## RESUMED 2026-10-16 00:15: epochs 3-4 of scale-w64kd-gate-cached-ctx2048 (user go: "You can resume the work")

- Launched by the main session with the resume command from the PAUSED section above, from anki-rwkv, appending (`>>`) to `runbook-next-scale-ctx2048.log`.
- Gates e1/e2 re-ran from the cached scores (000375, 000376: GO, same numbers as above). Epoch 3 training is ticket 000377 (started 00:16:36, ~62 min at ~98k reviews/s). The runbook continues on its own: epoch 4, decay, scoring (gates run only after epochs 1-2, see runbook-next.sh line 192; no e3/e4 gate). Epoch 3 finished 01:10 (000377, exit 3, 115,273 reviews/s, 53.4 min); epoch 4 is 000378.
- When the run is final: a fresh Sonnet medium agent runs the "LAUNCHED ~16:40" follow-up with chunk 2048 (absolute LogLoss/RMSE vs fork w8+s8), then an Opus re-assessment of priorities. No new GPU work starts without the user's go.

## DONE 03:00: result of scale-w64kd-gate-cached-ctx2048 (CPU report)

464-user scoring finished (000380/000381). F (better = `-dec`, w8+s8) = +2.16% curve / +2.16% instant vs fork (e4: +2.61% / +2.48%); S (e2 − e4, 100 gate users) = 0.00236 / 0.00260, still learning; branch **E** (not P: F above +1.0% / +1.5%). G from Q2 is still open and only affects Q7. Table and details: docs/SMALL_MODEL.md "Result 2026-10-16 03:00". The `e3-extend.json` gate file is a dry-run artefact (mtime 00:38), ignored.

## QUEUE 2026-10-16 from 03:01 (main session, plan in docs/SMALL_MODEL.md "Queue 2026-10-16" and "Re-assessment 2026-10-16 ~03:30")

- Q1-Q3 ran from `artifacts/small-model/jobs/chain-q1-q3.sh` (log `artifacts/small-model/chain-q1-q3.log`; the "exit" it prints is always 0, read the `[queue] ... finished with exit code` lines): 000382 Q1 SM04 CPU timing (exit 0, `cpu-timing.json`, e1ba148), 000383 Q2 SM13 in-window@2048 (exit 0, `@w2048` rows in `heldout.jsonl`, G in ee73eb5), 000384 Q3 S06 (exit 1, output dir pre-created; fixed da60b34, requeued as 000386, log `artifacts/small-model/speed-ctx-20261016.log`, output `speed-ctx-20261016/`).
- Q4 extend (epochs 5-6, gate after e5, decay, 464-user scoring) was started at 03:20 by `artifacts/small-model/jobs/launch-q4-after-chain.sh` (log `launch-q4.log`); runbook log `artifacts/small-model/runbook-2026-10-16-extend.log`; epoch 5 = 000385. experiment.json / run/run.json now say 6 epochs (backups `*.bak-4ep`).
- Next: Q5 (S05 batched scoring, branch s05-batched) and Q6 (ctx4096 decay probe) after Q4, then Q7 per the 03:30 re-assessment (needs the `cont` runbook stage, Sonnet writer).
- Override record (audit 000385): the run's epoch count was raised 4 -> 6 by stage `extend` (experiment.json and run/run.json, backups `*.bak-4ep`), 66-min budget per epoch (epochs 1-4 took 53-57 min plus smoke). Reason: branch E of the 2026-10-16 queue rule (F = +2.16% / +2.16% misses P; S = 0.00236 / 0.00260 still learning; docs/SMALL_MODEL.md "Result 2026-10-16 03:00"). Authority: the user's delegation of 2026-10-15/16 ("I trust your judgement and I don't mind allowing you re-assess the situation and trigger new GPU run anyway", "Feel free to plan work until tomorrow 8PM"). LR stays constant 7e-4 (independent of the epoch count), so epochs 5-6 continue the same protocol; stop rule after e5 (e4 - e5 <= 0 on both heads).
- Q3 rerun 000386 (exit 0, S06, `speed-ctx-20261016/summary.md`; case ctx2048x32 OOM) and SM18 timing 000387 (exit 0, `cpu-timing-sm18.json`, branch sm18 merged 2542e88) were enqueued by the main session between epoch 5 and the e5 gate; write-ups f5fef5f. Q4 runs from `artifacts/small-model/runbook-2026-10-16.sh` stage `extend`; its e5 gate 000388 (exit 0) wrote `gate-scale-w64kd-gate-cached-ctx2048-e5-extend.json`: GO (curve gain -0.00040, instant +0.00087; stop only if both <= 0). Epoch 6 = 000389, then decay into `scale-w64kd-gate-cached-ctx2048-e6-decay`, 464-user scoring of `-e6`/`-dec6`. Then `jobs/chain-q5-q6.sh` (log `chain-q5-q6.log`, 951b08e) runs Q5 and Q6 (`cal-ctx4096-e6`); Q7 = `runbook-2026-10-16b.sh cont` (be994ba) per the 03:30 re-assessment.
- DONE 06:35: extended-run result (464 users) written to docs/SMALL_MODEL.md "Result 2026-10-16 06:35": F6 = dec6 +1.76% curve / +1.83% instant (misses P); dec6-dec paired -0.00118 curve / -0.00086 instant; S6 = 0.00010 / 0.00082 (flat); if Q6 fails, fallback 2 (Muon) per table.
