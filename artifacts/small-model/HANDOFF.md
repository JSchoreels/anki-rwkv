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
