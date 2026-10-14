# Fast held-out evaluation (EV01)

`scripts/fast_evaluation.py` scores a checkpoint on a list of users and writes per-user metrics plus aggregates. It reproduces the full run's evaluation (`scripts/run_full_training.py` → `scripts/gpu_r01_training.run_case` with a `full_run` protocol) much faster. The original path stays available as `--mode reference`, and full runs keep using it unless `run.json` sets `"fast_evaluation": true`.

Status (2026-10-14): implemented, opt-in, committed. On the 300 quick-validation users the fast path takes **118.9 s instead of 965.9 s (8.1x; 186,641 vs 22,972 history reviews/s)**, with bitwise-identical inputs (`validation_input_sha256` equal to the full run's) but not bitwise-identical outputs: merging sections into one forward changes FP32 rounding (relative ~1e-7 per prediction). Per-user losses agree within 3.8e-8 absolute, aggregates within 5e-8, and the six-decimal benchmark metrics are equal for 299/300 users (LogLoss) and 278/300 (curve AUC, max 1.1e-5). On the 2,000-user final set (epoch 10) it takes **612 s instead of the run's 5,825 s (9.5x)** with the run's own `validation_input_sha256`. Because merged forwards are not exact, the path is opt-in, not the default. An exact mode (`--batch-chunks 8`, one section per forward) is bitwise identical to the reference on all 300 users and takes 722 s (1.34x).

## Use

```
.venv/Scripts/python.exe scripts/fast_evaluation.py --checkpoint <master.pth> \
    --users <run.json>:quick_validation_users --output <dir> [--model expertium|baseline] [--config module:ATTR] \
    [--mode fast|reference] [--workers 5] [--batch-chunks 512] [--input-hash] [--limit N] [--device cpu]
```

- `--users` takes `4,5,31`, `@users.json` (a JSON list) or `<run.json>:<key>`; join keys with `+` for a sorted union (`validation_users+final_test_users` is the final evaluation set). With a `run.json`, the dataset path, review counts and the `label-db` next to it are used; otherwise pass `--data-path` and `--label-db`.
- Output: `<dir>/result.json` with `per_user` (the same fields as a full-run `result.json`), `evaluation` (label-weighted and user-averaged Curve/Instant log loss, `scripts/run_gpu_batch1.aggregate`), `benchmark` (`scripts/full_run_metrics.aggregate`: LogLoss, RMSE(bins), AUC, RMSE per head, weighted and user-averaged), timing, reviews/s and provenance.
- From Python: `from scripts.fast_evaluation import evaluate_checkpoint; report = evaluate_checkpoint(checkpoint, users, output_dir, run-specific kwargs...)`, or `evaluate_fast(model, rows, label_db)` with a loaded FP32 model (what `run_case` calls).
- The CPU collection scorer case works directly: `--users 1 --data-path <collection-dataset> --label-db <collection-label-db> --device cpu --workers 0 --threads 2`.

## How the original evaluation works (profile)

Per user, in the main process: load the user's history (`get_rwkv_data` + `split_frame`), cut it into 1,024-review sections, prepare each section in a 5-worker spawn pool with the original Python `prepare(make_samples(...))` (seed `2025 + user + offset`), hash the inputs, copy each section to the GPU, run `master.get_loss` (FP32, TorchScript, default kernels) on that eight-chunk batch, loop in Python over every labelled position to fill prediction dictionaries, then compute the benchmark metrics with `get_stats` (Python loops, sklearn, pandas) for both heads. Every section forward synchronizes several times (NaN checks, four `.item()` counts, two loss reads, six `.cpu()` copies).

## What the fast path changes

| Stage | Original | Fast | Exactness |
|---|---|---|---|
| Sample construction | `make_samples` (pandas per chunk) | F30/F44/F45 `samples_for` (NumPy) | bitwise (check `inputs`) |
| Features and hierarchy splits | `prepare` (per-ID draws, nested loops) | F19 batched draws with the same RNG stream; split builder vectorized | bitwise per section (check `inputs`, and `--input-hash` reproduces `validation_input_sha256`) |
| Forward | one 8-chunk forward per 1,024-review section | sections from consecutive users share one forward (default 512 chunks), pipelined, no host syncs inside | see parity below |
| Per-user losses | per-section FP32 means, `.item() * count`, summed in Python | the same per-section FP32 sums on a fresh contiguous copy of each section's `[chunks, T]` block, same division and Python accumulation order | identical if the per-position values are identical |
| Prediction collection | Python loop over labelled positions into dicts | arrays gathered on the GPU, one read-back per batch; last-write-wins semantics kept | exact |
| Benchmark metrics | `get_stats` with dict/list inputs, main process | the same sklearn/pandas calls on arrays, in the worker pool, overlapped with the GPU | bitwise on 140 users with synthetic duplicates (check `metrics`) |

## Parity and timing

All runs: RTX 5080, torch 2.9.1+cu128, FP32 master, deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, graph-executor optimization off, 4 main-process threads, 5 preparation workers (the full run's `preparation_workers`), checkpoint `artifacts/full-run/run-8k-20261013/epoch-07/expertium/E89-F65e-F77a/master.pth`, the 300 `quick_validation_users` of `run.json` (22,188,946 history reviews, 21,821 sections, 38,929,906 scored Curve+Instant labels). Code `efebbf8` (dirty only by untracked job scripts and this file). Timing scope: model loaded, from the first preparation to the last per-user metric; the GPU was otherwise idle (queue ticket 000263); other sessions' CPU load was not controlled. Raw results: [artifacts/eval-speed/gpu/](../artifacts/eval-speed/gpu/) (`job1.sh`, `job1-queue.log`).

| Run | Evaluation s | History reviews/s | Labels/s | Speedup |
|---|---|---|---|---|
| Full run's own quick evaluation, epochs 7 / 8 / 9 (in-process, recorded) | 888.4 / 874.7 / 898.1 | 24,977 (epoch 7) | 43,822 | — |
| `--mode reference` (same code, standalone) | 965.9 | 22,972 | 40,303 | 1.0x |
| `--mode fast`, 512 chunks per forward | 118.9 | 186,641 | 327,453 | 8.1x |
| `--mode fast --input-hash` | 119.8 | 185,222 | 324,966 | 8.1x |
| Final 2,000-user evaluation in the full run (epoch 10, recorded) | 5,825.3 | — | — | — |

Settings sweep and the 2,000-user final set (queue ticket 000289; same machine and conditions):

| Run | Users | Workers / chunks | Evaluation s | History reviews/s | Labels/s | vs reference |
|---|---|---|---|---|---|---|
| fast, epoch 7 | 300 | 7 / 512 | 101.6 | 218,336 | 383,065 | 9.5x |
| fast, epoch 7 | 300 | 7 / 1024 | 89.6 | 247,626 | 434,454 | 10.8x |
| fast, epoch 7 | 300 | 10 / 512 | 89.0 | 249,306 | 437,400 | 10.9x |
| exact mode (one section per forward), epoch 7 | 300 | 7 / 8 | 722.3 | 30,721 | 53,899 | 1.34x, bitwise equal |
| fast `--input-hash`, epoch 10 | 2,000 | 7 / 512 | 612.4 | 238,080 | 412,712 | 9.5x vs the run's 5,825.3 s |
| fast, epoch 7 | 2,000 | 7 / 512 | 604.0 | 241,400 | 418,471 | — |

The 2,000-user set (`validation_users+final_test_users`) has 145,795,912 history reviews and 252,739,888 labels; the run's in-process final evaluation scored them at 25,028 reviews/s. In every fast run the main process mostly waits for preparation (for example 450 s of 612 s on 2,000 users, with 3,826 s of worker preparation), so more workers or larger batches help until the CPU saturates; GPU read-back waits stay below 0.2 s. 7 workers with 1,024 chunks together were not measured.

Fast vs the run's recorded epoch-10 final evaluation (2,000 users, `compare-final2000-recorded-vs-fast.json`): input hash identical (`41aa2ffa…`); label counts, sizes and coverage identical for every user; six-decimal benchmark metrics identical for 1,994–1,999 of 2,000 users (curve LogLoss 1,994, instant LogLoss 1,998, RMSE(bins) 1,999/2,000), AUC for 1,833 (curve, max 1.1e-5) and 1,947 (instant, max 2e-6); R01 loss fields max |Δ| 6.9e-8; aggregate label-weighted and user-averaged Curve/Instant log loss within 2.0e-10.

The exact mode (`compare-quick300-reference-vs-c8.json`): every per-user field and every aggregate of all 300 users is identical to `--mode reference` (and therefore to the run). It is launch-bound (683 s of launches for 21,821 forwards) and saves the main-process history loads, input hash, prediction loops and metrics.

### Profile of the original path (300 users, host timers in `--mode reference`)

| Stage | Seconds | Share |
|---|---|---|
| `get_loss` (section forward, launch-bound, several syncs) | 656.0 | 68% |
| Benchmark metrics (`get_stats`, Python loops, main process) | 155.3 | 16% |
| History load (`get_rwkv_data` + `split_frame`, main process) | 34.4 | 4% |
| Host-to-device copies (about 50 small tensors per section) | 30.6 | 3% |
| Waiting for preparation workers | 29.3 | 3% |
| `collect_predictions` (Python loop per labelled position) | 26.3 | 3% |
| Input hash | 12.2 | 1% |

21,821 forwards of eight chunks average 30 ms each; the GPU is mostly idle. In the fast path the main process waits on the workers 91 s of 119 s (the five workers did 498 s of preparation), launches take 26.5 s and GPU read-back waits are negligible: it is now bound by CPU preparation, not the GPU.

### Exactness evidence

| Check | Scope | Result | Evidence |
|---|---|---|---|
| Reference reproduces the run | 300 users, all per-user fields and aggregates vs epoch-07 `result.json` | 300/300 users identical; same `validation_input_sha256` (`5cb93a07…`) | `compare-recorded-vs-reference.json` |
| Inputs | per section: `build_batch(samples_for, section_features)` vs `prepare(make_samples)`: start, labels, review_th, every gather/shift/skip tensor, split lengths | 104 + 23 + 15 sections of 31 users bitwise equal (incl. last, short sections); packed IPC form equal | `artifacts/eval-speed/checks/inputs-*.json` |
| Inputs at full scale | fast path `--input-hash` over all 21,821 sections | equal to the run's recorded `validation_input_sha256` | `quick300-fast-hash/result.json` |
| Benchmark metrics | `benchmark_from_arrays` vs `full_run_metrics.benchmark_metrics` on real label-db users with synthetic predictions, repeated keys and missing reviews | 140/140 users identical (dict output compared as JSON) | `checks/metrics-*.json` |
| Forward per position | 64 sections of users 4, 5, 31: one section per forward vs all 64 in one forward (503 chunks) | not bitwise: `out_w` differs at nearly all positions, max 3.6e-7; p_curve equal at 58%, p_imm at 76%, max 3.6e-7 | `forward-quick3.json` |
| End to end | 300 users fast vs reference | see below | `compare-quick300.json` |

Fast vs reference per user (300 users): label counts, coverage, sizes, RMSE and RMSE(bins) for both heads identical for every user; instant LogLoss identical for 300; curve LogLoss for 299 (one differs by 1e-6, the six-decimal rounding step); AUC for 278 (curve, max 1.1e-5) and 295 (instant, max 1e-6). The R01 loss fields (computed in full precision, not rounded) differ for nearly every user: max |Δ| 3.8e-8 (curve), 3.4e-8 (curve raw), 1.9e-8 (instant), 5.3e-8 (rating cross-entropy); max relative 1.4e-6. Aggregates: label-weighted Curve 0.3603063603 vs 0.3603063600 (Δ 3.0e-10), Instant 0.3447268213 vs 0.3447268212 (Δ 6.8e-11); benchmark weighted Curve LogLoss Δ 2.9e-10, AUC Δ ≤ 5.0e-8.

Why not bitwise: the inputs are identical, and every per-section reduction is replayed on the original shape, so the differences come from the forward itself. With more rows in one forward, the FP32 GEMMs (cuBLAS chooses kernels by matrix size) round differently; the CPU shows the same effect with MKL. The tolerance is therefore the float reduction-order noise of a different GEMM shape: about 1e-7 relative per prediction, 1e-6 on six-decimal metrics (one rounding step) and up to 1e-5 on a single user's AUC when near-tied predictions swap rank. This is far below run-to-run differences of interest (the screening deltas in `docs/GPU_OPTIMIZATIONS.md` are 1e-4 and larger), but it is not parity, so the fast path stays opt-in and recorded runs compared with exact parity must use `--mode reference` or the exact mode below.

## Promotion campaign (2026-10-14)

Requested to decide whether fast evaluation may become the full-run default. Every row compares `--mode fast` (512 chunks per forward, input hash on) with a reference on the same checkpoint and users: the full run's own recorded `result.json` (written by the original in-run evaluation) where one exists, otherwise `--mode reference` on the same machine. Max |Δ| per user covers the four R01 losses and the six-decimal benchmark LogLoss, RMSE(bins) and RMSE of both heads; AUC is listed separately. Aggregates cover the R01 log losses (label-weighted and user-averaged) and the benchmark aggregates (LogLoss, RMSE(bins), AUC, RMSE per head, review-weighted and user-averaged; recomputed with `full_run_metrics.aggregate` for recorded results). Evidence: `artifacts/eval-speed/promotion/` and `artifacts/eval-speed/gpu/` (`compare-*.json`, field `tolerance`).

| Checkpoint / architecture | Users | Max aggregate abs Δ | Max per-user abs Δ (losses) | Max per-user abs Δ (AUC) | Input hash equal | Reference s | Fast s | Passes proposed tolerance |
|---|---|---|---|---|---|---|---|---|
| iter36 epoch 01 | 300 | 6.7e-9 | 1.0e-6 | 4e-6 | yes | 895.2 (recorded) | 100.8 | yes |
| iter36 epoch 02 | 300 | 2.3e-8 | 1.0e-6 | 8e-6 | yes | 828.9 (recorded) | 102.6 | yes |
| iter36 epoch 03 | 300 | 1.8e-8 | 1.0e-6 | 2e-6 | yes | 829.9 (recorded) | 100.5 | yes |
| iter36 epoch 04 | 300 | 3.0e-8 | 1.0e-6 | 5e-6 | yes | 827.3 (recorded) | 99.8 | yes |
| iter36 epoch 05 | 300 | 8.1e-8 | 1.0e-6 | 1.6e-5 | yes | 822.7 (recorded) | 99.6 | yes |
| iter36 epoch 06 | 300 | 7.7e-8 | 1.0e-6 | 1.0e-5 | yes | 831.6 (recorded) | 99.7 | yes |
| iter36 epoch 07 | 300 | 5.0e-8 | 1.0e-6 | 1.1e-5 | yes | 888.4 (recorded); 965.9 (standalone) | 119.8 (5 workers) | yes |
| iter36 epoch 08 | 300 | 2.7e-8 | 1.0e-6 | 1.0e-5 | yes | 874.7 (recorded) | 100.2 | yes |
| iter36 epoch 09 | 300 | 2.0e-8 | 1.0e-6 | 4e-6 | yes | 898.1 (recorded) | 99.6 | yes |
| iter36 epoch 10, final set | 2,000 | 2.0e-9 | 1.0e-6 | 1.1e-5 | yes | 5,825.3 (recorded) | 612.4 | yes |
| upstream 2.76M baseline (`RWKV_trained_on_5000_10000.pth`), fork-unseen users < 5000 | 100 | 5.0e-8 | 3.8e-8 | 1e-6 | yes | 335.5 (standalone) | 38.7 | yes |
| EV07 full-run smoke, iter36 trained 1 epoch on 50 users, `fast_evaluation` false vs true | 100 | P3-AGG | P3-LOSS | P3-AUC | P3-HASH | P3-REF | P3-FAST | P3-PASS |

Epochs 01–09 used 7 workers; epoch 07's fast row is the 5-worker run of job 1. The per-user loss maximum of 1.0e-6 is always the six-decimal rounding step of a benchmark LogLoss or RMSE (one to two users per checkpoint); the unrounded R01 losses differ by at most 9.6e-8 (epoch 02). In total 4,700 iter36 user-evaluations over 10 checkpoints plus 100 baseline user-evaluations, all with identical inputs and label counts.

Not covered: the Expertium width-80 variant on branch `expertium-catch-up` (`expertium/model/variant_model.py`) has a different head (GRU forgetting curve, no ahead residual, its own `_get_loss_variant`). `loss_terms` replicates only the iter36/upstream `_get_loss`, so `evaluate_fast` refuses other model classes (`supports_fast`), and a full run with such a model falls back to the reference evaluation and records `evaluation_mode` accordingly. Supporting it needs a variant-specific per-position term function (EV08).

## Proposed evaluation-tolerance rule

PROPOSED 2026-10-14, awaiting the human decision; not in force. The exact wording for `AGENT.md` and `docs/GPU_RETRAIN_LOOP.md` is the patch [proposals/eval-tolerance-rule.diff](proposals/eval-tolerance-rule.diff) (`git apply docs/proposals/eval-tolerance-rule.diff` once approved). `scripts/check_fast_evaluation.py compare` reports the check as `tolerance.passes`.

- Training changes still need bit-identical parity.
- The evaluation path may differ from the reference only by floating-point reduction order, with identical `validation_input_sha256`, identical label counts, benchmark sizes and coverage, and:

| Quantity | Proposed limit | Largest observed (P1–P3) | Margin |
|---|---|---|---|
| Any aggregate metric | ≤ 1e-6 | 8.1e-8 (epoch 05, benchmark AUC user-average) | 12x |
| Per-user loss (R01 losses; benchmark LogLoss, RMSE(bins), RMSE) | ≤ 1e-5 | 1.0e-6 (six-decimal rounding step); 9.6e-8 unrounded | 10x |
| Per-user AUC | ≤ 1e-4 | 1.6e-5 (epoch 05) | 6x |

- AUC caveat: AUC is a rank statistic. A 1e-7 change in a prediction can swap two near-tied predictions, so a single user's six-decimal AUC moves in steps of about 1e-5 even though every probability moved by 1e-7; that is why per-user AUC gets its own, wider limit. Aggregate AUC stays under the 1e-6 aggregate limit.
- Every result records the evaluation mode and commit (`mode`, `batch_chunks` and `provenance` in `fast_evaluation.py` results; `evaluation_mode` and `evaluation_batch_chunks` in full-run results). `--mode reference` and the exact `--batch-chunks 8` mode stay available for bit-identical claims.
- The limits sit far below the quality gates the loops use (R01 non-exact ceilings of 0.1%–1% relative, about 3e-4 to 3e-3 on these log losses; the accuracy loop's measured seed-noise bands) and above the observed reduction-order noise.

## Which mode to use

- Screening and repeated measurements of new checkpoints: `--mode fast` (default), preferably `--workers 7` (or 10) and `--batch-chunks 1024` when nothing else needs the CPU. Differences to the original are float noise (above), identical for both arms of a comparison run the same way.
- Numbers that must equal a recorded run bit for bit (parity claims, recorded protocols): `--mode reference`, or `--batch-chunks 8` (exact, 1.34x).
- Full runs: unchanged unless `run.json` sets `"fast_evaluation": true` (optional `"fast_evaluation_chunks"`, 8 for the exact mode). `run_full_training.write_protocol` then passes `full_run.fast_evaluation`, and `run_case` calls `evaluate_fast` with the run's preparation workers; `validation_input_sha256` is computed by the fast path (`input_hash=True`, measured cost under 1 s on 300 users).

## CPU collection scorer

`artifacts/collection-eval/2026-10-14/score_collection.py` (gitignored, unchanged) scores one user on CPU with the window protocol: 52.4 s for epoch 8 (48.9 s model, prepared inputs cached). The same checkpoint through `scripts/fast_evaluation.py --users 1 --data-path <collection-dataset> --label-db <collection-label-db> --device cpu --workers 0 --threads 2 --batch-chunks 512` took 39.2 s including preparation, with all eight benchmark metrics identical and R01 losses within 3e-10 (CPU GEMMs also round differently by batch shape). A modest win; the CPU recurrence dominates. Evidence: `artifacts/eval-speed/collection-cpu/`.

## Untested and follow-up ideas

Recorded as EV02–EV08 in [GPU_OPTIMIZATIONS.md](GPU_OPTIMIZATIONS.md#optimization-tracks).

- EV02: exact merged forward. The per-position differences come from shape-dependent FP32 kernels; padding or chunking each GEMM to the original section row counts, or row-invariant kernels, could give exact results with merged batching. Pending, needs a kernel-level check of which ops differ.
- EV03: more preparation throughput. The fast path is CPU-bound; try 7 workers with 1,024 chunks, a prestarted pool shared with training (F59), and caching prepared evaluation inputs across epochs (they do not depend on the checkpoint; about 0.8 kB per review, so tens of GB for the 2,000-user set).
- EV04: GPU-side feature encodings for evaluation (the training path's `gpu_packed_features`) would cut worker CPU and the ~50 MB per batch transfer, but GPU `sin/cos` differ from the CPU `prepare` inputs: inputs would no longer be identical.
- EV05: BF16 evaluation forward: faster on the GPU, which is not the bottleneck; changes results; screening only.
- EV06: `scripts/fast_evaluation.py` is now in `R01SourceWatch` (`scripts/run_gpu_r01.py`, listed only when present in the source archive, so older archives keep their watch set).
- EV07: a full-run smoke with `"fast_evaluation": true` to check the `run_case` wiring end to end (done in the promotion campaign above).
- EV08: per-position terms for Expertium variant heads (width-80 GRU curve) so they can use the fast path.
