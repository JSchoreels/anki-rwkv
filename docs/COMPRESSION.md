# Compressing the fork's 2.76M RWKV

Status 2026-10-14 (agent session `compression`, branch `compression`): post-training quantization (PTQ) measured on the user's collection and a CPU held-out screen; quantization-aware training (QAT) implemented and ready to run but **not yet run** (the GPU stayed busy with the 8k/2k full run, see [Next steps](#ready-to-run-qat)). Raw results: [artifacts/compression/](../artifacts/compression/).

## Summary

- **The fork's "int8" is not weight quantization.** Anki fork commit `d79d1ad` (2026-10-10, `rslib/src/rwkv/precision.rs`) compresses the stored per-card and per-note *recurrent state*. The weights, activations, arithmetic and deck/preset/global state stay float32; the shipped weight file `RWKV_trained_on_5000_10000.bin` is 11,082,637 bytes of float32 (2,762,884 parameters). The "<1%" figure is the fork's own replay of the collection (`docs/rwkv-state-precision-2026-10-10.MD` in the fork): log loss 0.45573652 → 0.45573628 (−0.00005%) for int8 state, +0.068% for int4 state, on 176,722 reviews at least one day apart; the state store shrinks by 68% (int8) / 80% (int4).
- **Weight PTQ is almost free down to 4 bits.** On the collection (full history, 173,690 benchmark test reviews) int8 weights change nothing measurable (≤ +0.02%), int4 group-32 costs +0.02% curve / +0.11% instant at 1.89 MB (5.8× smaller than 11.05 MB), int4 per-channel +0.02% / +0.42% at 1.64 MB. PTQ breaks at 3-bit per-channel (+11.8% instant) and ternary (+15–17%).
- **The fork's state compression composes with weight PTQ.** Emulating the fork's int8 card/note state in Python reproduces its result (±0.00%), and int4 group-32 weights plus int8 state equal int4 weights alone.
- **Held-out users agree but are more sensitive on the instant head** (CPU screen, small users only): int4 group-32 +0.0% curve / +0.4% instant, int4 per-channel −0.0% / +1.5%, 3-bit group-32 +0.1% / +1.0%.
- **Recommendation:** ship int4 group-32 (or group-64) PTQ weights with int8 state now; it needs no training. QAT is only worth running for ≤ 3-bit or per-channel int4 targets, and the fork's 2.76M model at 4 bits (1.6–1.9 MB) is still 2–2.5× larger than the 192.8k Expertium model in float32 (0.77 MB) while ~3% better on the collection; see [Recommendation](#recommendation).

## 1. The fork's int8 scheme (read from `C:\Users\jonat\workspace\anki`, read-only)

| Aspect | Fork int8 / int4 ("RWKV state" preference, default int8) |
|---|---|
| Tensors | Card and note modules' WKV state matrices (4 heads × 32 × 32 per layer; card 3 layers, note 2). Time-shift and channel-shift vectors of those modules are stored as IEEE f16. Deck, preset and user state stay f32. No weights, embeddings or activations are quantized. |
| Granularity | One f32 scale per 32-element matrix row (row = value index, columns = key index; the decay multiplies columns) |
| Codes | Symmetric signed, abs-max scale (`max/127` or `max/7`), round-half-to-even, int8 −127…127, int4 −7…7 packed two per byte |
| When | After every answer, once that answer's f32 output is computed; the next answer reads the rounded state. Packed codes and scales are kept, so checkpoints add no second rounding. |
| Bytes per entity | Card 52,224 B (f32) → 15,360 B (int8) → 9,216 B (int4); note 34,816 → 10,240 → 6,144 (headers excluded) |
| Measured by the fork | Rust runtime replay of a 2026-10-10 collection backup: log loss −0.00005% (int8) and +0.068% (int4), RMSE(bins) +0.009% / +5.70%, largest single-prediction change 0.68 / 14.1 percentage points; state store 1,247 → 395 / 253 MB |

There is also a test-only experiment in `rslib/src/rwkv/mod.rs` (`StateCompression`: low-rank rank-2/4 state with int8/int4 factors and int4 shifts) that is not wired into the runtime.

## 2. Method

- Weights: the fork's `RWKV_trained_on_5000_10000.pth` (identical to the Rust `.bin`, checked in the 2026-10-14 collection evaluation), scored with this repo's vendored upstream `rwkv/` model in float32.
- Weight PTQ (`scripts/compress_quant.py`): every `nn.Linear` weight (265 tensors, 2,719,744 of 2,762,884 parameters) is quantized symmetrically with abs-max scales and round-half-to-even, like the fork's state codes; ternary (2-bit, codes −1…1) uses the BitNet b1.58 abs-mean scale. Groups are runs of `g` input columns of one output row; `pc` is one scale per output row. Biases, LayerNorm/GroupNorm parameters and lerp/bonus vectors (43,140 values) stay float32. Bytes = packed codes + one f32 scale per group + float32 other parameters; the table also gives bytes with f16 scales and f16 other parameters.
- State emulation: a patched copy of `rwkv/model/rwkv_model.py` (generated at runtime, the vendored file is unchanged) rounds the card and note WKV state per 32-element row after every step of the CPU reference recurrence and rounds their shift vectors to f16, as the fork does. With state precision off it reproduces the float32 window-protocol scores exactly (curve 0.468221, instant 0.469254, the 2026-10-14 Model B values).
- Collection: the user's collection as converted for the 2026-10-14 evaluation (main checkout `artifacts/collection-eval/2026-10-14`, read-only), **full-history protocol** (the fork's replay and B's training context), 173,690 FSRS-6-filtered benchmark test reviews, upstream `get_stats`. CPU, 2 threads.
- Held-out users: validation users of the 8k/2k run (`run.json`) outside dataset users 5000–10000 (the fork's training users): 732 users. The CPU screen scored the first users (by ID) with at most 20,000 reviews of a seeded sample, full-history protocol, benchmark labels from the run's label DB, as a light screen while the full run held the GPU. It is biased towards small histories and is not a validation (AGENT.md requires a user-requested ≥ 500-user run).

## 3. Results

### Collection (full history, 173,690 reviews; Δ% vs fork fp32 = curve 0.4417 / instant 0.4366)

"Fork int8" is the fork's actual int8 option (float32 weights, int8 card/note state); because it equals fp32 to four decimals, Δ% vs fork int8 is the same as Δ% vs fp32 within ±0.01 points.

COLLECTION_TABLE

Expertium iter36 (192.8k parameters, 771,200 B float32) on the same collection and protocol, from the main checkout's `collection-epochs.jsonl`: epoch 7 curve 0.4546 (+2.9%) / instant 0.4525 (+3.6%), epoch 8 0.4569 / 0.4576.

### Held-out users (CPU screen, full history)

HELDOUT_TABLE

## 4. QAT

Implemented in `scripts/compress_qat.py`, not yet run. Starting point: the fork weights. Data: training users of the 8k/2k run outside 5000–10000 (3,969 users, 286M reviews), so held-out users stay unseen and the fork's own training users are not revisited. Contiguous history segments of ≤ 16,384 reviews (upstream trains on ≤ 32,768) packed into batches of ≤ 49,152 reviews. Upstream training recipe (fp32 master, bf16 child, the upstream AdamW parameter groups, clip 0.5, loss = `average_loss`); fake quantization of every Linear weight in the child each step with scales recomputed from the master weights, straight-through gradients. LR 1e-4, 30-step linear warmup, cosine decay to 0 over the time budget (2,400 s). `--bits 0` is the matched fine-tune-only control, needed because fine-tuning on new users can move the scores by itself.

Data pipeline dry run (CPU, one worker): about 1 s per 49k-review batch. GPU memory, throughput (reviews/s) and losses are unmeasured.

### Ready-to-run QAT

Run from the worktree through the shared queue once the GPU is free (each job < 60 min):

```
Q=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
$Q run --owner compression --purpose "QAT smoke (40 steps) + 6-user held-out smoke" --claim gpu -- bash artifacts/compression/jobs/smoke.sh
$Q run --owner compression --purpose "QAT int4 per-channel, 40 min" --claim gpu -- bash artifacts/compression/jobs/qat.sh 4 0 w4pc-run1
$Q run --owner compression --purpose "fine-tune control (no quantization), 40 min" --claim gpu -- bash artifacts/compression/jobs/qat.sh 0 0 fp32-control1
$Q run --owner compression --purpose "held-out GPU eval, 300 users" --claim gpu -- bash artifacts/compression/jobs/heldout-gpu.sh qat-w4pc=artifacts/compression/qat/w4pc-run1/final.pth ft-control=artifacts/compression/qat/fp32-control1/final.pth
```

Then on CPU: `scripts/compress_score_collection.py --checkpoint artifacts/compression/qat/w4pc-run1/final.pth --tag qat-w4pc-run1 --variants w4-pc,fp32` (and the same for the control with `--variants fp32`), then `scripts/compress_report.py`. Check the smoke's `peak` MiB first; if it exceeds ~13 GiB, halve `--batch-reviews`.

## Recommendation

RECOMMENDATION

## Backlog

| ID | Idea | Status | Next step |
|---|---|---|---|
| Q01 | int4 group-32/64 PTQ weights in the fork's Rust runtime (with the existing int8 state) | Measured in Python only | Port: pack nibbles + f32 (or f16) group scales, dequantize per matmul tile; re-check the collection replay in Rust |
| Q02 | QAT int4 per-channel (recover the +0.4% collection / +1.5% held-out instant gap) | Implemented, pending GPU | Run the four queue jobs above |
| Q03 | QAT ternary / 3-bit (1.0–1.3 MB, near Expertium float32 size) | Pending; PTQ fails (+12–17%) | `qat.sh 2 32 w2g32-run1`, then `qat.sh 3 0 w3pc-run1` |
| Q04 | QAT / PTQ of Expertium width-80 (being added in `wt/expertium-catch-up`) | Pending | Apply `compress_quant.fake_quant` to its Linear weights; int8 should be free as here |
| Q05 | Distil the 2.76M fork model into Expertium-sized students | Pending | Use the fork model's curve/instant probabilities as soft targets on the run's train users |
| Q06 | State quantization for deck/preset/user modules and low-rank state (`StateCompression` in the fork's tests) | Pending | Extend `set_state_bits` to all modules; measure memory vs accuracy |
| Q07 | f16 storage of scales and non-Linear parameters | Accounted only (table column) | Confirm no accuracy change (expected) |
| Q08 | Better PTQ: MSE-optimal clipping / GPTQ-style error feedback for 3-bit | Pending | Cheap CPU experiment on the collection before QAT |
