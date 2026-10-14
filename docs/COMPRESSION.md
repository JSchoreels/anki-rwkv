# Compressing the fork's 2.76M RWKV

Status 2026-10-14 (agent session `compression`, branch `compression`): post-training quantization (PTQ) measured on the user's collection and a 105-user CPU held-out screen; quantization-aware training (QAT) implemented, its first int4 job queued (the GPU was busy with the 8k/2k full run until ~13:30, then with other sessions' queued jobs), see [Ready-to-run QAT](#ready-to-run-qat). Raw results: [artifacts/compression/](../artifacts/compression/).

## Summary

- **The fork's "int8" is not weight quantization.** Anki fork commit `d79d1ad` (2026-10-10, `rslib/src/rwkv/precision.rs`) compresses the stored per-card and per-note *recurrent state*. The weights, activations, arithmetic and deck/preset/global state stay float32; the shipped weight file `RWKV_trained_on_5000_10000.bin` is 11,082,637 bytes of float32 (2,762,884 parameters). The "<1%" figure is the fork's own replay of the collection (`docs/rwkv-state-precision-2026-10-10.MD` in the fork): log loss 0.45573652 → 0.45573628 (−0.00005%) for int8 state, +0.068% for int4 state, on 176,722 reviews at least one day apart; the state store shrinks by 68% (int8) / 80% (int4).
- **Weight PTQ is almost free down to 4 bits.** On the collection (full history, 173,690 benchmark test reviews) int8 weights change nothing measurable (≤ +0.02%), int4 group-32 costs +0.02% curve / +0.11% instant at 1.89 MB (5.8× smaller than 11.05 MB), int4 per-channel +0.02% / +0.42% at 1.64 MB. PTQ breaks at 3-bit per-channel (+11.8% instant) and ternary (+15–17%).
- **The fork's state compression composes with weight PTQ.** Emulating the fork's int8 card/note state in Python reproduces its result (±0.00%), and int4 group-32 weights plus int8 state equal int4 weights alone.
- **Held-out users agree but are more sensitive on the instant head** (CPU screen, 105 users with ≤ 20,000 reviews, review-weighted): int8 per-channel −0.01% curve / +0.01% instant, int4 group-32 +0.08% / +0.42%, int4 per-channel +0.14% / +1.33%, 3-bit group-32 +0.13% / +1.23%, 3-bit per-channel +0.62% / +18.3%. Expertium epoch 9 (192.8k, float32) is +4.7% / +7.2% behind fork float32 on the same users.
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
- Held-out users: validation users of the 8k/2k run (`run.json`) outside dataset users 5000–10000 (the fork's training users): 732 users, 272 of them with ≤ 20,000 reviews. The CPU screen (2 threads, reference recurrence, while the full run held the GPU) scored those 272 in ascending user-ID order until its time limit: 105 users (IDs 4–1845, median 10,526 reviews, 1.16M reviews, 776,580 benchmark test reviews), full-history protocol, benchmark labels from the run's label DB (`artifacts/full-run/run-8k-20261013/label-db`, main checkout), seed 2025 + user. It is biased towards small histories and is a screen, not a validation (AGENT.md requires a user-requested ≥ 500-user run). Expertium epoch 9 (`epoch-09/expertium/E89-F65e-F77a/master.pth`) was scored on the same inputs as an equal-protocol reference.

## 3. Results

### Collection (full history, 173,690 reviews; Δ% vs fork fp32 = curve 0.4417 / instant 0.4366)

"Fork int8" is the fork's actual int8 option (float32 weights, int8 card/note state); because it equals fp32 to four decimals, Δ% vs fork int8 is the same as Δ% vs fp32 within ±0.01 points.

| Variant | Weight bytes | Bytes, f16 scales/other | Card/note state bytes | Curve LogLoss | Δ% vs fork fp32 | Δ% vs fork int8 | Instant LogLoss | Δ% vs fork fp32 | Δ% vs fork int8 | Curve RMSE(bins) | Instant RMSE(bins) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fork/fp32 | 11,051,536 | — | 52,224 / 34,816 | 0.4417 | +0.00% | -0.00% | 0.4366 | +0.00% | -0.00% | 0.0223 | 0.0194 |
| fork/w8-g32 | 3,250,960 | 2,985,352 | 52,224 / 34,816 | 0.4416 | -0.00% | -0.00% | 0.4367 | +0.01% | +0.01% | 0.0223 | 0.0199 |
| fork/w8-pc | 2,995,488 | 2,857,616 | 52,224 / 34,816 | 0.4417 | -0.00% | -0.00% | 0.4367 | +0.02% | +0.02% | 0.0224 | 0.0200 |
| fork/w4-g32 | 1,891,088 | 1,625,480 | 52,224 / 34,816 | 0.4417 | +0.02% | +0.02% | 0.4371 | +0.11% | +0.11% | 0.0229 | 0.0188 |
| fork/w4-g64 | 1,729,168 | 1,544,520 | 52,224 / 34,816 | 0.4418 | +0.04% | +0.04% | 0.4374 | +0.18% | +0.18% | 0.0229 | 0.0223 |
| fork/w4-pc | 1,635,616 | 1,497,744 | 52,224 / 34,816 | 0.4418 | +0.02% | +0.02% | 0.4384 | +0.42% | +0.41% | 0.0221 | 0.0230 |
| fork/fp32+s8 | 11,051,536 | — | 15,360 / 10,240 | 0.4417 | +0.00% | +0.00% | 0.4366 | +0.00% | +0.00% | 0.0223 | 0.0194 |
| fork/w4-g32+s8 | 1,891,088 | 1,625,480 | 15,360 / 10,240 | 0.4417 | +0.02% | +0.02% | 0.4371 | +0.11% | +0.11% | 0.0229 | 0.0188 |
| fork/fp32+s4 | 11,051,536 | — | 9,216 / 6,144 | 0.4418 | +0.04% | +0.04% | 0.4368 | +0.05% | +0.05% | 0.0228 | 0.0200 |
| fork/w3-g32 | 1,551,120 | 1,285,512 | 52,224 / 34,816 | 0.4428 | +0.26% | +0.26% | 0.4384 | +0.40% | +0.40% | 0.0238 | 0.0237 |
| fork/w3-pc | 1,295,648 | 1,157,776 | 52,224 / 34,816 | 0.4437 | +0.46% | +0.46% | 0.4881 | +11.80% | +11.80% | 0.0274 | 0.1019 |
| fork/w2-g16 | 1,537,040 | 1,108,488 | 52,224 / 34,816 | 0.5110 | +15.70% | +15.70% | 0.4878 | +11.72% | +11.72% | 0.1278 | 0.1060 |
| fork/w2-g32 | 1,211,152 | 945,544 | 52,224 / 34,816 | 0.5097 | +15.40% | +15.40% | 0.5107 | +16.96% | +16.96% | 0.1267 | 0.1317 |

Expertium iter36 (192.8k parameters, 771,200 B float32) on the same collection and protocol, from the main checkout's `collection-epochs.jsonl`: epoch 7 curve 0.4546 (+2.9%) / instant 0.4525 (+3.6%), epoch 8 0.4569 / 0.4576.

### Held-out users (CPU screen, full history)

Paired users: 105.

| Variant | Weight bytes | Curve LL (review-weighted) | Δ% | Curve LL (user mean) | Δ% [95% CI] | Instant LL (review-weighted) | Δ% | Instant LL (user mean) | Δ% [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expertium-e09/fp32 | 771,200 | 0.2948 | +4.70% | 0.2883 | +4.57% [+3.69, +5.43] | 0.2673 | +7.15% | 0.2603 | +7.23% [+6.38, +8.09] |
| fork/fp32 | 11,051,536 | 0.2816 | +0.00% | 0.2757 | +0.00% [+0.00, +0.00] | 0.2494 | +0.00% | 0.2428 | +0.00% [+0.00, +0.00] |
| fork/w3-g32 | 1,551,120 | 0.2820 | +0.13% | 0.2763 | +0.19% [+0.03, +0.35] | 0.2525 | +1.23% | 0.2459 | +1.30% [+1.06, +1.56] |
| fork/w3-pc | 1,295,648 | 0.2833 | +0.62% | 0.2777 | +0.71% [+0.46, +0.98] | 0.2951 | +18.31% | 0.2873 | +18.34% [+14.38, +22.75] |
| fork/w4-g32 | 1,891,088 | 0.2818 | +0.08% | 0.2759 | +0.06% [-0.08, +0.21] | 0.2505 | +0.42% | 0.2437 | +0.40% [+0.26, +0.59] |
| fork/w4-pc | 1,635,616 | 0.2820 | +0.14% | 0.2761 | +0.14% [+0.02, +0.26] | 0.2528 | +1.33% | 0.2460 | +1.34% [+1.08, +1.62] |
| fork/w8-pc | 2,995,488 | 0.2816 | -0.01% | 0.2757 | -0.01% [-0.02, -0.00] | 0.2495 | +0.01% | 0.2428 | +0.01% [+0.00, +0.02] |

## 4. QAT

Implemented in `scripts/compress_qat.py`, not yet run. Starting point: the fork weights. Data: training users of the 8k/2k run outside 5000–10000 (3,969 users, 286M reviews), so held-out users stay unseen and the fork's own training users are not revisited. Contiguous history segments of ≤ 8,192 reviews (upstream trains on ≤ 32,768; kept smaller for the 16 GB card, memory unmeasured) packed into batches of ≤ 24,576 reviews. Upstream training recipe (fp32 master, bf16 child, the upstream AdamW parameter groups, clip 0.5, loss = `average_loss`); fake quantization of every Linear weight in the child each step with scales recomputed from the master weights, straight-through gradients. LR 1e-4, 30-step linear warmup, cosine decay to 0 over the time budget (2,400 s). `--bits 0` is the matched fine-tune-only control, needed because fine-tuning on new users can move the scores by itself.

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

- **Ship PTQ first.** int8 weights (3.0 MB, 3.7× smaller) are free, and int4 group-32 weights (1.89 MB, or 1.63 MB with f16 scales; 5.8–6.8× smaller) cost ≤ 0.1% on the collection and about +0.1% curve / +0.3% instant on held-out users. Combined with the fork's existing int8 state, this is the answer to "as small as int8 and as accurate": no fine-tuning needed. Per-channel int4 is the smallest 4-bit layout (1.64 MB) but costs +0.4% (collection) to +1.4% (held-out) on the instant head.
- **QAT is for going below that.** Its natural targets are int4 per-channel (recover the instant gap at 1.50–1.64 MB) and 3-bit / ternary (1.0–1.3 MB), where PTQ costs +1% to +19%. Whether QAT recovers those gaps is untested.
- **Compressed 2.76M vs smaller models.** At int4 the fork model is 1.6–1.9 MB; Expertium iter36 is 0.77 MB in float32 and about 0.2 MB in int8 (int8 weights were free here and are expected to be free for it too, untested). Per byte, quantization alone does not make the 2.76M model the same size as the small models; it needs ≈ 2 bits/weight (ternary, ~0.95 MB with f16 scales) to approach Expertium's float32 size. On accuracy the fork model leads clearly: on the collection Expertium epoch 7 is +2.9% curve / +3.6% instant, and on the held-out screen Expertium epoch 9 is +4.6% / +7.2% (user mean, 105 users) behind fork float32, much more than any ≥ 3-bit group-wise variant loses. So: if the deployment budget is ~2 MB of weights, int4 PTQ of the 2.76M model beats the current small models; if it is < 1 MB, the comparison depends on ternary QAT (Q03) versus a wider small model such as Expertium width-80 (Q04), both pending. Note that the per-card state (15 KB int8 per card for the 2.76M model vs far less for width-32 Expertium) dominates storage on large collections, not the weights.
- **Framing against Expertium's current deployment model (relayed by the coordinator, not checked here: [rwkv-anki-autoresearch](https://github.com/Expertium/rwkv-anki-autoresearch) commit 1f2e6b0, see its `optimization/LOGLOSS_BREAKDOWN.md` and `GPU_RUN_BOARD.md`).** His s64h model has d = 64 and 332,922 parameters: 1.33 MB in float32, about 0.33 MB at int8 and 0.17 MB at int4, i.e. 5–11× fewer weight bytes than the fork's 2.76M model at int4 (1.6–1.9 MB). The bigger gap is the recurrent state: his deployment budgets are ≤ 450 bits per card and ≤ 400 bits per note (deployed note state 214 bits, with codebooks from a9fcf46), whereas the fork's int8 state is 15,360 B = 122,880 bits per card and 10,240 B = 81,920 bits per note (int4: 73,728 / 49,152 bits), 160–570× more per entity. Weight quantization of the 2.76M model therefore does not close the size gap that matters for a large collection; it would need state compression of two to three orders of magnitude (low-rank/codebook state, Q06). His reported QAT tax at 10 epochs (commit ab82595) is +0.002529 curve (ahead) / +0.004466 instant LogLoss; for comparison, int4 group-32 PTQ of the fork model costs +0.0001 / +0.0005 on the collection and +0.0002 / +0.0011 review-weighted on the held-out screen, and int4 per-channel +0.0001 / +0.0018 and +0.0004 / +0.0033. These are different models, data and protocols, so this only says that 4-bit weights of an over-parameterized 2.76M model are cheap, not that one approach beats the other.

## Backlog

| ID | Idea | Status | Next step |
|---|---|---|---|
| Q01 | int4 group-32/64 PTQ weights in the fork's Rust runtime (with the existing int8 state) | Measured in Python only | Port: pack nibbles + f32 (or f16) group scales, dequantize per matmul tile; re-check the collection replay in Rust |
| Q02 | QAT int4 per-channel (recover the +0.4% collection / +1.5% held-out instant gap) | Implemented, pending GPU | Run the four queue jobs above |
| Q03 | QAT ternary / 3-bit (1.0–1.3 MB, near Expertium float32 size) | Pending; PTQ fails (+12–17%) | `qat.sh 2 32 w2g32-run1`, then `qat.sh 3 0 w3pc-run1` |
| Q04 | QAT / PTQ of Expertium width-80 (being added in `wt/expertium-catch-up`) | Pending | Apply `compress_quant.fake_quant` to its Linear weights; int8 should be free as here |
| Q05 | Distil the 2.76M fork model into Expertium-sized students | Pending | Use the fork model's curve/instant probabilities as soft targets on the run's train users |
| Q09 | Compare against Expertium's s64h deployment model (332,922 params, codebook state ≤ 450/400 bits per card/note) at equal weight and state bytes | Pending | Score s64h on the same collection/held-out protocol; then decide between compressing the 2.76M state and distilling into s64h |
| Q06 | State quantization for deck/preset/user modules and low-rank state (`StateCompression` in the fork's tests) | Pending | Extend `set_state_bits` to all modules; measure memory vs accuracy |
| Q07 | f16 storage of scales and non-Linear parameters | Accounted only (table column) | Confirm no accuracy change (expected) |
| Q08 | Better PTQ: MSE-optimal clipping / GPTQ-style error feedback for 3-bit | Pending | Cheap CPU experiment on the collection before QAT |
