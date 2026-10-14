# Integrating int4 group-32 weights into the Anki fork

Package for "int4 group-32 weights + int8 state" (`fork/w4-g32+s8` in [COMPRESSION.md](COMPRESSION.md)). Nothing in the fork (`C:\Users\jonat\workspace\anki`) was edited; this describes the changes.

## What ships

| File | Bytes | Notes |
|---|---:|---|
| Current `qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.bin` (`ARWKVWEIGHTS1`, float32) | 11,082,637 | 504 tensors, 2,762,884 parameters |
| [`artifacts/compression/export/RWKV_trained_on_5000_10000_w4-g32.bin`](../artifacts/compression/export/RWKV_trained_on_5000_10000_w4-g32.bin) (`ARWKVWEIGHTS2`) | **1,923,753** | SHA-256 `e4faab45adf92c7007901b52713870e2a2bb205febe91f89443893d98d058f37`; 5.76× smaller |
| [`artifacts/compression/export/test-vectors-w4-g32.json`](../artifacts/compression/export/test-vectors-w4-g32.json) | — | Per-tensor SHA-256 of the dequantized float32 values, worked examples, Python predictions on a public input |

Regenerate with `scripts/compress_export.py --scheme w4-g32 [--dequantized-v1] [--predictions]` (CPU, seconds; `--predictions` about a minute). The script refuses to finish unless reading the file back gives tensors bit-identical to the model that was evaluated, and unless its reader also parses the fork's current `.bin` identically to the `.pth`.

The int8 state needs no change: it is the fork's existing `rwkvStatePrecision` option (default `int8`, `rslib/src/rwkv/precision.rs`), applied to card and note state independently of how the weights were stored.

## File format `ARWKVWEIGHTS2`

Little-endian; identical to the fork's `ARWKVWEIGHTS1` (`WeightMap::load`, `rslib/src/rwkv/mod.rs`) except for the magic and one encoding byte per tensor.

```
magic   "ARWKVWEIGHTS2"                       13 bytes
u32     tensor count                          504
per tensor (PyTorch state_dict order):
  u16   name length, then UTF-8 name
  u8    rank, then rank x u32 dims            PyTorch shape; a Linear weight is [out, in]
  u8    encoding                              0 = f32, 4 = int4 grouped, 8 = int8 grouped
  encoding 0:   prod(dims) x f32, row-major   (exactly the ARWKVWEIGHTS1 payload)
  encoding 4/8: rank must be 2, dims = [rows, cols]
    u32 G                                     group size along the last (input) axis, 1 <= G <= cols
    rows x ceil(cols/G) x f32 scales          row-major; scale[r][j] covers columns j*G .. min(cols,(j+1)*G)-1
    codes, row-major over rows*cols elements
      int8: one i8 per element, -127..127
      int4: ceil(rows*cols/2) bytes; element 2k in the LOW nibble, 2k+1 in the HIGH nibble of byte k,
            signed two's complement, -7..7 (-8 is invalid). Same packing as the fork's int4 state rows.
end of file (no trailing bytes)
```

Dequantization: `value = (code as f32) * scale`, a single IEEE float32 multiply. This reproduces the evaluated model's weights bit for bit.

Encoder (for reference; `scripts/compress_quant.fake_quant`): per group, `scale = max|w| / 7` in float32 (a zero group gets 1.0), `code = round_half_to_even(w / scale)` clamped to ±7. G = 32 except where a row has fewer than 32 inputs: the 64 LoRA up-projections (`*.B_and_lamb.weight`, `*.lora_B_g.weight`, shapes 128×8 and 128×16) are stored with G = cols, which is the same single group the encoder used; the 92-input `features2card.0.weight` has groups 32, 32, 28.

Which tensors: every 2-D `.weight` (all are `nn.Linear`; 265 tensors, 2,719,744 values) is encoding 4. Everything else stays encoding 0 (float32): biases, LayerNorm/GroupNorm weights and biases, `rkvdag_lerp`, `bonus`, `lerp_k` (239 tensors, 43,140 values). Byte budget: 1,359,872 B codes + 358,656 B scales + 172,560 B float32 tensors + 32,665 B headers and encoding bytes = 1,923,753 B. Storing the scales as f16 would save 179,328 B more, but that is a different model (scales rounded) and was not evaluated.

## Rust changes (described; in the fork)

1. **`rslib/src/rwkv/mod.rs`, `WeightMap::load` (line ~3166).** Read the 13-byte magic and branch: `ARWKVWEIGHTS1` keeps the current loop; `ARWKVWEIGHTS2` reads one `u8` encoding after the dims. For 0, read f32 values as now. For 4/8, require rank 2, read `u32 G` (reject 0 or > cols), read `rows * ceil(cols/G)` f32 scales (reject non-finite or <= 0, matching `precision.rs` validation), read the codes (`cursor.bytes(n)`), validate the code range, and fill `values[r*cols + c] = code(r, c) as f32 * scales[r*ngroups + c/G]`. The resulting `Tensor { shape, values }` is the same type as today, so `linear`, `layer_norm`, `group_norm`, the transposition to `weight_by_input` and all fused/precomputed matrices further down are unchanged. `Cursor::expect_magic` consumes bytes, so read the magic with `cursor.bytes(13)` and compare against both constants. Keep `cursor.expect_end()`.
   - int4 code of element `i`: `let b = bytes[i / 2]; let n = if i % 2 == 0 { b & 0x0F } else { b >> 4 }; let code = ((n as i8) << 4) >> 4;` (sign-extend the nibble), reject `code == -8`.
2. **`rslib/src/bin/rwkv_bench.rs`** has its own copy of `WeightMap::load` (line ~245); apply the same change or share the loader.
3. **Weight file name.** Either replace `RWKV_trained_on_5000_10000.bin` in place, or add the new file and update the name in `build/configure/src/aqt.rs` (`qt:aqt:rwkv_weights` copy rule), `qt/aqt/rwkv_scheduler.py` (`_EMBEDDED_RWKV_MODEL_FILENAME`), the Rust test defaults that hard-code the path (`rslib/src/rwkv/mod.rs` ~9150, ~9308, ~9436, ~9662) and the Python tests that name the file (`qt/tests/test_rwkv_inference_process.py`, `qt/tests/test_rwkv_scheduler.py`). `ANKI_RWKV_MODEL_PATH` can point at the new file for trials without any rename.
4. **State cache.** No code change. The cache key (`_rwkv_model_cache_key` in `rwkv_scheduler.py`) contains the model file's size and SHA-256, so the first start with the new weights rebuilds the card/note/deck/preset/global state from retained history once (about 10–11 s on the 2026-10-10 benchmark collection with int8 state, per the fork's precision benchmark), as for any weight change.
5. **Memory.** This keeps dequantize-on-load: resident weights stay float32 (≈ 11 MB plus the existing fused copies). The saving is in the installer/app bundle and on disk (11.08 → 1.92 MB). Keeping int4 resident and dequantizing per matmul tile would also cut ≈ 9 MB of RAM but changes every matmul path; it is not worth it next to the state store (395 MB int8 on the benchmark collection).

## Verification for the Rust side

- **Weights, bit-exact.** After loading the new file, the SHA-256 of each tensor's little-endian float32 values must equal `dequantized_sha256[name]` in the test vectors (504 entries; unquantized tensors hash the original float32 values). `examples` lists, for three tensors, row 0's scales (decimal and hex f32), its first 40 codes, the first eight dequantized values as hex f32, and the last group's codes of the last row (exercises the 28-wide partial group of `features2card.0.weight`).
- **Model, bit-exact.** `compress_export.py --dequantized-v1` also writes the same dequantized weights as an `ARWKVWEIGHTS1` float32 file (11,082,637 B, not committed). Because only the loader changes, the Rust runtime must give bitwise identical predictions and states with either file, on any collection (e.g. the existing `collection_benchmark.rs` replay).
- **Python reference predictions.** `predictions` in the test vectors: curve and instant recall probabilities of the fp32 and int4 models (Python reference recurrence, float32) for the 152 labelled reviews of the first 256 rows of public dataset user 4 (anki-revlogs-10k), one history, encoding seed 2025 + user. They are for checking a Python or independent port; the Rust feature pipeline differs from the Python one (correlation 0.998 on the collection in the 2026-10-14 evaluation), so do not expect Rust to match these to the bit. Single predictions early in a history move by up to 0.106 (curve) / 0.063 (instant) between fp32 and int4, while aggregate log loss moves by about 0.1%; the fork documents the same pattern for int4 state (largest single change 14.1 points at +0.07% log loss).
- **Accuracy.** Rerun the fork's precision/collection benchmark with the new weights and int8 state; its log-loss change against today's int8 should be close to the Python measurement below (well within ±0.2%).

## Measured accuracy and size (Python reference)

Collection (the user's own, full-history replay, 173,690 FSRS-6-filtered test reviews; fork float32 = curve 0.4417, instant 0.4366):

| Variant | Weight bytes (file) | Card / note state bytes | Curve LogLoss | Δ% vs fp32 | Instant LogLoss | Δ% vs fp32 |
|---|---:|---:|---:|---:|---:|---:|
| fork fp32 weights, f32 state | 11,082,637 | 52,224 / 34,816 | 0.4417 | — | 0.4366 | — |
| fork fp32 weights, int8 state (today's default) | 11,082,637 | 15,360 / 10,240 | 0.4417 | +0.00% | 0.4366 | +0.00% |
| **int4 group-32 weights, int8 state** | **1,923,753** | 15,360 / 10,240 | 0.4417 | +0.02% | 0.4371 | +0.11% |

Held-out dataset users: a 105-user CPU screen (small histories; weights only, f32 state) gives int4 group-32 +0.08% curve / +0.42% instant (review-weighted). The stratified ~1,000–2,500-user GPU evaluation of the exact shipped combination (`w4-g32+s8`, with `fp32`, `fp32+s8`, `w8-pc+s8`, `w4-g64+s8`, `w4-g32+s4`) is queued; results will be added to [COMPRESSION.md](COMPRESSION.md).
