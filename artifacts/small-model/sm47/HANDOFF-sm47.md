# SM47 handoff: float16 row scales (h) and int8 shift vectors (x8)

Branch `sm47` in worktree `C:/Users/jonat/workspace/wt/sm45`, code commit 4c2750ee. **Do not edit source in this
worktree until the job has finished**: the queued job runs from this checkout.

## What is running

- Launcher (detached, started 2026-10-20 19:19): `artifacts/small-model/sm47/launch-sm47.sh`, log
  `artifacts/small-model/sm47/launch-sm47.log`. It waits for the SM46 runbook to log `[sm46 <date>] done` or `STOP`
  (`C:/Users/jonat/workspace/wt/sm46/artifacts/small-model/sm46/runbook-sm46.log`; SM46 itself is still waiting for SM44),
  then waits while `C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/HOLD-chain` exists, then enqueues one
  ticket (owner small-model, claim gpu, `--expected-minutes 115`, launched by "detached launcher sm47"). The queue will
  not start the ticket if it would cross the GPU break (2026-10-20 22:00 to 2026-10-21 01:00).
- Job: `artifacts/small-model/sm47/job-sm47.sh`. The job's own `[sm47 ...]` lines and the queue's lines land in the
  launcher log.
  1. GPU check, 3 smallest 732-pool users (1425, 1893, 2388), 8 variants (w8+s8, w8+s4c8n and the 6 new ones), against
     `check-cpu.jsonl` (|dLogLoss| <= 1e-5 per head), plus w8+s8 / w8+s4c8n against SM45's GPU rows
     (`../sm45/check-gpu.jsonl`; this shows that the kernel without the new flag is unchanged). The result goes to `check-gpu.json`.
     The job logs `STOP: ...` on a mismatch.
  2. Scoring pass 1 (`--time-limit 7200`) and pass 2 (`--time-limit 1500`, resumes and retries OOM rows) over the 732
     pool for the 6 new variants, giving `heldout-sm47-pools.jsonl`. The memory cap is the default 0.8. With that cap SM45 had the same 9
     large users OOM (70, 650, 1479, 1534, 1601, 4379, 4501, 4625, 4922), so expect about 723 scored users.
  3. `sm47_summary.py` writes `sm47-summary.md` / `sm47-summary.json`, then the job logs `[sm47 ...] done`.
- Time estimate: SM45 took about 33 min for pass 1 with 2 variants. For 6 variants that gives about 100 min, plus the
  per-step shift hook on the x8 variants, the GPU check (~1-2 min) and pass 2 (~5 min of OOM retries). The ticket
  estimate is 115 min.

## Follow-up (fresh agent, when the launcher log shows `[sm47 ...] done` or `STOP`)

```bash
cd C:/Users/jonat/workspace/wt/sm45
grep -aE '^\[sm47|\[queue\]' artifacts/small-model/sm47/launch-sm47.log | tail -20
cat artifacts/small-model/sm47/check-gpu.json artifacts/small-model/sm47/sm47-summary.md
# re-run the summary by hand if needed (CPU, ~1 min):
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm47/sm47_summary.py
git add -f artifacts/small-model/sm47/{check-gpu.jsonl,check-gpu.json,heldout-sm47-pools.jsonl,sm47-summary.md,sm47-summary.json,launch-sm47.log}
git -c user.name="Jonathan Schoreels" -c user.email=jonathan.schoreels@gmail.com commit   # message ends with the Co-Authored-By line
git push
```
On `STOP: GPU check mismatch`, compare `check-gpu.json` per_variant. If only new variants mismatch, the kernel's
f16_round_scale or the CUDA `.half()` shift rounding differs from CPU. If w8+s8 / w8+s4c8n differ from SM45's GPU rows,
the kernel edit changed the default path.

## Variants and bits (scripts/state_vq_report.int_entity_bits)

Per WKV row: 16 codes plus one scale (f32, or f16 with h). Per token-shift vector: 64 x f32, or with x8 64 int8 codes
plus one per-vector scale (f16 with h, else f32). Card = 2 layers x 4 heads x 16 rows + 3 shift vectors. Note = 1 x 4 x 16 rows
+ 2 shift vectors. Collection = 13,973 cards + 13,198 notes, MB = 10^6 bytes.

| Variant | Card bits | Note bits | Collection MB |
|---|---:|---:|---:|
| w8+s8 (ref) | 26,624 | 14,336 | 70.15 |
| w8+s4c8n (ref, SM45) | 18,432 | 14,336 | 55.84 |
| w8+s8h | 24,576 | 13,312 | 64.89 |
| w8+s8x8 | 22,112 | 11,328 | 57.31 |
| w8+s8hx8 | 20,016 | 10,272 | 51.91 |
| w8+s4c8nh | 16,384 | 13,312 | 50.58 |
| w8+s4c8nx8 | 13,920 | 11,328 | 43.00 |
| w8+s4c8nhx8 | 11,824 | 10,272 | 37.60 |

## Implementation (commit 4c2750ee)

- `scripts/small_model_wkv.py`: `F16_SCALE = 256` flag on the kernel's `bits` argument. The kernel source (patched
  textually from compress_wkv_cuda.SOURCE, which is untouched) gains `f16_round_scale`: RNE to float16 with subnormals
  kept and a clamp at 65504. A NaN or 0 result becomes 1, as on the f32 path. Bits 0/4/8 run the old arithmetic. The torch twin
  is `round_scale_f16` (clamp then `.half().float()`). On 5M random and tie values the emulated device algorithm matches
  torch and numpy float16 bit for bit. The source compiles with NVRTC for K = 16 and 32 (arch 89; no GPU used).
  `self_test` adds `bits8h` / `bits4h` entries, which are reported but not enforced.
- `scripts/state_vq.py`: `int_row_quant(bits | F16_SCALE)`. `configure(..., shift_bits)` sets the CPU Hooks.shift on the card/note
  stored shift vectors (the `+vq` hook point). `IntScoreCodec` / `int_score_codecs` handle GPU scoring through the state-VQ slots
  (`set_scoring`): the WKV goes through `wkv_long` (the int kernel, F16 flag) and the shift through `int_row_quant`.
- `scripts/small_model_heldout.py`: variant parser `(s8|s4c8n)(h?)(x8)?` (`IntState`). New rows add
  `state_scale_bits`, `state_shift_bits` and `state_entity_bits`, while old variants' rows keep their fields unchanged. The new flag
  `--emulate-gpu-path` (check only) runs the CUDA dispatch on CPU tensors.
- `scripts/state_vq_report.py`: `int_entity_bits`. `bits_of` reads `state_entity_bits` and now also counts int4 / s4c8n.

## CPU check (check-sm47-cpu.json: passed)

- (i) w8+s8 and w8+s4c8n reproduce SM45's committed CPU rows exactly (every logged metric).
- (ii) All 6 new variants differ from their base. Per-user benchmark |dLogLoss| is at most 8e-6 (s8h), 1.9e-5 (s8x8),
  1.4e-5 (s8hx8), 1.2e-5 (s4c8nh), 5.8e-5 (s4c8nx8) and 5.5e-5 (s4c8nhx8). These are 3 small users and are not results.
- (iii) The GPU dispatch run on CPU tensors (`--emulate-gpu-path`) equals the CPU scorer exactly (0.0) for all 6.
- Float16 subnormal scales (`f16-scale-stats.json`, `f16_scale_stats.py`): on the 3 users, 49% of the s8hx8 scale roundings
  and 14% of the s4c8nhx8 ones are float16 subnormals (scale < 2^-14, i.e. int8 rows with absmax < 7.8e-3). They are kept as
  IEEE subnormals: codes use the rounded scale, and the worst added error is about 127 x 2^-25 = 3.8e-6 absolute from clipping
  the max element. About 1.6e-4 of the roundings underflow to 0, and those rows/vectors are stored as zeros (|values| < 3.8e-6).
  A runtime could avoid subnormals by storing f16(absmax) instead of f16(absmax/qmax), which is the same 16 bits. That option
  was not tested.
