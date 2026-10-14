# Compression handoff (2026-10-14 ~17:00)

Worktree `C:\Users\jonat\workspace\wt\compression`, branch `compression` (not pushed). Docs: [docs/COMPRESSION.md](../../docs/COMPRESSION.md) and [docs/COMPRESSION_INTEGRATION.md](../../docs/COMPRESSION_INTEGRATION.md).

## Done (committed)
- The fork's "int8" is card/note **state** precision, not weights. Weight PTQ on the collection, full history: int8 ±0.02%; int4 g32 +0.02% curve / +0.11% instant; int4 per-channel +0.02/+0.42; 3-bit per-channel and ternary break (+12–17%). The fork's int8 state is reproduced at ±0.00%.
- 105-user CPU held-out screen (weights only): `heldout-cpu-screen.jsonl`.
- Integration package A: `export/RWKV_trained_on_5000_10000_w4-g32.bin`, 1,923,753 B, ARWKVWEIGHTS2; test vectors; the Rust change spec.
- Cohort B: `b-users.json` (4,701 users, stratified round-robin). Labels for the first 2,500 are in `label-db-b/` (ignored, not committed; rebuild with `scripts/compress_labels.py --count 2500`).

## Queued GPU tickets (detached wrappers, shared queue, owner `compression`)
| Ticket | Command | Log | Produces |
|---|---|---|---|
| 000309 | `jobs/heldout-b.sh 1000` (chunk 1, ≤3,200 s scoring) | `logs/heldout-b-run1.log` | rows in `heldout-b.jsonl`: 6 variants × user |
| 000310 | the same, chunk 2 (resumes) | `logs/heldout-b-run2.log` | more rows |
| 000311 | `jobs/qat.sh 4 0 w4pc-run1 2400` | `qat/w4pc-run1-console.log` | `qat/w4pc-run1/{steps.jsonl,summary.json,step-*.pth,final.pth}` |

All three are untested on the GPU.
- B chunk start sequence:
  1. Compile the new CuPy kernel (`scripts/compress_wkv_cuda.py`).
  2. Run its self-test against the float32 reference; the job aborts if the fp32 error is above 1e-4.
  3. Check parity on user 1745 against the CPU scores (aborts above 2e-4).
- If a chunk fails, read the first lines of its log.

## After the jobs
1. Report B: `.venv python scripts/compress_report_b.py`. It prints paired Δ% with 95% CIs per head and metric, splits at 20k reviews, and writes `heldout-b-summary.json`. Then paste the tables into the COMPRESSION.md held-out section and the COMPRESSION_INTEGRATION.md accuracy section, and commit `heldout-b.jsonl` and the summary with `git add -f`.
2. More users:
   - Re-enqueue `bash <abs>/artifacts/compression/jobs/heldout-b.sh 1000` until 1,000 users are scored.
   - Decide from the w4-g32+s8 CI whether 2,500 are needed; if so, use `heldout-b.sh 2500`.
   - Speed: unmeasured. Read `review-evals/s` in the log.
3. QAT result on the collection (CPU, ~5 min per variant):
   - `scripts/compress_score_collection.py --checkpoint artifacts/compression/qat/w4pc-run1/final.pth --tag qat-w4pc-run1 --variants w4-pc,w4-pc+s8,fp32`
   - then `scripts/compress_report.py`.
   - Held-out: needs a variant of `compress_heldout_b.py` that takes the QAT weights. That does not exist yet; add a `--weights` option.
   - Report reviews/s from `summary.json`.
4. Fine-tune-only control (needed to read the QAT result): `bash <abs>/jobs/qat.sh 0 0 fp32-control1 2400`, then score it like step 3 with `--variants fp32`.

## Pitfalls
- Queue commands run from the main checkout's cwd, so always pass the **absolute** path of the job script. Relative paths failed (tickets 290 and 296–303).
- Vendored NVRTC WKV: at most 383 timesteps.
  - Full-history evaluation therefore uses `compress_wkv_cuda`, which is forward-only.
  - QAT trains on 128-review windows (R01 regime), shorter context than upstream's ≤32k segments. That is why the fp32 control matters.
- Read LMDB with `map_size=10_000_000_000` (MapResizedError otherwise).
- State rounding applies only in the patched model: `compress_quant.install_state_quant_model()` must run before any `rwkv` model import.
- Nothing private is committed: `preds/` parquets, `label-db-b`, `_generated/`, the 11 MB dequantized `.bin` and logs stay untracked. Commit only aggregate/per-dataset-user metrics.
- The QAT checkpoint holds latent fp32 weights. Always score it through the int4 fake quantization (`--variants w4-pc…`).

## Fork alignment (fork commit fdf9c689f, docs/rwkv-runtime-precision-2026-10-14.MD)

The fork's recommended default is f32 weights + int8 card/note state. Int4 weights are treated as storage-only there: on an M1 Max with a NEON packed kernel they save about 15 MB RSS of about 506 MB, take 15.8 s to rebuild against 12.4 s, and score the queue about 3.7x slower. Its int4 recipe matches ours (group 32, symmetric -7..7, rank-2 `.weight` only, 1,922,693 B). On one collection it measures +0.14% log loss and UM+ max 0.0096 -> 0.0149.

Follow-ups when processing 000309/000310/000311:
1. Report UM+ plus the max and p99 per-prediction |delta p| for w4-g32 against fp32 on held-out users, to compare with the fork's 13 pp max shift.
2. Frame int4 as a download/storage format that is decoded to f32 at load. The fork's own decoded-at-load row matches the f32 baseline in RAM and CPU.
3. Check whether our export stores f16 or f32 group scales. Align with the fork's `ARWKVINT4POC1` (`packed_weights.rs`, f32 scales) before proposing a format.

## Codex cross-check (2026-10-14)
- Our ARWKVWEIGHTS2 and the fork's ARWKVINT4POC1 decode bit-identically for all 504 tensors, and both use f32 scales. The 1,060 B size difference is 265 four-byte group-size fields. Follow-up 3 above is resolved.
- Fixed a fixture bug. Fake quantization keeps -0.0, while a packed zero decodes to +0.0, and `torch.equal` misses that difference. `dequantized_sha256` and the `_dequantized_f32.bin` companion now come from the decoded export, and the regenerated test vectors changed 260 hashes. The int4 .bin is unchanged (sha e4faab45…).
- The held-out 105-user screen used f32 state and short-history users. The matched w4+s8 quality numbers (000309/000310) are still owed, with QAT 000311.
