# Wide kernels: handoff (2026-10-14)

Worktree `C:\Users\jonat\workspace\wt\wide-kernels`, branch `wide-kernels` (pushed). Code commit `e36982e`. Ledger rows W01–W05 in `docs/GPU_OPTIMIZATIONS.md`.

## What changed

The exact width-32 kernels now also run on the Expertium head-size-16 models (w64 = s64h, d=64; w80 = iter-53, d=80), each bitwise equal to the generic path:

- `exact_index` and `skip_padding_sources`: `rows_backward_wide` reproduces PyTorch 2.9.1's general `indexing_backward_kernel` for rows wider than 32 (one rounding to the destination dtype per duplicate). The interleaved variant's per-step `index_select`/`index_copy` use `OrderedIndexSelect` (one sort per stream) and `ExactIndexCopy`.
- `exact_group_norm` for C = 64/80 (separate kernels; C = 32 untouched).
- `row_tiles_exact` at K=16 (text rewrite of the F31 kernel; K=32 source unchanged).
- `scripts/expertium_options.compatible_options(options, wide_kernels=True)` keeps these switches. The default call still gives the old generic cases. Still dropped: `native_wkv`, `wkv_tile`, `native_index`, `native_group_norm`, `native_glue` (W05).

## Parity (done)

`scripts/check_wide_kernels.py`: 312/312 bitwise (`check-parity.json`, `check-parity.log`). The checks cover row select (C 16–160, duplicates, padding), index copy, GroupNorm, WKV K=16 (rows 1/2/4/8, tiles 16/32, T up to 383), and the full w80/w64/w80-sequential/w64-sequential models on 3 real multi-user batches (loss, predictions, every gradient, perturbed weights). The checks disable JIT graph optimization, as the trainer does. With it on, the scripted GRU head's first call differs from later calls on the generic path itself. CPU catch-up checks still pass (`check-cpu.json`).

## Pending GPU job

- Ticket **000323-wide-kernels** (`--timing`, expected 45 min) runs `artifacts/wide-kernels/jobs/speed.sh`.
  - Log: `artifacts/wide-kernels/speed-20261014/queue-console.log`, `all.log`.
  - Results: `artifacts/wide-kernels/speed-20261014/<case>/expertium/E89-F65e-F77a/result.json`, `summary.md`.
- Seven tickets were ahead at 18:45 (about 3.5 h, two of them 57-min compression chunks), so the job cannot finish before the 21:00–01:00 quiet window. It runs **overnight**: expected start about 01:00, finish about 01:45.
- Cases, in order:
  1. w32 (accepted control)
  2. w80-b256-wide
  3. w64-b256-wide
  4. w80-b256 (generic control)
  5. w64-b256 (generic)
  6. w80-sequential-b256-wide
  7. w64-wide (512 chunks)
- New cases stop starting after 26 min. Same 160 users and seed as `artifacts/expertium-catch-up/speed-20261014`.

## Follow-up (fresh Sonnet agent, medium effort)

```
cd C:\Users\jonat\workspace\anki-rwkv
.venv/Scripts/python.exe scripts/r01_queue.py status            # 000323 finished?
cd C:\Users\jonat\workspace\wt\wide-kernels
cat artifacts/wide-kernels/speed-20261014/all.log               # also check the nvidia-smi lines: no foreign GPU jobs
/c/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe scripts/expertium_speed.py report --out artifacts/wide-kernels/speed-20261014
```

Then fill the W01–W04 "Measured" cells with reviews/s, main/machine CPU s and peak MiB per case, plus the 8k/2k 10-epoch ETA. Commit as Jonathan Schoreels and push `wide-kernels`. A case that did not start (budget) or failed can be rerun with `CASES="<names>"` through the same queue command (finished cases are skipped):

```
cd C:\Users\jonat\workspace\anki-rwkv
.venv/Scripts/python.exe scripts/r01_queue.py run --owner wide-kernels --purpose "wide-kernels: rerun cases" --claim gpu --timing --expected-minutes 20 -- "C:\Program Files\Git\usr\bin\bash.exe" /c/Users/jonat/workspace/wt/wide-kernels/artifacts/wide-kernels/jobs/speed.sh
```

Launch it detached (`Start-Process` with one quoted argument string; PowerShell 5.1 does not quote array arguments).
