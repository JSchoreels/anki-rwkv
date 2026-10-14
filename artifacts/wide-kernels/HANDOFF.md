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

## Update after 000323 (2026-10-14, ran 20:06:46-20:24:55 local, exit 0)

Results are in `artifacts/wide-kernels/speed-20261014/` (clean cases: `w80-sequential-b256-wide`, `w64-wide`) and `run1-contaminated/` (the five cases to rerun; moved there so a rerun fills the original paths).

Why `summary.md` showed "failed" and dashes: (1) the **w32 control failed in 6 s**: its accepted options use the native extension, whose build lives in the main checkout's `artifacts/cache/r01-native/<key>` (key f1b68c9d... matches), and the worktree has no such folder; `native_launch.load()` looks under `GPU_R01_CONTROLLER_ROOT` or the worktree root, so it died on a missing `build.json`. The other cases drop the native switches, so they ran. (2) `report()` printed dashes for every row when no w32 baseline existed, although the data was in each `result.json`. Fixed: it now falls back to the accepted w32 figure (178,242 reviews/s, docs/GPU_OPTIMIZATIONS.md) with a note, and has `--parallel-until <UTC ISO>` which adds an Overlap column. (3) `speed.sh` had no coreutils on the queue's PATH (`date: command not found`, so "after 0 min" and the 26-min budget never applied; harmless this time). `speed.sh` now exports PATH and `GPU_R01_CONTROLLER_ROOT`.

Contamination: an accuracy collection scoring ran until 20:15:10 local (18:15:10 UTC). Share of training time overlapping it: w80-b256-wide 100%, w64-b256-wide 100%, w80-b256 100%, w64-b256 21%, w80-sequential-b256-wide 0%, w64-wide 0%.

| Case | Reviews/s | vs accepted w32 (178,242) | Training s | Peak alloc MiB | Main / machine CPU s | Overlap |
|---|---|---|---|---|---|---|
| w32 control | failed | | | | | |
| w80 ported (256) | 111,281 | -37.6% | 111.5 | 8,417 | 145 / 827 | 100% |
| w80 generic (256) | 62,216 | -65.1% | 199.4 | 8,530 | 204 / 1,048 | 100% |
| w64 ported (256) | 141,274 | -20.7% | 87.8 | 6,687 | 117 / 756 | 100% |
| w64 generic (256) | 74,584 | -58.2% | 166.3 | 6,778 | 133 / 802 | 21% |
| w80-sequential ported (256) | 119,808 | -32.8% | 103.6 | 8,399 | 128 / 583 | 0% |
| w64 ported (512 chunks) | 39,041 | -78.1% | 317.8 | 13,042 | 151 / 1,357 | 0% |

Ported vs generic: w80 1.79x, w64 1.89x (both sides overlapped; the ported side more so, so the ratio is probably conservative). Machine CPU for the overlapped rows includes the foreign scoring, so the CPU columns are not comparable. w64 at 512 chunks does only 287 updates' worth in 318 s and nearly fills 16 GB (still paging; earlier generic 18,711 reviews/s, 2.09x). Peak MiB is peak allocated.

**A clean rerun is needed** (control failed; four rows contaminated). It restarts only those five cases (clean ones are skipped), about 12 min of cases, so 20 min expected. Queue it after 01:00 (`--timing`; launch detached, `Start-Process` with one quoted argument string):

```
cd C:\Users\jonat\workspace\anki-rwkv
C:\Users\jonat\workspace\anki-rwkv\.venv\Scripts\python.exe scripts/r01_queue.py run --owner wide-kernels --purpose "wide-kernels: clean rerun of w32 control and four contaminated cases" --claim gpu --timing --expected-minutes 20 -- "C:\Program Files\Git\usr\bin\bash.exe" /c/Users/jonat/workspace/wt/wide-kernels/artifacts/wide-kernels/jobs/speed.sh
```

Afterwards (fresh Sonnet medium agent): run `C:\Users\jonat\workspace\anki-rwkv\.venv\Scripts\python.exe scripts/expertium_speed.py report --out artifacts/wide-kernels/speed-20261014` in the worktree (no `--parallel-until` needed; check no foreign CPU load in `all.log` and the machine), replace the W04 numbers in `docs/GPU_OPTIMIZATIONS.md`, commit as Jonathan Schoreels, push `wide-kernels`.

## Update after 000327 (2026-10-15; ticket 000327-wide-kernels, ran 00:10-00:36 local, 22:10:04-22:36:16 UTC, exit 0)

Ticket **000327-wide-kernels** (`--timing`, `--expected-minutes 20`, ran 26.2 min: the spilled w80 cases took 12 + 10 min; if the rerun below does not spill, 20 minutes is enough, about 12 min of cases plus startup). Outputs: `artifacts/wide-kernels/speed-20261014/` (`summary.md`, `summary.json`, `all.log`, `queue-console-rerun.log`, `<case>/expertium/E89-F65e-F77a/{result.json,gpu_power.jsonl}`); first-run contaminated cases are in `run1-contaminated/`. Ledger: W01-W04 in `docs/GPU_OPTIMIZATIONS.md` updated.

**Cause of the odd numbers: GPU memory spill, not an error.** The card has 16.3 GB. w32, w80-b256-wide and w80-b256 each reserved 15.3-15.5 GB (peak allocated only 8.4-9.5 GB; the caching allocator holds the rest). Other desktop processes held 1.7-2.2 GB of VRAM at start (1.1 GB in the first run; Chrome, Spotify, Steam, iCUE, Codex python appear in `nvidia-smi`), so the total exceeded 16.3 GB and Windows paged into shared memory. Signature in `gpu_power.jsonl`: w80 runs at 95-99% "utilization" but only 90-105 W (non-spilling cases draw 170-210 W), device memory above 15.5 GB in 91-94% of samples (first run 54-66%), and the rate falls from ~110-160k to 5-20k reviews/s after the first 2.3M reviews, recovering only at the end. Same position in both w80 cases. Not the cause: no script error (all cases exit 0; PATH/controller-root fix worked, w32 ran with the native build and the budget guard worked), clocks and temperature normal (SM 3.14-3.18 GHz, 42-56 C), no other queue job (history: 000329 ended 22:10:04 UTC just as 000327 started, gpu held alone until 22:36:16), no small-model activity (its worktree was created at 00:36:02 local, after the cases ran). `w64-b256` generic did not start: the 26-min budget stopped it after the slow w80 cases. The table's `Overlap` column is empty (no `--parallel-until` given); the first-run overlap shares are in the "after 000323" section.

Valid numbers (accepted w32 = 178,242 reviews/s, 287 updates; 160 users):

| Case | Parameters | Reviews/s | vs accepted w32 | Training s | Peak alloc MiB | Verdict |
|---|---|---|---|---|---|---|
| w64 ported (256) | 331,511 | 165,018 | -7.4% | 75.2 | 6,687 | valid (reserved 11.9 GB, no spill) |
| w80-sequential ported (256) | 558,209 | 119,808 | -32.8% | 103.6 | 8,399 | valid first run, borderline (66% of samples above 15.5 GB) |
| w64 ported (512 chunks) | 331,511 | 39,041 | -78.1% | 317.8 | 13,042 | valid but paging by design |
| w32 control | n/a | 127,086 | | 97.6 | 9,491 | **invalid** (partial spill, 16% of samples above 15.5 GB); accepted 178,242 stands |
| w80 ported (256) | 558,209 | 17,247 | | 719.3 | 8,417 | **invalid** (spill; first run 111,281 also partly contaminated) |
| w80 generic (256) | 558,209 | 20,810 | | 596.2 | 8,530 | **invalid** (spill; first run 62,216 contaminated) |
| w64 generic (256) | 331,511 | not run | | | | first run 74,584 had 21% overlap |

Parameter counts for the small-model comparison: **w64 331,511; w80 558,209** (w32 baseline model 2.76M-class fork RWKV is a different family; Expertium's s64h is 332,922).

8k/2k 10-epoch ETA (581,942,738 train reviews per epoch, the figure in `docs/ACCURACY_LOG.md`): ported w64 at 165,018 reviews/s is **58.8 min per epoch, 9.8 h for 10 epochs** (plus about 10 min fast final evaluation). w80 is not established; indicative 111-120k reviews/s gives 81-87 min per epoch, 13.5-14.5 h, but those rows are not clean.

**A rerun is needed for the w80 pair, w64 generic and the w32 control** (w64 ported, w80-sequential and w64-wide stay). Precondition: the GPU must be empty apart from the desktop, with device memory in use <= 1,300 MiB before it starts (`nvidia-smi --query-gpu=memory.used --format=csv`); close Chrome, Spotify, Steam, iCUE, Discord and any Codex python (a Codex python.exe held ~12 GB when checked at 00:37). If the card still has >1.3 GB in use, expect the same spill and do not queue. Do not queue it from this agent. Command (move the invalid cases first, then launch detached, `Start-Process` with one quoted argument string, `--timing`, after the quiet window):

```
cd C:\Users\jonat\workspace\wt\wide-kernels\artifacts\wide-kernels\speed-20261014
mkdir run2-spilled; mv w32 w80-b256-wide w80-b256 run2-spilled/
$env:CASES = "w32 w80-b256-wide w80-b256 w64-b256"
cd C:\Users\jonat\workspace\anki-rwkv
C:\Users\jonat\workspace\anki-rwkv\.venv\Scripts\python.exe scripts/r01_queue.py run --owner wide-kernels --purpose "wide-kernels: w32 control, w80 pair and w64 generic without GPU memory spill" --claim gpu --timing --expected-minutes 20 -- "C:\Program Files\Git\usr\bin\bash.exe" /c/Users/jonat/workspace/wt/wide-kernels/artifacts/wide-kernels/jobs/speed.sh
```

Afterwards (fresh Sonnet medium agent): check each `result.json`/`gpu_power.jsonl` for spill (power >= 150 W, device memory not pinned above 15.5 GB), run `scripts/expertium_speed.py report --out artifacts/wide-kernels/speed-20261014`, update W04 and the ETA, commit as Jonathan Schoreels, push `wide-kernels`.
