# SM44: compressed card/note state on the GPU scorer (handoff, 10-20)

Goal: score the state-compression variants (`+vq<card bits>x<note bits>`) on the 732-user standard pool in minutes
instead of hours, and try the bigger ~800-bit budget. The 464 pool is a subset of the 732 pool, so its readout needs
no extra GPU pass.

Worktree `C:/Users/jonat/workspace/wt/sm32c`, branch `sm44`. It is the SM32c tree with `small-model` merged in, for the
732/2225 pools, the GPU memory cap and the OOM retry.

## What was built
- `scripts/state_vq.py`, section "GPU scoring": `install_scoring`, `score_codecs`, `score_wkv`, `score_shift`,
  `set_scoring`.
  - The eager model's own `state_vq_slot` sends the card and note mixers to forward-only scoring functions.
  - Placement is the CPU scorer's. The answer uses the float update. The stored state is the codec's reconstruction
    before that entity's next review. A token shift that reads the previous review gets the coded vector.
  - Deck, preset and user state stay float on the long WKV kernel.
- WKV codec on CUDA:
  - Rank-1 streams use the **unchanged** fused kernel `scripts/kernels/state_vq_wkv.cu` (`KernelCodec`). That covers
    every card recipe and the 412-bit note. The kernel factorizes by Gram squaring (the `...p` codec), in **float64**
    by default (`--vq-factor-dtype`), matching the CPU scorer.
  - The kernel records the state before every step. The scorer does not need that record, so each split runs in row
    blocks of at most 131,072 rows x steps (512 MB). The rows are independent entities, so this changes nothing.
  - Rank-2 streams have no fused kernel. This only affects the 828-bit note of `vq770x828`, which runs a per-step torch
    loop on the device with the recipe's own eigh float64 codec.
- `scripts/small_model_heldout.py`: `+vq` variants on `--device cuda`.
  - Same `--vq-codebooks` option as on CPU; the codebooks' sha256 is written into every row.
  - New option `--vq-factor-dtype`.
  - New row field `state_vq_path`, which records which computation each stream used.
- `scripts/check_state_vq_score_gpu.py`: `cpu` mode (CPU-first check) and `gpu` mode (the queue's check ticket).
- `artifacts/small-model/sm44/sm44_summary.py`: CPU pairing summary.

## What was verified (CPU, no GPU)
`sm44/check-sm44-cpu.json` (log `sm44/logs/check-sm44-cpu.log`): **ALL OK**.
- The GPU path's code ran on CPU tensors: the slots, plus the "sq" codec in torch, i.e. the kernel's arithmetic. It
  **reproduces the CPU scorer (eigh codec) bitwise**: max |dp| = 0 and dLogLoss = 0 on both heads.
  - SM42 `+vq378x412`: the 5 smallest pool users (2388, 4676, 1712, 1745, 2253; 3,685-4,311 reviews).
  - SM42 `+vq770x828`: 3 users, including the rank-2 note torch loop.
  - SM32c `+vq378x412` with its learned books: 2 users.
- So the Gram-squaring codec picks the same codes as the deployed eigh codec here, as in SM32b.
- The codec was active: float state moves p by up to 0.19 on the same users.
- The slot path ran: 9-13 WKV and 15-22 shift calls per user.
- Stored CPU references for the GPU ticket: `sm44/check-ref/*.npz`. This includes float-state references, so the GPU
  check can also report the base GPU vs CPU numerics.

- Not run: the MSVC emulation of the .cu kernel inside the scorer path. It spins 256 threads on barriers and loaded
  the CPU to 94% next to the SM43 training, so I stopped it after ~5 min.
  - The kernel's agreement with the torch codec was already checked: bitwise stored states in SM32b, 0 pick mismatches
    on GPU in SM32c.
  - The GPU check ticket compares the real CUDA kernel with the CPU scorer.

## Running (detached)
- Runbook: `C:/Users/jonat/workspace/wt/sm32c/artifacts/small-model/sm44/runbook-sm44.sh`
- Log: `C:/Users/jonat/workspace/wt/sm32c/artifacts/small-model/sm44/runbook-sm44.log`
- Dry run: `sm44/runbook-sm44-dryrun.log`
- Launched 10-20 15:45:49 from commit `8c5770d5`. The runbook bash is MSYS pid 1708 (Windows pid 16516).
- At launch it was waiting for the SM43 full launcher to end. There was no SM44 ticket in the queue yet; the queue was
  running only `000555-small-model` (SM43 A).
- Order of steps:
  1. Wait for the SM43 full launcher to end (log line `done`, `A failed` or `B failed`). SM43 holds several tickets in a
     row (A, then B, then its scoring), and SM44 must not slot in between them.
  2. Wait while HOLD-chain exists. This check runs before every ticket.
  3. **GPU check ticket** (10 min): CUDA path vs the stored CPU-scorer predictions on the 5 smallest pool users. Gate:
     benchmark LogLoss within 1e-5 per user and head. It also reports the float-state GPU vs CPU difference (base
     numerics) and measures reviews/s on 2 near-median users. **The runbook stops on a mismatch.**
  4. Scoring tickets on the 732 pool, one label at a time:
     - SM42 `w8+vq378x412` (k-means books)
     - SM32c `w8+vq378x412` (its learned books)
     - SM32c `w8+s8`
     - SM42 `w8+vq770x828` (k-means books)
     - SM42 `w8+s8`, re-scored as a reproducibility check against the wt/sm42 rows
     
     Each ticket runs at most 90 min, with up to 4 passes per label. GPU memory is capped at 0.8. Rows go to
     `sm44/heldout-sm44.jsonl`.
  5. CPU summary: `sm44/sm44-summary.md` and `.json`, for pools 732 and 464.
- Expected start: after SM43 full ends, about 20:50 (10-20).
- Expected duration (estimate, **to be measured** by the GPU check's timing): check 10 min; each `w8+s8` ~19 min (the
  measured w64 rate, 11 min per 31.87M reviews; the pool has 49.2M); each `+vq` label unknown, roughly 25-70 min.
  Total roughly 2.5-4 h, ending around 23:30-01:00.
- GPU scorer speed with the codec: **to be measured** (`check-sm44-gpu.json` `timing.reviews_per_second`; also the
  model-time rate of every label in the summary).

## Models, labels, codebooks
- SM42 (best model), label `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes`:
  `C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay1ep32k-lanes/expertium/E89-F65e-F77a/master.pth`
- SM32c, label `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes-vqlearn`:
  `wt/sm32c/artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay1ep32k-lanes-vqlearn/expertium/E89-F65e-F77a/master.pth`
  - Learned books: `state-vq-books.npz` next to the checkpoint, sha256 `a4d487b1...`.
- k-means books: `artifacts/small-model/state-vq/codebooks.npz`, sha256 `eab6db58...`. This one file holds every
  recipe: WKV 8 + 4 stages, shift m2 and m4 x 3 stages, norm grids 1-8 bits.
- Bits per entity (card / note): `vq378x412` = 378 / 412; `vq770x828` = 770 / 828.

## Outputs
- `sm44/check-sm44-gpu.json`: GPU check
- `sm44/heldout-sm44.jsonl`: rows
- `sm44/sm44-summary.md` and `sm44/sm44-summary.json`
- Summary content:
  - Review-weighted LogLoss, with a paired user bootstrap (2,000 resamples, seed 0, 2.5/97.5 percentiles).
  - % vs the same model's w8+s8, vs SM42 w8+s8, and vs fork w8+s8 (the wt/small-model `heldout.jsonl` +
    `heldout-pools.jsonl` rows; 731 of the 732 users).
  - SM42 w8+s8 reproducibility (max |dLogLoss|) and reviews/s.

## Follow-up commands
```bash
cd C:/Users/jonat/workspace/wt/sm32c
grep -aE '^\[sm44|\[gpu-check\]|\[queue\]' artifacts/small-model/sm44/runbook-sm44.log | tail -30
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -c "import json; r=json.load(open('artifacts/small-model/sm44/check-sm44-gpu.json')); print(r['passed'], r['fails'], r.get('timing',{}).get('reviews_per_second'), r.get('base_float_state_gpu_vs_cpu'))"
cat artifacts/small-model/sm44/sm44-summary.md
# runbook stopped or labels incomplete: rerun it; it resumes, done users are skipped and a passed GPU check is reused.
# WAIT_SM43=0 skips the SM43 wait.
nohup bash C:/Users/jonat/workspace/wt/sm32c/artifacts/small-model/sm44/runbook-sm44.sh >> C:/Users/jonat/workspace/wt/sm32c/artifacts/small-model/sm44/runbook-sm44.log 2>&1 & disown
# re-pair after extra rows (CPU, seconds)
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm44/sm44_summary.py
# commit (rows + summary + check + log), then push
git add -f artifacts/small-model/sm44/heldout-sm44.jsonl artifacts/small-model/sm44/sm44-summary.md artifacts/small-model/sm44/sm44-summary.json artifacts/small-model/sm44/check-sm44-gpu.json artifacts/small-model/sm44/runbook-sm44.log
git -c user.name="Jonathan Schoreels" -c user.email=jonathan.schoreels@gmail.com commit -m "SM44: GPU scoring of compressed state on 732 users (rows, summary)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```
- If the GPU check fails, read `check-sm44-gpu.json`:
  - `base_float_state_gpu_vs_cpu` large too: base GPU numerics, not the codec.
  - Only `+vq` off: a codec pick flip on GPU (FMA, log2f/exp2f ulps). Look at `n_dp_over_1e-4` per user.
- Do not edit `scripts/` in wt/sm32c while a scoring ticket runs (the runbook refuses to start with uncommitted
  source changes; the scorer itself has no source watcher). Never edit the runbook while it runs.
