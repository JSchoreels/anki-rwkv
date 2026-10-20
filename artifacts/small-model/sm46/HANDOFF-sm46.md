# SM46: large-budget compressed state (~6,000 / ~3,000 bits) on SM36 (handoff, 10-20)

Asked by the outside researcher whose codec we reimplemented: the codebook state compression at about 6,000 bits per
card and 3,000 per note, scored after training (no retraining) on the best small model, SM36
(`s-w64kd-gate-sc-lanes-dec12-1ep32k-muon`). int8 card state is 26,624 bits, so this is ~4.5x smaller than int8.

Worktree `C:/Users/jonat/workspace/wt/sm46`, branch `sm46` (= sm45 + this). Docs: docs/SMALL_MODEL.md section
"Large-budget compressed state (SM46)"; docs/GPU_OPTIMIZATIONS.md row `SM46-vq`.

## Recipe `vq5902x3028` (scripts/state_vq.py)
- Card **5,902 bits** = 8 WKV heads x 590 + 3 shift vectors x 394.
- Note **3,028 bits** = 4 heads x 560 + 2 x 394.
- Per 16 x 16 head:
  - rank 6, eigh factorization (rank 6 keeps 99.8% of the head energy, rank 4 98.9%);
  - joint (u, v) residual VQ with 10-bit index stages: (15, 13, 10, 7, 5, 3) for the card, (15, 12, 9, 7, 4, 3) for the
    note;
  - 10-bit log2 norm per component.
- Shift vectors: m4 sub-vectors x 8 stages of 12-bit codes + 10-bit norm.
- New in the code (only the new recipes use it):
  - `fit_norm_range` with bits >= 10 spans the whole sample (largest norm, 20 octaves below). The 1-99% quantile search
    clipped the top norms, which dominated the norm error at 10 bits.
  - On CUDA the torch-loop streams use 32,768-row nearest-centroid chunks (`SCORE_TORCH_CHUNK`).

## How it was chosen
1. Corpus: SM36's stored states from 50 SM36 training users (protocol.json rows, 3k-60k reviews, none of the 732 pool),
   1,500 samples per stream-layer and user (`books/corpus`, 918 MB, not committed).
2. k-means on 45 users (every 10th corpus user held out): WKV ranks 1-6 with (32, 24, 16, 12, 8, 6) stages, shift m4 x 8
   stages, norms 6/8/10 bits.
3. Model-free proxy (`alloc_proxy.py` -> `alloc-proxy.json`): relative squared reconstruction error on the 5 held-out
   users.
   - Rank 6 with equal per-component coding errors: 1.04e-2 at 630 bits per head.
   - Rank 4 at 600 bits: 1.45e-2. Rank 3 at 590 bits: 2.87e-2. Rank-6 truncation alone: 2.06e-3.
   - Shift m4: x6 = 2.0e-3 (time mixer) / 1.1e-3 (channel mixer); x8 = 6.8e-4 / 4.3e-4.
4. Training-user screen (`state_vq_fit.py tune`, 8 SM36 training users, 115k reviews, model-side LogLoss vs w8+s8;
   `screen-summary.txt`, rows `books/tune.jsonl`):

   | Recipe | Card / note bits | Curve | Instant |
   |---|---|---:|---:|
   | `vqr6exact` (rank-6 truncation only) | - | -0.000015 | -0.000001 |
   | `vq5934x2996` (shift m4 x 6, more WKV stages) | 5,934 / 2,996 | +0.000048 | +0.000152 |
   | `vq5902x3028` (shift m4 x 8; screen name `vqS46sh`) | 5,902 / 3,028 | +0.000058 | +0.000141 |

   The two recipes tie within noise (instant better on 4 of 8 users). Instant is the binding head, so `vq5902x3028`.
   For scale: the SM32 screen's 1,506 / 828-bit point cost +0.00009 / +0.00058 (another model and other users).
5. A "zero norm code" for rank-deficient states was tried in the proxy and then removed: with the full-range 10-bit grid
   it changed nothing (2.073e-3 both ways).

## CPU check (`check-sm46-cpu.json`, log `logs/check-sm46-cpu.log`): ALL OK
- The GPU scorer's code ran on CPU tensors: both streams on the torch step loop with the eigh float64 codec (rank >= 2
  has no fused kernel).
- It reproduces the CPU scorer `+vq5902x3028` **bitwise** on the 3 smallest 732-pool users (2388, 4676, 1712; 3,685-4,090
  reviews): max |dp| = 0, dLogLoss = 0 on both heads.
- The codec is active: float state moves p by up to 0.04. The slot path ran: 10-13 WKV and 17-22 shift calls per user.
- Stored references for the GPU ticket are in `check-ref/` (recipe + float state).
- CPU cost:
  - 3.5 ms per review (CPU scorer, 4 threads, the small check users);
  - 134 reviews/s in the screen at 2 threads, vs 1,020/s for w8+s8 (7.6x slower).

## GPU time estimate
- Both streams use the per-step torch loop on the GPU, about 570 small ops per codec step.
- Step profile (`step-profile-732.json`, 76 of the 732 pool users by id; stopped early to free the CPU): ~19.3k codec step
  iterations (card active steps x 2 layers + note) per 1M reviews. That is ~0.95M on the 732 pool (49.2M reviews) and
  ~0.62M on the 464 pool.
- At an assumed 3-8 ms per step (not measured on the GPU), plus the base model (~17 min), the 732 pool takes **~1.2-2.5 h**
  and the 464 pool ~0.8-1.6 h.
- The runbook does not trust this estimate. It uses the GPU check's measured reviews/s on 2 near-median users (+20%):
  - 732 projects <= 240 min: it scores the 464 pool first, then the rest of 732;
  - otherwise: only the 464 pool, logged as "scoring only the 464 pool".
- Tickets are <= 110 min each, with up to 6 passes per pool. `--time-limit` leaves room for the largest remaining user.
- The dry run used a placeholder rate of 3,000/s and therefore printed "464 only". Ignore that line.

## Running (detached)
- Runbook/launcher: `C:/Users/jonat/workspace/wt/sm46/artifacts/small-model/sm46/runbook-sm46.sh`
- Log: `C:/Users/jonat/workspace/wt/sm46/artifacts/small-model/sm46/runbook-sm46.log`
- Dry run: `runbook-sm46-dryrun.log`.
- Launched 10-20 17:26:21 from commit `053b5e34`. MSYS bash pid 3350, under the nohup wrapper 3348.
- Order of steps:
  1. Wait for the SM44 runbook log (`wt/sm32c/artifacts/small-model/sm44/runbook-sm44.log`) to show `[sm44 ...] done`
     or `[sm44 ...] STOP`. The SM44 runbook itself waits for SM43 full (~20:50) and the 22:00-01:00 break, so SM46 will
     likely start after ~02:00-04:00 on 10-21.
  2. Wait while HOLD-chain exists. This check runs before every ticket.
  3. GPU check ticket, 25 min expected. Gate: benchmark LogLoss within 1e-5 per user and head vs the stored CPU
     predictions. It also reports base float-state GPU vs CPU numerics and reviews/s. **The runbook stops on a
     mismatch.**
  4. Scoring tickets as described above. Rows go to `heldout-sm46.jsonl`. GPU memory: the scorer's default cap 0.8.
  5. CPU summary `sm46_summary.py` writes `sm46-summary.md` / `.json`, for the 732 pool and its 464 subset:
     - review-weighted LogLoss;
     - delta and % vs SM36 w8+s8 (wt/sm42 heldout-sm36*.jsonl), paired user bootstrap 2,000 / seed 0 / 2.5-97.5;
     - % vs fork w8+s8 (wt/small-model heldout*.jsonl) on the same users.

## Codebooks (not committed: 29.7 MB)
- `artifacts/small-model/sm46/books/codebooks-sm46.npz`, sha256
  `af46e6a253c4f18e349792527c815860006924c301f70c3924a4872c38cf81f7`. The runbook pins this hash.
- Regeneration (CPU, ~75 min with the two fits in parallel). k-means is seeded, but multi-threaded float sums may not
  reproduce the file bitwise: check the sha256 and re-run the CPU check if it differs.
```bash
cd C:/Users/jonat/workspace/wt/sm46
CK=C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon/expertium/E89-F65e-F77a/master.pth
P=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
A="--student $CK --out-dir artifacts/small-model/sm46/books"
for i in 0 1; do $P -X utf8 scripts/state_vq_fit.py $A --exclude-pool 732 collect --users 50 --shard $i/2 --per-user 1500 & done; wait
VQ_FIT_THREADS=4 $P -X utf8 scripts/state_vq_fit.py $A fit --parts wkv --holdout-every 10 --max-points 200000 --wkv-stages 32 24 16 12 8 6 --norm-bits 6 8 10 --out codebooks-sm46-wkv.npz &
VQ_FIT_THREADS=4 $P -X utf8 scripts/state_vq_fit.py $A fit --parts shift --holdout-every 10 --max-points 300000 --shift-m 4 --shift-stages 8 --norm-bits 6 8 10 --out codebooks-sm46-shift.npz & wait
$P -X utf8 scripts/state_vq_fit.py $A merge --wkv codebooks-sm46-wkv.npz --shift codebooks-sm46-shift.npz --out codebooks-sm46.npz
VQ_FIT_THREADS=4 $P -X utf8 scripts/state_vq_fit.py $A refit-norms --codebooks codebooks-sm46.npz --holdout-every 10 --max-points 200000 --norm-bits 10
```
(The WKV fit ran before the full-range 10-bit norm rule existed; `refit-norms --norm-bits 10` above set the 10-bit ranges
the recipe uses.)

## Follow-up commands
```bash
cd C:/Users/jonat/workspace/wt/sm46
grep -aE '^\[sm46|\[gpu-check\]|\[queue\]' artifacts/small-model/sm46/runbook-sm46.log | tail -30
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -c "import json; r=json.load(open('artifacts/small-model/sm46/check-sm46-gpu.json')); print(r['passed'], r['fails'], r.get('timing',{}).get('reviews_per_second'), r.get('base_float_state_gpu_vs_cpu'))"
cat artifacts/small-model/sm46/sm46-summary.md
# runbook stopped or pool incomplete: rerun it (resumes; a passed GPU check is reused). WAIT_SM44=0 skips the SM44 wait.
nohup bash C:/Users/jonat/workspace/wt/sm46/artifacts/small-model/sm46/runbook-sm46.sh >> C:/Users/jonat/workspace/wt/sm46/artifacts/small-model/sm46/runbook-sm46.log 2>&1 & disown
# re-pair after extra rows (CPU, seconds)
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm46/sm46_summary.py
git add -f artifacts/small-model/sm46/heldout-sm46.jsonl artifacts/small-model/sm46/sm46-summary.md artifacts/small-model/sm46/sm46-summary.json artifacts/small-model/sm46/check-sm46-gpu.json artifacts/small-model/sm46/runbook-sm46.log
git -c user.name="Jonathan Schoreels" -c user.email=jonathan.schoreels@gmail.com commit -m "SM46: GPU scoring of vq5902x3028 on SM36 (rows, summary)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```
- If the GPU check fails, read `check-sm46-gpu.json`:
  - `base_float_state_gpu_vs_cpu` large too: the base GPU numerics, not the codec.
  - Only `+vq` off: cuSOLVER eigh vs LAPACK, or matmul rounding flipping near-tie picks. Look at `n_dp_over_1e-4` per
    user.
- If the SM44 runbook dies without a `done` / `STOP` line, this runbook waits forever. Kill it (pid 3350) and relaunch
  with `WAIT_SM44=0` once the GPU is free.
- Do not edit `scripts/` in wt/sm46 before the scoring ends: the runbook checks for a clean tree only at start, but the
  scorer imports the files when each ticket starts. Never edit the runbook while it runs.
- Follow-ups:
  - Fill the SM46 results into docs/SMALL_MODEL.md and the factory page.
  - If the cost vs w8+s8 is ~0, a mid budget (~1,500-3,000 bits) is the next point of the curve.
  - A rank-r fused kernel (GPU_OPTIMIZATIONS `SM46-vq`) only if large-budget recipes become a regular column.
