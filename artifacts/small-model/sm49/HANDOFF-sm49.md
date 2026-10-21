# SM49 handoff: smaller int states for SM36 (int4 notes, int4 shift vectors, per-head scales)

Branch `sm49`, worktree `C:/Users/jonat/workspace/wt/sm49`, code commit 3e7cde73. **Status: GPU ticket stopped at the strict GPU check; nothing is queued. Requeue needs the user's decision on the check rule (below).**

## Variants (all f16 scales; bits from scripts/state_vq_report.int_entity_bits; collection 13,973 cards + 13,198 notes)

| Name | Variant | Card bits | Note bits | Collection MB | vs w8+s8 |
|---|---|---:|---:|---:|---:|
| SM36 ref | w8+s8 | 26,624 | 14,336 | 70.15 | — |
| SM47 smallest | w8+s4c8nhx8 | 11,824 | 10,272 | 37.60 | -46.4% |
| A | w8+s4hx8 | 11,824 | 6,176 | 30.84 | -56.0% |
| B | w8+s4hx4 | 11,056 | 5,664 | 28.65 | -59.2% |
| C | w8+s4hgx4 | 9,136 | 4,704 | 23.72 | -66.2% |
| C' (C cards, SM47 notes) | w8+s4hgx4c8hx8n | 9,136 | 10,272 | 32.90 | -53.1% |

Names: `s4` = int4 card and note rows, `x4` = int4 shift vectors, `g` = one scale per head (16 rows) instead of per row (kernel flag `HEAD_SCALE = 512` in scripts/small_model_wkv.py: warp max of the row absmax before the f16 rounding); per-stream form = card spec + `c`, note spec + `n`, as in `s4c8n`.

## Checks

- CPU (check-sm49-cpu.json, passed): w8+s8, w8+s4c8n, w8+s8hx8, w8+s4c8nhx8 reproduce SM47's CPU rows exactly, and the emulated w8+s4c8nhx8 reproduces SM47's emulated rows; the GPU dispatch run on CPU tensors equals the CPU scorer exactly (0.0) for the 4 new variants; bits as expected.
- GPU (ticket 000593, 2026-10-21 08:59, 23 s; check-gpu-000593.json / .jsonl): the 4 old variants match the CPU rows within 2e-6 and SM47's GPU rows exactly (0.0), so the kernel without the new flag is unchanged. New: A 6e-6 (pass), B 2.0e-5, C 2.2e-5, C' 2.7e-5 (fail at the strict 1e-5 rule).
- Diagnosis (CPU only, noise_check.py, noise-cpu.jsonl, check-gpu-diagnosis.json): a relative perturbation of +-2^-23 on every value entering the int rounding moves the CPU results by 1.6e-5 (A), 2.4e-5 (B), 2.9e-5 (C), 2.7e-5 (C'), against 1e-6 to 3e-6 for the old variants. The GPU gaps are the same size as this float-noise effect, so they come from rounding flips in the coarse int4 formats (int4 shift vectors with 15 levels, int4 note rows), not from a bug. Under the discrete-codec rule (per user |d| <= 3e-4, |review-weighted mean| <= 1e-4) all 4 pass: worst per-user 2.7e-5, worst mean 1.4e-5.

## User decision needed

AGENT.md keeps the strict 1e-5 rule for int formats; these int4-note / int4-shift formats cannot meet it. Either approve the discrete rule for SM49's new variants (old variants stay strict), or drop B/C/C'. The job supports this: `SM49_GPU_RULE=discrete` (job-sm49.sh; default strict).

Requeue (after approval; ~60 min of GPU, must end before 22:00; the launcher waits for HOLD-chain and quiet windows):

```powershell
Start-Process -FilePath 'C:\Program Files\Git\usr\bin\bash.exe' -ArgumentList '-c', '"export PATH=/usr/bin:/mingw64/bin:$PATH SM49_GPU_RULE=discrete; /usr/bin/bash C:/Users/jonat/workspace/wt/sm49/artifacts/small-model/sm49/launch-sm49.sh >> C:/Users/jonat/workspace/wt/sm49/artifacts/small-model/sm49/launch-sm49.log 2>&1"' -WindowStyle Hidden
```

Then watch `launch-sm49.log` for `[sm49 <date>] done` or `STOP`. Results: `heldout-sm49-pools.jsonl` (723 users = the 732 pool minus the 9 never-fit users, users-sm49.json, `--lanes 1`), `sm49-summary.md` / `.json` (paired vs SM36 w8+s8 and vs each base: A, C' vs SM47 w8+s4c8nhx8; B vs A; C vs B; 464 subset too). Rerun the summary by hand: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm49/sm49_summary.py`. Commit with `git add -f` of check-gpu.json(l), heldout-sm49-pools.jsonl, sm49-summary.*, launch-sm49.log.

## Part 4: lossless compression on the own collection (CPU, done; collection-zstd.md / .json, aggregate only)

States after the whole history of the 2026-10-14 collection copy (anki-rwkv/artifacts/collection-eval/2026-10-14, read-only), 13,953 cards / 13,178 notes with a review. Dumps are private (scratch, outside git).

| Variant | Raw MB | zstd-19 columnar | zstd-19 one file of records | per-entity zstd-19 | per-entity zstd-3 + 64 KB dict | order-0 entropy |
|---|---:|---:|---:|---:|---:|---:|
| w8+s4c8nhx8 | 37.54 | 32.15 (1.17x) | 33.45 (1.12x) | 36.20 (1.04x) | 36.01 (1.04x) | 32.90 |
| C w8+s4hgx4 | 23.68 | 13.58 (1.74x) | 13.97 (1.70x) | 17.08 (1.39x) | 16.07 (1.47x) | 14.80 |

With one scale per head most int4 codes are small (2.4-2.6 bits of order-0 entropy instead of 4), so C compresses much better than per-row formats. Collection LogLoss of the same runs: w8+s4c8nhx8 -0.06% / +0.11% vs fork w8+s8, C +0.31% / +0.48%.
