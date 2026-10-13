# F71: one preparation worker per physical core (2026-10-13)

The human asked to pin the preparation workers one per core after the [baseline-drift diagnostic](../live/summary.md) showed Windows sometimes packs them two per physical core. F71 is implemented as `host_placement: "physical"` on branch `f71-worker-core-placement`, commit `f1ebc8b73d707126a8e8557ebe90467e581e3cd1` (worktree `../rwkv-f71`, not merged into `main`). Each prestarted worker gets the first logical CPU of its own physical core; the main process gets both SMT siblings of the best-ranked remaining cores (here CPUs 6, 7, 10, 11, CPU-set class 6). The previous main-process affinity is restored right after `training_seconds` is recorded, because the evaluation pool would otherwise inherit four CPUs (the external mode-C screen showed validation rising from about 29 s to 46–48 s; with the restore, validation stayed 34–35 s in both arms). Placement only: arithmetic and batch order are unchanged.

## Measurement

Uncounted interleaved replicate blocks (not a counted R01 iteration; no round budget consumed), run from the worktree's own artifacts tree under the main checkout's `gpu` claim (`scripts/r01_claim.py`, owner `baseline-drift-f71`) with [run_f71_blocks.py](run_f71_blocks.py). Protocol: the accepted 0072 protocol re-frozen as diagnostic iteration 9171 at `f1ebc8b` (source archive `66657730…`), adding `E33-F48-F50-F65b-F71` = accepted 0072 options + `host_placement: "physical"`; both arms declared study tracks. Order control → candidate in every block. Raw results, updates, resources and console logs: [runs/](runs/); statistics: [f71-replicate-stats.json](f71-replicate-stats.json).

| Block | Control E33-F48-F50-F65b | F71 | Ratio |
|---|---:|---:|---:|
| 1 | 168,404 | 179,076 | +6.34% |
| 2 | 173,083 | 180,481 | +4.27% |
| 3 | 164,864 | 187,529 | +13.75% |
| 4 | 162,475 | 188,443 | +15.98% |
| Mean | 167,159 | 183,835 | +9.98% |

`scripts/r01_replicate_stats.py --calibrated-sd 1.01`: trimmed 5-MAD per-update gain **+9.22%**, one-sided 95% lower bound **+8.32%**, t interval [+7.53%, +10.93%] → promote under the replicated-block rule. All eight runs share one final parameter hash `8141e9b120b6…`, one validation-input hash and an identical evaluation object (all four held-out losses), 167 updates and 7,654,975 reviews.

Mechanism (per-run telemetry): controls ran in the packed state (summed preparation 156.6–158.4 s, worker CPU 339–364%); F71 held preparation at 113.0–113.6 s and worker CPU at 257–270%, and main-thread dispatch fell from 39.7–42.5 s to 36.5–38.6 s. Queue wait was 0.11–0.53 s with F71 versus 0.42–0.76 s.

Caveats: control always ran first in a block; the gain widened as the session progressed (controls drifted 168→162k while F71 rose 179→188k). A slow monotone drift cannot plausibly produce an 8% lower bound, but an alternating-order confirmation is cheap. The size of the gain depends on how often the machine is in the packed state: on 2026-10-13 earlier runs in the spread state, external pinning showed smaller, unproven gains (live mode B/C).

## Next steps

- Promotion into the accepted line requires a counted iteration (0074) on the human's explicit request; the human also decides whether to merge `f71-worker-core-placement` (it changes training source, so it needs the `training-source` and `git` claims).
- F72 main-thread budget audit: per-update `time.thread_time` by phase (receive/unpack, launches, label counting, telemetry, GC) to size zero-copy shared-memory/pinned batch slots and to complement F05 CUDA Graphs (owned by the `r01-cuda-graphs` session).
- F73 worker count with F71 pinning (4/5/6): pinned preparation is faster, so fewer workers may keep up and free cores for the main process.
