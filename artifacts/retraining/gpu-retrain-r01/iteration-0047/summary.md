# Iteration 0047: calibration replicates of the 0042 configuration (diagnostic, uncounted)

Human direction, 2026-10-12: "better do more runs and have a better statistical understanding of our results than just merely doing 7% from the best we ever saw". Six full-history replicates of E33-F48-F50 ran back to back on 2026-10-12/13 UTC through `scripts/r01_replicates.py`. Source: frozen commit `2da7761eef93025fc415a5399bc150e7080cae5d`, runtime `runtime-0047-bf96cbadcfe3`. Each run launched only after a 5-second system CPU sample was at most 10%. Setup matches 0042: 80/20 users, seed 2025, BF16/FP32, 512 × 128-review batches, six workers, 167 updates. Record: [iteration-0047-replicates.json](../iteration-0047-replicates.json). Analysis: [calibration summary](../../../diagnostics/r01-calibration-0047/summary.md).

| Replicate | Reviews/s | Training s | Pre-launch CPU | Busy power | Notes |
|---|---:|---:|---:|---:|---|
| r1 | 179,276 | 42.70 | 8.9% | 135 W | |
| r2 | 184,497 | 41.49 | 7.4% | 140 W | Trained normally; export recovered by hand (stale record copy in the runtime) |
| r3 | 182,498 | 41.95 | 9.8% | 141 W | |
| r4 | 171,735 | 44.57 | 8.3% | 139 W | Loss concentrated in the first work segment (−28%), a start-up transient |
| r5 | 183,071 | 41.81 | 6.6% | 141 W | |
| r6 | 180,635 | 42.38 | 8.6% | 142 W | |

- **Exactness.** All six runs match 0042's initial and trained weights, held-out inputs and all four scores exactly. No allocator retries; SM clock 2,475 MHz throughout.
- **Absolute speed.** Geometric mean 180,236 reviews/s; SD of log rate 2.57%; median pairwise spread 2.12%. 0042's single 197,499 is 9.6% above this mean, and 0042–0045 ran at about 2.5 W per utilization point against about 1.7 now. The cause of that faster earlier state is unknown. Absolute reviews/s claims should use replicate means, not one promoted run.
- **Attempts.** A first attempt was refused after 20 minutes because a game held system CPU at 31–46%; the refusal is recorded. A second, detached attempt disappeared while still waiting, without launching anything; its log is kept as `iteration-0047-replicates-runner-attempt-0002.log`. The six runs came from two ordinary batches.
