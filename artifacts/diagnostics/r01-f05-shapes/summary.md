# F05 CUDA Graphs on the R01 path: shape audit and decision (r01-structural, 2026-10-13)

**Recommendation: drop whole-update CUDA Graphs (exact-shape and bucketed) for R01; do not spend a counted iteration on it.** No graph runtime was built, because the recorded data shows nothing for a graph to replay. The shape-independent tail is the only replayable part, and it is too small to clear the timing-noise band.

## Evidence

All CPU data are measured from the accepted bundle (0072 options from `codex-round-0071-0073-handoff.json`, protocol `iteration-0072/protocol.json`, the same 80 users, 167 updates and 512-chunk spans), prepared with the production preparation path; raw records in [audit-1/result.json](audit-1/result.json) (produced by `scripts/gpu_r01_graphs.py audit`, 125 s on one CPU thread).

1. **Exact shapes never repeat.** 167 updates give 167 distinct hierarchy keys (the `DynamicGraphs` key: sample count, split lengths, every index tensor shape) and 167 distinct full keys (including the `gpu_packed_features` table sizes). Second-use admission therefore yields **0 hits**; every update would stay eager, and a captured graph would never be replayed. 87 updates have the full 512 samples, but their split structure still differs.
2. **Hierarchical buckets (existing `gpu_shape_buckets.py`) do not create repeats either.** Default policy (lengths 1..256, count quantum 8, work cap 1.5): 166 admitted, 1 eager, recurrent padding **+9.9%**, but **166 distinct keys, 0 replays**. Coarser quanta: q32 +11.8% (159 keys), q128 with cap 2.0 +20.1% (154 keys), power-of-two counts with cap 2.0 +53.1% (166 keys). Eight counts per length bucket per stream must all match, which they practically never do.
3. **Fixed-skeleton tiers (pad every update of a tier to one capacity) do produce replays but cost 3 to 4 times the work.** [fixed_tier_padding.txt](fixed_tier_padding.txt): one fixed tier padded to the per-bucket maximum gives 165 replays at **+369% recurrent positions**; four tiers by sample count (16/64/256/512) give 162 replays at **+265%**; using the 90th percentile capacity leaves most updates eager. A device-bound large update would run 3 to 4 times longer, far more than any dispatch saving (a projection from position counts, not a timing).
4. **The shape-independent tail is small.** Bounded microbenchmark ([tail-1/result.json](tail-1/result.json), gpu claim, 10 warm-up and 50 repeats): the accepted grouped master/child downcast, gradient transfer, clipping and AdamW over 444 tensors take **5.39 ms host (5.31 to 5.60) and 5.36 ms device** per update, about 2.4% of the 225 ms median recorded per-update host dispatch (0072, `updates.jsonl`). The host and device times are equal, so a graph could remove at most the launch overhead, below the 6.4% Expertium noise band. It is also not bitwise safe by default (a capturable AdamW/learning-rate path changes the arithmetic).
5. **Existing machinery is stale for the accepted bundle.** `GraphTrainingAdapter` copies only `start`, `sub_gather`, `time_shift_selects`, `skips`, labels into static buffers; the accepted bundle also carries packed feature tables (`gpu_packed_features`), worker scatter orders and shared shift orders, and `grouped_sync` is explicitly refused together with graph replicas. Making it work would need a rewrite for a variant that has nothing to replay.

## Considerations that were not needed

Capture cost, VRAM headroom (14.2 to 14.9 of 16 GB reserved in accepted runs, a 3 to 4 times larger padded activation set for the tier variant would not fit), NVRTC launch capture and graph-safe dropout RNG were not measured: the audit removes the variants before those questions arise. Not built: `check_r01_cuda_graphs.py`, wiring in `gpu_r01_training.py`, the prefetch allowlist, `R01SourceWatch`, `describe()` (no variant to wire; the training-source claim was held only for the microbenchmark). No peak-VRAM or allocator-retry numbers exist for F05 because no graph ran.

## What could still make F05 worthwhile (not pursued)

- Changing the batch schedule so shapes repeat by construction (fixed-size packed batches). That changes training exposure and needs the human's scope decision (P-E) and the quality gates.
- Regional compilation or fusing of the roughly 60 small ops per layer instance, which is a different track (F08 style fusion), not a graph.

## Next command if the human nevertheless wants a counted F05 iteration

There is no runnable F05 option for the accepted bundle. The older `dynamic_graphs` option is not in the `gpu_r01_prefetch.py` allowlist and cannot replay anything on this cohort. Suggested ledger row text for F05 (the ledger was not edited by this session): "R01 audit 2026-10-13 (r01-structural): 167/167 distinct exact shapes, 0 second-use hits; default buckets +9.9% padding but 166 distinct keys; fixed-skeleton tiers +265% to +369% work; update tail 5.4 ms host/5.4 ms device (about 2.4%). Whole-update graphs dropped for R01; evidence `artifacts/diagnostics/r01-f05-shapes/summary.md`."
