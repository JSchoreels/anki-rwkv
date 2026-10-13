# R01 0066-0068 candidate preparation

Inherited campaign 23 candidates, index 18 complete before this worker, five authorized slots left; original loss baseline and cumulative 0.001 ceiling unchanged. Accepted 0042 E33-F48-F50 remains the independent control. No recurrence prefetch alternatives are silently stacked.

## Written continuation order

1. 0066 F62b whole active-group AdamW: pack parameter, gradient and two moments by optimizer group, device, dtype and current step; apply the same foreach AdamW to one packed tensor, then scatter values back. None gradients skip all decay/state/step work. Five real batch loss/gradient/transfer/norm/three-update checks include changing active sets and signed zeros; passed 5/5.
2. 0067 F34e exact eight-way gather/lerp forward: one original kernel, with dtype conversion and interpolation formula unchanged; reconstruct the original gather/lerp autograd graph in backward to preserve deterministic reductions. Synthetic FP32/BF16 boundary-weight/signed-zero shapes passed 16/16; five-real-batch loss/every-gradient check passed 5/5.
3. 0068 F62c reusable whole-group AdamW: same mathematical buckets as F62b, but cache the four packed device buffers by active parameter identities and use cat(out=). This is a distinct persistent allocation mechanism, not an identical replay. Five real optimizer checks passed 5/5, including divergent per-parameter steps.

Implementation commit `dab8fee`; all implementation edits preceded checks. Current isolated launchers, source archives, preflight, smoke, full runs, profile and interleaved control/candidate blocks remain required. These are preliminary screens; Alex and broader validation paused. Feasibility checks consume no candidate iteration. The packing method is whole active-group arithmetic without changing weight-storage alignment; it is independent of rejected flat-storage 0050.

## Remaining preparation after this worker

Prioritize genuinely deferred memory-neutral recurrence overlap with main-stream outputs, or substantive F51/F52 token batching preserving split weight-gradient order and individual dropout calls. Exact shift/lerp backward fusion remains a distinct pending idea only if its original rounding and reductions can be preserved; avoid replaying rejected minor recurrence schedules.

## Reporting repairs

0066 post-replicate direct report failed with Windows `OSError: [Errno 22] Invalid argument` opening `docs/GPU_OPTIMIZATIONS_LOG.md`; explicit `r01_iterate.py --from-step report --agent codex` succeeded. 0067 and 0068 initial reports failed the same way, preserving their `iteration-NNNN-report-console.log` and resource sidecars; report-only numbered attempts succeeded. All completed training/profile work was reused, with no new smoke or full replay. Cause remains unresolved.

## Concrete pending mechanisms

Exact lerp backward consolidation can replace graph reconstruction in `rwkv/model/fused_shift.py` while retaining PyTorch op-boundary rounding, leading-eight reductions, `(B,T)` weight reductions and deterministic gather contribution order. This is a changed mechanism requiring synthetic and five-real-batch loss/every-gradient checks. Substantive F51 can batch input-side pointwise normalization/linear preparation across a stream's splits, but must retain original split parameter-gradient reductions, base alignment and dropout call/RNG sequence; giant linear backward reduction is not an exact substitute. F52 should remove downstream padded work rather than replay the rejected padding-index mapping, but must reproduce skip-state outputs plus residual/bonus/normalization behavior and all original dropout draws. Source review: `expertium/model/srs_model.py` split loop 314–340; `rwkv/model/rwkv_model.py` recurrence/normalization/bonus/dropout 350–378. Deferred F49 requires main-stream preallocated outputs and genuinely delayed joins with a memory-neutral prerequisite; earlier immediate joins serialized work and split streams grew reserved memory. These remain pending with throughput —.

## Final counted decisions

0066 F62b rejected at two blocks (mean -0.860%, lower -2.254%); 0067 F34e retained unproven at six (mean +0.229%, lower -0.588%); 0068 F62c rejected at two (mean -0.783%, lower -2.178%). All three full candidates and twenty replicas exact, zero allocator retries/OOMs; peak reserved 12,188-12,192 MiB. Accepted 0042 unchanged. Campaign 21/23 complete, exactly two slots remain. Continue 0069/previous0068; parent controls final chat/shutdown.
