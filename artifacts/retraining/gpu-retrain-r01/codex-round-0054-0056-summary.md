# Codex R01 round: iterations 0054–0056

The accepted preliminary expertium reference is **0042: 197,499 reviews/s**, +0.00% versus the round’s starting accepted reference and +558.22% versus the fixed original baseline. Implementation commit `9072bc032a61a55878f5e73aa8c3b4beab7f234b` is separate from experiment revisions: 0054 full froze before it; later runs originated from this implementation. Raw source archives remain authoritative.

Authorization: I want you to follow R01 cycles, spawn each time a new GPT 6.1 Sol medium thread for 3 iterations, until you reach iteration number 70. Go up to 70 and then shutdown the computer.. Explicit count: 9 of 23 iterations completed; 14 remain authorized. Consecutive misses do not end this counted round.

## Full-history results

Training seconds and raw review presentations/s include user transitions; complete process seconds also include setup, discarded primer, checkpoint saving and evaluation. Gains below use each candidate’s own recorded previous accepted reference, which can differ between rows. One full timing per candidate does not establish repeatability; broader validation remains pending.

| Iteration | Bundle | Training seconds | Reviews/s | Gain vs prior accepted | Complete process seconds | Queue wait s | Summed worker preparation s | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [0042](iteration-0042/summary.md) | E33-F48-F50 | 38.759 | 197,499 | — | 90.735 | 0.47 | 118.3 | Starting accepted reference |
| [0054](iteration-0054/summary.md) | E33-F48-F50-F58-F34d | 45.485 | 168,298 | -14.79% | 99.870 | 0.79 | 153.8 | retain previous accepted run; speed target missed |
| [0055](iteration-0055/summary.md) | E33-F48-F50-F58-F51a | 45.492 | 168,272 | -14.80% | 99.725 | 0.86 | 155.3 | retain previous accepted run; speed target missed |
| [0056](iteration-0056/summary.md) | E33-F48-F50-F58-F52a | 45.320 | 168,910 | -14.48% | 99.248 | 0.79 | 153.5 | retain previous accepted run; speed target missed |

**0054, E33-F48-F50-F58-F34d.** Fuse the two bonus products into one launch with BF16 op-boundary rounding and the original backward reduction order. Stack only retained F58; omit rejected gate/channel slices. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -14.79% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.79 versus 0.47; summed preparation s: 153.75 versus 118.28; validation s: 31.87 versus 29.63; primer s: 15.36 versus 15.31; worker CPU %: 59.76 versus 53.23; GPU activity %: 62.89 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

**0055, E33-F48-F50-F58-F51a.** Batch split input gathers into aligned storage, retain independent original row-backward nodes and execute modules/dropout in the original split order. Omit rejected bonus/gate/channel slices; stack only F58. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -14.80% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.86 versus 0.47; summed preparation s: 155.31 versus 118.28; validation s: 31.68 versus 29.63; primer s: 15.32 versus 15.31; worker CPU %: 60.15 versus 53.23; GPU activity %: 65.45 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

**0056, E33-F48-F50-F58-F52a.** Map negative hierarchy indices to row zero directly in one row-load kernel, eliminating separate clamp launches and preserving the original F50 padding-excluding backward. Omit rejected batching/bonus/glue slices; stack only F58. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -14.48% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.79 versus 0.47; summed preparation s: 153.47 versus 118.28; validation s: 31.38 versus 29.63; primer s: 15.38 versus 15.31; worker CPU %: 59.45 versus 53.23; GPU activity %: 58.25 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

## Replicated-block decisions and preparation

The frozen speed gate uses 5-MAD trimmed per-update effects, looks 2/4/6, Pocock boundary 1.9927 and SD at least calibrated 1.01%. The historical full-run deltas above do not decide promotion. Contemporaneous block decisions are:

| Iteration | Mechanism stacked on F58 | Blocks | Mean gain | Lower bound | Decision |
|---|---|---:|---:|---:|---|
| 0054 | Exact bonus products | 2 | -1.226% | -2.615% | Reject |
| 0055 | Aligned split gathers | 2 | -1.103% | -2.494% | Reject |
| 0056 | Fused padding-index mapping | 4 | -0.365% | -1.408% | Reject |

0056 extended from +0.261% / −1.149% at two blocks to a negative mean at four blocks; look six was therefore not required. All sixteen replicas and all three full candidates retain exact recorded parity; all four held-out losses equal accepted 0042, within the inherited cumulative 0.001 ceiling. Best-of paired wall diagnostics are −1.30%, −1.57%, −0.25%; these never replace the gate. Zero allocator retries/OOMs, matching 3157 MHz median SM clocks and approximately 12,188 MiB reserved (F51a 12,184 MiB) give no concrete anomaly warranting repetition. Bundle results do not isolate the new pieces independently from F58.

[Preparation, checks, failed diagnostics and state flags](../../diagnostics/r01-round-0054-0056/preparation.md). Thirty synthetic BF16/FP32 cases and five real loss/every-gradient batches per candidate passed on final source before freezing. Batched gather's initial TorchScript binding failure was repaired before any counted run; final checks refreshed. Report-only resumes repaired transient Windows run-log open failures in 0055/0056, without training repeats. First and final look records remain preserved. Standalone full launches passed the quiet gate at 4.0%, 3.4%, 3.7% CPU; all replicas used the existing at-most-10% gate.

F51a is the conservative input-gather hybrid, retaining split module/dropout and weight-gradient order; full per-token batching is pending. F52a removes clamp/index work only; padded mixer/recurrence arithmetic remains. Exact eight-way lerp/reductions, genuine deferred F49 overlap, whole-group F62 Adam, F58 recompute/per-row prefetch and P3b attribution remain pending.

Accepted options remain exactly 0042 E33-F48-F50. Retain prior 0048 F58 `recurrence_prefetch=true` only as an unproven stack component; omit all newly rejected slices. No new stack option survives. Slot 10 consolidation must not count an identical 0048 replay: choose a concretely changed pending mechanism. Next `--iteration 0057 --previous 0056`, budget 23/index 9, 14 remaining through 0070 and loss ceiling 0.001. This worker completed exactly 0054–0056, created no follow-on chat and did not shut down; parent controls sequencing and shutdown. Alex and broader validation stay paused.

## Separate held-out binary losses

| Iteration | Metric | Loss | Delta vs prior accepted | Delta vs original baseline |
|---|---|---:|---:|---:|
| 0054 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0054 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0054 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0054 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0055 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0055 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0055 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0055 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0056 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0056 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0056 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0056 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |

## Checks and provenance

Full recorded parity is exact in 3/3 cases. The recorded acceptance gates and decisions are reused; this report does not change them or substitute parity for broader quality validation.

- 0054: experiment `20ba77cc3dd4ea55f10dd505189fdce466f1b39e`, origin `c4ec5c46766ce3aa24fea7ee55e578f30a93fc5a`, dirty at process start: True; [source archive](../../sources/experiment-sources/5635f8bdc2b88e4f1681ead931d5031072a386c7c26493bebc233ea4b442d987/source.zip), [protocol](iteration-0054/protocol.json), [comparison](iteration-0054/comparison.json).
- 0054 pre-run background: 4.0% total CPU; 0 other Python GPU processes detected.
- 0055: experiment `c46f1052be674b8863d7505be340b722b694220b`, origin `9072bc032a61a55878f5e73aa8c3b4beab7f234b`, dirty at process start: True; [source archive](../../sources/experiment-sources/9b14a4671c2e7c7e6dc695de82b29caa8775eeafae9f383298ca3e3f20386324/source.zip), [protocol](iteration-0055/protocol.json), [comparison](iteration-0055/comparison.json).
- 0055 pre-run background: 3.4% total CPU; 0 other Python GPU processes detected.
- 0056: experiment `fa1ba472b232faa3ba436f8840aae8e51074c216`, origin `9072bc032a61a55878f5e73aa8c3b4beab7f234b`, dirty at process start: True; [source archive](../../sources/experiment-sources/1c2af31b3551c60ee16531fc426f0927909d19396acccad046209779eddf45ff/source.zip), [protocol](iteration-0056/protocol.json), [comparison](iteration-0056/comparison.json).
- 0056 pre-run background: 3.7% total CPU; 0 other Python GPU processes detected.

## Pending next steps

- Continue --iteration 0057 --previous 0056; inherit campaign budget 23/index 9, 14 remaining through 0070 and cumulative loss ceiling 0.001. This worker stops after 0056.
- Accepted 0042 E33-F48-F50 unchanged. Retain only prior 0048 F58 recurrence_prefetch=true as positive unproven stack; omit rejected F55/F62 and all 0051-0056 glue/gather bundles.
- Slot 10 consolidation has no new surviving stack component. Do not count an identical 0048 repeat; choose a concretely changed pending mechanism such as F58 recompute/per-row prefetch, genuine deferred F49 overlap or whole-group F62 Adam with exactness prerequisites.
- Full F51 per-token batching, F52 padding-free mixer/recurrence, exact eight-way lerp and P3b attribution remain pending. Preserve dropout calls, per-split weight-gradient order, compatible checks, quiet launches and frozen replicated-block looks. Alex/broader validation paused; parent controls new chats and shutdown.
