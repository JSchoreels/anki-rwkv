# Codex R01 round: iterations 0051–0053

The accepted preliminary expertium reference is **0042: 197,499 reviews/s**, +0.00% versus the round’s starting accepted reference and +558.22% versus the fixed original baseline. Implementation commit `4e9bae201a7b1a457a6b95a3f4680a2543218543` is separate from experiment revisions: counted candidates froze source before it; the last 0053 replica originated from it. Raw archives remain authoritative.

Authorization: I want you to follow R01 cycles, spawn each time a new GPT 6.1 Sol medium thread for 3 iterations, until you reach iteration number 70. Go up to 70 and then shutdown the computer.. Explicit count: 6 of 23 iterations completed; 17 remain authorized. Consecutive misses do not end this counted round.

## Full-history results

Training seconds and raw review presentations/s include user transitions; complete process seconds also include setup, discarded primer, checkpoint saving and evaluation. Gains below use each candidate’s own recorded previous accepted reference, which can differ between rows. One full timing per candidate does not establish repeatability; broader validation remains pending.

| Iteration | Bundle | Training seconds | Reviews/s | Gain vs prior accepted | Complete process seconds | Queue wait s | Summed worker preparation s | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [0042](iteration-0042/summary.md) | E33-F48-F50 | 38.759 | 197,499 | — | 90.735 | 0.47 | 118.3 | Starting accepted reference |
| [0051](iteration-0051/summary.md) | E33-F48-F50-F58-F34a | 45.710 | 167,468 | -15.21% | 99.843 | 0.80 | 154.8 | retain previous accepted run; speed target missed |
| [0052](iteration-0052/summary.md) | E33-F48-F50-F58-F34b | 45.747 | 167,334 | -15.27% | 99.833 | 0.82 | 154.8 | retain previous accepted run; speed target missed |
| [0053](iteration-0053/summary.md) | E33-F48-F50-F58-F34c | 45.762 | 167,279 | -15.30% | 100.032 | 0.83 | 154.5 | retain previous accepted run; speed target missed |

**0051, E33-F48-F50-F58-F34a.** P4 two-stream splits increased reserved memory; recurrence-only joins serialize calls. Fall back to exact fused time-mixer gate addition and multiplication, preserving BF16 rounding and original backward operations, stacked on F58. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -15.21% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.80 versus 0.47; summed preparation s: 154.76 versus 118.28; validation s: 31.69 versus 29.63; primer s: 15.29 versus 15.31; worker CPU %: 58.71 versus 53.23; GPU activity %: 64.12 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

**0052, E33-F48-F50-F58-F34b.** F34 forward-only slice was negative in 0051. Test the changed forward/backward gate mechanism: fuse independent backward products into one launch, preserve the same rounded gradient for both add inputs, stack on retained F58. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -15.27% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.82 versus 0.47; summed preparation s: 154.76 versus 118.28; validation s: 31.55 versus 29.63; primer s: 15.37 versus 15.31; worker CPU %: 58.81 versus 53.23; GPU activity %: 62.51 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

**0053, E33-F48-F50-F58-F34c.** Gate forward and forward/backward slices were negative under contemporaneous blocks. Test exact channel-mixer ReLU-square forward glue with original threshold/square backward order, stacked only on F58; omit both gate variants and rejected F55/F62. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -15.30% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.83 versus 0.47; summed preparation s: 154.51 versus 118.28; validation s: 31.79 versus 29.63; primer s: 15.32 versus 15.31; worker CPU %: 58.45 versus 53.23; GPU activity %: 66.33 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

## Replicated-block decisions and preparation

The frozen gate uses 5-MAD trimmed per-update effects, looks 2/4/6, Pocock boundary 1.9927 and SD at least calibrated 1.01%. All bundles had negative means at the first look; no extension was required. Reference means below are contemporaneous replicas.

| Iteration | Blocks | Reference mean reviews/s | Candidate mean reviews/s | Mean gain | Lower bound | Decision |
|---|---:|---:|---:|---:|---:|---|
| 0051 | 2 | 170,204 | 167,599 | -1.654% | -3.037% | reject |
| 0052 | 2 | 169,482 | 167,212 | -1.288% | -2.676% | reject |
| 0053 | 2 | 169,990 | 167,592 | -1.207% | -2.596% | reject |

All twelve replicas and three counted candidates retain exact recorded parity; all four held-out losses equal 0042 and satisfy the 0.001 cumulative ceiling. No new stack candidate remains. These F58 bundles do not isolate independent F34 effects. Best-of paired wall diagnostics are −1.79%, −1.40%, −1.41%; they support attribution but never replace the gate. Replicas show zero retries/OOMs, 12,188 MiB peak reserved and similar high clocks; no concrete anomaly justified a repeat. Standalone pre-launch CPU: 15.7%, 7.9%, 9.2%; replicas obeyed the at-most-10% gate.

P4 recurrence-only calls were exact and effectively memory-neutral but immediate parent joins serialize dependent calls. Two bounded split streams were exact but grew reserved memory 94–436 MiB on smaller five-batch diagnostics; full-history overlap is unmeasured. Thus 0051 used gate-forward F34 fallback. Its rejection led to changed fused backward in 0052, then distinct channel ReLU-square in 0053 with gate omitted. Thirty BF16/FP32 synthetic cases and five real batches per slice passed before binding. A duplicated launcher insertion was repaired before protocol freeze and binding checks refreshed. [Preparation, checks and paired state flags](../../diagnostics/r01-round-0051-0053/preparation.md).

Accepted options remain precisely 0042 E33-F48-F50. Prior 0048 F58 remains positive unproven (+0.110% at six blocks); retain `recurrence_prefetch=true` only for stacking. Omit F55/F62 and all new glue bundles. Next `--iteration 0054 --previous 0053`, inherited budget 23/index 6, 17 remaining, loss ceiling 0.001. Alex and broader validation remain paused; this worker created no follow-on chat and did not shut down.

## Separate held-out binary losses

| Iteration | Metric | Loss | Delta vs prior accepted | Delta vs original baseline |
|---|---|---:|---:|---:|
| 0051 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0051 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0051 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0051 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0052 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0052 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0052 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0052 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0053 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0053 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0053 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0053 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |

## Checks and provenance

Full recorded parity is exact in 3/3 cases. The recorded acceptance gates and decisions are reused; this report does not change them or substitute parity for broader quality validation.

- 0051: experiment `1518f099728c54cd6bf4f79547aaf9a9506149c5`, origin `015e78d3a93e108e536dcc6be875a8f3be52633f`, dirty at process start: True; [source archive](../../sources/experiment-sources/241f02ce7dae767744d50ff945a9f31adc9fec02080565efb2242e8ace787403/source.zip), [protocol](iteration-0051/protocol.json), [comparison](iteration-0051/comparison.json).
- 0051 pre-run background: 15.7% total CPU; 0 other Python GPU processes detected.
- 0052: experiment `291db556a3d410b91cc01ac4a24db34d21dc8134`, origin `015e78d3a93e108e536dcc6be875a8f3be52633f`, dirty at process start: True; [source archive](../../sources/experiment-sources/cb10504953d31e28fab7eac5195839237c4fa9cbcd0f0fa4fb6ba38c5b552687/source.zip), [protocol](iteration-0052/protocol.json), [comparison](iteration-0052/comparison.json).
- 0052 pre-run background: 7.9% total CPU; 0 other Python GPU processes detected.
- 0053: experiment `397b44ac282bd581b07c7c8eeb001192b61501f0`, origin `015e78d3a93e108e536dcc6be875a8f3be52633f`, dirty at process start: True; [source archive](../../sources/experiment-sources/1efb531c9bb0330a4c9df290cb01e12f77d8a707500453d745483e5ac771ac0c/source.zip), [protocol](iteration-0053/protocol.json), [comparison](iteration-0053/comparison.json).
- 0053 pre-run background: 9.2% total CPU; 0 other Python GPU processes detected.

## Pending next steps

- Continue --iteration 0054 --previous 0053; inherit campaign budget 23/index 6 with 17 remaining through 0070 and cumulative loss ceiling 0.001; do not reset authorization.
- Accepted reference remains 0042 E33-F48-F50. Retain 0048 F58 recurrence_prefetch=true as positive unproven stack; omit rejected F55/F62 and 0051-0053 F34 gate/channel bundles.
- For 0054 slot 7, establish exact feasibility for a distinct F34 group (eight-way shift/lerp reductions, bonus/GroupNorm glue or vectorized loading); absent feasibility advance to F51 RNG-preserving split batching while preserving dropout and per-split weight-gradient order.
- F49 needs main-stream preallocated outputs plus genuinely independent deferred recurrence scheduling; bounded two-stream split memory grew 94-436 MiB and immediate recurrence joins serialize calls. P3b attribution, whole-group F62 Adam, F58 recompute/per-row prefetch remain pending. No Alex/broader validation; parent manages follow-on chats and shutdown.
