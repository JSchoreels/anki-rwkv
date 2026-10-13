# Codex R01 round: iterations 0066–0068

The accepted preliminary expertium reference is **0042: 197,499 reviews/s**, +0.00% versus the round’s starting accepted reference and +558.22% versus the fixed original baseline. Implementation commit `dab8feead8d278233f1972bd24683903a239aea7` is recorded separately from each experiment revision; 0066 froze before that commit and 0067/0068 after it.

Authorization: I want you to follow R01 cycles, spawn each time a new GPT 6.1 Sol medium thread for 3 iterations, until you reach iteration number 70. Go up to 70 and then shutdown the computer.. Explicit count: 21 of 23 iterations completed; 2 remain authorized. Consecutive misses do not end this counted round.

## Full-history results

Training seconds and raw review presentations/s include user transitions; complete process seconds also include setup, discarded primer, checkpoint saving and evaluation. Gains below use each candidate’s own recorded previous accepted reference, which can differ between rows. One full timing per candidate does not establish repeatability; broader validation remains pending.

| Iteration | Bundle | Training seconds | Reviews/s | Gain vs prior accepted | Complete process seconds | Queue wait s | Summed worker preparation s | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [0042](iteration-0042/summary.md) | E33-F48-F50 | 38.759 | 197,499 | — | 90.735 | 0.47 | 118.3 | Starting accepted reference |
| [0066](iteration-0066/summary.md) | E33-F48-F50-F62b | 45.677 | 167,588 | -15.14% | 99.439 | 0.90 | 160.3 | retain previous accepted run; speed target missed |
| [0067](iteration-0067/summary.md) | E33-F48-F50-F34e | 45.325 | 168,892 | -14.48% | 99.134 | 0.79 | 159.9 | retain previous accepted run; speed target missed |
| [0068](iteration-0068/summary.md) | E33-F48-F50-F62c | 45.705 | 167,485 | -15.20% | 99.935 | 0.83 | 159.4 | retain previous accepted run; speed target missed |

**0066, E33-F48-F50-F62b.** Pack active AdamW parameters and moments by optimizer group and equal per-parameter steps; execute whole-group arithmetic and scatter without changing child alignment. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -15.14% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.90 versus 0.47; summed preparation s: 160.29 versus 118.28; validation s: 31.24 versus 29.63; primer s: 15.29 versus 15.31; worker CPU %: 61.22 versus 53.23; GPU activity %: 66.21 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

**0067, E33-F48-F50-F34e.** Fuse eight-way shift gather and lerp forward into one kernel; rebuild the original gather/lerp autograd backward graph to retain dtype rounding and deterministic reductions. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -14.48% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.79 versus 0.47; summed preparation s: 159.91 versus 118.28; validation s: 31.22 versus 29.63; primer s: 15.27 versus 15.31; worker CPU %: 61.14 versus 53.23; GPU activity %: 62.75 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

**0068, E33-F48-F50-F62c.** Cache four whole active-group AdamW packing buffers by active parameter identity set; cat out reuses device storage while preserving optimizer groups and individual step buckets. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -15.20% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.83 versus 0.47; summed preparation s: 159.44 versus 118.28; validation s: 31.75 versus 29.63; primer s: 15.25 versus 15.31; worker CPU %: 60.70 versus 53.23; GPU activity %: 69.19 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

## Replicated-block decisions and interpretation

The frozen gate uses 5-MAD trimmed per-update effects at looks 2/4/6, Pocock boundary 1.9927 and SD at least 1.01%. Historical full-run speed differences do not decide promotion.

| Iteration | Mechanism on accepted 0042 | Blocks | Mean gain | Lower bound | Decision |
|---|---|---:|---:|---:|---|
| 0066 | Whole active-group AdamW packing F62b | 2 | -0.860% | -2.254% | Reject |
| 0067 | Exact eight-way shift/lerp forward F34e | 6 | +0.229% | -0.588% | Retain unproven option |
| 0068 | Persistent whole-group AdamW packing buffers F62c | 2 | -0.783% | -2.178% | Reject |

0067 extended from +0.449% / -0.963% at two blocks to +0.192% / -0.806% at four, then +0.229% / -0.588% at six. No promotion occurred. Every full candidate and all twenty replicas preserved parameters, held-out inputs and all four losses exactly. Zero allocator retries/OOMs, 3157 MHz median SM clocks and 12,188-12,192 MiB peak reserved were recorded. [Recorded audit](../../diagnostics/r01-round-0066-0068/round-audit.json), [look progression](../../diagnostics/r01-round-0066-0068/frozen-look-progress.json). Best-of paired timing diagnostics do not replace this all-block gate.

Accepted 0042/options unchanged. Preserve prior unproven 0048 F58, 0061 F58e/mode5 and 0064 F58g/mode7; add independent 0067 F34e exact forward as unproven. Modes5/7 remain mutually exclusive. No additive gains are established. Omit rejected F62b/F62c without conflating these whole-group mechanisms with earlier aligned flat-storage 0050. Preserve 0061's single-fast-block caveat and all earlier rejections.

Both AdamW candidates passed five-real-batch loss/transfer/clipping/moment/step/master-child checks over three updates, including None gradients, changing active sets, divergent steps and signed zeros. F34e passed 16 FP32/BF16 synthetic boundary-weight/signed-zero shape cases and five real loss/every-gradient batches. All edits preceded source-compatible checks. Full quiet gates passed at 5.0% / 6.7% / 6.1% CPU with no other Python GPU process. Replicas used the same at-most-10% CPU gate. The known Windows run-log open failure affected 0066's direct post-block report and 0067/0068 initial reports; explicit report-only resumes reused completed training/profile and preserved attempt logs. [Preparation/repairs](../../diagnostics/r01-round-0066-0068/preparation.md).

Continue --iteration 0069 --previous 0068 with original budget 23/index21 complete and exactly two authorized candidates remaining through 0070. Inherit original baseline and cumulative loss ceiling 0.001. Prioritize exact lerp backward without graph reconstruction or substantive F49/F51/F52 with their semantics and prerequisites; uncounted feasibility failures require a concretely changed useful fallback. This worker stops after 0066-0068. Alex/broader validation remain paused; parent handles final chat and shutdown.

## Separate held-out binary losses

| Iteration | Metric | Loss | Delta vs prior accepted | Delta vs original baseline |
|---|---|---:|---:|---:|
| 0066 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0066 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0066 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0066 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0067 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0067 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0067 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0067 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0068 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0068 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0068 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0068 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |

## Checks and provenance

Full recorded parity is exact in 3/3 cases. The recorded acceptance gates and decisions are reused; this report does not change them or substitute parity for broader quality validation.

- 0066: experiment `59f451ce232a9f2cdf195170c8866d7f1dd50c62`, origin `78ad3ecebdd75a203011113dae49f8097156c8e6`, dirty at process start: True; [source archive](../../sources/experiment-sources/c72ac7f38a778f84a1d72ec162c31087ed44cfa31939504ff0be8b24483728e4/source.zip), [protocol](iteration-0066/protocol.json), [comparison](iteration-0066/comparison.json).
- 0066 pre-run background: 5.0% total CPU; 0 other Python GPU processes detected.
- 0067: experiment `95d005f47b9b0b087618e6e6ac6d898286cc225c`, origin `dab8feead8d278233f1972bd24683903a239aea7`, dirty at process start: True; [source archive](../../sources/experiment-sources/acbdb48f818e79de63b0ae56f6f839f6e6fcd1eb1e19970a86fae489bedb2ce1/source.zip), [protocol](iteration-0067/protocol.json), [comparison](iteration-0067/comparison.json).
- 0067 pre-run background: 6.7% total CPU; 0 other Python GPU processes detected.
- 0068: experiment `69fca82591a648c3539d8c71fbb1729b7ef00e64`, origin `dab8feead8d278233f1972bd24683903a239aea7`, dirty at process start: True; [source archive](../../sources/experiment-sources/09cc797eca0ea7ccf262f3830ff01654d648f9d57d0719009d3aa15d5e59e25d/source.zip), [protocol](iteration-0068/protocol.json), [comparison](iteration-0068/comparison.json).
- 0068 pre-run background: 6.1% total CPU; 0 other Python GPU processes detected.

## Pending next steps

- Continue --iteration 0069 --previous 0068: campaign budget 23/index21 complete, exactly two authorized candidates remain through 0070; inherit original baseline and cumulative 0.001 ceiling.
- Accepted 0042 E33-F48-F50/options unchanged. Retain 0048 F58, 0061 F58e mode5, 0064 F58g mode7 and 0067 F34e exact shift lerp forward as unproven options; modes5/7 cannot stack. Omit rejected whole-group F62b/F62c and prior rejected schedules/bundles.
- Prioritize exact shift lerp backward consolidation removing graph reconstruction while preserving original op-boundary rounding and reductions; check synthetic and five real loss/every-gradient batches before freeze.
- Other substantive pending mechanisms: memory-neutral genuinely deferred F49 with main-stream outputs, F51 token work preserving dropout and split weight-gradient order, F52 downstream padding removal preserving skip/residual/bonus/dropout semantics. Retain failures and choose a concretely changed fallback. Alex/broader validation paused; parent controls final chat/shutdown.
