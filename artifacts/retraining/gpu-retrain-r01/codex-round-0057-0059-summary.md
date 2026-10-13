# Codex R01 round: iterations 0057–0059

The accepted preliminary expertium reference is **0042: 197,499 reviews/s**, +0.00% versus the round’s starting accepted reference and +558.22% versus the fixed original baseline. Implementation commit `3f42024b77a2cbac699eb56da89b29620ed99615` was committed before the counted runs; prerequisite checks used the same training source while uncommitted. Experiment revisions and isolated source archives remain separate and authoritative.

Authorization: I want you to follow R01 cycles, spawn each time a new GPT 6.1 Sol medium thread for 3 iterations, until you reach iteration number 70. Go up to 70 and then shutdown the computer.. Explicit count: 12 of 23 iterations completed; 11 remain authorized. Consecutive misses do not end this counted round.

## Full-history results

Training seconds and raw review presentations/s include user transitions; complete process seconds also include setup, discarded primer, checkpoint saving and evaluation. Gains below use each candidate’s own recorded previous accepted reference, which can differ between rows. One full timing per candidate does not establish repeatability; broader validation remains pending.

| Iteration | Bundle | Training seconds | Reviews/s | Gain vs prior accepted | Complete process seconds | Queue wait s | Summed worker preparation s | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [0042](iteration-0042/summary.md) | E33-F48-F50 | 38.759 | 197,499 | — | 90.735 | 0.47 | 118.3 | Starting accepted reference |
| [0057](iteration-0057/summary.md) | E33-F48-F50-F58-F58a | 45.183 | 169,423 | -14.22% | 99.898 | 0.81 | 153.4 | retain previous accepted run; speed target missed |
| [0058](iteration-0058/summary.md) | E33-F48-F50-F58-F58b | 45.270 | 169,097 | -14.38% | 99.305 | 0.79 | 153.4 | retain previous accepted run; speed target missed |
| [0059](iteration-0059/summary.md) | E33-F48-F50-F58-F58c | 45.425 | 168,520 | -14.67% | 99.863 | 0.73 | 154.2 | retain previous accepted run; speed target missed |

**0057, E33-F48-F50-F58-F58a.** Prefetch next recomputation scalar inputs and each row v before the current shuffle tree; retain exact arithmetic and inherited scalar F58, omit rejected slices. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -14.22% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.81 versus 0.47; summed preparation s: 153.35 versus 118.28; validation s: 31.88 versus 29.63; primer s: 15.33 versus 15.31; worker CPU %: 59.67 versus 53.23; GPU activity %: 69.75 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

**0058, E33-F48-F50-F58-F58b.** Pipeline per-row backward gradient loads ahead of current recurrence work, preserving arithmetic and reduction order. Replace the previous pipeline mode; retain inherited scalar F58 and omit rejected slices. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -14.38% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.79 versus 0.47; summed preparation s: 153.38 versus 118.28; validation s: 31.64 versus 29.63; primer s: 15.19 versus 15.31; worker CPU %: 59.76 versus 53.23; GPU activity %: 64.21 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

**0059, E33-F48-F50-F58-F58c.** Pipeline forward recurrence scalar loads ahead of current recurrence work, preserving arithmetic and reduction order. Replace the previous pipeline mode; retain inherited scalar F58 and omit rejected slices. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -14.67% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.73 versus 0.47; summed preparation s: 154.21 versus 118.28; validation s: 31.94 versus 29.63; primer s: 15.35 versus 15.31; worker CPU %: 59.78 versus 53.23; GPU activity %: 64.99 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

## Replicated-block decisions and interpretation

The frozen gate uses 5-MAD trimmed per-update effects, looks 2/4/6, Pocock boundary 1.9927 and SD at least 1.01%. Historical full-run deltas above do not decide promotion.

| Iteration | Changed F58 mechanism | Blocks | Mean gain | Lower bound | Decision | Mean reference reviews/s | Mean candidate reviews/s |
|---|---|---:|---:|---:|---|---:|---:|
| 0057 | Recomputation inputs/per-row v | 4 | -0.236% | -1.230% | Reject | 169,871 | 169,652 |
| 0058 | Per-row backward gradients | 2 | -0.424% | -1.824% | Reject | 170,119 | 169,658 |
| 0059 | Forward scalars | 2 | -0.580% | -1.977% | Reject | 170,645 | 169,555 |

0057 extended from +0.150% / -1.258% at two blocks to a negative mean at four; six blocks were not required. 0058/0059 rejected at their first look. All sixteen replicas and three full candidates retain exact recorded parity and all four held-out losses equal accepted 0042, within the inherited cumulative 0.001 ceiling. Best-of paired wall diagnostics are -0.376%, -0.475%, -0.665%; these do not replace the gate. All runs show zero allocator retries/OOMs, 3157 MHz median SM clocks and 12,188 MiB peak reserved. No concrete anomaly warrants an identical repeat. [Audit](../../diagnostics/r01-round-0057-0059/round-audit.json).

[Preparation, compatible checks and failed attempts](../../diagnostics/r01-round-0057-0059/preparation.md). Each mechanism passed 216 synthetic BF16/FP32 cases and five real loss/every-gradient batches before freezing. Initial checker dict.get wiring and a reused failed output-directory attempt were repaired before counted runs. Transient Windows run-log errors in 0058/0059 were repaired by report-only resumes, without training repeats. Standalone full launches passed the at-most-10% quiet gate at 6.3%, 5.7%, 6.4% CPU; every replica used the same gate.

Each mode is an alternative on inherited F58, so these bundle results do not isolate the new mechanism from the original unproven scalar prefetch. Accepted 0042 E33-F48-F50 and only prior 0048 F58 recurrence_prefetch=true remain unchanged. Omit recurrence_load_pipeline modes 1/2/3 and all other rejected slices. P3b twin A/B attribution is still pending; use it before further blind F58 stacking. Forward per-row v prefetch, whole-group AdamW, exact eight-way lerp, genuinely deferred overlap and substantive F51/F52 work remain pending and unmeasured.

Continue --iteration 0060 --previous 0059: original campaign budget 23/index 12 complete, 11 remaining through 0070, loss ceiling 0.001. This worker completed exactly 0057-0059. Alex/broader validation stay paused; parent controls fresh chats and final shutdown.

## Separate held-out binary losses

| Iteration | Metric | Loss | Delta vs prior accepted | Delta vs original baseline |
|---|---|---:|---:|---:|
| 0057 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0057 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0057 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0057 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0058 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0058 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0058 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0058 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0059 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0059 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0059 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0059 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |

## Checks and provenance

Full recorded parity is exact in 3/3 cases. The recorded acceptance gates and decisions are reused; this report does not change them or substitute parity for broader quality validation.

- 0057: experiment `1d109ff0c08b1d96489eba617ccc18dd75855970`, origin `3f42024b77a2cbac699eb56da89b29620ed99615`, dirty at process start: True; [source archive](../../sources/experiment-sources/78526172eda17b1f52fc0f7ec23604a898a49ad765ef97a13011a563fa36afa1/source.zip), [protocol](iteration-0057/protocol.json), [comparison](iteration-0057/comparison.json).
- 0057 pre-run background: 6.3% total CPU; 0 other Python GPU processes detected.
- 0058: experiment `ec88c3128d3bcc317ee8bf7731722461a466cd84`, origin `3f42024b77a2cbac699eb56da89b29620ed99615`, dirty at process start: True; [source archive](../../sources/experiment-sources/f68da6c77cc60804f772f5bb7a2ab0fe2b24fad8b533d8911a36b4692e5d5062/source.zip), [protocol](iteration-0058/protocol.json), [comparison](iteration-0058/comparison.json).
- 0058 pre-run background: 5.7% total CPU; 0 other Python GPU processes detected.
- 0059: experiment `28c9eb5864801855bd61614ff66b4939872e4f47`, origin `3f42024b77a2cbac699eb56da89b29620ed99615`, dirty at process start: True; [source archive](../../sources/experiment-sources/24708aed07b2cac960010cfc0efd2850b3eb9141fa3bd84b9c84cfc3ad733b2a/source.zip), [protocol](iteration-0059/protocol.json), [comparison](iteration-0059/comparison.json).
- 0059 pre-run background: 6.4% total CPU; 0 other Python GPU processes detected.

## Pending next steps

- Continue --iteration 0060 --previous 0059 without resetting authorization/baseline: budget 23, index 12 complete, 11 remaining through 0070, cumulative loss ceiling 0.001.
- Accepted 0042 E33-F48-F50 unchanged. Retain only 0048 F58 recurrence_prefetch=true as unproven; omit all rejected 0049-0059 slices including recurrence_load_pipeline modes 1/2/3.
- Prepare P3b twin A/B attribution before further F58 stacking, then a changed pending mechanism: forward per-row v prefetch with five real plus synthetic checks, or whole-group F62 AdamW preserving None-gradient, steps and signed zeros.
- Pending exact eight-way shift/lerp, independent deferred F49 overlap and substantive F51/F52 work must preserve dropout calls and split weight-gradient order. Follow extended written order and quiet frozen blocks. Alex/broader validation paused; parent manages chats/shutdown.
