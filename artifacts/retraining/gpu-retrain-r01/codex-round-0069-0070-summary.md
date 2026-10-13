# Codex R01 round: iterations 0069–0070

The accepted preliminary expertium reference is **0042: 197,499 reviews/s**, +0.00% versus the round’s starting accepted reference and +558.22% versus the fixed original baseline. Implementation commit `a9ed12390b5d49edc17367726cdf3bb9ab64aa5a` was made after the experiments.

Authorization: I want you to follow R01 cycles, spawn each time a new GPT 6.1 Sol medium thread for 3 iterations, until you reach iteration number 70. Go up to 70 and then shutdown the computer.. The selected iteration budget is complete; another round requires a new human request.

## Full-history results

Training seconds and raw review presentations/s include user transitions; complete process seconds also include setup, discarded primer, checkpoint saving and evaluation. Gains below use each candidate’s own recorded previous accepted reference, which can differ between rows. One full timing per candidate does not establish repeatability; broader validation remains pending.

| Iteration | Bundle | Training seconds | Reviews/s | Gain vs prior accepted | Complete process seconds | Queue wait s | Summed worker preparation s | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [0042](iteration-0042/summary.md) | E33-F48-F50 | 38.759 | 197,499 | — | 90.735 | 0.47 | 118.3 | Starting accepted reference |
| [0069](iteration-0069/summary.md) | E33-F48-F50-F34f | 45.460 | 168,390 | -14.74% | 99.649 | 0.82 | 160.6 | retain previous accepted run; speed target missed |
| [0070](iteration-0070/summary.md) | E33-F48-F50-F34g | 45.496 | 168,257 | -14.81% | 99.810 | 0.82 | 161.8 | retain previous accepted run; speed target missed |

**0069, E33-F48-F50-F34f.** Replace reconstructed autograd lerp/gather backward with explicit original derivative operations and original reductions; skip singleton reductions to preserve signed zeros. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -14.74% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.82 versus 0.47; summed preparation s: 160.60 versus 118.28; validation s: 31.66 versus 29.63; primer s: 15.31 versus 15.31; worker CPU %: 61.35 versus 53.23; GPU activity %: 65.49 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

**0070, E33-F48-F50-F34g.** Fuse shift gather and three lerp derivative elementwise outputs into one kernel, rounding complement/difference/products at original dtype boundaries and preserving original PyTorch reductions and deterministic scatter/add. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -14.81% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.82 versus 0.47; summed preparation s: 161.81 versus 118.28; validation s: 31.68 versus 29.63; primer s: 15.16 versus 15.31; worker CPU %: 61.47 versus 53.23; GPU activity %: 67.96 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

## Replicated-block decisions and final campaign state

The frozen promotion statistic uses 5-MAD trimmed per-update effects, looks 2/4/6, Pocock boundary 1.9927 and SD at least 1.01%. Historical single-run throughput comparisons above are descriptive and do not decide promotion.

| Iteration | Mechanism on accepted 0042 | Blocks | Mean gain | Lower bound | Decision |
|---|---|---:|---:|---:|---|
| 0069 | Explicit original shift/lerp backward F34f | 2 | -0.075% | -1.480% | Reject |
| 0070 | Fused rounded derivative elements F34g | 6 | +0.137% | -0.678% | Retain unproven option |

0070 extended from +0.188% / -1.221% at two blocks to +0.140% / -0.858% at four, then +0.137% / -0.678% at six. Every full candidate and all sixteen reference/candidate replicas preserved initial/final parameters, held-out inputs and all four losses exactly. All eighteen full-history cases passed the at-most-10% CPU gate (observed 3.7%-8.8%) and recorded zero allocator retries/OOMs, 3157 MHz median SM clocks and 12,188 MiB peak reserved. [Audit](../../diagnostics/r01-round-0069-0070/round-audit.json), [look progression](../../diagnostics/r01-round-0069-0070/frozen-look-progress.json).

F34f removes graph reconstruction using original derivative arithmetic and reductions; its first synthetic attempt found four singleton B=T=1 weight-gradient signed-zero failures. Skipping singleton sum-to-size reductions repaired the issue, preserving the failed evidence. F34g additionally fuses gather and three elementwise derivative tensors, explicitly rounding complement/difference/product boundaries and retaining PyTorch reductions plus deterministic scatter/final addition. Both passed sixteen FP32/BF16 synthetic edge cases and five real loss/every-gradient batches before their respective freezes; all relevant source edits preceded compatible checks. Smoke checks matched accepted 0042. The known Windows Errno22 run-log open failure affected 0070's initial report; report-only resume reused completed training/profile and retained both attempts. [Preparation and checks](../../diagnostics/r01-round-0069-0070/preparation.md).

Campaign 23/23 is complete at 0070, with zero authorized candidates remaining. Original baseline and cumulative loss ceiling 0.001 were inherited without reset. Accepted 0042 E33-F48-F50/options remain unchanged. Retain unproven 0048 F58, 0061 F58e/mode5, 0064 F58g/mode7, 0067 F34e and 0070 F34g independently; modes5/7 and the lerp paths are alternatives, with no additive gain established. Preserve 0061's single-fast-block caveat and all earlier rejections; add rejected 0069 F34f. No 0071 or broader validation was prepared or run. Parent verifies the records and performs the authorized shutdown; this worker does not shut down or message other chats.

## Separate held-out binary losses

| Iteration | Metric | Loss | Delta vs prior accepted | Delta vs original baseline |
|---|---|---:|---:|---:|
| 0069 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0069 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0069 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0069 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0070 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0070 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0070 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0070 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |

## Checks and provenance

Full recorded parity is exact in 2/2 cases. The recorded acceptance gates and decisions are reused; this report does not change them or substitute parity for broader quality validation.

- 0069: experiment `ccca98c8906717fd54f39baf9a38edb5f6936592`, origin `163c7b7d8c5d75542d0e38f666dce4b362bff7e3`, dirty at process start: True; [source archive](../../sources/experiment-sources/9207a1c95ce85f4246e53008c5202aa438a6cf26fa33237d1e87d67e66f1a245/source.zip), [protocol](iteration-0069/protocol.json), [comparison](iteration-0069/comparison.json).
- 0069 pre-run background: 5.8% total CPU; 0 other Python GPU processes detected.
- 0070: experiment `fd3129c29801cd3ca43fc7790b28a9ee4d934174`, origin `163c7b7d8c5d75542d0e38f666dce4b362bff7e3`, dirty at process start: True; [source archive](../../sources/experiment-sources/44eb4d7efd50a6ff745d724109a2b54ac1efb6254012b526b13b236d630a8aa7/source.zip), [protocol](iteration-0070/protocol.json), [comparison](iteration-0070/comparison.json).
- 0070 pre-run background: 6.1% total CPU; 0 other Python GPU processes detected.

Post-completion [process audit](../../diagnostics/r01-round-0069-0070/final-process-audit.json) found zero owned R01 Python processes and no NVIDIA compute applications. No own GPU job remains.

## Pending next steps

- Campaign complete: 23/23 candidates through 0070; zero authorized candidates remain. Do not prepare or run 0071. Parent verifies saved records and handles authorized shutdown; this worker does not shut down or message other chats.
- Accepted 0042 E33-F48-F50/options unchanged. Retain unproven 0048 F58, 0061 F58e/mode5, 0064 F58g/mode7, 0067 F34e and 0070 F34g independently; modes5/7 and lerp paths are alternatives, gains not additive. Omit rejected 0069 F34f and all prior rejected schedules/bundles.
- Untested future mechanisms require new authorization: genuinely deferred memory-neutral F49 with main-stream outputs, substantive F51 token batching preserving dropout/split weight-gradient order, F52 downstream padding removal preserving skip/residual/bonus/dropout semantics, exact F63 dropout prerequisites. Alex/broader validation remain paused.
