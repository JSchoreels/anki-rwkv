# Codex R01 round: iterations 0060–0062

The accepted preliminary expertium reference is **0042: 197,499 reviews/s**, +0.00% versus the round’s starting accepted reference and +558.22% versus the fixed original baseline. Implementation commit `f127c14f93da8d9ea7f82426fc8f891cb0e65b27` was made after 0060 protocol preparation and before 0061/0062 preparation; prerequisite checks used the final uncommitted source. Experiment revisions and isolated source archives remain separate and authoritative.

Authorization: I want you to follow R01 cycles, spawn each time a new GPT 6.1 Sol medium thread for 3 iterations, until you reach iteration number 70. Go up to 70 and then shutdown the computer.. Explicit count: 15 of 23 iterations completed; 8 remain authorized. Consecutive misses do not end this counted round.

## Full-history results

Training seconds and raw review presentations/s include user transitions; complete process seconds also include setup, discarded primer, checkpoint saving and evaluation. Gains below use each candidate’s own recorded previous accepted reference, which can differ between rows. One full timing per candidate does not establish repeatability; broader validation remains pending.

| Iteration | Bundle | Training seconds | Reviews/s | Gain vs prior accepted | Complete process seconds | Queue wait s | Summed worker preparation s | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [0042](iteration-0042/summary.md) | E33-F48-F50 | 38.759 | 197,499 | — | 90.735 | 0.47 | 118.3 | Starting accepted reference |
| [0060](iteration-0060/summary.md) | E33-F48-F50-F58d | 45.130 | 169,622 | -14.12% | 99.208 | 0.80 | 154.4 | retain previous accepted run; speed target missed |
| [0061](iteration-0061/summary.md) | E33-F48-F50-F58e | 44.836 | 170,734 | -13.55% | 99.082 | 0.80 | 154.9 | investigate retuning freed resources |
| [0062](iteration-0062/summary.md) | E33-F48-F50-F58f | 45.328 | 168,881 | -14.49% | 99.778 | 0.87 | 159.9 | retain previous accepted run; speed target missed |

**0060, E33-F48-F50-F58d.** Pipeline forward per-row value loads before current shuffle work, preserving arithmetic and reduction order. Test independently on accepted 0042; omit unproven scalar F58 after inconclusive P3b attribution. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -14.12% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.80 versus 0.47; summed preparation s: 154.35 versus 118.28; validation s: 31.68 versus 29.63; primer s: 15.30 versus 15.31; worker CPU %: 59.75 versus 53.23; GPU activity %: 67.96 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

**0061, E33-F48-F50-F58e.** Pipeline forward scalar and per-row value loads before current shuffle work, preserving arithmetic and reduction order. Test independently on accepted 0042; omit unproven scalar F58 after inconclusive P3b attribution. Recorded decision: investigate retuning freed resources; exact full recorded parity: True. Training throughput changed -13.55% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.80 versus 0.47; summed preparation s: 154.94 versus 118.28; validation s: 31.74 versus 29.63; primer s: 15.29 versus 15.31; worker CPU %: 59.78 versus 53.23; GPU activity %: 58.83 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

**0062, E33-F48-F50-F58f.** Pipeline recomputation per-row value loads without scalar prefetch before current shuffle work, preserving arithmetic and reduction order. Test independently on accepted 0042; omit unproven scalar F58 after inconclusive P3b attribution. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -14.49% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.87 versus 0.47; summed preparation s: 159.92 versus 118.28; validation s: 31.72 versus 29.63; primer s: 15.32 versus 15.31; worker CPU %: 61.01 versus 53.23; GPU activity %: 69.93 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

## Replicated-block decisions and interpretation

The frozen gate uses 5-MAD trimmed per-update effects, looks 2/4/6, Pocock boundary 1.9927 and SD at least 1.01%. Historical full-run deltas do not decide promotion.

| Iteration | Independent mechanism on 0042 | Blocks | Mean gain | Lower bound | Decision | Mean reference reviews/s | Mean candidate reviews/s |
|---|---|---:|---:|---:|---|---:|---:|
| 0060 | Forward row-value prefetch | 2 | −0.199% | −1.602% | Reject | 169,051 | 169,153 |
| 0061 | Forward scalar-plus-row-value prefetch | 6 | +2.714% | −2.143% | Unproven stack candidate | 169,168 | 173,955 |
| 0062 | Recomputation row-value-only prefetch | 2 | −0.064% | −1.469% | Reject | 168,897 | 168,756 |

0061 extended from +0.421%/−0.991% at two blocks to +0.360%/−0.640% at four. The sixth-look mean is dominated by block 5's +15.964% effect; the other five effects average +0.252% (descriptive sensitivity only, not a replacement gate). In that pair, candidate r10 versus control r9 training was 38.962 versus 45.210 s, preparation 116.972 versus 154.739 s and validation 28.893 versus 31.838 s. This broad speed shift makes component attribution uncertain; recorded quiet gates passed, clocks remained 3157 MHz and memory remained 12,188 MiB reserved. Preserve all blocks and the frozen unproven decision. Best-of paired diagnostics are +0.23%, +10.91%, −0.13%; the 0061 diagnostic selects the fast-state candidate and cannot establish repeatability. [Audit](../../diagnostics/r01-round-0060-0062/round-audit.json).

All three full candidates and twenty replicas retain exact final parameters, held-out inputs and four binary losses against 0042, within the inherited cumulative 0.001 ceiling. All replicas have zero allocator retries/OOMs. Every new mode passed 216 BF16/FP32 synthetic cases and five real loss/every-gradient batches on final source before freezing. Full quiet gates passed at 5.2%, 6.6%, 6.5% CPU; replicas used the same at-most-10% gate. 0061/0062 encountered transient Windows run-log file-open errors; report-only resumes reused completed training/profile and preserved failures. [Preparation and checks](../../diagnostics/r01-round-0060-0062/preparation.md).

P3b attribution is now implemented: twenty alternating AB/BA real-batch pairs restore CPU/CUDA RNG after warmup and compare every gradient/loss. Original F58 mean −0.226% with descriptive interval [−1.350%, +0.899%] remains inconclusive. These diagnostics motivated standalone F58d/e/f comparisons without blindly inheriting F58. Original 0048 F58 remains unproven; 0061 F58e/mode 5 joins it as a separately checked unproven candidate. Neither a combined stack nor independent broader validation is established. Omit rejected modes 1/2/3/4/6 and other rejected slices.

Continue --iteration 0063 --previous 0062: original budget 23/index 15 complete, 8 remaining through 0070, original baseline and cumulative ceiling 0.001 unchanged. Whole-group F62 and exact eight-way lerp remain deferred pending mechanisms, with deferred memory-neutral F49 and substantive F51/F52. A future F58+F58e combination needs compatible synthetic and five-real-batch checks before counting. This worker stops exactly after 0060–0062; parent controls chats/shutdown. Alex/broader validation paused.

## Separate held-out binary losses

| Iteration | Metric | Loss | Delta vs prior accepted | Delta vs original baseline |
|---|---|---:|---:|---:|
| 0060 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0060 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0060 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0060 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0061 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0061 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0061 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0061 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0062 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0062 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0062 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0062 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |

## Checks and provenance

Full recorded parity is exact in 3/3 cases. The recorded acceptance gates and decisions are reused; this report does not change them or substitute parity for broader quality validation.

- 0060: experiment `248381d7bbd79bae2ce942ddeb5c8d87ccbbe5bd`, origin `7ddaed78b82d31907a7169bcd56049e3f47379f0`, dirty at process start: True; [source archive](../../sources/experiment-sources/9394eff8d5a439604e8b6d79baffae9c2b74caa9265285ca83060abb17ddad1f/source.zip), [protocol](iteration-0060/protocol.json), [comparison](iteration-0060/comparison.json).
- 0060 pre-run background: 5.2% total CPU; 0 other Python GPU processes detected.
- 0061: experiment `76ca4383baa08179137c6895f5c250d51f5eaf8a`, origin `f127c14f93da8d9ea7f82426fc8f891cb0e65b27`, dirty at process start: True; [source archive](../../sources/experiment-sources/5ca7ec40a86b64bc3a284c63910e6d7caf25d9f0de32dbadece1aeeaefa5c2ca/source.zip), [protocol](iteration-0061/protocol.json), [comparison](iteration-0061/comparison.json).
- 0061 pre-run background: 6.6% total CPU; 0 other Python GPU processes detected.
- 0062: experiment `43b6c2a2e426d2d4c3db6df14a573b3797e808d6`, origin `f127c14f93da8d9ea7f82426fc8f891cb0e65b27`, dirty at process start: True; [source archive](../../sources/experiment-sources/dfc4e5aa3d87fd13a6ef6432404350f001f0f144fe302d744b8eed3c23ff5df6/source.zip), [protocol](iteration-0062/protocol.json), [comparison](iteration-0062/comparison.json).
- 0062 pre-run background: 6.5% total CPU; 0 other Python GPU processes detected.

## Pending next steps

- Continue --iteration 0063 --previous 0062: original campaign budget 23/index 15 complete, 8 remaining through 0070; inherited original baseline and cumulative loss ceiling 0.001.
- Accepted 0042 E33-F48-F50/options unchanged. Retain original 0048 F58 and new 0061 F58e mode 5 as unproven; omit rejected modes 1/2/3/4/6 and all other rejected components.
- 0061 mean is dominated by one +15.964% block, with faster validation too; review state audit before combining. A new combined F58 plus F58e needs compatible synthetic and five-real-batch checks; no identical counted replay.
- Prepare whole-group F62 AdamW preserving None gradients, per-parameter step counters and signed zeros, or exact eight-way shift/lerp with original dtype rounding/reduction order. Deferred memory-neutral F49 and substantive F51/F52 preserving dropout and split weight-gradient order remain pending. Alex/broader validation paused; parent controls chats/shutdown.
