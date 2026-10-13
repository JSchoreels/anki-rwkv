# Codex R01 round: iterations 0063–0065

The accepted preliminary expertium reference is **0042: 197,499 reviews/s**, +0.00% versus the round’s starting accepted reference and +558.22% versus the fixed original baseline. Implementation commit `ea776234cf7703b1305c1159d49603b53bf51a36` was made after the prerequisite checks and before all three counted experiments; no training-source edits followed the checks.

Authorization: I want you to follow R01 cycles, spawn each time a new GPT 6.1 Sol medium thread for 3 iterations, until you reach iteration number 70. Go up to 70 and then shutdown the computer.. Explicit count: 18 of 23 iterations completed; 5 remain authorized. Consecutive misses do not end this counted round.

## Full-history results

Training seconds and raw review presentations/s include user transitions; complete process seconds also include setup, discarded primer, checkpoint saving and evaluation. Gains below use each candidate’s own recorded previous accepted reference, which can differ between rows. One full timing per candidate does not establish repeatability; broader validation remains pending.

| Iteration | Bundle | Training seconds | Reviews/s | Gain vs prior accepted | Complete process seconds | Queue wait s | Summed worker preparation s | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [0042](iteration-0042/summary.md) | E33-F48-F50 | 38.759 | 197,499 | — | 90.735 | 0.47 | 118.3 | Starting accepted reference |
| [0063](iteration-0063/summary.md) | E33-F48-F50-F58-F58e | 45.173 | 169,458 | -14.20% | 99.525 | 0.86 | 159.2 | retain previous accepted run; speed target missed |
| [0064](iteration-0064/summary.md) | E33-F48-F50-F58g | 45.247 | 169,181 | -14.34% | 99.467 | 0.83 | 161.2 | retain previous accepted run; speed target missed |
| [0065](iteration-0065/summary.md) | E33-F48-F50-F58h | 45.363 | 168,749 | -14.56% | 99.150 | 0.84 | 159.3 | retain previous accepted run; speed target missed |

**0063, E33-F48-F50-F58-F58e.** Combine retained backward scalar F58 and forward scalar/value F58e pipelines, preserving exact arithmetic and reduction order. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -14.20% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.86 versus 0.47; summed preparation s: 159.17 versus 118.28; validation s: 31.74 versus 29.63; primer s: 15.36 versus 15.31; worker CPU %: 60.89 versus 53.23; GPU activity %: 66.57 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

**0064, E33-F48-F50-F58g.** Select forward scalar/value prefetch only for histories at least 32 steps, preserving baseline elsewhere; test sequential latency hiding versus live-set pressure. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -14.34% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.83 versus 0.47; summed preparation s: 161.17 versus 118.28; validation s: 31.68 versus 29.63; primer s: 15.31 versus 15.31; worker CPU %: 61.32 versus 53.23; GPU activity %: 68.31 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

**0065, E33-F48-F50-F58h.** Select forward scalar/value prefetch only for histories at most 16 steps, preserving baseline elsewhere; test broad short-history scheduling separately from long chains. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -14.56% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.84 versus 0.47; summed preparation s: 159.33 versus 118.28; validation s: 31.26 versus 29.63; primer s: 15.33 versus 15.31; worker CPU %: 60.73 versus 53.23; GPU activity %: 65.67 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

## Replicated-block decisions and interpretation

The frozen gate uses 5-MAD trimmed per-update effects at looks 2/4/6, Pocock boundary 1.9927 and SD at least 1.01%. Historical full-run deltas do not decide promotion.

| Iteration | Mechanism on accepted 0042 | Blocks | Mean gain | Lower bound | Decision |
|---|---|---:|---:|---:|---|
| 0063 | Combined original F58 backward scalar and F58e forward scalar/value prefetch | 2 | -0.015% | -1.421% | Reject bundle |
| 0064 | F58g forward scalar/value prefetch only for T >= 32 | 6 | +0.017% | -0.797% | Retain unproven alternative |
| 0065 | F58h forward scalar/value prefetch only for T <= 16 | 2 | -0.296% | -1.698% | Reject |

0064 extended from +0.118% / -1.289% at two blocks to +0.092% / -0.905% at four, then +0.017% / -0.797% at six. No promotion occurred. Full candidates and all twenty replicas preserve final parameters, held-out inputs and four binary losses exactly against 0042, within the inherited cumulative 0.001 ceiling. Zero allocator retries/OOMs, 3157 MHz median SM clocks and 12,188 MiB reserved were recorded for every replica. Best-of paired wall diagnostics are -0.49%, -0.08%, -1.00%; these diagnostics select fast cases and do not replace the frozen all-block decision. [Audit](../../diagnostics/r01-round-0063-0065/round-audit.json), [look progression](../../diagnostics/r01-round-0063-0065/frozen-look-progress.json).

Original 0048 F58 and 0061 F58e remain unproven individually; their checked 0063 combination rejected. 0061's earlier +2.714% mean remains dominated by one +15.964% fast-state block with faster preparation/validation. New 0064 F58g joins the unproven alternatives; modes 5 and 7 are mutually exclusive schedules, not additive components. Omit rejected mode 8 and the combined 0063 bundle without conflating bundle rejection with independent component effects.

Each candidate passed 216 BF16/FP32 synthetic skip/edge cases and five real loss/every-gradient batches on final source. All full quiet gates passed at 7.0% / 9.1% / 8.6% background CPU with no other Python GPU process detected. Replicas used the same at-most-10% CPU gate; 0065 delayed its first candidate at 10.7% until quiet. 0064/0065 reporting hit the historical Windows Errno 22 opening the run log; explicit report-only resumes preserved failed attempts and reused completed training/profile. [Preparation](../../diagnostics/r01-round-0063-0065/preparation.md).

Continue --iteration 0066 --previous 0065: original budget 23/index 18 complete, five remain through 0070, original baseline/ceiling unchanged. Prioritize substantive whole-group F62, exact eight-way lerp, memory-neutral deferred F49 or larger F51/F52 with their semantics/checks. This worker stops exactly after 0063-0065; parent controls chats/shutdown. Alex/broader validation paused.

## Separate held-out binary losses

| Iteration | Metric | Loss | Delta vs prior accepted | Delta vs original baseline |
|---|---|---:|---:|---:|
| 0063 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0063 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0063 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0063 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0064 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0064 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0064 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0064 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0065 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0065 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0065 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0065 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |

## Checks and provenance

Full recorded parity is exact in 3/3 cases. The recorded acceptance gates and decisions are reused; this report does not change them or substitute parity for broader quality validation.

- 0063: experiment `a6c343efcee4d1134440de899152aa35547d0c37`, origin `ea776234cf7703b1305c1159d49603b53bf51a36`, dirty at process start: True; [source archive](../../sources/experiment-sources/481c7f4fab9b6420ac3896e6081000ea1c3740cf50669bfae5b1d4b8cbfef36f/source.zip), [protocol](iteration-0063/protocol.json), [comparison](iteration-0063/comparison.json).
- 0063 pre-run background: 7.0% total CPU; 0 other Python GPU processes detected.
- 0064: experiment `55ae4ae59b2b7adc7afef1054c2085800f8aedcb`, origin `ea776234cf7703b1305c1159d49603b53bf51a36`, dirty at process start: True; [source archive](../../sources/experiment-sources/194fb68e302fa63ba351259f9c80895f0374f03f7b0bc9f9cc0e392e4e1019e9/source.zip), [protocol](iteration-0064/protocol.json), [comparison](iteration-0064/comparison.json).
- 0064 pre-run background: 9.1% total CPU; 0 other Python GPU processes detected.
- 0065: experiment `d1125376ce231c262b9eb7f078ba79a7a592a000`, origin `ea776234cf7703b1305c1159d49603b53bf51a36`, dirty at process start: True; [source archive](../../sources/experiment-sources/807ea0e95471fdf903bb9c405369a2acacae409a07b427fe869c72e9c2275c9c/source.zip), [protocol](iteration-0065/protocol.json), [comparison](iteration-0065/comparison.json).
- 0065 pre-run background: 8.6% total CPU; 0 other Python GPU processes detected.

## Pending next steps

- Continue --iteration 0066 --previous 0065: original campaign budget 23/index 18 complete, five remain through 0070; original baseline and cumulative ceiling 0.001 inherited.
- Accepted 0042 E33-F48-F50/options unchanged. Retain original 0048 F58, 0061 F58e and 0064 F58g as unproven alternatives; omit rejected 0063 combination and mode 8 plus prior rejected slices.
- Modes 5 and 7 are alternative forward schedules, not additive components. Preserve 0061 single-fast-block caveat; do not repeat rejected 0063 or count no-op consolidation.
- Prepare substantive whole-group F62 AdamW with None gradients/per-parameter steps/signed zeros, exact eight-way lerp with original dtype rounding/reduction order, memory-neutral deferred F49, or substantive F51/F52 retaining dropout and split weight-gradient order. Alex/broader validation paused; parent controls chats/shutdown.
