# Codex R01 round: iterations 0071–0073

Stopped at the human request after concluding the active 0073 iteration: **3 of 10 full cycles complete, 7 unconsumed**. Accepted 0072 E33-F48-F50-F65b improves current matched controls by **+1.81426%, lower bound +0.38273%**; the historical peak was not recovered. Its options are accepted 0042 plus `native_wkv=true`; native GroupNorm remains off and is retained as an exact, unproven option. No 0074/0075 protocol, fresh GPU check, new agent or administrator request will start without explicit resumption. The ten-count campaign, 0.001 loss ceiling and inherited replicated-block Pocock rule remain unchanged.

The historical best 197,499.49 reviews/s and current matched control 166,338.05 reviews/s used identical accepted 0042 options/configuration, 7,654,975 reviews, 167 updates, 80 training/20 evaluation users, initial/final parameter hashes, evaluation-input hash and complete held-out evaluation results. Current training took 46.02059 s versus 38.75947 s, a 7.26112 s gap. Source and machine conditions evolved; the cause is unresolved. Queue/GC changes explain only a small part, preparation overlaps, historical clock/thread evidence is incomplete, and added timing events remain a hypothesis. Native WKV's +1.81426% matched gain is local to current controls; it does not establish recovery of the old peak. Read the [Claude handoff](../../diagnostics/r01-native-toolchain/baseline-drift/claude-handoff.md), [offline audit](../../diagnostics/r01-native-toolchain/baseline-drift/summary.md) and [toolchain/counter evidence](../../diagnostics/r01-native-toolchain/summary.md).

| Cycle | Exactness and unchanged losses | Matched blocks | Mean gain / lower bound | Final interpretation |
|---|---|---:|---:|---|
| 0071 F60 deferred evaluation metrics | Full and 4 replicas exact; all 4 held-out losses equal | 2 | −0.00166% / −1.40766% | No training-speed promotion; two observed evaluation savings 0.369 s / 0.154 s are small and inconclusive |
| 0072 F65b native WKV | 72 synthetic and 5 real checks exact; full and 4 replicas exact; all 4 losses equal | 2 | +1.81426% / +0.38273% | Promoted preliminary against current matched controls |
| 0073 F65c native GroupNorm | 48 synthetic and 5 real checks exact; full and 12 replicas exact; all 4 losses equal | 6 | +0.96950% / −0.21613% | Final stack candidate; speed benefit unproven, not promoted |

0074 native shift and 0075 native padded-row source/fixture artifacts are drafts only: they were not built, checked, frozen or counted. F70 bounded timing-event reuse remains pending, preserving host-wall estimator, paired acceptance and GPU-span semantics. F59/F67/F69 also remain pending; no evaluation was skipped and no broader validation was authorized. The successful counter capture is B37/H1/T256 with 0.15 waves; the dense production-shape bottleneck remains pending.

Accepted 0072's single full case recorded 169,934 reviews/s. The full-history table below compares single recorded cases with their inherited prior-reference records, including historical 0042; those unmatched raw percentages are descriptive and are not the promotion estimator. Promotion used contemporaneous matched replicate controls with the same accepted options and unchanged Pocock gate. Implementation commit `7dfa19dbe3627971543d9c8204d4bba805ef7421` was made after the experiments.

Authorization: Ten full R01 training cycles. Explicit count: 3 of 10 iterations completed; 7 remain authorized. Consecutive misses do not end this counted round.

## Full-history results

Training seconds and raw review presentations/s include user transitions; complete process seconds also include setup, discarded primer, checkpoint saving and evaluation. Raw changes below use each candidate’s recorded previous accepted reference, which can differ between rows and dates. Inherited 0042 historical rows do not replace contemporaneous replicated controls. One full timing per candidate does not establish repeatability; broader validation remains pending.

| Iteration | Bundle | Training seconds | Reviews/s | Raw change vs prior accepted (unmatched) | Complete process seconds | Queue wait s | Summed worker preparation s | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [0042](iteration-0042/summary.md) | E33-F48-F50 | 38.759 | 197,499 | — | 90.735 | 0.47 | 118.3 | Starting accepted reference |
| [0071](iteration-0071/summary.md) | E33-F48-F50-F60 | 46.552 | 164,441 | -16.74% | 102.029 | 0.73 | 162.4 | retain previous accepted run; speed target missed |
| [0072](iteration-0072/summary.md) | E33-F48-F50-F65b | 45.047 | 169,934 | -13.96% | 99.676 | 0.84 | 161.1 | retain main improvement; preliminary only |
| [0073](iteration-0073/summary.md) | E33-F48-F50-F65b-F65c | 45.152 | 169,538 | -0.23% | 100.231 | 0.86 | 162.2 | retain previous accepted run; speed target missed |

**0071, E33-F48-F50-F60.** Read the original per-batch held-out metric scalars once per user, preserving reductions, counts, finite checks and Python accumulation order. This measures scalable evaluation and process time; training speed estimator and promotion gates stay unchanged. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -16.74% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.73 versus 0.47; summed preparation s: 162.36 versus 118.28; validation s: 33.16 versus 29.63; primer s: 15.31 versus 15.31; worker CPU %: 60.48 versus 53.23; GPU activity %: 65.77 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

**0072, E33-F48-F50-F65b.** Launch the unchanged accepted four-row NVRTC WKV kernels through a C++ dispatcher and C++ autograd node, replacing Python/CuPy forward/backward wrapping while preserving tensor types, checkpoint layout, current stream and argument bytes. Recorded decision: retain main improvement; preliminary only; exact full recorded parity: True. Training throughput changed -13.96% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.84 versus 0.47; summed preparation s: 161.11 versus 118.28; validation s: 32.44 versus 29.63; primer s: 15.24 versus 15.31; worker CPU %: 62.81 versus 53.23; GPU activity %: 63.55 versus 66.54. These observations include machine-state variation and do not independently attribute a component’s effect.

**0073, E33-F48-F50-F65b-F65c.** Replace accepted exact GroupNorm Python autograd and CuPy gamma/beta dispatch with C++ autograd, identical ATen forward/input gradients and retained BF16 gamma/beta kernels, preserving reduction order and current stream. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -0.23% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.86 versus 0.84; summed preparation s: 162.19 versus 161.11; validation s: 32.82 versus 32.44; primer s: 15.29 versus 15.24; worker CPU %: 63.52 versus 62.81; GPU activity %: 64.82 versus 63.55. These observations include machine-state variation and do not independently attribute a component’s effect.

## Separate held-out binary losses

| Iteration | Metric | Loss | Delta vs prior accepted | Delta vs original baseline |
|---|---|---:|---:|---:|
| 0071 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0071 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0071 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0071 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0072 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0072 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0072 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0072 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0073 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0073 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0073 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0073 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |

## Checks and provenance

Full recorded parity is exact in 3/3 cases. The recorded acceptance gates and decisions are reused; this report does not change them or substitute parity for broader quality validation.

- 0071: experiment `a01ee5789139f561bb86bff0d353df4bf0a33d1c`, origin `7f73b6d94b9e9058670a8ef838dbf7352ecc2985`, dirty at process start: True; [source archive](../../sources/experiment-sources/956c7894cf177a4ab7b5321d7635bcccb39666370409937645db42d7e3562693/source.zip), [protocol](iteration-0071/protocol.json), [comparison](iteration-0071/comparison.json).
- 0071 pre-run background: 5.1% total CPU; 0 other Python GPU processes detected.
- 0072: experiment `50424589f196b51555988c981a71856afd152d98`, origin `7f73b6d94b9e9058670a8ef838dbf7352ecc2985`, dirty at process start: True; [source archive](../../sources/experiment-sources/b5a7c6f5149b10a8370c939b8f615f5c757f2aa370424ee2852a37b10c1dd817/source.zip), [protocol](iteration-0072/protocol.json), [comparison](iteration-0072/comparison.json).
- 0072 pre-run background: 9.0% total CPU; 0 other Python GPU processes detected.
- 0073: experiment `e4ebdabab770d7c6541f7947b0ad3f6f3ecaa16e`, origin `123acf9f04127635020d2754a7c8018647db0ad4`, dirty at process start: True; [source archive](../../sources/experiment-sources/54b7f2582ad4243df9f30196558442e162491f362f1141c801f694d89f723313/source.zip), [protocol](iteration-0073/protocol.json), [comparison](iteration-0073/comparison.json).
- 0073 pre-run background: 6.5% total CPU; 0 other Python GPU processes detected.

## Pending next steps

- Human explicitly stopped: no 0074 or later cycle, fresh GPU check, new agent or administrator prompt until explicit resumption; budget remains 3/10 completed and 7 unconsumed.
- Claude: read artifacts/diagnostics/r01-native-toolchain/baseline-drift/claude-handoff.md; historical 197499.49 versus current166338.05 reviews/s uses identical accepted options/config/exposure/initial+final parameters/evaluation input and scores. Gap cause unresolved; current native gain is local matched-only.
- After explicit resumption only: next 0074 previous 0073 preserves inherited10-count budget and 0.001 loss ceiling. Native shift and padded-row artifacts are untested drafts, not completed candidates.
- F70 event-lifetime/telemetry investigation remains pending; preserve host-wall estimator, paired gate and equivalent GPU span diagnostics. F59/F67/F69 remain pending; no broader validation authorized.
