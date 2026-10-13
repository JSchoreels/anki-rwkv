# gpu-retrain-r01

R01 retraining screens, resource profiles, fixed cohorts and statistical checks.

- **Open first:** [Latest claude round, 0076–0076](claude-round-0076-0076-summary.md); [compact handoff](claude-round-0076-0076-handoff.json). Previous: [Latest codex round, 0071–0073](codex-round-0071-0073-summary.md).
- **Earlier work:** Claude rounds [0025–0027](claude-round-0025-0027-summary.md), [0021–0024](claude-round-0021-0024-summary.md), [0010–0012](claude-round-summary.md); [Codex 0018–0020](codex-round-summary.md), numbered iteration reports and seed studies.
- **Evidence:** Raw JSON/JSONL, checkpoints, frozen protocols and isolated source metadata stay in their iteration folders; runtime folders preserve execution source.
- **Diagnostic repeat:** [Historical E33 during active desktop use](iteration-0040/summary.md): one authorized repeat, +7.05% throughput with exact recorded parity; accepted 0033 and the optimization stop remain unchanged.
- **Instrumented repeat:** [0042 with per-update timing](iteration-0046/summary.md): uncounted, exact parity, −18.13% throughput in a high-background slow state; it would raise the max/min noise band to 22.149%. Next round order: [docs/R01_NEXT_ROUND_PLAN.md](../../../docs/R01_NEXT_ROUND_PLAN.md).
- **Noise calibration:** [six replicates of 0042](iteration-0047/summary.md): all exact; 180,236 reviews/s mean, 2.57% run SD. Per-update trimmed estimator SD 1.01% against 3.74% for totals ([analysis](../../diagnostics/r01-calibration-0047/summary.md)).
