# R01 0063–0065 preparation

Inherited budget 23/index 15 complete, eight remaining through 0070; exactly 0063–0065 are assigned here. Accepted 0042/options, original loss baseline and cumulative ceiling 0.001 inherited. Alex/broader validation paused; parent manages chats/shutdown.

0063 combines retained original 0048 F58 recurrence_prefetch=true and 0061 F58e recurrence_load_pipeline=5. This is a new bundle requiring combined synthetic and five-real-batch checks. 0061 mean +2.714% was dominated by one +15.964% block with faster preparation and validation; this remains a caveat, not proof of component benefit.

0064 F58g selects mode 5 only for histories T >= 32; 0065 F58h selects it only for T <= 16. The nine recorded real WKV shapes include 256/68-step chains and broad 2/4/10/12-step batches. Two distinct host-selected schedules test long sequential latency versus short-history live-set overhead, retaining accepted four-row layout and all arithmetic/checkpoint/reduction/skip semantics. Middle histories retain the baseline. This prioritizes a feasible shape schedule over deferred whole-group F62/eight-way lerp/F49/F51/F52; those mechanisms remain pending. No rejected option mode or unchanged counted replay is reused.

All relevant source edits finish before checks. Existing watched backend/check files and recurrence_load_pipeline allowlist are reused. Synthetic checks cover both selected and fallback shapes; real checks compare five losses/every gradient against accepted 0042. No claims of broader quality validation.

All final-source checks passed: synthetic-combo/long/short each 216 BF16/FP32 cases; real-combo/long/short each five losses/every-gradient batches. Implementation ea776234cf7703b1305c1159d49603b53bf51a36 committed after checks and before every counted experiment. No subsequent training-source edits.

0063 combination rejected at two blocks (-0.015%, lower -1.421%). 0064 long-history schedule extended through 2/4/6 looks (+0.118%/+0.092%/+0.017%) and retained unproven at the cap (lower -0.797%). 0065 short-history schedule rejected at two (-0.296%, lower -1.698%). All three full cases and twenty replicas are exact; zero retries/OOMs, 3157 MHz median SM clocks and 12,188 MiB reserved throughout. [Audit](round-audit.json).

0064/0065 first report attempts hit Windows Errno 22 opening GPU_OPTIMIZATIONS_LOG.md; explicit report-only resumes succeeded without repeating training/profile. Failed logs are preserved. 0065 first candidate replica waited at 10.7% background CPU until the at-most-10% gate cleared. Campaign 18/23 complete, five remaining; continue 0066/previous 0065 with original baseline and ceiling 0.001. Accepted 0042 unchanged; retain original 0048 F58, 0061 F58e and 0064 F58g as unproven alternatives. Modes 5/7 are mutually exclusive schedules. Omit combined 0063 and mode 8, plus all prior rejections; no identical counted replay or no-op consolidation.
