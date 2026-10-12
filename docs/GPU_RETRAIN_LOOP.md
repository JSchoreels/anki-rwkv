# R01: retraining optimization loop

## Goal
- Improve training speed, measured as raw review presentations per second.
- Support both Alex (default) and Expertium (smaller), while respecting the human-authorized model scope. Alex remains paused until explicitly resumed.

## Procedure
- Create each model's first baseline only when no compatible recorded baseline exists and that model is authorized.
- Reuse each model's original baseline and previous accepted run for later comparisons. Rerun a reference only for a stated comparability problem.
- Use the existing fixed 100-user cohort for R01 screens: 80 training users and 20 disjoint held-out users. Reuse compatible resource profiles; a cohort of 100 total users is not 100 evaluation users.
- Reuse applicable profiles; run focused smoke checks when changed code needs verification.
- Before each launch, reconcile the current LOOP with the frozen protocol or its amendment. Reuse matching results and checks; repeat only for a stated incompatibility or repaired failure, and give each new smoke a specific changed behaviour to check.
- Launch through the current workspace controller, using isolated source from the first trial.
- Select the next candidate from measured bottlenecks and concrete mechanisms; adapt selection after each result.
    - First address underused resources, then reduce work when utilization is already high. Treat roughly 80% utilization as a heuristic, not proof that a resource is the bottleneck.
    - Prefer exact recorded parity. For non-exact changes, require no more than 1% measured relative loss regression against the previous accepted run; no more than 2% against the fixed original baseline is preferred.
    - Apply these limits separately to Curve and Instant binary log loss, both label-weighted and user-averaged. Report uncertainty separately; the composite objective and rating cross-entropy are not these four losses.
    - Preserve failed, inconclusive and quality-regressing results in the records; recording a speed gain does not accept the candidate.
- Implement and try this improvement.
    - Commit successful implementation separately from experiment records; preserve the actual experiment revision, dirty-tree status and immutable tested-source snapshot.
    - If quality is unchanged or improved and speed is within the recorded timing spread, retain measured resource savings as pending retuning rather than claiming a speed improvement.
    - Keep faster candidates with worse losses in a separate opt-in fastmode, with all four measured regressions explicit. They do not replace the accepted main reference.
    - Keep fastmode separate from the main accepted configuration and its comparison reference.
- Promote adaptations with exact recorded parity when their measured reviews/s gain strictly exceeds the timing-noise band frozen from recorded comparable repeat evidence. The band is the median, over all pairs of complete full-history repeats with identical cohort, settings, options, config and trained weights, of max/min − 1 reviews/s (human decision 2026-10-12, replacing the widest group's max/min, which only grows and let one slow-state repeat set it). Under symmetric run noise, a truly neutral change beats this median about one time in four; record stack candidates and use paired timing to follow up. Exact parity requires initial weights, every logged update and gradient norm, final weights, held-out inputs and all four scores to match the previous accepted run. There is no additional fixed 2% or 5% promotion minimum; equality with the noise boundary is insufficient. Missing measured noise leaves promotion pending, rather than treating unknown noise as zero. Report the noise evidence and timing scope; exceeding an observed band does not establish timing repeatability.
- The original 5% throughput target was a proxy for timing noise, not an extra hard promotion floor. Non-exact changes retain the recorded timing-spread and quality rules, with 5% as their nominal miss-counter target. Future protocols record `exact_promotion_rule=timing_noise`; historical protocols and decisions retain their original rules.
- A round is a sequence of new candidate iterations. When the human specifies a count, complete that count even after consecutive speed misses. At round start, pass the quoted request with `--round-authorization` and the count with `--round-iterations N`; freeze the count and progress in each protocol. Diagnostics, smokes and control repeats do not substitute for new candidate iterations; repaired checks or run resumes do not create extra iterations. Keep the speed and quality rules unchanged.
- Stop at the requested count and require fresh authorization for more iterations. A partial handoff preserves the already-authorized remaining count; it does not close the budget or reset progress. Choose substantive candidates rather than repeating unchanged failures to fill the count; repair or report an actual blocker instead of claiming the count complete.
- When no explicit count is requested, stop each model before another iteration after three consecutive misses of its applicable target: failure to exceed measured noise for bitwise-equal adaptations, or the nominal 5% target for other changes.
- R01 results are preliminary screens. Larger cohorts or broader quality validation require a separate explicit human request; follow AGENT.md for that validation phase.

## Artifacts
- Keep GPU_OPTIMIZATIONS.md concise, with stable track IDs, decisions, pending ideas and links to detailed evidence. Preserve raw results, checkpoints, protocols and source snapshots.
- Add columns to the LOG table only after the user explicitly confirms them.
- Maintain one row per recorded run in GPU_OPTIMIZATIONS_LOG.md, using the existing columns and separate values rather than X/Y pairs:
    - Timestamp of the start of the training
    - The full experiment commit ID recorded when the process starts. Record dirty-tree status and tested-source snapshot in the raw evidence; keep later implementation commit IDs separate.
    - Model.
    - The stable improvement IDs from GPU_OPTIMIZATIONS.md and a concise human-readable explanation of the candidate (1–3 sentences). Put detailed evidence in the iteration artifacts.
    - The number of training users, distinguished from held-out users. Reuse the existing seeded 100-, 250- and 500-total-user cohort definitions and their 20% disjoint user holdout. Larger stages are definitions for separately authorized work, not permission to run them or broader validation.
    - The number of reviews trained on.
    - Training duration. Also keep complete process time and setup, preparation, transfer, validation and capture timing explicit in the detailed evidence; overlapping worker sums are not elapsed wall time.
    - Reviews/s and percentage change against the previous accepted run and fixed original baseline, retaining their original measurements. Disclose timing drift and any changed exposure, seeds, precision, batching or update schedule.
    - Binary log loss and its percentage change against the previous accepted run and fixed original baseline, separately for each metric.
        - Use different columns for Instant User, Instant Weighted, Curve User, Curve Weighted
    - Raw result link.
    - Notes.
    - Round percentages to two decimal places.
