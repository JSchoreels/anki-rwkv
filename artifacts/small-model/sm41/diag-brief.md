Task: diagnostic of the small model in the review simulator (SM41) full-run result (CPU only). Worktree `C:/Users/jonat/workspace/wt/sm41`, branch `sm41`. Always `git fetch` + `git merge --ff-only` first (if SSH to GitHub fails: `git -c credential.helper= -c credential.helper='!gh auth git-credential' fetch|push https://github.com/JSchoreels/anki-rwkv.git sm41`).

Context: the full run (commit 6321b689, `artifacts/small-model/sm41/fullrun-result.md`, DESIGN.md section 10) found that scheduling with the instant head (pick the candidate interval whose predicted recall hits the desired retention) costs +14.5% +/- 7.4 more minutes per memorised card than FSRS-7 at matched realised retention (116 users, paired). The simulated learner IS the small model, so the instant head predicts its own world; the user asked why a more accurate model schedules less efficiently. Hypotheses to separate:
- H1 per-card spread: "hit r on every card" is not cost-optimal; FSRS-7's per-card errors (some cards above, some below the target) give a cheaper mix at the same average.
- H2 selection: realised retention is the pass rate at review time only; a policy can match it with fewer reviews while holding less knowledge (compare memorised counts by the curve head vs the instant head: rows carry `memorised_instant` and `memorised_curve`).
- H3 counterfactual miscalibration: the policy reads the instant head at candidate intervals, outcomes are drawn by the simulated learner (check DESIGN.md section 1 for which head draws); a gap at long intervals pushes intervals shorter.
- H4 horizon: 60 days favours policies that let cards slide early.

Do:
1. Read DESIGN.md (sections 1, 9, 10), `scripts/sm41_engine.py`, `scripts/sm41_fullrun.py`. Keep the engine's numerics untouched; add an optional per-review / per-card log (off by default) so default runs stay bit-identical (prove it on 1 user: rows identical with logging off vs before the change).
2. Re-run a sample of ~20 users of the full run (a fixed seed-0 random pick from `fullrun/users.json`, 1 split each, the 0.90 lanes only: ihr-0.90 and fsrs7-0.90, seeds 0-3), logging per review: card, day, interval chosen, the instant head's predicted recall at that interval, the drawn outcome, and the learner's true recall probability if the engine has it. Output under `artifacts/small-model/sm41/diag/`. Check first that the sample's lane totals match the full-run rows for the same users/splits/lanes.
3. Analyse (paired per user, CIs over users): per-card realised retention spread (SD, quantiles) per policy; interval distributions (median, p90) per policy; where FSRS-7 saves reviews (by card difficulty tercile from the real history and by interval band); predicted-vs-drawn calibration of the instant head at the chosen intervals by interval band (H3); memorised by curve vs instant head (H2, from the full-run rows, all 116 users); cost gap by 15-day bucket (H4, from the full-run rows).
4. Write `artifacts/small-model/sm41/diag-result.md`: a verdict per hypothesis (supported / not / unclear, with the numbers), and what a cost-optimising policy would need. Commit (explicit paths, `git add -f` for artifacts/) and push.

Rules: commit as `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ends with a blank line + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; never --no-verify; push right after. Never commit `.claude/`, `docs/GPU_OPTIMIZATIONS_LOG.md`, `artifacts/coordination/queue/quiet.json`. Don't touch the main checkout or `pages/`. CPU only: at most 4 processes, each < 6 GB RAM; re-read `C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json` and stop before any window (next: 20:00 today). If the re-run would take more than ~3 h of wall time, shrink the sample and say so. Naming: plain name + ID in parentheses ("the small model in the review simulator (SM41)"), never a bare ID; never write "Expertium".

If you must wait on a long run (> 20 min): launch it detached (WMI Win32_Process.Create, ShowWindow=0, like HANDOFF.md's relaunch command), write `artifacts/small-model/sm41/HANDOFF-diag.md` with the pid, output paths and exact follow-up steps, commit + push it, and stop; reply with the expected end time.

Reply in under 250 words: commit hash, sample check, the verdict per hypothesis with key numbers, and the next step you recommend.

---
PART B (user 11-05 00:35: "explore more why FSRS-7 is better than RWKV in terms of efficiency, and what kind of scheduling could help with that"). Do it after part A, in the same worktree, as a separate commit and note `artifacts/small-model/sm41/policy-result.md`. The user's request lifts the pre-launch gate on the time-aware objective (DESIGN.md 9.2) for this exploration.

1. From part A's verdicts, write down (in the note, before running anything) which policies follow, the metric (the same primary metric: minutes per memorised card at matched realised retention, paired over users, 95% CI, vs FSRS-7) and the decision rule.
2. Candidate policies to test in the existing engine (pick the ones part A supports; at least 3; keep each a small addition to `scripts/sm41_engine.py` with defaults bit-identical):
   - curve-head target: pick the interval from the small model's curve head instead of the instant head (what FSRS-7 does with its own curve);
   - cost-aware interval: pick the interval that maximises expected memorised-card gain per expected minute (recall from the model, review time from the answer-time head or the per-grade costs), instead of hitting a fixed retention;
   - per-card target: retention target varying with the card (e.g. by predicted difficulty / stability growth), tuned on a few users and checked on held-out ones;
   - hybrid: FSRS-7's interval corrected by the small model (e.g. scaled by the ratio of the two models' predicted recall at FSRS-7's interval).
   Optionally: a short-horizon lookahead (two-step) policy, if cheap. Outside research (FSRS's "compute minimum recommended retention" and SSP-MMC-style cost-optimal scheduling) may be read for ideas, pointers only; keep our pipeline.
3. Screen on ~20 users (fixed seed-0 pick, 1 split, seeds 0-3, a few targets), then confirm the best 1-2 policies on the full-run users if a screen beats FSRS-7 or comes within the CI. Same holds/quiet rules; hand off before any wait > 20 min.

Final reply (under 300 words): part A verdicts with key numbers; part B policies tried, their gap vs FSRS-7 with CI, the best one, and the recommended next step (including whether it could become an Anki scheduler).

---
ADDED (user 11-05 00:45): "I wonder if FSRS inaccuracy leads to lower actual R cards, which are often more efficient than higher R". Test this hypothesis (call it H5, a sharper form of H1) explicitly in part A:
- per review, log the simulated learner's TRUE recall probability at review time (not the policy's own prediction); compare its distribution per policy (mean, SD, quantiles) at matched realised retention;
- efficiency vs true R: per review, the gain it buys (e.g. growth of the learner's predicted retention / time-to-forget after the review, from the curve head) and its cost (minutes), binned by true R at review; is the gain per minute higher at lower R, and is the relation convex (so a spread beats a constant at the same mean)?
- where FSRS-7 lands low: is its low-R mass concentrated on cards where a late review pays most (correlate FSRS-7's prediction error with the card's gain-per-minute)?
- causal check in part B (cheap, do it first): the instant-head policy with deliberate per-card noise on its target (e.g. target logit + N(0, s) per review, s chosen to match FSRS-7's true-R spread), and a variant that sets lower targets on cards with high gain-per-minute. If noise alone closes the gap, H5 is confirmed and the useful policy is the targeted variant.
