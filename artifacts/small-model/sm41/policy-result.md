# The small model in the review simulator (SM41): policy search (part B)

User request (11-05 00:35): explore why FSRS-7 is more efficient than the small model's instant head, and what
scheduling could help. The request lifts the pre-launch gate on the time-aware objective (DESIGN.md 9.2) for this
exploration. Brief: [diag-brief.md](diag-brief.md), part B. Part A: [diag-result.md](diag-result.md).

## Plan (written before any part B run)

### What part A implies

- The gap is **seconds per review** (+7.9 ± 1.6% at matched retention, 116 users), not knowledge held (H2), not
  miscalibration (H3), not the horizon (H4).
- The instant head reads slow answers as weak memory, so it brings slow cards back sooner. FSRS-7, blind to time,
  reviews them late (true recall 0.859 vs 0.874) and reviews fast cards early (0.908 vs 0.888).
- A spread of true recall at the same mean buys nothing in this learner: gain per minute rises with recall, and the
  Jensen check is -0.4 ± 7.4%.
- So the policy that part A supports is a **time-aware per-card target**. Noise and the curve head are controls.

### Policies (each a small option of `scripts/sm41_engine.py`; the default lanes stay bit-identical: user 1606 split 660, 9 default lanes, rows identical to before the change, `policy/bitcheck/`)

| lane | policy | why |
|---|---|---|
| `ihr-R-n0.5` | instant head, target logit + N(0, 0.5) per decision | **H5 causal check**: s = 0.5 adds about the per-card true-recall spread FSRS-7 has on top of the instant head (SD over cards 0.082 vs 0.063: sqrt(0.082² - 0.063²) = 0.052 in recall, / R(1-R) ≈ 0.106 at R ≈ 0.88). If noise alone closes the gap, the spread is the cause. |
| `ihr-R-s<β>` | instant head, per-card target: logit R - β ln(the card's last later-day answer time / the user's median later-day answer time before the split) | **the targeted variant**: lower targets on cards that are slow to answer (expensive), higher on fast ones. Part A's mechanism. β = 1 first, then one more β. |
| `curve-R` | the curve head's interval (the commit row's GRU power curves), as FSRS-7 uses its own curve | control: does a curve-based target (it sees the answer time only through the state) behave like FSRS-7? |
| `hyb-R-a0.5` | hybrid: FSRS-7's interval × (ln R / ln P_instant(FSRS-7's interval)) ^ 0.5 | half-way between FSRS-7 and the instant head on each card. Is the best of both in between? |

### Not tested

- The cost-aware "maximise gain per minute" interval: part A's gain per minute rises with recall, so a one-step
  ratio would push every card to high retention. That is the opposite of what the metric rewards: the cheapest
  policies have the lowest targets.
- The SSP-MMC-style table: for the same reason it has no lever here beyond the per-card answer time, which `s<β>`
  already tests.

### Metric and decision rule (the full run's, unchanged)

- **Metric.** Minutes on days 30-59 per memorised card at day 60 (the instant head's count), at matched realised
  retention.
  - Each policy's curve (3 targets: 0.80 / 0.85 / 0.90, 4 seeds each) is compared with FSRS-7's 6-target curve over
    their common realised-retention range (grid step 0.005; overlap at least 0.02).
  - A user's splits are averaged. Mean over users with a 95% CI (t), paired.
  - The FSRS-7 and instant-head curves come from the full-run rows for the same users, splits and seeds. Lanes are
    independent and share their draws card by card (proof P1; part A's check reproduced these rows to 1e-6).
  - Also reported: the gap against the instant head's curve on the same 3 targets. Did the policy improve on the
    instant head?
- **Screen.** The 20 users of part A (1 split each, seeds 0-3).
- **Decision rule.**
  - A policy goes on to confirmation if its screen gap vs FSRS-7 is < 0, or its CI includes 0.
  - At most the best 2 policies are confirmed, on the full-run users (all 117 users' splits). Confirmation passes
    only if the gap is < 0 with the CI excluding 0 ("beats FSRS-7"). "Within the CI" counts as "as good as FSRS-7".
  - For `s<β>`, β is tuned on the 20 screen users. Confirmation on all users includes 97 users that were not used
    for tuning, and the gap is also reported on those 97 alone (held out).
  - **H5 causal check:** H5's spread form is confirmed only if `n0.5` moves the gap vs FSRS-7 by at least half of the
    instant head's gap on the same users, toward 0.
- **Holds.** CPU only, 4 processes x 1 thread, below-normal priority, `quiet.json` holds (next window 20:00). Any
  wait over about 20 minutes: launch detached, write a handoff, commit, stop.

## Results

### Screen 1 (20 users, 01:32-01:59; `policy/screen1/screen1-tables.md`)

| policy | users | gap vs FSRS-7, % (95% CI) | median | cheaper than FSRS-7 | gap vs instant head (same 3 targets) |
|---|---|---|---|---|---|
| instant-head desired retention, targets 0.80-0.90 (the reference, full-run rows) | 19 | +5.6 ± 5.0 | +3.8 | 26% | — |
| instant head + target noise s = 0.5 (`ihr-*-n0.5`) | 18 | +4.2 ± 4.5 | +1.2 | 33% | -2.2 ± 2.8 |
| instant head + time-aware target β = 1 (`ihr-*-s1`) | 19 | **-6.9 ± 6.0** | -5.5 | 79% | **-10.9 ± 4.2** |

- **H5 causal check.** Noise moves the gap by 1.4 points, less than half of the reference's 5.6. The spread form of
  H5 is **not confirmed**, as part A predicted.
- **The time-aware target** beats FSRS-7 on the screen users, with the CI excluding 0. Per target it holds about the
  same memorised count with fewer minutes:
  - at 0.90: 22.0 against 25.5 min/day, 1084 against 1102 memorised;
  - FSRS-7 at 0.90: 23.4 min/day, 1090 memorised.

Screens 2 (`curve-R`, `hyb-R-a0.5`) and 3 (β = 0.5 and 2), the choice of what to confirm, and the confirmation on
all full-run users are running: see [HANDOFF-policy.md](HANDOFF-policy.md).
