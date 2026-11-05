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

### Screens 2 and 3 (same 20 users; `policy/screen-tables.md`, `policy/lanes-confirm.json`)

| policy | users | gap vs FSRS-7, % (95% CI) | gap vs instant head |
|---|---|---|---|
| curve head (`curve-*`) | 19 | +9.9 ± 8.2 | +1.5 ± 6.7 |
| hybrid, alpha = 0.5 (`hyb-*-a0.5`) | 19 | +6.0 ± 5.3 | -0.5 ± 3.9 |
| noise s = 0.5 (`ihr-*-n0.5`, control) | 18 | +4.2 ± 4.5 | -2.2 ± 2.8 |
| time-aware beta = 0.5 (`ihr-*-s0.5`) | 18 | +1.2 ± 3.9 | -4.9 ± 2.4 |
| time-aware beta = 1 (`ihr-*-s1`) | 19 | -6.9 ± 6.0 | -10.9 ± 4.2 |
| time-aware beta = 2 (`ihr-*-s2`) | 19 | **-18.7 ± 8.4** | -21.2 ± 6.7 |

- The curve head and the hybrid do not help. The gap shrinks steadily as beta grows; beta = 2 is the largest tried
  (the optimum may lie beyond it).
- **Passed to confirmation** (rule: the best time-aware family, plus the best other family that passes, at most 2):
  `ihr-*-s2` and `ihr-*-n0.5`. The noise rule is a **control**: it was picked because its screen CI included 0,
  not because it looked good.

### Confirmation (117 users, 304 splits, 60 days; `policy/confirm-tables.md`, `policy/confirm-heldout-tables.md`)

| policy | users | gap vs FSRS-7, % (95% CI) | median | users cheaper | gap vs instant head | 60-day cost gap vs FSRS-7 |
|---|---|---|---|---|---|---|
| instant head, targets 0.80-0.90 (reference), all users | 115 | +9.0 ± 3.3 | +7.1 | 20% | — | +8.7 ± 2.8 |
| noise s = 0.5 (control), all users | 116 | +5.3 ± 2.8 | +4.6 | 31% | -3.2 ± 1.1 | +6.5 ± 2.2 |
| **time-aware beta = 2, all users** | 110 | **-22.6 ± 3.2** | -21.7 | 92% | -26.1 ± 2.9 | -20.3 ± 2.8 |
| instant head (reference), 97 held-out | 96 | +10.0 ± 3.8 | +8.8 | 17% | — | +9.3 ± 3.2 |
| noise s = 0.5 (control), 97 held-out | 97 | +5.9 ± 3.2 | +5.0 | 28% | -3.4 ± 1.3 | +7.3 ± 2.5 |
| **time-aware beta = 2, 97 held-out** | 91 | **-23.6 ± 3.5** | -21.8 | 92% | -27.3 ± 3.2 | -21.1 ± 3.2 |

Per target, all users (days 30-59):

| policy | target | realised retention | min/day | memorised |
|---|---|---|---|---|
| time-aware beta = 2 | 0.80 / 0.85 / 0.90 | 0.825 / 0.858 / 0.893 | 16.2 / 18.5 / 21.6 | 1151 / 1174 / 1201 |
| instant head | 0.80 / 0.85 / 0.90 | 0.801 / 0.838 / 0.879 | 20.2 / 23.4 / 28.1 | 1195 / 1225 / 1252 |
| FSRS-7 | 0.80 / 0.85 / 0.90 / 0.95 | 0.769 / 0.820 / 0.875 / 0.929 | 19.6 / 23.9 / 29.9 / 38.9 | 1167 / 1203 / 1240 / 1283 |

At about the same realised retention (beta = 2 at 0.893 vs FSRS-7 at 0.875) the time-aware rule costs 21.6 against
29.9 min/day.

- **Verdict (the plan's rule): the time-aware target beats FSRS-7** (gap < 0, CI excluding 0), on all users and on
  the 97 held-out users alike (-22.6 vs -23.6; 92% of users cheaper in both). The held-out users agree.
- **H5 causal check:** noise moves the gap from +9.0 to +5.3 (41% of the instant head's gap, less than half), and
  from +10.0 to +5.9 held-out (41%). The spread form of H5 is **not confirmed**. Noise is a control, not a
  candidate: its gap stays positive.
- The instant head alone stays costlier than FSRS-7 (+9.0 ± 3.3), as in the full run.

### Could it become an Anki scheduler?

The target needs only the card's last answer time and the user's median answer time, both in the revlog. So it is a
per-card desired-retention rule on top of any recall model: in the simulator it needs the instant head, and in Anki
it would need the fork's RWKV runtime. Caveat: the simulator's learner is the same small model, so the gain is
inside the model's world. Beta = 2 was the largest value tried. Next step, which needs the user's go-ahead: an
offline check on the real revlog (are slow-answer cards cheaper to review late?), and a larger-beta screen.

### Run cost (CPU only)

4 processes x 1 thread. Simulation about 445 reviews/s per process on the confirmation (35.9M simulated reviews,
22.4 h), 619/s on screen 2 and 431/s on screen 3; about 26 process-hours of simulation plus 4 of warm-up, about 30
CPU process-hours, 7.5 h wall (02:01-09:31).
