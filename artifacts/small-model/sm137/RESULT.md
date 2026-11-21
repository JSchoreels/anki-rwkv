# Trained smaller state for Tiny (SM137): result

Verdict: **no go**. The Instant-Only chain passed the main rule but failed the pre-registered long-card check, so the chain ends as no go; the two-head chain did not run, because the launcher runs it only after an Instant-Only go.

What was tried: the six-layer Tiny (SM130)'s card and note state cut to 16 value rows x 8 of 16 key lanes per head (`w64-c1-d2p1u1-v16k8`), retrained with the same two-pass Instant-Only cool-down, so that the fork stores **803 B per card and per note** instead of 1,315 B at the release-candidate format w8+s8hx8.

## Main rule (passed)

The rule from `design.md`: per trained head, d = 100 x (the student at the release format w8+s8hx8 / the longer Small Instant-Only model (SM113) at its published w8+s8 - 1); go iff the 95% CI upper bound is at most +1.00%, every rule user is finite, and the long-card check passes. 722 users of the 732 set, show-time inputs, no non-finite user (`sm137-decision-io-v16k8.json`).

| comparison (instant head) | % [95% CI] | LogLoss new / reference | better / worse users |
|---|---|---|---|
| the student at w8+s8hx8 vs the longer Small Instant-Only model (SM113) at w8+s8 (the rule) | +0.576% [+0.534, +0.619] | 0.277522 / 0.275933 | 27 / 695 |
| the student at w8+s8 vs the longer Small Instant-Only model (SM113) at w8+s8 | +0.575% [+0.533, +0.618] | 0.277518 / 0.275933 | 29 / 693 |
| the student at w8+s8hx8 vs the six-layer Tiny (SM130) at w8+s8hx8 | +0.010% [+0.005, +0.015] | 0.277522 / 0.277494 | 305 / 417 |
| the student at w8+s8 vs the six-layer Tiny (SM130) at w8+s8 | +0.010% [+0.005, +0.015] | 0.277518 / 0.277490 | 294 / 425 |

The upper bound +0.619% is under the +1% ceiling, so the main rule passes. Against the six-layer Tiny (SM130) the smaller state costs about 0.01%, which is below the 0.05% minimum effect we treat as meaningful.

## Long-card check (failed)

The rule from `design.md`: on the 50 users of `users-longcard.json`, CPU scorer, the student and the six-layer Tiny (SM130) final both at w8+s8hx8, excess = (LogLoss cost vs the six-layer Tiny (SM130) on reviews of cards with 128 or more prior answers) - (the cost on all reviews), paired user bootstrap; pass iff no student user is non-finite and the excess CI upper bound is at most +0.25% (`sm137-longcard-io-v16k8.json`).

| instant head, 50 users | % [95% CI] |
|---|---|
| cost on the 30,089 reviews of cards with 128+ prior answers | -0.003% [-0.167, +0.417] |
| cost on all reviews | +0.005% [-0.010, +0.020] |
| excess (the rule) | -0.008% [-0.175, +0.412] |

No user was non-finite. The CI upper bound +0.412% is above +0.25%, so the check fails and the pre-registered verdict is **no go**.

An honest reading: the failure comes from a wide interval, not from a measured loss. The point estimate of the excess is about zero (-0.008%), and the interval reaches both well below and well above zero. With 50 users and a bucket of about 0.8% of reviews (30,089 reviews), the check had little power to show that long cards lose nothing. The rule is kept exactly as pre-registered; it is not reinterpreted or re-run here.

## Speed and timeline (Sat 11-21)

- Pilot 01:31 to 02:58 (163,493 reviews/s), go; readout `sm137-pilot-io-v16k8.md`.
- Full plan launched 03:19 (`launch-sm137-full.sh`, log `gpu/launch-sm137-full.log`).
- Instant-Only cool-down pass 1: 03:19 to 05:51, 581,942,724 reviews, 151.7 min training, 63,916 reviews/s; learning rate ended at 0.2929 x peak (expected [0.2925, 0.2935]); dropped key and value rows of the master exactly 0.
- Pass 2: 05:52 to 08:24, 581,942,724 reviews, 151.9 min training, 63,853 reviews/s; learning rate ended at 5.8e-11 x peak; dropped rows exactly 0.
- Final scoring on the GPU: 08:24 to 08:48 (the student at w8+s8 and w8+s8hx8, then the six-layer Tiny (SM130) final at w8+s8hx8, 723 users each).
- Long-card CPU rows: 08:54 to 09:15 (50 users, 2 processes at a time, memory guard 5.9 GB per process); rule and verdict at 09:15; launcher done 09:16.
- At about 63,900 reviews/s, a full 8k-user, 10-epoch run would take about 25 h of training (5.72 billion reviews), plus about 1.2 h per 2k-user held-out evaluation pass.

Launcher bug: at 08:48 the long-card step aborted because the Windows Python output carried a carriage return into a bash arithmetic expression, and the launcher wrote a false "done (full: io not run, th not run)" line. Commit 2ce902f8 strips the carriage return and makes a chain that ends without a verdict file stop with STOP. The launcher was relaunched at 08:54; finished steps (both passes, the scoring) were skipped, and only the long-card check and the rules ran. No result was changed by the bug.

## What this means for the release

Per the week plan, the trained smaller state for Tiny (SM137) stays experimental and is not in this release. The six-layer Tiny (SM130) at 1,315 B per card and per note stays the release Tiny.

A possible follow-up for a later release, as an idea only: a long-card check with more power, using more long-card users from outside the 732 set, pre-registered before it runs, CPU-only under the 5.9 GB memory guard. Nothing is planned or launched for it.

## Files

- Rules and readouts: `design.md`, `sm137-decision-io-v16k8.{json,md}`, `sm137-longcard-io-v16k8.json`, `sm137-pilot-io-v16k8.{json,md}`.
- Logs and rows: `gpu/launch-sm137-full.log`, `gpu/sm137-rows.jsonl`, `gpu/tickets.jsonl`, `gpu/segments.jsonl`, `gpu/verdict-full-io`, `gpu/verdict-full-th`, `cpu/sm137-longcard-rows.jsonl`.
- Model folders (local, weights not committed): `artifacts/small-model/sm137-io-v16k8-dec2ep32k-p1`, `artifacts/small-model/sm137-io-v16k8-dec2ep32k-p2`.
