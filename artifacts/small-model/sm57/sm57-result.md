# The instant-only small model (SM57): result (2026-11-07)

Unit: points = % of the big model (1 leak-free epoch) LogLoss, paired user bootstrap 2,000, seed 0, 95% CI. Pool: 722 users of the 732 set (rule), 725 with the 3 long users. Leak-free scoring, normal inputs. Full tables: `sm57-decision.md` / `.json`.

## Verdict: GO

Rule: instant-only - joint on the instant head, CI entirely below zero and gain >= 0.05 points. Result -0.147 [-0.162, -0.133] points: go. The 0.05-point minimum was added by the user on 11-07 ~00:40, before the joint arm was scored (commit 6a2caedb). Go means stage 2 is proposed, not run.

## Numbers (722 users)

| comparison | head | points [95% CI] | users better / worse |
|---|---|---|---|
| instant-only - joint (the rule) | instant | -0.147 [-0.162, -0.133] | 643 / 78 |
| instant-only - the release (SM63) | instant | -0.162 [-0.176, -0.148] | 646 / 75 |
| joint - the release (SM63): one more epoch | instant | -0.015 [-0.020, -0.009] | 442 / 278 |
| joint - the release (SM63): one more epoch | curve | +0.034 [-0.018, +0.119] | 377 / 344 |
| instant-only - the big model (1 leak-free epoch) | instant | +0.122 [+0.098, +0.145] | 237 / 484 |
| joint - the big model (1 leak-free epoch) | instant | +0.269 [+0.239, +0.298] | 139 / 583 |
| joint - the big model (1 leak-free epoch) | curve | +0.390 [+0.217, +0.590] | 232 / 489 |
| the release (SM63) - the big model (1 leak-free epoch) | instant / curve | +0.283 / +0.356 | |
| instant-only, untrained curve head - the big model | curve | +16.773 [+15.319, +18.478] | 13 / 709 |

732+long (725 users): instant-only - joint -0.161 [-0.190, -0.139] (646 / 78); instant-only - the release -0.175 [-0.205, -0.153]; joint - the release curve +0.023 [-0.032, +0.105], instant -0.014 [-0.020, -0.009].

Smaller-model bar (<= +0.5% curve / <= +0.4% instant vs the big model (1 leak-free epoch)): joint passes (+0.390 / +0.269), the release passes (+0.356 / +0.283), instant-only passes on instant only (+0.122; its curve head is untrained, so the curve column is meaningless: +16.8%).

## What it means

- Giving the whole small model to the instant task is worth about 0.15 points of instant LogLoss, and 86% of users gain. It brings the small model to +0.12% of the big model (1 leak-free epoch) on instant, against +0.27 for the joint model. The gain is small in absolute terms (LogLoss 0.27609 vs 0.27650).
- One-more-epoch effect (joint - the release, same recipe plus epoch 26): -0.015 instant [-0.020, -0.009], +0.034 curve [-0.018, +0.119] (CI includes zero). So an extra epoch of the release is worth about 0.015 points on instant and nothing measurable on curve. The instant-only gain is about 10 times that effect. Implication: more epochs of the release (SM63 epochs 26-30) are not worth a heavy GPU spend; at most ~0.07 points of instant if the trend were linear, and it will flatten. Dropping the curve loss buys far more than five more epochs.
- Caveat: the instant-only model cannot schedule by itself; its curve head is untrained (curve +16.8% worse than the big model).

## Training (per arm and stage)

| arm / stage | wall min | reviews/s | peak allocated / reserved MiB | nvidia-smi max / baseline MiB |
|---|---|---|---|---|
| instant-only / epoch 26 | 111.2 | 87,706 | 10,892 / 13,014 | 15,710 / 2,627 |
| instant-only / cool-down | 477.1 | 20,337 | 7,428 / 12,840 | 15,429 / 2,305 |
| joint / epoch 26 | 111.2 | 87,562 | 10,892 / 13,014 | 15,700 / 2,305 |
| joint / cool-down | 370.0 + ~102 + 6 (no-op check) = ~478 | 20,3k first segment; 21,425 last (the segment table only covers the last) | 7,420 / 12,780 | 15,106 / 464 (first segment) |

Joint cool-down wall clock: segment 1 by the launcher 11-06 13:47-19:57 (370 min), segment 2 by `gpu/extra-segment-1106.sh` 20:00-21:42 (~102 min; the user moved the break to 22:00 that night), then a no-op completion check at 01:06 (6 min). `gpu/segments.jsonl` has only the launcher's segments. Speed and memory are the same for both arms (the instant-only loss skips no compute: the curve head still runs forward). With 87.6k reviews/s on the 8k set, a 10-epoch run of either arm is ~18.5 h of epochs (111 min each) plus a ~8 h cool-down; the 2k-set ETA is about a quarter of the epoch time (~28 min/epoch), ~4.6 h for 10 epochs, assuming the same speed.

## Stage 2 proposal (not run)

From-scratch comparison at small scale: instant-only vs joint from the same random init and seeds, 2,048 windows, 5 epochs each, same data order, scored on the 722-user rule pool with the same paired bootstrap, go if the instant CI is below zero. Question: does the curve task help instant learn early (then the joint model wins from scratch and the gain here is a late-training effect) or does it only compete for capacity. Estimated GPU cost (estimate, not measured at this scale): ~28 min per epoch x 5 x 2 arms = ~4.7 h, plus 2 x ~15 min scoring, about 5.2 h total, fits in one evening window and uses ~13 GB (cap 0.8). A 10-epoch version is about 10 h.

## Using an instant-only model in practice

It cannot schedule by itself. Options: (a) a second model for instant next to the normal one (the release for the curve, the instant-only model for instant): two models to ship and run, about double the small-model cost for a ~0.15 point instant gain; (b) an instant-only RWKV-Instant scheduler that schedules from instant predictions alone (FSRS7-Instant style scheduling on top of the instant head): one model, but it needs a scheduling path that does not use the curve. Given the small gain, (b) is the only option with a clear payoff, and it needs its own validation (simulator) before any release. The normal default stays the release (SM63).
