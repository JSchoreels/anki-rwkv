# The smaller instant-only model (SM101): result (2026-11-11)

Launcher finished 06:36 Wed 2026-11-11 with "done (verdict go)". The rule `sm101_rule.py` ran in the launcher; see
`sm101-decision.json` / `sm101-decision.md`. Units: "%" = % of the big model (1 leak-free epoch)'s normal-input instant
LogLoss 0.275757 (732 set, 722 users, review-weighted, paired user bootstrap 2,000, seed 0, 95% CI), unless stated.
Chain: the show-time instant-only model, stage 2 (SM98)'s epoch 35 cut to 1 card layer (w64-c1), 3 constant-LR
instant-only epochs (36-38) on show-time inputs, an early check after epoch 36, then the release's one-epoch 32k Muon
cool-down on show-time inputs from epoch 38 (curve loss weight 0).

## Verdict: go

- A, the bar: the smaller instant-only model (SM101) minus the big model (1 leak-free epoch), instant head, normal
  inputs: **+0.298% [+0.268, +0.329]** (0.276578 vs 0.275757). Pass needs the CI upper end <= +0.40%: **pass**.
- B, what the fork ships for RWKV-Instant: the smaller instant-only model (SM101) minus the show-time retrain of the
  release (SM92), instant head, show-time inputs: **-0.057% [-0.079, -0.034]** (0.27672 vs 0.276876), 460 users better /
  261 worse. Pass needs the CI below 0 and a gain of at least 0.05%: **pass**.

The curve head is untrained (weight 0) and is not part of the verdict.

## Secondary (732 set, instant head, %)

| new minus | inputs | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|---|
| the show-time instant-only small model (SM96) | show-time | 0.27672 | 0.276287 | +0.157 [+0.137, +0.178] | 140 / 580 |
| the show-time instant-only model, stage 2 (SM98) | show-time | 0.27672 | 0.276191 | +0.192 [+0.171, +0.215] | 114 / 607 |
| the release (SM63) | show-time | 0.27672 | 0.277718 | -0.362 [-0.463, -0.284] | 574 / 148 |
| the show-time instant-only small model (SM96) | normal | 0.276578 | 0.276134 | +0.161 [+0.141, +0.182] | 140 / 582 |
| the show-time instant-only model, stage 2 (SM98) | normal | 0.276578 | 0.27606 | +0.188 [+0.167, +0.211] | 119 / 603 |
| the show-time retrain of the release (SM92) | normal | 0.276578 | 0.276757 | -0.065 [-0.088, -0.042] | 472 / 249 |
| the big model (1 leak-free epoch) on normal inputs | show-time vs normal | 0.27672 | 0.275757 | +0.349 [+0.314, +0.387] | 108 / 614 |
| the new model on normal inputs (its show-time cost) | show-time vs normal | 0.27672 | 0.276578 | +0.051 [+0.039, +0.065] | 186 / 511 |

So the smaller instant-only model (SM101) is about 0.16-0.19% worse than the show-time instant-only small model (SM96)
and the show-time instant-only model, stage 2 (SM98), as expected from dropping one card layer, but it still beats the
show-time retrain of the release (SM92) and the release (SM63).

## 1,500 validation users outside the 732 set, show-time inputs (1,487 users with a release reference), % of the release's own LogLoss

| comparison | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|
| new minus the release (SM63) | 0.279693 | 0.2806 | -0.323 [-0.374, -0.282] | 1216 / 271 |
| new minus the show-time instant-only small model (SM96) | 0.279693 | 0.279249 | +0.159 [+0.141, +0.180] | 333 / 1152 |
| new minus the show-time instant-only model, stage 2 (SM98) | 0.279693 | 0.279162 | +0.190 [+0.171, +0.212] | 244 / 1240 |

Row check on the 6 check users: max abs LogLoss difference 0.0. Only 767 of the 771 pool users were scored: the last 4
did not fit even at the 0.92 GPU memory cap; 1,487 users have a release reference.

## Early check (after epoch 36)

The smaller instant-only model (SM101)'s epoch 36 minus the show-time instant-only model, stage 2 (SM98)'s epoch 35 (the
cut's source), instant head, show-time inputs, 722 users: **+0.051% [-0.004, +0.109]**, 378 better / 343 worse (0.278963
vs 0.278821). The pre-registered rule stops above +0.25%, so it said continue.

## Speed

| step | reviews | wall min (segments) | training min (log) | reviews/s (log) | reviews/s (wall) | nvidia-smi max (baseline) |
|---|---|---|---|---|---|---|
| epoch 36 | 581,942,724 | 105.9 | 105.6 | 91,875 | 91,587 | 13,729 MiB (729) |
| epoch 37 | 581,942,724 | 109.4 | 109.2 | 88,837 | 88,657 | 14,982 MiB (737) |
| epoch 38 | 581,942,724 | 110.1 | 109.5 | 88,602 | 88,093 | 13,967 MiB (1,764) |
| cool-down segment 1 (17:09-19:57 on 11-10, stopped for the break with a snapshot) | partial | 168.8 | n/a | n/a | n/a | 12,900 MiB (1,613) |
| cool-down segment 2 (01:00-01:56 on 11-11, cut by a machine reboot; resumed from its snapshot, a few minutes lost) | partial | about 56 | n/a | n/a | n/a | not logged (no segment line) |
| cool-down segment 3 (01:58-05:52 on 11-11, the rest) | 291,028,216 | 233.6 | 232.8 | 20,839 | 20,761 | 15,297 MiB (2,207) |
| 1-epoch 32k Muon cool-down, total | 581,942,724 | about 458 | n/a | n/a | about 21,160 | 15,297 MiB |

The cool-down total is the three segments added (168.8 + about 56 + 233.6 wall minutes), so about 21,160 reviews/s
includes the few minutes redone after the reboot; the last segment's logged 20,839 reviews/s is the clean figure (the
cool-down of the show-time instant-only model, stage 2 (SM98), ran at 20,232). Overall nvidia-smi max: 15,297 MiB
(cool-down segment 3), logged by the launcher; epochs peaked at 13,729-14,982 MiB. Segment 2's own max was not logged
because the reboot killed it before the segment line was written.

10-epoch ETA at the log speeds (8k tier = 581.94M reviews per epoch, 2k tier = 147.6M); constant-LR epochs use epoch 36's
91,875 reviews/s (epochs 37-38 run at 88.6-88.8k, which gives 4.6 h / 18.2 h):

| phase | 2k tier | 8k tier |
|---|---|---|
| 10 constant-LR epochs at 91,875 reviews/s | 4.5 h | 17.6 h |
| 10 cool-downs at 20,839 reviews/s | 19.7 h | 77.6 h |

## Total GPU time and tickets

25 queue tickets (owner sm101, `history.jsonl`: smokes, scoring passes, 3 constant-LR epochs, and the 2 cool-down
tickets that finished) summing to 796 GPU minutes between start and finish of each ticket, plus about 56 minutes of the
reboot-cut cool-down segment that left no ticket, so about 852 GPU minutes (about 14.2 h). The launcher's
`gpu/tickets.jsonl` holds 16 rows (scoring and smokes only). Run window: first ticket 11:12:45 Tue 2026-11-10 (+01:00) to
the last ticket 06:36:43 Wed 2026-11-11 (+01:00), with the 20:00-01:00 break in between.

## State bytes (the engine side)

When the smaller instant-only model (SM101) runs alone, its state is 3,134 B per card instead of 4,554 B (-1,420 B, -31%);
2,110 B with int4 state. The fork's paired case gives no saving today: `MODULE_LAYERS` is a compile-time constant, so
pairing a 1-card-layer model with the default small model (SM63) (RWKV-Curve and RWKV-Instant both on) would need an
engine change, and it stays at 9,146 B.

Since 2026-11-10, fork main runs the show-time instant-only model, stage 2 (SM98) ALONE whenever no preset uses
RWKV-Curve. The smaller instant-only model (SM101) would replace exactly that case and save memory and CPU there.
