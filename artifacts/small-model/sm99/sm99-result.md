# The show-time instant-only big model (SM99): result (2026-11-12)

Launcher finished Thu 2026-11-12 07:43 with "done (verdict ceiling above SM98 / sm99-big-inst-e2/w8+s8)". The rule
`sm99_rule.py` ran in the launcher; see `sm99-decision.json` / `sm99-decision.md`. Units: "%" = % of the big model
(1 leak-free epoch)'s normal-input instant LogLoss 0.275757 (732 set, 722 users, review-weighted, paired user bootstrap
2,000, seed 0, 95% CI), unless stated.
Chain: the show-time big model (SM91 step 2) trained for 2 more instant-only epochs on show-time inputs (curve loss
weight 0), with an early check after epoch 1. Epoch 2 was interrupted by a machine reboot at about 21:44 Wed 11-11 and
resumed at 01:02 Thu 11-12.

## Verdict: ceiling above SM98

- Primary: the show-time instant-only big model (SM99) minus the Small Instant-Only model (SM98), instant head,
  show-time inputs: **-0.108% [-0.133, -0.084]** (0.275894 vs 0.276191), 492 better / 230 worse. "Ceiling above SM98"
  needs <= -0.05% with the CI below 0: **met**, so a teacher experiment is worth it.
- Right after it: the show-time instant-only big model (SM99) minus the show-time big model (SM91 step 2), instant head,
  show-time inputs: **-0.105% [-0.118, -0.093]** (0.275894 vs 0.276183), 578 better / 143 worse.
- Versus the big model (1 leak-free epoch) on show-time instant: 0.275894 vs 0.277157 = **-0.456%**.

The final model is epoch 2. The curve head is untrained (weight 0) and is not part of the verdict.

## Read-out (732 set, instant head, %)

| new minus | inputs | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|---|
| the show-time big model (SM91 step 2) | show-time | 0.275894 | 0.276183 | -0.105 [-0.118, -0.093] | 578 / 143 |
| the show-time big model (SM91 step 2) | normal | 0.275748 | 0.276028 | -0.102 [-0.115, -0.090] | 581 / 139 |
| the show-time instant-only small model (SM96) | show-time | 0.275894 | 0.276287 | -0.143 [-0.168, -0.119] | 514 / 207 |
| the show-time retrain of the release (SM92) | show-time | 0.275894 | 0.276876 | -0.356 [-0.392, -0.322] | 630 / 91 |
| the release (SM63) | show-time | 0.275894 | 0.277718 | -0.662 [-0.765, -0.577] | 662 / 59 |
| the Small Instant-Only model (SM98) | normal | 0.275748 | 0.27606 | -0.113 [-0.138, -0.089] | 490 / 230 |
| the big model (1 leak-free epoch) | normal | 0.275748 | 0.275757 | -0.004 [-0.024, +0.016] | 392 / 329 |
| the final model on normal inputs (its show-time cost) | show-time vs normal | 0.275894 | 0.275748 | +0.053 [+0.042, +0.067] | 170 / 534 |
| epoch 1 minus the Small Instant-Only model (SM98) | show-time | 0.276002 | 0.276191 | -0.069 [-0.094, -0.044] | 444 / 277 |
| epoch 2 minus epoch 1 | show-time | 0.275894 | 0.276002 | -0.039 [-0.046, -0.033] | 512 / 208 |

## 1,500 validation users outside the 732 set, show-time inputs (1,487 users with a release reference), % of the release's own LogLoss

| comparison | new | ref | % [95% CI] | better / worse |
|---|---|---|---|---|
| new minus the Small Instant-Only model (SM98) | 0.278406 | 0.279162 | -0.271 [-0.311, -0.235] | 1203 / 284 |
| new minus the show-time big model (SM91 step 2) | 0.278406 | 0.278727 | -0.115 [-0.124, -0.107] | 1240 / 246 |
| new minus the show-time instant-only small model (SM96) | 0.278406 | 0.279249 | -0.302 [-0.340, -0.268] | 1234 / 252 |
| new minus the release (SM63) | 0.278406 | 0.2806 | -0.782 [-0.849, -0.719] | 1413 / 73 |

Row check on the 6 check users: max abs LogLoss difference 0.0 (new and the show-time big model (SM91 step 2)).

## Early check (after epoch 1)

The show-time instant-only big model (SM99)'s epoch 1 minus the Small Instant-Only model (SM98), instant head,
show-time inputs, 722 users: **-0.069% [-0.094, -0.044]**, 444 better / 277 worse (0.276002 vs 0.276191). The
pre-registered rule continues to epoch 2 when the point estimate is below 0, so it said continue. Epoch 1 minus the
show-time big model (SM91 step 2), its start: -0.066% [-0.076, -0.056]. The final model is epoch 2.

## Speed

| step | reviews (last attempt) | wall min (segments) | training min (log) | reviews/s (log) | nvidia-smi max (baseline) | peak allocated / reserved | process tree RSS max |
|---|---|---|---|---|---|---|---|
| epoch 1 | 370,292,439 | 545.1 | 544.5 | 11,334 | 14,593 MiB (482) | 11,060 / 13,020 MiB | 12,974 MiB |
| epoch 2, segment 1 (stopped by the reboot, about 21:44 Wed) | partial | 212.0 | n/a | n/a | 15,008 MiB (1,592) | not logged | not logged |
| epoch 2, segment 2 (resumed 01:02 Thu) | 227,371,580 | 323.7 | 322.9 | 11,737 | 15,600 MiB (3,231) | 10,961 / 13,020 MiB | 9,111 MiB |
| epoch 2, total | | 535.7 | | 11,737 (last attempt) | 15,600 MiB | | |

Segment 1 left no segment line with memory figures (the reboot killed it), so epoch 2's RSS max is the resumed
segment's. Overall: nvidia-smi max 15,600 MiB, RSS max 12,974 MiB (epoch 1), reserved 13,020 MiB.

10-epoch ETA at epoch 2's 11,737 reviews/s (8k tier = 581.94M reviews per epoch, 2k tier = 147.6M):

| tier | per epoch | 10 epochs |
|---|---|---|
| 2k tier | 3.49 h | 35.0 h |
| 8k tier | 13.77 h | 137.7 h |

## Total GPU time

Segments: 545.1 + 212.0 + 323.7 = 1,080.8 min. Launcher tickets (`gpu/tickets.jsonl`, 16 smokes and scoring passes):
93 min. Total about 1,174 min, about 19.6 GPU hours.
