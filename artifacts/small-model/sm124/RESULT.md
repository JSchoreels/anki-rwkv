# Result: the show-time two-head cool-down for Tiny (SM124)

Verdict: **go**, held for the next release decision on 11-28 (user 11-14: no new release soon). Nothing here is released and the fork is untouched.

On the 722 users of the 732 set, with show-time inputs, the show-time two-head cool-down for Tiny (SM124) beats the Tiny two-head model (SM106) on both heads. The curve head is -0.103% [-0.142, -0.067] and the instant head is -0.096% [-0.106, -0.087] (paired user bootstrap, 95% CI). It closes 0.244 of the curve gap and 0.223 of the instant gap to the show-time big model (SM91s2). It is still +0.319% [+0.163, +0.483] (curve) and +0.334% [+0.304, +0.365] (instant) behind the show-time big model (SM91s2). LogLoss: curve 0.300369 (was 0.300676), instant 0.277105 (was 0.277369). The 1,500 validation users outside the 732 set (1,487 scored) agree: -0.078% curve and -0.098% instant versus the Tiny two-head model (SM106). Full tables: `sm124-decision.md` and `sm124-decision.json`.

## Stages and speed

| stage | minutes | reviews/s |
|---|---|---|
| reference scoring of the Tiny two-head model (SM106) after 4,000 users | 12 | scoring |
| pass 1 of the cool-down (learning rate 7e-4 to 0.29 x peak), 581.9M reviews | 328.7 | 29,511 |
| pass 2, first segment (16:58 Tue, killed at 18:54 by a Windows power-off event, no reboot) | n/a | not recorded (ticket 001311 removed as stale) |
| pass 2, second segment (relaunched 19:19 from the 2,800-user snapshot, about 6 min of training lost; stopped for the 20:00 break) | 38.4 | n/a (segment) |
| pass 2, third segment (resumed 01:00, ended 04:04 Wed 11-18), 319.7M reviews | 183.2 | 29,083 |
| early-stop scoring plus final scoring (732 set show-time and normal, 1,500 pool) | about 70 in total | scoring |

Both passes ended at the expected learning rates (0.2929 x peak after pass 1, 5.8e-11 x peak after pass 2). The early stop after pass 1 said continue: pass 1 was -0.052% curve and -0.127% instant against the snapshot of the Tiny two-head model (SM106) after 4,000 users.

Launcher: started 11:05 Tue 11-17, ended 04:46 Wed 11-18. Recorded GPU time 617 min over 20 tickets (the killed segment is not in it).
