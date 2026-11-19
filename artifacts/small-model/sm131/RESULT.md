# Compact state for Small and Big (SM131): result

Verdict (ended 17:20 Thu 11-19): **Small: compact state free; Big: compact state free.** Details and CIs in `sm131-decision.md`.

- Compact (`w8+s4c8nhx8`) vs fork default (`w8+s4c8n`), % [95% CI]:
  - the show-time two-head cool-down for Small (SM118): curve +0.000% [-0.001, +0.001], instant +0.001% [+0.000, +0.002]
  - the longer Small Instant-Only model (SM113): instant +0.001% [+0.000, +0.002]
  - the show-time big model (SM91s2): curve -0.000% [-0.001, +0.000], instant +0.000% [-0.000, +0.001]
  - the show-time instant-only big model (SM99): instant +0.000% [-0.000, +0.001]
- Users: Small 722 of 722 rule users; Big 721 of 722 (user 3876: the fork default overflows, non-finite, on both Big models; compact and card-only stay finite, so it counts as missing). Unparseable lines: 0.
- Bytes per card / note: Small 1,963 / 1,567 -> 1,521 / 1,315 (card + note -19.7%); Big 9,271 / 10,283 -> 7,747 / 9,267 (-13.0%).
- Card-only (report only), vs fork default: Small two-head +0.161% curve / +0.198% instant, Small Instant-Only +0.278%, Big two-head +0.122% / +0.155%, Big Instant-Only +0.160%.
- GPU: 2 tickets, 160 minutes (41 + 119, `gpu/tickets.jsonl`).
