# Long-card state drift on real users (SM136): result

Verdict: **int8 card rows recommended (amended coverage).** The fork should store int8 card rows with compact notes (format w8+s8hx8) for the 11-28 release candidate, for Big, Small and Tiny.

## Question

Do int4 card rows cost LogLoss on reviews of cards with many prior answers, compared with the published int8 format (w8+s8)? Rule: int4 is acceptable if no user goes non-finite and every head's 95% CI upper bound is at most +0.25% in the 128-299 and 128+ prior-answer buckets. The benchmark itself cannot see these cards (it stops at 127 prior answers), so the scorer predicts every review and groups by prior answers.

## Coverage and the amendment

Scoring ran on the CPU (the GPU smoke test disagreed on long cards). 51 of the 66 pre-registered users were scored per model (50 for the two Tiny models). The 15-16 longest users per size (186k to 531k reviews) were killed by the 5.9 GB memory guard. The coverage rule was therefore amended on 11-21 from "at least 64 of 66 users" to "the complete users", with the user's "continue" after the recommendation. The missing users are the longest, where drift is largest, so they can only make int4 look worse. What this cannot rule out: a pass of int4 on the complete users would not have covered them. It was not needed, because int4 already fails on the complete users.

## Memory finding

Peak memory comes from one evaluation of the whole review history. It grows with the review count (about 3.2 GB at 102,545 reviews, 5.6 GB at 150,500) and does not depend on the format, the label split, the thread count or row chunking. The killed users would need about 6-20 GB each. This PC has 31 GB total and about 18 GB free, so they cannot run here as the code stands.

Open option (not needed for the decision): an exact time-chunked CPU layer that carries the token-shift and recurrence state between chunks, checked bit for bit against the unchunked run on short users, to finish the 15-16 users per size for the record. A machine with about 24 GB free would also do.

## Numbers (CI upper bound, % vs w8+s8, complete users; worst of the two buckets and both heads)

| model | compact int4 (w8+s4c8nhx8) | fork default int4 (w8+s4c8n) | int8 cards, compact notes (w8+s8hx8) |
|---|---|---|---|
| the show-time big model (SM91s2) | +1.09 (fail) | +0.84 (fail) | +0.03 |
| the show-time instant-only big model (SM99) | +0.50 (fail) | +0.53 (fail) | +0.01 |
| the show-time two-head cool-down for Small (SM118) | +3.77 (fail) | +0.32 (fail) | +0.11 |
| the longer Small Instant-Only model (SM113) | +0.46 (fail) | +0.21 (meets ceiling, but non-finite user) | +0.01 |
| the show-time two-head cool-down for Tiny (SM124) | +4.55 (fail) | +0.44 (fail) | +0.01 |
| the longer cool-down for Tiny (SM115) | +0.30 (fail) | +0.34 (fail) | +0.02 |

- Compact int4 fails for every size. Examples of point estimates in the 128+ bucket: Small two-head curve +0.74%, Tiny two-head curve +0.65%.
- The fork default int4 fails for Big and Tiny, and for Small two-head. Only the longer Small Instant-Only model (SM113) meets the ceiling, and it still has the non-finite user.
- One user (the one with a card of thousands of updates) is non-finite for the fork default in five of six models (not the longer cool-down for Tiny (SM115)); compact and int8-card formats stay finite.
- Int8 cards with compact notes stay within +0.12% everywhere, so the release should use int8 card rows. Bytes per card: Tiny 1,315 (compact int4: 803), Small 2,545 (1,521), Big 13,891 (7,747).

## Files

`sm136-decision.json` (per-model numbers, coverage, flag amended), `sm136-interim.json`, `design.md` (amendment 11-21), `cpu/` run logs, `HANDOFF.md`. No raw review data.
