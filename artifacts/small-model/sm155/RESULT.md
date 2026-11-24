# Bounded-memory long-card scorer (SM155): result

## Exactness (pre-registered rule): PASS

All 264 (model, user, label) cells pass: complete rows from both engines, worst max |p_stream - p_stock| 1.19e-07 over
every predicted review of both heads (limit 1e-5), benchmark metrics equal. Details per cell: `sm155-exactness.txt`.
Peak working set on user 1458 (92,307 reviews) for Big two-head (SM91s2) at REF: stock engine 5.25 GB, stream engine
1.40 GB. In the first use the memory guard log (10 s samples) shows a highest value of 3.67 GB while the largest user
(3735, 531,091 reviews) ran, against the guard limit of 5.9 GB.

## First use (11-24, 01:29 to 05:02, 2 processes at a time)

All 18 missing Big long-card users (13 with 87k to 151k reviews, 5 that never fit on the GPU with 424k to 531k), each
for Big two-head (SM91s2) and Big Instant-Only (SM99), at T24 (`w8+lr2.16.4q8x8`) and REF (`w8+s8hx8`), `--engine
stream`, chunk rows 16384. The bitwise gate (user 2388, REF and T24, stock vs stream) passed with max |dp| 0. Rows:
`firstuse/sm155-big-longcard-rows.jsonl` (72 rows); read-out `firstuse/sm155_firstuse_report.py` ->
`firstuse/sm155-firstuse-result.{json,md}`.

Known launcher bug: after the last user the launcher hung in its final `wait`, because that `wait` also waited for its
own memory-guard process, which only exits on `firstuse/memguard.off`, written after the `wait`. The main session wrote
`memguard.off` by hand at 06:44 and the launcher then finished (`firstuse/done`). Fix (not applied; the launcher is
untouched): `wait "${PIDS[@]}"`, or touch `memguard.off` before waiting.

### Speed and CPU

Measured per process, review-evaluations per second (reviews x 4 labels per user process, from the log lines): 15.73M
review-evaluations in 386.6 process-minutes, so 678 review-evals/s per process on average (per user 476 to 821; the
larger the user, the faster: the 13 users 476 to 720, the 5 large users 687 to 821). From the rows' model seconds (383.8
process-minutes): Big two-head (SM91s2) REF 708, T24 621; Big Instant-Only (SM99) REF 754, T24 664 review-evals/s. T24 costs
about 12% more time than REF. Wall time was 213 min for the 18 users (2 processes), against the 180 min forecast of about 3 h.
The stock CPU scorer ran at about 900 reviews/s per process (low-rank int8 states test (SM148)), so the stream engine
ran at about 0.75 times that, as forecast.

### Long-card read-out (reviews of cards with 128+ earlier answers, 50 long-card users; warning only)

Same statistic as the low-rank int8 states test (SM148): % = 100 x (T24 LogLoss / REF LogLoss - 1), review-weighted,
paired user bootstrap 2,000 draws, 95% CI. Coverage (users per label of 50): Big two-head (SM91s2) REF and T24 37 -> 50;
Big Instant-Only (SM99) REF and T24 37 -> 50. The 128+ bucket grows from 15,173 to 30,089 reviews.

| model, head | before (37 users): point [CI] | after (50 users): point [CI] |
|---|---|---|
| Big two-head (SM91s2), curve | -0.007% [-0.140, +0.167] | -0.016% [-0.092, +0.086] |
| Big two-head (SM91s2), instant | +0.011% [-0.133, +0.219] | +0.026% [-0.084, +0.110] |
| Big Instant-Only (SM99), instant | -0.011% [-0.152, +0.168] | +0.006% [-0.105, +0.061] |

By answer-count bucket (after, T24 vs REF, point [CI], all 50 users): see `sm155-firstuse-result.md`. The picture does not
change: every 128+ upper bound stays far below the long-card tolerance (point 0.5%, CI upper 1%), and the CI narrows by
about a third to a half. The Big budget of +0.15% vs its own int8 applies to the whole 732 set, not to long cards.
