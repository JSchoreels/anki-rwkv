# Release-candidate checks for the family (SM139): result

Summary (11-21 16:15, status pass): all six release candidates are finite for every user (C1), the format cost of w8+s8hx8 against w8+s8 is at most +0.002% on every head, on the 732 set and on the extension, far below the +0.05% ceiling (C2), and coverage holds (C3: 722 users on the 732 set plus 761 to 764 extension users; the extension is 771 users with 6 check users, a few never-fit users are missing for every label). What users get, report only: Big is unchanged (the show-time big model (SM91s2) and the show-time instant-only big model (SM99): format only, +0.000%). Small gets slightly better: the show-time two-head cool-down for Small (SM118) against the show-time retrain of the release (SM92) is -0.086% curve and -0.095% instant, and the longer Small Instant-Only model (SM113) against the Small Instant-Only model (SM98) is -0.091%. Tiny costs more: the six-layer Tiny two-head model (SM130) against the Tiny two-head model (SM106) is +0.264% curve [+0.218, +0.309] and +0.242% instant, and the six-layer Tiny Instant-Only model (SM130) against the Tiny Instant-Only model (SM101) is +0.269%; this is the known price of the faster six-layer Tiny chosen on 11-21 (the fork bench showed the pair replaying 39% faster). Time heads were not rescored (design.md). 271 GPU minutes over 19 queue tickets.

Status: **pass** (C1 every user finite at the release-candidate format, C2 format cost CI upper bound <= +0.05% per trained head on the 732 set and on the extension, C3 coverage; design.md).

Users: the 732 set = 722 rule users; the extension = 765 validation users of the 1,500 pool outside the 732 set (ids >= 5000: never seen by the Small and Tiny models, but training users of the two Big models, so Big's extension numbers are a format and robustness check only); all = both. Show-time inputs, leak-free, w8 weights; % of the reference's own LogLoss, paired user bootstrap 95% CI.

## Checks per release-candidate model (format cost = w8+s8hx8 vs the same model at w8+s8)

| model | head | finite (C1) | cost on the 732 set | cost on the extension | cost on all | users 732 + ext (C3) | C2 |
|---|---|---|---|---|---|---|---|
| the show-time big model (SM91s2) | curve | True | +0.001% [-0.000, +0.001] | -0.000% [-0.000, +0.000] | +0.000% [-0.000, +0.001] | 722 + 764 | pass |
| the show-time big model (SM91s2) | instant | True | +0.000% [+0.000, +0.000] | +0.000% [+0.000, +0.000] | +0.000% [+0.000, +0.000] | 722 + 764 | pass |
| the show-time instant-only big model (SM99) | instant | True | +0.000% [-0.000, +0.000] | +0.000% [-0.000, +0.000] | +0.000% [+0.000, +0.000] | 722 + 764 | pass |
| the show-time two-head cool-down for Small (SM118) | curve | True | +0.000% [-0.000, +0.000] | -0.000% [-0.000, +0.000] | -0.000% [-0.000, +0.000] | 722 + 761 | pass |
| the show-time two-head cool-down for Small (SM118) | instant | True | +0.000% [+0.000, +0.001] | +0.001% [+0.000, +0.001] | +0.001% [+0.000, +0.001] | 722 + 761 | pass |
| the longer Small Instant-Only model (SM113) | instant | True | +0.001% [+0.001, +0.001] | +0.001% [+0.001, +0.002] | +0.001% [+0.001, +0.001] | 722 + 761 | pass |
| the six-layer Tiny two-head model (SM130) | curve | True | +0.000% [+0.000, +0.001] | +0.000% [-0.000, +0.000] | +0.000% [+0.000, +0.000] | 722 + 761 | pass |
| the six-layer Tiny two-head model (SM130) | instant | True | +0.000% [+0.000, +0.001] | +0.001% [+0.000, +0.001] | +0.001% [+0.000, +0.001] | 722 + 761 | pass |
| the six-layer Tiny Instant-Only model (SM130) | instant | True | +0.001% [+0.001, +0.002] | +0.001% [+0.001, +0.002] | +0.001% [+0.001, +0.002] | 722 + 761 | pass |

## What users get from the update (report only): the candidate at w8+s8hx8 vs the published 11-13 model at w8+s8

| size | kind | candidate | published | head | 732 set | extension | all | LogLoss all (candidate / published) | model only, all (both w8+s8) |
|---|---|---|---|---|---|---|---|---|---|
| Big | two-head | the show-time big model (SM91s2) | the show-time big model (SM91s2) (same model: format only) | curve | +0.001% [-0.000, +0.001] | -0.000% [-0.000, +0.000] | +0.000% [-0.000, +0.001] | 0.296149 / 0.296149 | +0.000% [+0.000, +0.000] |
| Big | two-head | the show-time big model (SM91s2) | the show-time big model (SM91s2) (same model: format only) | instant | +0.000% [+0.000, +0.000] | +0.000% [+0.000, +0.000] | +0.000% [+0.000, +0.000] | 0.274684 / 0.274683 | +0.000% [+0.000, +0.000] |
| Big | Instant-Only | the show-time instant-only big model (SM99) | the show-time instant-only big model (SM99) (same model: format only) | instant | +0.000% [-0.000, +0.000] | +0.000% [-0.000, +0.000] | +0.000% [+0.000, +0.000] | 0.274354 / 0.274354 | +0.000% [+0.000, +0.000] |
| Small | two-head | the show-time two-head cool-down for Small (SM118) | the show-time retrain of the release (SM92) | curve | -0.113% [-0.176, -0.062] | -0.065% [-0.089, -0.039] | -0.086% [-0.116, -0.060] | 0.301229 / 0.30149 | -0.086% [-0.116, -0.060] |
| Small | two-head | the show-time two-head cool-down for Small (SM118) | the show-time retrain of the release (SM92) | instant | -0.099% [-0.109, -0.090] | -0.092% [-0.102, -0.082] | -0.095% [-0.102, -0.088] | 0.278347 / 0.278612 | -0.096% [-0.102, -0.089] |
| Small | Instant-Only | the longer Small Instant-Only model (SM113) | the Small Instant-Only model (SM98) | instant | -0.093% [-0.102, -0.084] | -0.090% [-0.099, -0.082] | -0.091% [-0.097, -0.085] | 0.277683 / 0.277937 | -0.093% [-0.099, -0.087] |
| Tiny | two-head | the six-layer Tiny two-head model (SM130) | the Tiny two-head model (SM106) | curve | +0.246% [+0.154, +0.332] | +0.278% [+0.234, +0.325] | +0.264% [+0.218, +0.309] | 0.302629 / 0.301832 | +0.264% [+0.218, +0.309] |
| Tiny | two-head | the six-layer Tiny two-head model (SM130) | the Tiny two-head model (SM106) | instant | +0.245% [+0.219, +0.275] | +0.239% [+0.216, +0.265] | +0.242% [+0.224, +0.261] | 0.279781 / 0.279105 | +0.241% [+0.223, +0.261] |
| Tiny | Instant-Only | the six-layer Tiny Instant-Only model (SM130) | the Tiny Instant-Only model (SM101) | instant | +0.280% [+0.254, +0.310] | +0.260% [+0.237, +0.284] | +0.269% [+0.252, +0.287] | 0.279206 / 0.278458 | +0.267% [+0.251, +0.286] |

## Row checks (report only)

- The 6 check users, extension ticket vs 732 ticket, max |LogLoss difference| per label: sm91s2-big-showtime/w8+s8 0.0e+00, sm91s2-big-showtime/w8+s8hx8 0.0e+00, sm99-big-inst-e2/w8+s8 0.0e+00, sm99-big-inst-e2/w8+s8hx8 0.0e+00, sm118-e26-dec2ep32k-t91s2/w8+s8 0.0e+00, sm118-e26-dec2ep32k-t91s2/w8+s8hx8 0.0e+00, sm113-inst-e35st-st-dec2ep32k-t99/w8+s8 0.0e+00, sm113-inst-e35st-st-dec2ep32k-t99/w8+s8hx8 0.0e+00, sm92-e26-dec1ep32k/w8+s8 0.0e+00, sm98-inst-e35st-st-dec1ep32k/w8+s8 0.0e+00, sm130-th-dec2ep32k-t91s2/w8+s8 0.0e+00, sm130-th-dec2ep32k-t91s2/w8+s8hx8 0.0e+00, sm130-io-dec2ep32k-t99/w8+s8 0.0e+00, sm130-io-dec2ep32k-t99/w8+s8hx8 0.0e+00, sm106-c1-e29st-st-dec1ep32k/w8+s8 0.0e+00, sm101-c1-e38st-st-dec1ep32k/w8+s8 0.0e+00
- This tree vs recorded rows of earlier runs (same label, common users): sm91s2-big-showtime/w8+s8 729 users max 0.0e+00 (sm91s2-inputs.jsonl), sm118-e26-dec2ep32k-t91s2/w8+s8 729 users max 0.0e+00 (sm118-rows.jsonl), sm113-inst-e35st-st-dec2ep32k-t99/w8+s8 729 users max 0.0e+00 (sm113-rows.jsonl), sm130-th-dec2ep32k-t91s2/w8+s8 729 users max 0.0e+00 (sm130-rows.jsonl), sm130-io-dec2ep32k-t99/w8+s8 729 users max 0.0e+00 (sm130-rows.jsonl), sm130-io-dec2ep32k-t99/w8+s8hx8 729 users max 0.0e+00 (sm137-rows.jsonl), sm92-e26-dec1ep32k/w8+s8 729 users max 0.0e+00 (sm92-inputs.jsonl), sm106-c1-e29st-st-dec1ep32k/w8+s8 1502 users max 0.0e+00 (sm106-rows-1500.jsonl/sm106-rows.jsonl)
