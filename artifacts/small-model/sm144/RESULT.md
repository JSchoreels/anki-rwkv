# Earlier int8 switch (SM144): result

Verdict (report decided 12:32 Sun 11-22), pre-registered rules: **Big goes to AN32 (-43.2% state bytes); Small and Tiny stay int8 cards and notes (`w8+s8hx8`).** Details: `sm144-result.md` (per-model tables), `sm144-result.json`; rules and bytes: `design.md`.

## Question

If a card (and a note) switches from int4 to int8 earlier, at 16 or 32 answers instead of 128, does the long-card cost found by the tiered card state by answer count (SM141) disappear while most of the saving stays? No training; the six release-candidate models; the 732 set (723 users, 722 rule users) on the GPU; the 50 long-card users on the CPU, every label scored directly (no dominance shortcut).

Labels: A, A32, A16 = int4 cards below 128 / 32 / 16 answers and int8 from there, notes int8; AN32, AN16 = the same cards switch at 32 / 16 and the notes follow the cards.

## Result under the pre-registered rules (R1 finite, R2 dataset-wide CI upper bound <= +0.05% per head, R3 cards with 128+ answers CI upper bound <= +0.15% per head)

| size | passing schemes | pick | state bytes vs int8 |
|---|---|---|---|
| Big (show-time big model (SM91s2) and show-time instant-only big model (SM99)) | AN32, AN16, A32, A16 | AN32 | -43.2% |
| Small (show-time two-head cool-down for Small (SM118) and longer Small Instant-Only model (SM113)) | none | int8 stays | 0.0% |
| Tiny (six-layer Tiny two-head model (SM130) and six-layer Tiny Instant-Only model (SM130)) | none | int8 stays | 0.0% |

- Big: AN32 has R2 upper bounds +0.025% to +0.030% and R3 upper bounds +0.029% to +0.114%; A (switch at 128, measured directly now) fails R3 (+0.356% and +0.354% upper bounds for the show-time big model (SM91s2)).
- Small: no scheme passes. Every scheme fails R3 on the show-time two-head cool-down for Small (SM118) curve head (AN16: +0.065% [-0.014, +0.386]; AN32: +0.230% [-0.021, +1.381]); several also fail on the longer Small Instant-Only model (SM113) instant head (AN16 upper bound +0.073%).
- Tiny: no scheme passes under R3 either (AN16 curve head of the six-layer Tiny two-head model (SM130): +0.047% [-0.006, +0.299]); AN32 also fails R2 on the six-layer Tiny Instant-Only model (SM130): +0.049% [+0.044, +0.055].

## Where the cost sits (answer-count buckets; GPU all users, CPU long-card users)

- Switch at 128 (A): the cost grows with age and peaks in the long buckets: 32-127 answers +0.07% to +0.17% and 128+ answers +0.09% to +0.33% (GPU, instant and curve heads over the six models).
- Switch at 32 (A32, AN32): 128+ answers is back near 0 (GPU point estimates -0.007% to +0.066%); the cost moves to 16-31 answers (+0.03% to +0.13%, largest for AN32 on the six-layer Tiny Instant-Only model (SM130)) and 32-127 answers (+0.02% to +0.08%).
- Switch at 16 (A16, AN16): 16-31 answers +0.01% to +0.05%, 32-127 answers and 128+ about 0 (+0.00% to +0.02%). So the earlier switch removes the long-card cost on the dataset-wide statistic; what remains is a small cost in the 5-31 answer buckets (+0.01% to +0.08%).
- The only large long-card outliers left are the upper bounds on the show-time two-head cool-down for Small (SM118) curve head (+0.39% to +1.38%) and the six-layer Tiny two-head model (SM130) curve head (+0.21% to +0.34%), with 50 users and wide intervals.

## Run facts

- Smoke ticket passed (75 checks, GPU vs CPU within tolerance). Tickets (`gpu/tickets.jsonl`): Tiny 49 + 1 min, Small 88 + 2 min, Big 132 + 1 min (about 4.6 GPU hours, 01:11 to 05:43). CPU long-card chain 01:10 to 12:32 (11.4 h wall, 2 processes).
- Coverage: 722 GPU users per model (no non-finite user); long-card CPU users 50 of 50 for Tiny and Small, 49 of 50 for Big (user 1714 killed by the memory guard at 5.97 GB, the 48-user minimum is met).
- The int8 reference (REF) reproduces the SM141 GPU rows exactly (max |difference| 0.0 in benchmark LogLoss, 723 users, every model). GPU vs CPU on the 128+ bucket: mean LogLoss differs by at most 0.0015 (show-time two-head cool-down for Small (SM118)), 0.0005 to 0.0009 for the others, so the CPU is the reference for R3 as registered.

## Re-read under the user's 11-22 long-card tolerance (not pre-registered)

After the design was registered, the user relaxed the long-card rule on 11-22: "If really long cards have to somewhat lose 0.5 or 1% precision but just them, it's ok." Our reading: R3 becomes point estimate <= +0.5% and 95% CI upper bound <= +1.0% per head; R2 (upper bound <= +0.05%), R1 and the byte order are unchanged. Recomputed from `sm144-result.json` (the table below is not the decision rule that was registered):

| size | passing schemes | pick | state bytes vs int8 | what fails |
|---|---|---|---|---|
| Big | AN32, AN16, A, A32, A16 | AN32 | -43.2% | nothing |
| Small | AN16, A, A32, A16 (not AN32) | AN16 | -36.3% | AN32 fails R3 on the show-time two-head cool-down for Small (SM118) curve head (upper bound +1.381%); A fails there too (+1.489%) |
| Tiny | AN16, A, A32, A16 (not AN32) | AN16 | -35.3% | AN32 fails R2 on the six-layer Tiny Instant-Only model (SM130): +0.049% [+0.044, +0.055], upper bound over +0.05% (AN16: +0.030% [+0.027, +0.033]) |

- Small, AN16 on the show-time two-head cool-down for Small (SM118) curve head: +0.065% [-0.014, +0.386] on the long cards, inside the new tolerance (the old +0.15% bound is what it missed).
- The bucket tables give bucket means only, not per-card worst cases, so they do not show whether a few individual long cards are badly off.

## For the fork

Under the pre-registered rules only Big changes (AN32: int4 cards and notes below 32 answers, int8 from there). If the user's 11-22 tolerance is adopted, Small and Tiny can take AN16 (int4 below 16 answers, int8 from there, notes follow the cards): -36.3% and -35.3% state bytes. That is the user's decision; the long-card tolerance was not part of the registered design.
