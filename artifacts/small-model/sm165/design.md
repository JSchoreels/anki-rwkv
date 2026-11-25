# Big long cards with 6-bit factors (SM165): design

Pre-registered on 11-25 before any row of this test exists. Branch `sm165`, cut from `origin/sm155` (cdb38316).

## Question

The release format for Big is T24 with 6-bit factors (F6, `w8+lr2.16.4q6x8`). It is the only final format without its
own long-card read-out (cards with 128 or more answers before the review). The bounded-memory long-card scorer
(SM155) gave that read-out for Big at T24 with 8-bit factors (`w8+lr2.16.4q8x8`) on all 50 long-card users: T24 vs
int8 REF (`w8+s8hx8`) on 128+ cards was -0.016% [-0.092, +0.086] for the curve head and +0.026% [-0.084, +0.110] for
the instant head of Big two-head (SM91s2), and +0.006% [-0.105, +0.061] for Big Instant-Only (SM99). How much does F6
add on those cards?

## Branch and code

`origin/sm155` holds both the bounded-memory stream engine (`scripts/stream_scorer.py`, `--engine stream`) and the
`q6` factor label of smaller shift vectors and factors on T24 states (SM152): the sm155 branch was cut from
`origin/sm152`, and neither sm152 (8f4604c7) nor the release-candidate re-check (SM164) changed `scripts/`, `rwkv/`
or `expertium/` after that point (`git diff --stat` empty). No merge was needed and no source file is changed here.
The stream engine was bitwise equal to the stock CPU scorer on `w8+lr2.16.4q6x4` (int6 factors) for Big two-head
(SM91s2) in SM155's exactness run; F6 differs only by keeping the shift vectors at int8 (the T24 shift path).

## Method (as SM155's first use, CPU)

SM148 and SM155 scored long cards on the CPU, the stock scorer for 37 users and the stream engine (bitwise equal) for
the 13 users that did not fit in 6 GB. The GPU is not used: REF and T24 rows are CPU rows, so F6 must be CPU rows too
for a paired comparison.

- Users: the 50 long-card users of `artifacts/small-model/sm148/users-longcard.json` (5,673 to 151,106 reviews,
  3,505,515 in all), largest first.
- Models and label: Big two-head (SM91s2) `sm91s2-big-showtime` and Big Instant-Only (SM99) `sm99-big-inst-e2`
  (`artifacts/small-model/sm148/models.sh`; the launcher checks the checkpoints' sha256), F6 `w8+lr2.16.4q6x8` only.
- Scorer: `scripts/small_model_heldout.py --device cpu --pool 732 --engine stream --chunk-rows 16384 --inputs
  e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets 0,1,3,5,9,16,32,128,300,1000
  --dump-scope all`, `PYTORCH_JIT=0`, `OMP_NUM_THREADS=4` (the scorer sets 2 torch threads), one fresh process per
  user with both models. The stream engine is used for every user (memory is set by the entities, not the reviews).
- 3 processes at a time (SM155 used 2; the users here are at most 151k reviews, about 2 GB per stream process).
  Memory guard 5.9 GB per process (`memguard.ps1`, SM155's copy). quiet.json windows of scope all / cpu respected
  between users. Dumps go to `work/sm165-dump/` (never committed).
- Queue: two tickets with `--claim cpu-heavy` (so speed benchmarks wait), owner `sm165`, purpose prefix
  `Big long cards with 6-bit factors (SM165): `. The GPU memory cap (0.8, then 0.92) does not apply: nothing runs on
  the GPU.

## Gate (must pass before any F6 row is scored)

Ticket 1. Users 2388 and 2883 (long cards), Big two-head (SM91s2), F6 and T24, stock (`--engine parallel`) vs stream,
same flags: every (user, label) cell bitwise equal (`sm155_compare.py`: `4/4 pass; worst max|dp| 0.00e+00`), and the
T24 stream rows equal the reused T24 rows (SM155 gate row for 2388, SM148 CPU row for 2883) on benchmark LogLoss,
RMSE(bins) and AUC of both heads within 1e-6. If the gate fails nothing is scored.

## Read-out (`sm165_report.py`, the statistic of SM148 and SM155)

% = 100 x (sum LogLoss new / sum LogLoss base - 1) over the reviews of each bucket of answers before the review
(1-2, 3-4, 5-8, 9-15, 16-31, 32-127, 128+), review-weighted, paired user bootstrap 2,000 draws, seed 0, 95% CI.
Comparisons per model and head (Big two-head (SM91s2) curve and instant, Big Instant-Only (SM99) instant):

1. F6 vs REF (the long-card cost of the release format against int8);
2. F6 vs T24 (what the 6-bit factors add);
3. T24 vs REF, recomputed from the reused rows: must reproduce SM155's 50-user numbers exactly (a check of the
   merge, not a result).

REF and T24 rows are reused, not rescored: `artifacts/small-model/sm148/cpu/sm148-longcard-rows.jsonl` (37 users) and
`artifacts/small-model/sm155/firstuse/sm155-big-longcard-rows.jsonl` (13 users).

Tolerance (user, 11-22): on 128+ cards a long-card cost of about 0.5% (point) or 1% (95% CI upper bound) vs int8.
F6 vs REF above either is a **warning only**, never a blocker; it does not change the Big F6 go of the release-candidate
re-check (SM164). Coverage target: 50 of 50 users per model; the read-out is pending below 48 (SM148's MIN_LONG).
