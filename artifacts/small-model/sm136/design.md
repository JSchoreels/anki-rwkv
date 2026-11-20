# Long-card state drift on real users (SM136): design (2026-11-20, pre-registered before any GPU ticket)

## Question

The fork overflow check (SM133, `artifacts/small-model/sm133-fork-compact/check-and-scope.md`, sections 2-3 and 7) found
on synthetic single-card histories that the fork's default stored state `w8+s4c8n` drifts and grows geometrically.
That format has int4 card rows, float32 abs-max / 7 scales and int8 note rows, and it is re-quantized after every
answer. On those cards it eventually overflows on every size, Big first. After about 200-500 answers on a card, the
mean |Δr| vs float32 state reaches 0.02-0.06, while int8 card rows stay at 0.002-0.015. The compact format
`w8+s4c8nhx8` (float16 scales, int8 shifts) stays finite, because it saturates at the float16 maximum, but it keeps
the drift.

The real-user evidence so far cannot see this:
- The compact state for Small and Big (SM131) compared compact with the default, and both formats share the drift.
- Both are about +0.015% vs the published `w8+s8` on the whole 722-user pool.
- One user (3876, one card with 4,733 updates) overflowed the default on both Big models.
- The older int4 drift check (SM78) measured +0.129% on the instant head in the 32-63-review bucket (smaller model
  (SM72s), 110 users).

**For the 11-28 release candidate: on real users, how much LogLoss do int4 card rows (default and compact) lose against
int8 card rows (`w8+s8`), and against float32 state, on reviews of cards with many prior answers?** The answer decides
whether the fork should store int8 card rows (all sizes or Big only) instead of int4.

## A finding that shapes the readout: the benchmark reviews stop at 127 prior answers

The benchmark test reviews of every earlier run (the `benchmark` LogLoss) never include a review of a card with 128 or
more prior answers. Counted from the label DB and the revlogs before any scoring, on the 723 users, the largest number
of prior answers of any benchmark review is 127, for 64 users. The FSRS benchmark cuts card histories there. So
**the standard LogLoss cannot see long cards at all.** This run therefore buckets **every review the scorer predicts**
(the "all" scope: every revlog row with at least one prior answer of its card, both heads). That is the same set as the
scorer's own `curve_log_loss` / `instant_log_loss`. The benchmark-scope buckets are kept as well, as a check and for
continuity.

## Models (one per size and head chain, the family candidates; read only)

| plain name (ID) | size | kind / arch | heads | checkpoint (sha256 prefix) |
|---|---|---|---|---|
| the show-time big model (SM91s2) | Big | fork | curve, instant | `wt/sm91s2/.../big-showtime/master.pth` (67ee8e9b) |
| the show-time instant-only big model (SM99) | Big | fork | instant | `wt/sm99/.../big-inst/master.pth` (c49e45d6) |
| the show-time two-head cool-down for Small (SM118) | Small | `expertium:w64` | curve, instant | `wt/sm118/...-p2/.../master.pth` (b4fe1175) |
| the longer Small Instant-Only model (SM113) | Small | `expertium:w64` | instant | `wt/sm113/...-p2/.../master.pth` (e4574e2a) |
| the show-time two-head cool-down for Tiny (SM124) | Tiny | `expertium:w64-c1` | curve, instant | `wt/sm124/...-p2/.../master.pth` (c78b24d8) |
| the longer cool-down for Tiny (SM115) | Tiny | `expertium:w64-c1` | instant | `wt/sm115/...-p2/.../master.pth` (c59d7049) |

The full paths and hashes are in `launch-sm136.sh`. They are the same checkpoints that the compact state for Tiny (SM129)
and for Small and Big (SM131) used.

## Formats (no training; w8 weights; scorer `scripts/small_model_heldout.py` of this tree = `origin/sm131` + the flag below)

| format | variant | card rows | note rows | shift vectors | role |
|---|---|---|---|---|---|
| int8 everywhere (published) | `w8+s8` | int8, float32 scales | int8, float32 scales | float16 (Big) / float (Small, Tiny) | **reference** |
| fork default | `w8+s4c8n` | int4, float32 scales | int8, float32 scales | as above | rule |
| compact (release candidate) | `w8+s4c8nhx8` | int4, float16 scales | int8, float16 scales | int8 + one float16 scale | **rule, decides** |
| int8 cards, compact blocks | `w8+s8hx8` | int8, float16 scales | int8, float16 scales | int8 + one float16 scale | report only (the int8 option) |
| float32 state | `w8` | float32 | float32 | float32 | report only |

The scorer's grammar already expresses every format:
- `s8hx8` = `s8` with `h` (float16 row scales) and `x8` (int8 shift vectors) on cards and notes (SM47 / SM49 syntax);
- `w8` alone sets state bits 0, so no state rounding at all, on both the `fork` and the `expertium` scorer paths.

Bytes per card / note (`sm136_bytes.py` -> `sm136-bytes.json`): SM131's reference encoder of the fork layout, plus the
`w8+s8hx8` arm. It reproduces all known fork totals.

| size | w8+s8 | fork default w8+s4c8n | compact w8+s4c8nhx8 | int8 cards, compact w8+s8hx8 |
|---|---:|---:|---:|---:|
| Tiny | 1,567 / 1,567 | 1,055 / 1,567 | 803 / 1,315 | 1,315 / 1,315 |
| Small | 2,987 / 1,567 | 1,963 / 1,567 | 1,521 / 1,315 | 2,545 / 1,315 |
| Big | 15,415 / 10,283 | 9,271 / 10,283 | 7,747 / 9,267 | 13,891 / 9,267 |

## Scorer change: `--prior-buckets` (opt-in, default output bit for bit unchanged)

`--prior-buckets 0,32,128,300,1000` adds a row field `prior_buckets`:
- for both scopes (`benchmark`, `all`) and both heads (`ahead`, `instant`), it holds the review count `n` and the
  LogLoss sum `ll` per bucket of the card's prior answers, plus the min and max prior answers;
- a card's prior answers for a review = the number of earlier revlog rows of the same card. This uses the frame order
  of `get_rwkv_data` (`review_th` = 1 + the parquet row index), so it equals the number of state updates the card
  stream has applied before that review;
- the LogLoss is sklearn `log_loss(..., normalize=False)` on the same kept reviews as the benchmark, so the benchmark
  scope adds up to the benchmark LogLoss;
- without the flag, nothing changes. The code is additive only (64 added lines, 0 removed in
  `scripts/small_model_heldout.py`; nothing else under `scripts/` changes).

## Users (fixed before scoring by `sm136_pool.py`, from revlog card ids only -> `pool-sm136.json`, aggregates only)

| pool | rule | users | revlog rows | predicted reviews per bucket 0-31 / 32-127 / 128-299 / 300-999 / 1000+ | users per bucket |
|---|---|---:|---:|---|---|
| **main (rule)** | the 723 fit users of the 732 set with a card of >= 128 prior answers | **66** | 8,414,821 | 6,256,630 / 1,485,011 / 78,555 / 2,723 / 5,997 | 66 / 66 / 66 / 6 / 1 |
| extension (report only) | validation users of the 8k split with id >= 5000, outside the 732 set, with a card of >= 300 prior answers | 10 | 2,775,665 | 1,596,737 / 862,993 / 188,835 / 27,670 / 1,021 | 10 / 10 / 10 / 10 / 2 |

- Real users rarely have long cards. Only 6 of the 723 users have any card with 300 or more prior answers, and only
  one user (3876) has 1,000 or more. So the 300+ buckets cannot reach 20 users in the 732 set.
- The extension adds the users of the larger validation pool (`--pool 1500`, minus the 732 set) that have such cards.
  The Small and Tiny models never saw them, but they are training users of the big models. The never-fit users of the
  732 set are excluded, as in every 732 run.
- The extension is report only. It brings the 300+ bucket to 16 users and about 37,000 reviews (main + extension).
- Inputs: show-time (`--inputs e2s-query`), leak-free (`--split-existing --deleted-cards neighbour`),
  `--nonfinite-rows` (a non-finite evaluation is a `nonfinite` row, done and not retried). GPU cap 0.8 (0.92 only in a
  retry).

## Statistic and pre-registered rule (`sm136_rule.py`, selftest passes now)

- Per model, format, head and bucket (0-31, 32-127, 128-299, 300-999, 1000+, pooled 128+ and 300+, and all), on the
  "all" scope: % = 100 x (sum of the format's LogLoss / sum of w8+s8's LogLoss - 1) over the bucket's reviews. This is
  review-weighted.
- CI: paired user bootstrap over the users with reviews in the bucket, 2,000 draws, seed 0, 95% CI.
- The bucket counts must be identical across formats (they depend only on the inputs). Otherwise `decide` fails
  (exit 3).
- **Rule buckets:** the buckets with >= 128 prior answers that have >= 20 main-pool users. These are the candidates
  128-299, 300-999, 1000+, 300+ and 128+. By the pool table above, the rule buckets will be **128-299 and 128+**
  (66 users each). 300-999, 300+ and 1000+ are report only.
- **acceptable(model, int4 format)** = both of:
  - no main-pool user has a non-finite row for the format (an overflowing card: the fork errors or panics);
  - in every rule bucket, every head's 95% CI upper bound <= **+0.25%** (two-head: curve and instant; Instant-Only:
    instant).
- **Int4 card rows acceptable for a size** = both of its models acceptable. Otherwise: **recommend int8 card rows for
  that size.**
- The release decision follows the **compact** format (approved for the 11-28 release candidate). The fork default's
  verdict is reported too. It already fails for Big if user 3876 is non-finite again.
- Coverage: at least 64 of the 66 main-pool users must have a row (finite or non-finite) for every model and format.
  Otherwise `decide` exits 2 (pending, a `STOP` line) and the main session decides.
- Reported, no verdict:
  - every format vs w8+s8 and vs float32, per bucket and head;
  - the benchmark-scope buckets (only 0-31 and 32-127 can have reviews);
  - the non-finite users per format;
  - the extension pool, and the combined main + extension 300+ bucket;
  - w8+s8 vs the published GPU rows (max |difference|, an integrity check);
  - the bytes;
  - the whole-pool context from the compact states for Tiny (SM129) and for Small and Big (SM131): compact and default
    vs w8+s8 on 722 users, benchmark reviews.
- Selftest (synthetic rows, CPU):
  - identical formats give 0% and acceptable everywhere;
  - the rule buckets are 128-299 and 128+;
  - a hand-computed two-user statistic gives 25%;
  - compact x 1.004 on 128-299 for the show-time instant-only big model (SM99) gives int8 for Big only;
  - x 1.01 on the report-only 300-999 bucket decides nothing;
  - a non-finite default user fails the default for Big only;
  - curve +0.3% on 128+ for the show-time two-head cool-down for Tiny (SM124) gives int8 for Tiny;
  - 3 missing users give pending and 2 give a decision on 64 users;
  - a bucket-count mismatch raises an error;
  - torn and NUL-filled lines are skipped and counted.

Why +0.25%: the published family release accepts per-size accuracy steps of a few tenths of a percent. A long-card cost
below +0.25% even at the CI upper bound is smaller than any per-size decision this track makes. The worst bucket SM78
found was +0.129%.

## Proofs

CPU (`sm136_check.py` -> `check-cpu-sm136.json`; the launcher repeats it under the memory guard, 5.9 GB per CPU process):
- **K1 bytes:** the table above, the round trip and the known totals.
- **K2 pool:** `sm136_pool.py --check` reproduces the committed pools and counts.
- **K3 formats:** the five variants map to the intended state bits, and `w8` sets no state rounding.
- **K4 default output unchanged:**
  - a. the scorer diff against `origin/sm131` is additive only;
  - b. without the flag, this tree's CPU rows of the show-time big model (SM91s2) for users 2388 and 4676 at SM131's
    four formats equal SM131's committed CPU rows bit for bit (every field except timing and provenance);
  - c. with the flag, the rows equal the rows without it plus `prior_buckets`. This is checked on the fork path (the
    show-time big model (SM91s2)) and the `expertium` scorer path (the show-time two-head cool-down for Tiny (SM124)), users
    2388 and 1866.
- **K5 bucket sums:**
  - benchmark scope: the counts add up to the benchmark size, and sum(ll) / sum(n) equals the benchmark LogLoss within
    its 6-decimal rounding (<= 5.1e-7); every benchmark review has 1-127 prior answers;
  - "all" scope: the counts add up to the scorer's curve / instant label counts, and the mean equals `curve_log_loss` /
    `instant_log_loss` within 1e-4.
- **K6 CPU smoke rows** (`cpu/smoke-cpu-rows.jsonl`): users 1866 and 2883, the two smallest main-pool users (cards up to
  147 and 138 prior answers), all 30 labels on CPU:
  - every row is finite;
  - w8+s8 is within 1e-4 of the published GPU rows;
  - the counts are equal across formats;
  - each int4 format differs from w8+s8.
- **K7 rule selftest.**

GPU smoke (the first step of the GPU ticket; a mismatch = `STOP`): the same 2 users and 30 labels on the GPU vs the CPU
rows:
- the bucket counts are identical;
- the benchmark LogLoss is within 1e-5 for w8+s8 (also vs the published rows) and within 1e-4 for the other formats;
- the "all"-scope bucket means (buckets with >= 20 reviews) are within 1e-4.

## Cost and ETA

| step | GPU | ends about |
|---|---|---|
| CPU proofs (about 10 min, memory guard), commit, push, launch | | ~03:00 Fri 20 Nov |
| wait in the queue behind the six-layer Tiny with longer recovery (SM135)'s Instant-Only epoch 42 (started 02:01, expected 70 min) | | ~03:15 Fri |
| ONE chained ticket: GPU smoke (~5 min); Big 10 labels (~30 min); Small 10 (~18 min); Tiny 10 (~18 min); extension 30 labels (~22 min); in-ticket retries | ~1.5 h (1.2-2 h) | ~04:45 Fri |
| retry tickets only if main-pool users are still missing (each waits behind one SM135 ticket) | minutes | (+~1.5 h if needed) |
| rule `decide` (CPU, seconds) | | **~04:45 Friday 20 November 2026** (range 04:15-06:30) |

The estimates come from the model seconds in SM131's and SM129's rows on these 66 users: per label, Big 2.0 min, Small
1.2 min and Tiny 1.2 min of model time; the wall time is about 1.5x the model time. **Total about 1.5 GPU h**, inside the
4 GPU h budget. The full 723-user pool would cost about 4.7 GPU h for these 30 labels and adds nothing for the long
buckets. Its whole-pool numbers for the default and compact formats already exist (SM129, SM131).

## Deviations from the brief

1. **Buckets on every predicted review, not the benchmark reviews.** The benchmark set stops at 127 prior answers, so
   it cannot answer the question (above). The benchmark scope is reported too.
2. **Rule buckets 128-299 and 128+, not 300+.** Only 6 of the 723 users have a card with 300 or more prior answers, and
   1 has 1,000 or more. The suggested rule ("every bucket with >= 300 prior answers with >= 20 users") would test
   nothing. The rule keeps the brief's threshold (+0.25%) and its >= 20-user condition, and applies them from 128 prior
   answers up. The 300+ buckets are reported, with a CI where there are at least 2 users, including the extension.
3. **Restricted pool:** the 66 users with any card with >= 128 prior answers (fixed before scoring), not all 723. Users
   without such cards contribute nothing to the rule buckets.
4. **Extension pool (report only):** 10 validation users with id >= 5000 and a card with >= 300 prior answers (big-model
   training users, unseen by Small and Tiny).
5. **The int8-card + compact-notes variant is `w8+s8hx8`:** int8 card **and** note rows with the compact blocks. The
   grammar cannot mix `h` on one stream only (`s8c8hx8n` is refused: "card and note need the same scale precision").
   Its note rows are exactly the compact notes, so it is the int8-card option of the compact format.
6. One chained GPU ticket (smoke, Big, Small, Tiny, extension) with in-ticket out-of-memory retries.

## Risks

1. **Big users out of GPU memory.** The main pool includes 2267 (488,983 reviews; one of the SM128 out-of-memory users,
   fine in SM131 at cap 0.8). The extension has users up to 754,676 reviews. Handling: retries at 0.8 and 0.92, the
   64-of-66 coverage floor, and the extension is report only.
2. **User 3876** (one card with 4,733 updates): the fork default is expected to be non-finite on both Big models
   (SM131), so the fork default fails for Big. Compact saturates and stays finite. Its 1000+ bucket is one user and is
   report only.
3. **GPU vs CPU rounding** of the flagged int kernels: the GPU smoke bounds it at 1e-4. A saturated state on long cards
   may differ more between kernels. Such cards are rare, and they are paired within the GPU rows.
4. This is the scorer's emulation of the fork's state rounding. SM133 confirmed that the fork's Rust runtime follows the
   same arithmetic.
5. The heads of the Instant-Only models: only the instant head is judged.
