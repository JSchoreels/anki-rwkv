# Training-free state check (SM116): design (2026-11-18, fixed before the score)

## Question

How much of the show-time retrain (SM92)'s card and note state carries information, and what does it cost to keep
only 70% of that state, with no training at all? Origin: an outside challenge (11-14) to cut parameters and state with
no gradient descent and no loss. The full challenge is unlikely (cutting one card layer still cost +0.11% curve /
+0.18% instant after 3 retrained epochs in the Tiny two-head model (SM106)), so this is a time-boxed CPU diagnostic:
one pre-registered score, no tuning on the 732 set. Low priority: a smaller state saves little time (Tiny pair 7.2 s
vs Small pair 7.8 s).

## Model and state

The show-time retrain (SM92): `wt/sm92/artifacts/small-model/sm92-e26-dec-1ep32k-muon/expertium/E89-F65e-F77a/master.pth`
(read only), architecture w64 (d = 64, 4 heads of 16). The state a card or a note keeps between its reviews, as the
stock CPU scorer (`scripts/small_model_heldout.py`, `scripts/state_vq.py` Hooks) stores it:

| stream | layer | WKV state | token-shift vectors | floats |
|---|---|---|---|---|
| card | 0 | 4 heads x 16 x 16 | time mixer, channel mixer | 1,024 + 128 |
| card | 1 | 4 heads x 16 x 16 | time mixer (its channel mixer is stripped) | 1,024 + 64 |
| note | 0 | 4 heads x 16 x 16 | time mixer, channel mixer | 1,024 + 128 |

Card 2,240 + note 1,152 = 3,392 floats. 17 blocks: 12 WKV heads (256 values each) and 5 shift vectors (64 values each).
In w8+s8 (the standard scoring) the WKV rows are int8-rounded when stored; the shift vectors stay float.

## Pools

- **"The 750 users"** is not a documented pool. The closest documented evaluation pool is the 732 set; the base is
  exactly the show-time retrain (SM92)'s rule pool: the 732 set's 723 fitted users that the big model (1 leak-free
  epoch) rows have, 722 users (42.3M reviews). Validation users, unseen by the model.
- **Scored users: the 629 of those 722 with <= 125,000 reviews** (`users-score.json`, 21.4M reviews, 51% of the rule
  pool's reviews). The cap is forced by the 6 GB per-process RAM limit: the stock CPU scorer holds the whole history at
  once, and its peak grows faster than the review count (3.9 GB at 112k reviews, 6.9 GB at 158k, 14.8 GB at 531k; the
  two runs above 6 GB were stopped). The 93 heaviest users are not scored; the result says so. A memory guard
  (`memguard.ps1`) kills any scorer above 5.9 GB and stops the launcher.
- **Direction users: 94 users outside every scored user** (`users-directions.json` holds the 100 drawn):
  `random.Random(116).sample` of the 1,500 validation pool minus the 732 set and its 3 long users, i.e. validation
  users with id >= 5000 (also unseen by the show-time retrain (SM92)), drawn with <= 200,000 reviews (694 candidates);
  the 6 drawn users above 125,000 reviews were dropped for the RAM limit (the first collection run went to 6.9 GB on a
  158k-review user and was stopped; `collect-b2.sh` re-ran its half without them). 94 users, 3,450,496 reviews (`users_used` in that file).
  Training users cannot be used: the scorer's label database has no labels for them.
- **Proof users:** 8 scored users with 2,000-15,000 reviews (`users-proof.json`, seed 116). The proof touches no
  decision (the 100% projection is the unmodified model).

## State directions (no training)

`sm116_state.py collect` runs the stock scorer (w8+s8, show-time inputs `e2s-query`, leak-free flags
`--split-existing --deleted-cards neighbour`) on the 94 direction users and accumulates, per block, n, sum x and
sum x x^T in float64:
- WKV: every stored state after every update of a card or note (the float update, before the int8 rounding);
- shift: every vector a review reads from the entity's previous review (query rows read too, so about 2 reads per
  stored vector).

`sm116_state.py basis`: per block a centered PCA (covariance eigendecomposition, float64), giving the mean mu and the
directions ordered by variance. `directions.json` holds the explained-variance curve of every block and the number of
dimensions for 50 / 90 / 95 / 99 / 99.9% of the variance. Summary in section "Directions found".

## The 70% state

Fixed before the score, no alternatives tried: every block keeps its first round(0.7 x D) directions: 179 of 256 per
WKV head, 45 of 64 per shift vector. Card 1,567 of 2,240, note 806 of 1,152, total **2,373 of 3,392 floats
(69.96%)**. Uniform per block (no allocation by variance across blocks: variances of different blocks are not on a
common loss scale, and choosing an allocation would be tuning).

## Fold method: exact where the recurrence allows, projection elsewhere

The stored state x of a block becomes mu + P P^T (x - mu), P = the kept directions (D x k, orthonormal). Equivalently
the entity stores only the k coefficients z = P^T (x - mu).

- **Token-shift vectors: an exact fold.** A time mixer reads the previous vector only through static lerps followed by
  a linear map: W_i lerp(x, x_prev, m_i) = W_i diag(1 - m_i) x + W_i diag(m_i) x_prev for each of its 8 branches
  (r, k, v, decay, a, gate, k-scale, v-scale; the channel mixer's key likewise). With x_prev = mu + P z, every
  reader becomes W_i diag(m_i) P z + W_i diag(m_i) mu: a folded D x k matrix and a bias, exactly. The writer is one new
  k x D matrix (z = P^T (x - mu)). No training.
- **WKV state: no exact fold into the existing weights.** The update S' = S diag(w_t) - (S kd_t)(a_t kd_t)^T + v_t k_t^T
  has a data-dependent diagonal decay and a data-dependent removal term, so no fixed subspace of the 16 x 16 state is
  preserved (except the coordinate axes); the values and keys are L2-normalised per head and the output goes through a
  per-head group norm, so a value-side or key-side subspace cannot be folded into W_v / W_k / W_r / W_o either. What is
  exact is a reduced-coordinate recurrence: the entity stores z (179 coefficients per head) and each review computes
  S = mu + P z, runs the stock update, and stores z' = P^T (S' - mu). That is the **projection at every stored step**
  scored here; it keeps the storage at 70% exactly but adds two fixed matrices per head, so it is **the upper bound of a
  weight fold** (a fold into the existing weights could at best match it).
- **Int8:** the projected WKV state then goes through the same int8 row rounding as the unmodified model (w8+s8), so
  the only difference between the two scores is the projection. Shift vectors stay float, as in w8+s8.
- Within a review nothing changes: the answer's output uses the float update from the (projected) stored state; only
  what is stored for the next review is projected (the state-VQ placement, `scripts/state_vq.py`).

Implementation: `sm116_state.py score` wraps `scripts.state_vq.configure` (stored WKV = s8(mu + P P^T (nxt - mu)),
read shift = mu + P P^T (x - mu), float64 matmul, cast back to float32) and runs the stock scorer unchanged.

## CPU proof (before the score)

`proof.sh` -> `check-cpu-sm116.json`: the 100% projection (P = all directions, P P^T = I in float64) on the 8 proof
users against (a) the unmodified model on the same CPU scorer: max |difference| of the benchmark LogLoss and of the
full-history loss per user and head <= 1e-6; (b) the show-time retrain (SM92)'s GPU rows (`wt/sm92/.../gpu/sm92-inputs.jsonl`,
label `sm92-e26-dec1ep32k/w8+s8@e2s-query`): <= 1e-5. (b) also bounds the CPU-vs-GPU gap of the comparison below.
Smoke (user 2388, basis from that user only): the 100% projection gave the unmodified full-history losses bit for bit.

## Directions found (94 users, before the score)

Stored WKV states: 3,450,496 per block (one per card / note update); read shift vectors: 9.96M (card), 10.77M (note).
Dimensions needed for 90 / 99 / 99.9% of the variance, and the variance the 70% state keeps (`directions.json` has the
full curves):

| block | dims | 90% | 99% | 99.9% | kept at 70% |
|---|---|---|---|---|---|
| card.L0.shift.c | 64 | 9 | 25 | 44 | 99.92% |
| card.L0.shift.t | 64 | 6 | 18 | 35 | 99.98% |
| card.L0.wkv.h0 | 256 | 21 | 98 | 191 | 99.86% |
| card.L0.wkv.h1 | 256 | 37 | 130 | 209 | 99.72% |
| card.L0.wkv.h2 | 256 | 32 | 116 | 195 | 99.83% |
| card.L0.wkv.h3 | 256 | 30 | 118 | 198 | 99.80% |
| card.L1.shift.t | 64 | 23 | 45 | 57 | 99.01% |
| card.L1.wkv.h0 | 256 | 74 | 191 | 237 | 98.65% |
| card.L1.wkv.h1 | 256 | 42 | 157 | 229 | 99.41% |
| card.L1.wkv.h2 | 256 | 41 | 154 | 226 | 99.46% |
| card.L1.wkv.h3 | 256 | 65 | 188 | 242 | 98.79% |
| note.L0.shift.c | 64 | 16 | 39 | 54 | 99.56% |
| note.L0.shift.t | 64 | 19 | 42 | 56 | 99.35% |
| note.L0.wkv.h0 | 256 | 58 | 177 | 236 | 99.07% |
| note.L0.wkv.h1 | 256 | 4 | 23 | 98 | 99.98% |
| note.L0.wkv.h2 | 256 | 24 | 131 | 220 | 99.65% |
| note.L0.wkv.h3 | 256 | 31 | 139 | 220 | 99.61% |

Most of the variance sits in few directions (90% in 4 to 74 of 256 per WKV head), but the tail is long: 99.9% needs
98 to 242 of 256. The card stream's second layer is the least compressible (98.65-99.46% kept). Variance is not loss:
only the score answers the question.

## CPU proof result

`check-cpu-sm116.json`: **PASS**. 8 proof users, the 100% projection vs the unmodified model on the same CPU scorer:
max difference 0.0 (bit for bit) for the benchmark LogLoss and the full-history losses, both heads; vs the show-time
retrain (SM92)'s GPU rows: max 2.0e-6 per user and head (tolerance 1e-5).

## Pre-registered score and rule (`sm116_rule.py decide`)

- **New:** the show-time retrain (SM92) with the 70% state, CPU scorer, w8+s8, show-time inputs, leak-free, the 629
  scored users (`rows/sm116-k70.jsonl`, label `sm116k70/w8+s8@e2s-query`).
- **Unmodified:** the show-time retrain (SM92)'s existing GPU rows on the same users, inputs and flags (the proof
  shows CPU = GPU on this scorer to <= 1e-5 per user).
- **Cost** per head (curve and instant) = review-weighted benchmark LogLoss, new - unmodified, in % of the big model
  (1 leak-free epoch) show-time LogLoss on the same users (rows `wt/sm31/.../sm31-inputs.jsonl`,
  `big-leakfree-1ep/w8+s8@e2s-query`; the 11-11 standard); paired user bootstrap, 2,000 resamples, seed 0, 95% CI.
  Also reported: % of the unmodified model's own LogLoss, users better / worse.
- **Rule:** "starting point" (the 70% state becomes the start of a trained smaller model) **iff both heads have the
  upper 95% CI bound of the cost below +0.05%**. Otherwise "drop". (A cost that is only "not shown to exceed 0.05%"
  is not enough: the CI itself must sit below +0.05%.)
- One score only; no other keep fraction, allocation or basis is scored on these users. `sm116_rule.py self-test`
  checks the bootstrap and verdict logic on synthetic rows (identical, cheap, costly, noisy).

## Cost and schedule

CPU only, at most 3 processes of 2 torch threads, < 6 GB RAM each (users <= 125k reviews, memory guard). Directions: 2 processes, about 30 min. Score: 3 processes over fixed thirds of the
scorer's stratified order, about 1,550 reviews/s each, 21.4M reviews, about 1.3 h. `launch-sm116.sh` stops starting new
users 25 min before the next quiet window (quiet.json, the daily 20:00-01:00 break, Sunday from 16:00), sleeps through
it and resumes (scored users are kept), then runs the rule.
