# Bounded-memory long-card scorer (SM155): design

Pre-registered on 11-24 before the formal exactness runs (`run-sm155-tests.sh`). The development tries before it
(Big two-head (SM91s2) on users 2388 and 2883, Small two-head (SM118) and six-layer Tiny two-head (SM130) on user 2883)
are listed under "Development history" and are not part of the result.

## Question

The CPU scorer (`scripts/small_model_heldout.py --device cpu`) evaluates a user's whole history in one forward pass,
so its peak memory grows with the review count (about 3.2 GB at 102,545 reviews and 5.6 GB at 150,500 in the long-card
state drift check (SM136); 6.0 to 7.3 GB for Big at 87k to 151k reviews in the low-rank int8 states test (SM148)). The
6 GB per-process limit therefore drops the longest users: 13 Big long-card users in the low-rank int8 states test
(SM148), 15 to 16 users per size in the long-card state drift check (SM136), and the 5 largest Big users (424k to 531k
reviews) never fit on the GPU. Can a scorer mode with memory set by the number of entities (cards, notes, decks,
presets, the user) give the same per-review predictions?

## Engine (`--engine stream`, `scripts/stream_scorer.py`)

* The history is replayed in chunks of `--chunk-rows` canonical rows (review order; default 16,384). Between chunks
  only the per-entity recurrent state of every stream and layer is kept: the WKV matrix, its update count, and the two
  token-shift vectors (time mixer, channel mixer).
* Nothing of the model or of the codecs is re-implemented. Per chunk and stream, the model's own modules run on
  [B, T] blocks of that chunk's rows (fork models: the stream module; interleaved Small and Tiny models: one block per
  layer step in the model's own order). Two module globals are swapped during an evaluation: the WKV reference (a step
  loop identical to the stock ones that starts from the carried state and calls the label's own codec callables) and
  `torch` (the token-shift gather puts carried vectors where the shift source lies in an earlier chunk). Heads and loss
  are the model's own `head_and_out` and `get_loss`, per chunk.
* Exact storage: card and note states of integer-row formats are kept as int8 codes and one float32 scale per row,
  low-rank formats as their factor pair, everything else as float32; every write is checked bit for bit against the
  stored state, with a float32 fallback per entity (counted in the row field `stream_engine`).
* Row placement reproduces the stock kernels' arithmetic. Two CPU kernel facts were measured on this machine
  (AVX512, 2 threads): MKL's GEMMs with 3 to 8 outputs (k and v scale, value LoRA, rating head) pair rows, so a row's
  result depends on the parity of its position in the block and on an odd last row; ATen's vectorised elementwise
  kernels (sigmoid, tanh on 4- to 16-wide tensors) run the last elements of a block, and of each thread's half when the
  block has more than 32,768 elements, through a scalar path with different rounding. Normal rows keep their stock
  parity in blocks whose size is a multiple of 16; rows within 32 of their stock block's end or middle run in a
  one-entity replica block of the same size modulo 64, the same side of every thread-split threshold, at the same
  distance from the end or the middle.

## Exactness rule (pre-registered)

PASS if and only if, for every (model, user, label) cell of the test matrix below:

1. both engines write a complete row (no error, no non-finite row);
2. for each head (curve = ahead, instant), the reviews predicted (`--dump-scope all`: every predicted review, long cards
   included) are the same set;
3. max |p_stream - p_stock| <= 1e-5 over every one of those reviews, both heads (float32 head precision);
4. the benchmark metrics of the row (6-decimal LogLoss, RMSE(bins), AUC) are equal within 1e-6.

The share of reviews whose prediction is bitwise equal is reported too, but it is not part of the rule. Any failing
cell fails the rule; there is no averaging.

## Test matrix

Scorer flags of the long-card chains of the low-rank int8 states test (SM148): `--inputs e2s-query --split-existing
--deleted-cards neighbour --nonfinite-rows --prior-buckets 0,1,3,5,9,16,32,128,300,1000`, `PYTORCH_JIT=0`, CPU, 2
threads, `--dump-scope all`. Both engines run the same command except `--engine stream --chunk-rows N`.

* Models: the six release models of `artifacts/small-model/sm148/models.sh`: Big two-head (SM91s2), Big Instant-Only
  (SM99), Small two-head (SM118), Small Instant-Only (SM113), six-layer Tiny two-head (SM130), six-layer Tiny
  Instant-Only (SM130).
* Users: the 3 smoke users (2388, 4676, 1866) and 5 long-card users of `artifacts/small-model/sm148/users-longcard.json`
  (users with a card of 128 or more prior answers): 2883, 3882, 1215, 280, 4407 (9,451 to 23,053 reviews; the 5
  smallest after the smoke user 1866, so the matrix stays within about an hour of CPU time).
* Main labels, every model and user: `w8` (int8 weights, float32 state: the int8-weights path), REF `w8+s8hx8`, T24
  `w8+lr2.16.4q8x8`, Q2 `w8+lr2q8x8`, at `--chunk-rows 2048` (many chunk boundaries per user).
* Format families, 3 smoke users, Big two-head (SM91s2) (fork model path) and Small two-head (SM118) (interleaved
  expertium path): `fp32`, `w4` (int4 weights), `w8+s8` (int8 rows, float32 scales), `w8+s4c8n` (int4 cards, int8
  notes), `w8+s4hgx4` (int4 rows, one float16 scale per head, int4 shift vectors), `w8+t4.128.8c8nhx8` (tiered by
  answer count), Q4 `w8+lr4q8x8`, `w8+lr2.16.4q6x4` (int6 factors, int4 shift vectors; smaller shift vectors and
  factors on T24 states (SM152)), U8 `w8+lrRq8x8`, F4X0 `w8+lr4fx0`, LR4 `w8+lr4` (rank-4 truncation of the
  low-rank structure probe (SM147)); Small two-head (SM118) also `w8+vq403x214` (state VQ, codebooks
  `artifacts/small-model/state-vq/codebooks-dec10.npz`). Chunk rows 1024.
* Default chunk size: user 4407, Big two-head (SM91s2), REF and T24 at `--chunk-rows 16384`.

Not supported (the scorer refuses them with `--engine stream`): CUDA, `--emulate-gpu-path`, `--window`,
`--legacy-metrics` and the learned initial state of the tiny-additions model (SM77). The spectrum probe `+lrsv` runs,
but its spectrum aggregates are summed in another order (its predictions are those of the float state).

## Memory and speed (reported, no pass/fail)

* Peak working set, stock vs stream, Big two-head (SM91s2) REF on user 1458 (92,307 reviews: the largest Big user the
  stock scorer finished under the 5.9 GB guard in the low-rank int8 states test (SM148)), `artifacts/small-model/
  sm136/peakrun.py` (working set polled every 2 s), chunk rows 16,384.
* Preparation alone (batch + row layout) for the largest Big user 3735 (531,091 reviews): `prep_probe.py`.
* Reviews per second from the rows' `model_seconds`, per model, stock vs stream (2 processes run side by side; both
  engines see the same load).

## First use (decided, not started on 11-24)

The 13 Big long-card users missing from the low-rank int8 states test (SM148) and then the 5 Big users that never fit
on the GPU, both Big models (two-head (SM91s2) and Instant-Only (SM99)), T24 and REF, with the SM148 long-card flags
and `--engine stream`; command and ETA in `HANDOFF.md`.

## Development history (not part of the result)

1. First version (blocks grouped by in-chunk length only): Big two-head (SM91s2), user 2388: fp32 within 4.8e-07, REF
   off by 3.1e-04; user 2883 (long cards): REF 1.8e-04, T24 1.6e-03. Ulp-level differences from the kernels above
   are amplified by the int8 rounding and the rank truncation on long cards.
2. Parity-preserving placement: user 2388 REF within 1.0e-06 (curve bitwise on 45% of reviews).
3. Replica blocks for rows near a stock block's end or middle: user 2388 fp32 and REF, user 2883 REF and T24 bitwise
   equal on every review (Big two-head (SM91s2)); Small two-head (SM118) and six-layer Tiny two-head (SM130) fp32, REF
   and T24 on user 2883 bitwise equal too.
