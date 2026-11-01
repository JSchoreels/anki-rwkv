# Codebook state compression sweep (SM90): design (pre-registered 2026-11-01, before any model-side result)

Goal: the smallest per-card / per-note state budget with codebooks (vector quantization, fitted after training, no
retraining) that is still nearly free in accuracy on the **released small model** (the leak-free small model (SM72s),
25 epochs + 1-epoch 32k Muon cool-down, `wt/sm72s/.../scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/.../master.pth`),
iterating from state compression at a large budget (SM46): 5,902 bits per card, 3,028 per note, on the best model of
10-21 (+0.00009 curve / +0.00015 instant vs its int8 state).

Worktree `C:/Users/jonat/workspace/wt/sm90`, branch `sm90` = origin/main + the leak-free small model (SM72s) tree
(merge) + state compression at a large budget (SM46) fixes (merge). Main has no small-model code of its own.

## What changed vs state compression at a large budget (SM46)

1. **Codebooks refitted on the released model's own states.** Corpus: 50 training users of its run (3k-60k reviews,
   none of the 732 set), 1,500 samples per stream-layer and user, **leak-free inputs** (`state_vq_fit.py
   --deleted-cards neighbour`, the scorer's `sm71_leak.neutralise`). k-means on 45 users; every 10th corpus user is held
   out for the proxy. WKV: rank components 0-7 with (20, 16, 12, 8, 6, 4, 4, 3) residual stages of 2^10 codewords
   (joint 32-dim (u, v), as SM46); shift: m4 x 8 and m8 x 4 stages of 2^12 codewords per role. 200k / 300k points,
   20 Lloyd iterations, seeded.
2. **Norm range rule for 6/8-bit norms.** The quantile search of the earlier recipes clips the top 1% of norms. On the
   note heads of this model that made the 8-bit norm error 5x the 10-bit one (rank 8, exact directions: 1.0e-2 vs
   2.1e-3, held-out proxy). New `fit_norm_range(top="max")` (hi = the largest norm, lo searched), used for the 6/8-bit
   ranges of `codebooks-sm90.npz` only (8-bit then 2.2e-3). Older books and recipes are untouched.
3. **Separate card and note ladders.** The proxy (below) shows the note heads are much higher-rank than the card heads,
   and that a note budget of SM46's size leaves the note state ~500x further from int8 than the card state is from
   int4. One shared ratio would hide which stream costs accuracy, so each ladder keeps the other stream at the fork
   default (new `state_vq.IntStream`: the exact `+s8` / `+s4c8n` paths inside a `+vq` recipe):
   - `vqC<bits>` = coded card + int8 note;
   - `vqN<bits>` = int4 card + coded note;
   - `vqC<c>N<n>` = both coded (only the screen's pick reaches the 732 set);
   - `vq5902x3028` = SM46's recipe as is (both coded), refitted books, for continuity.

## Model-free proxy (held-out corpus users, `proxy-sm90-wkv.json`; relative squared error sum|A - Â|² / sum|A|²)

| | Card heads | Note heads |
|---|---:|---:|
| int8 rows (`+s8`) | 3.2e-5 | 4.8e-5 |
| int4 rows (`+s4`) | 9.6e-3 (fork default) | 1.0e-2 (diverged with the previous release) |
| energy kept at rank 4 / 6 / 8 | 99.55% / 99.95% / 99.994% | 96.5% / 99.07% / 99.79% |
| best codec at 590 bits per head | 6.5e-3 (rank 5, 8-bit norms) | 2.4e-2 (rank 6) |
| ~440 / ~295 / ~185 / ~145 bits per head | 1.4e-2 / 2.9e-2 / 5.7e-2 / 7.9e-2 | 4.0e-2 / 7.5e-2 (280) / 1.2e-1 / 1.4e-1 |
| all fitted stages (rank 8, 794 bits) | | 1.5e-2 (approx.) |

Token shift (stored 64-float vectors, float in the fork default; `proxy-sm90-shift.json`): int8 1e-4, int4 2-3.5e-2.
m4 (12-bit codes, 10-bit norm) per vector: x8 (394 bits) 1.3e-4 to 3.0e-4, x6 (298) 6e-4 to 1.2e-3, x4 (202) 2.5e-3 to
5.1e-3, x3 (154) 5e-3 to 1.1e-2, x2 (106) 1.2e-2 to 2.5e-2. m8 at equal bits is within ~±20% (better on the channel-mixer
vectors at 394 bits, worse at <= 200), so m4 stays. 8-bit shift norms give the same error as 10-bit (a 2-bit saving,
not taken).

## Budget ladder (bits per entity; state MB = 13,973 cards x card + 13,198 notes x note, codes only)

Fork default (int4 cards + int8 notes, fork format) = 15,704 / 12,536 bits = 3,530 B per card + note = 48.1 MB;
int8 = 62.4 MB. Per-head allocation = the proxy's best rank, 8-bit norms and a greedy stage split (largest error drop
per bit); shift m4 with stages scaled with the rung + 10-bit norm.

| Recipe | Rung | Card bits | Note bits | State MB | Card heads (rank, stages) | Shift stages | Codebooks MB f32 / f16 | Encode ms / review (est.) |
|---|---|---:|---:|---:|---|---:|---:|---:|
| `vqC5902` | card 1x (SM46's card) | 5,902 | int8 | 31.0 | 5, (16, 14, 11, 8, 6) | 8 | 24.0 / 12.0 | 3.6 |
| `vqC4430` | card 3/4 | 4,430 | int8 | 28.4 | 4, (14, 11, 9, 7) | 6 | 18.0 / 9.0 | 2.7 |
| `vqC2958` | card 1/2 | 2,958 | int8 | 25.8 | 3, (11, 9, 7) | 4 | 11.9 / 6.0 | 1.9 |
| `vqC1934` | card 1/3 | 1,934 | int8 | 24.1 | 3, (8, 5, 3) | 3 | 8.4 / 4.2 | 1.3 |
| `vqC1470` | card 1/4 | 1,470 | int8 | 23.2 | 3, (6, 4, 2) | 2 | 5.8 / 2.9 | 0.9 |
| `vqN3964` | note 1.3x (all fitted stages) | int4 | 3,964 | 34.0 | 8, (20, 16, 12, 8, 6, 4, 4, 3) | 8 | 26.3 / 13.2 | 2.4 |
| `vqN3020` | note 1x (SM46's note) | int4 | 3,020 | 32.4 | 6, (13, 11, 9, 8, 6, 4) | 8 | 23.5 / 11.7 | 1.9 |
| `vqN1524` | note 1/2 | int4 | 1,524 | 29.9 | 5, (9, 7, 5, 2, 1) | 4 | 11.5 / 5.8 | 1.0 |
| `vq5902x3028` | SM46's recipe (both) | 5,902 | 3,028 | 15.3 | 6, (15, 13, 10, 7, 5, 3), 10-bit norms | 8 | 23.7 / 11.9 | 5.4 |

The 1/6 rung (~985 bits per card) is not screened: the proxy puts it at ~12x the int4 card error, and ~800 bits already
cost +0.37% / +0.67% on earlier models. A note rung above 1.3x needs more fitted stages and would still be ~100x the
int8 note error, so it waits for the screen (follow-up only if the 1.3x note rung is close).

Encode cost (`cost_sm90.py`): one card and one note update per review; per updated head a 16x16 Gram + eigendecomposition
and per stage a nearest-codeword search over 1,024 x 32 floats; per stored shift vector and stage 4,096 codewords per
sub-vector. Timed single-threaded with numpy at batch 1 (5 us per WKV stage, 17 us per eigh, 7-13 us per shift
stage-position) as the fork's Rust runtime would run them; decode is a sum of codewords (negligible). The int formats
cost no search. Codebooks: shared by card and note; the 12-bit shift catalogs are ~70% of the size.

## Knobs

Screened on the model (GPU): card budget (5 rungs), note budget (3 rungs), card vs note (the two ladders), SM46's
shape on the new books.

Measured on the proxy only (chosen per rung, not screened): rank vs stages (ranks 2-8 at 10 per-head budgets),
norm bits 6/8/10 and the norm range rule, greedy stage split, shift m4 vs m8 and stage counts.

Fitting-data knob (attempted 13:55, NOT run): the codec proof showed the deep WKV stages generalise poorly at 200k
fitting points (component 0, held-out / in-sample residual 1.15x at stage 1 rising to 4.4x at stage 20;
`proof/proof-codec-200k.json`). A refit on all 813,788 fit-split heads (`fit-big-sm90.sh`, 5 parallel processes) used
~5 GB RAM each, filled the machine and crashed another session's GPU job; it was killed at ~14:00 with no output. The
screen and the finalists use the 200k books (`codebooks-sm90.npz`, sha256 c12b659b...). Follow-up: the same refit with
ONE process at a time and the data streamed or subsampled (< 6 GB RAM), then the proxy pick in `choose_books.py`
(900k replaces 200k if lower held-out error for both `vqC5902` card and `vqN3020` note heads).

Not run this round, with reasons:
- fitting points (above), codebook size per stage (8 / 12-bit vs 10-bit), Lloyd iterations (50 vs 20):
  `knobs-sm90.sh` / `knobs_sm90.py` are written (proxy only) but were dropped before the 16:00 break to free the CPU
  for the rank-8 fit and the norm refit; they matter for speed / codebook size more than for accuracy.
- 10-bit shift codes (would cut the codebooks by ~3x): needs a shift refit.
- norms relative to component 0 (scale handling): a codec change; the full-range 8-bit grid already brings the norm
  error to the 10-bit level.
- per-layer bit split inside the card (layer 0 vs 1), separate u / v codebooks: second-order.
- retraining with the codec in the loop (QAT): out of scope (fitted after training only).

## Screen (pre-registered)

40 of the 723 fit users of the 732 set (`screen-users.json`: evenly spaced ranks by review count among the 653 users
with <= 150,000 reviews; 1.51M reviews), leak-free (`--deleted-cards neighbour --split-existing`), w8 weights, one GPU
ticket: `w8+s4c8n`, `w8+s8`, the 9 recipes above. The screen users are part of the final 732-set readout (mild
selection; the screen only picks what to score).

### Finalist pick (`sm90_rule.py screen`)
e(r) = the larger of the two heads' extra cost of recipe r vs `w8+s4c8n` on the screen users, in points (% of the big
model (1 leak-free epoch) LogLoss). Screen threshold 0.04 points (2x the final rule, for the 40-user noise).
- c* = the smallest card rung with e <= 0.04; n* = the smallest note rung with e <= 0.04.
- Finalists, in scoring order:
  1. `w8+s4c8n`;
  2. c* (if no card rung passes: the largest card rung, `vqC5902`);
  3. the both-coded `vqC<c*>N<n*>` if both exist;
  4. the next larger card rung than c* (safety), if any;
  5. n*, if any. No note rung passing = no note finalist: notes stay int8.

## Decision rule (pre-registered, `sm90_rule.py final`)

On the 723 fit users of the 732 set (users scored in every finalist label and in the big model's rows), leak-free:
- extra cost vs `w8+s4c8n` (the fork default) on the same users, points = 100 x ΔLogLoss / LogLoss of the big model
  (1 leak-free epoch), review-weighted, paired user bootstrap 2,000, seed 0, 95% CI reported;
- a budget is **nearly free** if the point estimate is <= +0.02 points on curve AND instant, AND its total stays inside
  the smaller-model bar: <= +0.5% curve / <= +0.4% instant vs the big model (1 leak-free epoch) (point estimates);
- the answer per stream = the smallest nearly-free card budget and the smallest nearly-free note budget (and the
  both-coded pair if it is nearly free);
- **practical** (reported with every budget, part of the verdict): codebooks <= 10 MB at float16 and estimated encode
  <= 1 ms per review; plus the GPU scorer's reviews/s. A nearly-free but impractical budget is reported as such.

Reported per budget: state MB (13,973 cards / 13,198 notes), codebook MB, GPU scoring reviews/s, estimated Rust encode
ms per review.

## CPU proof (before any GPU ticket)
- (a) fitting deterministic: two k-means fits with the same seed on the same corpus give bitwise the same books
  (`proof/proof-fit-det.json`);
- (b) encoding deterministic and decode(encode(x)) as expected: the explicit decode of the picks (sum of codewords,
  norms on the grid) equals the codec output bitwise; the held-out residual falls with every stage and is not below the
  in-sample residual (`proof/proof-codec.json`). The first run also required held-out <= 1.6x in-sample and failed
  (4.4x at stage 20, `proof/proof-codec-200k.json`): a generalisation gap of the deep stages, not a codec error, which
  started the fitting-data knob;
- (c) int paths bitwise unchanged: `w8+s8` and `w8+s4c8n` rows of the sm90 tree = the released tree (wt/sm72s) on 3
  users (`proof/proof-int8.json`);
- (d) the scorer with compression off reproduces the recorded released rows (`gpu/small-lf-neighbour.jsonl`) within the
  strict 1e-5 rule (same file);
- (e) the GPU codec path on CPU tensors equals the CPU scorer for every screen recipe (`check-sm90-cpu.json`), the GPU
  ticket then checks CUDA against these references with the discrete codec rule of AGENT.md.
