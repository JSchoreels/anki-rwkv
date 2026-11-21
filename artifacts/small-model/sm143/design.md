# Per-user adaptation (SM143): design (pre-registered 2026-11-21, before any GPU ticket)

## Question

The released RWKV models are trained on about 10,000 benchmark users and only read a new user's history as input. On the user's own collection the released RWKV model beats FSRS on ROC AUC (0.756 vs 0.737). The user asks (11-21) whether adapting the model to each user from that user's own earlier reviews gives a real further gain, in LogLoss and especially in within-user discrimination (AUC), measured on many benchmark users rather than on one collection.

A per-user recalibration (a monotone Platt refit, as in the rolling recalibration (SM125, SM128)) cannot change AUC. Adaptation must change the model's ranking of a user's reviews, so the candidate variants below change the network itself (a per-user vector, the instant head, or every weight). The Platt refit stays in as a control: it shows how much of any LogLoss gain is calibration alone.

## Users

- Pilot: 200 users drawn with `random.Random(143).sample` from the 723 users of the 732 set's fit list (`wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json`, the standard read-out's users; validation users with id below 5000 that no model here trained on). File `users-pilot.json` (11,771,250 reviews; median 30,305, largest 531,091).
- Full fine-tune subset: 50 users drawn with `random.Random(143).sample` from the pilot (`users-pilot-d.json`, 2,700,594 reviews), because variant (d) costs far more GPU time per user (see "GPU time").
- Smoke users: the two smallest users of the fit list, 2388 and 4676 (as in the long-card check (SM131) and the release-candidate checks (SM139)); user 2388 alone for the GPU real-step smoke.
- Full run (only if the pilot passes the go rule): the 723 users of the fit list (the 722 users of the standard read-out plus the one the big model's read-out lacks).

## Time split: the benchmark's own rolling split

The FSRS numbers of the benchmark already come from a time-series split: srs-benchmark fits FSRS per user with `TimeSeriesSplit(n_splits=5)`, i.e. fold k (k = 0..4) is fitted on every review before the fold and scored on the fold (an expanding window, refit 5 times). The RWKV benchmark reviews are the same kind of folds (`rwkv/find_equalize_test_reviews.py` builds the label database with the same `TimeSeriesSplit`), but RWKV is never fitted on them; it only reads the history.

Primary scheme (rolling, the closest to an app that refits from time to time, and exactly the FSRS-7 protocol): the 5 test folds of the published FSRS-7 protocol (`--short --secs --equalize_test_with_non_secs`), read from the scheduling-time vs review-time accuracy (SM81) per-user files (`wt/sm81/artifacts/small-model/sm81/gpu/fsrs/<user>.npz`: review_th, fold, y, p_fsrs7_fit). For fold k:

- fold start = the fold's first review_th (review_th = the row number of the user's revlog; rows are in time order and `get_rwkv_data` asserts that day_offset never decreases);
- fit set = the model's labelled instant query rows with review_th below the fold start, so every review before the fold, same-day reviews included, exactly as FSRS-7's training set for that fold (FSRS-7 with `--short` also trains on same-day reviews);
- causal validation slice = the last 10% of the fit set by review_th (at least 30 reviews); the training part = the rest;
- evaluation = fold k's FSRS-7 test reviews (reviews whose card was last reviewed on an earlier day) that are also RWKV benchmark reviews with an instant prediction (SM81's common-row rule); every predictor is scored on exactly these reviews, and the script stops if FSRS-7's y differs from the model's label on any of them;
- a fold with fewer than 200 fit rows keeps the base model (recorded as skipped); with the smallest pilot user (3,685 reviews) fold 0 already has 518 fit rows, so no pilot fold is expected to be skipped.

Why not a single 50% cut: the rolling scheme reuses FSRS-7's existing per-fold fits (no new FSRS run, the same published protocol), measures the gain at 5 history lengths per user (which the history bins below need), and costs nothing extra for variants (a) to (c), whose 5 refits run on features computed once.

Causality is asserted in code (`assert_causal` in `scripts/sm143_adapt.py`, with a selftest that a leaking set trips it): for every fold, every fit row's review_th is below the fold start and its day_offset is not after the fold start's day; for (d), the same for every review in its training windows. There is no absolute timestamp in the dataset (only day_offset and per-card intervals), so the revlog row order is the time order used. The model is causal (an RNN over the history), so the trunk features of a fit row depend only on that row and earlier rows. The evaluation predictions of fold k use the model adapted for fold k, which never saw a review of fold k.

## Models

- Primary: the longer Small Instant-Only model (SM113) (`sm113-inst-e35st-st-dec2ep32k-t99`, `expertium:w64`, 331,511 parameters), the release default size.
- Secondary: the six-layer Tiny Instant-Only model (SM130) (`sm130-io-dec2ep32k-t99`, `expertium:w64-c1-d2p1u1`), variants (a) to (c) only.
- Reference without adaptation: each model itself (variant "none", bit for bit the stock scorer's prediction), and the show-time instant-only big model (SM99) (`sm99-big-inst-e2`), unadapted, from the stock scorer's `--dump` files on the same reviews: does adaptation let Small beat Big?
- All on show-time inputs (`e2s-query`), leak-free (`--deleted-cards neighbour`), fp32 weights and state. The release runs w8+s8; fp32 is the cleaner first test, and the head-only variants would keep their few adapted values in float on a client anyway.

Checkpoint sha256 values are in `models.sh` and `design-pins.json`; the launcher refuses to run if they, the users files or the FSRS-7 files (one digest) changed.

## Variants (fixed in advance; refit per fold from the base weights)

The base forward keeps the trunk output x (the 64 values that enter the instant head's LayerNorm) of every row. Variants (a) to (c) are fitted on those features (no backward pass through the recurrence), so a client could fit them from features it already computes.

| Variant | What is adapted | Values | Fit |
|---|---|---|---|
| (a) Platt control | p' = sigmoid(exp(s) · logit(p) + b) on the instant probability | 2 | Adam, learning rate 1e-2, 100 full-batch steps, L2 1e-3 on (s, b) |
| (b) user vector | u added to the normalised pre-head features (after the LayerNorm, before the head's first linear layer) | 64 | Adam, learning rate 1e-2, 100 full-batch steps, L2 1e-3 on u |
| (c) instant head | the instant head's LayerNorm, first linear layer and output layer (prehead_norm, head_p, p_linear) | 8,964 (Small) | Adam, learning rate 1e-3, 100 full-batch steps, L2 1e-3 to the base values |
| (d) full fine-tune (upper bound) | every weight | 331,511 (Small) | Adam, learning rate 1e-4 (a seventh of the training peak 7e-4), 30 steps, each on 64 random 128-review windows of the training part (the training protocol: state reset at each window), L2 1e-3 to the base weights |

- Loss: binary cross-entropy of the instant head's recall probability (1 − p(Again), clamped like the model) on the labelled instant query rows: the quantity the benchmark scores.
- Variant (a) leaves each fold's ranking unchanged; the per-user AUC pooled over the 5 folds can still move a little, because each fold gets its own monotone map.
- Checkpoint choice: the validation-slice LogLoss every 10 steps for (a) to (c) and at steps 15 and 30 for (d) (each (d) checkpoint is scored by a full-history forward); the lowest wins, and step 0 (the base model) is always a candidate, so a fold whose validation slice does not improve keeps the base model. No refit on the validation slice afterwards.
- (d) windows are drawn with `numpy.random.default_rng([143, user, fold])`; window batches use prepare seed 143,000,000 + 100 · user + 10 · fold + step. Dropout stays off (eval mode).
- Exactness: with 0 steps every variant must reproduce the unadapted predictions bit for bit, and the unadapted row must equal the stock scorer's row (`scripts/small_model_heldout.py`) bit for bit on the same device; see "CPU proof".

## Metrics (per user on the evaluation reviews, then paired over users)

- LogLoss of the instant head, review-weighted (all evaluation reviews pooled), as % change vs the reference.
- Within-user AUC (the main metric for the user's question): the review-weighted mean of the per-user AUC difference (users with both classes); the user mean is in the JSON.
- Calibration: ECE with 10 equal-width bins per user, review-weighted mean of the per-user difference.
- Each adapted variant vs the unadapted model; every predictor vs FSRS-7 fitted per user on the same early part (p_fsrs7_fit of the same fold); Small (all variants) vs the unadapted big model (SM99).
- By the amount of earlier history: per user and fold, bins of the fold's fit rows (0-1,000, 1,000-5,000, 5,000-20,000, 20,000 or more), LogLoss % and AUC difference vs unadapted, user bootstrap within the bin. Adaptation should help most where the early history is long.
- Uncertainty: paired user bootstrap, 2,000 resamples, seed 0, 95% percentile CI.

## Go rule (repo standard)

For each candidate variant (b), (c) and (d) of the primary model vs the same model unadapted: go when the review-weighted LogLoss change has its 95% CI upper bound below 0 and its point estimate at or below −0.05%. The AUC difference is always reported with its CI, and so is the gap to FSRS-7 before and after adaptation. Variant (a) is the control (by construction it cannot change AUC). The pilot is a screen: a variant that passes goes to the full 723-user run, which decides with the same rule; three variants are screened without a multiplicity correction, so the full run is the confirmation. A pilot without any go ends the track unless the user asks for other variants.

## Cost (what a client would pay)

Measured per user and variant: adaptation seconds for all 5 folds, the share of folds whose validation slice kept the base model, the forward time, and the GPU peak memory (`gpu_max_mib`); on the CPU proof also the process peak working set. For a client the relevant figures are the CPU proof's (desktop CPU, 2 threads): see "CPU proof" below. The features kept for (a) to (c) are 64 float32 values per labelled review (256 bytes; 7.7 MB for 30,000 reviews), the adapted values 2 / 64 / 8,964 floats. First CPU figures (user 2388, 3,685 reviews, Small, 2 threads, from `cpu/real-small.jsonl`): the full-history forward 2.5 s; per fold (a) about 0.03 s, (b) about 0.1 s, (c) about 0.15 s, (d) 60 to 130 s (475 s for the 5 folds, almost all of it the 150 training steps; a long history makes (d)'s windows batches full at 64 windows and its 10 checkpoint forwards longer); process peak working set 2.9 GB (the forward, not the adaptation).

## CPU proof (done before any GPU ticket; rows in `cpu/`)

- `cpu/smoke-scorer-rows.jsonl`: the stock scorer on CPU, the 2 smoke users, both models at fp32 on e2s-query, neighbour.
- `cpu/zero-small.jsonl`, `cpu/zero-tiny.jsonl`: `scripts/sm143_adapt.py --device cpu --steps-scale 0 --variants a,b,c,d`. `sm143_check.py zero` passes with tolerance 0 for both models: the unadapted rows equal the scorer's rows bit for bit (totals and the benchmark dict), the head recomputed on the kept features equals the forward's p_imm bit for bit (head_identity_max_abs 0), (d)'s scoring forward at step 0 equals the base forward (d_identity_max_abs 0), every variant's evaluation predictions equal the unadapted ones, FSRS-7's y equals the model's labels on all common reviews (555 of 555 and 1,435 of 1,435 FSRS-7 test reviews in common), and every fold has its causal record (for user 2388 fold 0: 518 fit rows, the last at review_th 1,046 for a fold starting at 1,047).
- `cpu/real-small.jsonl`: the same 2 users with the real step counts (all four variants), for the cost figures and to see that the fits run; its numbers are not a result (2 users).
- `sm143_adapt.py selftest`: the causality assertion trips on a leaking fit set.

## GPU plan and time

One launcher (`launch-sm143.sh`, owner sm143, cap 0.8 then 0.92 after a pass without progress, quiet.json windows only, STOP file, resume, DRYRUN=1, final output check). Tickets in order, expected minutes:

| Ticket | What | Expected |
|---|---|---|
| smoke | 2 users, Small and Tiny, all variants at 0 steps vs the CPU rows (≤ 1e-5); a mismatch stops the run | 20 min |
| steps | all variants with their steps on user 2388 (the full fine-tune's training kernel on the GPU); a failure only skips the "full" ticket | 10 min |
| small | variants a, b, c of Small on the 200 pilot users | 45 min |
| big | the big instant-only model (SM99), unadapted, stock scorer with `--dump`, 200 users | 30 min |
| tiny | variants a, b, c of Tiny on the 200 pilot users | 35 min |
| full | variant d (and a, b, c again) of Small on the 50-user subset | 150 min |

Pilot total: about 4.8 GPU hours of expected ticket time (the head-only part about 2.3 hours). Basis: on the 723 users Small's forward took 5.5 minutes of model time and Tiny's 2.4 and Big's 9.1 (release-candidate checks (SM139) rows), with the scorer's whole ticket about 6 times its model time; the 200 pilot users hold 27.5% of the reviews; the head fits are 1,500 small full-batch steps per user and variant set; (d) adds 150 training steps on 64 windows and 10 extra full-history forwards per user.

A full 723-user run of variants (a) to (c) for Small, Tiny and the Big reference: about 3.6 times the pilot's head-only tickets, about 8 GPU hours. Variant (d) on all 723 users: about 14 times the 50-user ticket, about 35 GPU hours; it would only be worth running if (d) alone passes on the pilot, and even then a client could not afford it (see "Decision points").

## Decision points for the user (written before the result)

- If only (d) passes: the gain exists but needs a full backward pass through the recurrence on the user's device, many times the cost of reviewing. Options: (1) stop and report the upper bound; (2) design a cheaper variant that gets close to (d), e.g. fine-tuning only the last layer of each stream with a low-rank adapter; (3) do the adaptation on a server (privacy cost). Recommendation: (2), after reading (d)'s history bins: if the gain is concentrated in long histories, a desktop-only option for long collections could be enough.
- If (b) or (c) passes: run the full 723-user confirmation, then a w8+s8 check and a client-side cost measurement in the fork.
- If nothing passes but (a) improves LogLoss: the gain is calibration, already covered by the rolling recalibration track (SM125, SM128).

## Outputs

- `gpu/rows-small.jsonl`, `gpu/rows-tiny.jsonl` (per user: the scorer-format unadapted row, per fold the fit sizes, chosen steps, validation losses, seconds, and per predictor n, LogLoss sum, AUC, ECE, pooled and per fold), `gpu/rows-big.jsonl` (stock scorer rows): committed (benchmark users, per-user scores only).
- `gpu/npz-small/`, `gpu/npz-tiny/`, `gpu/dump-big/`: per-review arrays, not committed.
- `sm143-result.json`, `sm143-result.md`: `sm143_report.py report`.
