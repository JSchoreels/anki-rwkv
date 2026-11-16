# Time heads for the family (SM117): design and pre-registered rules

User 2026-11-15: "Add the time head for all 6 models and measure how reliable it is to predict R recall under S seconds."

Branch `sm117`, worktree `C:/Users/jonat/workspace/wt/sm117`. Scripts: `scripts/sm117_head.py` (plan, baselines, train, score, gradecompare, decide, report), `scripts/sm117_check.py` (the CPU proof), `scripts/sm117_timelib.py` (the answer-time code of the SM82 line, copied with provenance). Launcher: `launch-sm117.sh`. Everything here was fixed before any GPU ticket of SM117 ran.

**Branch base.** The branch is cut from `origin/sm115` (the tip of the experiment line: show-time inputs `scripts/sm31_inputs.py`, the one-card-layer preset `w64-c1`, the show-time scorer), not from `origin/main`: `main` holds the pages and coordination files but none of the small-model code (no `scripts/small_model_heldout.py`, no show-time inputs, no `w64-c1`). The answer-time code lives on the `sm82` / `sm82b` / `sm82c` branches, which have no show-time inputs; it is copied into `scripts/sm117_timelib.py` and `scripts/sm117_head.py` (header of each lists the source file and commit, `sm82c` at `7f5f4150`). No file of `scripts/`, `expertium/` or `rwkv/` from `sm115` is changed; SM117 only adds files.

## The six models (release `rwkv-family-2026-11-13`)

| model | size, kind | scorer kind | master (read only) | sha256 = package manifest |
|---|---|---|---|---|
| the Tiny two-head model (SM106) | Tiny, two-head | `expertium:w64-c1` | `anki-rwkv/artifacts/hf-family/repo-new/rwkv-tiny-two-head/master.pth` | `e9d0d0a0b336...` yes |
| the Small Instant-Only model (SM98) | Small, instant-only | `expertium:w64` | `.../rwkv-small-instant-only/master.pth` | `565b105e1e16...` yes |
| the Tiny Instant-Only model (SM101) | Tiny, instant-only | `expertium:w64-c1` | `.../rwkv-tiny-instant-only/master.pth` | `d1408ae9c883...` yes |
| the show-time instant-only big model (SM99) | Big, instant-only | `fork::` | `.../rwkv-big-instant-only/master.pth` | `c49e45d69819...` yes |
| the show-time big model (SM91s2) | Big, two-head | `fork::` | `.../rwkv-big-two-head/master.pth` | `67ee8e9b0d01...` yes |
| the show-time retrain (SM92) | Small, two-head | `expertium:w64` | `wt/sm92/artifacts/small-model/sm92-e26-dec-1ep32k-muon/expertium/E89-F65e-F77a/master.pth` (the package has the hash only) | `2091f8a879a5...` yes |

Every script verifies the sha256 against the package `manifest.json` before it uses a master (`sm117_head.py verified_master`; CPU proof C0 checks all six).

**README note (not edited).** Line 73 of the family README (`artifacts/hf-family/repo-new/README.md`) says the show-time big model (SM91s2) has an answer-time estimate. It does not: no answer-time head was ever trained on a Big model, the fork's `qt/aqt/rwkv_inference/` has `time_head.*` files only in model folders of the Small line (the show-time retrain (SM92) and an earlier small model), none beside `RWKV_trained_on_5000_10000.bin`, and the fork's `AnswerTimeHead` (`rslib/src/rwkv/engine.rs`) is built for the Small layout only (`FROZEN = (ANSWER_TIME_ROUNDS + 1) * D_MODEL + HEAD_DIM`, 474 inputs). The README should say only the show-time retrain (SM92) has one (and it is not in the release, line 59).

## The recipe (unchanged unless listed)

The answer-time head on richer frozen features (SM82c), `wt/sm82c/scripts/sm82c_head.py`: MLP inputs -> 128 -> 4 grades x 16 log-time bins (bin edges 500 ms x 120^(k/14), k = 0..14, last bin 60 s or more), ReLU, zero-initialised output layer (first loss ln 16), censored cross-entropy of the true grade's bins (time at a user cap below 60 s = "this bin or later"). Trained ALONE on a frozen copy of the model: every model parameter has `requires_grad` False and none is in the optimizer. Inputs at each instant (query) row: the trunk's residual stream after the input encoder and after each schedule unit, each layer-normalised without affine parameters, the instant head's hidden layer (`head_p` output), and the 26 explicit causal time features (card's last 3 answer times, the user's running median per grade, ...). Plan: SM82c's 1,991 users (none in the 732 pool), their seeded order, the first 400,000 reviews of each history; one pass, AdamW lr 2e-3 (2% warm-up, cosine to 0), no weight decay, batch 8,192, seed 0, shuffle buffer 1M rows; the explicit-only ablation (26 -> 128 -> 64) in the same pass.

SM117 changes:

1. **Show-time inputs everywhere** (training and scoring), `scripts/sm31_inputs.py` `e2s-query`, leak-free (`neighbour`), id / day seed 2025 + user: exactly the grade scorer's `LeakBatches(..., inputs="e2s-query")` batches (CPU proof C8: the training batch builder without a cap gives the scorer's features bitwise). SM82c and the shipped head of the show-time retrain (SM92) used normal (answer-to-answer) inputs.
2. **Same-day reviews included** (decision 2 below): label rule `time_labels(same_day=True)`.
3. **The head stays outside the model.** The training writes `head.pth`, `ablation-head.pth` and `master.pth` = the frozen model's own `state_dict` after training (checked bitwise against the release in the training, in the CPU proof C1 and by the GPU grade check). Shipping a head to the fork is not part of SM117.
4. **Big capture.** The Big models are the fork's `SrsRWKV` (sequential streams, a TorchScript module): `fork_forward_capture` runs `forward_batch` verbatim and keeps the residual after the encoder and after each stream (CPU proof C3: every output bitwise the stock forward's; the final layout is the canonical row order, asserted at every call).
5. **Big buffer 500k rows** (its rows are 1306 wide: 500k x 1306 x 4 B = 2.6 GB on the GPU, the concatenation peaks at twice that).

### Decision 1: input dimensions per model

The structure is the same for all six: one layer-normalised snapshot of the residual stream after the encoder and after every schedule unit (the last one is the final residual that feeds every head), the instant head's hidden layer, the 26 explicit features.

| model | d_model | schedule | units | snapshots | head_p | head inputs | head values (main) |
|---|---|---|---|---|---|---|---|
| Small (SM92, SM98), `w64` | 64 | interleaved, layer steps card 2 / deck 4 / note 1 / preset 3 / user 3 | 4 layer rounds | 5 x 64 | 128 | 474 | 69,056 |
| Tiny (SM106, SM101), `w64-c1` | 64 | interleaved, card 1 / deck 4 / note 1 / preset 3 / user 3 | 4 layer rounds (round 1 has no card layer; the deck stream sets 4 rounds) | 5 x 64 | 128 | 474 | 69,056 |
| Big (SM91s2, SM99), fork | 128 | sequential, card 3 / deck 4 / note 2 / preset 3 / user 4 | 5 streams | 6 x 128 | 512 | 1,306 | 175,552 |

Tiny keeps 474 inputs (one card layer fewer, still 4 rounds), so a Tiny head has the fork's Small layout. For Big the unit is the stream (the sequential schedule has no rounds): 6 snapshots of 128 + 512 + 26 = 1,306. The hidden width stays 128 for every model (the same head capacity; only the first layer's width follows the inputs). Checked in CPU proof C1/C3 (feature widths 448 / 448 / 1280 plus 26).

### Decision 2: same-day reviews

SM82 dropped every review on the same day as the card's previous one (about 30% of the time-labelled reviews) because with normal inputs the instant row's interval runs answer to answer and so contains that review's own answer time. On show-time inputs the instant row's interval is `floor(max(elapsed_seconds - duration / 1000, 0))`, the time up to the moment the card is shown, and every other input of an instant row is either a count, a day value, or the card's earlier history (`rwkv/data_processing.py add_queries`: `duration`, `scaled_duration`, rating and label columns are zeroed on query rows). Audit (CPU proof C4a, three models, a training user, 3 same-day and 3 later-day reviews): give the review a 5 s longer answer while keeping the moment the card was shown (duration + 5 s, its answer-to-answer interval + 5 s, the card's next interval - 5 s). With show-time inputs, every input row up to that review's instant row and every frozen feature of the instant rows predicting that review or an earlier one must stay bitwise equal (and later features must change); with normal inputs the review's own instant row must change (the leak SM82 avoided). Static part: show-time and normal batches may differ only on instant rows and only in the 6 interval columns 2-7 (scaled elapsed seconds, its sin / cos, scaled cumulative, its sin / cos), and instant rows carry a zero duration input. **Decision: same-day reviews are included** in training and scoring when C4a passes (the launcher refuses to start without a passing proof). Label rule: a review has a time label iff its time is recorded (duration > 0) and it is not the card's first review (first reviews have no instant row).

Effect on the rows: training 117,634,406 labelled rows (35,727,965 same-day; SM82's rule: 81,906,441), about 14,360 steps of 8,192; scoring 30,445,421 time-labelled test reviews of the 723 users (21,783,607 on a later day than the card's previous review). Every time metric is also reported on that later-day subset (the rows of SM82c and of the shipped head of the show-time retrain (SM92)), so the numbers stay comparable with the earlier heads.

### Decision 3: the show-time retrain (SM92) is retrained

Its shipped head (`wt/sm92/artifacts/small-model/sm92-head/master.pth`, gate 1.9580) was trained by the SM82c scripts on normal inputs and gated on normal-input grade rows (`sm92/head/gpu/grade-check-rows.jsonl` has no `inputs` field), while the fork runs the model on show-time inputs. So it is retrained under the identical SM117 recipe (last in the order: a Small two-head model is the case closest to SM82c), and the shipped head is scored on the same rows as a reference predictor (`shipped`).

### Decision 4: instant-only models

The head reads the instant head's hidden layer (`head_p`), which is trained in all six models, and the grade split P(grade) always comes from the instant (rating) head `p_rating` = softmax of `p_logits` at the instant row, in two-head and instant-only models alike. Curve outputs (`p_curve`, `w`, ahead logits) are never read: the curve head of the instant-only models (SM98, SM101, SM99) is untrained and must not be used.

## The measurement: P(recall within S seconds)

### What "answer time" means here (added 2026-11-15 after launch, at the user's request; documentation only, no rule changes)

The answer time is the **whole time on the card, front and back together**: from the moment the question is shown to the moment a grade button is pressed. Anki starts the card timer when the question is displayed (`qt/aqt/reviewer.py`, `self.card.start_timer()`) and reads it only when the card is answered (`self.card.time_taken()`), capped at the preset's maximum answer seconds (60 s by default). That single number is the revlog `time` column, the dataset's `duration`. The moment "Show answer" is pressed is **not stored** anywhere, so front-only (recall) time and back-only (reading the answer) time cannot be computed, either for anki-revlogs-10k or for a real collection's history. So every result of this experiment reads as "recalled (Hard, Good or Easy) **and answered** within S seconds of the question appearing, reading the back included". It is not a pure recall latency. The user chose (2026-11-15) to keep this definition rather than add front/back recording to the fork. Every write-up, factory row, package `INTEGRATION.md` and fork document about the time heads must state it.

For each time-labelled test review, at its instant row (show time):

p(S) = sum over g in {Hard, Good, Easy} of P(g) x P(t <= S | g),

P(g) from the model's instant head, P(t <= S | g) from the head's 16 bins of grade g: the bins below the one holding S, plus that bin's share linear in ln time, (ln S - ln lower edge) / (ln upper edge - ln lower edge) (bin 0 starts at 0.25 s as the median rule; S is never an edge). Label: grade >= Hard and recorded duration <= 1000 S ms.

- **S grid**: 3, 5, 8, 10, 15, 20, 30 s.
- **Rows**: the 30.4M time-labelled test reviews (label DB test reviews, SM117 rule) of the users scored. A user whose own time cap (`time_user_cap`, Anki's "maximum answer seconds" when below 60 s) is at or below S has no observable event at that S: all their rows are left out at that S (whole users, independent of the label).
- **Pool**: the 722 users of the 732 set (the 723 fit users that also have the big model (1 leak-free epoch) rows, as SM82c), minus any user a model could not score (reported).
- **Predictors**: the head (`model`), the explicit-only ablation, for the show-time retrain (SM92) the shipped head, and the baselines:
  - (a) the model's own P(g) x the user's running share of earlier answers of grade g at or under S: F_u(g, S) = (n_g(t <= S) + 10 F0(g, S)) / (n_g + 10), counted over the user's earlier time-labelled reviews (causal);
  - (b) the user's running rate of the event: E_u(S) = (n(recalled, t <= S) + 10 E0(S)) / (n + 10);
  - F0 / E0 are pooled over the 1,991 training users (`priors.json`, none in the 732 pool): E0 = 0.201, 0.374, 0.529, 0.593, 0.681, 0.726, 0.772 at the S grid.
- **Metrics per model and S**: binary LogLoss (p clipped to [1e-6, 1 - 1e-6]), Brier, ECE over 20 equal-mass bins, AUC, the reliability table (rows, mean predicted, observed per bin); ECE, AUC and the table come from per-user histograms over 1,000 fine logit bins in [-8, 8] (equal-mass bins = contiguous fine bins cut where the cumulative count crosses k / 20; AUC counts pairs inside one fine bin as half; CPU proof C9: within 0.002 / 0.001 of the exact per-row values). Paired head - baseline differences of LogLoss and Brier, review-weighted, with user-bootstrap 95% CIs (2,000 resamples, seed 0).
- **Time part alone**, as `sm92-head-decision.md`: censored 16-bin time LogLoss given the true grade and the mean |ln error| of the predicted median (uncensored rows), also grade-mixed (P(g)-weighted), for the head, the ablation, the shipped head (SM92 only) and SM82's baselines (c): the card's last recorded time and the user's running median per grade, each with pooled per-grade residual CDFs (`time-baselines.jsonl`, recomputed on the SM117 rows: card's last 2.4735, user median 2.3828 time LogLoss on the 723 users). All rows and the later-day subset.

### Gate per head (go / no-go, fixed)

Pass iff (1) **grade unchanged**: the show-time grade scorer (`small_model_heldout.py`, w8+s8, `--inputs e2s-query --split-existing --deleted-cards neighbour`) gives every grade field and every per-review dump array bitwise equal for the release and the trunk written after the head training, on SM82c's 5 grade users (4396, 722, 1286, 3455, 3781); and (2) the head beats **both** SM82 time baselines (card's last time, user's running median per grade) on the censored time LogLoss **and** the mean |ln error|, each paired review-weighted difference with its 95% CI below zero (722 users, all rows). The recorded show-time rows of each model are compared too (information only).

### What "reliable" means (a report, not a go / no-go; fixed)

At a given S the head's P(recall within S seconds) is **reliable** iff its ECE (20 equal-mass bins) is below 0.01 **and** its LogLoss is below both baseline (a) and baseline (b), each paired difference with the 95% CI below zero. A model is reliable iff it is reliable at all seven S. Reported with it: Brier, AUC, the largest bin gap of the reliability table per S, and the same numbers for the ablation and the baselines.

## CPU proof (`scripts/sm117_check.py`, models Tiny (SM106), Small (SM98), Big (SM91s2))

C0 masters (all six hashes), C1 frozen training on CPU (trunk bitwise, nothing trainable, ln 16 start, head sizes), C2 resume bitwise, C3 capture = stock forward bitwise (Big: plus `head_p` feeds `p_linear` exactly), C4a the show-time audit above, C4b explicit features causal, C5 labels = plan (same-day rows counted), C6 loss = metric, C7 grade scorer equal for release vs trunk and the SM117 scorer on 2 users, C8 training path = scoring path, C9 R-under-S arithmetic vs brute-force loops, histogram ECE / AUC vs exact, running baselines vs a loop and causal. Report: `check-cpu-sm117.json`.

## GPU plan (`launch-sm117.sh`, queue owner small-model, purposes start with "time heads for the family (SM117): ")

Expected minutes from the answer-time head on richer frozen features (SM82c) (Small head training 46.9 min for 136.1M forward reviews, time scoring of 723 users 12.7 min) and the show-time big model (SM91s2) scoring (723 users in 15 min, about 1.4x the Small scorer); the budget is what a ticket needs free before the next break to start.

| step | expected | budget |
|---|---|---|
| per Small / Tiny model: smoke, head training, grade check (5 users) | 3 + 55 + 7 min | 12 + 90 + 10 |
| per Big model: smoke, head training, grade check | 5 + 95 + 10 min | 20 + 170 + 15 |
| scoring per Small / Tiny model (723 users; a pass 2 at cap 0.92 only for users that failed at 0.8) | 18 min | 30 (+15) |
| scoring per Big model | 25 min | 45 (+15) |
| decide + report (CPU) | 1 min | 10 |

Total about 4 x 65 + 2 x 110 + 4 x 18 + 2 x 25 = **about 600 GPU min (10 h)**. Order: the Tiny two-head model (SM106) (no answer-time estimate at all today), the Small Instant-Only model (SM98), the Tiny Instant-Only model (SM101), the show-time instant-only big model (SM99), the show-time big model (SM91s2), the show-time retrain (SM92), then the scoring of all six, then `decide` per model and `report` -> `sm117-result.{md,json}`.

Timing: the GPU runs the longer cool-down for Tiny (SM115) (expected end about 14:15 Sunday 11-15, or about 05:30 if its early stop after pass 1 fires). The queue is first-in first-out, so the launcher enqueues nothing until the SM115 launcher log ends with a `done` or `STOP` line (a ticket waiting earlier would run between SM115's tickets and delay it). Every ticket then starts only if its budget fits before the next break (Sunday from 16:00, daily 20:00-01:00, quiet.json). If SM115 ends at about 14:15, the Tiny two-head model (SM106)'s smoke and training (12 + 90 min budget) still start Sunday only if the GPU is free by about 14:20; otherwise everything starts Monday 01:00 and the result is expected about 11:30 Monday 11-16 (about 10:30 if the Tiny training ran Sunday; about 16:00 Sunday plus the Monday remainder if the early stop frees the GPU at 05:30). Later if other tickets share the queue.

### Grade check for the big models: one model per scorer process (added 2026-11-15 15:30, no rule change)

The first smoke test of the show-time instant-only big model (SM99) scored the release and the trunk after head training in one scorer process and found them unequal (p_ahead and p_instant differ by at most 3e-7 on 2 users). The weights are identical (all 504 tensors `torch.equal`). A release-vs-itself run gave the same differences, at the same rows: in one process the second big (fork) model differs from the first by float32 rounding, while the same model in the same position is bitwise equal across processes (the trunk scored second equals the release scored second, bitwise, on both users). The grade check therefore now scores the release and the trunk in two separate scorer processes (each first in its process) and still requires bitwise equality. The Small and Tiny checks already passed in one process and are not re-run. Before that, the smoke test had stopped because the head scripts never registered the CUDA `compress::wkv_q` kernel the big models use (fixed in 42c916ea, with the scorer's self-test).

### Out of memory in the big head training (added 2026-11-16 01:20, no rule change)

The first full head training of the show-time instant-only big model (SM99) stopped after 3.9 min (step ~500, ~101 users): CUDA out of memory while allocating 2.29 GiB, with 5.48 GiB reserved but unallocated (fragmentation; the smoke's longest user peaked at 8.43 GiB). In the big models the OOM comes from inside TorchScript as a `RuntimeError`, which the per-user `torch.OutOfMemoryError` handler did not catch, so the whole run stopped instead of skipping a user. The training now catches both forms, frees the CUDA cache and retries the user once (the forward is deterministic, so a retry gives the same rows); a user that fails twice is skipped and listed in `train.json` as before. The scorer's error branch also frees the cache on an OOM (such users are recorded and retried by pass 2 at cap 0.92). CPU proof re-run on the changed script: PASS (22 checks, `work/check-cpu-oomfix.json`). The run resumes from `resume.pt`.
