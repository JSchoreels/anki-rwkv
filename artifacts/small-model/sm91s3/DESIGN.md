# Show-time interval retrain (SM91) step 3: design (2026-11-03)

## Question

Step 1 (`artifacts/small-model/sm91/sm91-decision.md`, GO): the small model trained on show-time inputs wins back ~2/3 of its show-time cost, but its KD teacher targets were still the big model (1 leak-free epoch) on normal (answer-to-answer) inputs. Step 2 (`artifacts/small-model/sm91s2/sm91s2-decision.md`, GO): the show-time big model (the big model (1 leak-free epoch) + one epoch on show-time inputs) beats the big model (1 leak-free epoch) on `e2s-query` by -0.055 curve [-0.117, +0.013] / -0.353 instant [-0.459, -0.276] points. Step 3: does distilling from the show-time big model, with targets computed on show-time inputs, make the show-time small model better on `e2s-query` than step 1's arm?

## What runs (the cheapest clean design)

| | New arm | Reference: step 1's arm (reused, read only) |
|---|---|---|
| Start | step 1's arm epoch 21 (`wt/sm91/artifacts/small-model/sm91-arm-e21`, as is) | the same checkpoint |
| Cool-down | step 1's quarter-epoch 32k Muon cool-down (1,991 users, 145,578,259 reviews, order / schedule from SM72s epoch 21, update offset 463,638, w64, show-time inputs), folder `sm91s3-arm-e21-dec-q32k-muon-stt` | `wt/sm91/artifacts/small-model/sm91-arm-e21-dec-q32k-muon` |
| KD replay targets (alpha 0.9) | **new**: the show-time big model (`wt/sm91s2/.../big-showtime/master.pth`, sha256 `67ee8e9b...`) replaying the full histories of the 1,991 cool-down users on show-time inputs (`scripts/sm11_replay_export.py --inputs e2s-query`, deleted cards neighbour) -> `anki-rwkv/artifacts/cache/kd-replay/big-showtime-cool1991-neighbour-e2s` | old: `big-leakfree-1ep-final8k-neighbour` (the big model (1 leak-free epoch), normal inputs) |
| Scoring | leak-free, w8+s8, the 732 set (723 users cap 0.8 + 3 long users cap 0.92), `--inputs e2s-query`, then normal inputs; label `sm91s3-arm-e21-q32k-stt`, rows `gpu/sm91s3-inputs.jsonl` | step 1's rows (`wt/sm91/.../gpu/sm91-inputs.jsonl`, `sm91-arm-e21-q32k/w8+s8[@e2s-query]`) |

The arms differ only in the cool-down's KD targets: `sm91s3_pair.py` requires the cool-down's `decay.json` / protocol to equal step 1's arm cool-down except `distillation.teacher` + `distillation.replay` and the folder name (same start sha256 `45a7e9a1...`, users, order, update offset, seeds, schedule, options, flag). The trainer and the scorer code are unchanged since step 1's launch commit `4f39eb92` (CPU proof E6).

**Confound / limitation (stated, accepted).** Epoch 21 itself replayed the old targets (both arms share it), so the new arm sees the show-time teacher only in the last quarter-epoch (145.6M reviews). The pair is clean (one difference), but it measures the teacher swap at cool-down scale, a lower bound on what a fully show-time-distilled small model would gain. The full version (new targets for all 8,000 users, ~4.8 h export, then a new epoch 21 + cool-down, ~2 h + ~2 h) does not fit before the user's 16:00 stop; it is the follow-up if this step shows a gain. Step 1's arm cool-down was resumed once (stopped at 16:17, resumed 19:01); the resumable decay (SM69) passed its GPU check bit for bit, so a segmented run is not a confound.

The brief's "37.6M reviews" is the last segment of step 1's cool-down only (its `complete:` line); the whole cool-down is 145.6M reviews (1,991 users), so the export covers those 1,991 users' full histories (145.6M reviews; 18 users > 600k run in passes with 300k burn-in).

## Why show-time targets fit the student

The student (show-time inputs) asks the curve at the show-time interval of the next review and the instant head at the show-time interval of the current review. With `--inputs e2s-query` the teacher sees exactly the scorer's `e2s-query` inputs, which step 1's proofs showed are bitwise the trainer's show-time rows; so the curve target for review j is the teacher's curve at j's show-time interval (the old target sat at the answer-to-answer interval), and the rating target is the teacher's distribution given only what is known when the card is shown. Targets are keyed as before (`label_review_th`, row = review_th - 1), so `ReplayTargets` reads them unchanged.

## Pre-registered rule (`sm91s3_rule.py`, fixed before launch)

Users: the 722 users of the 732 set that the big model (1 leak-free epoch) normal rows have. Review-weighted benchmark LogLoss, paired new - step 1's arm, both on `e2s-query` inputs, user bootstrap 2,000, seed 0, 95% CI; points = % of the big model (1 leak-free epoch) normal-input LogLoss on the same users (as steps 1-2).

- **adopt the show-time targets** if one head gains >= **0.05 points** with its 95% CI entirely below zero, and no head is worse by >= 0.05 points with its CI entirely above zero;
- **no gain** otherwise.

The bar is half of steps 1-2's (0.10): the change is smaller (targets for a quarter-epoch, not a training-input change), and a 0.05-point gain (curve ~0.00015, instant ~0.00014 LogLoss) is still clearly above the CI half-width of these pairs (~0.03-0.05 points).

Also reported (not the rule), for the new arm and for step 1's arm: vs the big model (1 leak-free epoch) on `e2s-query` (SM31 rows), vs the show-time big model on `e2s-query` (step 2 rows), vs the released small model on `e2s-query` (SM31 rows), vs the big model (1 leak-free epoch) on normal inputs (cost left); new normal - step 1's arm normal; the new arm's own show-time cost; the 725-user 732+long pool.

### Smaller-model bar (<= +0.5% curve / <= +0.4% instant), which reference for show-time models

Reported three ways (% of the reference's own LogLoss, CI from the same bootstrap): (a) normal inputs vs the big model (1 leak-free epoch) on normal inputs (the bar as defined); (b) `e2s-query` vs the big model (1 leak-free epoch) on `e2s-query`; (c) `e2s-query` vs the show-time big model on `e2s-query`.

Recommendation: for show-time models the bar should compare both models on the deployment inputs (`e2s-query`, what the fork feeds at show time) against the big model that would ship beside the small one: (c) once the show-time big model ships, (b) as long as the fork ships the big model (1 leak-free epoch). (a) measures the models on inputs neither sees in deployment and penalises a show-time model for not fitting answer-to-answer intervals it was not trained on. For reference, step 1's arm: (a) +0.59 / +0.60 (fails), (b) +0.42 / +0.14 (passes), (c) +0.48 / +0.49 (fails on instant).

Self-test (`sm91s3_rule.py self-test`): step 1's rule pair through the same code reproduces `sm91-decision.json` exactly (-0.086 [-0.126, -0.054] / -0.277 [-0.367, -0.209], users better / worse); step 1's arm with itself: 0 / no gain; synthetic instant -0.06: adopt; -0.04: no gain; instant -0.06 with curve +0.06: no gain.

## CPU proofs (`scripts/sm91s3.py cpu-check` -> `check-cpu-sm91s3.json`, run by the launcher)

Users 8081, 4001, 5804 of the cool-down (8,623 / 8,630 / 24,292 reviews).

- **E0** the show-time big model loads in the exporter's fork model (float32); sha256 recorded.
- **E1** exporter batches (`--inputs e2s-query`) == the duration-leak scoring (SM31) scorer's `e2s-query` batches bitwise in every tensor (52 fields); `--inputs normal` == the scorer's normal batches; e2s-query moves only `batch.start` and `batch.labels`.
- **E2** the show-time big model's tables cover exactly the reviews with a label (same NaN pattern as normal inputs), and curve and rating targets move (e.g. user 8081: 2,989 of 6,469 curve and 3,208 of 6,469 rating targets change).
- **E3** the export command end to end on CPU (`--workers 1`: the spawned loader worker gets the variant through `SM11_INPUTS`): tables == E2 bitwise; manifest == the old targets' manifest except `teacher`, `teacher_sha256` (== the file) and `"inputs": "e2s-query"`; `verify --inputs e2s-query` passes, a normal-input verify is refused; normal inputs + the old teacher reproduce the old cache (same manifest, same NaN pattern, curve within 1.7e-4 and ratings 2.4e-4 of the GPU-exported values, inside the verify tolerance).
- **E4** `cool-users-run.json` lists exactly the cool-down's 1,991 users with review counts == their available reviews, same dataset path / revision as the old targets.
- **E5** `sm91s3_pair.py`: new cool-down == step 1's arm cool-down except the targets; step 1's against itself refused.
- **E6** `scripts/ rwkv/ expertium/` unchanged since `4f39eb92` except files neither the cool-down nor the scorer import.
- **E7** the rule's self-test.

After the export (launcher, CPU): export status complete, `verify --inputs e2s-query --sample 8` (GPU export vs CPU recompute), `check_replay_label_windows.py` on the new cool-down (every user's file present, rows == reviews, NaN counts == the export record, no unlabelled window; no trims expected, the pair check would refuse any), the Muon fresh-state check, `sm91s3_pair.py --targets` (manifest teacher sha == the cool-down's teacher, inputs e2s-query, deleted cards neighbour, same dataset). Then the GPU trainer smoke (2 users x 512 reviews on a scratch copy: `ReplayTargets` accepts the new folder).

## The only code change

`scripts/sm11_replay_export.py --inputs {normal,e2s-query}` (default normal = unchanged, proven by E1 normal / E3d): `load_frame` applies `sm31_inputs.apply_frame(frame, "e2s-query")` after `get_rwkv_data` (deleted-card transform), and `rwkv.data_processing.add_queries` is wrapped once per process with the scorer's `sm31_inputs._query_rows_show_time`; the variant reaches the loader workers through `SM11_INPUTS` and is recorded in the manifest (`"inputs"`, absent for normal). `verify` refuses a variant other than the manifest's.

## Timeline and the 16:00 stop

GPU stop 2026-11-03 16:00 (quiet.json, the user: nothing restarts after it until the user says so). The launcher never waits through that window: when the next step does not fit before it (any window starting between 11-03 12:00 and 11-04 01:00 counts), a running cool-down gets a snapshot stop, the log says `STOP: user stop 16:00, resume needs the user's go`, and it exits (code 3). Relaunching resumes: finished steps are skipped, the export skips finished users, the cool-down resumes from its snapshot.

| # | step | expected min | clock (launch ~11:10) |
|---|---|---|---|
| - | CPU proofs + pair check | 2 | 11:10-11:12 |
| 1 | teacher targets, 1,991 users, 145.6M reviews (old export: 43 model-min for these users, ~34-40k reviews/s overall) | 60-75 | 11:12-12:25 |
| - | CPU: status, verify 8 users, label windows, Muon check, pair --targets | 3 | 12:25-12:28 |
| 2 | trainer smoke | 1-2 | 12:30 |
| 3 | cool-down (step 1: 89 + 28 min training) | 120 | 12:30-14:30 |
| 4-5 | scoring e2s-query 723 + 3 long users (step 1: 12 + 1 min) | 15 | 14:30-14:45 |
| - | rule (CPU) | 1 | 14:45 |
| 6-7 | scoring normal inputs (information) | 15 | 14:45-15:00 |
| - | rule again with the normal rows | 1 | 15:00 |

Margin ~1 h before 16:00. If the export or the cool-down runs slow, the order keeps the rule's rows (e2s-query) before the information rows (normal inputs); what does not fit waits for the user's go.
