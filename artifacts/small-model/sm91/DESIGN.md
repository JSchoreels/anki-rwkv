# Show-time interval retrain (SM91) step 1: design (2026-11-02)

## Question

The duration-leak scoring (SM31) found that the training data's interval `elapsed_seconds` runs from the card's previous answer to this answer, so it holds the review's own answer time, which an app cannot know when the card is shown. Scored on show-time intervals (`e2s-query`, what the Anki fork does at query time), the models lose without retraining (732 set, points = % of the big model's LogLoss): the released small model (the leak-free small model (SM72s)) +0.13 curve / +0.43 instant, the big model (1 leak-free epoch) +0.20 / +0.51 (`artifacts/small-model/sm31/sm31-result.md`). Part of that cost is train/test mismatch. Does training on show-time inputs win part of it back? If it does, step 2 (one big-model epoch on show-time inputs, ~9 GPU h) and step 3 (new teacher targets + small cool-down) follow; they are not part of this step.

## What runs

| | Arm "show-time" | Reference (reused, read only, re-scored) |
|---|---|---|
| Start | SM72s constant-LR epoch 20, w64 (the normal preset), as is | SM72s epoch 20 |
| Epoch 21 | SM72s's own epoch-21 protocol, rebuilt (`sm77_warm.py epoch --extra-options '{"show_time_intervals": true}'`): same 8,000 users, order, seeds, options, constant LR 7e-4, AdamW state of epoch 20; **inputs: show-time intervals** | SM72s's own epoch 21 |
| Cool-down | quarter-epoch 32k Muon cool-down, users / order / schedule from SM72s epoch 21 (1,991 users, 145,578,259 reviews, update offset 463,638), the reference's options + the flag | `wt/sm77/artifacts/small-model/sm77-ref-e21-dec-q32k-muon` (the tiny architecture additions (SM77) reference) |
| Scoring | leak-free (`--split-existing --deleted-cards neighbour`), w8+s8, the 732 set (723 fit users at cap 0.8 + 3 long users at 0.92), `--inputs e2s-query`; and normal inputs for information | the same on `--inputs e2s-query` (new rows); its normal-input rows exist (`wt/sm77/.../sm77/gpu/sm77-neighbour.jsonl`, label `sm77-ref-e21-q32k/w8+s8`) |

The only difference between the arms is the training inputs: same architecture, start weights, users, order, seeds, schedule, optimizer and KD replay targets (`sm91_pair.py`, checked before the epoch and again before the cool-down).

## The training inputs (track option `"show_time_intervals": true`)

Implemented in `scripts/gpu_r01_samples.py` (`show_time_query_rows`, `show_time_frame`) and `scripts/gpu_r01_prefetch.py` (the state-carry path that SM72s trains on); the flag absent leaves every batch bit for bit today's. Exactly the scorer's `e2s-query` transform (`scripts/sm31_inputs.py`): everything asked before the answer uses `floor(max(elapsed_seconds - duration/1000, 0))` (first-review sentinel -1 kept):

- the instant query row's interval, its scaled value and its sin / cos;
- the cumulative interval (the card's earlier answer-to-answer intervals + the show-time one), its scaled value and sin / cos;
- the curve query time (`label_elapsed_seconds` of the card's previous row = the next review's show-time interval), set on the whole user frame after the deleted-card transform and before the train split, as the scorer does on the full history.

Update rows (after the answer, when the full interval and the answer time are known) keep the full interval, as in the fork.

## KD teacher targets: kept unchanged in both arms

The replay targets (`artifacts/cache/kd-replay/big-leakfree-1ep-final8k-neighbour`, the big model (1 leak-free epoch) on normal, leaky inputs; looked up by `label_review_th`, alpha 0.9) stay as they are in both arms. Why:

- it keeps the pair clean: the arms differ only in the student's inputs, so the delta measures what retraining on show-time inputs buys, not a teacher change;
- the leaky teacher's targets are still sensible for a show-time student: a student that cannot see the answer time and matches them learns the teacher's prediction averaged over the answer times it cannot see, which is the right show-time target; the curve target sits at the answer-to-answer time, which differs from the student's show-time query by the answer time (at most 60-64 s on the checked users, CPU check T1) against intervals of minutes to months;
- a show-time teacher is step 2 / step 3 (the big model retrained on show-time inputs, then new targets); if this step already wins back part of the cost with the old targets, new targets can only add to it.

What this does not measure: how much of the instant head's cost comes from the teacher (alpha 0.9 puts most of the instant loss on the teacher's rating distribution). A "stop" here therefore says "retraining the student alone does not pay", not "the leak cannot be trained away".

## Pre-registered rule (`sm91_rule.py`, fixed before launch)

Users: the 722 users of the 732 set that the big model (1 leak-free epoch) rows have. Review-weighted benchmark LogLoss, paired arm - reference, both on `e2s-query` inputs, user bootstrap 2,000, seed 0, 95% CI; points = % of the big model's normal-input LogLoss on the same users (the SM31 unit).

- **go to step 2** if the arm gains >= **0.10 points** on at least one head (curve ~0.00030, instant ~0.00028 LogLoss) with that head's 95% CI entirely below zero;
- **stop** otherwise.

Also reported (not the rule): the reference's own show-time cost at this checkpoint (reference e2s-query - reference normal); the share of that cost and of SM31's +0.13 / +0.43 that the arm wins back; the arm on normal inputs vs the reference on normal inputs (what the retrain costs on answer-to-answer inputs); the 725-user 732+long pool. Self-test (`sm91_rule.py self-test`): SM31's released small model e2s-query vs normal through the same code gives +0.128 [+0.106, +0.160] curve / +0.428 [+0.337, +0.546] instant (= `sm31-result.json`), a pair with itself 0 / stop, a synthetic arm 0.2 points better on instant only: go.

## CPU proofs (`scripts/sm91.py cpu-check` -> `check-cpu-sm91.json`)

Users 2651, 8823, 2158 of SM72s's epoch-21 protocol (3,064 / 15,016 / 32,077 training reviews), deleted cards neighbour, SM72s's options.

- T1 trainer samples, flag on vs off: card features move only in the 6 interval columns (2-7) and only on instant query rows; global labels only in `label_elapsed_seconds` and only on update rows, never longer (max shortening 60-64 s); ids, permutations, review_ths, day offsets and skips equal.
- T2 trainer vs scorer: every training row (by review_th, is_query) equals the scorer's `e2s-query` row bitwise in every card feature and every label except has_label (train-split mask); the flag-off trainer rows equal the scorer's normal rows everywhere; the change (on - off) equals the scorer's (e2s-query - normal) on every row and column.
- T3 flag off == the current trainer: full training batches (`gpu_r01_prefetch.training_batches`, real options minus `packed_ipc`, span 16 x 2048) of this tree with the flag off and of the smaller model, card stream 1 layer (SM73 stage 2) tree (git archive of `eaf4128b`) are bitwise equal in every tensor; with the flag on only `batch.start` (packed features) and `batch.labels` change.
- T4 config: the arm epoch == SM72s's epoch-21 protocol except the flag and paths; a w64 cool-down prepared by this tree from SM72s epoch 21 with the flag == the reference cool-down except the flag and the folder name (start fields equal); the reference against itself is refused.
- T5 the rule's self-test above.

## Changes against the brief

- **Reference normal rows reused, not re-scored.** The share-of-cost read-out uses the reference's existing normal-input rows from the tiny architecture additions (SM77) scorer; the SM31 scorer's default rows were proven identical to its predecessor (`artifacts/small-model/sm31/cpu-proof/`). Not part of the rule.
- **Reference re-scoring runs first**, before the arm's smoke and epoch, and the launcher first waits for the smaller model, card stream 1 layer (SM73 stage 2) chain to end (its launch log ends with `done` or `STOP: ended`, at most 5 h), so its scoring tickets are not overtaken.
- **No GPU kernel check**: no kernel change; the flag only changes CPU-built input values. The first arm GPU ticket is the 2-user trainer smoke.
- Scoring tickets get 32 min (pass 1) instead of 12: the `e2s-query` scorer path took ~28 min for 723 users in SM31.
- The resumable decay (SM69) is already in this tooling: `expertium_decay.py run` snapshots every 400 users and resumes; the launcher sends a snapshot stop on `gpu/STOP` or 2 min before a window, so both the epoch and the cool-down stop between users and resume.

## Cost

GPU: reference scoring ~30 + ~10 min, smoke ~15, epoch ~110-125, cool-down ~150, arm scoring ~30 + ~15 (e2s-query) and ~15 + ~10 (normal): ~6.5-7 h with the queue free; with the 20:00-01:00 break the read-out lands after 01:00 on 2026-11-03.
