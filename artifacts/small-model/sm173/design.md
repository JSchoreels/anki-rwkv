# Deleted-card note ids (SM173): design

Written Wed 2026-11-25, before any 732-set score of the arms below (only a 3-user CPU smoke of the hook ran first, to check the patch: `cpu/smoke-cpu-rows.jsonl`). Branch `sm173` from `origin/main`, worktree `C:/Users/jonat/workspace/wt/sm173`. Run tree `C:/Users/jonat/workspace/wt/sm173-run`: a detached worktree at `e8ab44d4` (`origin/sm161` head, which holds the scorer with `--inputs e2s-query`, `--deleted-cards neighbour` and `--split-existing`), **never edited**. Track: small model, next-family item.

## Cause (found before this design)

The collapse comes from the shared preprocessing, `rwkv/data_processing.get_rwkv_data`, which both the training data and the scorer go through (training: the replay export, the LMDB build and the carried-state trainer's sample path; scoring: `scripts/sm31_inputs.samples` and `scripts/small_model_heldout.py`). It is not in the scorer's `--deleted-cards neighbour` transform (`scripts/sm71_leak.neutralise` leaves the note id as it is) and not in the fork.

1. **Note lanes.** The revlogs are merged with the cards table by a left join, so for any user with at least one card missing from the cards table (deleted before the export) the `note_id` column becomes float64. The deleted card's note id `ID_PLACEHOLDER + card_id` (314159265358979323 + card id, about 3.14e17) is exact in int64 but is then stored in that float64 column, where the spacing of representable numbers is 64. The dataset's card ids are small per-user integers (0, 1, 2, ...), so all deleted cards whose card ids fall in one 64-wide block get the same note id and share one note lane (one note memory) in `create_sample` (`_module_group_locations` groups on the pandas column). The smoke users: 362 deleted cards on 19 lanes (user 4), 1,078 cards on 18 lanes (user 311, every card deleted).
2. **Note codes.** `create_sample` casts the id column to int32 for the random id codes (`rwkv/prepare_batch.prepare`). A float64 value of 3.14e17 is out of int32 range and becomes -2^31, so in scoring all deleted cards of a user share ONE note code. The carried-state trainer (`scripts/gpu_r01_samples.create_sample_fast`, `int64_ids=True`, the release cool-downs) casts to int64 instead: one code per 64-block in training. The comment there already notes that "the int32 ids collide for placeholder note ids".
3. **The fork.** Not affected. `rslib/src/rwkv/engine.rs` `id_encoding_keys` builds the placeholder as `ID_PLACEHOLDER + card_id` in i64, and `qt/aqt/rwkv_inference/process.py` (the Python replay row builder) in Python integers: unique per card, both for lanes and codes. In practice the fork never meets a placeholder: a deleted card's reviews are simply absent from the collection (the structural gap of the deleted-card leak check (SM71) and the fork input parity check (SM161)). Fork worktree read: `wt/fork-parity` at 2de493c83 (`origin/rc-2026-11` + the SM161 test).

The real note ids of the existing cards are small integers, exact in float64 and in int32: only deleted cards are touched.

## Question

Phase 1: how large is the collapse in the training data and the evaluation pools, and what does it cost the released models at inference? If the deleted cards had a note of their own, as the integration rules say (`docs/SMALL_MODEL_INTEGRATION.md`: "Missing -> 314159265358979323 + card id (unique per card)"), would six-layer Tiny two-head (SM130) and Small two-head (SM118) score better on the 732 set?

## Phase 1a: size (CPU, `sm173_measure.py`)

Per user, aggregate counts only (no card ids): reviews, deleted-card reviews and cards, placeholder lanes (`float64(ID_PLACEHOLDER + card_id)` distinct values, exactly what `get_rwkv_data` stores), the deleted cards and reviews whose lane holds more than one card, the largest lane. Pools: the 8,000 training users of the 8k split (`run.json` `train_users`, the release models' training users), the 732 set, the 1,500 pool. Output `sm173-measure.json` (pool summaries) and `sm173-measure-users.jsonl` (per-user counts).

## Phase 1b: effect at inference (GPU queue, `sm173_score.py`)

No training. The scorer of the run tree, unchanged, behind a hook selected by `SM173_ARM` (installed at import, so the data-loader workers get it too):

| arm | note lanes of deleted cards | note codes of deleted cards | other codes |
|---|---|---|---|
| `cur` | today: one per 64-block | today: one for all (-2^31) | seed 2025 + user |
| `fix` | one per card (int64 `ID_PLACEHOLDER + card_id`, `note_id` column int64) | one per card | re-drawn for users with deleted cards (the set of note codes changes, so the seeded draw order moves: a reseed) |
| `lanes` | one per card (`_module_group_locations` groups placeholder rows by card) | today: one for all | exactly today's (no reseed) |
| `reseed` | today | today | seed 2025 + user + 1 (`--seed-offset 1`): the noise floor of a reseed |

`fix` is the change that would go into the next family's data. `lanes` isolates the memory-mixing part without any reseed. `reseed` shows how much of `fix` vs `cur` a pure reseed could explain. Users without deleted cards get bitwise the same inputs in `cur`, `fix` and `lanes` (checked on the smoke user 27, and checked again on every such user of the 732 set by the report).

Models (fp32 weights and states, the release checkpoints): six-layer Tiny two-head (SM130) `expertium:w64-c1-d2p1u1` `wt/sm130/artifacts/small-model/sm130-th-dec2ep32k-p2/expertium/E89-F65e-F77a/master.pth`; Small two-head (SM118) `expertium:w64` `wt/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/E89-F65e-F77a/master.pth`. Users: the 732 set minus the 9 never-fit users (`wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json`, 723 users). Scorer flags (as the low-rank int8 states experiment (SM148)): `--pool 732 --users-file <that file> --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows`, `PYTORCH_JIT=0`, GPU cap 0.8 (0.92 for a retry pass without progress; memory only). One queue ticket per arm (both models in one process), purpose prefix `Deleted-card note ids (SM173): `. Rows: `gpu/sm173-rows-732.jsonl`.

## Rules (pre-registered; `sm173_report.py`)

Statistic as SM148: % = 100 x (sum over users of LogLoss x size of the arm / the same sum of `cur` - 1), review-weighted over the benchmark reviews, per head (curve, instant), paired user bootstrap, 2,000 draws, numpy `default_rng(0)`, 95% percentile CI, on the rule users (users with a finite row for all 8 labels). Three review scopes from the scorer's rows: **overall** (`benchmark`), **affected** (`benchmark_deleted`: the reviews of deleted cards; the measurement gives the share of them on a shared lane), **existing** (`benchmark_existing`). For the affected scope the bootstrap draws users among those with affected reviews.

- **Primary:** `fix` vs `cur`, per model and head, overall and affected.
- **The fix matters** (verdict MATTERS) if, for at least one model and one head, (A) overall: the change is <= -0.05% with the CI upper bound below 0, or (B) affected reviews: the change is <= -1.0% with the CI upper bound below 0; and no model and head is worse overall with the CI lower bound above +0.05%. Then the fix goes into the next family's training data (the fork needs no change, see Cause 3) and phase 2 (a short fine-tune screen on six-layer Tiny two-head (SM130), fixed inputs vs a control) is designed and launched.
- **Harmless** (verdict HARMLESS) if neither (A) nor (B) holds for any model and head: the issue is recorded as harmless and the item stops (the int64 fix may still be folded into the next family's data at no cost, but no screen is run).
- **Mixed** (verdict MIXED) if (A) or (B) holds somewhere and some model and head is worse overall with the CI lower bound above +0.05%: the main session decides on adopting the fix; phase 2 is still designed and launched, because a model trained on collapsed lanes may lose at inference from clean lanes while gaining from them in training, which is exactly what the fine-tune screen separates.
- **Coverage:** at least 720 rule users and no non-finite row, else `pending` and the main session decides.
- **Reported, no verdict:** `lanes` vs `cur` and `reseed` vs `cur` (same scopes), `fix` vs `reseed`; existing-card reviews; users with deleted cards only; the bitwise check of the unaffected users.

Why inference only can understate the training effect: the released models were trained with collapsed lanes (and, through the carried-state trainer, one code per 64-block), so they may have learned to live with fake note siblings; the inference test measures what giving them clean note memory is worth today. Under `neighbour` the deleted-card flags are 0, so the models cannot tell a deleted card from an existing one.

## Phase 2 (only if MATTERS or MIXED)

A short fine-tune screen on six-layer Tiny two-head (SM130), like the state-energy penalty screen (SM171): two arms from the release weights and Muon state, same users and schedule (about 2,550 updates, 67.3M reviews), `fix` inputs vs today's inputs (the control), scored on the 732 set with the `fix` inputs. Designed in a section of this file before it is launched.

## Budget

GPU: 4 tickets, about 10 min each (fp32 scoring of the 732 set: about 2.5 min per Tiny label, about 5 min per Small label, plus start-up), about 40 min, plus a short retry pass if users are missing. CPU: the measurement (about 15 min, under 1 GB) and the report (seconds).
