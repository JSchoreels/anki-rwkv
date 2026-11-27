# Accuracy loop

The accuracy counterpart of the [R01 speed loop](GPU_RETRAIN_LOOP.md). The speed loop asks "same weights, faster?" and promotes on bitwise parity plus a timing-noise band. This loop asks "better held-out predictions?" and promotes on paired, seeded LogLoss deltas that exceed a measured seed-noise band. Ideas live in [ACCURACY_IDEAS.md](ACCURACY_IDEAS.md), results in [ACCURACY_LOG.md](ACCURACY_LOG.md), and the harness is `scripts/accuracy_loop.py`. [AGENT.md](../AGENT.md) still applies: provenance, `—` for unmeasured values, never deleting ideas, and the explicit-request rule for broader validation.

## Data and tiers

All tiers are fixed subsets of one user-disjoint split, the full run's `run.json` (`artifacts/full-run/run-8k-20261013/run.json` in the main checkout: 8,000 `train_users`, 1,500 `validation_users` of which 300 are `quick_validation_users`, and 500 sealed `final_test_users` including the preserved 100-user R01 final-test cohort). `scripts/accuracy_loop.py tiers` derives them once into `artifacts/accuracy/tiers.json` (seed 20261014, stratified by history-length decile, recorded with the split's SHA-256).

| Tier | Training users | Epochs | Evaluation users | Decides |
|---|---|---|---|---|
| screen | 2,000 (subset of the gate's) | 1 (+ short decay once A03 lands) | 1,000 validation users: the 300 quick-validation users plus 700 stratified others | drops ideas only |
| gate | 5,000 (contains the screen's) | 1 (+ short decay once A03 lands); more only when the idea needs it | all 1,500 validation users | promotes ideas |
| final | all 8,000 | multi-epoch (10 by default) | all 1,500 validation users; the 500 final-test users are scored once and reported separately | confirms a promoted bundle |

- The split has 1,500 validation users, not 2,000: the full run's "2,000-user final evaluation" is validation plus the sealed final test. The gate therefore uses all 1,500 validation users and never the final-test users.
- **Sealed users.** `final_test_users` are evaluated only in a `final` tier experiment, at most once per final candidate, and never used for decisions. The harness writes an empty `final_test_users` list into every non-final run folder, and switches the trainer's in-training evaluation off, so training never touches them.
- **Fork-seen users.** The fork's 2.76M model (`RWKV_trained_on_5000_10000`) was trained on dataset users 5000–10000. Every record reports metrics on all evaluation users and separately on `fork_unseen` (user id < 5000) and `fork_seen` (≥ 5000). Any comparison against the fork's model, or against a model distilled from it, uses `fork_unseen` only. Of the 1,000 screen users, the tier file records how many are unseen.
- One epoch plus decay is enough to rank recipes: Expertium's 2026-10-14 table (idea A17) shows ten warmup-stable epochs + one decay epoch beat one + one by only about 0.0012 (ahead) / 0.0015 (imm) at d=64, so screens and gates train one epoch, and multi-epoch training is reserved for the final tier.
- Smaller training sets hide capacity gaps: at 100 users the 192.8k model looked as good as the 2.76M model, while Expertium measured about +0.007 at 5k users. Tuning on 200 users did not transfer for him either. That is why capacity and size decisions need the gate.

## Pairing and seeds

- A comparison is always candidate minus control on identical evaluation users, identical held-out inputs (1024-review blocks of independent 128-review samples, encoding seed 2025 + user + offset, as in the full run) and the same training seed. The training-user order per epoch is fixed by the split seed, so it is identical too.
- The experiment seed is the trainer's `training_seed` option (initialization, dropout and batch encodings; 2025 is the trainer's default and the full run's seed). Seeds 2025 and 2026 are the screen baseline pair.
- `evaluate` can score several experiments in one pass when their environment and evaluation users match; the per-user inputs are then the same tensors by construction.

## Seed-noise band

- Each tier's band is measured on its own control: identical configuration, two (later three or more) training seeds, `scripts/accuracy_loop.py noise A B`. With two seeds the band is the observed |difference| of each aggregate metric, which is what a single-seed paired comparison sees under no effect. With three or more seeds the band is the across-seed SD × √2 (the SD of a two-run difference); a promotion margin is 2 × that.
- The band is recorded per metric (curve and instant benchmark LogLoss, review-weighted and user-averaged) in [ACCURACY_LOG.md](ACCURACY_LOG.md#noise-bands), with the run names it came from. It is re-measured when the control recipe changes (for example when WSD decay joins the screen recipe).
- Reference point: Expertium's seed noise is about 0.0004 LogLoss. Training-order noise is now measured (A15) and is much larger.
- Current screen band (2026-10-14, three seeds and three order seeds, 1,000 users, review-weighted): seed SD×√2 is 0.00030 curve / 0.00018 instant; order SD×√2 (same seed, same init, different training-user order) is 0.00311 curve / 0.00136 instant. The decision band is 2 × max(seed SD×√2, order SD×√2) = **0.0062 curve, 0.0027 instant** (user-averaged 0.0053 / 0.0025). **Training order dominates**, about 10× the seed component on curve and 7.7× on instant. Each SD rests on three runs, so treat it as uncertain by about a factor of 2. Values and raw files: [ACCURACY_LOG.md](ACCURACY_LOG.md#noise-bands). The earlier provisional 0.0004 is superseded.
- **Adopted screening protocol (user decision, 2026-10-14).** Training order dominates, so a screen is a paired, order-averaged comparison, never two independent runs.
  1. *Paired.* A candidate and its baseline share init seed and data-order seed (`--seed`, `--order-seed`) and are compared per user (paired delta, bootstrap CI). Continuations (decays, fine-tunes) start from the same checkpoint.
  2. *Averaged.* A candidate is judged on k order seeds (default k = 3), each paired with a baseline on the same order seed. Let delta_i be the paired review-weighted delta on order seed i, and sigma the single-run SD of the metric across order seeds (the larger of the baselines' and the candidates' observed SD; baselines alone give 0.00220 curve / 0.00096 instant). The mean of k deltas has SD sigma × √2 / √k, so **promote to the gate only if all k deltas have the same sign and |mean delta| > 2 × sigma × √2 / √k**. At k = 3 that is about 0.0036 curve / 0.0016 instant (k = 1: 0.0062 / 0.0027; k = 4: 0.0031 / 0.0014). `accuracy_loop.py screen CAND=BASE ...` prints it per head.
  3. *Reusing baselines.* Never retrain them: the order-seed baselines are `screen-w32-s2025` (order 0), `screen-w32-s2025-o1`, `screen-w32-s2025-o2`; train the candidate with `--seed 2025 --order-seed 0/1/2` to match. `screen-w32-s2026` and `-s2027` differ by init seed only (same order as s2025), so use them only for init-seed pairs. A new baseline recipe needs its own k runs.
  4. *Cost.* k = 3 triples screen GPU time (about 13 min training plus 6 min evaluation per run). k = 1 is acceptable as a cheap pre-filter: drop a candidate only if its single paired delta is worse than the k = 1 band (+0.0062 curve or +0.0027 instant); anything else gets the k = 3 run before it is called neutral or promising. A k = 1 result never promotes.
  Commands: `prepare ... --seed 2025 --order-seed N`, `train`, `evaluate`, then `$PY scripts/accuracy_loop.py screen cand-o0=screen-w32-s2025 cand-o1=screen-w32-s2025-o1 cand-o2=screen-w32-s2025-o2` (writes `comparisons/screen--*.json`); `compare` and `noise` remain for a single pair and for measuring sigma. A08's pending decay test (tickets 000330/000331: three decays from the same epoch-10 checkpoint, orders 1/2/3, scored against `fullrun-e10`) already follows this design.
- The user bootstrap in `compare` measures user sampling only. Two seeds of the same recipe already produce a user-averaged instant interval that barely excludes zero, so an effect must clear the seed band as well as its bootstrap interval.
- Missing measured noise leaves promotion pending; unknown noise is never treated as zero.

## Promotion rules

1. **Screen drops, never promotes.** Drop an idea when its paired delta (mean over k order seeds, see the adopted screening protocol) is worse than the band for that k on instant or curve LogLoss (review-weighted). Otherwise it becomes a gate candidate. A screen win is a reason to run the gate, not a result.
2. **Capacity and size need the gate.** For changes that add or remove capacity (width, layers, heads, state size, quantization, pruning, distillation into a smaller model), the screen is informative only: it can neither drop a larger model for looking neutral nor accept a smaller one for looking equal.
3. **Gate promotes.** On all 1,500 validation users, same seed as the control: promote when the instant LogLoss delta (review-weighted) is below −band of the gate tier, its 95% user-bootstrap interval excludes zero, and curve LogLoss is not worse by more than the band. When |delta| < 2 × band, run a second seed pair and decide on the mean. Until the gate band is measured, use the screen band and say so.
4. **Final confirms.** A promoted bundle is retrained on all 8,000 users for the planned epochs, evaluated on the validation users and, once, on the sealed final-test users. A bundle validates the bundle, not each component (AGENT.md).
5. Record dropped, inconclusive and rejected results in the log and update the idea's status; never delete the row.
6. Authorization: screen runs are covered by the accuracy-loop request of 2026-10-14. Gate and final runs, and any GPU job over 60 minutes, need an explicit human request; write the exact commands as a proposal in the log.

## Evaluation modes

- `fast` (default since 2026-10-14): the EV01 path from the `eval-speed` branch (`scripts/fast_evaluation.py`, merged at `c6530b9`, see [EVAL_SPEED.md](EVAL_SPEED.md)). Inputs are bitwise identical to the full run's evaluation; merged forwards change only FP32 rounding (aggregate LogLoss within 2e-10, per-user R01 losses within 6.9e-8 of the reference), far below the ~4e-4 seed noise. About 8–10× faster (2,000 users in 612 s instead of 5,825 s). Approved for screen and gate comparisons on 2026-10-14; every per-user row and result records `eval: fast (EV01, c6530b9)`. The R01 exactness rules still govern speed promotions, not accuracy comparisons.
- `reference` (`evaluate --mode reference`): the full run's in-training evaluation re-run standalone in `scripts/accuracy_eval.py`; bitwise equal to the recorded epoch-6 and epoch-7 per-user results on 12 users (`artifacts/accuracy/checks/check-eval-check-e0{6,7}.json`). It is resumable per user and can score several checkpoints on shared inputs.
- Never mix modes inside one paired comparison.

## What every result reports

- Held-out benchmark metrics (upstream `rwkv/get_result.get_stats` on the label-DB benchmark test reviews): **curve** and **instant** LogLoss separately, RMSE(bins), AUC, each review-weighted and user-averaged, on all evaluation users and on `fork_unseen` / `fork_seen`. The R01 all-label curve and instant losses are kept in the record too; the composite training objective is never compared.
- Training cost: training reviews, training seconds, **reviews/s**, updates, process wall time per epoch; evaluation and collection wall times. Report CPU savings alongside reviews/s when relevant.
- Model facts: parameters, per-card recurrent state bytes (float32: card layers × (heads × (d/heads)² + 2d) × 4 B; width 32 = 4,352 B), checkpoint bytes.
- The user's own collection in **full layout** only (whole history as one sequence, as the fork replays RWKV; `scripts/accuracy_collection.py`, CPU, about 6.5 minutes): curve and instant LogLoss in their own columns, each with Δ% = (ours − fork) / fork × 100 against the fork's full-layout curve 0.4417 and instant 0.4366 on the same 173,690 test reviews. Positive is worse.
- Provenance: the git commit, dirty flag and source snapshot recorded by the trainer at process start, the experiment's option overrides, environment hooks, seed and tier.

## Hooks

Sibling branches plug in through per-experiment fields of `experiment.json`, set by `prepare` flags, so nothing in the harness changes when they merge:

| Field | Flag | Used for |
|---|---|---|
| `options` | `--options JSON` | merged into the accepted R01 track options the trainer reads (`training_seed`, LR schedule, optimizer, distillation teacher, chunk length…) |
| `env` | `--env JSON` | environment for every train, evaluation and collection process (`RWKV_ZERO_FEATURES`, `RWKV_QAT_SCOPE`, width switches…); experiments with different `env` cannot share an evaluation pass |
| `model`, `track` | `--model`, `--track` | trainer model name and track id |
| `init` | `--init-master`, `--init-optimizer`, `--update-offset` | start epoch 1 from given weights and optimizer state (LR decay from a checkpoint, fine-tuning) |
| `checkpoint` | `--checkpoint` | score-only experiment (no training): an existing master.pth, such as the full run's epoch 10 |
| `checkpoint_transform` | `--checkpoint-transform MOD:FN` | `FN(path, out_dir, experiment) → path` applied before scoring (PTQ, pruning) |
| `evaluator` | `--evaluator MOD:FN` | replaces the default held-out scorer (fast evaluation, full-layout evaluation); same per-user row schema |
| `epoch_runner` | `--epoch-runner MOD:FN` | replaces the epoch entry point when a branch needs its own trainer |

## Workflow

Run from the worktree or checkout that holds the code under test, with the main checkout's interpreter. Every GPU step goes through the shared queue as its own ticket (`scripts/r01_queue.py run --owner <session> --purpose "<what>" --claim gpu -- …`, see [R01_PARALLEL_SESSIONS.md](R01_PARALLEL_SESSIONS.md)), and each must stay under 60 minutes. Training and evaluation are separate jobs. `train --epochs-per-job 1` runs one epoch per ticket (exit 3 means more epochs remain); `evaluate` stops starting new users after `--max-minutes` (default 55) and exits 3, and rerunning it resumes.

```
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
$PY scripts/accuracy_loop.py tiers                                   # once per split (CPU)
$PY scripts/accuracy_loop.py prepare screen-w32-s2025 --tier screen --seed 2025
$PY scripts/r01_queue.py run --owner <me> --purpose "screen train" --claim gpu -- $PY scripts/accuracy_loop.py train screen-w32-s2025
$PY scripts/r01_queue.py run --owner <me> --purpose "screen eval" --claim gpu -- $PY scripts/accuracy_loop.py evaluate screen-w32-s2025 [other experiments …]
$PY scripts/accuracy_loop.py collection screen-w32-s2025              # CPU, about 6.5 min, 2 threads
$PY scripts/accuracy_loop.py report screen-w32-s2025                  # result.json, results.jsonl, Markdown log row
$PY scripts/accuracy_loop.py compare <control> <candidate>            # paired deltas, user bootstrap
$PY scripts/accuracy_loop.py noise <seed A> <seed B> [...]
```

Outputs: `artifacts/accuracy/experiments/<name>/` holds `experiment.json`, the run folder (`run/`, a `scripts/run_full_training.py` layout with checkpoints), `eval/per_user.jsonl`, `phases.jsonl` (wall time per phase and job), `collection.json` and `result.json`. `artifacts/accuracy/results.jsonl` gets one compact line per report. Commit the small JSON records (`git add -f`), never checkpoints.

## Tier costs

Measured wall times per phase are in [ACCURACY_LOG.md](ACCURACY_LOG.md#tier-costs). Planning numbers before measurement: training at about 189k reviews/s, held-out evaluation at about 25k reviews/s in reference mode (full run epoch 7: 22.2M reviews in 888 s) and about 230k reviews/s in fast mode, collection about 6.5–9 CPU minutes.
