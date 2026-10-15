# Accuracy loop handoff (2026-10-14, ~17:00)

Branch `accuracy-loop` (merged into main at 26f2cb7). Rules: [docs/ACCURACY_LOOP.md](../../docs/ACCURACY_LOOP.md); results: [docs/ACCURACY_LOG.md](../../docs/ACCURACY_LOG.md); ideas: [docs/ACCURACY_IDEAS.md](../../docs/ACCURACY_IDEAS.md). Done and logged: screen baseline seeds 2025/2026, `fullrun-e10`, two-seed noise band, tier costs.

## Running detached (no session attached; the shared queue runs them in turn)

| Ticket | What | Runbook / log | Outputs |
|---|---|---|---|
| 000316 | A15: train `screen-w32-s2025-o1` and `-o2` (seed 2025, order seeds 1 and 2), then registers the third-seed training `screen-w32-s2027`, then one fast eval of all three, collections, reports, two `noise` runs | `runbook-2026-10-14b.sh` → `runbook-2026-10-14b-detached.log` | `experiments/screen-w32-s2025-o1`, `-o2`, `screen-w32-s2027`; `noise/screen-w32-s2025--screen-w32-s2026--screen-w32-s2027.json`, `noise/screen-w32-s2025--screen-w32-s2025-o1--screen-w32-s2025-o2.json` |
| 000317 | A08: fast eval of expertium-catch-up's epoch-10 decay checkpoint (`wt/expertium-catch-up/artifacts/expertium-catch-up/decay-8k-e10-q25/.../master.pth`, read-only) on the 1k screen users, then collection, report, compare vs `fullrun-e10` | `runbook-2026-10-14c.sh` → `runbook-2026-10-14c-detached.log` | `experiments/decay-e10-q25`; `comparisons/decay-e10-q25--vs--fullrun-e10.json` |
| 000324 | A15 step 2, registered by runbook 14b: train `screen-w32-s2027` (third seed). Finished exit 0 after 13.2 min | `runbook-2026-10-14b-detached.log` | `experiments/screen-w32-s2027` |
| 000328 | A15 step 3, registered by runbook 14b: one fast eval of `-o1`, `-o2`, `s2027` on the 1k screen users (~19 min). Then the runbook runs the CPU collections, reports and both `noise` runs | `runbook-2026-10-14b-detached.log` | `noise/*.json`, per-experiment eval |

Earlier tickets 000307/000308 were cancelled before running and re-registered as 000316/000317 when the runbooks were detached. Check with `scripts/r01_queue.py status` and `tail` of the logs. If a runbook died, rerun only the missing steps below. Every step is resumable or idempotent.

## Once they finish (from the worktree or main; PY = main checkout's .venv python)

```
AL=scripts/accuracy_loop.py
$PY $AL status
# only if a runbook stopped early:
$PY $AL train screen-w32-s2027
$PY $AL evaluate screen-w32-s2025-o1 screen-w32-s2025-o2 screen-w32-s2027 decay-e10-q25
for n in screen-w32-s2025-o1 screen-w32-s2025-o2 screen-w32-s2027 decay-e10-q25; do $PY $AL collection $n --threads 2; $PY $AL report $n; done
# always:
$PY $AL noise screen-w32-s2025 screen-w32-s2026 screen-w32-s2027          # seed component (now with SD)
$PY $AL noise screen-w32-s2025 screen-w32-s2025-o1 screen-w32-s2025-o2    # order component (same training seed)
$PY $AL compare fullrun-e10 decay-e10-q25                                 # A08 paired delta, 1k users
```

Log update: paste each `report` row into ACCURACY_LOG "Runs"; add seed and order rows to "Noise bands" (values, range, SD and SD×√2 from the noise JSONs). Then replace the provisional 0.0004 decision band in ACCURACY_LOOP.md "Seed-noise band" with 2 × max(seed SD×√2, order SD×√2), and say which component dominates. Record the A08 row: decay vs `fullrun-e10` (curve and instant, weighted and user-avg, 95% bootstrap), collection Δ% vs fork. The decay job's own quick-300 result was −0.0018 curve and −0.0013 instant. Update the A08/A15 statuses in ACCURACY_IDEAS.md. Commit the small records with `git add -f` (experiment.json, result.json, phases.jsonl, collection.json, eval/per_user.jsonl, epoch result.json, noise/, comparisons/, logs, results.jsonl), never checkpoints.

## Open items

- Untested hooks: `env`, `init`, `checkpoint_transform`, `evaluator`, `epoch_runner`, `--epochs-per-job`. `--order-seed` is exercised by 000316.
- After expertium-catch-up lands: re-baseline the screen with WSD decay (A03), then screen and gate s64h (A17) and width 80 (A01).
- No fork-model held-out scores in this harness yet. The default scorer builds the expertium model; compression's held-out eval B covers the fork. Fork comparisons must use `fork_unseen` (id < 5000).
- Gate and final runs, and any job over 60 minutes, need an explicit human request (ACCURACY_LOOP.md rule 6).

## Update after 000317 / A08 (2026-10-14)

000317 and `runbook-2026-10-14c.sh` finished (exit 0, 6.1 min eval). `decay-e10-q25` (epoch-10 checkpoint plus the 0.25-epoch WSD decay from expertium-catch-up) against `fullrun-e10`, 1,000 screen users, review-weighted with 95% user bootstrap:

| Subset | Curve Δ LogLoss | Instant Δ LogLoss | Better users (curve / instant) |
|---|---|---|---|
| all (1000) | −0.00136 [−0.00192, −0.00081] | −0.00103 [−0.00129, −0.00080] | 676 / 747 |
| fork_unseen (464) | −0.00152 [−0.00268, −0.00037] | −0.00116 [−0.00163, −0.00073] | 322 / 347 |
| fork_seen (536) | −0.00122 [−0.00162, −0.00088] | −0.00093 [−0.00119, −0.00070] | 354 / 400 |

Own collection, full layout (173,690 reviews), Δ% vs fork RWKV (curve 0.4417, instant 0.4366):

| Run | Curve LL (Δ%) | Instant LL (Δ%) |
|---|---|---|
| `fullrun-e10` | 0.4535 (+2.68%) | 0.4525 (+3.64%) |
| `decay-e10-q25` | 0.4536 (+2.70%) | 0.4518 (+3.49%) |

**Noise verdict.** Order noise (000324, the A15 `-o1`/`-o2` evaluation) is still pending: it waits for the GPU behind 000323-wide-kernels, so only the two-seed band exists (0.0001 to 0.0002; decision band provisionally 0.0004). Against that provisional band the gain is outside it: 3.4x on curve, 2.6x on instant, with consistent sign on both fork subsets and the quick-300 result (−0.0018 / −0.0013). Caveats: the bootstrap covers user sampling only, not training order; the full run's consecutive epochs moved 0.002 to 0.003 on quick-300, so an order band near that size would swallow the curve gain. The decay starts from the same weights and adds only the cool-down, which makes a pure order effect less likely but unproven. Collection is flat on curve (+0.0001) and 0.0007 better on instant, one user, not evidence either way. Verdict: outside the seed band, order band not yet measured, so "likely real, small".

**Recommendation (user decides).** Promote A08 to a full run: the cool-down is cheap (a fraction of an epoch from an existing checkpoint) and low-risk. Do not claim the gain as established until 000324 gives the order SD; if 2 x order SD x sqrt(2) exceeds 0.0014, treat it as unproven.

**Next steps.**
1. When 000324 lands: run the `noise` commands above, fill the Noise bands rows, replace the provisional 0.0004 band in ACCURACY_LOOP.md, and re-judge the A08 gain against it.
2. Re-baseline the screen with WSD decay (A03) before screening A17 and A01.
3. `per_user.jsonl` of `decay-e10-q25` stays uncommitted (per-user data); the aggregate comparison JSON is committed.


## Note (2026-10-14 20:45)
The runbook's `q()` does not pass `--expected-minutes`, so its tickets get the default 60 min. Add it in the next runbook. Don't edit runbook-2026-10-14b.sh while it runs, because bash reads scripts as it goes.

## Update: A08 promoted (2026-10-14)

User approved promoting A08. Main now has `final_decay_epochs` (default 0.25) in `scripts/run_full_training.py` (merge 1017383; `scripts/check_full_run_decay.py` is the CPU check). Status: promoted, provisional until A15 order band (000328). When 000328 lands, re-judge against the order SD; if the gain is inside the band, set the default to 0 (`FINAL_DECAY_EPOCHS`) and mark A08 accordingly. Next screens should use the cool-down as the baseline (A03 re-baseline).

## Update after 000328 / A15 (2026-10-14)

Runbook 14b finished at 22:11 (tickets 000316, 000324, 000328 exit 0). Noise JSONs are in `noise/`, logged in ACCURACY_LOG "Noise bands" and ACCURACY_LOOP "Seed-noise band". Review-weighted, 1,000 screen users:

| Head | Seed SD (s2025/26/27) | Seed SD×√2 | Order SD (s2025, o1/o2) | Order SD×√2 | Band 2 × max |
|---|---|---|---|---|---|
| curve LogLoss | 0.00021 | 0.00030 | 0.00220 | 0.00311 | **0.0062** |
| instant LogLoss | 0.00012 | 0.00018 | 0.00096 | 0.00136 | **0.0027** |

Training order dominates (about 10× on curve, 7.7× on instant). n = 3 per component, so each SD is uncertain by about 2×.

**A08 re-judged.** Gain −0.00136 curve / −0.00103 instant is inside the whole-training band (0.22× and 0.38×). Unproven against independent trainings. A08 is paired (same epoch-10 weights plus 0.25 epoch), so that band probably overstates its noise, but this is unmeasured. Status: promoted, provisional (default `final_decay_epochs` 0.25 stays). If the check below shows the spread of decays comparable to the gain, set `FINAL_DECAY_EPOCHS` to 0 and mark A08 accordingly.

**Queued (detached, from `C:\Users\jonat\workspace\anki-rwkv`, log `artifacts/accuracy-a08/queue-a08-decay-seeds.log`; held by the quiet window until 02:00):**

| Ticket | What | Output (main checkout) |
|---|---|---|
| 000330 | `expertium_decay.py auto` from `run-8k-20261013` epoch 10, 0.25 epoch, `one_minus_sin`, `--seed 2` (original `decay-8k-e10-q25` used `--seed 1`), `--evaluate quick`, expected 20 min | `artifacts/accuracy-a08/decay-8k-e10-q25-o2/expertium/E89-F65e-F77a/master.pth` |
| 000331 | same, `--seed 3` | `artifacts/accuracy-a08/decay-8k-e10-q25-o3/...` |

Both were `prepare`d already (different user subsets and order per seed), so `auto` goes straight to the run. Each job also writes `summary.md` (decayed vs epoch 10 on the 300 quick users; the earlier seed-1 result was −0.0018 / −0.0013).

**Follow-ups once both finish** (fresh agent, Sonnet medium; PY = main venv python, run from the accuracy-loop worktree):

```
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
AL=C:/Users/jonat/workspace/wt/accuracy-loop/scripts/accuracy_loop.py
M=C:/Users/jonat/workspace/anki-rwkv/artifacts/accuracy-a08
for s in 2 3; do
  $PY -X utf8 $AL prepare decay-e10-q25-o$s --tier screen --seed 2025 --checkpoint $M/decay-8k-e10-q25-o$s/expertium/E89-F65e-F77a/master.pth --note "A08 check: decay data-order seed $s"
done
```
(copy the extra `prepare` arguments from `experiments/decay-e10-q25/experiment.json`: model, track, options, `order_seed`, note). Then one GPU ticket for both evaluations (~13 min, fast mode), and CPU steps under `cpu-heavy`:

```
$PY $QUEUE run --owner accuracy-loop --purpose "accuracy-loop: fast eval of the two A08 decay order-seed checkpoints (~13 min)" --claim gpu --expected-minutes 16 -- $PY -X utf8 $AL evaluate decay-e10-q25-o2 decay-e10-q25-o3
for n in decay-e10-q25-o2 decay-e10-q25-o3; do
  $PY $QUEUE run --owner accuracy-loop --purpose "accuracy-loop: collection scoring $n" --claim cpu-heavy --expected-minutes 12 -- $PY -X utf8 $AL collection $n --threads 2
  $PY -X utf8 $AL report $n
  $PY -X utf8 $AL compare fullrun-e10 $n
done
$PY $QUEUE run --owner accuracy-loop --purpose "accuracy-loop: noise over the three decays" --claim cpu-heavy --expected-minutes 5 -- $PY -X utf8 $AL noise decay-e10-q25 decay-e10-q25-o2 decay-e10-q25-o3
```

Decision rule: A08 holds if all three decays beat `fullrun-e10` on both heads and the SD of the three decays (a decay-only order spread) times √2, doubled, is below the gain (0.0014 curve, 0.0010 instant); then record that paired band for continuation experiments. Otherwise treat the gain as unproven, ask the user about setting `--final-decay-epochs 0`, and say so in ACCURACY_LOG/IDEAS. Scoring `fullrun-e10` and the decays on the same 1k users is already the paired design. Per-user files of public benchmark users may be committed; collection results stay aggregate-only.

**Screening implication** (written into ACCURACY_LOOP.md as a recommendation): single-run screens cannot resolve gains below about 0.006 curve / 0.003 instant; use paired continuations or average several order seeds. The user decides.

Runbook note: queue tickets in runbooks now need `--expected-minutes` (see the 20:45 note).

## Adopted screening protocol (2026-10-14, user decision)

The A15 recommendation is adopted: screens are paired (shared init and order seed, per-user paired delta) and averaged over k = 3 order seeds. Promote only on a consistent sign across all k and |mean| > 2 × sigma × √2 / √k (about 0.0036 curve / 0.0016 instant at k = 3). k = 1 is a pre-filter only (drop if clearly worse than 0.0062 / 0.0027, never promote). Baselines `screen-w32-s2025`, `-o1`, `-o2` are reused. New command `accuracy_loop.py screen CAND=BASE ...` (smoke-checked on the A15 runs). Details in ACCURACY_LOOP.md "Seed-noise band". A08's pending test (000330/000331) already follows this design.

## Update: A08 settling evaluation queued (2026-10-15)

000330/000331 finished training the decays `decay-e10-q25-o2` and `-o3` (checkpoints in `anki-rwkv/artifacts/accuracy-a08/decay-8k-e10-q25-o2` and `-o3`). Both were `prepare`d (score-only, epochs 0, order_seed 20261013, 1k screen users) in `experiments/decay-e10-q25-o2` and `-o3`.

**Ticket 000333-accuracy-loop** (GPU claim, expected 30 min, not --timing): `accuracy_loop.py evaluate decay-e10-q25-o2 decay-e10-q25-o3`, queued detached from the main checkout behind 000332-small-model. Log: `artifacts/accuracy/queue-a08-decay-eval.log`. Outputs: `experiments/decay-e10-q25-o2/eval/` and `-o3/eval/` (`per_user.jsonl` stays uncommitted). Check with `r01_queue.py status`; if the wrapper died, rerun the evaluate command below under the queue.

Preliminary quick-300 signal (decay job summaries, NOT the official verdict), curve / instant vs epoch 10: seed1 -0.0018 / -0.0013, seed2 -0.00180 / -0.00132, seed3 -0.00228 / -0.00140.

**Follow-up commands** (fresh Sonnet medium agent, after 000333 exits 0; from the accuracy-loop worktree):

```
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
AL=C:/Users/jonat/workspace/wt/accuracy-loop/scripts/accuracy_loop.py
for n in decay-e10-q25-o2 decay-e10-q25-o3; do
  $PY $QUEUE run --owner accuracy-loop --purpose "accuracy-loop: collection scoring $n" --claim cpu-heavy --expected-minutes 12 -- $PY -X utf8 $AL collection $n --threads 2
  $PY -X utf8 $AL report $n
  $PY -X utf8 $AL compare fullrun-e10 $n
done
$PY $QUEUE run --owner accuracy-loop --purpose "accuracy-loop: noise over the three decays" --claim cpu-heavy --expected-minutes 5 -- $PY -X utf8 $AL noise decay-e10-q25 decay-e10-q25-o2 decay-e10-q25-o3
$PY -X utf8 $AL screen decay-e10-q25=fullrun-e10 decay-e10-q25-o2=fullrun-e10 decay-e10-q25-o3=fullrun-e10   # check usage with `screen --help`
```

**Verdict rule:** A08 HOLDS if all three decays (seeds 1/2/3) beat `fullrun-e10` on curve and instant AND 2 x SD x sqrt(2) of the three decays is below 0.0014 (curve) / 0.0010 (instant). If it fails, record it in ACCURACY_LOG/IDEAS and mark "needs user decision" (do not revert `final_decay_epochs`; the user decides on `--final-decay-epochs 0`). Log rows go in ACCURACY_LOG; commit small records with `git add -f`, never checkpoints or per_user files of the own collection.

## Update: A08 settled, HOLDS (2026-10-15, ticket 000333 processed)

000333 finished exit 0 (12.0 min). CPU steps done (collections, reports, compares, noise, screen). Paired deltas vs `fullrun-e10`, 1,000 screen users, review-weighted:

| Decay seed | Curve Δ | Instant Δ |
|---|---|---|
| 1 (`decay-e10-q25`) | −0.00136 | −0.00103 |
| 2 (`-o2`) | −0.00141 | −0.00107 |
| 3 (`-o3`) | −0.00157 | −0.00114 |
| mean / SD | −0.00145 / 0.00011 | −0.00108 / 0.00006 |

2·SD·√2 = 0.00031 curve (< 0.0014) and 0.00016 instant (< 0.0010); all three beat `fullrun-e10` on both heads. **Verdict: A08 HOLDS.** The provisional wording is removed from ACCURACY_LOG, ACCURACY_IDEAS, GPU_TRAINING.md and the `run_full_training.py` comment; `final_decay_epochs` stays 0.25. Paired decay band for continuation experiments: 0.0003 curve / 0.0002 instant. Note: when running `accuracy_loop.py` under `r01_queue.py run`, pass an absolute script path (the queue runs from the main checkout, so a relative path used its copy and missed this worktree's experiments). Files: `comparisons/*decay-e10-q25-o*`, `noise/decay-e10-q25--decay-e10-q25-o2--decay-e10-q25-o3.json`, `comparisons/screen--decay-*.json`.

**A08 order-check ticket ids (added 2026-10-15 from the flow audits).** GPU evaluation: 000333 (exit 0). CPU scoring steps (`cpu-heavy`): 000335-000337 failed (relative script path, queued from the main checkout; fixed by the queue's relative-path refusal), reruns 000338-000340 exit 0.
