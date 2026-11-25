# Tiny ensemble on the 1,500 users (SM169): design

Written Wed 11-25 about 09:35 (machine clock), before any row or per-review dump of this run existed. Branch `sm169`, worktree `C:/Users/jonat/workspace/wt/sm169`, cut from `origin/sm160` (3a5530cc, the result commit of the deploy-time ensemble test (SM160)). No source change: `scripts/` is SM160's, so every new row and dump comes from the same code as SM160's rows and dumps. GPU scoring plus a CPU read-out; no training.

## Question

The deploy-time ensemble test (SM160) found one go: for the six-layer Tiny, the logit average `E = sigmoid(w * logit(p_IO) + (1 - w) * logit(p_TH))` of the six-layer Tiny Instant-Only (SM130) `sm130-io-dec2ep32k-t99` and six-layer Tiny two-head (SM130) `sm130-th-dec2ep32k-t91s2` instant predictions beat the Instant-Only model alone by -0.067% [-0.072, -0.061] on the 732 set at T24, with w = 0.69 fitted on 100 training users (SM160's `users-wfit.json`). **With w = 0.69 fixed and not refitted, does the gain hold on the 1,500 validation users, excluding the 100 fit users?** This is the decisive check before the user decides whether the Tiny ensemble ships.

## Models, label and scorer settings

- The two six-layer Tiny models (`models.sh`, sha256 pinned and checked by the launcher; SM160's lines).
- One label: T24 `w8+lr2.16.4q8x8@e2s-query` (the Tiny release format). No REF.
- SM160's settings: show-time inputs `--inputs e2s-query`, leak-free `--deleted-cards neighbour`, `--nonfinite-rows`, `--split-existing`, `--prior-buckets 0,1,3,5,9,16,32,128,300,1000`, `PYTORCH_JIT=0`, `--dump` (per benchmark review: `review_th`, `bin`, `y`, `p_ahead`, `p_instant`; never committed).

## Users (the 1,500 pool minus SM160's 100 fit users)

The 1,500 pool (`scripts/eval_pools.py`) is every validation user of the 8k split minus the sealed final-test users: the 732 set plus 768 users with id >= 5000. All of them are held out from the six-layer Tiny models' training. Minus the 100 fit users, 1,400 users remain; 1,388 can be scored, in two parts:

| Part | Users | Rows | Dumps |
|---|---|---|---|
| 732 part | the 732 fit list (`C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json`, 723 users) | SM160's `../sm160/gpu/sm160-rows-732.jsonl` (reused, nothing scored again) | SM160's Tiny T24 dumps, copied to `work/sm169-dump/e732/` (1,446 files) |
| new part | `users-1500-new.json`: the 1,500 pool's users with id >= 5000 and at most 650,000 reviews (SM93's rule, as the release-candidate re-check (SM164)), minus the 100 fit users: 665 users, 50,180,975 reviews; equal to SM164's `users-new-rest.json` (checked by `sm169_report.py users`) | this run: `gpu/sm169-rows-1500.jsonl` | this run: `work/sm169-dump/new/` |

The 12 users left out: the 9 users of the 732 set outside its fit list (never fit the GPU scorer) and the 3 users with id >= 5000 and more than 650,000 reviews. SM164 has T24 rows for the new part but no dumps, so its rows serve only as a cross-check of the row LogLoss. In SM164, four of these users (7562, 7625, 7775, 9526) ran out of GPU memory for the Instant-Only model even at cap 0.92; expect the same here.

Before this design was committed, the copied 732-part dumps were read once by `sm169_report.py report` (with the new part still empty) to check the copy: they reproduce SM160's Tiny figure exactly (723 users, -0.0667% [-0.0722, -0.0611]). No row or dump of the new part existed then.

## Rules (pre-registered; `sm169_report.py`, its selftest passes)

The loss, logit, dump and bootstrap code is SM160's (`../sm160/sm160_report.py`, imported unchanged). Logits of p clipped to [1e-7, 1 - 1e-7]; per-review LogLoss as `sklearn.metrics.log_loss`.

Statistic: D = 100 x (sum over users of the ensemble's benchmark instant LogLoss x size / the same for the six-layer Tiny Instant-Only (SM130) alone - 1), review-weighted, paired user bootstrap 2,000 draws, seed 0, 95% CI, on the rule users: users of both parts with a benchmark row and a dump for both models. Users whose row is a GPU out-of-memory error for either model after the cap 0.92 pass are excluded and listed.

w = 0.69 fixed (SM160's `gpu/wfit-T24.json`, Tiny; the script checks it). Nothing is refitted.

**Go** (all of):
- (a) the CI upper bound of D below 0 AND D at most -0.05% (an improvement of at least 0.05%);
- (b) at least 1,375 rule users and no non-finite row (a `nonfinite` row) for either model;
- (c) every dump reproduces its row's instant LogLoss within 2e-6.

A go means the user can decide to ship the Tiny ensemble in the fork's default mode (a later, separate change; nothing ships from this test). The curve head is untouched by construction.

Report only (decide nothing): D on each part separately; the oracle w on the rule users and its D; the per-user share that improves (users whose summed ensemble loss is below the Instant-Only model's); the two-head model's instant head alone against the Instant-Only model; the ensemble against the two-head instant head; the aggregate curve at w = 0, 0.1, ..., 1; the largest difference of this run's new-part rows against SM164's rows of the same users and label (expected below 1e-4: GPU run-to-run noise); GPU minutes per ticket.

## Run (`launch-sm169-gpu.sh`, detached)

`scripts/r01_queue.py` tickets (owner sm169, claim gpu, `HELDOUT_GPU_MEMORY_FRACTION` 0.8, 0.92 only for memory-only retry passes after a pass without progress), quiet windows respected, purpose prefix "Tiny ensemble on the 1,500 users (SM169): ". Order: preconditions (files, clean source, model sha256 pins, report selftest, user lists unchanged, the 1,446 copied dumps present); a GPU smoke ticket on the 2 smallest new users (`users-smoke.json`: 5625, 9094), whose rows and dumps go into the run's own files (counted, not scored twice), checked by `sm169_report.py smoke` (rows within 1e-5 of SM164's rows, dumps reproduce the rows); then the new part, passes until every user fits or a pass at cap 0.92 makes no progress. Expected: SM160's Tiny T24 732-set ticket took 24 minutes for 30.5M reviews with the dump, so about 40 minutes for the 50.2M reviews of the new part, plus a few minutes of retry passes for the out-of-memory users.

## Read-out (a later, fresh agent)

`sm169_report.py report` -> `sm169-result.{json,md}` (exit 0 decided, 2 pending, 3 invalid), then `RESULT.md`.

## Files

`models.sh`, `users-1500-new.json`, `users-smoke.json`, `sm169_report.py`, `launch-sm169-gpu.sh`, `design.md`; outputs `gpu/` (rows, tickets, smoke check, logs), `sm169-result.{json,md}`, `RESULT.md`, `HANDOFF.md`. Not committed: `work/sm169-dump/` (per-review values).
