# Release-candidate re-check of the final state formats (SM164): design

Pre-registered Tue 11-24 (about 23:35), committed before any row of this run exists. Branch `sm164`, worktree `C:/Users/jonat/workspace/wt/sm164`, cut from `origin/sm152` (8f4604c7), the branch that added the int6 factor label `q6` to `scripts/state_lowrank.py` and produced the F6 rows reused here. No source change: `scripts/` is SM152's, so every new row comes from the same code as the reused F6 rows. No training.

## Question

The user's final state formats (11-24): Big and Small on F6, T24 with int6 factors (`w8+lr2.16.4q6x8`); the six-layer Tiny on T24 (`w8+lr2.16.4q8x8`). **Do the six release-candidate models stay inside the user's budgets on the 732 set (the standard read-out) and on the 1,500-user pool (the decisive check)?**

Models: Big two-head (SM91s2) `sm91s2-big-showtime`, Big Instant-Only (SM99) `sm99-big-inst-e2`, Small two-head (SM118) `sm118-e26-dec2ep32k-t91s2`, Small Instant-Only (SM113) `sm113-inst-e35st-st-dec2ep32k-t99`, six-layer Tiny two-head (SM130) `sm130-th-dec2ep32k-t91s2`, six-layer Tiny Instant-Only (SM130) `sm130-io-dec2ep32k-t99` (`models.sh`, sha256 pinned, the SM152 copy). Inputs: show-time (`e2s-query`), leak-free (`--deleted-cards neighbour`), the standard.

## Labels (`labels.sh`)

- REF `w8+s8hx8`: int8 card and note rows, the baseline of every budget.
- T24 `w8+lr2.16.4q8x8`: the six-layer Tiny's final format.
- F6 `w8+lr2.16.4q6x8`: the Big and Small final format.

## Pools

- **732 set**: the 722 rule users (`showtime_report.pools()`, as every 732 read-out). Fork-unseen.
- **1,500 pool**: `artifacts/small-model/eval-users-1500.json` (`scripts/eval_pools.py`): every validation user of the 8k split, minus the sealed final-test users = the 732 set + **768 users with id >= 5000**. The users with id >= 5000 are **training users of the Big models** (and the fork): `eval_pools.py` says the pool is "valid for small-vs-small comparisons only".

## What is reused and what is scored (nothing is scored twice)

Rows, the first source wins per (model, label, user):

| source | labels | users |
|---|---|---|
| low-rank int8 states (SM148) `../sm148/gpu/sm148-rows-732.jsonl` | REF, T24, all 6 models | 732 set (723 fit users, Big 718) |
| int6 factors / int4 shifts (SM152) `../sm152/gpu/sm152-rows-732.jsonl` | F6, all 6 models | 732 set (723, Big 718) |
| deploy-time ensemble (SM160) `wt/sm160/.../gpu/sm160-rows-wfit.jsonl` (sha256 `c03f1ac7...`, complete: SM160 GPU DONE 11-24 15:26) | REF, T24, all 6 models | 100 users with id >= 5000 (`--pool 1500`) |
| **this run** `gpu/sm164-rows-new.jsonl` | the rest, below | the ids >= 5000 part |

The bounded-memory long-card scorer (SM155) scored only the 50 long-card users (Big T24 / REF, CPU): no pool rows to reuse. SM160's 732-set REF / T24 rows duplicate SM148's and are not used.

Scored now (GPU, `--pool 1500`; every list is a subset of the ids >= 5000 part, rule in `sm164_report.py users`):

| job | labels | users file | users | reviews |
|---|---|---|---:|---:|
| Tiny, 2 models | T24, REF | `users-new-rest.json` | 665 | 50.2M |
| Small, 2 models | F6 | `users-new-all.json` | 765 | 55.7M |
| Small, 2 models | REF | `users-new-rest.json` | 665 | 50.2M |
| Big, 2 models | F6 | `users-big-all.json` | 754 | 49.9M |
| Big, 2 models | REF | `users-big-rest.json` | 654 | 44.4M |

- `users-new-all.json`: the 768 users with id >= 5000 with at most 650,000 reviews (the instant model check (SM93)'s scoring rule for this pool: larger users never fit) = 765. `users-new-rest.json` = that minus SM160's 100 users (they have REF and T24 rows).
- `users-big-all.json`: `users-new-all.json` with at most 450,701 reviews, the largest user the Big models fit on the GPU in SM148 / SM152 (their 5 misses have 424k to 531k reviews) = 754; `users-big-rest.json` = that minus SM160's 100.
- Small and Tiny F6 / T24 on the 732 set are not rescored; Small and Tiny F6 for Tiny, T24 for Small and Big are not needed.

## Rules (pre-registered, `sm164_report.py`; selftest passes)

Statistic: % = 100 x (sum of the label's LogLoss / sum of the baseline's LogLoss - 1), review-weighted over the benchmark reviews, paired user bootstrap, 2,000 draws, seed 0, 95% CI (`showtime_report.compare`, sha256 pinned; the SM148 / SM152 statistic). Heads: two-head models curve and instant, Instant-Only models instant. Budget = the CI upper bound, per head.

- **732 rule (the user's budget, as SM152):** Big final label vs its own int8: upper <= **+0.15%**. Small and Tiny final label vs the matching Big model at int8 (two-head vs Big two-head (SM91s2), Instant-Only vs Big Instant-Only (SM99)): TOTAL upper <= **+0.5%** (Small) / **+1.0%** (Tiny).
- **1,500 rule (decisive):**
  - Big: final label vs its own int8 on the 1,500 set, upper <= +0.15% (a comparison of a model with itself is valid on its training users).
  - Small and Tiny: a direct comparison with Big on the ids >= 5000 measures Big's in-sample advantage, not the state format. The total is therefore decomposed into its two parts, each read where it is valid: **upper(own int8 vs Big int8, 732 set) + upper(final label vs own int8, 1,500 set) <= +0.5% / +1.0%**. Adding the two upper bounds is conservative.
- **Verdict per cell (model, head):** PASS if both rules pass; FAIL if the 732 rule fails, a row is non-finite, or (Big) the 1,500 rule fails; **HOLD** if a Small / Tiny cell passes the 732 rule and fails only the 1,500 rule (reported to the main session; the user decides).
- **Verdict per size:** GO if every cell of both models passes, NO if one fails, HOLD otherwise.
- **Coverage (else `pending`, exit 2):** at least 715 users on the 732 set per label used (as SM152), and on the ids >= 5000 at least 97% of the scoring list per label used (Small / Tiny lists 765 -> 743, Big 754 -> 732; SM160's users count).
- **Report only, never a verdict:** the direct Small / Tiny vs Big comparison on the 1,500 set and on the ids >= 5000 alone; own int8 vs Big int8 on the ids >= 5000 (the in-sample effect itself); every label vs own int8 on the 732 set and on the ids >= 5000 alone.
- **Warnings, not rerun (the rule says warnings only):** long cards (128+ answers) and per-card |dR| are carried from SM152 (Small and Tiny F6), SM148 (T24, every size), SM155 (Big T24, all 50 long-card users). Big F6 has no long-card read-out; that is stated, not run.

### Seen before writing the rules (disclosure)

The 732-set numbers are known from SM148 / SM152 (all six cells pass; tightest: six-layer Tiny two-head (SM130) curve, upper +0.899% vs +1.0%; Small two-head (SM118) curve, upper +0.409% vs +0.5%). While checking the reuse I also read SM160's 100 users with id >= 5000: there, six-layer Tiny two-head (SM130) T24 vs Big two-head (SM91s2) at int8 is +3.1% (curve) and +1.3% (instant), against +0.72% / +0.75% on the 732 set. That size of gap is Big's in-sample advantage on its own training users, and it is why the Small / Tiny 1,500 rule is decomposed rather than read directly. The format parts on the ids >= 5000 were not looked at.

## Integrity

- GPU smoke ticket (first ticket, STOP on failure): 5 smoke users x 6 models x REF, T24, F6 (`users-smoke.json`): SM152's 3 smoke users (2388, 4676, 1866) must equal SM152's GPU smoke rows, and the 2 smallest of SM160's 100 users must equal SM160's REF / T24 rows, within **1e-5** per user and head, with equal review counts and finite rows. This also checks that `--pool 1500` (here) and `--pool 732` (SM152) give the same rows.
- Launcher preconditions: clean source tree, model sha256, SM160 rows sha256, report selftest, users lists equal to their rule.

## Run (`launch-sm164-gpu.sh`; `DRYRUN=1` for a dry run)

Queue tickets in the order Tiny, Small F6, Small REF, Big F6, Big REF; passes at cap 0.8, then 0.92 after a pass without progress (memory only); `PYTORCH_JIT=0`; quiet windows respected (`sm135_block.py`). Expected (SM160's per-label 732-set minutes scaled by reviews): smoke 2 min, Tiny about 45 min, Small F6 about 47, Small REF about 20, Big F6 about 65, Big REF about 23: about 3.4 GPU hours with retries. CPU read-out afterwards: a few minutes, under 1 GB.

## Files

`models.sh`, `labels.sh`, `users-{new-all,new-rest,big-all,big-rest,smoke}.json`, `sm164_report.py` (users, smoke, report, selftest), `launch-sm164-gpu.sh`, `HANDOFF.md`. Outputs: `gpu/` (log, tickets, smoke rows and check, `sm164-rows-new.jsonl`), `sm164-result.{json,md}`.
