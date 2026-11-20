# Six-layer Tiny with the card-only state (SM134): design (2026-11-20, pre-registered before any GPU ticket)

## Question

Six-layer Tiny (SM130) passed its Tiny-shrink rule on Fri 11-20 01:21 in both chains (CI upper bound of 100 x (six-layer / Small - 1) <= +1.00% per head):

| model | head | published w8+s8 | compact w8+s4c8nhx8 |
|---|---|---|---|
| the two-head six-layer Tiny (SM130) | curve | +0.467% [+0.398, +0.546] vs the show-time two-head cool-down for Small (SM118) | +0.483% [.., +0.562] |
| the two-head six-layer Tiny (SM130) | instant | +0.524% [.., +0.567] vs the show-time two-head cool-down for Small (SM118) | +0.545% [.., +0.589] |
| the Instant-Only six-layer Tiny (SM130) | instant | +0.564% [.., +0.607] vs the longer Small Instant-Only model (SM113) | +0.603% [.., +0.647] |

The card-only state format `w8+s4hgx4c8hx8n` (the second integer-state test (SM49)'s arm C': per-head int4 card rows with float16 scales, int8 token-shift vectors, int8 notes) stores **619 bytes per Tiny card** instead of 803 bytes compact (fork default 1,055); the note state stays at **1,315 bytes** (`artifacts/small-model/sm131/sm131-bytes.json`, arm Tiny (w64-c1): six-layer Tiny has the same one card layer and one note layer, so the same bytes). **Are the six-layer models still within the same +1.00% budget against Small with the card-only state?**

| plain name (ID) | role here |
|---|---|
| the two-head six-layer Tiny (SM130) | `sm130-th-dec2ep32k-t91s2`, final cool-down `wt/sm130/artifacts/small-model/sm130-th-dec2ep32k-p2/expertium/E89-F65e-F77a/master.pth` (sha256 48a5e724cf38b53a32d570d4d034791f1da2cfcbe878572ff03ad434d7fb75af), `w64-c1-d2p1u1`, 185,057 parameters, curve + instant |
| the Instant-Only six-layer Tiny (SM130) | `sm130-io-dec2ep32k-t99`, `.../sm130-io-dec2ep32k-p2/.../master.pth` (sha256 23f037282379199bd113a5297737c920fcbefd702b7b76eecc0095b8a31186a2), instant |
| the show-time two-head cool-down for Small (SM118) / the longer Small Instant-Only model (SM113) | the references (committed w8+s8 rows, sha256 pinned in `sm134_rule.py`) |
| the second integer-state test (SM49) | the origin of the card-only format |
| compact state for Small and Big (SM131) | the pipeline this run is adapted from (scorer with `--nonfinite-rows`, one chained ticket, quiet.json-only breaks, memory guard) |

## Pre-registered rule (`sm134_rule.py decide`)

Same references, users and statistic as six-layer Tiny (SM130)'s rule: show-time inputs (`--inputs e2s-query`), leak-free (`--split-existing --deleted-cards neighbour`), 723 users of the 732 set, w8 weights, the 722 rule users of `showtime_report.py pools()` (sha256 pinned), review-weighted LogLoss, % = 100 x (new / reference - 1), paired user bootstrap 2,000, seed 0, 95% CI.

**Go iff every trained head's 95% CI upper bound of 100 x (six-layer card-only / Small published w8+s8 - 1) <= +1.00%**, per chain, on all 722 rule users with finite rows (two-head: curve and instant vs the show-time two-head cool-down for Small (SM118); Instant-Only: instant vs the longer Small Instant-Only model (SM113)). Verdict: go iff both chains go. If card-only rows are missing for any rule user, `decide` exits 2 (pending), and the launcher retries the missing users.

Report only (no verdict): card-only vs the six-layer compact rows, card-only vs the six-layer published rows (both from six-layer Tiny (SM130)'s committed rows, copied byte for byte into `inputs/sm130-rows.jsonl`, sha256 f4b38086f872...), published and compact vs Small as in SM130, bytes per card / note (619 / 1,315 card-only; 803 / 1,315 compact; 1,055 / 1,567 fork default).

Expected: the card-only format cost vs compact is small (the second integer-state test (SM49) measured it on Tiny); the upper bounds sit at +0.60% to +0.70% in compact, so a cost of up to ~+0.3% still passes. This is a prediction, not a gate.

## What is scored (training-free)

Only the card-only format `w8+s4hgx4c8hx8n` for the two final models (tags `sm130-th-dec2ep32k-t91s2/w8+s4hgx4c8hx8n`, `sm130-io-dec2ep32k-t99/w8+s4hgx4c8hx8n`), one process, the exact scorer flags of six-layer Tiny (SM130)'s final scoring, plus `--nonfinite-rows`. The tree is branch sm134 = `origin/sm131` (which has the card-only scorer variant) plus the 27-line preset `w64-c1-d2p1u1` of six-layer Tiny (SM130) (`expertium/variants.py`, commit 2369f0ae of branch sm130; the Muon split change in `check_muon_fresh_decay.py` is training-only and not needed).

## Checks done before the launch (CPU)

- Rule selftest: reproduces SM130's published and compact numbers from the copied rows (th curve +0.467 / +0.483, instant +0.524 / +0.545; io instant +0.564 / +0.603, upper bounds too), the Tiny bytes, an end-to-end relabelled run (go, card-only vs compact exactly 0), a scaled run (no go), a dropped user (exit 2), junk lines counted.
- CPU smoke (`cpu/smoke-cpu-rows.jsonl`, users 2388, 4676, 1712, the two-head six-layer Tiny (SM130), three formats): every row finite, 185,057 parameters; w8+s8 equals SM130's published GPU rows within 1e-5 per head (the SM131 smoke tolerance; SM131 saw 2e-6 / 1e-5), compact within 1e-4, card-only differs from compact. `sm134_check.py` -> `check-cpu-sm134.json`.
- GPU smoke (first step of the ticket, same users and formats on the GPU): w8+s8 within 1e-5 of the CPU and published rows, compact within 1e-4 of both, card-only within 1e-4 of its CPU row; a mismatch = STOP (exit 3).
- `DRYRUN=1 launch-sm134.sh` OK.

## Run plan

One chained queue ticket (`sm134_ticket.sh 0.8 smoke score`, GPU cap 0.8 via `scripts/r01_queue.py`): smoke ~3 min, then both models x one format x 723 users, expected ~25-45 min in total (SM130's two-format final scoring took ~35 min per model; one format roughly halves the model time). In-ticket retries for missing users at cap 0.8, then 0.92 (memory only, same numerics). Quiet windows: quiet.json scope all / gpu only (`sm134_block.py`); no daily break. Memory: CPU processes under 6 GB (`memguard.ps1`, kill above 5.9 GB), GPU cap 0.8. No `.pth` / `.npz` committed; `wt/sm130`, the fork and every other worktree are read only / untouched.
