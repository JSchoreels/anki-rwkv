# Session inputs confirmation (SM181): design

Written Sat 2026-11-28 00:45, before any GPU ticket and before any scoring row of this run exists (only the CPU proofs below ran first). Branch `sm181` (from `origin/sm179` at `8aab3a56`: design, scripts, launcher, hand-off), worktree `C:/Users/jonat/workspace/wt/sm181`. Run tree: `C:/Users/jonat/workspace/wt/sm179-run`, branch `sm179-run` at `cd220759` (SM179's run tree with the gated feature code `rwkv/extra_inputs.py`), clean and not edited; the new arm folder lives there, ignored by git. Track: small model.

## Question

The session inputs screen (SM179) found, for six-layer Tiny two-head (SM130) after a short fine-tune (about 2,550 updates from the release weights, 6 new inputs through zero-initialised weights), an instant-head LogLoss change of -0.065% [-0.073, -0.058] against the matched control (SM171's control cool-down) on the 732 set, with the curve head unchanged (+0.001%). **Question: does that gain hold (1) on the decisive users, for the same two Tiny models, and (2) for Small two-head (SM118) with the same short fine-tune?**

| plain name (ID) | what it is here |
|---|---|
| six-layer Tiny two-head (SM130) | `w64-c1-d2p1u1`; axis 1 reuses SM179's two checkpoints: the session arm `sm179-run/.../sm179-tiny-ses` and the control `sm171-run/.../sm171-tiny-ctl` (not retrained) |
| Small two-head (SM118) | `w64`, release folder `wt/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2` (`master.pth` sha256 b4fe1175...) |
| the rank-2 cool-down test on Small (SM149) | its control arm `sm149-run/.../sm149-sm118-ctl` (codec off: the stock trainer, no track option) is axis 2's control: 794 users, 67,256,032 reviews, 0 -> 7e-5 over 64 updates then one_minus_sin, seed 2025, trained in `sm149-run` at `cea0f931` (the base of this run tree), 38.8 min, 28,883 reviews/s |
| the session inputs screen (SM179) | the 6 inputs (`SM179_INPUTS=session`: gap, reviews in the last 10 and 60 minutes, minutes since the session start, position in the session, the session's recent Again share), their fixed normalisation constants (in `rwkv/extra_inputs.py`, from SM179's 944 training users, used unchanged for Small) and the zero-column widening of the first Linear |
| the release-candidate re-check (SM164) | the decisive user list and the Tiny REF rows used for the size integrity check |

## Axis 1: decisive users (six-layer Tiny two-head (SM130), no training)

**Users.** The 1,500 pool (`artifacts/small-model/eval-users-1500.json`, `scripts/eval_pools.py`) is every validation user of the 8k split minus the sealed final-test users: the 732 set plus 768 users with id >= 5000. The decisive list is the latter part with at most 650,000 reviews (SM93's rule, as SM164 and SM169): 765 users, 55,729,876 reviews, a copy of SM164's `users-new-all.json` in `users-decisive.json`. The 732 set is excluded by construction.

**About Big's training users.** SM164 recorded that the users with id >= 5000 are training users of the Big models, so a direct Small or Tiny vs Big comparison there measures Big's in-sample advantage. They are held out from the training of both Tiny models compared here (and of both Small models), so a Tiny-vs-Tiny comparison is valid on them (`eval_pools.py`: "small-vs-small only"). This check never compares with Big; both models share the same KD teacher (the show-time Big model (SM91s2)), so its in-sample knowledge of these users enters the arm and the control alike. No further user is removed.

**Scoring.** The run tree's scorer, one ticket per model so that each process has its own `SM179_INPUTS`: the control with the flag unset, the session arm with `session`. `--pool 1500 --users-file users-decisive.json --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets 0,1,3,5,9,16,32,128,300,1000 --dump <abs path>`, `PYTORCH_JIT=0`, label REF `w8+s8hx8` only (show-time inputs, curve and instant), cap 0.8, then retry passes at cap 0.92 for missing users while a pass makes progress (memory only).

## Axis 2: Small two-head (SM118), the same short fine-tune

**Arm** `sm181-small-ses` (`sm181_prepare.py arm`): SM149's control cool-down with only the inputs changed. Start = the release model's final `master.pth` and `optimizer.pth` with 6 zero columns inserted at input index 24 of `features2card.0.weight` (128 x 92 -> 128 x 98) and of its Muon momentum (`muon[67].momentum_buffer`), so that step 0 computes exactly what the release model computes. Rows, schedule, snapshots, options and seed equal SM149's control (`sm181_prepare.py pair`: ok). Trained through `sm181_run.py` (SM179's guarded entry: the environment's flag must equal the folder's, the run tree must be clean).

**Control** `sm149-sm118-ctl`, not retrained: the same users, reviews, learning rate, seed and options, on the stock code (`cea0f931`), which the run tree with the flag unset equals bit for bit (K2a). Integrity: its rows rescored here must equal SM149's rows of that model (max |LogLoss difference| <= 1e-6, at least 720 users).

**Scoring** on the 732 set (`--pool 732 --users-file <sm72s eval-users-732-fit.json>`), otherwise as axis 1.

## Report only: Tiny on the 732 set with the dump

SM179 scored its models without the per-review dump. For the AUC on the 732 set, its two Tiny models are rescored here with the dump (same flags as SM179); their rows must equal SM179's rows (max |LogLoss difference| <= 1e-6). This adds no verdict; it runs last and can be skipped if the GPU is needed.

## Proofs (CPU, before the GPU tickets)

`sm181_check.py cpu` -> `check-cpu-sm181.json`: **ALL OK** (118 s).

- **K1** show-time data of every user read with the flag on: all 794 Small training users (67,256,032 rows) and all 765 decisive users (55,729,876 rows) pass `rwkv/extra_inputs.py`'s own alignment (unique shared row index, rating and duration equal), all session values finite, 0 answer-time inversions.
- **K2** Small two-head (SM118), the real trainer path on CPU (release weights and Muon state, release options, 256-review windows, users 1088 and 2104, 3 updates each): (a) flag off in the run tree = SM149's stock tree bit for bit (loss, model input, every gradient, 6 updates); (b) flag on from the padded start: first-update loss equal to stock (relative difference 0.0), width 98, extra columns and their gradients non-zero, finite.
- **K3** the scorer: Small release (flag off) vs the padded Small start (flag on) on 2 small users of the 732 set, and six-layer Tiny release vs SM179's padded session start with `--pool 1500` on the 2 smallest decisive users (5625, 9094): same benchmark LogLoss (max |difference| 0.0, fp32 and REF).

## Rules (pre-registered; `sm181_report.py`)

Statistic as SM179: % = 100 x (sum over users of LogLoss x reviews of the arm / the same of the control - 1), per head, paired user bootstrap, 2,000 draws, numpy `default_rng(0)`, 95% CI, on each axis's rule users (a finite row for both models, equal benchmark sizes).

- **Confirmed (per axis):** the instant change is <= -0.05% with the CI upper bound below 0, and the curve point estimate is <= +0.05%. The instant head is the head SM179 found; a curve-only gain does not confirm.
- **Verdict:** CONFIRMED if both axes are confirmed; PARTIAL if one is (the main session decides); NOT CONFIRMED if neither.
- **Pending** (exit 2, the main session decides): axis 1 fewer than 743 rule users (97% of 765, SM164's coverage rule); axis 2 fewer than 720; a non-finite row; or an integrity check fails (axis 1: every benchmark size equals SM164's Tiny REF row of the same user, at least 600 common users; axis 2: the control's rows equal SM149's).
- **Reported, no verdict:** both heads with CI per axis; the change per answer-count bucket; Tiny on the 732 set and the decisive users together; Small training reviews/s, minutes and GPU memory vs the control's 28,883 reviews/s and 38.8 min.

## Secondary metric (reported, not part of the go rule): discrimination (AUC)

Added at the user's request (11-28) before any scoring row of this run existed. From the per-review dumps (benchmark reviews; `y`, `p_ahead`, `p_instant`), per axis (decisive, Small 732) and for the report-only Tiny 732 rescoring, per head (curve, instant):

- **Overall AUC:** the AUC of all rule users' benchmark reviews pooled (exact, rank-based with ties counted one half).
- **Within-user AUC:** the unweighted mean of the per-user AUCs over the rule users with both outcomes.
- **Change:** arm minus control (absolute AUC difference, also as % of the control's value), 95% CI from the paired user bootstrap with the same 2,000 user draws as the LogLoss. In the bootstrap the pooled AUC uses per-user histograms over 16,384 logit bins (the point estimate is exact; on SM169's dumps the binned and exact values agreed to 5 decimals). The within-user mean in a draw skips drawn users with one outcome.
- Dumps: `C:/Users/jonat/workspace/wt/sm181/work/sm181-dump/{decisive,small732,tiny732}/` (never committed). Users without both models' dumps are left out of the AUC only (their count is reported).

## Estimates and budget

| step | GPU minutes |
|---|---|
| trainer smoke, Small arm | about 3 |
| axis 1: two Tiny models on the decisive users with the dump (SM169: two Tiny models, 665 such users, with the dump, 23 min) | about 15 each, plus retry passes about 5 |
| axis 2: `sm181-small-ses` (control: 38.8 min) | about 40 |
| axis 2: two Small models on the 732 set with the dump | about 15 each, plus retries |
| report only: two Tiny models on the 732 set with the dump (SM179: about 10 each) | about 10 each |
| **total** | **about 2.4 h** (budget 5 GPU hours) |

Memory: cap 0.8 of 16.3 GB for every ticket (0.92 only for memory retry passes). CPU: the launcher, the proofs and the report stay under 4 GB (the report loads the dumps one axis at a time, about 1 GB). No 8k / 2k 10-epoch ETA: this is a short cool-down, not a training run.

## If confirmed (not part of this run)

As SM179 planned: the session inputs in a full Tiny cool-down (the release recipe, all 8,000 users), then Small, then the fork's feature builder for the 6 columns with test vectors. The true time of day and weekday (A19) remain the next inputs to test on donated data.
