# The instant model for RWKV-Instant (SM93): handoff (2026-11-07 ~11:05)

User approval (11-07 ~10:40): "Yeah I think everything shows that an instant model would help a lot so I'd say go for
it": ship the instant-only small model (SM57) (stage 1 instant-only arm) as the model of the RWKV-Instant scheduler in the
Anki fork; RWKV-Curve keeps the release from epoch 25 (SM63, key `sm72e25`). **No release, not even a draft.**

Branch `sm93` (cut from `sm57` at `e89bc9fa`), worktree `C:/Users/jonat/workspace/wt/sm93`. Always `git fetch` +
`git merge --ff-only` first in every repo (the user works on several computers), the fork included.
`wt/sm57` is read-only (stage 3 GPU run there; editing its `scripts/`, `expertium/`, `rwkv/` kills it); never touch
`wt/sm41`; never write to `C:\Users\jonat\AppData\Roaming\Anki2\User 1`.

## Step 1: decisive check (running)

**Pool change, and why:** the brief said the 2,225-user pool, but that pool is invalid for both models: its 1,493 extra
users are training users of every final-tier (8,000-user) model, the release and the instant-only model included
(`artifacts/eval/2225-labels/DESIGN.md` on main, 11-01; it also has no labels for them). The valid larger pool for a
small-vs-small comparison is every validation user of the 8k split: **1,500 users** (107.6M reviews; held out from
both models). New `--pool 1500` in `scripts/eval_pools.py` / `scripts/small_model_heldout.py` (commit `24bd3c68`), file
`artifacts/small-model/eval-users-1500.json`; the 732 and 2,225 files are unchanged.

- The 732-set rows are reused (the release: `wt/sm72s/.../sm72s/gpu/small-lf-neighbour.jsonl`; the candidate:
  `wt/sm57/.../sm57/gpu/sm57-rows.jsonl`). This check scores both models on the other 768 users (minus 3 above 650k
  reviews: 6329, 8060, 9875, the 732 set's never-fit size) + **6 check users of the 732 set** that must match the
  earlier rows (max |LogLoss diff| <= 1e-5 per head) before rows are merged.
- Unit: % of the release's LogLoss on the same users (the big model (1 leak-free epoch) is trained on the ids >= 5000,
  so it cannot be the denominator there). On the 722 users 1% of the release's LogLoss = 1.003 points of the big model's,
  so the 0.05 minimum applied to it is slightly stricter. Bar (fixed before any new row): instant, candidate - release,
  95% CI upper < 0 and gain >= 0.05; paired user bootstrap 2,000, seed 0.
- Rule `sm93_rule.py decide` -> `sm93-decision.{json,md}` (exit 0 = go/stop, 2 = pending). Dry run on the existing
  rows reproduces the SM57 number (722 users, points of the big model: instant -0.162 [-0.176, -0.148], 646 / 75).
- Launcher `launch-sm93.sh`, WMI pid **32500** (bash.exe), log `gpu/launch-sm93.log`, rows `gpu/sm93-rows.jsonl`,
  scorer output `gpu/score.out`. Ticket 000984 (owner small-model, cap 0.8, 0.92 for no-progress passes) queued 10:41
  behind stage 3's epoch-4 ticket 000983; ~45 min of scoring (55.7M reviews x 2 models), then the rule and `done`.
  Expected done ~11:45-12:00 (later if stage 3's tickets go first).

## Step 2: export (done, CPU)

`artifacts/models/sm57` (this worktree): `RWKV_DELETED_CARDS=neighbour scripts/small_model_export.py package
--checkpoint <wt/sm57>/artifacts/small-model/sm57-inst-e26-dec-1ep32k-muon/expertium/E89-F65e-F77a/master.pth --arch w64
--label sm57-inst-dec1ep32k`. Parity vs PyTorch **PASS**: fp32 max |dp| 3.6e-7 curve / 1.9e-6 instant / 2.3e-6 rating,
int8 4.2e-7 / 1.5e-6 / 2.1e-6 (release sm72e25: 1.3e-6 / 1.2e-6; tolerance 1e-5). `architecture.json` byte-identical to
sm72e25's (`9bd41f11...`). Test vectors (the sm72e25 normal-input `make_testvectors.py`, users 4676 + 1745) and the
NumPy checker: **PASS** (tolerance 1e-5, max 8.6e-7). Hashes: weights.f32 `08448a2c...`, weights.int8 `3848d90d...`,
master.pth `09331f22...` (master.pth NOT committed, as SM92). Left: `sm93_package.py` (neutral names, evaluation from
`sm93-decision.json`, `use` = instant head only, re-hash) after the rule; then commit `manifest.json` + `parity.json`.
INTEGRATION.md: copy sm72e25's and add a short section on top: instant / rating head only, curve head untrained.

## Follow-up (fresh agent; Opus for the fork design, the rest is mechanical)

```bash
cd C:/Users/jonat/workspace/wt/sm93 && git fetch && git merge --ff-only
tail -3 artifacts/small-model/sm93/gpu/launch-sm93.log          # ends with "rule exit 0: [sm93-rule] ..." then "done"
tasklist //FI "PID eq 32500"                                     # launcher alive?
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status
cat artifacts/small-model/sm93/sm93-decision.md
```

Relaunch only if pid 32500 is gone and the log is not at `done` (resumes; the scorer skips scored users):

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm93/artifacts/small-model/sm93/launch-sm93.sh >> C:/Users/jonat/workspace/wt/sm93/artifacts/small-model/sm93/gpu/launch-sm93.nohup 2>&1"'; ProcessStartupInformation=$si}
```

1. Check `row_check` in the decision (both labels <= 1e-5 on 6 users; else verdict `invalid`: report, stop).
2. Commit (`git add -f` explicit paths, as Jonathan Schoreels, Co-Authored-By line, push): `gpu/launch-sm93.log`,
   `gpu/launch-sm93.nohup`, `gpu/score.out`, `gpu/sm93-rows.jsonl`, `sm93-decision.{json,md}`, this file.
3. **If stop:** report (candidate - release on instant and curve, CI, users better / worse; the ids < 5000 / >= 5000
   parts) and stop. No fork change.
4. **If go:** run `sm93_package.py`, commit the package (`artifacts/models/sm57/{INTEGRATION.md,architecture.json,
   manifest.json,parity.json,weights.f32.bin,weights.int8.bin}`, `reference/*.py`, `testvectors/{*.py,*.npz}`; never
   `master.pth`), push, then the fork (below).

## Step 3 notes: fork integration (SM93), from reading the fork at `201b6e34b` (main, 11-05)

Facts:
- One model per profile: key `rwkvModel` (`qt/aqt/rwkv_model.py`, default `sm72e25`), bundled folders
  `qt/aqt/rwkv_inference/<key>/`, build copy list `build/configure/src/aqt.rs` (with the time-head copy loop),
  `ModelKind` + `leak_free_inputs()` in `rslib/src/rwkv/runtime.rs`, PyO3 class `RwkvInference` in
  `pylib/rsbridge/lib.rs`, Python wrapper `_RustRwkvRuntime` / `EmbeddedRwkvReviewerBackend` in
  `qt/aqt/rwkv_srs_benchmark.py`, one global `_reviewer_backend` in `qt/aqt/rwkv_scheduler.py` (27k lines). The SM92
  integration commit `6caa5fb0e` lists every file a new key touches; `97cd0ad3e` switched the default back to SM63.
- **What RWKV-Instant reads:** the R-only path (`predict_retrievability_many*`, queue scoring and stats) returns
  `1 - P(Again)` of the instant/rating head (`engine.rs` `review_retrievability_features`). Full predictions
  (`review` / `predict_many`) return `retrievability` (instant), `button_probabilities` (rating head),
  `curve_retrievability`, `intervals`, `s90s` (curve head).
- **The answer-time head (SM83)** is loaded by the engine beside the weights (`time_head.{f32,int8}.bin`) and reads that
  engine's frozen features; it feeds only the "RWKV : Answer Time by Button" display; scheduling never reads it.
- The user / preset / deck streams span every card, so the instant model must replay the **whole** collection history,
  not only the Instant presets' cards: two full state sets whenever both models run.
- State persistence: one SQLite store `state-v12.sqlite3` (plus `.state-v12-building` / `-compacting` temp files that
  Python renames), the snapshot binary `ARWKVSNAPSHOT12` (written by Rust, parsed by Python: id -> blob maps + the
  runtime cache bytes), opaque per-entity state blobs and `RwkvRuntimeState` undo states. A sibling store file for a
  second engine would not follow Python's renames.

Recommended design (honest option for the time head: **it keeps running on the SM63 state**, the features it was
trained on; no retrain needed because SM63 stays resident whenever the instant model runs; a retrain on SM57's features
would cost ~1 h GPU (SM82c recipe, as SM92) for no gain):
- A **paired runtime** in Rust: the release engine plus the instant-only model's weights, one feature builder and one
  curve cache (identical inputs: both leak-free, feature 22 zeroed, normal intervals), per-entity state = a versioned
  pair (release state, instant state) serialized as one blob, so the SQLite store, snapshots, undo states and cache
  bytes keep a single file / blob per entity. Answers and warm-up advance both recurrences (rayon::join);
  R-only queries run only the instant model (queue scoring cost unchanged); full predictions take curve, intervals,
  S90 and the answer time from the release and `retrievability` + `button_probabilities` from the instant model.
  New `ModelKind` for the instant model (leak-free), never allowed as a profile's sole model (its curve is untrained).
- A distinct cache key for the pair (e.g. `sm72e25+sm57`) so no cached state of one layout is reused by the other.
- Load the pair only when the profile model is `sm72e25` and RWKV-Instant is enabled on some preset (or a profile
  key, default on); otherwise the single engine as today. `sm92` and `legacy` stay as they are (one model serves both,
  as now). Cost when paired: +1.35 MB weights, +1 state set (~3.1 KB per card + note at int4 cards / int8 notes; e.g.
  40k cards ~ +125 MB), answers and warm-up ~2x CPU (parallel), queue R-only unchanged.
- Tests: Rust parity of the new key on a fixture made with `qt/tools/rwkv_sm40_vectors.py` from
  `artifacts/models/sm57` (500 rows/user, as SM92), the paired runtime vs each single engine (instant outputs equal to
  the instant engine alone, curve / intervals / S90 / answer time equal to the release alone, bit for bit), the store /
  snapshot / undo round trips in paired mode; Python + TS tests of the key; harness
  `C:/Users/jonat/workspace/anki/rwkvharness` (`. ./env.sh; cargo test --offline --release`; 4 pre-existing
  `fused_packed` failures expected), `rustfmt --edition 2021 --check`. Fork pushes:
  `GIT_TERMINAL_PROMPT=0 git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push origin <branch>`.

## Done (2026-11-07 ~12:55)

- Step 1: rule **GO**, row check 0.0 on 6 users for both labels; `sm93_package.py` run (use text: curve +16.0% of the
  release's LogLoss on 1,487 users), `INTEGRATION.md` written (instant-only section on top of the sm72e25 manual);
  commit `225ac5ff` (package + decision + logs), pushed.
- Step 2, fork `JSchoreels/anki` **main `64211f498`** "Give RWKV-Instant the instant-only small model (SM57)" (also on
  branch `sm93-instant-model`), pushed; no remote commits had to be integrated (origin/main stayed at `201b6e34b`).
  User requirement added 11-07 ~12:25: only RWKV-Instant -> SM57 alone (`InstantOnly` runtime: no curve, intervals,
  S90, answer time or workload simulation); both -> SM63 shared by default, Advanced toggle "Use specialized model for
  RWKV-Instant (2x RAM if used alongside RWKV-Curve)" (`rwkvSpecializedInstantModel`) -> `Paired` runtime (SM57 for
  Instant, SM63 for Curve and the answer-time head); only Curve -> SM63. Applies at profile reopen.
  Report: `docs/rwkv-sm93-2026-11-07.MD` in the fork. No release, no draft.
- Tests run: harness default features 225 passed / 4 failed (the pre-existing `fused_packed` ones) / 30 ignored;
  `--no-default-features` same 4 only; new tests `rwkv_paired_runtime_matches_each_model_alone`,
  `rwkv_paired_runtime_state_round_trips`, `rwkv_instant_only_runtime_serves_no_curve`, `rwkv_sm57_*`, `paired::tests`
  pass; full 14,403-row vectors 2.3e-6 f32 / 2.0e-6 INT8; rustfmt clean; clippy clean on the new code. Python / TS /
  rsbridge tests could not run locally (no built `out/`): they run in fork CI.

## Follow-up: fork CI (not waited on)

CI run **42369925888** on main `64211f498` (~18 min). Check:

```bash
cd C:/Users/jonat/workspace/anki && git fetch && git merge --ff-only
gh run view 42369925888 --json status,conclusion,jobs --jq '.status, .conclusion, (.jobs[] | select(.conclusion != "success") | .name)'
gh run view 42369925888 --log-failed | tail -80     # only if it failed
```

If it failed: fix on a branch from main (Python ruff format / mypy on `qt/aqt/rwkv_scheduler.py`, `qt/aqt/rwkv_model.py`,
`qt/aqt/mediasrv.py`, the new tests in `qt/tests/test_rwkv_model.py` / `test_rwkv_inference_process.py`, the svelte
toggle, the proto field `specialized_instant_model`), harness `cd rwkvharness && . ./env.sh && cargo test --offline
--release -- runtime paired sm57`, commit as Jonathan Schoreels, fetch + ff before committing and pushing.

## Done (2026-11-07, main session)

Fork CI: run 42369925888 on 64211f498 failed (ruff import order; 3 Windows pytest backend-creation tests). Fixed in fork main 74c1131d6 (only `qt/aqt/rwkv_scheduler.py`: import order; the instant model is passed to the backend only when two models run together); CI run 42370821200 green on Linux, macOS and Windows. No release.
