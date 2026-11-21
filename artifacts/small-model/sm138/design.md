# Answer-time heads for the six-layer Tiny (SM138): design

Pre-registered Sat 11-21, before any GPU ticket of SM138.

Why: the user decided on 11-21 that the six-layer Tiny (SM130) replaces the current Tiny models (the show-time two-head cool-down for Tiny (SM124) and the longer cool-down for Tiny (SM115)) in the next family release. Every family model ships with an answer-time head, and a head is tied to the trunk it was trained on, so both final models of the six-layer Tiny (SM130) need their own head.

Branch `sm138`, worktree `C:/Users/jonat/workspace/wt/sm138`. The branch is cut from `origin/sm130` (the six-layer architecture `w64-c1-d2p1u1` and the scorer that produced SM130's recorded rows) with `origin/sm119` merged in (the head scripts `scripts/sm117_head.py`, `scripts/sm117_check.py`, `scripts/sm117_timelib.py` and the read-only SM117 inputs under `artifacts/small-model/sm117/`). It is not cut from `origin/main`, because main has none of the small-model code (main and the small-model branches split on 10-15). Launcher: `launch-sm138.sh`.

## Models

| model | size, kind | scorer kind | master (read only) |
|---|---|---|---|
| the six-layer Tiny two-head model (SM130), key `sm130th` | Tiny, two-head | `expertium:w64-c1-d2p1u1` | `wt/sm130/artifacts/small-model/sm130-th-dec2ep32k-p2/expertium/E89-F65e-F77a/master.pth` (sha256 `48a5e724cf38...`, scorer label `sm130-th-dec2ep32k-t91s2`) |
| the six-layer Tiny Instant-Only model (SM130), key `sm130io` | Tiny, Instant-Only | `expertium:w64-c1-d2p1u1` | `wt/sm130/artifacts/small-model/sm130-io-dec2ep32k-p2/expertium/E89-F65e-F77a/master.pth` (sha256 `23f037282379...`, scorer label `sm130-io-dec2ep32k-t99`) |

The labels are those of SM130's recorded show-time rows (`wt/sm130/artifacts/small-model/sm130/gpu/sm130-rows.jsonl`), which the trained smaller state for Tiny (SM137) also used as its reference: the tag t99 / t91s2 names the teacher, and the master is pass 2's for both. No release package exists yet for SM130, so the full sha256 of each master is pinned in the model table of `scripts/sm117_head.py` (read from the files on 11-21; both files date from the end of SM130's pass 2, 11-19 10:30 and 11-20 01:08).

**Head input width.** The recipe's inputs are the trunk's residual stream after the input encoder and after each interleaved schedule round (layer-normalised, no affine), the instant head's hidden layer (128) and the 26 explicit time features. The six-layer Tiny keeps card 1, deck 2, note 1, preset 1 and user 1 layers, so its schedule has 2 rounds instead of the 4 of the four-round Tiny (`w64-c1`): 3 snapshots x 64 + 128 + 26 = **346 inputs** instead of 474 (head 52,672 values instead of 69,056). This is the same rule applied to a shallower trunk, as the Big models got 1,306; `Frozen` in `sm117_head.py` derives the width from the model and needed no change. The fork's head runtime must accept a 346-wide head for these two models (a packaging point for later, not a rule change).

## Rules

Exactly the rules of the time heads for the new candidates (SM119), which are those of the time heads for the family (SM117) (`../sm117/design.md`): same recipe (MLP -> 128 -> 4 grades x 16 log-time bins, zero-initialised output, trained alone on the frozen model, the explicit-only ablation in the same pass, 1,991 training users, first 400,000 reviews each, one pass, AdamW lr 2e-3 with 2% warm-up and cosine, batch 8,192, seed 0), show-time inputs (`e2s-query`), same-day reviews included, the same S grid (3, 5, 8, 10, 15, 20, 30 s), the same baselines, the 732 set (722 users), leak-free. Steps per model: smoke, head training, grade-unchanged check, then gate scoring and the scoring of P(recall within S seconds), then `decide` per model and `report` on the CPU.

**Gate per head (go / no-go, fixed):** pass iff (1) grade unchanged: the show-time grade scorer (`small_model_heldout.py`, w8+s8, `--inputs e2s-query --split-existing --deleted-cards neighbour`) gives every grade field and every per-review dump array bitwise equal for the release and the trunk written after the head training, on SM82c's 5 grade users, in two separate scorer processes; and (2) the head beats both answer-time baselines (the card's last time, the user's running median per grade) on the censored time LogLoss and on the mean |ln error|, each paired review-weighted difference with its 95% CI below zero (722 users, all rows). SM130's recorded show-time rows are compared too (information only).

**Reliable (a report, not a go / no-go):** at a given S the head's P(recall within S seconds) is reliable iff its ECE (20 equal-mass bins) is below 0.01 and its LogLoss is below both baseline (a) (the model's own P(grade) x the user's running share of earlier answers of that grade under S) and baseline (b) (the user's running rate of the event), each paired difference with its 95% CI below zero. A model is reliable iff it is reliable at all seven S.

**Answer time** = front + back together, from the moment the question is shown to the moment a grade button is pressed, capped at the preset's maximum answer seconds. **Recall within S s** = recalled (Hard, Good or Easy) and answered within S s, reading the back included. The moment "Show answer" is pressed is not stored, so a front-only time cannot be computed.

## Changes against SM119 and their proof

- `scripts/sm117_head.py`: two model entries (`sm130th`, `sm130io`), the outputs written to `artifacts/small-model/sm138/` (`<key>-gate.json`, `sm138-result.{md,json}`), and the answer-time definition in the report header. No recipe or rule code changed.
- `scripts/sm117_check.py`: the expected head width of C1 is 346 for `w64-c1-d2p1u1` (474 for the other small models, 1,306 for Big).
- CPU proof, run on 11-21 before any GPU ticket: `sm117_head.py check --models sm130th,sm130io` -> **PASS, 16 checks, 84 s** (`check-cpu-sm138.json`, log `check-cpu-sm138.out`): C0 masters, C1 frozen training (trunk bitwise the release, nothing trainable, both first losses ln 16, 346 inputs), C2 resume bitwise, C3 capture = stock forward bitwise, C4a show-time audit, C4b explicit features causal, C5 labels, C6 loss = metric, C7 grade scorer equal for release vs trunk and the time scorer on 2 users, C8 train path = score path, C9 R-under-S arithmetic. The launcher refuses to start unless this file passed.
- Launcher (`launch-sm138.sh`, from `launch-sm119.sh`): models `sm130th sm130io`; no wait for another run; queue owner `sm138`, purposes starting "Answer-time heads for the six-layer Tiny (SM138): "; the daily 20:00-01:00 break and the Sunday-from-16:00 break removed (no planned breaks since 11-19; only quiet.json windows of scope all or gpu count); carriage returns stripped from every Python output that bash compares or evaluates (the failure of the trained smaller state for Tiny (SM137)'s launcher on 11-21); GPU cap 0.8, STOP file, resume as in SM119.

Outputs: `<key>-gate.json`, `sm138-result.md`, `sm138-result.json` here; heads in `artifacts/small-model/sm138-<key>-head/`; logs and rows in `gpu/`.
