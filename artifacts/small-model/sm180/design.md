# 4-bit weights (SM180): design

Pre-registered Fri 2026-11-27, committed before any row of this screen exists. Track: small model, model size. User request (11-27): keep exploring ways to improve LogLoss or shrink the models.

- Design branch `sm180` (from `origin/main`): this folder (design, launcher, report, arm preparation, CPU checks, hand-off), worktree `C:/Users/jonat/workspace/wt/sm180`.
- Run tree `C:/Users/jonat/workspace/wt/sm180-run`, branch `sm180-run`, cut from `origin/sm164` (c30a0070: the scorer that produced the release-candidate rows of the final state formats) plus **one gated commit** (below). The launcher refuses to start if the run tree has uncommitted source edits.

## Question

The release models ship int8 weights (`weights.int8.bin`: one float32 scale per row, int8 codes, for every 2-D `.weight`; the rest float32). **Can those weights go to 4 bits (int4 codes, one float32 scale per group of 32 or 16 input columns of a row) at a negligible accuracy cost?** First by post-training quantization (PTQ); if that costs too much for Small or Tiny, by a short quantization-aware fine-tune (QAT, straight-through estimator).

| plain name (ID) | tag in the rows | weights (2-D `.weight` / all) | state format scored |
|---|---|---|---|
| Big two-head (SM91s2) | `sm91s2-big-showtime` | 265 tensors / 2,762,884 parameters | F6 `lr2.16.4q6x8` |
| Big Instant-Only (SM99) | `sm99-big-inst-e2` | same architecture | F6 |
| Small two-head (SM118) | `sm118-e26-dec2ep32k-t91s2` | 217 tensors, 315,423 / 331,511 | F6 |
| Small Instant-Only (SM113) | `sm113-inst-e35st-st-dec2ep32k-t99` | same architecture | F6 |
| six-layer Tiny two-head (SM130) | `sm130-th-dec2ep32k-t91s2` | 105 tensors, 176,403 / 185,057 | T24 `lr2.16.4q8x8` |
| six-layer Tiny Instant-Only (SM130) | `sm130-io-dec2ep32k-t99` | same architecture | T24 |

Model files and sha256: `models.sh` (the SM164 copy). The state formats are the user's final choices (11-24): Big and Small F6 (T24 with int6 factors), Tiny T24.

## Which tensors, which group sizes

- **Quantized:** exactly the int8 tensors of the release (`architecture.json` `int8_tensors`): every 2-D `.weight` (all are `nn.Linear` weights; the 1x1 dummy weights included). Everything else (biases, norms, lerp / bonus vectors, `gru_*`) stays float32, as today.
- **Codes:** symmetric int4, -7..7 (code -8 never used), scale = abs-max of the group / 7 (float32; 1.0 if 0), round-half-even (`scripts/compress_quant.fake_quant`, the function the int8 weights already use with 127).
- **Group sizes: 32 (`w4g32`) and 16 (`w4g16`).** A group is a run of consecutive input columns of one output row; a row shorter than the group is one group. Why not 64 or per row: most Small and Tiny weights have 64 input columns (Small 283,904 of 315,423 parameters, Tiny 151,168 of 176,403), so 64 would be one scale per row for them, and per-row int4 was clearly worse in the earlier compression screen of the original fork model (below).
- **Bytes:** counted as the scorer does (`compress_quant.weight_bytes`): packed codes (2 per byte) + one float32 scale per group, other parameters float32; the int8 reference is codes + one float32 scale per row. With float32 scales g32 is about 5 bits per weight and g16 about 6, against about 8.5 for int8 per row on 64-column rows. Float16 scales would save a little more; they change the values and are not scored here.

## Seen before writing the rules (disclosure)

The earlier compression screen of the original fork model (`artifacts/compression/report.md` in the run tree; 105 held-out users, float32 state, user-mean statistic) measured int4 weights against float32: g32 +0.06% curve / +0.40% instant, g64 worse, per row +0.14% / +1.34%; int8 per row +0.01% / +0.01%. So the instant head is the sensitive one, and PTQ at the +0.15% bar may well fail; that is why the QAT step is pre-registered here. No row of the six release models with 4-bit weights exists yet.

## The gated code addition (run tree commit, `sm180-run`)

- `scripts/compress_quant.py`: one new scheme `"w4-g16": (4, 16)` (`"w4-g32"` existed).
- `scripts/small_model_heldout.py` `parse_variant`: weight prefixes `w4g32` (= the existing `w4`) and `w4g16`. `w8`, `fp32` and `w4` are unchanged; no other line of the scorer changes. A label is `<weights>+<state format>`, e.g. `w4g32+lr2.16.4q6x8`.
- `scripts/weight_qat.py` (new) and a 4-line dispatch at the top of `scripts/state_vq.install_training`: the track option `"state_vq": {"recipe": "weight-qat", "bits": 4, "group": 32}` makes the trainer's master / child sync (`MasterChildSync.downcast`, release option `grouped_sync`) overwrite, after every master -> child copy, each 2-D weight of the child (bfloat16, the TorchScript model that runs forward and backward) with `fake_quant(float32 master weight)` cast to the child dtype. The gradient is computed at the quantized weights and applied to the latent float32 master unchanged: the straight-through estimator. No module, parameter name or state_dict key changes; the flat-buffer sync and float32 training are refused while it is installed; every downcast is counted (`downcasts` in result.json, a `[weight-qat] first downcast` line in the log). No other recipe reaches the new code. `state_vq` is teacher-blind (`expertium_options.TEACHER_BLIND`), so the KD replay cache is the release one. (A first version switched the `nn.Linear` class; the pre-launch GPU trainer smoke, ticket 001805, showed that the training model is TorchScript, so it was replaced before launch; ticket 001806 then applied 105 quantized weights on six-layer Tiny two-head (SM130) and finished 2 updates.)
- **int8 unchanged, bitwise:** the GPU smoke (below) rescores the release int8 labels and must equal SM164's GPU smoke rows exactly (difference 0).
- **CPU proofs** (`sm180_check.py`, before launch, `check-cpu-sm180.json`): the new labels parse to the intended schemes and the old ones to the old; the scorer's w8 weights equal the int8 per-row rule of `weights.int8.bin` bitwise; its w4 state dicts equal `fake_quant` of every 2-D weight with codes in -7..7 and leave every other tensor bitwise equal; on the trainer's own master / child pair (Tiny, CPU eager) the plain downcast is unchanged before install, every downcast after install writes exactly the scorer's w4g32 weights into exactly the scorer's tensor list, the other child tensors and the master stay as without QAT, the gradient transfer to the master is unchanged, uninstall restores.

## Step 1: PTQ (every model, both group sizes)

Scorer (run tree, unchanged flags of SM148 / SM152 / SM171): `--pool 732 --users-file artifacts/small-model/sm72s/eval-users-732-fit.json` (723 users) `--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets 0,1,3,5,9,16,32,128,300,1000`, `PYTORCH_JIT=0`, GPU cap 0.8 (0.92 for a retry pass without progress: memory only, same numerics). Show-time inputs, leak-free, curve and instant.

Labels scored now: `w4g32+FMT` and `w4g16+FMT` for each of the six models (FMT = F6 for Big and Small, T24 for Tiny). The int8 baselines `w8+FMT` are **reused, not rescored**: SM152's F6 rows and SM148's T24 rows of the 732 set (`wt/sm164/artifacts/small-model/sm152/gpu/sm152-rows-732.jsonl` sha256 1e716ec4..., `.../sm148/gpu/sm148-rows-732.jsonl` sha256 b5d37215..., pinned); the smoke proves the run tree reproduces them.

## Rules (pre-registered, `sm180_report.py`; selftest passes)

Statistic (as SM148 / SM152 / SM164): % = 100 x (sum of the label's LogLoss / sum of the baseline's LogLoss - 1), review-weighted over the benchmark reviews, paired user bootstrap, 2,000 draws, seed 0, 95% CI (a verbatim copy of `showtime_report.compare`). Users: the 723-user list, those with rows for both labels.

- **PTQ cell** (model, head, group size): int4 `w4gG+FMT` vs the same model's int8 `w8+FMT`. **Free** if the CI upper bound is at most **+0.15%** (the bar of Big's own format budget). Heads: two-head models curve and instant, Instant-Only models instant.
- **Per size and group size:** FREE if every cell of both models of the size is free and every row finite; otherwise NOT FREE.
- **Per size:** the chosen group size is 32 if FREE (fewer bytes), else 16 if FREE, else **PTQ fails**.
- **Coverage:** at least 715 users per label on the list (as SM152), else `pending` (exit 2).
- **Reported, no verdict:** every cell's value and CI, the weight bytes and the saving vs int8 per model, the users per comparison.

## Step 2: QAT (only for Small or Tiny when PTQ fails; Big gets no QAT here)

- **Target per failing size:** the two-head model (both heads) if it is not free at g32 on some head, otherwise the Instant-Only model. At most one arm per size.
- **Recipe (the SM171 short fine-tune, unchanged except the one option):** start = the release model's final `master.pth` AND `optimizer.pth` (sha256 checked); the release folder's options, loss and targets (show-time inputs, strided-lane carry, 32,768-review windows, replay KD targets, deleted cards neighbour); data = whole users of `random.Random(171).shuffle` of the release folder's training rows (sorted by user id first) until 2^26 reviews (Tiny: 944 users, 67,310,673 reviews); learning rate 0 -> 7e-5 over 64 updates, then one_minus_sin to 0 (about 2,550 updates); snapshots every 100 users. The QAT arm adds `"state_vq": {"recipe": "weight-qat", "bits": 4, "group": 32}` (group 32 for every arm).
- **Control:** the identical fine-tune without the option. For six-layer Tiny two-head (SM130) the SM171 control `sm171-tiny-ctl` is reused (`wt/sm171-run`, master.pth sha256 fe233f0c...): same start, rows, learning rate and release options, trained by the same trainer code (cea0f931 and this run tree differ only in scorer files and this commit), its only option difference was the SM171 hook at weight 0, which is the stock arithmetic bitwise (SM171 proof P4). `sm180_prepare.py pair` proves the arm and its control differ only by the option. Any other target trains its own control (`sm180-<model>-ctl`, the release options unchanged).
- **Scoring:** same scorer and flags; labels `w4g32+FMT` and `w8+FMT` for the arm and for the control.
- **QAT rule, per head:** arm at `w4g32+FMT` vs control at `w8+FMT`, CI upper bound at most **+0.15%** -> **QAT free** for that model (then the other model of the size would follow the same recipe, not part of this screen). Otherwise QAT not free: the size keeps int8 weights.
- **Reported, no verdict:** arm w4g32 vs control w4g32 (what QAT buys over PTQ on equally fine-tuned weights), arm w8 vs control w8, arm w4g32 vs the original release model at w8 (the SM148 / SM152 rows), control w8 vs the original at w8, training reviews/s and GPU memory.

## GPU smoke (first ticket, STOP on failure)

SM164's 5 smoke users (`artifacts/small-model/sm164/users-smoke.json`, `--pool 1500` as SM164's smoke), 6 models x {REF `w8+s8hx8`, T24, F6} and the two int4 labels of each model: (1) every w8 row must equal SM164's GPU smoke rows **exactly** (difference 0 on curve and instant LogLoss, equal review counts); (2) every int4 row finite, same review counts as w8, smaller `weight_bytes`, and different from w8 on at least one user per model. The QAT step starts with the trainer's own 2-user smoke of each new arm (`expertium_decay.py smoke`).

## Run and budget

One detached launcher `launch-sm180-gpu.sh` (WMI hidden, `HANDOFF.md`); every GPU step is one `scripts/r01_queue.py` ticket, owner `sm180`, purpose prefix `4-bit weights (SM180): `, cap 0.8; quiet windows respected (`sm135_block.py`). Order: smoke; PTQ Tiny, Small, Big (2 models x 2 labels each); CPU PTQ report; then for each failing size (Tiny first) arm preparation, trainer smoke, training, scoring, CPU QAT report.

| step | GPU minutes (from SM164's per-label minutes, SM171's training speeds) |
|---|---|
| smoke | about 3 |
| PTQ Tiny / Small / Big | about 32 / 40 / 68 (+ retry passes) |
| QAT Tiny two-head: arm (control reused), scoring 4 labels | about 25 + 32 |
| QAT Small two-head: arm + control (28.7k reviews/s), scoring 4 labels | about 45 + 42 + 40 |
| **total** | **about 2.5 h PTQ, up to about 3 h QAT: at most about 6 h** |

An Instant-Only target costs more (Small Instant-Only (SM113) trains at about 20k reviews/s: about 2 h for arm + control); the launcher logs the estimate before it starts. Memory: cap 0.8 of 16.3 GB per ticket; CPU steps under 1 GB, never above 6 GB. No 8k / 2k 10-epoch ETA: no training run of that kind.

## Files

`models.sh`, `sm180_check.py` (CPU proofs), `sm180_report.py` (smoke, ptq, qat, selftest), `sm180_prepare.py` (arm, pair), `launch-sm180-gpu.sh`, `HANDOFF.md`. Outputs: `gpu/` (log, tickets, `smoke-gpu-rows.jsonl`, `sm180-rows-ptq.jsonl`, `sm180-rows-qat.jsonl`, `qat-targets.txt`), `sm180-ptq.{json,md}`, `sm180-qat.{json,md}`; QAT folders `sm180-<model>-qat` / `-ctl` under the run tree's `artifacts/small-model/`.
