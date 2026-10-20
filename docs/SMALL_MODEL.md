# Small model: lighter than the fork's 2.76M RWKV, close to its accuracy

Goal (user, 2026-10-15): a model clearly lighter than the fork's 2.76M RWKV (fewer parameters, smaller file, cheaper CPU inference) whose accuracy is close to it, and that competes with the fork's simple int8 variant (int8 weights + int8 card/note state: about +0.00% LogLoss vs f32, 2.99 MB of weights). Branch `small-model` (worktree `C:\Users\jonat\workspace\wt\small-model`, off main `eb42997`, with `accuracy-loop` `b7717a2` and `wide-kernels` `1a38ef1` merged in). Night plan and follow-up: [artifacts/small-model/HANDOFF.md](../artifacts/small-model/HANDOFF.md).

## 1. Frontier from existing results (no new GPU work)

Δ% = (candidate − fork) / fork × 100 on the same users and protocol; positive is worse. "—" = not measured. File bytes: float32 = 4 B per parameter; int8 = per-output-channel int8 codes of every 2-D weight + one f32 scale per row + f32 for the other parameters (the fork's `w8-pc` layout, `compress_quant.weight_bytes`).

| Candidate | Parameters | File f32 | File int8 | CPU inference cost | LogLoss curve / instant | RMSE(bins) curve / instant | Users and protocol |
|---|---:|---:|---:|---|---|---|---|
| Fork 2.76M, f32 (fork default: + int8 state = identical) | 2,762,884 | 11,051,536 | — | — (fork doc: f32 is its runtime baseline; no reviews/s figure in our records) | 0.30563 / 0.27859 | 0.03969 / 0.02049 | Held-out B: 1,000 fork-unseen users, full history |
| Fork 2.76M, int8 weights + int8 state (`w8-pc+s8`) | 2,762,884 | — | 2,995,488 | — | −0.004% / +0.009% | −0.06% / +0.32% | Held-out B, same 1,000 users |
| Fork 2.76M, f32 | same | same | — | — | 0.4417 / 0.4366 | 0.0223 / 0.0194 | Own collection, full layout, 173,690 reviews |
| Fork 2.76M, f32 | same | same | — | — | 0.2816 / 0.2494 | — | 105-user CPU screen (validation users ≤ 20k reviews), full history |
| Our w32 full run, epoch 9 (iter36) | 192,800 | 771,200 | 245,696 | — | **+4.70% / +7.15%** | — | 105-user CPU screen, full history (only held-out comparison with the fork) |
| Our w32 full run, epoch 10 (`fullrun-e10`) | 192,800 | 771,200 | 245,696 | — | **+2.68% / +3.64%** (0.4535 / 0.4525) | — | Own collection, full layout |
| Our w32, epoch 10 + 0.25-epoch decay (`decay-e10-q25`, A08) | 192,800 | 771,200 | 245,696 | — | **+2.70% / +3.49%** (0.4536 / 0.4518) | — | Own collection, full layout |
| `fullrun-e10` | | | | | 0.33518 / 0.31853 | 0.06319 / 0.05602 | 1,000 screen validation users, window protocol (128-review samples); fork never scored here |
| `decay-e10-q25` | | | | | 0.33383 / 0.31749 | 0.06092 / 0.05426 | same (paired gain −0.00135 / −0.00104) |
| w32 screen baselines (2,000 users, 1 epoch, order seeds 0/1/2) | 192,800 | 771,200 | 245,696 | — | 0.35258, 0.35141, 0.35567 / 0.33341, 0.33149, 0.33229 | 0.0785–0.0832 / 0.0661–0.0694 | same window protocol |
| w64 (Expertium s64h, `"architecture": "w64"`) | 331,511 | 1,326,044 | 410,459 | — | — (never trained on our split) | — | — |
| w80 (Expertium iter-53, `"w80"`) | 558,209 | 2,232,836 | 665,291 | — | — | — | — |
| Distilled students (compression Q05, catch-up X05) | — | — | — | — | — (none trained) | — | — |
| Fork 2.76M QAT int4 per-channel (compression Q02, run 1) | 2,762,884 | — | 1,635,616 (int4) | — | — (never scored) | — | — |

Recurrent state per entity (the other size that ships; `compress_quant.state_bytes` formula, int8 = one f32 scale per 32/16-value row + codes + f16 shifts):

| Model | Card f32 | Card int8 | Note f32 | Note int8 |
|---|---:|---:|---:|---:|
| Fork 2.76M (card 3 layers, note 2; d 128, 4×32) | 52,224 | 15,360 | 34,816 | 10,240 |
| w32 iter36 (card 1, note 3; d 32, 1×32) | 4,352 | 1,280 | 13,056 | 3,840 |
| w64 (card 2, note 1; d 64, 4×16) | 9,216 | 3,072 | 4,608 | 1,536 |
| w80 (card 2, note 1; d 80, 5×16) | 11,520 | 3,840 | 5,760 | 1,920 |

Training speed (RTX 5080, 160-user R01 speed runs; inference unmeasured): w32 accepted 178,242 reviews/s; w64 with the wide kernels 141,274 (contaminated run, W04); w80 with them 111,281; w32 + online fork teacher (KD) 113,121.

Final k=3 held-out frontier (runbook 2026-10-15, tickets 000332-000348 all exit 0; full history, 430 fork-unseen users every label has, 34 of 464 cut by time limits; Δ% vs fork w8+s8; accuracy lane, no speed figures; details in the HANDOFF and `artifacts/small-model/heldout-report.md`):

| Candidate | Parameters | Curve LL / Δ% | Instant LL / Δ% | Instant RMSE(bins) Δ% |
|---|---:|---|---|---:|
| Fork fp32 | 2,762,884 | 0.29418 / +0.01% | 0.26919 / -0.01% | -0.4% |
| Fork w8+s8 | 2,762,884 | 0.29416 / ref | 0.26921 / ref | ref |
| w32 decay-e10-q25 (w8+s8) | 192,800 | 0.31385 / +6.69% | 0.28886 / +7.30% | +88.6% |
| w32 fullrun-e10 | 192,800 | 0.31461 / +6.95% | 0.28947 / +7.53% | +89.6% |
| w32 screen o0 / o1 / o2 | 192,800 | +11.42 / +12.37 / +12.78% (mean +12.19) | +15.24 / +17.20 / +16.96% (mean +16.47) | +212 / +229 / +211% |
| w64 screen o0 / o1 / o2 | 331,511 | +11.37 / +11.30 / +12.36% (mean +11.68) | +11.96 / +12.19 / +12.25% (mean +12.13) | +152 / +153 / +150% |
| w64kd screen o0 / o1 / o2 | 331,511 | +10.17 / +8.63 / +10.56% (mean +9.79) | +9.39 / +9.86 / +11.51% (mean +10.25) | +115 / +127 / +152% |
| w64 o0 w8+s8 | 331,511 | 0.32758 / +11.36% | 0.30144 / +11.97% | +151.9% |
| w64kd o0 w8+s8 | 331,511 | 0.32412 / +10.18% | 0.29449 / +9.39% | +115.1% |
| **Scale-up w64kd, 3 epochs on 5,000 users + 0.25-epoch decay (`scale-w64kd-dec`, w8+s8)** | 331,511 | 0.30970 / +5.28% [+4.17, +6.82] | 0.28534 / +5.99% [+5.42, +6.61] | +85.0% |
| scale-w64kd-dec fp32 | 331,511 | 0.30970 / +5.28% | 0.28532 / +5.98% | +84.9% |
| scale-w64kd-e3 fp32 (before decay) | 331,511 | 0.31037 / +5.51% | 0.28533 / +5.99% | +81.3% |

Paired k=3 vs w32 (mean of cand - base LogLoss; band = larger of full-history and window band): w64 curve -0.00151 (band 0.00359, inside), instant -0.01166 (0.00471, better); w64kd curve -0.00706 (0.00490, better), instant -0.01673 (0.00488, better); kd-vs-w64 curve -0.00555, instant -0.00507 (both negative, better). Gap closure vs w32 (pp): w64 0.51 curve (fails 1.0) / 4.34 instant; w64kd 2.40 / 6.22. int8 cost (o0, w8+s8 minus fp32): w64 +0.01 pp / +0.01 pp, w64kd +0.01 pp / 0.00 pp. Verdict: w64kd is promising (R) but its EPOCH_MIN is 61.6 > 57 and w64 does not pass, so the HANDOFF rule does not permit the scale-up. The screens (o0-o2) stay above the full-run w32 (+6.7% / +7.3%), so none reaches the good-candidate bar.

Scale-up result (tickets 000349-000354, all processed; w64kd, EPOCHS=3, 62 min/epoch, 430 users every label has; accuracy lane, no speed figures): `scale-w64kd-dec` (w8+s8) is +5.28% curve / +5.99% instant vs fork w8+s8, instant RMSE(bins) +85.0%. It beats the w32 `decay` (+6.69% / +7.30%) on both heads, paired (k = 1) curve -0.00415, instant -0.00348 LogLoss, but a single run: inside the report's window band on curve (0.00622) and marginally past it on instant (0.00272); the full-history band is not defined for k = 1. int8 is free (w8+s8 minus fp32 +0.00 / +0.01 pp). Verdict: NOT a good candidate (bar: <= +1.0% / +1.5%, RMSE(bins) <= +15%); "closer, not yet close enough" (gap about +4.3 pp curve / +4.5 pp instant). Packaged as `artifacts/models/scale-w64kd-dec/` (main checkout, not in git): int8 file 436,391 B (410,459 B of weights; 331,511 params), NumPy-vs-PyTorch parity PASS (max |dp| 1.4e-6 fp32, 3.7e-6 w8). The decay `compare` crash on ticket 000352 (post-run, KeyError per_user) was fixed in bcc8c9e; the checkpoint was valid.

### Comparability and gaps

- **The only held-out comparison of our model with the fork** is the 105-user CPU screen (epoch 9, f32 weights and state, users ≤ 20k reviews): +4.7% curve / +7.2% instant. The collection gives +2.7% / +3.5–3.6% but is one user.
- **Held-out B cannot score our models:** 840 of its 1,000 users are training users of the 8k/2k run. Only 160 are validation users.
- **The accuracy-loop screen users** (window protocol) have never been scored with the fork. Its full-history-trained model would be penalised by 128-review windows, so the window protocol does not suit a fork comparison.
- **w64, w80 and distilled students** have no quality result on our data at all.
- **CPU inference cost** is unmeasured for every candidate in a runtime-like setting. Per-review compute scales roughly with the matrix parameters: w64 has 8.3× fewer than the fork, w32 14.3× fewer.
- **Comparable set for tonight:** the 464 screen evaluation users with id < 5000. All are 8k/2k validation users: unseen by the fork, the full run and every screen. They are scored in full history (the fork's replay protocol) with the fork's benchmark labels. 100 of them are also in held-out B, and the fork is scored through the same code path as held-out B, so those 100 users must reproduce `heldout-b.jsonl` (a free check). Our models' int8 state rounds the card/note WKV rows like the fork but leaves their shift vectors in f32 (the fork rounds them to f16).

## 2. Route

| Route | Evidence | Verdict |
|---|---|---|
| int8 on our models | Fork: int8 weights + state cost ≤ 0.01%; ours is expected to be similar, measured tonight (`w8+s8` labels) | Necessary but closes none of the 3–7% gap |
| w32 + A08 cool-down + int8 | Cool-down gain −0.0014 / −0.0010 (≈ 0.4%, window protocol) | Too small against a 2.7–7.2% gap |
| Narrower / fewer-layer fork variants | Need upstream-style long-sequence training; our only fork-architecture trainer (QAT) runs at 269 reviews/s | Not feasible tonight |
| **Width 64 (s64h)** | Expertium's paired table (his split, 2,279 users): d=64 333k params 1 WS + 1 decay 0.2979 / 0.2643 vs d=128 2.58M 10 + 2 0.2963 / 0.2612, i.e. +0.54% / +1.19% at 7.7× fewer parameters with far less training; d=80 10 + 2 +0.24% / +0.46% | **Chosen**: biggest lever with outside evidence; 8.3× fewer parameters, int8 file 0.41 MB (7.3× smaller than fork int8), card state int8 3,072 B (5× smaller) |
| **Distillation from the fork (KD)** | Expertium's iters 32/39/45 adopted KD from a d=128 teacher (alpha 0.9, decay 0.5); our teacher is exactly the model to match | **Chosen, on top of w64**, with w64 as its control |
| w80 | +0.00003 / +0.0004 better than d=64 in his matched 10 + 1 confirmation, at 1.7× the parameters and 0.79× the speed | Deferred |

Caveats:
- The w64 screens run at 256 chunks per update. The w32 baselines run at 512 chunks, because w64 pages at 512 on 16 GB. So w64 vs w32 confounds width with twice the updates.
- The teacher was trained on users 5000–10000. 1,003 of the 2,000 screen training users are in that range, where its targets are fitted to the users themselves. Evaluation uses fork-unseen users only.
- Per AGENT.md and ACCURACY_LOOP rule 2, a screen cannot accept a smaller model as "equal". These runs choose the recipe for the scale-up; they validate nothing.

## 3. Tonight (2026-10-15, about 01:15–08:00)

Runbook [artifacts/small-model/runbook-2026-10-15.sh](../artifacts/small-model/runbook-2026-10-15.sh), one queue ticket per step, each under 60 minutes, first ticket 000332:

1. Screens: `screen-w64-s2025{,-o1,-o2}` and `screen-w64kd-s2025{,-o1,-o2}`. Screen tier: 2,000 users, 1 epoch, init seed 2025, order seeds = those of `screen-w32-s2025{,-o1,-o2}`. Paired k = 3: w64 vs w32 (capacity, informative) and w64kd vs w64 (KD effect).
2. Matched evaluation: `scripts/small_model_heldout.py`, full history on the 464 fork-unseen screen users:
   - fork f32 and fork int8 weights + int8 state;
   - `decay-e10-q25` (f32 and int8+int8), `fullrun-e10`;
   - every screen run (f32), and int8+int8 of the two o0 w64 runs.
   - New code: an any-length head-size-16/32 WKV (`scripts/small_model_wkv.py`, ledger S01), self-tested against the float32 reference at each ticket start.
3. Report: `scripts/small_model_report.py` writes the frontier table vs fork int8 and the paired k = 3 verdicts. Sigma is measured on the three full-history baselines, with the window-protocol band printed beside it.

Decision criteria, scale-up plan and exact commands: [HANDOFF.md](../artifacts/small-model/HANDOFF.md).

## 4. From checkpoint to Anki

### Where checkpoints live (all kept, none overwritten)

| Run | Checkpoint |
|---|---|
| Screens (tonight) | `artifacts/accuracy/experiments/screen-<w64\|w64kd>-s2025[-o1\|-o2]/run/epoch-01/expertium/E89-F65e-F77a/master.pth` (one folder per run; a retried epoch first moves the old folder to `epoch-01-attempt-K`) |
| Scale-up, every epoch | `artifacts/accuracy/experiments/scale-<R>-gate[-<RUN_TAG>]/run/epoch-0{1..4}/expertium/E89-F65e-F77a/{master,optimizer}.pth` |
| Scale-up, decayed final | `artifacts/small-model/scale-<R>-gate[-<RUN_TAG>]-decay/expertium/E89-F65e-F77a/master.pth` |

All paths are under the small-model worktree (`C:\Users\jonat\workspace\wt\small-model`), gitignored, not a temp or scratch folder. Nothing in the trainers deletes a `.pth`. The one risk is removing the worktree (or `git clean -x`), so `package` below copies the chosen checkpoint to the main checkout's `artifacts/models/<label>/`.

Re-launching `runbook-scaleup.sh` with the same recipe resumes: finished epochs and a finished decay are skipped. Before the 2026-10-15 fix, a re-launch silently skipped the decay, because `expertium_decay.py prepare` refuses an existing folder. `RUN_TAG=<tag>` gives a fresh run its own experiment, decay folder and held-out labels (`scale-<R>-<tag>-dec`).

### Can the fork load our model? Not as it is

The fork's runtime is Rust (`rslib/src/rwkv/mod.rs`). The Python under `qt/aqt/rwkv_inference/` is the same fixed 2.76M architecture (`architecture.py`). The runtime loads an `ARWKVWEIGHTS1` file (`WeightMap::load`, mod.rs:3192), selected by `ANKI_RWKV_MODEL_PATH` or the embedded `RWKV_trained_on_5000_10000.bin` (`qt/aqt/rwkv_scheduler.py:213`, `:4014`). That container is generic: named f32 tensors with their shapes. The model built from it is hard-coded:
- `D_MODEL = 128`, `HEADS = 4`, `HEAD_SIZE = 32` (mod.rs:57-61), `MODULE_LAYERS = [3, 4, 2, 3, 4]`, `CHANNEL_MIXER_DIMS` (mod.rs:78-79), `NUM_CURVES = 128` (mod.rs:62);
- LoRA ranks 8/16/16/16 in `TimeMixer::load` (mod.rs:5822-5860), `MAX_LORA_RANK`/`MAX_CHANNEL_MIXER_DIM` (mod.rs:296-298);
- fixed-size scratch arrays `[f32; D_MODEL]` and `[f32; HEAD_SIZE]` throughout (e.g. mod.rs:3447-3477, 5460-5482, 5776);
- a strictly sequential stream order card → deck → note → preset → user, all layers of a stream before the next (`review_features`, mod.rs:3703);
- the 128-curve softmax mixture plus the ahead-logit residual head (`SrsModel::from_weights`, mod.rs:3490).

Our tensor names follow the same scheme (`features2card.*`, `rwkv_modules.{i}.blocks.{l}.time_mixer.W_r`, `head_w.*`, `p_linear`, ...), and the block equations are the same. Every shape differs, so `load` fails at its first tensor shape check:
- **w32 (iter36):** d = 32, 1 head of 32, layers [1, 4, 3, 3, 3], channel mixers 32 wide, 64 curves / 64 points.
- **w64:** d = 64, 4 heads of 16, layers [2, 4, 1, 3, 3].

w64 also has switches the fork lacks:
- an interleaved layer schedule;
- 9 stripped channel mixers and a stripped layer-0 value LoRA;
- input feature 22 zeroed;
- a 3-term power-curve head (`gru_*` tensors) instead of the softmax mixture, and no ahead residual.

**The fork can load neither our w32 nor our w64 checkpoints.**

### What this repository now produces (`scripts/small_model_export.py`, `scripts/small_model_numpy.py`)

`package` writes a self-contained folder:
- `master.pth`: a copy of the checkpoint.
- `weights.f32.bin`: the fork's `ARWKVWEIGHTS1` container. The writer re-encodes the fork's own `RWKV_trained_on_5000_10000.bin` byte for byte (checked 2026-10-15).
- `weights.int8.bin`: `ARWKVWEIGHTS8PC1`, int8 per-output-channel linears (the scorer's `w8`) plus the f32 rest.
- `architecture.json`: every shape and switch, the layer schedule, the constants and the input layout.
- `parity.json`: the NumPy vs PyTorch check below.
- `manifest.json`: sha256 of every file, parameters, architecture, training config and phases, and the held-out summary with Δ% vs `fork/w8+s8`.

`small_model_numpy.py` is a pure-NumPy forward and the executable spec for a port. It reads only those files and returns `p_curve`, `p_imm` and `p_rating`.

| | w32 decay (iter36) | w64 |
|---|---:|---:|
| Parameters | 192,800 | 331,511 |
| `weights.f32.bin` | 798,493 B | 1,351,553 B |
| `weights.int8.bin` | 273,436 B | 436,391 B |
| Fork 2.76M f32 `.bin`, for reference: 11,082,637 B | | |

Command (CPU, about 1 min). It refuses an existing `--out`, so a package is never overwritten:
```
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
cd C:/Users/jonat/workspace/wt/small-model
$PY -X utf8 scripts/small_model_export.py package --checkpoint <master.pth> --arch <w64|iter36> --label <heldout tag> \
    --out C:/Users/jonat/workspace/anki-rwkv/artifacts/models/<heldout tag>
```

### Verified vs not

Verified on CPU (2026-10-15): the NumPy forward against the PyTorch model, on full-history batches of the smallest held-out users. The table gives max |Δp| over labelled positions, across the curve, instant and 4-rating outputs.

| Model | Users | fp32 (f32 file vs PyTorch) | int8 (int8 file vs PyTorch with `w8-pc` weights) |
|---|---|---:|---:|
| w32 `decay-8k-e10-q25` | 4676, 1745, 516 | 9.5e-7 | 7.7e-7 |
| w64 (noised random init) | 4676, 1745 | 3.6e-7 | 4.2e-7 |
| w64-sequential (noised random init) | 4676, 1745 | 3.6e-7 | 3.6e-7 |

Also verified:
- Control: on the same weights, the interleaved and sequential w64 schedules differ by up to 0.056 in p, so the check tells the schedules apart.
- The int8 file dequantizes bit-exactly to `compress_quant` w8-pc.

Not verified:
- Anything in Rust: nothing was compiled or run in the fork.
- int8 state (`s8`) in NumPy, which uses fp32 state only.
- The one-review-at-a-time form. Its reference is `expertium/model/variant_rnn.py`, checked against the parallel model by `scripts/check_expertium_catch_up.py`.
- A trained w64 checkpoint: none exists yet, so the w64 parity used noised random weights.

### Fork changes needed (for the user or Codex; this repository never edits the fork)

1. **Make the shapes data-driven.** Read `architecture.json` next to the weights file.
   - Turn `D_MODEL`, `HEADS`/`HEAD_SIZE` (per stream), `MODULE_LAYERS`, `CHANNEL_MIXER_DIMS`, the LoRA ranks (`TimeMixer::load`) and `NUM_CURVES`/num_points into runtime fields.
   - Replace the `[f32; D_MODEL]`/`[f32; HEAD_SIZE]` scratch arrays with `Vec`s, or use const generics with one instantiation per shipped architecture (d128/h32, d32/h32, d64/h16).
   - Keep the 2.76M path as the default when no `architecture.json` is present.
2. **Layer schedule.** For `layer_schedule == "interleaved"`, `review_features` runs `layer_steps` in round order (card L0, deck L0, note L0, preset L0, user L0, card L1, ...) instead of whole streams. Each stream keeps its own `v0`, and each (stream, layer) its own per-entity state, as in `VariantSrsRWKVRnn`.
3. **Variant switches.**
   - Skip the channel mixer of every layer in `channel_mixer_stripped_layers`. Its tensors are 1×1 dummies: accept them and ignore them.
   - Accept the unused 1×1 dummy `v_lora_simple` on layer 0.
   - Zero input feature 22 before `features2card`.
4. **Curve head.**
   - Power mixture: R(t) = Σᵢ softmax(w)ᵢ (1 + t / (1e-7 + Sᵢ))^(−dᵢ), where S, d = exp(clamp(·, ±25)) come from `gru_{w,s,d}_{weight,bias}` on the `head_w` output.
   - Without an ahead residual, the curve logit gets +1e-5; skip `head_ahead_logits`/`ahead_linear` (1×1 dummies).
   - For iter36, use 64 curves / 64 points instead of 128.
5. **State store.**
   - Per-layer state is H·K² per (stream, layer): 1,024 floats for w64 vs 4,096, so the state blobs change.
   - Bump `STATE_CACHE_STORE_SCHEMA_VERSION` (mod.rs:65), or key the cache by the model's sha256 from `manifest.json`, so states cached by the 2.76M model are never reused.
   - int8 state rows are `HEAD_SIZE` long (`precision.rs:49`), so 16 for w64.
6. **int8 weights (optional).** Production loads f32 only; `packed_weights.rs` is test-only int4 (mod.rs:43). Either dequantize `ARWKVWEIGHTS8PC1` at load, which shrinks the shipped file but not RAM, or keep the int8 codes resident with per-row scales.
7. **Acceptance.** The fork's Rust replay of a held-out user must reproduce `small_model_numpy.py` (or the PyTorch scorer) to about 1e-5. Then run the fork's collection replay against the 2.76M int8 baseline.

## 5. Next step: KD cost and calibration (design, 2026-10-15; nothing launched)

Starting point: `scale-w64kd-dec` (w8+s8) is +5.28% curve / +5.99% instant LogLoss and +85% instant RMSE(bins) vs fork w8+s8 (430 users). The good-candidate bar is ≤ +1.0% / +1.5% LogLoss and ≤ +15% instant RMSE(bins) (tightened 2026-10-16 16:40 by the user to ≤ +1.0% on both heads, see P). Runbook: [artifacts/small-model/runbook-next.sh](../artifacts/small-model/runbook-next.sh). All rates and times below are accuracy-lane estimates from logged training rates and queue durations, not benchmarks.

### 5.1 Where the calibration gap is (CPU diagnostic, no GPU)

Data:
- The 430-user report rows (`heldout.jsonl`), split by history size.
- New per-review dumps (`small_model_heldout.py --dump`) of fork w8+s8 and `scale-w64kd-dec` w8+s8 on the first 64 held-out users with ≤ 80k reviews: 1,087,000 benchmark reviews, CPU.
- The same 64 users scored again in the training protocol (`--window 128`: independent 128-review samples, state reset at each).
- Analysis: `scripts/small_model_calibration.py`, written to `artifacts/small-model/calibration-{full,window}.json`.

Instant RMSE(bins) by user size (430 users, size quintiles of benchmark reviews):

| Size quintile | Share of reviews | Fork | w64kd | Δ% | Users worse |
|---|---:|---:|---:|---:|---:|
| 2,000–6,865 | 2% | 0.0471 | 0.0550 | +17% | 98% |
| 6,875–13,975 | 5% | 0.0414 | 0.0532 | +29% | 99% |
| 14,125–27,405 | 9% | 0.0343 | 0.0480 | +40% | 99% |
| 27,430–66,015 | 19% | 0.0251 | 0.0414 | +65% | 100% |
| 66,180–427,540 | 66% | 0.0163 | 0.0362 | +122% | 100% |

The gap is systematic: almost every user is worse. It grows with history length. On long histories the fork's binned error falls towards the noise floor; ours stays at about 0.036, a bias that more reviews do not average away. The review-weighted +85% is dominated by the largest users.

On the 64 dumped users (instant head; the curve head is similar):
- **Not a global temperature or offset.**
  - A Platt layer (a · logit + b), fitted on half the users and applied to the other half, moves RMSE(bins) from 0.0496 to 0.0484. The fork is at 0.0343, so it closes 8% of the gap, and LogLoss improves by 0.1%.
  - The fitted slope is a ≈ 1.07–1.09 instant and 1.09–1.19 curve, against 1.00 for the fork: our model is slightly under-confident.
  - An oracle per-user constant shift (the upper bound of any per-user offset) gives 0.0485.
  - So a calibration layer on the head is cheap (it could be folded into the export) but is not the fix.
- **Range compression.** In the hardest decile the model predicts 0.388 instant against 0.361 observed (fork 0.321 against 0.325). On the curve head it predicts 0.440 against 0.407, and it is pessimistic in the 0.85–0.97 band.
- **The gap grows along the history.** Instant LogLoss gap by position in the user's history: +2.1% for reviews 0–1k, +3.7% for 1k–5k, +5.1% for 5k–20k, +5.7% beyond 20k. The instant bias drifts to +0.0056 beyond 20k reviews, against −0.0009 for the fork.

**The decisive check.** The same 64 users and reviews, scored in the training protocol (independent 128-review samples):

| Protocol | Head | Fork LogLoss | w64kd LogLoss | Δ% | Fork RMSE(bins) | w64kd RMSE(bins) | Δ% |
|---|---|---:|---:|---:|---:|---:|---:|
| 128-review windows (training protocol) | instant | 0.3382 | 0.3385 | **+0.1%** | 0.0646 | 0.0657 | **+1.7%** |
| 128-review windows | curve | 0.3519 | 0.3521 | +0.1% | 0.0694 | 0.0719 | +3.6% |
| Full history (deployment, the held-out protocol) | instant | 0.2995 | 0.3153 | +5.3% | 0.0343 | 0.0496 | +45% |
| Full history | curve | 0.3254 | 0.3381 | +3.9% | 0.0504 | 0.0652 | +29% |

In the protocol it was trained and distilled in, the 331k student already matches the 2.76M teacher, within 0.1% LogLoss and 2–4% RMSE(bins). The whole remaining gap is the benefit of long context:
- the fork gains 11.4% instant LogLoss going from windows to full history (7.5% curve);
- the student gains 6.9% (4.0% curve).

Caveats: 64 users with ≤ 80k reviews, one checkpoint, no noise band computed for this comparison; the window protocol handicaps the fork (it was trained on long segments), which is the point. So capacity is not the binding limit at this point. Neither the KD weight nor more epochs of the same recipe can teach what neither the data chunks nor the teacher's window targets contain.

**Diagnosis (confirmed by the window check).** Our students train on independent 128-review chunks, with recurrent state reset between chunks (`CHUNK_REVIEWS = 128`, hard-coded in `scripts/gpu_r01_training.py`). They are then deployed and scored over full histories of up to 430k reviews. The fork was trained on long segments (`MAX_TRAIN_GLOBAL_LEN = 66000`). The KD targets come from the teacher run on the same 128-review chunks, so they are the fork's *window* predictions, not its full-history ones. This is idea A09 (ACCURACY_IDEAS), now measured. It also explains why more of the same training saturates: the w32 at 8k users × 10 epochs is still at +6.7%. Head temperature, KD alpha and per-head calibration act on the output and cannot add the missing long-context behaviour. With longer training chunks, the online teacher also sees the longer chunks, so its KD targets improve with them (the cache is keyed by batch options, so it keeps them apart).

### 5.2 KD cost: cache the teacher's targets (implemented, CPU-checked)

Measured: KD epochs run at 108.2k reviews/s (`scale-w64kd-gate` epochs 1–3, 57.2 min tickets at 5k users), against 167.0k for w64 without KD. The teacher forward costs about 3.3 µs per review on top of the student's 6.0.

Options assessed:

| Option | Verdict |
|---|---|
| Teacher in bf16 | Already the case: `teacher_dtype` defaults to bfloat16 and none of our experiments overrides it |
| int8 teacher | No int8 GPU path in this repo. The teacher cost is the WKV recurrence and the gathers, not d=128 matmuls. Not worth building |
| Teacher targets once per dataset from a full-history replay | The scorer replays one user per stream at about 27k reviews/s, so 5k users take ≈ 3.8 h: more than the whole saving at 6 epochs. It would also change the targets (full-history instead of window predictions), so it is a recipe change, not a speed change. Kept as calibration idea SM11 |
| **Cache the online targets per training batch** | **Chosen.** A training batch is a pure function of (user, offset, reviews, seed = training_seed + user + offset); only the user order changes between epochs. So the targets of epoch 1 are exactly those of every later epoch, and of the decay too (same training seed and span) |

Implementation (`distillation.cache`, `scripts/expertium_options.py` `Teacher`; call site `scripts/gpu_r01_training.py`):
- The first time a batch is seen, its online targets are also written. Only the labelled rows the loss reads are stored: the curve probability at ahead rows and the 4 rating probabilities at query rows, float32.
- Later epochs read the file and scatter it back into zero tensors. The masked-out rows do not reach the loss.
- Each file carries a fingerprint (label shape and sum), so another batch under the same key is refused.
- The folder is `<cache>/batches-<hash>`. The hash covers every track option except the distillation spec and the bitwise-exact kernel switches, so runs that could build batches differently never share targets. A manifest pins the teacher's sha256 and dtype.
- Counters (`hits`, `writes`, `online`) go into the epoch's `result.json` as `kd_cache`.

`scripts/check_kd_cache.py` (CPU, 10 s, `artifacts/small-model/check-kd-cache.json`), on real batches of user 7 with the fork teacher and a w64 student:
- a miss returns the online targets bit for bit;
- a hit gives the same student loss and all 354 gradients bit for bit (eager mode; the control, two online passes, is bit-identical too);
- decay_alpha is applied on hits;
- a foreign batch and a foreign teacher are both refused;
- 9.9 B per training review: **3.7 GB for the 5k gate users, 5.8 GB for 8k**. That goes on disk (2.2 TB free; read rate ≈ 1.7 MB/s at 167k reviews/s), not VRAM (16 GB, full) and not pinned RAM (31 GB, shared with 5–6 preparation workers).

`check_expertium_catch_up.py` still passes (0 failed). Not yet verified on GPU: the cached epoch's real rate. The first cached epoch logs it in `phases.jsonl` (accuracy lane); a `--timing` ticket is only needed to quote it as a speed result.

Epoch estimates (accuracy-lane):

| | Online KD | Cached: epoch 1 (online + writes) | Cached: later epochs (hits) | 0.25-epoch decay |
|---|---:|---:|---:|---:|
| Gate, 5k users, 369.5M reviews | 57 min (measured) | ≈ 59 min | ≈ 38 min | 20 min online (measured), ≈ 11 cached |
| Final, 8k users, 581.9M reviews | ≈ 91 min | ≈ 93 min | ≈ 59 min | ≈ 16 cached |

Plans (training + decay + scoring of `dec` and `eN` in w8+s8, about 22 min):

| Plan | Online KD | Cached KD |
|---|---:|---:|
| 5k × 6 epochs | 6.4 h | **4.7 h** |
| 5k × 8 epochs | 8.3 h | 6.0 h |
| 8k × 4 epochs | 6.8 h | **5.1 h** |
| 8k × 6 epochs | 9.9 h | 7.1 h |
| 8k × 10 epochs | 16 h | 11.0 h |

**8k users × 6–10 epochs does not fit 6 h, even with free KD.** Six epochs of 582M reviews at 167k/s are 349 min of pure training, before the decay and scoring. In one 6 h window: 5k × 6–7 or 8k × 4. 8k × 6–10 needs two windows; the runbook resumes per epoch. Given 5.1, more epochs of the same 128-chunk recipe are expected to give diminishing returns. Spend the next window on the screen below first.

### 5.3 Calibration screen (paired k = 3, docs rule)

Design: continue from the same checkpoint (`scale-w64kd-gate` epoch 3) with 0.25-epoch decays, as A08 did. The arms differ only in the tested switch. Decay order seeds 1/2/3 give k = 3. The control is decay_alpha 0.5 at 128-review chunks; its seed 1 is the existing `scale-w64kd-dec`. Branching from one checkpoint makes the paired noise smaller than between independent trainings (A15).

| Arm | Switch | Why | Ready |
|---|---|---|---|
| A (optional), `cal-kd09-s{1,2,3}` | decay_alpha 0.9 (teacher-heavy decay) instead of 0.5 | e3 → dec with 0.5 improved curve LogLoss by 0.23 pp (instant unchanged) but raised instant RMSE(bins) from +81.3% to +84.9% (fp32). KD lowered RMSE(bins) on the screens (o0 +152% → +115%), though inside the band at k = 3 (instant −0.0043, band 0.0066; `small_model_report.py --rmse-bins`). But the window check shows the student already matches the teacher in-window, so the expected gain is small | Yes, `runbook-next.sh kd09` |
| **B (recommended)**, `cal-ctx1024-s{1,2,3}` | 1,024-review training chunks (`batch_chunks` 32, so the same 32,768 reviews per update), decay_alpha 0.5 | Tests the diagnosis in 5.1: does more context in training reduce the position drift and the RMSE(bins) gap? | Code ready (P1); runs after the GPU `check` stage passes |

Both arms share the control: seeds 2 and 3 are `runbook-next.sh control` (ready); seed 1 exists. B is the real test of the diagnosis. A is launchable today but low-value; run it only if the GPU would otherwise idle.

Readout: `small_model_report.py` on the 464 held-out users, with `--rmse-bins` (new: the same paired rule applied to RMSE(bins)):
- **LogLoss:** all three deltas negative on both heads and |mean| beyond the larger band.
- **RMSE(bins)**, also needed for a calibration fix: the same rule. The band on the 1-epoch screens was 0.0076 curve / 0.0066 instant; decays from one checkpoint should be tighter.
- For B, also re-run `small_model_calibration.py` on dumps of `cal-ctx1024-s1`. Expected if the diagnosis holds: the drift beyond 20k reviews shrinks and the gap stops growing with history length.

**Result of arm B (2026-10-15, 464 held-out users, int8 w8+s8, details in the HANDOFF "RESULT 2026-10-15"):** the diagnosis holds. 1,024-review chunks beat 128 on all four metrics under the paired k = 3 rule: LogLoss curve -0.00471 (band 0.00053, window band 0.00359) and instant -0.00541 (0.00056 / 0.00157); RMSE(bins) curve -0.00643 (0.00073) and instant -0.00915 (0.00123); all three seed deltas negative. Δ% vs fork int8: curve +5.53% to +3.95%, instant +6.06% to +4.07%, RMSE(bins) instant +89% to +45%. Cost: about 15% lower training reviews/s (accuracy lane, indicative). Recommendation: `scale` uses 1,024-review chunks; needs the user's go.

P1 for arm B, long-context training (implemented 2026-10-15; CPU-checked; its GPU check is runbook stage `check`, not yet run):
1. **Track option `chunk_reviews`** (default 128, unchanged when unset), read in:
   - `scripts/gpu_r01_training.py`: span `chunks * chunk_reviews`, `CHUNK_REVIEWS`; it refuses sample/preprocessing caches and `pack_users` with a chunk other than 128;
   - `scripts/gpu_r01_samples.samples_for`, which every preparation path calls;
   - the R01 prefetch allowlist;
   - `expertium_decay` (protocol and options).

   Evaluation inside the trainer keeps its 128-review samples. `LONG_CHUNKS_READY` marks the code.
2. **WKV beyond 383 steps.** The NVRTC kernels (generic and row-tiled exact) loop over any T and checkpoint every 32 steps; the cap was only in `nvrtc.validate`. `nvrtc.set_max_timesteps` now raises it explicitly (default still 383). The trainer raises it to 4 × chunk_reviews when the chunk exceeds 191 reviews, and restores it at the end of the case. A 1,024-review sample has 2,048 user-stream rows with its queries.
3. **CPU check, `scripts/check_long_chunks.py`** (0 failed, `artifacts/small-model/check-long-chunks.json`):
   - 1,024-review samples;
   - the fast columnar construction equals the original construction on every batch field;
   - a w64 student + fork teacher KD step on 1,024-review samples has finite loss and gradients;
   - the teacher cache round-trips the long batch;
   - the timestep limit is raised and restored;
   - decay `prepare` carries the option.

   `check_kd_cache.py` and `check_expertium_catch_up.py` still pass.
4. **GPU check, `scripts/check_long_wkv.py`** (runbook stage `check`; ticket 000355, 2026-10-15 12:18, **35/35 passed**, `artifacts/small-model/check-long-wkv.json`):
   - for K = 16 and 32 at T = 512, 1,024 and 2,100: the generic NVRTC float32 forward and all gradients against a float64 autograd reference (whose forward matches `small_model_wkv.reference` to 1.1e-7 on CPU);
   - the first 383 outputs of a long run bit-identical to a 383-step run;
   - the row-tiled exact kernel bit-identical to the generic kernel;
   - a probe of one real 32,768-review update (user 101) as 256 × 128 vs 32 × 1,024 chunks, both with the same 49,828 labelled rows: teacher + student fwd/bwd, median ms, peak MiB.

   `context` refuses to run until this check has passed. Size arm B from the probe: if 1,024-review updates are more than about 2× slower, use 0.125-epoch decays.

   Results:
   - parity: relative max error ≤ 1.9e-7 against float64, forward and all gradients, up to T = 2,100;
   - prefix and row-tiled checks: bit-identical;
   - probe (eager step times, not training rates): 499 ms at 256 × 128 against 632 ms at 32 × 1,024, about 1.3× slower; peak 5.6 → 6.0 GB.

   Arm B keeps its 0.25-epoch decays. `control` and `context` were launched at 12:19 (tickets 000356 onwards; HANDOFF).

Not proposed now (backlog):
- Head temperature folded into the export (≈ 8% of the gap, free);
- KD only on fork-unseen users (SM05);
- full-history teacher targets (SM11);
- w80 (SM06).

### 5.4 Runbook (`artifacts/small-model/runbook-next.sh`, not launched)

Launch from `C:\Users\jonat\workspace\anki-rwkv`, one stage per launch, detached (heavy GPU: the user's go first):
```
nohup "C:/Program Files/Git/usr/bin/bash.exe" C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-next.sh control \
  > C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-next-control.log 2>&1 & disown
```

| Stage | Tickets | Estimate | Output |
|---|---|---:|---|
| `check` | 1 ticket: WKV parity beyond 383 steps + 32 × 1,024 vs 256 × 128 update probe | ≈ 10 min | `artifacts/small-model/check-long-wkv.json`; gate for `context` |
| `control` | completion ticket (fork w8+s8 on the 34 users cut by time limits, `scale-w64kd-e1/e2` w8+s8 for the epoch slope; ≈ 40 min) + 2 decays (`cal-ctl-s2`, `cal-ctl-s3`; ≤ 24 min each with the smoke) + scoring (≈ 27 min) | ≈ 1.9 h | `artifacts/small-model/cal-ctl-s*/expertium/E89-F65e-F77a/master.pth`; rows in the worktree's `artifacts/small-model/heldout.jsonl` (all stages write there); teacher cache filled for about 1,300 users per seed |
| `context` | refuses until `check` passed; 3 decays at 1,024-review chunks (≈ 40 min each; re-size from the probe) + scoring (≈ 38 min) | ≈ 2.7 h | `cal-ctx1024-s*` |
| `kd09` (optional) | 3 decays + scoring | ≈ 1.6 h | `cal-kd09-s*` |
| `scale` | e.g. `TIER=gate EPOCHS=6 EPOCH1_MIN=62 EPOCH_MIN=41 EXTRA='{"chunk_reviews": 1024, "batch_chunks": 32}'` (if B wins; `SAME_RECIPE=1` for more of the reference recipe). The epoch gate (5.6) runs after epochs 1 and 2; "stop" ends the launch, `FORCE=1` continues | ≈ 4.9 h at 128-review chunks if both gates say go; ≈ 1.1–1.3 h if epoch 1 says stop (re-estimate at 1,024 from the probe) | `scale-w64kd-gate-cached[-tag]`, labels `s-w64kd-gate-{e1,e2,e6,dec}`, decisions `artifacts/small-model/gate-<run>-e{1,2}.json` |

Every training ticket (decays and epochs) starts with a 60–120 s smoke: `expertium_decay.py smoke` runs the same options and checkpoint on the trainer's smoke rows (2 users × 512 reviews) in a scratch copy under `artifacts/small-model/smoke/`, without the teacher cache. Training starts only if the smoke passes. The smoke is not run on GPU yet. CPU steps (`prepare`, reports, calibration analysis) run outside the queue and claim nothing.

After `control` and an arm, run on all 464 users, then again with `--min-user-id 101` to exclude the upstream validation users 1–100 (today +5.27% / +5.99% either way). Report both:
```
$PY -X utf8 scripts/small_model_report.py --rmse-bins \
  --labels fork/w8+s8,scale-w64kd-dec/w8+s8,cal-ctl-s2/w8+s8,cal-ctl-s3/w8+s8,cal-ctx1024-s1/w8+s8,cal-ctx1024-s2/w8+s8,cal-ctx1024-s3/w8+s8 \
  --screen "ctx1024-vs-ctl:cal-ctx1024-s1/w8+s8=scale-w64kd-dec/w8+s8,cal-ctx1024-s2/w8+s8=cal-ctl-s2/w8+s8,cal-ctx1024-s3/w8+s8=cal-ctl-s3/w8+s8"
```
For arm A, replace `cal-ctx1024` with `cal-kd09`.

Recommended order:
1. `check` (≈ 10 min). P1 code and CPU checks are done.
2. `control` and `context` in one night (≈ 4.6 h). `control` does not depend on `check`.
3. If B passes, size a full long-context run from B's measured rate: train from scratch or from epoch 3 at long chunks, cached KD. If it does not, the window check still says the gap is long-context behaviour. Next would be SM11 (full-history teacher targets) or longer chunks, not `scale` with the same 128-chunk recipe. `scale` 5k × 6 cached (4.7 h) stays available as the plain more-training baseline.


### 5.5 Review notes from the main session (2026-10-15), and what was adopted

- **Fairness.** The 464 users are fork-unseen (id < 5000) and outside every student training set, and the fork is re-scored by our code. The main bias favours the fork: 128-chunk training against full-history scoring, as 5.1 measures. Held-out B contains student training users: never use it for students.
- **Time-limit cuts.** The 34 users cut by time limits are long-history users, 14% of reviews. Adopted: `control` first completes `fork/w8+s8` on them, so reports can use all 464 users.
- **Users with id ≤ 100** (upstream validation range). Adopted as an option, `small_model_report.py --min-user-id 101` (418 users). `scale-w64kd-dec` stays +5.27% / +5.99%.
- **Scoring cost.** Full-history scoring was 42% of small-model GPU time: one user per stream, plus resumes.
  - Fork rows are scored once and reused from `heldout.jsonl`. Every stage scores w8+s8 only, with tickets sized by label count.
  - Batching several users per pass is feasible but not done. `collect_predictions` keys predictions by review index only, so a batched pass needs a per-sample user map and length bucketing. It also needs a GPU equivalence and speed check before any result depends on it (ledger S05).
- **Epoch slope.** Asked for a 1-epoch point at 5k users before multi-epoch runs. Adopted:
  - `control` scores the existing `scale-w64kd-gate` epoch-1 and epoch-2 checkpoints at no training cost;
  - every multi-epoch run passes the epoch gate (5.6) after epochs 1 and 2, the first epoch of the real run doubling as the cheap check.
- **Smoke checks.** Adopted: every training ticket begins with `expertium_decay.py smoke` (60–120 s, same options) and trains only if it passes.
- **Longer training context and matching KD targets.** Adopted as P1 (5.3). With longer chunks the online teacher sees the same long chunks, so its targets match the student's context. The cache keeps them apart from 128-review targets (batch-option hash). Full-history teacher targets (SM11) remain a later option.


### 5.6 Epoch gate: the first epoch of the real run is the cheap check (user request, 2026-10-15)

The check is never a throwaway run. `runbook-next.sh scale` (and any later multi-epoch runbook, via its `epoch_gate` function):
1. Trains epoch 1 of the real experiment. The checkpoint is saved in the run folder, and later epochs resume from it.
2. Scores that checkpoint (w8+s8) on a **fixed 100-user subset**, `artifacts/small-model/gate-users.json`:
   - 10 users per history-length decile of the 464, evenly spaced ranks inside each decile, ids ≥ 101;
   - 21.9% of the 464 users' reviews (3.7k to 861k reviews per user).

   The rows go into `heldout.jsonl` under the run's normal label (`<lbl>-e1/w8+s8`). The ticket also scores the fork and the reference run `scale-w64kd-gate` (e1, e2, e3, dec) on the same users, skipping rows that already exist:
   - first gate ever: ≈ 15 min (the reference labels are new);
   - later gates: ≈ 3–4 min (one label, 2.3 min of scoring + model loading).
3. Decides on the CPU: `scripts/small_model_gate.py` writes `artifacts/small-model/gate-<run>-e<K>.json` (metrics, rule, thresholds, verdict).
4. **Go:** the runbook continues with the remaining epochs, the decay and full scoring. **Stop:** the launch ends cleanly and logs why; everything stays resumable, and a re-launch with `FORCE=1` continues from the saved epoch.
5. Re-checks the same way after epoch 2 (runs of 3 or more epochs).

Rule: paired, review-weighted LogLoss on the subset users. Both heads must pass every test.

| Test | When | Passes if | Why this threshold |
|---|---|---|---|
| A, matching point | every gate | cand_eK − ref_eK < 0 with its 95% user-bootstrap upper bound < 0, and ≤ −band on at least one head. Band: 0.0062 curve / 0.0027 instant | The k = 1 whole-training noise band (A15): a single run must beat one seed's noise |
| A′, no regression | `SAME_RECIPE=1` (more users or epochs of the reference recipe, which cannot beat itself) | cand_eK − ref_eK ≤ +band on both heads | Same band, as a non-inferiority margin |
| B, projection | every gate | projected final ≤ (1 − 1%) × ref_dec. Projection: cand_eK + b · (ln N − ln K) + (ref_dec − ref_e3) | b is the log-epoch slope: the reference's least-squares fit over its epochs 1–3 at K = 1, the candidate's own (e2 − e1)/ln 2 at K = 2. 1% of LogLoss is about 1 pp of the gap to the fork, the smallest gain worth a multi-hour run |
| C, not flattening | K = 2 | cand_e1 − cand_e2 ≥ 0.5 × (ref_e1 − ref_e2) | Stop early if the gain per epoch has halved against the reference |

A subset user without a row for any label gives verdict "stop" with reason "incomplete". The log-epoch projection is a prior for a go/no-go, not a forecast to report.

Checked on CPU:
- `small_model_gate.py --self-test`: 8 synthetic cases (go; inside the band; flattening; still improving; beats the band but short of the margin; same recipe with more epochs; same recipe with the same epochs; missing row), all as intended.
- The `epoch_gate` bash branch, run with a stubbed queue: go → continue; stop → launch ends with exit 0; stop + `FORCE=1` → continue.
- A real-data dry run reports the reference rows still missing on the subset. The `control` stage, or the first gate ticket, scores them.
- The subset scoring has not run on GPU: queuing a GPU ticket from this session was refused, so it needs the user's go.

Time (5k users × 6 epochs cached, 128-review chunks, accuracy-lane estimates):

| Outcome | GPU time used | Saved vs the full run (≈ 4.9 h) |
|---|---:|---:|
| Stop at epoch 1 | ≈ 1.1–1.3 h (61 min epoch + 4–15 min gate) | ≈ 3.6–3.8 h |
| Stop at epoch 2 | ≈ 1.9 h | ≈ 3.0 h |
| Go at both gates | ≈ 4.9 h (+ ≈ 8 min for the two gates) | — (+3%) |

For 8k users (final tier), epoch 1 has 1.6× the reference's data per epoch, so A is easier there. Read that decision with this in mind.

### 5.7 Chunk-size probe, then the real run (user go 2026-10-15 ~16:30)

Screen B showed the context is the main lever: 1,024-review chunks beat 128 by about 9x the k = 3 band. Before spending ~6 h of GPU on `scale`, one cheap decay asks whether longer still helps.
- **Probe:** `cal-ctx2048-s1`, a 0.25-epoch decay at 2,048-review chunks (batch_chunks 16, same 32,768 reviews per update), seed 1, paired with `cal-ctx1024-s1` (k = 1). The long-WKV parity check covers T up to 2,100.
- **Pick** (`scripts/small_model_chunk_pick.py`, decision in `artifacts/small-model/chunk-pick.json`):
  - use 2,048 only if it has all 464 users,
  - and improves LogLoss by at least 0.001 on both heads, with the 95% user-bootstrap upper bound < 0,
  - and its decay trains at most 1.5x as long as the 1,024 one;
  - otherwise use 1,024, which is also the fallback if the probe fails (for example out of memory).
  - The margin is about 2x the k = 3 band and a fifth of the 128 → 1,024 effect, so a single seed cannot pass on noise.
- **Real run:** `runbook-next.sh probe-scale` chains the probe, the pick and `scale` with TIER=gate, EPOCHS=4 and RUN_TAG=ctx<chunk>. The labels are `s-w64kd-gate-ctx<chunk>-e<K>` and `-dec`.
  - The epoch gate (5.6) runs after epochs 1 and 2 against `scale-w64kd-gate` (128 chunks).
  - Epoch estimates are 61 min scaled by the measured decay times (about 73 min at 1,024).

## Priorities (re-assessed after each final result; AGENT.md "Priority re-assessment")

Previous order (2026-10-15 16:30, after screen B): 1. `probe-scale`; 2. SM11 + SM12; 3. SM06, SM04, SM05, run-110 port.

### Re-assessment 2026-10-16 ~01:00 (Opus, AGENT.md "Priority re-assessment"; proposal, nothing enqueued)

Input: `scale-w64kd-gate-cached-ctx2048` (w64, cached fork KD, 2,048-review chunks x 16, gate tier, 4 epochs), epoch gates on the fixed 100-user subset, w8+s8 vs fork w8+s8 (subset fork 0.28665 / 0.26105):

| Checkpoint | Curve | vs fork | Instant | vs fork | Δ vs reference, upper95 (curve / instant) |
|---|---:|---:|---:|---:|---|
| e1 | 0.29771 | +3.86% | 0.27421 | +5.04% | −0.0038 / −0.0060 |
| e2 | 0.29595 | +3.24% | 0.27078 | +3.73% | −0.0042 / −0.0062 |
| e2 projection of the final (gate rule B) | 0.29368 | +2.45% | 0.26707 | +2.31% | — |
| Previous best, `scale-w64kd-dec` (128 chunks) | 0.30092 | +4.98% | 0.27731 | +6.23% | — |

- Reading: the long-context recipe roughly halves the gap again, and its epoch slope is steeper than the 128-chunk reference (e1→e2 −0.00176 / −0.00343 vs −0.00105 / −0.00303). The projection stays above the good-candidate bar (≤ +1.0% curve / +1.5% instant, instant RMSE(bins) ≤ +15%).
- The subset is not the frontier: for the 128-chunk run the 464-user Δ% was 0.3–1.1 pp higher on curve than the subset (e1 +6.85 vs +5.79, e2 +5.72 vs +5.42, dec +5.40 vs +4.98) and about equal on instant. Expected 464-user final: about +2.8–3.5% curve / +2.3–2.5% instant. Pruning is therefore not expected to start from this run alone.
- Training rate (accuracy lane, `phases.jsonl`): epoch 1 80,842 reviews/s (teacher online, cache writes), epoch 2 102,143 (cache hits); 8k/2k 10-epoch ETA at 102k: 95 min per epoch, about 16 h.
- **Correction to the run's expected sequence:** `runbook-next.sh scale` gates only after epochs 1 and 2 (`[ "$epoch" -le 2 ]`). There is no e3 or e4 gate: epoch 3 → epoch 4 → 0.25-epoch decay (cached, ≈ 18–27 min) → 464-user scoring of `-e4` and `-dec` (≈ 22 min) → resume ticket. GPU free at about **03:10** (±10 min), not 02:50.

Ranked top 3:
1. **Finish the ctx2048 run, then continue it conditionally** (queue item Q4: epochs 5–6, or pruning if the final already meets the bar). Reason: the only lever with measured momentum; the log-slope says 4→6 epochs gives about −0.35 pp curve / −0.75 pp instant. GPU: run ends ≈ 03:10; continuation ≈ 3 h. Decides whether more training of w64 alone approaches the bar.
2. **SM13, in-window@2048 diagnostic** (Q2). Reason: at 128 the student matched the teacher in-window (+0.1%), which showed the gap was context, not capacity; the same check at 2,048 says whether the residual is capacity (then w80 or more epochs help, and pruning will cost accuracy) or context beyond 2,048 reviews (then only longer context helps, and pruning can be tried). GPU ≈ 10–15 min. Decides the daytime run (Q7).
3. **SM04, CPU inference timing fork vs w64 vs w32** (Q1). Reason: "lighter" is half the goal and the user asked how much CPU w64 saves; we only have the 8.3x matrix-parameter estimate. Cost: one quiet-machine CPU ticket, ≈ 15 min, plus a small script. Decides the CPU claim in the frontier table and whether w64 already saves enough CPU to justify the accuracy gap.

Next after these: S05 batched full-history scoring (speed of every gate and final; queue item Q5), the S06 long-context training speed profile (Q3), then SM11 / SM16 (context beyond 2,048, design).

Made pointless or mis-sized by this result:
- `runbook-next.sh kd09` (arm A): pointless. KD weight is not the lever (5.1), and its decays branch from the 128-chunk `scale-w64kd-gate` epoch 3.
- `scale` 5k x 6 epochs at 128-review chunks, kept in 5.4 as the "plain more-training baseline": pointless; the 128-chunk recipe is superseded on every metric.
- SM06 as written (w80 + KD with a longer decay, 128 chunks): mis-sized; only at 2,048 chunks and only if SM13 says the student is capacity-bound (Q7).
- The epoch gate's reference (`scale-w64kd-*`, 128 chunks) is now a weak bar: test B passed by 0.0030 / 0.0050 (e1) and 0.0042 / 0.0075 (e2) below its target. Later runs need the ctx2048 run's own epochs as reference (gate script option, Q4/Q7 code).
- SM11 (full-history teacher targets) was sized at ≈ 3.8 h of batch-1 replay; with S05 it shrinks, and its value now depends on SM13.
- Wide-kernels rerun (w32 control, w80 pair, w64 generic): cannot run while desktop apps hold 3.4 GB of VRAM (precondition ≤ 1.3 GB); stays parked, not in the queue.
- The unconditional `score` resume ticket (≈ 1–2 min of GPU and 15 queue minutes even when 0 users remain): the parked resume-skip tweak; apply it after the running launch ends (Sonnet medium, CPU only).

### Queue 2026-10-16 (from ~02:50 until 20:00, ≈17 h)

Main-session launch list, in order; every GPU step goes through `scripts/r01_queue.py` (detached). `--timing` items run only in the quiet night window (until ≈ 08:00); daytime slots, when the user may be at the PC with ≈ 3.4 GB of VRAM in desktop apps, take accuracy runs. Abbreviations: `PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe`, `QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py`, `WT=C:/Users/jonat/workspace/wt/small-model`, `SM=$WT/artifacts/small-model`, `SC=$WT/scripts/small_model_heldout.py`, `DEC_PY=$WT/scripts/expertium_decay.py`, `CKPT=expertium/E89-F65e-F77a/master.pth`, `TEACHER=C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth`, `RUN=$WT/artifacts/accuracy/experiments/scale-w64kd-gate-cached-ctx2048`, `LBL=s-w64kd-gate-ctx2048`.

**Branch inputs**, computed on CPU when the run ends (Sonnet medium, `small_model_report.py --rmse-bins` on all 464 users):
- **F** = 464-user Δ% vs `fork/w8+s8` of the better of `$LBL-dec` and `$LBL-e4` (w8+s8), per head.
- **S** = slope check on the 100 gate users: e2 − e4 LogLoss. The e1→e2 log-slope predicts −0.00176 curve / −0.00343 instant for e2→e4. "Still learning" = at least half of that (≥ 0.0009 curve and ≥ 0.0017 instant), the gate's test C fraction.
- **G** = SM13's in-window gap (Q2): student@w2048 vs fork@w2048, review-weighted, same users. "Context-bound" if G ≤ the k = 3 window band expressed in Δ% (≈ +1.0% curve, ≈ +0.5% instant: 0.00359 / 0.00157 of ≈ 0.35 window LogLoss); "capacity-bound" if above it on either head.

**Thresholds:**
- **P (start pruning):** F ≤ +1.0% curve and ≤ +1.0% instant vs fork w8+s8 on the 464 users (user, 2026-10-16 16:40; was ≤ +1.5% instant before, verdicts above that date used the old bar; RMSE(bins) reported beside it). Pruning only makes sense with that headroom: Expertium's d=128 → d=64 cost +0.5% / +1.2%.
- **E (extend the same run):** not P, F ≤ +4.0% on both heads (4 epochs must at least beat a single 0.25-epoch ctx1024 decay, +3.91% / +4.13%), and S "still learning".
- **D (else):** no continuation of this recipe tonight; Q4 stays empty.

| # | Lane | Item | Decides | Ready? Who writes | Launch | GPU min | Skip / stop |
|---|---|---|---|---|---|---:|---|
| Q1 | speed/goal, night, `--timing` | **SM04 CPU inference timing**: fork (`srs_model_rnn`, fork repo read-only import), w64 and w32 recurrent step, 1 thread, f32 and int8 weights; per-review latency at batch 1 (runtime case) and at batch 256 cards (amortises Python dispatch), plus analytic MACs per review | The CPU saving of w64 vs the fork (now only the 8.3x parameter estimate) | Ready and CPU-checked (2026-10-16): `scripts/small_model_cpu_timing.py`; smoke `python -X utf8 scripts/small_model_cpu_timing.py --threads 1 --smoke --out <json>` runs in ≈ 14 s | `$PY $QUEUE run --owner small-model --purpose "small-model: SM04 CPU inference timing (1 thread)" --claim gpu --timing --expected-minutes 15 -- $PY -X utf8 $WT/scripts/small_model_cpu_timing.py --threads 1 --out $SM/cpu-timing.json` (claims the GPU so no training workers load the CPU) | 0 (15 of machine) | Skip if the script is not CPU-checked by 03:10; then run it after Q5, still before 08:00 |
| Q2 | accuracy | **SM13 in-window@2048 diagnostic** on the 100 gate users | G (capacity vs context beyond 2,048) | Runnable as-is (`--window` exists) | `$PY $QUEUE run --owner small-model --purpose "small-model: SM13 in-window@2048 fork vs ctx2048 dec" --claim gpu --expected-minutes 15 -- $PY -X utf8 $SC --users-file $SM/gate-users.json --window 2048 --model fork=fork::$TEACHER --model $LBL-dec=expertium:w64:$WT/artifacts/small-model/scale-w64kd-gate-cached-ctx2048-decay/$CKPT --only fork/w8+s8 --only $LBL-dec/w8+s8 --time-limit 900 --out $SM/heldout.jsonl` (rows get `@w2048`, never mixed with full history) | 10–15 | Users with an OOM row on either label are dropped from G (report how many); if more than 20 drop, rerun with `--max-reviews 200000` |
| Q3 | speed, night, `--timing` | **S06 long-context training speed profile** (160 users, `expertium_speed.py`): w64 wide 256 x 128 control, 32 x 1,024, 16 x 2,048, 32 x 2,048; no KD; all at `gpu_memory_fraction` 0.8 | Whether ctx2048 costs the recurrence (few rows, long T) or the KD cache (102k cached vs the no-KD rate): next speed work is cache I/O (S07) or the K=16 native WKV / bigger batches (W05) | Ready 2026-10-16: 4 cases `w64-ctx128-b256`, `w64-ctx1024-b32`, `w64-ctx2048-b16`, `w64-ctx2048-b32` in `scripts/expertium_speed.py cases()` + `$SM/jobs/speed-ctx.sh` (logs `nvidia-smi` first; exits 2 if VRAM in use > 3.6 GB at start). CPU-checked: `prepare --users 160` writes the 4 cases (w64, wide ports kept, no KD, fraction 0.8) and all their option keys are in the `gpu_r01_prefetch.py` allowlist; `bash -n` ok; nothing run on GPU | `$PY $QUEUE run --owner small-model --purpose "small-model: S06 long-context speed profile" --claim gpu --timing --expected-minutes 25 -- "C:/Program Files/Git/usr/bin/bash.exe" $SM/jobs/speed-ctx.sh` | 25 | A case is invalid if `gpu_power.jsonl` shows < 150 W or memory pinned above 15.5 GB (spill); skip the item if VRAM in use exceeds 3.6 GB at start |
| Q4 | accuracy | Branch **E**: epochs 5–6 of the same run, a 100-user score after epoch 5, decay from e6, 464-user scoring of `$LBL-e6` and `$LBL-dec6`. Branch **P**: w48 pruning run, see Q7-P (it moves here). Branch **D**: nothing | Whether 6 epochs reach the instant bar (log-slope expectation ≈ +1.6–1.8% instant, ≈ +2.5–3.2% curve on 464 users) | **Ready 2026-10-16** (`$SM/runbook-2026-10-16.sh extend`, CPU-checked and dry-run with a stubbed queue; the LR is constant 7e-4 and independent of the epoch count, so 4 → 6 in `experiment.json` and `run/run.json` is consistent; it refuses unless epoch-04 is complete, the old runbook has exited and the e4 decay exists; the decay goes to `$SM/scale-w64kd-gate-cached-ctx2048-e6-decay`). Spec: a NEW file `$SM/runbook-2026-10-16.sh` (never edit `runbook-next.sh`), stage `extend`: CPU-check that an epoch-5 `protocol.json` has the same LR/optimizer fields as epoch 4 and resumes from `epoch-04`, then set `experiment.json` epochs 4 → 6; one ticket per epoch (smoke first, `--expected-minutes 66`); score e5 on the gate users (`--expected-minutes 6`); decay into `$RUN-e6-decay` (≈ 27); scoring (30) and a resume ticket only if users remain. Dry run with a stubbed queue, as done for `epoch_gate` | `TIER=gate EPOCHS=6 RUN_TAG=ctx2048 nohup bash $SM/runbook-2026-10-16.sh extend > $SM/runbook-2026-10-16-extend.log 2>&1 & disown` | ≈ 175 | Stop after epoch 5 if e4 − e5 ≤ 0 on both heads (flattened); do not start later than 05:00 (it must finish before Q7) |
| Q5 | speed, night, `--timing` | **S05 batched full-history scoring** (`--batch-users N`, length-bucketed): tolerance + timing on the 100 gate users against existing rows of `$LBL-e2/w8+s8` | Whether gates and finals can move to 464 users at today's 100-user cost (model time is ≈ 8.6 of the ≈ 11 min per 464-user label, all of it batch-1 recurrence at 62–78k reviews/s) | Code first, **Opus** (design: per-sample user map in `collect_predictions`, padding, per-user int8 state rows; ≈ 2 h) + CPU equivalence check on ≈ 10 small users (`--device cpu`), thresholds as AGENT.md "Evaluation tolerance" | `$PY $QUEUE run --owner small-model --purpose "small-model: S05 batched scoring tolerance + timing" --claim gpu --timing --expected-minutes 12 -- $PY -X utf8 $SC --users-file $SM/gate-users.json --batch-users 8 --model $LBL-e2=expertium:w64:$RUN/run/epoch-02/$CKPT --only $LBL-e2/w8+s8 --out $SM/s05-check.jsonl` (then a CPU compare against `heldout.jsonl`) | 12 | Skip if the CPU check has not passed by 07:30 (no timing after 08:00); adopt only if within tolerance and ≥ 2x faster in model seconds |
| Q6 | accuracy, day | **ctx4096 decay probe**: 0.25-epoch decay of the best epoch (e6 after E, else e4) at 4,096-review chunks x 8, k = 1 vs that epoch's 2,048 decay | Whether chunk length still pays beyond 2,048 (1,024 → 2,048 gave −0.00095 / −0.00106); if not, longer context needs SM16/SM11 (design) | Runnable as-is (explicit commands; cold cache for 4,096 chunks; T = 8,192 rows relies on the any-T kernel loop and the smoke, as 2,048 did) | `$PY -X utf8 $DEC_PY prepare --source-run $RUN/run --epoch <K> --out $SM/cal-ctx4096-e<K> --fraction 0.25 --shape one_minus_sin --evaluate none --seed 1 --extra-options '{"distillation": {"teacher": "<TEACHER>", "alpha": 0.9, "decay_alpha": 0.5, "cache": "C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-teacher/fork-bf16-s2025"}, "chunk_reviews": 4096, "batch_chunks": 8}'`; then `$QUEUE run ... --expected-minutes 40 -- bash -c "'$PY' -X utf8 '$DEC_PY' smoke --out '$SM/cal-ctx4096-e<K>' && '$PY' -X utf8 '$DEC_PY' run --out '$SM/cal-ctx4096-e<K>'"`; then scoring `--expected-minutes 16 -- $PY -X utf8 $SC --model cal-ctx4096-e<K>=expertium:w64:$SM/cal-ctx4096-e<K>/$CKPT --only cal-ctx4096-e<K>/w8+s8 --time-limit 3000 --out $SM/heldout.jsonl` | ≈ 55 | Pick rule as `chunk_pick` (≥ 0.001 better on both heads, upper95 < 0, ≤ 1.5x slower); skip in branch P |
| Q7 | accuracy, day | Main daytime run, chosen at ≈ 08:30 from F (after Q4), G and Q6 (chunk = 4,096 if Q6 passed, else 2,048), 4 epochs + gates after **every** epoch + decay + 464-user scoring: **P** (F ≤ bar): **w48** student (3 heads x 16, 195,949 parameters, 14x fewer than the fork), same KD recipe, non-inferiority gates (test A′, band 0.0062 / 0.0027) against the ctx2048 run's own epochs. **Capacity-bound** (G above band): **w80** (558,209) at the chosen chunk, test-A gates against the ctx2048 run. **Context-bound, Q6 passed**: 2 more epochs of the best checkpoint at 4,096 chunks + decay. **Context-bound, Q6 failed**: epochs 7–8 only if e4→e6 gave ≥ half its expectation (0.0005 / 0.0010 on the gate users), else nothing (design day for SM16/SM11) | The next recipe step toward the bar, or the first pruning step | Code first, Sonnet medium (≈ 1.5 h, during the night): stage `arch` in `runbook-2026-10-16.sh` (ARCH and reference-label variables, scoring with `expertium:<ARCH>:`, gate after every epoch), `w48` preset in `expertium/variants.py` (`w80_config(heads=3, mult=2)`, CPU param count 195,949), `small_model_gate.py --reference-prefix` (self-test extended), and `check_wide_kernels.py` accepting w48 (one GPU parity ticket ≈ 3 min before training). Optional speed add-on S07 (Opus review): share the KD cache across student architectures | `TIER=gate EPOCHS=4 ARCH=<w48 or w80> CHUNK=<2048 or 4096> nohup bash $SM/runbook-2026-10-16.sh arch > $SM/runbook-2026-10-16-arch.log 2>&1 & disown` | w48 ≈ 330, w80 ≈ 420–450, 4,096 continuation ≈ 220, epochs 7–8 ≈ 175 | Any gate "stop" ends the run; nothing may start after 15:00 that cannot finish its scoring by 20:00; otherwise leave the GPU idle |

Expected timeline (branch E, context or capacity verdict unknown): Q1 03:10–03:25, Q2 03:25–03:40, Q3 03:40–04:05, Q4 04:05–07:10, Q5 07:10–07:25, Q6 07:25–08:30, Q7 08:30–≈ 16:00 (w80) or ≈ 14:00 (w48), idle after. In branch P, Q7-P takes Q4's slot at ≈ 04:15 and Q5/Q6 drop to the next quiet window. Alternation: speed Q1, accuracy Q2, speed Q3, accuracy Q4, speed Q5, accuracy Q6–Q7 (daytime is accuracy-only because timing needs a quiet machine). Writers to start now: Sonnet medium for Q1, Q3, Q4 and the post-verdict report; Opus for Q5; Sonnet medium for Q7 once F and G are known (≈ 03:45).

**Is the 100-user gate scoring still relevant now that evaluation is faster?** Yes, keep it for go/stop, but score every epoch. The faster evaluation (EV01, 300 users in ≈ 2 min) is the in-training window-protocol evaluator: it scores 128-review samples and falls back to the reference path for Expertium variant heads (EV08). The gap we are closing is the full-history one (5.1: +0.1% in-window vs +5% full history), and full-history scoring is still one user per pass: ≈ 1.5 min of model time out of a ≈ 3.4 min ticket for 100 users, ≈ 11 min per label for 464. Moving every gate to 464 users would add ≈ 7.6 min per gate (≈ 26 min on a 4-epoch run with a gate after each epoch, about 9% of its training) for no change in decision: both gates passed with upper95 at −0.0038 to −0.0062 (curve CI half-width ≈ 0.0017, instant ≈ 0.0007), far from 0 and from the bands. The subset is less useful for the level vs the fork (its curve Δ% ran 0.3–1.1 pp below the 464-user value), so finals stay on 464 users. Recommendation: (1) keep 100 users for gates; (2) add a 100-user score after every epoch (≈ 3.4 min, ≈ 5% of an epoch), because the missing e3 point is why tonight's continuation rule must use e2→e4; (3) escalate a single gate to 464 users when it is close to its boundary (upper95 > −0.001, or projection within 0.002 of the target); (4) once S05 passes (Q5) and a 464-user label costs ≤ ≈ 4 min, move gates to all 464 users, so the last epoch's gate doubles as its final score.

### Result 2026-10-16 03:00: scale-w64kd-gate-cached-ctx2048 (4 epochs + decay, 464 users)

Full-history scoring of `s-w64kd-gate-ctx2048-e4` and `-dec` finished at 03:00 (tickets 000380 + 000381, exit 0); all 464 users scored. `small_model_report.py --rmse-bins`, reference `fork/w8+s8`, review-weighted, 95% user-bootstrap CI. Only w8+s8 rows exist for the student labels (no fp32 rows; `fork/fp32` has 430 users and is not paired).

| Label (w8+s8) | Curve LogLoss | Δ% [CI] | Instant LogLoss | Δ% [CI] | Curve RMSE(bins) | Δ% | Instant RMSE(bins) | Δ% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fork/w8+s8 | 0.29766 | +0.00% | 0.27216 | +0.00% | 0.03867 | +0.0% | 0.02084 | +0.0% |
| s-w64kd-gate-ctx2048-e4 | 0.30542 | +2.61% [+2.05, +3.33] | 0.27891 | +2.48% [+2.29, +2.69] | 0.04648 | +20.2% | 0.02678 | +28.5% |
| s-w64kd-gate-ctx2048-dec | 0.30408 | +2.16% [+1.62, +2.90] | 0.27804 | +2.16% [+1.97, +2.35] | 0.04441 | +14.8% | 0.02481 | +19.0% |

- **F** (better of the two = `-dec`): **+2.16% curve / +2.16% instant**. The 464-user value was inside the expected +2.8–3.5% / +2.3–2.5% or better on both heads (previous best, 128-chunk `scale-w64kd-dec`: +5.28% / +5.99%).
- **S** (100 gate users, w8+s8, fork 0.28665 / 0.26105): e2 0.29595 / 0.27078, e4 0.29359 / 0.26818 (dec 0.29233 / 0.26731), so S = e2 − e4 = **0.00236 curve / 0.00260 instant**. Threshold for "still learning" is 0.0009 / 0.0017: both heads pass, with the e2→e4 slope at 1.3x / 0.76x of the log-slope prediction (−0.00176 / −0.00343).
- **Verdict: branch E.** F misses P (curve +2.16% > +1.0%, instant +2.16% > +1.5%; RMSE(bins) +14.8% / +19.0% is also above the +15% instant bar), F is within 4.0% on both heads, and S says still learning. G (Q2) only affects Q7. The file `gate-scale-w64kd-gate-cached-ctx2048-e3-extend.json` exists (mtime 2026-10-16 00:38) and was ignored (`verdict: incomplete`, dry-run artefact).

### Result 2026-10-16 03:06: SM04 CPU inference timing

Queue Q1 (ticket 000382) finished at 03:06, exit 0, in 4.5 min (15 expected). Script `scripts/small_model_cpu_timing.py`, result `artifacts/small-model/cpu-timing.json`. Recurrent step, 1 thread, Ryzen 7 9850X3D, per-review medians (1,000 steps at batch 1, 60 at batch 256, 5 repeats). w32 uses random init (no checkpoint; speed does not depend on values).

| Model | Params | MACs/review | f32 b1 (ms) | f32 b256 (ms) | int8 b1 (ms) | int8 b256 (ms) | Speed-up vs fork f32 (b1 / b256) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fork | 2,762,884 | 2.91M | 3.854 | 0.1871 | 10.150 | 0.2530 | 1.00x / 1.00x |
| w64 | 331,511 | 0.341M | 3.452 | 0.1008 | 7.054 | 0.1374 | 1.12x / 1.86x |
| w32 | 192,800 | 0.190M | 4.289 | 0.0606 | 9.839 | 0.0738 | 0.90x / 3.09x |

- **Batch 1 is dispatch-bound.** w64 has 8.5x fewer MACs but is only 1.12x faster than the fork (w32 is slower than the fork), so the per-step cost is PyTorch per-op overhead, not arithmetic.
- **Batch 256 amortises dispatch**: w64 1.86x and w32 3.1x faster, closer to (but still well under) the MAC ratio.
- "int8" is torch dynamic-quantized Linear (onednn) and is slower than f32 here; it is an int8-kernel indication only, not the fork's runtime.
- **Implication:** with plain eager PyTorch the small model saves memory and disk, not CPU per review at batch 1. The CPU saving is realised only with a fused/low-overhead runtime (hand-written step, TorchScript/compiled graph) or by batching cards (SM18).

### Re-assessment 2026-10-16 ~03:30 (G and Q7)

Opus, CPU only (nothing enqueued or cancelled). Queue at 03:25: 000385 (Q4 epoch 5/6) running since 03:20, 000386 (S06 rerun, `--timing`) waiting. Epochs 3–5 run at ≈ 54–57 min (109–115k reviews/s), not the 66 budgeted, so Q4 (e5 + gate score, S06 25 min, e6, decay, 464-user scoring) ends at ≈ **06:50–07:35**.

**G (SM13, ticket 000383, 03:19, exit 0).** 100 gate users, w8+s8, each history cut into independent 2,048-review samples (the training protocol); `small_model_report.py`, review-weighted, 95% user-bootstrap CI. 100/100 users scored on both labels (no OOM rows, none dropped).

| Label (w8+s8), 100 gate users | Curve LogLoss | Δ% vs fork, same protocol [CI] | Instant LogLoss | Δ% [CI] | Curve RMSE(bins) Δ% | Instant RMSE(bins) Δ% |
|---|---:|---:|---:|---:|---:|---:|
| fork @w2048 | 0.31664 | +0.00% | 0.29666 | +0.00% | +0.0% | +0.0% |
| s-w64kd-gate-ctx2048-dec @w2048 | 0.31557 | **−0.34%** [−0.93, +0.11] | 0.29808 | **+0.48%** [+0.32, +0.66] | +2.5% | +3.4% |
| fork, full history | 0.28665 | +0.00% | 0.26105 | +0.00% | +0.0% | +0.0% |
| s-w64kd-gate-ctx2048-dec, full history | 0.29233 | **+1.98%** [+1.33, +2.64] | 0.26731 | **+2.40%** [+2.01, +2.79] | +12.9% | +17.9% |

- **G = −0.34% curve / +0.48% instant: context-bound** (band ≈ +1.0% / +0.5%). Instant is at the edge of the band (CI excludes 0), curve is level.
- **What it means.** Inside a 2,048-review window the w64 student (8.3x fewer parameters) predicts as well as the fork on curve and within 0.5% on instant, so with the same context it does not lack capacity. With the full history the gap is +1.98% / +2.40%. Both models gain from the history beyond 2,048 reviews (the fork 0.0300 curve / 0.0356 instant LogLoss, the student 0.0232 / 0.0308), but the student realises only 77% / 86% of the fork's gain. Split of the full-history gap (0.00568 curve / 0.00626 instant): curve is entirely beyond-window (in-window −0.00107), instant is ≈ 77% beyond-window (0.00484) and ≈ 23% in-window (0.00142). Had the in-window gap been close to the full-history one, the gap would be capacity (w80 or more epochs of the same recipe). It is not: what is missing is the use of context longer than the training chunk. All 100 gate users have more than 2,048 reviews (86 have more than 8,192: full +2.00% / +2.40%, window −0.34% / +0.48%; 14 with 2,049–8,192: full +0.53% / +2.02%, window −0.50% / +0.51%).
- Consequences: **w80 (SM06) is out** for Q7. Pruning is likely cheaper in accuracy than feared (in-window capacity is not binding), but it still waits for P. The binding head for P is curve (F = +2.16%, the e6 log-slope projection is ≈ +1.8%), and curve is purely long-context. So the next GPU hours go to context length: Q6 (4,096 chunks), then a 4,096 continuation. SM16 (state carry-over) and SM11 (full-history teacher targets) gain value if 4,096 stops paying.

**Muon (user OK'd weighing a w64 Muon run).** Not primary. Evidence in this repo: F82 (`artifacts/experiments/f82-muon-curve`, main checkout, paused) is w32, 128-review samples, 10 test users, one seed. At 300 training users AdamW 0.3851 / 0.3850 vs Muon×1 0.3873 / 0.3785; the curves cross several times, which is noise level. Muon×3 never ran and the AdamW curve stops at 310 users. X03 (`"optimizer": "expertium-iter53"`, `scripts/expertium_options.py` `make_optimizer`) was CPU-checked only; it was never run on GPU. Upstream adopted Muon at w80 with a d=128 teacher, which is a different regime. An optimizer improves fit, and in-window fit is already at parity (curve) or within 0.5% (instant), so Muon can reach at most the small instant residual. It does not address the beyond-window part that dominates the gap. A Muon run is also a from-scratch run (≈ 5 h with gates, decay and scoring, against ≈ 3.5 h for the continuation). Muon is therefore **fallback 2** (below): it beats leaving the GPU idle, and it is gated after every epoch against the ctx2048 run's own epochs.

**Q5 / Q6 placement.** S05 code exists on branch `s05-batched` (worktree `C:/Users/jonat/workspace/wt/s05-batched`, e4a7a06 + 0294eea; CPU equivalence check passed, 11 users, 1.95x on CPU; correction by the main session 03:40: this re-assessment looked only at the small-model branch). Q5 therefore runs between Q4 and Q6 (≈ 07:35, 12 min, `--timing`) with the Launch command of the Q5 row (script from the s05-batched worktree), then `small_model_batch_compare.py --allow-missing-input-sha`; adopt if `passes` and speed-up ≥ 2. **Q6 runs right after Q4** (or after Q5), K = 6, as written. That gives ≈ 07:00–08:30, and Q7 starts at **≈ 08:00–08:45**, not at 07:15, because Q7 depends on Q6. Q6 pick rule, amended before the result because G shows curve is the head that context fixes: **pass** if dec6@4096 − dec6@2048 (464 users, paired) ≤ −0.001 curve with upper95 < 0, instant not worse by more than +0.0005, and ≤ 1.5x slower per review (the old rule asked for ≥ 0.001 on both heads).

**Q7 decision at ≈ 08:30.** Inputs: F6 = 464-user Δ% of the better of `$LBL-e6` / `$LBL-dec6`, S6 = e4 − e6 on the gate users (e6 gate rows are a subset of the 464-user scoring), and Q6.

| Condition (first match) | Q7 | Runs | GPU | Ends (start 08:30) |
|---|---|---|---:|---|
| F6 meets P (≤ +1.0% curve, ≤ +1.5% instant) | SM15 w48 pruning, `arch` stage, AdamW, chunk 4,096 if Q6 passed else 2,048 | as in the Q7 row of the queue table | ≈ 330 | ≈ 14:00 |
| **Q6 passed (primary, expected)** | **SM19: epochs 7–8 of the ctx2048 run at 4,096-review chunks x 8 (32,768 reviews/update, unchanged), from e6, in a new experiment `scale-w64kd-gate-cached-ctx4096-cont`; 100-user score after e7 and e8; 0.25-epoch decay at 4,096 from the last epoch; 464-user scoring** | `TIER=gate FROM=6 TO=8 CHUNK=4096 nohup bash $SM/runbook-2026-10-16b.sh cont > $SM/runbook-2026-10-16b-cont.log 2>&1 & disown` | ≈ 205–240 (e7 ≈ 70–80 with the cold 4,096 KD cache, e8 ≈ 57–65, scores 2 x 4, decay ≈ 25, scoring ≈ 30, smokes and queue ≈ 10) | ≈ 12:00–12:30 |
| Q6 failed, S6 ≥ 0.0005 / 0.0010 (half the expectation) | Fallback 1: epochs 7–8 at 2,048 (the same `cont` stage with `CHUNK=2048`; `extend` is not reusable for it) | `TIER=gate FROM=6 TO=8 CHUNK=2048 nohup bash $SM/runbook-2026-10-16b.sh cont > $SM/runbook-2026-10-16b-cont.log 2>&1 & disown` | ≈ 175 | ≈ 11:30 |
| Q6 failed, S6 flat | Fallback 2: **Muon w64** from scratch, ctx2048, `"optimizer": "expertium-iter53"`, 4 epochs, gate after every epoch against `$LBL-e<N>` (test A: must beat AdamW at the same epoch; stop at the first epoch where upper95 ≥ 0 on both heads), decay, 464-user scoring. Design work on SM16 starts in parallel (Opus, CPU) | `cd C:/Users/jonat/workspace/wt/small-model && SM=$PWD/artifacts/small-model && TIER=gate ARCH=w64 OPTIMIZER=expertium-iter53 CHUNK=2048 nohup bash $SM/runbook-2026-10-16c.sh arch > $SM/runbook-2026-10-16c-arch.log 2>&1 & disown` (written 2026-10-16 06:45, CPU only; dry run `DRYRUN=1 TIER=gate ARCH=w64 OPTIMIZER=expertium-iter53 CHUNK=2048 bash $SM/runbook-2026-10-16c.sh arch`, passed). New experiment `scale-w64kd-gate-cached-ctx2048-muon` (source run options + `"optimizer"`, same users/seed/order seed/KD teacher), labels `s-w64kd-gate-ctx2048-muon-e<N>`. Gate after every epoch on the 100 gate users vs `s-w64kd-gate-ctx2048-e<N>` (the AdamW e3 row does not exist: the stage scores it first, ≈ 4 min); test A reported, stop if upper95 ≥ 0 on both heads (`FORCE=1` continues). **Cold KD cache**: `"optimizer"` is part of the cache key (`batches-2b2206de733f`, not `batches-e66b1ec9cc5b`), so epoch 1 recomputes teacher targets (≈ 76 min like AdamW e1) and writes +6 GB; epochs 2+ hit. Untested: Muon on GPU (CPU only: grouping 309,888 Muon / 21,623 AdamW parameters, effective Muon LR 0.0025, optimizer-state round trip, decay prepare); smoke runs with a fresh Muon state | ≈ 300–345 (e1 ≈ 80–85, e2–4 ≈ 56–65 + unmeasured Muon step cost, decay ≈ 27, gates and scoring ≈ 55) | ≈ 13:30–14:15 |

**Q7 Ready / Launch (written 2026-10-16 ~03:35, CPU only).** New file `artifacts/small-model/runbook-2026-10-16b.sh`, stage `cont` (`runbook-2026-10-16.sh` and `runbook-next.sh` untouched; the stage only reads the source run). Env: `TIER` (required), `FROM` (6), `TO` (8), `CHUNK` (4096), `BATCH_CHUNKS` (32768/CHUNK), `RUN_TAG` (ctx2048), `FORCE=1`, `DRYRUN=1`. It refuses (exit 2) unless epoch FROM of the source is complete, no `runbook-2026-10-16.sh`/`runbook-next.sh` process is running, and heldout.jsonl holds the 100 gate users for `s-w64kd-gate-ctx2048-e<FROM>` and `-dec<FROM>` (the Q4 464-user scoring). It then copies into `scale-w64kd-gate-cached-ctx<CHUNK>-cont` (result/protocol stubs of epochs 1..FROM and the epoch-FROM master/optimizer only, no snapshots), sets chunk_reviews/batch_chunks/epochs, runs the CPU resume check, and enqueues: one training ticket per epoch (smoke first; 80 min for the first epoch at CHUNK > 2048, else 66), a 100-user score per epoch (`s-w64kd-gate-ctx<CHUNK>c-e<N>`), the stop rule after FROM+1 (e<FROM> reference rows from the 464-user scoring; a stop decays from e<FROM+1>, `FORCE=1` continues), the 0.25-epoch decay into `$SM/scale-w64kd-gate-cached-ctx<CHUNK>-cont-decay`, and 464-user scoring of `-e<last>` and `-dec<last>` with a resume ticket only if users remain. Relaunching resumes (existing experiment is not re-copied, scored labels are skipped).
- Launch primary (Q6 passed): `cd C:/Users/jonat/workspace/wt/small-model && SM=$PWD/artifacts/small-model && TIER=gate FROM=6 TO=8 CHUNK=4096 nohup bash $SM/runbook-2026-10-16b.sh cont > $SM/runbook-2026-10-16b-cont.log 2>&1 & disown`
- Launch fallback 1 (Q6 failed, S6 ≥ half): same with `CHUNK=2048` (BATCH_CHUNKS defaults to 16; new experiment `...-ctx2048-cont`, labels `s-w64kd-gate-ctx2048c-e<N>`).
- Dry run (no enqueue, scratch folders only): `DRYRUN=1 TIER=gate FROM=4 TO=6 bash $SM/runbook-2026-10-16b.sh cont`.
- Not written (skipped, not cheap): `arch` stage, `w48` preset, `--reference-prefix` for the gate, Muon grouping check. Fallback 2 and branch P therefore still need the Q7-P writer.

Decision rule after the primary (464 users, w8+s8 vs `fork/w8+s8`, F8 = the better of e8 and dec8@4096):
- F8 meets P: pruning starts. Q8 = w48 at 4,096 via `arch` (≈ 330 min) if it can start by 14:30; otherwise first in tonight's slot.
- Not P, and dec8@4096 beats dec6@4096 (Q6's label) by ≥ 0.001 curve (upper95 < 0): Q8 = epochs 9–10 at 4,096 (`cont FROM=8 TO=10`, ≈ 3.5 h, start ≤ 15:00).
- Otherwise: chunk length and epochs are exhausted at w64. Q8 = Muon fallback 2 only if it can start by 14:30; SM16/SM11 design (Opus, CPU) becomes the top accuracy item.
- Stop rule inside the run: after e7, if e6 − e7 ≤ 0 on both heads (gate users), skip e8 and decay from e7.

**To write before launch (one Sonnet medium writer, CPU only, start now, done by ≈ 08:00; in order):**
1. **`cont` stage** in `$SM/runbook-2026-10-16.sh` (never edit `runbook-next.sh` or the source run). Env: `TIER FROM TO CHUNK BATCH_CHUNKS RUN_TAG(=ctx2048) DRYRUN`. Step one: create `$WT/artifacts/accuracy/experiments/scale-w64kd-$TIER-cached-ctx${CHUNK}-cont` by copying the source `experiment.json`, `run/run.json` and `run/epoch-01..FROM` (≈ 65 MB each). In the copies set `chunk_reviews`=CHUNK, `batch_chunks`=BATCH_CHUNKS, `epochs`=TO, and a new name/note; the distillation cache path stays the same. Refuse (exit 2) unless epoch FROM is complete and the `extend` runbook has exited. CPU check: the epoch FROM+1 `protocol.json` resumes from `epoch-FROM` with `update_offset` = its `update_offset_end`, the same LR and optimizer fields, and the new chunk fields. The new option keys must be in the `gpu_r01_prefetch.py` allowlist. Then one ticket per epoch (smoke first; `--expected-minutes 80` for the first, 66 after), a 100-user score after each epoch (labels `s-w64kd-$TIER-ctx${CHUNK}c-e<N>`), the e7 stop rule above, a decay at CHUNK into `$SM/scale-w64kd-$TIER-cached-ctx${CHUNK}-cont-decay` (the Q6 `prepare` command with `--source-run <new>/run`), 464-user scoring of `-e<last>` and `-dec<last>` (`--time-limit 3000`, `--expected-minutes 30`), and a resume ticket only if users remain. Dry run with `DRYRUN=1 FROM=4 TO=6` and a stubbed queue, as for `extend`.
2. **`extend` generalisation**: decay folder and labels from `EPOCHS` (today hard-wired to e6), and the stop rule after `FROM+1`. Dry run `FROM=6 EPOCHS=8`.
3. **`arch` stage** (fallback 2 and P): `ARCH` (w64 | w48), `OPTIMIZER` (absent | `expertium-iter53`), `CHUNK`; gate after every epoch with `small_model_gate.py --reference-prefix s-w64kd-gate-ctx2048-e` (extend its self-test); add a `w48` preset in `expertium/variants.py` (`w80_config(heads=3, mult=2)`, CPU parameter count 195,949); CPU-check the Muon grouping on w64 (log the Muon/LoRA/AdamW parameter counts) and that `"optimizer"` passes the prefetch allowlist. If items 1–2 take the whole night, item 3 can be dropped: fallback 2 then becomes "idle + design", which is acceptable.

**Speed in parallel: yes, start SM18 now** (second Sonnet medium writer, CPU only, no GPU, no queue). The Anki fork runs the model in Rust (SM08), where the per-op overhead that dominated SM04 does not exist. A low-overhead step for both models is therefore the honest proxy for the real CPU saving. Brief: in `scripts/small_model_cpu_timing.py` add a `--runtime {eager,numpy,script}` axis for the fork and w64. `numpy` is the `small_model_numpy.py` step with preallocated buffers and no per-step allocation; `script` is `torch.jit.script` of the step. Exactness: max |Δp| ≤ 1e-5 vs eager on 3 users x 500 reviews. Smoke only (1 thread, light enough to run beside training). The timing ticket (`--timing`, ≈ 15 min) goes into tonight's quiet window, not during the day. Escalate to Opus only if neither runtime gives w64 ≥ 3x over the fork at batch 1.

### Result 2026-10-16 04:28: S06 long-context speed profile

Queue Q3 (ticket 000386, exit 0 after 6:57; the first attempt, ticket 000384, failed and was fixed in da60b34). w64, no KD, 160 users, `expertium_speed.py`; result `artifacts/small-model/speed-ctx-20261016/summary.md` + `speed.json` (log `artifacts/small-model/speed-ctx-20261016.log`).

| Case | Reviews/s | vs ctx128 | Peak MiB |
|---|---:|---:|---:|
| ctx128 x 256 (control) | 150,589 | 1.00x | 6,687 |
| ctx1024 x 32 | 124,733 | 0.83x (-17%) | 6,832 |
| ctx2048 x 16 | 101,429 | 0.67x (-33%) | 6,845 |
| ctx2048 x 32 | failed: CUDA OOM (12.72 GiB allowed, 12.01 allocated by PyTorch) | — | — |

- **Chunk length costs throughput**: -17% at 1,024 and -33% at 2,048 relative to 128 (fewer parallel sequences per step, the recurrence is the bottleneck). The real ctx2048 KD run did 103-115k reviews/s, consistent (its KD targets are cached).
- **b32 at 2048 does not fit**: the b16 case peaked at only 6,845 MiB, yet doubling the batch chunks ran out of memory at 12 GiB, so memory more than doubles with chunks x length (activations/backward are not linear in the b16 peak). Larger batches at 2,048 are not an option without lowering memory use.
- **ctx4096 x 8 (Q6/Q7) is not covered.** Cautious extrapolation from the trend (-17%, -33%): about -45% to -50%, i.e. roughly 75-85k reviews/s, to be confirmed by the real run's reviews/s.
- Decision: training is recurrence-bound at long chunks (S07 cache I/O is not the lever); larger batches are blocked by memory at 2,048.

### Result 2026-10-16 04:33: SM18 low-overhead CPU step

Ticket 000387, exit 0 after 4:44, `--timing`, 1 thread, `--impl eager,numpy` (merged from branch `sm18`, `scripts/small_model_fast_step.py`); result `artifacts/small-model/cpu-timing-sm18.json`. Per-review medians (ms), batch 1 / batch 256:

| Model | numpy f32 | eager f32 | eager int8 |
|---|---:|---:|---:|
| fork | 0.939 / 0.2479 | 3.954 / 0.1896 | 10.151 / 0.2011 |
| w64 | 0.509 / 0.0613 | 2.364 / 0.0721 | 5.733 / 0.0915 |
| w32 | 0.592 / 0.0502 | 2.661 / 0.0408 | 6.841 / 0.0659 |

- **At batch 1, w64 numpy is 7.8x faster than the eager fork (0.509 vs 3.954 ms) and 1.84x faster than the numpy fork (0.939 ms)**, so the CPU saving of the small model appears once per-op dispatch is removed (SM04 only showed 1.12x). w32 numpy (0.592) is slower than w64 numpy at batch 1.
- At batch 256 the numpy step is not better than eager except for w64 (0.0613 vs 0.0721 ms); for the fork (0.2479 vs 0.1896) and w32 (0.0502 vs 0.0408) eager is faster: the numpy step is meant for batch 1, not batching.
- Exactness vs eager: max |dp| <= 4.2e-7 on probabilities (sm18 commit, 1,490 real reviews of user 7). The s8 int8-state path is not covered.
- **Eager discrepancy vs SM04** (w64 batch 1: 2.364 here vs 3.452 ms in `cpu-timing.json`; w32 2.661 vs 4.289; fork 3.954 vs 3.854 unchanged). The diff of `scripts/small_model_cpu_timing.py` between e1ba148 and 3e97f25 only adds the numpy path, `--impl` and the exactness check; the eager timing code, settings (1,000 / 60 steps, 5 repeats, warmup 50) and torch/CPU are identical. Not a code change, so unexplained run-to-run variance (the fork matches, w64 and w32 were about 30-40% slower in SM04; possible causes are machine load or process warm-up state, e.g. here the numpy runs and exactness check execute before the eager runs in the same process). Use the SM18 numbers as the current measurement; the eager w64 batch-1 speedup over the eager fork is 1.67x here rather than 1.12x.

### Result 2026-10-16 06:35: extended run (epochs 5-6 + decay, 464 users)

Full-history scoring of `s-w64kd-gate-ctx2048-e6` and `-dec6` finished at 06:35 (ticket 000391, exit 0). `small_model_report.py --rmse-bins`, reference `fork/w8+s8`, review-weighted, 95% user-bootstrap CI; w8+s8 only. The tool's common-user set is the 451 users with ids >= 101 and a benchmark row in every label (same set as the 03:00 table; 464 rows exist per label).

| Label (w8+s8) | Curve LogLoss | Δ% [CI] | Instant LogLoss | Δ% [CI] | Curve RMSE(bins) | Δ% | Instant RMSE(bins) | Δ% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fork/w8+s8 | 0.29766 | +0.00% | 0.27216 | +0.00% | 0.03867 | +0.0% | 0.02084 | +0.0% |
| s-w64kd-gate-ctx2048-e4 | 0.30542 | +2.61% [+2.05, +3.33] | 0.27891 | +2.48% [+2.29, +2.69] | 0.04648 | +20.2% | 0.02678 | +28.5% |
| s-w64kd-gate-ctx2048-dec | 0.30408 | +2.16% [+1.62, +2.90] | 0.27804 | +2.16% [+1.97, +2.35] | 0.04441 | +14.8% | 0.02481 | +19.0% |
| s-w64kd-gate-ctx2048-e6 | 0.30435 | +2.25% [+1.70, +2.94] | 0.27799 | +2.14% [+1.96, +2.32] | 0.04560 | +17.9% | 0.02576 | +23.6% |
| s-w64kd-gate-ctx2048-dec6 | 0.30291 | +1.76% [+1.30, +2.37] | 0.27714 | +1.83% [+1.67, +1.98] | 0.04323 | +11.8% | 0.02418 | +16.0% |

Paired per-user differences (LogLoss, review-weighted, user bootstrap 95% CI, 4,000 resamples; cand − base, negative = better):

| Pair | Curve | Instant |
|---|---:|---:|
| dec6 − dec | −0.00118 [−0.00162, −0.00081] | −0.00086 [−0.00097, −0.00077] |
| e6 − e4 | −0.00102 [−0.00177, −0.00020] | −0.00093 [−0.00107, −0.00079] |
| dec6 − e6 (decay gain) | −0.00142 [−0.00191, −0.00103] | −0.00081 [−0.00091, −0.00072] |

- **F6** (better of e6/dec6 = `-dec6`): **+1.76% curve / +1.83% instant** (previous F +2.16% / +2.16%). Two more epochs plus decay bought −0.40 / −0.33 points; the gain over the 4-epoch result is significant on both heads.
- **S6** (definition from the 03:30 re-assessment: e4 − e6 on the 100 gate users, w8+s8; fork 0.28665 / 0.26105): e4 0.29359 / 0.26818, e6 0.29349 / 0.26736, so **S6 = 0.00010 curve / 0.00082 instant**. Thresholds 0.0005 / 0.0010 (both heads, as for S): both fail, so the gate users say **flat**. The e5 gate row (curve −0.00040, instant +0.00087 vs e4) already showed this. The 464-user paired e6 − e4 (−0.00102 / −0.00093) is larger on curve than the gate subset suggests (the 100-user subset has curve CI [−0.00061, +0.00074] for e4 − e6, i.e. consistent with zero); the Q7 table is defined on S6, so S6 is used.
- **P verdict: not met.** Curve +1.76% > +1.0%, instant +1.83% > +1.5%; RMSE(bins) +11.8% / +16.0% (instant still above the +15% bar). Row 1 of the Q7 table (pruning) does not apply.
- **If Q6 fails** (Q6 = cal-ctx4096-e6 vs dec6 at 2048 is still running and decides the primary): S6 is flat, so **fallback 2 (Muon w64 from scratch)** applies, not fallback 1 (epochs 7-8 at 2,048). Caveat for the decision: the 464-user improvement from e4 to e6 is real, only the gate subset is flat; if the user prefers, fallback 1 is the cheaper alternative (≈ 175 min) with an expected gain of roughly another −0.0005 curve / −0.0005 instant by the log-slope.

### Result 2026-10-16 07:20: Q5 S05 batched scoring (rerun)

Branch `s05-batched` (not merged). First run, ticket 000392 (06:35): crashed with an uncaught `torch.AcceleratorError` (OOM) after 56 of 100 users; the partial result was within tolerance at 2.05x (`artifacts/small-model/s05-gpu-check.json`). Fix df64fd9: robust OOM catch, split-in-half fallback, `--batch-length` cap of 800k padded reviews. Rerun, ticket 000394 (07:15, exit 0, 3:16 wall, job `artifacts/small-model/jobs/q5-rerun.sh`; `s05-gpu-check2.json`, `s05-check2.jsonl`): all 100 users scored, 0 fallbacks, passes the evaluation tolerance (max aggregate |Δ| 9e-8 vs the batch-1 rows).

| Scoring | Model time | Review-evals/s | Speed-up |
|---|---:|---:|---:|
| batch 1 (current) | 89.68 s | 77.7k | 1.00x |
| batched (s05-batched) | 114.98 s | 60.6k | 0.78x |

- **Verdict: not adopted** (rule: ≥ 2x). Batched scoring is slower overall, although the partial first run (56 users, the shorter ones scored first by bucket order) reached 2.05x.
- Reading: short users gain about 2x from batching, long users lose (padding waste and long sequences with a large batch dimension). Follow-up **S05b** (backlog): batch only users below a length threshold and keep long users at batch 1. `s05-batched` stays unmerged.

### Result 2026-10-16 07:33: Q6 ctx4096 decay probe

Decay at 4,096-review chunks from the e6 checkpoint of the ctx2048 run: ticket 000393 (exit 0, 38:35 wall against 27:03 for the 2048 decay; accuracy tickets, not a timing measurement — the ratio 1.43x is an indication only and includes a cold KD cache for the 4096 chunks); 464-user scoring ticket 000395 (exit 0, 14:55). Label `cal-ctx4096-e6`. `small_model_report.py --labels fork/w8+s8,s-w64kd-gate-ctx2048-dec6/w8+s8,cal-ctx4096-e6/w8+s8 --reference fork/w8+s8 --pair cal-ctx4096-e6/w8+s8=s-w64kd-gate-ctx2048-dec6/w8+s8 --rmse-bins` (w8+s8):

| Label | Curve LogLoss | Δ% [CI] | Instant LogLoss | Δ% [CI] | Curve RMSE(bins) Δ% | Instant RMSE(bins) Δ% |
|---|---:|---:|---:|---:|---:|---:|
| fork/w8+s8 | 0.29766 | +0.00% | 0.27216 | +0.00% | +0.0% | +0.0% |
| s-w64kd-gate-ctx2048-dec6 | 0.30291 | +1.76% [+1.30, +2.37] | 0.27714 | +1.83% [+1.67, +1.98] | +11.8% | +16.0% |
| cal-ctx4096-e6 | 0.30265 | +1.68% [+1.22, +2.30] | 0.27677 | +1.69% [+1.55, +1.84] | +10.7% | +12.9% |

Paired cand − base (cal-ctx4096-e6 − dec6): curve −0.00025, instant −0.00037. Per-user paired 95% bootstrap CI (451 users with id ≥ 101, review-weighted, 4,000 resamples): curve −0.00025 [−0.00036, −0.00012], instant −0.00036 [−0.00041, −0.00031]. The effect is real in sign but tiny.

- **Rule** (amended 03:30): pass needs curve ≤ −0.001 with upper95 < 0, instant ≤ +0.0005, and ≤ 1.5x slower. Curve −0.00025 is a quarter of the bar → **FAIL**. The speed clause was judged on accuracy-ticket wall time only (indication 1.43x, no `--timing` run), so it is unverified; the accuracy clause alone decides the FAIL.
- `cal-ctx4096-e6` is nevertheless now the best checkpoint (+1.68% / +1.69%; instant RMSE(bins) +12.9% meets the +15% bar), but the difference to dec6 is small. Longer chunks alone do not close the gap; Q7 = 4,096 continuation (SM19) is dropped.

### Decision 2026-10-16 07:34: Q7 = epochs 7-8 at 2,048, Q8 = Muon w64

Q6 failed and the 100-user S6 was flat, so the 03:30 table pointed to fallback 2 (Muon from scratch). The main session ran fallback 1 instead: epochs 7-8 at 2,048 (`runbook-2026-10-16b.sh cont`, `FROM=6 TO=8 CHUNK=2048`, experiment `scale-w64kd-gate-cached-ctx2048-cont`, labels `s-w64kd-gate-ctx2048c-e7`, `-e8`, `-dec8`, log `artifacts/small-model/runbook-2026-10-16b-cont.log`). Reason: the 464-user paired gains of epochs 5-6 were real (dec6 − dec −0.00118 [−0.00162, −0.00081] curve, −0.00086 instant; e6 − e4 −0.00102 curve) and only the 100-user subset looked flat; fallback 1 is also the cheaper option (≈ 175 min).

- **Launch**: the first launch at 07:34 failed within 1 s (the stage read `$EXP.partial` after renaming it; fixed in 3d9c15d); relaunched 07:35, ticket 000396 = epoch 7 (running; do not touch).
- **Expected**: e7 ≈ 08:40, e7 gate, e8 ≈ 09:45, decay ≈ 10:15, 464-user scoring ≈ 10:50.
- **Q8 = Muon w64** (`runbook-2026-10-16c.sh arch`, 70e94db, ≈ 300-345 min) kept for the afternoon: start ≈ 11:00, end ≈ 16:30-17:00, before the 20:00 limit; it must start before 15:00.
- **Cost note**: Muon epoch 1 recomputes the KD teacher targets (≈ 76 min, +6 GB) because `"optimizer"` is part of the KD cache key, although the teacher targets do not depend on the optimizer. Recorded as backlog SM20.

### Design 2026-10-16: training with carried state (SM21)

Opus, CPU only (branch `state-carry`, worktree `C:/Users/jonat/workspace/wt/state-carry`; nothing enqueued, no runbook or running script touched). User question 08:40: training cuts each history into independent windows that start from a zero state, while scoring and Anki run the whole history with the state carried. Can the student learn to use carried state? User direction 08:45: try saving the previous state and reusing it (option 3), and cost it in time and disk.

**What the code does today.** One update = one user's 32,768-review span, cut into 16 consecutive 2,048-review windows that run in parallel (`gpu_batch1.training_batches` → `prepare_job` → `gpu_r01_samples.samples_for`, fast columnar path). Users run in protocol order; spans run in offset order. Every entity sequence (card, deck, note, preset, user stream) starts from S = 0, and the token shift at its first row uses the row itself (`prepare` sets `time_shift_select` to 0). The fork teacher runs on the same windows, so the KD targets (alpha 0.9) are window-local too (cache key = batch options). **The fork itself was trained on independent sections of up to 32,768 reviews** (`rwkv/data_processing.py` `job`: `MAX_BATCH_SIZE = 65,536` rows // 2; one section per batch slot, `MAX_TRAIN_GLOBAL_LEN = 66,000`). It never carried state either; its windows are 16x longer than ours.

**Mechanism, measured on the data (CPU, `scripts/state_carry_sizes.py`, 1,000 of the 5,000 gate training users, every 5th by length, scaled to the 369.5M training reviews; `artifacts/small-model/state-carry-sizes.json`).** A card's previous review is often far back, so in training the card stream starts cold where at deployment it has the card's whole history:

| Window (reviews) | Labelled reviews whose card's previous review lies before the window start | Option 3 store per epoch, int8 (all / only entities seen before) | fp32 (all) |
|---:|---:|---:|---:|
| 2,048 (today) | **46.3%** | 739 / 576 GB (2.0 KB per review) | 2.45 TB |
| 4,096 (Q6) | 36.4% | 613 / 450 GB | 2.03 TB |
| 8,192 | 26.8% | 492 / 328 GB | 1.63 TB |
| 16,384 | 18.2% | 386 / 222 GB | 1.28 TB |
| 32,768 (fork) | 11.5% | 303 / 140 GB | 1.01 TB |

Burn-in reach on 2,048 windows (share still cold when the window is preceded by N unlabelled reviews): N = 2,048 → 26.8%, 4,096 → 19.5%, 8,192 → 12.7%, 32,768 → 3.6%. w64 state per entity (WKV 4 x 16 x 16 per time-mixer layer + 64-wide token-shift vectors; stripped channel mixers keep none): card 2,240 values (8.96 KB fp32, 2.69 KB int8 with fp16 row scales), note 1,152 (4.6 / 1.4 KB), deck 4,480, preset and user 3,264 each. An in-memory "latest state per entity" table for one user: median 57 MB, max 1.25 GB fp32 (≈ 0.4 GB int8).

This is consistent with the evidence so far: going from 2,048 to 4,096 chunks only moved the cold share from 46% to 36%, and Q6 gained −0.00025 / −0.00036. The fork's own protocol has 11.5% cold, and the w64 student's realised share of the fork's window → full-history gain rose with training context at a fixed state size (≈ 55–60% at ctx128, section 5.1; 77% / 86% at ctx2048, SM13).

**Options against the code.**

| | Where it plugs in | What breaks in the fast path | Cost vs today (≈ 103–115k reviews/s) | Effort | Verdict |
|---|---|---|---|---|---|
| (1) Burn-in: N unlabelled reviews before each window | `samples_for`: section `[s − N, s + 2,048)`, zero `has_label` (labels[..., 4]) on the prefix rows | Nothing structural; new option → new KD cache key (cold epoch). 16 x (2,048 + N) rows do not fit (2,048 x 32 OOM'd in S06), so batch_chunks drops to 8 = half the labelled reviews per update (LR schedule changes) | N = 2,048: ≈ 2.9x per labelled review (cold share 26.8%) | Sonnet 2–3 h | **Dominated by (5)**: 8,192 chunks reach the same 26.8% cold share at ≈ 2.3x, with labels on every row and no schedule change. Q6 already bounds it |
| (2) Carry between consecutive windows of a user (TBPTT, detached) | New: state table + `carry_forward`. One update holds 16 *consecutive* windows in parallel, so window k cannot read window k−1 of the same update | Needs the WKV kernel to take an initial state and return the final one; per-user consistent id/day encodings (today they are re-drawn per window by `prepare` / `batched_id_encodings` on the GPU) | ≈ 1.05–1.15x (estimate: state gather/scatter of ≈ 33k entity states per update, on the GPU) | Opus ≈ 1.5 days incl. kernel | **Recommended as "fresh carry"** (below), not as a batch-order redesign |
| (3) Stale states from the previous pass | Same model plumbing as (2); a store keyed by (user, window) | Disk: 576–739 GB per epoch int8 at 2,048 (2.45 TB fp32; 2.2 TB free) and ≈ 200 MB/s read + write at 100k reviews/s. States are one epoch stale, and their own initial states were stale, so the reachable history grows by about one window per epoch | Reusing the training pass's own end states: no extra forward. A separate refresh: student full-history replay at 50–67k reviews/s (measured, 464-user scorer) = 1.5–2 h per epoch | as (2) + cache I/O | **Only at lane boundaries** (see v2); as the main mechanism it costs too much disk for stale context |
| (4) SM11: teacher targets from a full-history fork replay instead of window-local predictions | Scorer exports per review (curve probability at review rows, 4 rating probabilities at query rows) for the training users; `Teacher` scatters them by `batch.label_review_th` instead of running or caching the teacher | The cache becomes keyed by (user, review), independent of chunk, architecture and optimizer (also fixes SM20 and the cold-cache epoch at every chunk change) | One replay: fork at 19.8k reviews/s (measured: 31.87M reviews of the 464 users in 1,609 s model time) → **5.2 h for the 5k gate users** (8.2 h for 8k); disk ≤ 10 B/review fp16 = **3.7 GB** (5.8 GB for 8k; fp32 7.4 GB) | Sonnet ≈ 3 h | **Needed with carry**: with carried state the window-local targets would pull the student back to cold-card behaviour. Alone it gives targets the student cannot see in a window |
| (5) Fork-length windows: chunk 8,192–32,768, same 32,768 reviews per update | Option already wired (`chunk_reviews`, validator ≤ 8,192; nvrtc limit 4 x chunk); `runbook-2026-10-16b.sh cont` computes `BATCH_CHUNKS = 32768 / CHUNK` | 8,192: no code. 16,384 / 32,768: one validator line, plus memory and kernel smoke | Recurrence-bound: update time ≈ 0.19 s + 34 µs x rows per window (fits S06: 128 → 165k vs 151k measured, 1,024 → 126k vs 125k, 2,048 → 100k vs 101k; Q6's 1.43x at 4,096). **8,192 ≈ 44k/s (2.3 h/epoch), 16,384 ≈ 25k/s (4.1 h), 32,768 ≈ 14k/s (7.5 h)** | none (8,192) | **Run today at 8,192** as the context test; 16k/32k are too slow for a gate-size run |

**Prototype (option 2/3 mechanism, CPU, `scripts/state_carry.py`, check `scripts/check_state_carry.py`).** `carry_forward` is the w64 interleaved forward with, per (stream, layer, entity row), the carried WKV state S, the time-mixer shift vector (used at every row before the entity's first non-skipped row in the window) and the channel-mixer shift vector, and it returns each row's final state. Sources: `FreshCarry` (latest state per entity in memory, filled by the training pass itself) and `StaleCache` (option 3: end-of-window states of the previous pass, int8 rows + fp16 shifts, looked up as "end of the entity's last window before this one"). `consistent_start` gives windows the input rows of the user's full-history sample (one id/day encoding draw per user). Entity keys come from the frame (int64): `RWKVSample.ids` is int32 and the loader's placeholder ids for missing notes (ID_PLACEHOLDER + card id) overflow it, so all such notes collide. The first check run found exactly this. Results (`artifacts/small-model/check-state-carry.json`, user 1340, 5,641 reviews, 3 windows; `check-state-carry-u8254.json`, 10k reviews, 5 windows; e6 checkpoint of the ctx2048 run, fp32, eval mode):
- A **carry disabled = today's batches**: the trainer's `prepare_job` batch (run options minus the packed-IPC transport) and the prototype's batch are identical (sha256 of every tensor). The trainer is not modified.
- B zero carried state = stock `forward_batch_variant`, **bit for bit**.
- C **exact carry = full history**: windows with fp32 state carried window to window reproduce the single full-history forward on every row, max |Δp| 5.4e-7 / 5.1e-7. Without carry the same windows differ by up to 0.64 (mean of the per-row max |Δp| on labelled rows 0.016 / 0.027). int8 carried state (s8, the deployment precision): max |Δp| 3.0e-3 / 2.3e-3, mean 5e-5 / 9e-5.
- D option 3 store, int8: 986 B and 1,714 B per review for these two users (the 1,000-user average is 2.0 KB, table above).
- Not covered: GPU kernels (nvrtc `RWKV7_WKV` and the row-tiled exact kernel have no initial-state input and no final-state output; the forward checkpoints its state every `_checkpoint_length` steps, so starting from S0 only changes the initial load, and backward is unchanged because S0 is detached), bf16, dropout, the fast columnar builder with consistent encodings, and the KD path.

**Recommended design (v1, tomorrow): fresh carry at update granularity + SM11 targets.** Keep the batch layout (one user, 16 x 2,048 per update), the order seeds and the shuffles. Keep a per-user GPU table of the latest state per (stream, layer, entity): int8 rows for card and note (the deployment s8), fp16 for deck, preset and user. All 16 windows read the table as of the update start. After the update (detached, no gradient) each entity's final state from its latest window is written back, and the table is cleared at the user boundary. Cold share falls from 46% to about the 32,768-span share, 11.5%. **v2: strided lanes.** Update u holds windows {u, u + U, …} of the user (U = updates of the user), so every window reads its predecessor fresh. Only the 16 lane starts per user remain, which take option 3 stale states (≤ 140 GB int8 per epoch, the 32,768 "warm" figure) or zero. Batches are still single-user, so the KD cache stays per batch. Work: (a) kernel S0/S_T for the generic nvrtc and row-tile kernels + GPU parity against `wkv_with_state` (Opus 3–4 h, one 5-min GPU ticket); (b) trainer: `state_carry` option in the `gpu_r01_prefetch` allowlist, int64 entity keys and consistent encodings in the columnar workers and in `batched_id_encodings` / `gpu_packed_features`, table lookup and commit around `catch_up.loss` (Opus 4–6 h, Sonnet 2 h checks: carry off = today's batches and gradients bitwise, the existing checks); (c) SM11 export + replay-target `Teacher` mode (Sonnet 3 h) and the 5.2 h replay ticket (can run tonight or early tomorrow). Experiment: continue from the best checkpoint (not from scratch: in-window fit is at parity, the new skill is using incoming state), 2 epochs at 2,048 x 16 with carry + SM11 targets, decay with carry, 464-user scoring. Paired against the non-carry epochs from the same start. Rule as Q6: curve ≤ −0.001 with upper95 < 0 and instant ≤ +0.0005. ≈ 2.5–3 h GPU.

**Context vs state size (w64 card state 3 KB int8 vs the fork's 15 KB).**
- Free evidence for context: the realised share of the fork's window → full-history gain rose from ≈ 55–60% (ctx128) to 77% / 86% (ctx2048) at a fixed state size, and the cold-card share is the mechanism above.
- Cheap discriminator (GPU ≈ 10 min, scoring only, after a run at 8,192): G at 8,192 = student@w8192 vs fork@w8192 on the 100 gate users, as SM13 did at 2,048. If in-window parity holds at 8,192 (where cards keep 4x longer histories), the 3 KB state carries those contexts as well as the fork's 15 KB, and the residual is context. If a gap opens despite training at 8,192, it is state size → w80 at 2,048 (≈ 7 h) before any carry work.
- Optional before deciding (≈ 15 min, scoring only): fork and `cal-ctx4096-e6` @w8192 and @w32768 on the gate users. This shows how much of the fork's full-history advantage is already present in 8k / 32k windows, i.e. the ceiling of option 5 versus carry.

**Today (slot ≈ 11:00–21:30).** Carry cannot be GPU-ready today (kernel plus trainer integration, ≈ 1.5 days), so the afternoon tests the premise with option 5, which needs no code:
- ≈ 10:50 (after the ctx2048 cont scoring), optional window probe, 15 min.
- ≈ 11:00 launch `cd C:/Users/jonat/workspace/wt/small-model && SM=$PWD/artifacts/small-model && TIER=gate FROM=6 TO=8 CHUNK=8192 nohup bash $SM/runbook-2026-10-16b.sh cont > $SM/runbook-2026-10-16b-cont8192.log 2>&1 & disown` (new experiment `scale-w64kd-gate-cached-ctx8192-cont`, labels `s-w64kd-gate-ctx8192c-e7/-e8/-dec8`, batch 4 x 8,192, from the same e6 as the running 2,048 continuation). Run `DRYRUN=1` first. Note that the runbook's queue estimates (80/66 min per epoch) are low: they only matter for GPU quiet windows.
- Estimate: e7 ≈ 2.6–3 h (cold 8,192 KD cache), e8 ≈ 2.3 h, gates 2 x 5 min, decay ≈ 40 min, 464-user scoring ≈ 30 min → **≈ 6.5–7 h, ends ≈ 17:30–18:00**. A 1.3x slower rate still ends before 21:30.
- Stop rule: if e7 runs below 30k reviews/s, stop after e7's gate and decay from e7 instead.
- Decision (pre-registered, 464 users, w8+s8, paired): dec8@8192 − dec8@2048 (`s-w64kd-gate-ctx2048c-dec8`) ≤ −0.001 curve with upper95 < 0 and instant ≤ +0.0005 → context pays; build carry v1 (it gives fork-length context at ≈ 2,048 speed). |Δ| < 0.0005 on both heads → window context is not the lever at 4x; run G@8192, then w80 if G opens.
- Muon w64 (Q8, 5.5 h) no longer fits after this run and targets in-window fit, which is at parity. It is postponed.

**Status 2026-10-18: v1 implemented, CPU-checked; GPU parity ticket and experiment pending** (Opus, branch `state-carry`, CPU only; handoff `HANDOFF-SM21.md` at the worktree root).
- Kernels (work item a): `rwkv/model/wkv_state.py` (`RWKV7_WKV_State`: forward from a detached S0, returns S_T; stock backward from checkpoints that start at S0). Generic NVRTC forward (`nvrtc.forward_with_state`, `STATE_REWRITES`) and the exact row tiles (`gpu_row_tiles.STATE_REWRITES`, also inside `row_tiled_backend`) are text rewrites compiled into separate modules: the stock text, compilation and launches are unchanged. `scripts/check_state_carry_kernels.py --cpu`: all 8 state variants compile with NVRTC, CPU reference from zeros = `reference_rwkv7` bitwise, chained halves bitwise (`artifacts/small-model/check-state-carry-kernels-cpu.json`). GPU parity part not yet run.
- Trainer (work item b): opt-in track option `"state_carry": true | {"precision": "deploy" | "fp32", "carry": false}` (in the `gpu_r01_prefetch` allowlist), `scripts/state_carry_train.py`. Per-user inputs: user-level day features, int64 entity ids, one id/day encoding draw per user (hash of training_seed + user). Per-user table per (stream, layer, entity): int8 S rows + fp16 scales for card/note, fp16 for deck/preset/user, fp16 shift vectors; all windows of an update read it as of the update start, each entity's latest window writes back; reset at the user's first update; searchsorted + gather/where only (no host sync, deterministic). The scripted forward hands over in `_interleaved_streams` (jit.ignore) for the student only. `"carry": false` = the paired control (same inputs, zero start).
- CPU check `scripts/check_state_carry_trainer.py` (users 1340 and 8254, e6 checkpoint): ALL OK (`artifacts/small-model/check-state-carry-trainer.json`). A option absent = the tree before SM21 (all batches, loss and every gradient sha256 equal); B empty table = stock forward bitwise (train mode, dropout); C windows' input rows = the full-history sample's rows bitwise; D exact carry through the trainer (one 2,048 window per update, fp32 table) = full-history forward, max |dp| 6e-7 (deploy table 1.4e-3 / 7.8e-4; zero start 0.65 / 0.64); E two windows per update = prototype `FreshCarry` bitwise; F table cleared at the user boundary; G scripted model: hand-over reaches the student only.
- Left: GPU parity ticket (`artifacts/small-model/jobs/sm21-gpu-parity.sh`, ~15 min), SM11 targets (parallel branch `sm11`), the experiment runbook (from e10 of `scale-w64kd-gate-cached-ctx32768-cont`, 2 epochs at 2,048 x 16 + decay, carry vs `"carry": false`, 464-user scoring).

**Status 2026-10-18 (later): v2 strided lanes implemented, CPU-checked; GPU check and experiment pending** (Opus, branch `sm21v2`, worktree `C:/Users/jonat/workspace/wt/sm21v2`, CPU only; handoff `HANDOFF-SM21.md` there). Why: in v1, 29.4% of card reviews and 21.2% of note reviews start from a stale state, and the user stream skips about 15k reviews at 92% of windows (`artifacts/small-model/feature-screen/sm21-stale-share.txt`). Arm A's gate result is consistent with that (curve = 32k training, instant +0.0011).
- Option `"state_carry": {"lanes": "strided"}`. v1 and the absent option are unchanged bit for bit.
  - Update u holds window u of each of the user's 16 lanes of consecutive windows.
  - Ragged rule: U = ceil(N / 16), the first 16 - m lanes hold U - 1 windows and the last m hold U. So every update keeps today's window and review count; offsets and seeds stay u x 32,768.
  - Table: per (lane, stream, layer, entity), for the pairs that occur. Lane starts start from zero; seeding them with stale states (option 3) is the alternative, not implemented.
- CPU check `scripts/check_state_carry_lanes.py`: ALL OK (`artifacts/small-model/check-state-carry-lanes.json`).
  - Option absent = pre-SM21, and v1 = 5a2455e (sha256 of every batch, loss and gradient).
  - The layout matches; first update = stock bitwise; user boundary; JIT.
  - fp32 v2 carry = full history: lane-0 rows max |dp| 8.3e-7, and with exactly seeded lane starts every row is at most 8.3e-7 (deploy 1.4e-3).
- Stale share in v2 (`sm21-stale-share-v2.txt`, the same 295 users):

  | Card reviews | v2 | v1 |
  |---|---:|---:|
  | stale | **0.0%** | 29.4% |
  | exact carry | **19.2%** | 8.9% |
  | zero at a lane cut | **19.1%** | — |

  - The lane cut splits into 9.4% in lane-start windows and 9.7% for entities first seen later in a lane.
  - Notes: stale 0 (v1 21.2%), exact 14.5%, cut 13.2%.
  - User stream: only the lane starts of lanes 1–15 enter with zero, which is 31% of windows. Every other window is exact.
  - Table: 2.5x v1's card slots and 2.7x note slots overall (max user 9.9x / 16x). Largest-user memory is measured by the GPU check.
- Pending:
  - GPU check `artifacts/small-model/jobs/sm21v2-gpu-check.sh`: P/X parity, T cost stock/v1/v2, M largest-user memory.
  - Experiment `artifacts/small-model/runbook-2026-10-18-sm21v2.sh`: `scale-w64kd-gate-sc-lanes-cont`, labels `s-w64kd-gate-sc-lanes-*`, the same recipe as arms A/B, DRYRUN OK. Decision: lanes - zero (arm B) by the Q6 rule; lanes - v1 carry (arm A) is reported.

### SM11 full-history replay targets: implementation and CPU check (2026-10-18; GPU ticket not launched)

Opus, CPU only (branch `sm11`, worktree `C:/Users/jonat/workspace/wt/sm11`; nothing enqueued). Implements option (4) of the SM21 design above, work item (c). Handoff: `HANDOFF-SM11.md` at the worktree root.

- **Export** (`scripts/sm11_replay_export.py export|status|verify`): the fork (float32 weights and state, the scorer's any-length recurrence; seed 2025 + user, as the 464-user scorer) replays each training user's whole history once. Per review it stores what `Teacher.targets` gives the loss: the curve of the ahead row labelled by that review (as a float16 logit of the clamped probability, which keeps 1 − p accurate near 1) and the 4 rating probabilities of its query row. One file per user, `users/<user>.npz`, float16 [reviews, 5], row = review_th − 1, NaN where the review has no such label; written to .tmp and renamed, so the export resumes per user. The manifest pins the teacher sha256, dataset path and revision, precision and pass settings, and a rerun with other settings is refused. `verify` recomputes sampled users on the CPU after the GPU run.
- Long users: histories above 600k reviews (the largest user the GPU scorer has replayed is 599,569) run in passes of 600k with 300k reviews of burn-in. That covers 38 of the 5k gate users (39.2M reviews) and adds 16.4M replayed reviews (+4.4%). A GPU out-of-memory error halves the pass and retries.
- **Teacher mode**: `"distillation": {"teacher": "<fork .pth>", "alpha": 0.9, "decay_alpha": 0.5, "kd_source": "replay", "replay": "<export dir>"}` (`ReplayTargets` in `scripts/expertium_options.py`). No teacher model is loaded. The labelled rows are looked up by `batch.label_review_th` and scattered into zero tensors exactly like a cache hit; alpha and decay_alpha are unchanged. It needs single-user batches (the batch key), and it raises on a missing user, a labelled row without a stored target, another teacher, or another dataset (`GPU_BATCH1_DATA_PATH`).
- Option absent = today's behaviour bit for bit. Cache digest: replay never uses a cache folder ("cache" together with replay is refused), and "distillation" is already TEACHER_BLIND, so online and cached runs keep their folders. No allowlist change: `distillation` is already in the `gpu_r01_prefetch` allowlist and the new keys are nested in it.
- Recipe differences beyond the context: replay targets come from a float32 teacher, while the window teacher ran in bfloat16 (the `teacher_dtype` default). They use one encoding draw per user instead of one per training batch. Expected speed: every epoch, the first included, runs without a teacher forward, as cached epochs do now. Per update the trainer adds one device-to-host copy of `label_review_th` and a CPU gather. Not measured on GPU.

CPU check (`scripts/check_sm11_replay_targets.py`, **ALL OK, 18 checks, 109 s**; `artifacts/small-model/check-sm11-replay-targets.json`; gate training users 793 and 2188, 3,463 and 3,367 reviews):
- a. Option absent compared with the code before SM11 (7fada06): online targets (plain and decaying), cache folder digest, cache miss and hit targets and the stored file, and the CatchUp loss with all 354 student gradients are all bitwise equal. `"kd_source": "online"` is bitwise equal too.
- b. Stored targets match the full-history forward through the scorer's prediction path (`evaluate_with_predictions`). Max |Δ| on the curve: 9.2e-5 and 9.9e-5; ratings: 2.44e-4. The float16 bounds are 1.2e-4 and 2.5e-4. The NaN pattern is exact.
- b. On a 1,024-review window starting at the user's first review, with the full-history encodings copied in, the online window teacher equals the full-history forward to 3.6e-7 (float32). The replay Teacher equals the window teacher within 9.4e-5 (curve) and 2.44e-4 (ratings), and it is zero at unlabelled rows.
- b. The replay Teacher runs end to end through `CatchUp.loss` (finite loss, 354 gradients, decay_alpha applied). All 7 refusals work.
- c. Resume works: a pass stopped after one user exits 3, and the rerun (through a loader worker, the GPU path's IPC) skips the finished user, discards a stale .tmp and gives byte-identical targets. Other settings on the same folder are refused. A multi-pass export covers every review exactly once.
- d. **10.27 B per review**: 10 B of payload plus about 910 B per file. That gives **3.70 GB for the 5k gate users** and 5.83 GB for the 8k final run.
- Informational (user 793, training batch 4 × 512): window-local targets against replay targets differ by a mean |Δ curve| of 0.0015 on the window that starts at the first review (encoding draw only). On the three cold-start windows the mean is 0.009–0.028, with a max of 0.32, and the mean rating |Δ| is 0.046–0.052. The encoding-draw noise floor (full history, seed 2025 against 2026) is a curve mean of 0.0009–0.0029, with a max of 0.015–0.025. So on cold windows the replay changes the targets about 10x more than the encoding noise does.

GPU ticket, not launched: `artifacts/small-model/runbook-sm11-replay.sh` (DRYRUN OK). It runs queue tickets with `--expected-minutes 95` and `--time-limit 5100` until `status` reports all 5,000 users, then runs `verify`. Estimate: 386M replayed reviews at the scorer's 18.4–19.8k reviews/s ≈ 5.4–5.8 h, so 4–5 tickets. Output: `C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-gate5k`. Not verified yet: GPU memory for 600k-review passes on training users, the GPU rate, the trainer on GPU with replay targets (a decay smoke), and any quality effect. Use: the SM21 v1 experiment (carry + SM11). An SM11-only screen (decay from the best epoch with replay targets, paired with the same decay with window targets) would separate the target effect from the carry effect.

### Feature screen 2026-10-17: candidate inputs and review type (CPU, user request)

Question (user): which other inputs could the dataset give, raw or engineered, and how would we test them? The user
also asked to drop **review type** (revlog `state`) where possible: the dataset comes from SM-2 users, and the fork's
curve visibly depends on review type, which breaks for users whose review types do not follow SM-2.

**What the models read today.** 24 per-review columns (`rwkv/data_processing.CARD_FEATURE_COLUMNS`: intervals in days
and seconds, cumulative intervals, interval mod 24 h, duration, rating one-hot, day of week, new cards / reviews today,
gaps, `scaled_state`, `is_query`) + card/note/deck/preset id encodings + day-offset encodings = 92 inputs. The w64 student
already **zeroes input 22 = `scaled_state`** in training and scoring (Expertium's recipe, `expertium/variants.py`), so
it never sees review type; the fork reads it on every history row. Query rows carry no state for either model. Unused
today: the deck hierarchy (`parent_id` is dropped on load), the card's review/lapse counts (`i` is computed, not fed),
and everything in the **-id dataset** (`data/anki-revlogs-10k`): raw Anki ids (creation ms) and `review_time`. Its rows
match the original by pandas index (ties within a day can be reordered; 3 of 34,149 rows of user 1901 sit one day
apart), but its intervals are re-derived from show times and its durations are not rounded, so it is a different build,
not just extra columns.

**Method** (`artifacts/small-model/feature-screen/`, CPU, 2 threads per job, scratch outputs only):
- per-review predictions (`small_model_heldout.py --device cpu --dump`) of fork w8+s8, the best student
  (`dec8-L32k` w8+s8) and the fork with its review-type input removed (`fork_nostate.py`: `features2card.0.weight[:, 22]
  = 0`, the same as a 0 input; the fork is a ScriptModule, no hooks), on 68 test users (every 6th of the 464 by size,
  <= 150k reviews, 1.68M benchmark reviews; the heaviest users are left out, so this sample's curve gap is +0.49%, not
  +1.15%);
- 19 candidate features per review (`features.py`): card prior reviews / lapses / pass streak / interval ratio,
  default-parameter FSRS-6 R, S, D, review type of the predicted and the previous review, user pass-rate and duration
  EWMAs, same-day reviews of the card, gap since any review and session position (from `review_time`), hours since the
  day rollover (shift fitted to the day_offset changes, 100% match on the checked user), days a card waited before its
  first review (raw card id), deck depth and note siblings (dropped: some users have no cards table);
- residual screen (`screen.py`): per model and head, logistic recalibration of the model's own logit (base) vs base +
  feature bins and their interaction with the logit, fitted on half of the users and scored on the other half (2-fold),
  95% interval by resampling users. A gain on both models = information neither uses; on the student only = capacity.

**Results.**
1. **No candidate adds information to either model, on either head** (all within about ±0.15% of log loss, mostly
   slightly negative = cross-user noise). The only large gain, user recent pass rate on the curve (+1.1% fork, +0.9%
   student), is a **protocol leak**: the curve is forecast at the card's previous review, and the same feature taken as of
   that review gives −0.3% (`atprev.py`). Instant: no feature helps either model (≤ +0.04%).
2. **The fork leans hard on review type.** Without it the fork's curve log loss rises **+8.4%** and instant **+3.3%**
   (student, which never had it: +0.49% / +0.94% on this sample). Zeroing an input a model was trained with overstates
   what a model trained without it would lose, but the dependence is real; on the de-stated fork, review type and card
   history features recover +1.1–1.3% (they do nothing for the normal fork).
3. **The student's curve gap is entirely on learning-step reviews** (`bytype.txt`):

| review type of the predicted review | share | curve: student vs fork | share of the curve gap | fork without review type | instant: student vs fork |
|---|---|---|---|---|---|
| learning | 35.2% | +1.16% | 72% | +21.9% | +1.28% |
| review | 52.5% | **−0.03%** | −3% | +2.2% | +0.74% |
| relearning | 11.9% | +1.01% | 28% | +3.4% | +0.86% |
| same-day review of the card: yes / no | 36.4% / 63.6% | +1.17% / +0.14% | 81% / 19% | +16.7% / +4.2% | +1.15% / +0.82% |

   On review-state reviews (the ones the scheduler's interval choice depends on) the student already matches the fork's
   curve on this sample. The fork's lead is on learning and relearning steps, where it reads the SM-2 review type and
   where the student has no scheduler-independent substitute (same-day count, streak, FSRS state: no gain in the screen).
4. **Instant** is behind on every slice (+0.74% to +1.28%), with no feature explaining it: capacity or fit, as the
   per-user view already suggested.
5. **Deck / preset / collection counts** (user follow-up, `deckfeat.py`, `screen-deck.txt`): deck size as of the review
   (cards created by then, from raw card ids) and in the export snapshot, cards introduced so far per deck / preset /
   collection, deck progress (introduced / created), reviews so far per deck / preset, each also as of the card's previous
   review: all between −0.42% and +0.01% for both models on both heads, no signal.
6. **Deck and preset ids are not used by value.** `rwkv.prepare_batch.add_encodings` (and the batched GPU path
   `scripts/gpu_id_encodings.py`) give every distinct card / note / deck / preset id a fresh random code per sequence
   (8-12 dims in {−1.5, −0.5, 0.5, 1.5}), and the day-offset encodings get a random phase. The code says only "same
   deck as that row"; deck difficulty is learned per user through the deck stream's recurrent state, never as "deck 2 is
   hard". (Collisions are possible but rare: 4^8 = 65,536 deck codes, 4^12 card codes.)

**Plan** (backlog SM24–SM28; GPU items queue after the overnight Q11 chain):
- **SM24 (decision support, ~35 min GPU):** the same breakdown on all 464 users: GPU scoring with `--dump` of fork and
  `dec8-L32k` (and the Q11 models), per review type and same-day. If it holds, the user decides whether bar P is also
  reported on review-state (or not-same-day) reviews, where review type is not needed, next to the official all-review
  bar.
- **SM25 (review-type robustness, CPU):** a realistic non-SM-2 perturbation instead of zeroing: score the fork and the
  student with history review types rewritten as an FSRS / no-learning-steps scheduler would label them (same-day
  reviews of graduated cards as "review"). The student is invariant by construction; the fork's change measures the risk
  the user saw in the fork. ~1 h CPU on the 68 users.
- **SM26 (instant capacity, ~1.2 h GPU):** instant-weighted final decay: 0.25 epoch from ctx8192 e8 (or Q11 e9) with the
  instant loss weight doubled, paired against the matching decay; then, if it moves, a wider instant head only.
- **SM27 (long-history summary inputs for windowed training, ~0.5 day code + ~3.2 h GPU):** the screen measures
  full-history inference only and finds nothing there, but in 2,048-review training windows the model starts cold and
  cannot see a card's earlier history. Card prior reviews / lapses, FSRS-6 S/D/R (scheduler-independent, available in
  Anki's Rust FSRS) and the long user pass rate as of the previous review, computed over the full history, would give
  windows that information. Test: append them after the 92 inputs with zero-initialised weights (the teacher keeps its
  92; add the option to `TEACHER_BLIND` so the KD cache is reused), continue ctx2048 e6 for 2 epochs at 2,048 + decay,
  paired against the existing `ctx2048c-dec8` (+1.66% / +1.66%). Success = moves toward `ctx8192c-dec8` (+1.24% /
  +1.21%) at 2,048 speed. Prior is modest (the screen is negative); it competes with SM21 for the same cold-start gap
  and would be cheaper to train and deploy.
- **SM28 (not planned):** session/clock features (`review_time`), deck depth, card creation age: no signal in the screen
  and they need the -id build at training and the review clock at inference. Revisit only if SM24 points there.

#### Follow-up 2026-10-17: SM25 (fork under non-SM-2 review types) and per-user rep-number statistics (CPU, 68 users)

**SM25** (`feature-screen/sm25.py`, `sm25_table.py`, `sm25.txt`): the fork scored with its history review types rewritten
as another scheduler would label them (`base_transform_state` patched; the predicted review carries 0 as before).
*No steps* (FSRS without learning/relearning steps): a card's first review stays new, every later review is "review"
(36–43% of history rows change). *Day steps* (steps shorter than a day): a learning/relearning review after a gap of
≥ 1 day becomes "review" (7–17% change). Log loss vs the normal fork, by review type of the predicted review:

| model | curve all | learning | review | relearning | instant all | learning | review | relearning |
|---|---|---|---|---|---|---|---|---|
| student (never reads review type) | +0.49% | +1.16% | −0.03% | +1.01% | +0.93% | +1.28% | +0.74% | +0.86% |
| fork, review type zeroed | +8.39% | +21.87% | +2.24% | +3.36% | +3.31% | +5.84% | +2.09% | +2.63% |
| fork, no-steps labels | **+0.43%** | **+1.17%** | −0.04% | +0.69% | +0.55% | +1.08% | +0.17% | +0.85% |
| fork, day-steps labels | +0.20% | +0.75% | −0.09% | +0.19% | +0.20% | +0.32% | +0.09% | +0.35% |

- A plausible relabelling costs the fork far less than zeroing (+0.43% vs +8.39% curve): zeroing is an
  out-of-distribution input, a consistent other labelling is not.
- With no-steps labels the fork's curve lead over the student disappears (all +0.43% vs +0.49%, learning +1.17% vs
  +1.16%): **the fork's curve advantage on learning steps is the SM-2 review label itself**. For a user whose review
  types do not follow SM-2 learning steps, the student's curve is on a par with the fork. Instant keeps a real gap
  (student +0.93% vs relabelled fork +0.55%, review-state +0.74% vs +0.17%).

**Per-user rep-number statistics** (user idea: with SM-2 a card's learning steps are its reviews 2, 3 (4) and
relearning steps the first reviews after an Again, so a user's own pass rate at "review k of a card" could stand in for
the review type; `feature-screen/repfeat.py`, `repscreen.py`, `repscreen.txt`). The data supports the premise: 88% of
second reviews and 35% of fourth reviews are learning-type, and the per-user pass rate at review 2 ranges 0.31–1.00
(median 0.91). Features, causal (as of the card's previous review for the curve, of the review itself for instant):
the user's smoothed pass rate minus their overall rate, keyed by review number k, by (ever had an interval ≥ 1 day,
reviews since the last Again), by (preset, k) and by (deck, k), and the model's own running residual (y − p) on the same
keys. Screen as above (logistic recalibration + feature bins × logit, 2-fold by user), gain in % of the fork's log loss
per slice: **no feature helps either model on any slice** (best single: student residual by phase on learning reviews
+0.14% curve, interval crosses 0; combined sets −0.1 to −0.3%). The student's running residual by review number has
mean ≈ 0 (sd 0.02): it already calibrates per user and per review number from the history it reads.

**A stacked corrector** (`feature-screen/stack.py`, `stack.txt`; gradient boosting on the model's logit + card-history
features, fitted on other users, 4 folds by user) makes unseen users worse (student curve +0.53% → +1.06%; a first
version with user-level features +2.0%). Adding review type to it recovers ~0.6 points on learning reviews, again
locating the remaining learning-step information in the SM-2 label, not in counts.

**Consequence.** The student's learning/relearning curve gap is what the SM-2 review label carries and what a
scheduler-independent model should not rely on; rep-number statistics, per-user or per-preset, do not recover it and a
post-hoc corrector does not transfer across users. The review-state slice and a relabelled-fork reference are the
fair comparisons for non-SM-2 users; instant (+0.4 to +0.6 points beyond the relabelled fork) is the real model gap.

### Results 2026-10-16/17: longer windows (decay and training), per-user view, shared history

Student vs fork w8+s8 (LogLoss gap, 464 users). Source: `artifacts/small-model/HANDOFF.md` (2026-10-16 18:30 to 2026-10-17 09:30).

| model | curve | instant | paired delta (curve / instant) |
|---|---|---|---|
| ctx2048c dec8 | +1.66% | +1.66% | |
| ctx8192c dec8 | +1.24% | +1.21% | |
| dec8-L16k (8k e8, decay at 16k) | +1.17% | +1.16% | vs 8k decay: -0.00020 / -0.00014 |
| dec8-L32k (8k e8, decay at 32k) | +1.15% | +1.12% | vs 8k decay: -0.00028 / -0.00022; vs L16k: -0.00008 / -0.00009 |
| ctx16384c e9 (undecayed) | +2.93% | +2.27% | |
| ctx16384c dec9 (e9 at 16k, decay at 16k) | +1.03% | +1.04% | vs dec8-L16k: -0.00042 (-0.00076..-0.00017) / -0.00032 (-0.00036..-0.00027) |
| **ctx16384c dec9-L32k** | **+1.01%** | **+1.01%** | vs dec9: -0.00007 (-0.00017..+0.00003) / -0.00009 (-0.00011..-0.00007) |

- Decay length alone has diminishing returns: 8k -> 16k ~0.06 points, 16k -> 32k ~0.03.
- Training one epoch at 16k pays ~0.13 points per head on top of a 16k decay (dec9 vs dec8-L16k, 300/464 curve and 385/464 instant users better): the SM22 open point is answered, training at 16k beats decay-only.
- Undecayed checkpoints are noisy (e9: +2.93% / +2.27%; 100-gate-user score +2.68% / +2.51%, no divergence in updates): constant-LR end-of-epoch noise, judge only after decay.
- Bar P (<= +1.0% on both heads) missed by ~0.01 points at dec9-L32k.
- Epoch 10 at 32k (Q12) is running as ticket 000420 (epoch ~6 h, results ~17:00); paired test is dec10 vs dec9-L32k.

**Per-user view** (plain mean of per-user LogLoss; tertiles by user review count <= 11,220 / <= 37,940 / <= 655,000):

| model | curve per-review | curve per-user | curve light/mid/heavy | instant per-review | instant per-user | instant light/mid/heavy |
|---|---|---|---|---|---|---|
| ctx2048c dec8 | +1.66% | +1.19% | +0.89/+1.21/+1.46 | +1.66% | +1.40% | +1.30/+1.36/+1.54 |
| ctx8192c dec8 | +1.24% | +0.99% | +0.80/+1.08/+1.09 | +1.21% | +1.11% | +1.13/+1.10/+1.10 |
| dec8-L16k | +1.17% | +0.97% | +0.82/+1.06/+1.03 | +1.16% | +1.09% | +1.13/+1.08/+1.06 |
| dec8-L32k | +1.15% | +0.97% | +0.82/+1.05/+1.01 | +1.12% | +1.08% | +1.13/+1.08/+1.04 |

- The curve gap is concentrated on heavy users, and longer windows close mostly that part (heavy +1.46 -> +1.01, light +0.89 -> +0.82).
- Instant is a uniform ~+1.1% on every tertile and worse than fork on ~449/464 users: a systematic gap, not a user-type effect, so more context will not fix it (capacity, calibration, KD instead).

**Shared history** (imported decks; relation graph DQ02/DQ05, common blocks >= 20 reviews; L32k model vs fork, w8+s8): 75/464 test users share some history with fork-training users (21 share >= 10%, 6 >= 50%); 72/464 share some with the student's training users.

| group | users | review share | curve per-review | curve per-user | instant per-review | instant per-user |
|---|---|---|---|---|---|---|
| no overlap | 389 | 74.5% | +1.15% | +0.93% | +1.17% | +1.07% |
| some overlap | 75 | 25.5% | +1.13% | +1.16% | +1.00% | +1.11% |
| >= 10% overlap | 21 | 2.5% | +1.39% | +1.80% | +1.05% | +1.39% |

- The fork leads most on heavy-overlap users (consistent with training on copies of those histories), but they are 2.5% of reviews; on the 389 clean users the gap is the same. Shared history does not explain the bar P miss. Open (user's decision): score bar P on clean users too.

### State quantization test (SM32, post-training)

2026-10-17, CPU only. How much LogLoss the w64 student (`scale-w64kd-gate-cached-ctx16384-cont-decay-L32k`, E89 master) loses when its card/note state is vector-quantized the way Expertium compresses his d=64 state, with no retraining (PTQ). Code: `scripts/state_vq.py` (codec, hooks, recipes, bit accounting), `scripts/state_vq_fit.py` (corpus, k-means, training-user screen), `scripts/state_vq_report.py` (table); variants `+vq<card bits>x<note bits>` in `scripts/small_model_heldout.py`. Outputs: `artifacts/small-model/state-vq/` (`summary.md`, jsonl rows, `tune-summary.txt`).

**Scheme** (reimplemented from his `rust/rwkv-infer/src/model.rs` at 1f2e6b0; same placement as `+s8`: the answer uses the float update, the stored state is replaced by its reconstruction before the next review of that card/note). Per head (16 x 16 WKV matrix): rank-r truncation via the Gram eigendecomposition, sign canonicalisation, joint (u, v) 32-dim unit-direction code by residual VQ (stages of 10-bit indices), one norm sqrt(sigma) per component in log2 at n bits. Token-shift vectors (2 time + 1 channel mixer for the card, 1 + 1 for the note): unit direction in m sub-vectors of 12-bit codes (residual stages), norm in log2. Deck/preset/user state float. Bits per entity are counted as he counts them: his deployed recipe (rank 1, 4 stages, 1-bit norms, shift m2b12 + 1-bit norm) is 403 / 214 bits, one stage 163 / 94 (both reproduced exactly by the accounting). Global codebooks (not counted per entity): k-means on 150k WKV heads and 150k shift vectors sampled from 50 training users of the student (protocol.json rows; no evaluation user).

**Deviations from his scheme**: k-means PTQ codebooks (his deployed catalogs are learned in QAT); a separate catalog per rank component (his joint catalog is shared); norm ranges fitted on the corpus (energy-weighted log2 MSE); exact nearest-centroid search (no warm start); the 400 / 800 budget recipes below are ours (picked on 8 other training users, model-side loss): **6-bit norms** (1-bit norms were most of the PTQ loss), more WKV stages, 2 shift stages, and for the 828-bit note rank 2 (8 + 4 stages) with shift m4 x 3 stages.

**Result** (34 users = every 2nd of the 68 feature-screen users in the stratified order, 1.21M reviews; fork w8+s8 scored fresh on the same users):

| Label | Card bits | Note bits | State, largest user | Curve LogLoss | Δ% vs fork w8+s8 | Δ vs st w8+s8 | Instant LogLoss | Δ% vs fork w8+s8 | Δ vs st w8+s8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fork w8+s8 | 122,880 | 81,920 | 4.58 GB | 0.32132 | ref | −0.00052 | 0.28497 | ref | −0.00238 |
| student fp32 | 71,680 | 36,864 | 2.57 GB | 0.32184 | +0.16% [−0.23, +0.56] | +0.00001 | 0.28733 | +0.83% [+0.64, +1.12] | −0.00001 |
| student w8+s8 | 26,624 | 14,336 | 959 MB | 0.32183 | +0.16% [−0.24, +0.56] | 0 | 0.28734 | +0.83% [+0.64, +1.12] | 0 |
| w8 + his 1 stage | 163 | 94 | 5.9 MB | 0.32838 | +2.20% [+1.50, +2.91] | +0.00655 | 0.29738 | +4.36% [+3.55, +5.26] | +0.01003 |
| w8 + his deployed | 403 | 214 | 14.5 MB | 0.32702 | +1.78% [+1.11, +2.43] | +0.00519 | 0.29556 | +3.72% [+3.05, +4.45] | +0.00822 |
| w8 + ~400/400 | 378 | 412 | 15.6 MB | 0.32303 | +0.53% [+0.10, +0.97] | +0.00120 | 0.28962 | +1.63% [+1.31, +2.09] | +0.00228 |
| w8 + ~800/400 | 770 | 412 | 27.7 MB | 0.32255 | +0.38% [−0.05, +0.81] | +0.00071 | 0.28892 | +1.39% [+1.11, +1.77] | +0.00157 |
| w8 + ~800/800 | 770 | 828 | 31.7 MB | 0.32251 | +0.37% [−0.04, +0.78] | +0.00067 | 0.28870 | +1.31% [+1.03, +1.71] | +0.00135 |

State = card + note state of the largest dataset user (246,801 cards, 76,887 notes); the student's s8 keeps shift vectors f32 (fork s8 rounds them to f16). [ ] = 95% user-bootstrap CI.

- At his own recipe the PTQ cost (+0.0052 curve / +0.0082 instant) is ~3x his QAT tax (+0.0018 / +0.0028). Almost all of it is the 1-bit norms: on the training-user screen his WKV codec alone (float shift) cost +0.0059 / +0.0071, his whole codec +0.0060 / +0.0078, the whole codec with 4-bit norms (436 / 232 bits) +0.0012 / +0.0025, rank-1 truncation alone +0.0003 / +0.0009, his shift codec alone −0.0001 / +0.0008.
- Same ~400-bit card, norms at 6 bits and a 412-bit note: the cost drops to +0.0012 / +0.0023 (14.5 → 15.6 MB). Doubling the card (800/400) buys −0.0005 / −0.0007, doubling the note again (800/800) only −0.0000 / −0.0002. Instant stays the expensive head, as in his ledger.
- A near-lossless upper point (1,506 / 828 bits, rank 2 everywhere, shift m4 x 3) cost +0.0001 / +0.0006 on the 8 screen users (not scored here, time cap).
- Caveats: PTQ overstates the cost vs QAT (his catalogs and the model adapt to the quantizer); Expertium found that PTQ rankings of some quantizer settings did not transfer to QAT, so the allocation picked here (6-bit norms, shift stages, rank 2) needs a QAT run before it is trusted. 34 users only; the 400/800 recipes were tuned on 8 training users (small screen, model-side loss). Codebooks (`codebooks.npz`, 14 MB) are not committed: `state_vq_fit.py collect --users 50 --shard i/2` then `fit --shift-stages 3` and `refit-norms` regenerate them (seeded).
- Checks: `+vqexact` (rank-16 factorization, no codebook, exact norms, the same hooks) reproduces the fp32 student to 3e-8 in loss and to the 6th decimal in benchmark LogLoss (3 users); bit accounting printed per entity type at load.
- Found on the way: before this change, `+s8` was silently a no-op for expertium models on CPU (RWKV7TimeMixer calls `reference_rwkv7`, not the patched `RWKV7_WKV`, for CPU tensors; checked: w8 and w8+s8 identical on user 4676). GPU rows (the 464-user bar) are unaffected; CPU `st/w8+s8` rows from before 2026-10-17 (e.g. the feature screen) are w8 rows. The int8 state itself costs ~0 on these users (fp32 vs w8+s8 above).
- Runtime: corpus 10 min (2 processes), k-means 25 min, screen 30 min, fork 47 min and student scoring ~80 min wall (3 processes in parallel; VQ variants 1.5–2.5x the plain CPU time). About 3 h wall in total, including setup.

### State quantization, QAT in the final decay (SM32b)

2026-10-17, CPU only: implemented and CPU-checked; the GPU run is prepared (runbook below), not launched. Evening: made fast enough to run (S09, "Speed" below): the run now uses the Gram-squaring codec `vq378x412p` on fused CUDA kernels, projected ~75-80 min instead of 257 min (420 queued); a ~5-min GPU micro-benchmark measures it before the launch. Expertium's advice: keep the 10-epoch checkpoint and redo only the decay epoch with the state compression in the loop (his decay-only QAT, iter39, removed the quantization cost; QAT from scratch, iter40, ended +0.0118 instant worse).

**What runs.** The decay of `s-w64kd-gate-ctx32768c-dec10` (464 users: 0.30096 curve / 0.27471 instant; fork 0.29766 / 0.27216) redone with the codec in the loop and nothing else changed: epoch-10 master and AdamW state of `scale-w64kd-gate-cached-ctx32768-cont` (read from the small-model worktree), `expertium_decay.py prepare --epoch 10 --fraction 0.25 --shape one_minus_sin --seed 1` = the same 1,192 users and 92,400,532 reviews in the same order, 32,768-review chunks, 1 chunk per update (3,465 updates), LR 7e-4 → 0, KD decay alpha 0.5 from the same teacher cache (`batches-458c9a61e4c9`). One extra track option: `"state_vq": {"recipe": "vq378x412p", "codebooks": "<state-vq>/artifacts/small-model/state-vq/codebooks.npz", "sha256": "eab6db58…", "factor_dtype": "float32", "impl": "cuda"}` (S09; a plain `"vq378x412"` string = the original eigh codec on the torch loop, same codebook file). Output: `artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay-qat378x412p` in the state-vq worktree (the run executes this worktree's code).

**Where the fake-quant sits** (the CPU scorer's `+vq` placement, so training, scoring and deployment use one codec):
- Card stream (2 WKV layers) and note stream (1 layer), 4 heads of 16 x 16: the answer to a review uses the float update; the stored state that the entity's next review starts from is the codec's reconstruction (rank-1 direction of the Gram matrix, sign canon, joint (u, v) residual VQ, 6-bit log2 norm; the direction by `linalg.eigh` in `vq378x412`, by Gram squaring in `vq378x412p`, same picks, see Speed). Deck, preset and user state stay float on the CUDA kernel.
- Token-shift vectors (card: 2 time + 1 channel mixer, note: 1 + 1): a review that reads its entity's previous review reads the codec's reconstruction (product code + 6-bit norm); an entity's first review reads its own input, as before.
- Straight-through estimator: forward value = the reconstruction exactly, gradient = identity through the codec. Codebooks get no gradient.
- Code: `expertium/model/state_vq_hooks.py` (`VQStateWKV`, `VQStateWKVKernel`, `ste_backward`, `vq_wkv`, `vq_shift`; `@torch.jit.ignore`, called from the scripted mixers when `state_vq_slot >= 0`), `scripts/state_vq.py` (`install_training`, device-aware `StateCodec`, `factorize_sq`, `attach_kernel`), `scripts/state_vq_cuda.py` + `scripts/kernels/state_vq_wkv.cu` (fused kernels), one block in `scripts/gpu_r01_training.py`. The card/note recurrence becomes a float32 per-step loop over each split's length (the per-entity sequences of a chunk: ~144 card and ~844 note steps per update on average, sampled over 40 of the decay's updates). Its backward is explicit (the plain WKV backward along the quantized trajectory; equal to autograd through the straight-through loop to 4e-16), so it keeps one 16 x 16 state per head and step (~1 GB at 32k chunks) instead of ~10 recorded intermediates.

**Codebooks: frozen (SM32 k-means), not learned.** (1) Train = score = deploy: the 34-user scorer row loads the same file (sha256 pinned in the option, written into the training result and into every scorer row), so no export step can drift. (2) The warm start stays dec10's: learned catalogs add an optimizer group the epoch-10 AdamW state does not have, and a new checkpoint format; with frozen codebooks the only difference from dec10 is the codec in the forward, a 1:1 comparison. (3) In a 0.25-epoch decay with the LR going to 0, the weights adapting to a fixed quantizer is the larger effect (his iter39: pure quantization cost +0.00002 instant after QAT vs +0.0036 for PTQ). Learned catalogs (his gradient into the picked centroids, or EMA k-means) and a CPU re-fit of the k-means on the QAT'd model's states are the follow-ups if QAT closes less than half of the PTQ gap. `factor_dtype` "float32" is a speed fallback (faster GPU eigendecomposition; picks can differ from the float64 scorer on near-ties); the run uses float64.

**Default off.** Without the option no mixer gets a slot and the scripted graphs only gain an untaken branch. `state_vq` is teacher-blind for the KD cache key (same folder as dec10, so cache hits) and is in the `gpu_r01_prefetch` allowlist.

**CPU check** (`scripts/check_state_vq_qat.py all`, `artifacts/small-model/state-vq/qat/check.json`): ALL OK.
- (a) Option absent: 3 decay updates (bf16 student + fp32 master, epoch-10 AdamW state, decay LR, dropout on, 1,024-review batches of the decay's first user, plain loss) are bitwise equal to a clean export of HEAD 4e07e6b in losses, gradient norms, gradient hashes and final weights, scripted (graph optimizer off, as in the trainer) and eager. The KD-cache digest of dec10's options is unchanged with `state_vq` added.
- (b) Option on, scripted training model vs the CPU scorer's eager `+vq378x412` path, w8 weights of epoch 10, full history of the 3 smallest users34 users (3,716 / 5,224 / 5,773 reviews, 28k predictions): max |Δp| 3.0e-7, mean 1–2e-8, loss/metric |Δ| ≤ 6e-8.
- (c) `VQStateWKV` backward = autograd through the straight-through reference loop (float64, card and note codecs): forward bitwise, gradients to 4e-16 relative. Shift hook: gradient passed unchanged, stored positions coded, first reviews unchanged. 3 QAT updates: finite, non-zero card/note gradients, losses differ from the option-off run (0.7976 / 0.6177 / 0.8540 vs 0.7949 / 0.6021 / 0.8297).

**Cost of the first design (CPU measurement, `check_state_vq_qat.py timing`, `qat/timing.json`).** 320–355 aten ops per loop step (forward ~190–230 incl. views, backward ~130), 2.4–2.5 µs per op on the CPU at 1 row; 144 card (x 2 layers) + 844 note step calls per 32k update. Assuming 6 µs of CUDA launch per op on top: +3.4 s per update over dec10's 1.12 s, 260 min for 3,465 updates unpadded (queued 420 min; the float64 `eigh` also syncs the host every step). dec10 itself took 65 min (23,908 reviews/s).

**Speed (S09 in docs/GPU_OPTIMIZATIONS.md; 2026-10-17 evening, CPU only; commit `6b7231e`).** The user chose "speed it up first". Three changes; the codec's output is unchanged:
1. *Factorization without an eigensolver.* `factorize_sq` (scripts/state_vq.py) finds the top eigenvector of the same normalised Gram G = An An^T by repeated squaring: P = G / tr G, then 12 squarings P ← P P (= G^4096) with a trace renormalisation before every 4th, then u = G p / |G p| where p is the column of P with the largest diagonal. v, sigma, the valid mask and the sign canon are computed as before. Recipes `vq378x412p` / `vq770x412p` are the same StreamRecipe with `factor="sq"` (same bits, codebooks and norm ranges). On all 90,419 entities of the SM32 state corpus (50 training users, 361,676 heads, λ2/λ1 up to 0.995, 99.9% quantile 0.94), the stored reconstruction equals the deployed eigh codec bitwise in float64 and to ≤ 1.1e-7 relative in float32 (rounding only; no pick differs; `qat/check.json` `b_codec_sq_vs_eigh`). **LogLoss effect:** the scorer's metrics with `vq378x412p` equal those with `vq378x412` exactly (Δ = 0) on the 6 smallest users34 users (3,716–7,102 reviews, epoch 10, w8). The scripted training path (float32 squaring) differs from the deployed scorer by at most 1.3e-6 on any metric (`b_codec_change_logloss`).
2. *Fused CUDA forward* (`scripts/kernels/state_vq_wkv.cu`, compiled by NVRTC through CuPy as in rwkv/model/nvrtc.py; glue in `scripts/state_vq_cuda.py`). One launch per split runs all steps. One 16 x 16 thread block per (row, head) does the recurrence, the squaring factorization, the 3- or 7-stage joint residual VQ (1,024 centroids, dimension-major codebooks for coalesced reads) and the log2 norm. It writes the output and the stored state before every step. Skipped steps (padding, 69% of the row steps) skip the codec. The *backward* is the validated K=16 WKV backward kernel (`rwkv7_wkv_backward_kernel<1, float>`, a checkpoint at every step) run along that stored trajectory, which is the straight-through gradient. Option `impl`: "cuda" (both kernels), "hybrid" (torch forward + the backward kernel), or "torch" (the first design; it is the default, so earlier option dicts keep their meaning). Each split now costs 2 launches instead of ~330 ops per step.
3. *Shorter chains* (measured, not used): at 2,048 x 16 an update has 78 card + 497 note steps (653 step calls, -42%) vs 144 + 844 at 32k. With the old torch step that is still ~136 min (CPU-based estimate), and the base is weaker (ctx2048c dec8 is +1.66% vs fork, dec10 +1.11%). With the fused kernels, the 32k recipe itself fits.

**CPU validation of the kernels** (no GPU used). The UNCHANGED .cu sources were compiled with MSVC behind a thread shim (`scripts/state_vq_cuda_emu.py`: 256 OS threads per block, std::barrier for __syncthreads, shuffles through an exchange buffer). They store bitwise the torch codec's states, both on synthetic sequences and on the busiest and longest rows of captured real inputs (epoch 10, w8, user 4676; card [483 x 13] / [128 x 35], note [430 x 13] / [3 x 308] truncated to 40 steps). Outputs differ by ≤ 2.4e-7, and the backward differs from the explicit STE backward by ≤ 3e-7 relative. The deployed eigh float64 codec applied to the kernel's states agrees to ≤ 1.5e-8. NVRTC for sm_120: forward 64 registers and 0 spills (float32 and float64 factorization), backward 39 registers. GPU-only risks left:
- FMA contraction and log2f/exp2f ulps can cause rare near-tie flips; `cuda-codec` and the bench gate them at ≤ 0.1% of coded steps.
- Launch and CuPy argument passing.
- The real speed.

**Projection** (model, to be measured by the bench). The fused forward reads 128 KB of codebook per VQ stage per coded head step from L2 (~190 GB per update) and has ~1,130 sequential steps of ~10–30 µs. Expected: +0.05–0.15 s per update, i.e. 32k ~75–80 min (dec10 65 min) and 2,048 x 16 ~35 min (plain decay 26 min). **Recommended recipe: 32k x 1 from epoch 10, as planned.** Reasons: it pairs 1:1 with dec10 and its PTQ row, it starts from the long-context base, and its chains match deployment. Use 2,048 x 16 only if the bench shows the 32k cuda path above ~2 h.

**GPU micro-benchmark** (`check_state_vq_qat.py bench`, ~5 min of GPU, no training). It first checks the kernels' correctness on CUDA: vs the torch path on CUDA and on the CPU, vs the deployed codec, backward vs the STE backward, and the shift codec CUDA vs CPU. It then measures per-update seconds of the plain WKV kernels and of the paths old / old+kbwd / torch-p / torch-p+kbwd / cuda, plus the shift codec, on sampled updates of both chunkings (`qat/timing.json`, `qat/timing-2048.json`: 40 updates each with their splits and coded steps). It prints the projected ticket minutes. `--dry-run` runs the same code on the CPU with emulated kernels and tiny shapes (done). The ticket's first command, `cuda-codec`, is the same correctness check plus the re-estimate for the run's chunking and impl; it stops above `--max-minutes`.

**Deliverable** (CPU, after training; the 464-user GPU scorer cannot run the codec): users34.json, `dec10/w8+s8`, `dec10/w8+vq378x412` (PTQ of dec10), `qat/w8+vq378x412`, plus `qat/w8+s8` (the QAT model's float-state quality, his "fp32 regress"), Δ% vs `fork/w8+s8` of the same users (`state-vq/fork.jsonl`), table `artifacts/small-model/state-vq/qat/summary.md`. Results: pending.

**Runbook** `artifacts/small-model/runbook-2026-10-18-qat.sh` (state-vq worktree; S09: `RECIPE=vq378x412p`, `FACTOR=float32`, `IMPL=cuda`, `MINUTES=110` until the bench gives the number, `MAX_MINUTES=180`; scoring with the deployed `vq378x412` codec): preconditions (codebooks sha, check ALL OK for the recipe, committed source) → prepare and compare with dec10's decay.json (only `state_vq` may differ) → one queue ticket (`cuda-codec` gate → trainer smoke → decay) → three CPU scoring processes → table. `DRYRUN=1` prepares in a scratch folder and echoes the ticket. Launcher `launch-after-repflags-qat.sh` waits for the SM30 rep-flags runbook's last line (`] done` or `ended with exit`). Follow-up commands: `HANDOFF-SM32b.md`. Do not change this worktree's scripts while the ticket runs (the trainer's source watch stops the run).

### Compressed state on the GPU scorer (SM44)

2026-10-20. The `+vq<recipe>` variants of `scripts/small_model_heldout.py` now run on CUDA too (before: CPU only, ~720 reviews/s, 68 users took 1 h), so compression is scored on the 732 pool. Code: `scripts/state_vq.py` section "GPU scoring" (`install_scoring`, `score_codecs`, `score_wkv`, `score_shift`), check `scripts/check_state_vq_score_gpu.py`, runbook and outputs `artifacts/small-model/sm44/`.

- **Placement** unchanged (the CPU scorer's): the answer uses the float update, the stored card/note state is the codec's reconstruction before that entity's next review; a token shift that reads the entity's previous review reads the coded vector; deck/preset/user state stays float on the long WKV kernel. The eager model's own `state_vq_slot` routes the card/note mixers to forward-only scoring functions (no straight-through Function, no per-step record kept).
- **WKV codec on the GPU**: rank-1 streams (every card recipe, the 412-bit note) run the unchanged fused kernel `scripts/kernels/state_vq_wkv.cu`, i.e. the Gram-squaring factorization (the `...p` codec), in float64 (`--vq-factor-dtype`, default float64 = the CPU scorer's type). A split is launched in row blocks of at most 131,072 rows x steps (rows are independent entities), which bounds the kernel's per-step state record. Rank-2 streams (the 828-bit note of `vq770x828`) have no fused kernel: a per-step torch loop on the device with the recipe's own eigh float64 codec. Shift vectors: `StateCodec.shift` on the device. Rows record `state_vq_path` and the codebook sha256.
- **Same picks as the deployed eigh codec**: on the CPU, the GPU path's code (slots + the "sq" codec in torch) reproduces the CPU scorer (deployed eigh codec) bitwise: max |dp| = 0 and dLogLoss = 0 on every head for SM42 `+vq378x412` (the 5 smallest pool users, 3,685-4,311 reviews), SM42 `+vq770x828` (3 users) and SM32c `+vq378x412` with its learned books (2 users), see `sm44/check-sm44-cpu.json`. The codec was active (float state moves p by up to 0.19 on the same users) and the slot path ran (9-13 WKV and 15-22 shift calls per user). The GPU check ticket compares the CUDA run with those stored CPU predictions (gate: benchmark LogLoss within 1e-5 per user and head) and measures reviews/s. The MSVC emulation of the kernel was not run here: it spins 256 threads and loaded the CPU to 94% next to the SM43 training.
- **Results**: pending (`sm44/sm44-summary.md`; follow-up in `sm44/HANDOFF-sm44.md`).

### Large-budget compressed state (SM46, ~6,000 / ~3,000 bits)

2026-10-20, asked by the outside researcher whose codec we reimplemented: the same codebook compression (post-training, no retraining) at about 6,000 bits per card and 3,000 per note (int8 card state: 26,624 bits, so ~4.5x smaller), on SM36 (`s-w64kd-gate-sc-lanes-dec12-1ep32k-muon`). Branch `sm46`, folder `artifacts/small-model/sm46/` (`HANDOFF-sm46.md`).

- **Recipe `vq5902x3028`** (`scripts/state_vq.py`): card 5,902 bits = 8 heads x 590 + 3 shift vectors x 394; note 3,028 bits = 4 x 560 + 2 x 394. Per 16 x 16 head: rank 6 (rank 6 keeps 99.8% of the head energy, rank 4 98.9%), joint (u, v) residual VQ stages (15, 13, 10, 7, 5, 3) for the card and (15, 12, 9, 7, 4, 3) for the note, 10-bit log2 norms. Shift: m4 sub-vectors x 8 stages of 12-bit codes + 10-bit norm.
- **Codebooks** fitted on SM36's own states: corpus from 50 SM36 training users (none of the 732 pool; 1,500 samples per stream-layer and user, 918 MB, not committed), k-means on 45 of them (5 held out for the proxy), WKV ranks 1-6 with (32, 24, 16, 12, 8, 6) stages and shift m4 x 8 stages; 10-bit norm ranges span the whole corpus (`fit_norm_range` for bits >= 10: the 1-99% quantile search clipped the largest norms, which dominated the norm error). `books/codebooks-sm46.npz`, 29.7 MB (not committed; sha256 and regeneration commands in the handoff).
- **How it was chosen**: a model-free held-out proxy (`sm46/alloc_proxy.py`, relative squared reconstruction error on the 5 held-out users) picked rank 6 with equalised per-component coding errors (relative WKV error 1.04e-2 at 630 bits per head; rank 4 at 600 bits 1.45e-2, rank 3 2.87e-2). A training-user screen (`state_vq_fit.py tune`, 8 SM36 training users, 115k reviews, model-side LogLoss vs w8+s8) then compared two splits of the budget: `vq5902x3028` (more shift bits) +0.000058 curve / +0.000141 instant, `vq5934x2996` (shift m4 x 6, more WKV stages) +0.000048 / +0.000152; rank-6 truncation alone (`vqr6exact`) -0.000015 / -0.000001. A tie within noise; instant is the binding head, so `vq5902x3028`. For scale: the 1,506 / 828-bit point of the SM32 screen cost +0.00009 / +0.00058 (other model and users).
- **CPU check** (`sm46/check-sm46-cpu.json`, ALL OK): the GPU scorer's code on CPU tensors (both streams on the torch step loop with the eigh codec; rank >= 2 has no fused kernel) reproduces the CPU scorer bitwise on the 3 smallest pool users (max |dp| = 0, dLogLoss = 0); the codec is active (float state moves p by up to 0.04). CPU cost: 3.5 ms per review on those users (4 threads), 134 reviews/s at 2 threads in the screen (w8+s8: 1,020/s).
- **GPU run**: detached runbook `sm46/runbook-sm46.sh` waits for the SM44 runbook to end, then a GPU check ticket and time-capped scoring tickets (464 pool first, then the rest of 732; only 464 if 732 projects above 4 h at the check's rate). Estimate from the step profile: ~0.95M codec steps on the 732 pool, ~1.2-2.5 h. Results: pending (`sm46/sm46-summary.md`).

### Faster held-out scoring (SM48)

2026-10-20, user request (focus on scoring speed). Branch `sm48` (from `sm46`). Profile: [sm48/profile.md](../artifacts/small-model/sm48/profile.md).

- **Where the time went** (SM45's 732-pool pass, 726 users x 2 variants, 1,978 s): 51% model call, 49% outside it (~10.8 µs per review-evaluation, flat across user sizes); preparation is overlapped by the 4 workers. Outside the model, the Python loops over labelled reviews (`collect_predictions`, upstream `get_stats`) cost 8.70 µs per review and variant on CPU. Inside the model, the preset/user/deck WKV recurrences are serial chains on 4 blocks of 16 threads (B = 1, 4 heads), ~13-19 dependent steps per review on 4 of the GPU's 84 SMs.
- **Implemented**: (1) numpy metrics (`scripts/heldout_fast_metrics.py`, default; `--legacy-metrics` = old path): bitwise equal, 0.85 µs per review and variant; (2) `--lanes N`: N users at once, threads with their own model copies and CUDA streams, in-flight reviews capped by `--lane-review-budget` (400k default, ~25 KB of GPU memory per review); +vq variants run one at a time.
- **CPU checks**: numpy vs dict metrics on real labels of 7 users bitwise (`sm48/check-fast-metrics.json`); the scorer with both paths vs the pre-SM48 scorer, 4 users x (w8+s8, w8+s4c8n), 8/8 rows bitwise (`check-cpu-orig-vs-array.json`, `check-cpu-orig-vs-dict.json`); lanes 2 vs 1 on CPU (fp32/w8; scheduler logic only) 8/8 bitwise.
- **GPU timing ticket** (`--timing`, ~30 min): detached launcher `sm48/launch-sm48.sh` waits for the SM47 ticket to end and for `HOLD-chain` to be absent; job `sm48/job-sm48.sh`: 40 users of the 732 pool (2.26M reviews), SM36 w8+s8 + w8+s4c8n, arms old / numpy / lanes 2 / lanes 4 in order old arr l2 l4 l4 l2 arr old, rows compared with the first old run (stop on a tolerance failure). Results: `sm48/timing/summary.md` (pending). Follow-up: `sm48/HANDOFF-sm48.md`.

## Backlog

| ID | Idea | Status | Next step |
|---|---|---|---|
| SM01 | Full-history held-out frontier vs fork int8 (464 users) | Done 2026-10-15 (430 users every label has) | See frontier table; 34 users unscored (time limits) |
| SM02 | w64 vs w32 and w64+KD vs w64, paired k = 3 screens | Done 2026-10-15: w64kd promising, w64 not | Scale-up blocked by the EPOCH_MIN fallback rule; main session decides (HANDOFF k=3 verdict) |
| SM03 | Scale-up of the winning recipe (gate tier, ≤ 6 h) | Done 2026-10-15 (main session override: w64kd, 3 epochs): `scale-w64kd-dec` +5.28% / +5.99% vs fork w8+s8, beats w32 decay, not a good candidate; packaged | Superseded by SM10: the next scale-up uses long-context chunks (5.7) |
| SM04 | CPU inference cost of fork vs w32/w64 (recurrent step, 1 thread, `--timing`) | Done 2026-10-16 03:06 (ticket 000382): batch 1 w64 only 1.12x faster than the fork, batch 256 1.86x (w32 3.1x); dispatch-bound | See "Result 2026-10-16 03:06: SM04 CPU inference timing"; follow-up SM18 |
| SM05 | KD only on fork-unseen training users (teacher targets on its own training users may be overfit) | Untested idea | Needs a per-user KD mask in the trainer |
| SM06 | w80 + KD, longer decay (1 epoch, Expertium's choice) | Untested; re-scoped 2026-10-16: at 128 chunks it is pointless | Only at 2,048 (or 4,096) chunks and only if SM13 says capacity-bound: queue Q7 |
| SM07 | Window-protocol and collection evaluation of w64 checkpoints (`fast_evaluation` and `accuracy_collection` only build the w32 model) | Gap | Add a variant-aware evaluator hook |
| SM08 | Fork runtime port of the exported format (section 4, "Fork changes needed") | Specified, not started | User or Codex in the fork; acceptance = Rust vs `small_model_numpy.py` within 1e-5 |
| SM09 | Teacher-target cache for KD (section 5.2) | In use since 2026-10-15. ctx2048 run (accuracy lane): epoch 1 80,842 reviews/s (online + writes), epoch 2 102,143 (hits), below the hoped-for no-KD rate | S06 (queue Q3) measures the no-KD rate at 2,048 chunks; if the gap is the cache read, S07 (cache I/O, cross-architecture sharing) |
| SM10 | Long-context training (A09): in the 128-window protocol w64kd matches the fork (+0.1% LogLoss); the full-history gap is long-context behaviour (section 5.1) | Confirmed 2026-10-15 (screen B, paired k = 3). `scale-w64kd-gate-cached-ctx2048` (2,048 x 16, 4 epochs): gates e1 +3.86% / +5.04%, e2 +3.24% / +3.73% vs fork on the 100-user subset (projection +2.45% / +2.31%); epochs 3–4, decay and 464-user scoring running 2026-10-16 | Final verdict (F) at ≈ 03:10; continuation per Priorities, queue Q4; chunk 4,096 probe Q6 |
| SM11 | KD targets from the teacher's full-history replay instead of its 128-chunk predictions | Implemented and CPU-checked 2026-10-18 (branch `sm11`: `scripts/sm11_replay_export.py`, Teacher `"kd_source": "replay"`; check ALL OK; 10.27 B/review = 3.70 GB for 5k users); GPU replay not launched | Run `artifacts/small-model/runbook-sm11-replay.sh` (4–5 tickets of 95 min, ≈ 5.6 h), then use it in SM21 v1 or an SM11-only decay screen (`HANDOFF-SM11.md`) |
| SM12 | Head temperature (Platt) folded into the export | Measured offline: closes ≈ 8% of the instant RMSE(bins) gap, LogLoss −0.1% | Cheap add-on once the recipe is fixed; fit on training users, never on the 464 |
| SM13 | In-window check at 2,048: student@w2048 vs fork@w2048 on the 100 gate users (the 5.1 decisive check at the new context) | Done 2026-10-16 03:19 (ticket 000383): G = −0.34% curve / +0.48% instant, **context-bound** (full history on the same users +1.98% / +2.40%) | See "Re-assessment 2026-10-16 ~03:30 (G and Q7)": Q7 = 4,096-chunk continuation (SM19) if Q6 passes |
| SM14 | Extend `scale-w64kd-gate-cached-ctx2048` to 6 epochs + decay from e6 | Proposed 2026-10-16 (log-slope: ≈ −0.35 pp curve / −0.75 pp instant) | Queue Q4, branch E; new `runbook-2026-10-16.sh extend` (Sonnet medium), never `runbook-next.sh` |
| SM15 | Pruning step 1: w48 student (3 heads x 16, 195,949 parameters, 14x fewer than the fork) with the same long-context KD recipe, non-inferiority gates vs the ctx2048 run | Proposed 2026-10-16; starts only when the bar is met (user: accuracy first, then prune) | Queue Q7-P (or Q4 if F ≤ bar); needs the `w48` preset, gate `--reference-prefix`, ARCH-generic runbook stage |
| SM16 | Context beyond the chunk: carry the recurrent state (detached) from a user's chunk k to chunk k+1 (truncated BPTT), so training sees deployment-length histories | Idea 2026-10-16 (design, Opus); conflicts with the per-batch purity the KD cache relies on, so it needs a batch-order redesign | Only if SM13 says context-bound and Q6 says longer chunks no longer pay |
| SM17 | Epoch scoring after every epoch on the 100 gate users; 464 users for finals and for gates near their boundary; all-464 gates once S05 lands | Proposed 2026-10-16 (answer to the user's 100-user question, Priorities) | Built into the new runbook stages (Q4, Q7) |
| SM18 | Low-overhead CPU step (fused/compiled) to realise the MAC saving at batch 1 | Done 2026-10-16 (merged `sm18`; ticket 000387, `--timing`, 1 thread): at batch 1 w64 numpy 0.509 ms = 7.8x faster than the eager fork (3.954) and 1.84x faster than the numpy fork (0.939); exact to 4.2e-7 (s8 state path not covered) | See "Result 2026-10-16 04:33: SM18 low-overhead CPU step"; eager w64 differs from SM04 (2.364 vs 3.452 ms), eager code unchanged, unexplained variance |
| SM19 | Continue the ctx2048 run at 4,096-review chunks (epochs 7–8 from e6, x8 batch) + decay at 4,096 | Proposed 2026-10-16 ~03:30 (G context-bound) | Q7 primary if Q6 passes; needs the `cont` stage in `runbook-2026-10-16.sh` (Sonnet medium) |
| S05b | Batch only users below a length threshold in full-history scoring, keep long users at batch 1 (S05 batched was 0.78x overall: short users ~2x, long users slower) | Proposed 2026-10-16 07:20 | Opus/Sonnet: threshold from the per-user lengths in `s05-check2.jsonl`; branch `s05-batched` stays the base; rule ≥ 2x |
| SM20 | Drop `"optimizer"` from the KD cache key (teacher targets do not depend on the optimizer) | Proposed 2026-10-16 07:34: Muon epoch 1 would recompute the teacher targets (≈ 76 min, +6 GB) | Fixed in 5cbfe15 (branch `sm20`): `TEACHER_BLIND = ("distillation", "optimizer")` in `scripts/expertium_options.py`; only `optimizer` added (lr_schedule etc. kept in the key). Verified on CPU: all 4 existing manifests under `artifacts/cache/kd-teacher/fork-bf16-s2025` keep their digest and their manifest identity (no ValueError); AdamW dict, `+"optimizer": "expertium-iter53"` and `+{"kind": "muon"}` give the same digest (`scripts/check_kd_cache.py`). Muon epoch 1 reuses the AdamW targets once merged |
| SM21 | Training with carried state (supersedes SM16): fresh carry at update granularity (v1), strided lanes + stale states at lane starts (v2), with SM11 full-history teacher targets | Design + CPU prototype 2026-10-16 (`scripts/state_carry.py`; check A/B/C pass: carry off = today's batches, zero state bitwise, exact carry = full history to 5e-7; option 3 store 576–739 GB/epoch int8 at 2,048) | 2026-10-18: v1 kernels + trainer option `state_carry` implemented on branch `state-carry`, CPU checks ALL OK (`HANDOFF-SM21.md`); GPU parity passed (000435, +9%/update); arm A (v1) trained and arm B (zero) running from `state-carry`. v2 `{"lanes": "strided"}` on branch `sm21v2`: CPU check ALL OK, stale share 0% (cards: exact 19.2%, zero at lane cuts 19.1%). Next: v2 GPU check (`jobs/sm21v2-gpu-check.sh`, 15 min), then arm C `runbook-2026-10-18-sm21v2.sh` |
| SM22 | Window curriculum: most epochs at 2,048-review chunks (≈ 105k reviews/s), the last 1–2 epochs + decay at 8,192 (or longer) | Proposed 2026-10-16 13:50 (user). Evidence: from the same e6, one epoch at 8,192 took the 100-gate-user gap vs fork from +2.38% / +2.41% to +1.66% / +1.80% (curve / instant), while two more epochs at 2,048 only reached +2.04% / +1.99% and flatten; 8,192 trains at 32,528 reviews/s (3.2x slower). Q8 (`scale-w64kd-gate-cached-ctx8192-cont`, e7–e8 + decay) is the first instance. Status 2026-10-17: training at 16k (e9) beats decay-only (dec9 +1.03% / +1.04% vs dec8-L16k +1.17% / +1.16%; dec9-L32k +1.01% / +1.01%); e10 at 32k in progress (ticket 000420) | Default recipe until SM21 lands. Open: where to switch (after e4? e6?), and whether 16k/32k for the last epoch pays (cold share 2,048 46% / 4,096 36% / 8,192 26.8% / fork 32k 11.5%) |
| SM23 | More parallel windows at long chunks: 8,192 x 8 (65,536 reviews/update) instead of x4, to refill the GPU (the WKV recurrence is sequential in time, so fewer longer windows per update leave it partly idle) | Proposed 2026-10-16 13:50 (user) | `--timing` ticket first (speed claim, VRAM fit on 16 GB); if it pays, an LR check at the doubled update size (sqrt scaling or a short sweep) before any accuracy run; compare reviews/s and accuracy per GPU-hour against SM22 |
| SM24 | Gap by review type / same-day on all 464 users (GPU `--dump` scoring of fork and the best students) | Proposed 2026-10-17 (feature screen): on 68 users the curve gap is all on learning/relearning reviews, −0.03% on review-state reviews | ~35 min GPU after Q11; then a user decision on also reporting bar P on review-state reviews |
| SM25 | Review-type robustness: score fork and student with history review types rewritten as a non-SM-2 scheduler would label them | Done 2026-10-17 (CPU, 68 users): no-steps labels cost the fork +0.43% curve / +0.55% instant (zeroing +8.4% / +3.3%) and remove its learning-step curve lead over the student (see the follow-up under the feature screen) | Report a relabelled-fork reference beside bar P |
| SM26 | Instant-weighted final decay (instant loss weight x2), then a wider instant head only | Proposed 2026-10-17: instant gap uniform on every slice, no feature explains it. Code on branch `sm26` (checked, unmerged) | ~1.2 h GPU, paired against the matching decay. Also report both heads by interval length: Expertium/obezag (2026-10-17) found that removing the curve heads helps instant on short intervals but hurts it on very long ones, so upweighting instant may cost long-interval reviews. Prior from his research_5k: re-weighting the loss gave mode trades (iters 17/19, a binary-recall term: instant better, curve worse, linear in the dose) and head-capacity changes were nulls (iter 12 head resolution, iter 48 R-coupled rating head), so expect a trade and drop the "wider instant head" follow-up |
| SM27 | Long-history summary inputs (card prior reviews/lapses, FSRS-6 S/D/R, user pass rate as of the previous review) for windowed training, zero-initialised extra input columns | Proposed 2026-10-17 (feature screen negative at full-history inference; targets the cold-window gap) | ~0.5 day code (data columns, query keep list, fast-path columns, input dim, teacher slice, `TEACHER_BLIND`) + ~3.2 h GPU vs `ctx2048c-dec8` |
| SM28 | Session/clock (`review_time`), deck depth, card creation age, note siblings | Screened 2026-10-17: no signal for either model | Not planned |
| SM31 | Deployment-realistic scoring without the duration leak: score fork and student with end-to-start intervals (`EXPERTIUM_END_TO_START=1`), i.e. the interval Anki knows when it shows the card | Proposed 2026-10-17 (Expertium: his leak fix cost +0.0020 instant, e2s +0.0005). Our published layout has the same class of leak (query intervals run answer-to-answer and include the review's own duration, docs/EXPERTIUM_CATCH_UP.md section 4); student runs and the fork both train and score with it, so bar P is like-for-like but optimistic for deployment | CPU on the 68 feature-screen users first (both models, normal vs e2s), then 464 if the gap moves; retraining with e2s only if the deployed gap differs |
| SM32 | Expertium's state quantization on the w64 student: codebook card/note state with tunable bit budgets (his deployed ~450 / 400 bits per card / note), LogLoss vs bits | PTQ test done 2026-10-17 (section "State quantization test (SM32, post-training)": his 403/214 recipe +0.0052 / +0.0082 PTQ, 378/412 with 6-bit norms +0.0012 / +0.0023, 770/828 +0.0007 / +0.0014 vs student w8+s8, 34 users); QAT in the final decay (SM32b) implemented and CPU-checked 2026-10-17 (section "State quantization, QAT in the final decay (SM32b)"), GPU runbook ready, not launched; next: run it, then the 34-user table. Proposed 2026-10-17 (Expertium: "a lot of work done on quantization, no need to continue int8/int4"). Largest dataset user (246,801 cards, 76,887 notes): fork f32 state 15.6 GB, fork int8 4.6 GB, w64 int8 0.88 GB, his budgets ~14–18 MB. Relates to COMPRESSION Q06/Q09 | Port or reuse his quantizer (rwkv-anki-autoresearch) for the w64 student; curve of LogLoss vs card/note bits on the 464 users; replaces our int8-state (s8) column once it works. Cost to expect (his d=64, 4 residual stages, 403 / 214 bits, QAT): +0.0018 ahead / +0.0028 imm (~+0.6% / +1.1%). Expertium's recommendation (2026-10-17): raise the budget to 400 / 400 bits (card / note) and measure LogLoss; if tuning does not close the gap, try 800 / 400 and 800 / 800. His measured tax vs bits (QAT, plain twin as reference): d=64 1 stage 163 / 94 b +0.0028 / +0.0040; d=64 4 stages 403 / 214 b +0.0018 / +0.0028; d=80 3 stages 385 / 205 b +0.0016 / +0.0026; d=80 5 stages 585 / 305 b +0.0013 / +0.0021 (doubling bits cuts the tax ~25%) |
| SM33 | Grade-order sanity: does the student keep interval(Again) <= interval(Hard) <= interval(Good) <= interval(Easy)? | Proposed 2026-10-17 (Expertium enforces it in his curve head, costing +0.0020; PAVA was deliberately not ported here, GPU_OPTIMIZATIONS X06) | CPU: violation rate and size for fork and student on the 68 users; enforce only if violations are frequent enough to matter for scheduling. His costs: rectification (PAVA pooling + current duration zeroed) +0.0019..0.0036 ahead; a head ordered by construction (R2) +0.0004..0.0006 ahead, imm +0.0002 better |
| SM34 | Re-sync with Expertium's research (rwkv-anki-autoresearch; HEAD is still our pinned 1f2e6b0) | 2026-10-17: `research_5k.md`, `LEDGER.md` and `GPU_RUN_BOARD.md` read (main session); new rows SM35–SM38 below. His bottom line: d=64 plain, leak-free, 333k params, 10+1 epochs = +0.0029 ahead / +0.0010 imm vs the old d=128 model (VAL half; ahead pessimistic by ~0.002 because his rows are rectified); deployed with ~1,000x smaller state +0.0048 / +0.0039. His d=128 retrained 10+2 in his pipeline (`d128e10`): 0.29628 / 0.26116 vs the old 0.29461 / 0.26356. Dead ends to skip: by-user and unscored-row loss weighting, SAM, LAWA, PCGrad, self-distillation imm→ahead, R-coupled head, deck tree, self-feed, next-rating ordinal supervision, duration dropout, shrink-perturb, warmup-only KD, head resolution 64→128, channel-mixer and LoRA-rank grafts, peak LR x0.5 / x2 | Still unread: `research_5k_notes.md` (162 KB) and `research_5k_verbose.md` (370 KB); a Sonnet pass only if a row below needs their detail |
| SM35 | 1-epoch decay instead of our 0.25 epoch | Proposed 2026-10-17 from his research: decay ratio 1.0 was one of his largest gains (iter 34, bundled: ~−0.0015 both heads with the Muon LR change), and 1 epoch beat 2 and 3 after a 10-epoch WS (LEDGER 14b). Our decays are all 0.25 epoch | Cheapest form: a 1-epoch decay at 2,048 windows from the best undecayed checkpoint, paired against its 0.25-epoch decay (≈1 h); at 16k/32k a 1-epoch decay costs 4.5–7 h, so test short first |
| SM36 | Muon optimizer for the student (promote from parked) | His Muon gains: hybrid Muon+AdamW −0.00014 ahead / −0.00049 imm (iter 29, the largest imm gain of that phase), Muon on LoRA −0.0002 / −0.0001 (iter 53), on the k/v scale matrices imm −0.0001 (iter 67). Our student is AdamW only; `"optimizer": "expertium-iter53"` exists here | Imm is our binding gap: a continuation or decay with Muon from the best checkpoint vs the same with AdamW; from-scratch only if that is ambiguous |
| SM37 | Real-timestamp input features (the `-id` dataset: card age, tenure, time since any review, creation batch…) | His featB + clock-leak fix net: about −0.0005 ahead / −0.0008 imm (LEDGER rows 10 + 14; row 10 alone −0.0007 / −0.0029 was mostly the clock leak). Our post-hoc screen (SM28) found no signal at inference, which does not rule out a training gain | Needs the `-id` build in training and review timestamps at inference (Anki has them); after SM35/SM36 |
| SM39 | Continue Expertium's tuning queue on the 333k model (his recommendation 2026-10-17) | His open queue at 1f2e6b0: Muon LR x0.5 (`wtmuon0p5`, running then) and x2 (`wtmuon2`), WS warmup 400 → 2,000 steps (`wtwarmup2000`), QAT weight decay 0.0 / 0.03 (control `s64hcq4`); results never pushed. Our w64 student is his s64h trunk (331,511 vs his 333,178 params: no sibling-rating inputs, no PAVA thetas) | Route decision (user): port his remaining wins into our pipeline (KD, long windows, 464-user bar) or run his repository and data build directly |
| SM38 | Size of the label-free fork reference recipe | His LEDGER row 1: retraining d=128 with a 1.25-epoch recipe costs +0.0037 / +0.0043 vs the upstream weights; his 10+2-epoch d=128 retrain is mixed vs upstream (ahead +0.0017, imm −0.0024) | Supports fine-tuning the fork without review type (cheap) over a from-scratch retrain for the label-free reference |
