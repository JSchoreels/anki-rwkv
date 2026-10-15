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

Starting point: `scale-w64kd-dec` (w8+s8) is +5.28% curve / +5.99% instant LogLoss and +85% instant RMSE(bins) vs fork w8+s8 (430 users). The good-candidate bar is ≤ +1.0% / +1.5% LogLoss and ≤ +15% instant RMSE(bins). Runbook: [artifacts/small-model/runbook-next.sh](../artifacts/small-model/runbook-next.sh). All rates and times below are accuracy-lane estimates from logged training rates and queue durations, not benchmarks.

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
4. **GPU check, `scripts/check_long_wkv.py`** (runbook stage `check`, ≈ 10 min, not run yet: queuing it needs the user's go):
   - for K = 16 and 32 at T = 512, 1,024 and 2,100: the generic NVRTC float32 forward and all gradients against a float64 autograd reference (whose forward matches `small_model_wkv.reference` to 1.1e-7 on CPU);
   - the first 383 outputs of a long run bit-identical to a 383-step run;
   - the row-tiled exact kernel bit-identical to the generic kernel;
   - a probe of one real 32,768-review update (user 101) as 256 × 128 vs 32 × 1,024 chunks, both with the same 49,828 labelled rows: teacher + student fwd/bwd, median ms, peak MiB.

   `context` refuses to run until this check has passed. Size arm B from the probe: if 1,024-review updates are more than about 2× slower, use 0.125-epoch decays.

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
| `scale` | e.g. `TIER=gate EPOCHS=6 EPOCH1_MIN=62 EPOCH_MIN=41 EXTRA='{"chunk_reviews": 1024, "batch_chunks": 32}'` (if B wins). Trains epoch 1 and scores it (slope point, ≈ 75 min), then stops; `CONTINUE=1` resumes epochs 2–6, decay and scoring | ≈ 4.7 h at 128-review chunks (re-estimate at 1,024 from the probe) | `scale-w64kd-gate-cached[-tag]`, labels `s-w64kd-gate-{e1,e6,dec}` |

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
  - `scale` stops after epoch 1 and its score until re-launched with `CONTINUE=1`.
- **Smoke checks.** Adopted: every training ticket begins with `expertium_decay.py smoke` (60–120 s, same options) and trains only if it passes.
- **Longer training context and matching KD targets.** Adopted as P1 (5.3). With longer chunks the online teacher sees the same long chunks, so its targets match the student's context. The cache keeps them apart from 128-review targets (batch-option hash). Full-history teacher targets (SM11) remain a later option.

## Backlog

| ID | Idea | Status | Next step |
|---|---|---|---|
| SM01 | Full-history held-out frontier vs fork int8 (464 users) | Done 2026-10-15 (430 users every label has) | See frontier table; 34 users unscored (time limits) |
| SM02 | w64 vs w32 and w64+KD vs w64, paired k = 3 screens | Done 2026-10-15: w64kd promising, w64 not | Scale-up blocked by the EPOCH_MIN fallback rule; main session decides (HANDOFF k=3 verdict) |
| SM03 | Scale-up of the winning recipe (gate tier, ≤ 6 h) | Done 2026-10-15 (main session override: w64kd, 3 epochs): `scale-w64kd-dec` +5.28% / +5.99% vs fork w8+s8, beats w32 decay, not a good candidate; packaged | More users/epochs, w80 + KD (SM06) or longer decay to close the remaining ~4.3 / 4.5 pp |
| SM04 | CPU inference cost of fork vs w32/w64 (recurrent step, 1 thread, `--timing`) | Pending | Small timing ticket with the RNN classes (`expertium.variants.build(..., rnn=True)`, fork `srs_model_rnn`) |
| SM05 | KD only on fork-unseen training users (teacher targets on its own training users may be overfit) | Untested idea | Needs a per-user KD mask in the trainer |
| SM06 | w80 + KD, longer decay (1 epoch, Expertium's choice) | Untested | After SM03 |
| SM07 | Window-protocol and collection evaluation of w64 checkpoints (`fast_evaluation` and `accuracy_collection` only build the w32 model) | Gap | Add a variant-aware evaluator hook |
| SM08 | Fork runtime port of the exported format (section 4, "Fork changes needed") | Specified, not started | User or Codex in the fork; acceptance = Rust vs `small_model_numpy.py` within 1e-5 |
| SM09 | Teacher-target cache for KD (section 5.2) | Implemented 2026-10-15, CPU check bit-identical (`check_kd_cache.py`); 3.7 GB / 5.8 GB for 5k / 8k users | First real use in `runbook-next.sh control`; read the cached epoch's reviews/s from `phases.jsonl` |
| SM10 | Long-context training (A09): in the 128-window protocol w64kd matches the fork (+0.1% LogLoss); the full-history gap is long-context behaviour (section 5.1) | Diagnosed 2026-10-15 (CPU, 64 users); P1 implemented and CPU-checked (`chunk_reviews`, `check_long_chunks.py`) | `runbook-next.sh check` (GPU parity + probe), then `context` |
| SM11 | KD targets from the teacher's full-history replay instead of its 128-chunk predictions | Idea | Only if SM10 confirms the context diagnosis; one replay ≈ 3.8 h for 5k users at batch 1 (needs batching across users) |
| SM12 | Head temperature (Platt) folded into the export | Measured offline: closes ≈ 8% of the instant RMSE(bins) gap, LogLoss −0.1% | Cheap add-on once the recipe is fixed; fit on training users, never on the 464 |
