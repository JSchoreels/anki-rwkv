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
- **P (start pruning):** F ≤ +1.0% curve and ≤ +1.5% instant (the user-agreed good-candidate LogLoss bar; RMSE(bins) reported beside it). Pruning only makes sense with that headroom: Expertium's d=128 → d=64 cost +0.5% / +1.2%.
- **E (extend the same run):** not P, F ≤ +4.0% on both heads (4 epochs must at least beat a single 0.25-epoch ctx1024 decay, +3.91% / +4.13%), and S "still learning".
- **D (else):** no continuation of this recipe tonight; Q4 stays empty.

| # | Lane | Item | Decides | Ready? Who writes | Launch | GPU min | Skip / stop |
|---|---|---|---|---|---|---:|---|
| Q1 | speed/goal, night, `--timing` | **SM04 CPU inference timing**: fork (`srs_model_rnn`, fork repo read-only import), w64 and w32 recurrent step, 1 thread, f32 and int8 weights; per-review latency at batch 1 (runtime case) and at batch 256 cards (amortises Python dispatch), plus analytic MACs per review | The CPU saving of w64 vs the fork (now only the 8.3x parameter estimate) | Code first: `scripts/small_model_cpu_timing.py`, Sonnet medium (≈ 1 h, CPU smoke of 100 steps per model) | `$PY $QUEUE run --owner small-model --purpose "small-model: SM04 CPU inference timing (1 thread)" --claim gpu --timing --expected-minutes 15 -- $PY -X utf8 $WT/scripts/small_model_cpu_timing.py --threads 1 --out $SM/cpu-timing.json` (claims the GPU so no training workers load the CPU) | 0 (15 of machine) | Skip if the script is not CPU-checked by 03:10; then run it after Q5, still before 08:00 |
| Q2 | accuracy | **SM13 in-window@2048 diagnostic** on the 100 gate users | G (capacity vs context beyond 2,048) | Runnable as-is (`--window` exists) | `$PY $QUEUE run --owner small-model --purpose "small-model: SM13 in-window@2048 fork vs ctx2048 dec" --claim gpu --expected-minutes 15 -- $PY -X utf8 $SC --users-file $SM/gate-users.json --window 2048 --model fork=fork::$TEACHER --model $LBL-dec=expertium:w64:$WT/artifacts/small-model/scale-w64kd-gate-cached-ctx2048-decay/$CKPT --only fork/w8+s8 --only $LBL-dec/w8+s8 --time-limit 900 --out $SM/heldout.jsonl` (rows get `@w2048`, never mixed with full history) | 10–15 | Users with an OOM row on either label are dropped from G (report how many); if more than 20 drop, rerun with `--max-reviews 200000` |
| Q3 | speed, night, `--timing` | **S06 long-context training speed profile** (160 users, `expertium_speed.py`): w64 wide 256 x 128 control, 32 x 1,024, 16 x 2,048, 32 x 2,048; no KD; all at `gpu_memory_fraction` 0.8 | Whether ctx2048 costs the recurrence (few rows, long T) or the KD cache (102k cached vs the no-KD rate): next speed work is cache I/O (S07) or the K=16 native WKV / bigger batches (W05) | Ready 2026-10-16: 4 cases `w64-ctx128-b256`, `w64-ctx1024-b32`, `w64-ctx2048-b16`, `w64-ctx2048-b32` in `scripts/expertium_speed.py cases()` + `$SM/jobs/speed-ctx.sh` (logs `nvidia-smi` first; exits 2 if VRAM in use > 3.6 GB at start). CPU-checked: `prepare --users 160` writes the 4 cases (w64, wide ports kept, no KD, fraction 0.8) and all their option keys are in the `gpu_r01_prefetch.py` allowlist; `bash -n` ok; nothing run on GPU | `$PY $QUEUE run --owner small-model --purpose "small-model: S06 long-context speed profile" --claim gpu --timing --expected-minutes 25 -- "C:/Program Files/Git/usr/bin/bash.exe" $SM/jobs/speed-ctx.sh` | 25 | A case is invalid if `gpu_power.jsonl` shows < 150 W or memory pinned above 15.5 GB (spill); skip the item if VRAM in use exceeds 3.6 GB at start |
| Q4 | accuracy | Branch **E**: epochs 5–6 of the same run, a 100-user score after epoch 5, decay from e6, 464-user scoring of `$LBL-e6` and `$LBL-dec6`. Branch **P**: w48 pruning run, see Q7-P (it moves here). Branch **D**: nothing | Whether 6 epochs reach the instant bar (log-slope expectation ≈ +1.6–1.8% instant, ≈ +2.5–3.2% curve on 464 users) | **Ready 2026-10-16** (`$SM/runbook-2026-10-16.sh extend`, CPU-checked and dry-run with a stubbed queue; the LR is constant 7e-4 and independent of the epoch count, so 4 → 6 in `experiment.json` and `run/run.json` is consistent; it refuses unless epoch-04 is complete, the old runbook has exited and the e4 decay exists; the decay goes to `$SM/scale-w64kd-gate-cached-ctx2048-e6-decay`). Spec: a NEW file `$SM/runbook-2026-10-16.sh` (never edit `runbook-next.sh`), stage `extend`: CPU-check that an epoch-5 `protocol.json` has the same LR/optimizer fields as epoch 4 and resumes from `epoch-04`, then set `experiment.json` epochs 4 → 6; one ticket per epoch (smoke first, `--expected-minutes 66`); score e5 on the gate users (`--expected-minutes 6`); decay into `$RUN-e6-decay` (≈ 27); scoring (30) and a resume ticket only if users remain. Dry run with a stubbed queue, as done for `epoch_gate` | `TIER=gate EPOCHS=6 RUN_TAG=ctx2048 nohup bash $SM/runbook-2026-10-16.sh extend > $SM/runbook-2026-10-16-extend.log 2>&1 & disown` | ≈ 175 | Stop after epoch 5 if e4 − e5 ≤ 0 on both heads (flattened); do not start later than 05:00 (it must finish before Q7) |
| Q5 | speed, night, `--timing` | **S05 batched full-history scoring** (`--batch-users N`, length-bucketed): tolerance + timing on the 100 gate users against existing rows of `$LBL-e2/w8+s8` | Whether gates and finals can move to 464 users at today's 100-user cost (model time is ≈ 8.6 of the ≈ 11 min per 464-user label, all of it batch-1 recurrence at 62–78k reviews/s) | **Ready** (2026-10-16, Opus): branch `s05-batched` (code e4a7a06; not merged into `small-model`), worktree `wt/s05-batched`. Per-user encoding seeds (inputs equal batch 1), per-row int8 state rounding (already per user), batch-1 loss path on each user's sliced outputs, length buckets (≤ 8 users, padded sequence length ≤ 800,000 = users x longest x 1.9), OOM fallback (split the group in half, recursively, batch 1 last); expertium models only. CPU check passed (`$SM/s05-cpu-check.json`, rows `s05-cpu-b1/b8.jsonl`): 11 gate users ≤ 7,000 reviews, `$LBL-e2/w8+s8`, batch 8 vs 1: input SHA-256 equal, counts exact, max |Δ| aggregate 2.4e-7 (≤ 1e-6), per-user loss 1.2e-7 (≤ 1e-5), per-user AUC 2e-6 (≤ 1e-4); CPU model time 23.1 s vs 45.1 s (1.95x, 2 threads). **First GPU run (ticket 000392, 2026-10-16 06:35) crashed at group 8/17** (`torch.AcceleratorError: CUDA error: out of memory` from `torch.gather`, not a `torch.OutOfMemoryError`, so the old fallback did not fire): the 56 users of groups 1–7 (up to 8 x 48,045 reviews, longest sequence 84,142) were within tolerance (max aggregate 1.8e-7, per-user loss 1.6e-6, AUC 6e-6) and 2.05x faster in GPU model time (87k → 179k review-evals/s); group 8 (8 x 61,058 reviews, padded est. 928k) did not fit. **Fixed** (see GPU_OPTIMIZATIONS S05): groups capped by padded length, OOM of any kind splits the group. **Rerun** with the command below but `--out $SM/s05-check2.jsonl` (new file) and the compare against `s05-check2.jsonl` (`--out $SM/s05-gpu-check2.json`); `s05-check.jsonl` keeps the 56 good users | `$PY $QUEUE run --owner small-model --purpose "small-model: S05 batched scoring tolerance + timing" --claim gpu --timing --expected-minutes 12 -- $PY -X utf8 C:/Users/jonat/workspace/wt/s05-batched/scripts/small_model_heldout.py --users-file $SM/gate-users.json --batch-users 8 --model $LBL-e2=expertium:w64:$RUN/run/epoch-02/$CKPT --only $LBL-e2/w8+s8 --out $SM/s05-check.jsonl` (`$SC` once merged); then on CPU `$PY -X utf8 C:/Users/jonat/workspace/wt/s05-batched/scripts/small_model_batch_compare.py --reference $SM/heldout.jsonl --candidate $SM/s05-check.jsonl --label $LBL-e2/w8+s8 --allow-missing-input-sha --out $SM/s05-gpu-check.json` (`tolerance.passes` and `timing.speedup` vs the 89.7 s batch-1 model time decide; the reference rows have no input SHA, input identity rests on the CPU check because `prepare` runs on CPU in both paths) | 12 | Skip if the CPU check has not passed by 07:30 (no timing after 08:00); adopt only if within tolerance and ≥ 2x faster in model seconds |
| Q6 | accuracy, day | **ctx4096 decay probe**: 0.25-epoch decay of the best epoch (e6 after E, else e4) at 4,096-review chunks x 8, k = 1 vs that epoch's 2,048 decay | Whether chunk length still pays beyond 2,048 (1,024 → 2,048 gave −0.00095 / −0.00106); if not, longer context needs SM16/SM11 (design) | Runnable as-is (explicit commands; cold cache for 4,096 chunks; T = 8,192 rows relies on the any-T kernel loop and the smoke, as 2,048 did) | `$PY -X utf8 $DEC_PY prepare --source-run $RUN/run --epoch <K> --out $SM/cal-ctx4096-e<K> --fraction 0.25 --shape one_minus_sin --evaluate none --seed 1 --extra-options '{"distillation": {"teacher": "<TEACHER>", "alpha": 0.9, "decay_alpha": 0.5, "cache": "C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-teacher/fork-bf16-s2025"}, "chunk_reviews": 4096, "batch_chunks": 8}'`; then `$QUEUE run ... --expected-minutes 40 -- bash -c "'$PY' -X utf8 '$DEC_PY' smoke --out '$SM/cal-ctx4096-e<K>' && '$PY' -X utf8 '$DEC_PY' run --out '$SM/cal-ctx4096-e<K>'"`; then scoring `--expected-minutes 16 -- $PY -X utf8 $SC --model cal-ctx4096-e<K>=expertium:w64:$SM/cal-ctx4096-e<K>/$CKPT --only cal-ctx4096-e<K>/w8+s8 --time-limit 3000 --out $SM/heldout.jsonl` | ≈ 55 | Pick rule as `chunk_pick` (≥ 0.001 better on both heads, upper95 < 0, ≤ 1.5x slower); skip in branch P |
| Q7 | accuracy, day | Main daytime run, chosen at ≈ 08:30 from F (after Q4), G and Q6 (chunk = 4,096 if Q6 passed, else 2,048), 4 epochs + gates after **every** epoch + decay + 464-user scoring: **P** (F ≤ bar): **w48** student (3 heads x 16, 195,949 parameters, 14x fewer than the fork), same KD recipe, non-inferiority gates (test A′, band 0.0062 / 0.0027) against the ctx2048 run's own epochs. **Capacity-bound** (G above band): **w80** (558,209) at the chosen chunk, test-A gates against the ctx2048 run. **Context-bound, Q6 passed**: 2 more epochs of the best checkpoint at 4,096 chunks + decay. **Context-bound, Q6 failed**: epochs 7–8 only if e4→e6 gave ≥ half its expectation (0.0005 / 0.0010 on the gate users), else nothing (design day for SM16/SM11) | The next recipe step toward the bar, or the first pruning step | Code first, Sonnet medium (≈ 1.5 h, during the night): stage `arch` in `runbook-2026-10-16.sh` (ARCH and reference-label variables, scoring with `expertium:<ARCH>:`, gate after every epoch), `w48` preset in `expertium/variants.py` (`w80_config(heads=3, mult=2)`, CPU param count 195,949), `small_model_gate.py --reference-prefix` (self-test extended), and `check_wide_kernels.py` accepting w48 (one GPU parity ticket ≈ 3 min before training). Optional speed add-on S07 (Opus review): share the KD cache across student architectures | `TIER=gate EPOCHS=4 ARCH=<w48 or w80> CHUNK=<2048 or 4096> nohup bash $SM/runbook-2026-10-16.sh arch > $SM/runbook-2026-10-16-arch.log 2>&1 & disown` | w48 ≈ 330, w80 ≈ 420–450, 4,096 continuation ≈ 220, epochs 7–8 ≈ 175 | Any gate "stop" ends the run; nothing may start after 15:00 that cannot finish its scoring by 20:00; otherwise leave the GPU idle |

Expected timeline (branch E, context or capacity verdict unknown): Q1 03:10–03:25, Q2 03:25–03:40, Q3 03:40–04:05, Q4 04:05–07:10, Q5 07:10–07:25, Q6 07:25–08:30, Q7 08:30–≈ 16:00 (w80) or ≈ 14:00 (w48), idle after. In branch P, Q7-P takes Q4's slot at ≈ 04:15 and Q5/Q6 drop to the next quiet window. Alternation: speed Q1, accuracy Q2, speed Q3, accuracy Q4, speed Q5, accuracy Q6–Q7 (daytime is accuracy-only because timing needs a quiet machine). Writers to start now: Sonnet medium for Q1, Q3, Q4 and the post-verdict report; Opus for Q5; Sonnet medium for Q7 once F and G are known (≈ 03:45).

**Is the 100-user gate scoring still relevant now that evaluation is faster?** Yes, keep it for go/stop, but score every epoch. The faster evaluation (EV01, 300 users in ≈ 2 min) is the in-training window-protocol evaluator: it scores 128-review samples and falls back to the reference path for Expertium variant heads (EV08). The gap we are closing is the full-history one (5.1: +0.1% in-window vs +5% full history), and full-history scoring is still one user per pass: ≈ 1.5 min of model time out of a ≈ 3.4 min ticket for 100 users, ≈ 11 min per label for 464. Moving every gate to 464 users would add ≈ 7.6 min per gate (≈ 26 min on a 4-epoch run with a gate after each epoch, about 9% of its training) for no change in decision: both gates passed with upper95 at −0.0038 to −0.0062 (curve CI half-width ≈ 0.0017, instant ≈ 0.0007), far from 0 and from the bands. The subset is less useful for the level vs the fork (its curve Δ% ran 0.3–1.1 pp below the 464-user value), so finals stay on 464 users. Recommendation: (1) keep 100 users for gates; (2) add a 100-user score after every epoch (≈ 3.4 min, ≈ 5% of an epoch), because the missing e3 point is why tonight's continuation rule must use e2→e4; (3) escalate a single gate to 464 users when it is close to its boundary (upper95 > −0.001, or projection within 0.002 of the target); (4) once S05 passes (Q5) and a 464-user label costs ≤ ≈ 4 min, move gates to all 464 users, so the last epoch's gate doubles as its final score.

## Backlog

| ID | Idea | Status | Next step |
|---|---|---|---|
| SM01 | Full-history held-out frontier vs fork int8 (464 users) | Done 2026-10-15 (430 users every label has) | See frontier table; 34 users unscored (time limits) |
| SM02 | w64 vs w32 and w64+KD vs w64, paired k = 3 screens | Done 2026-10-15: w64kd promising, w64 not | Scale-up blocked by the EPOCH_MIN fallback rule; main session decides (HANDOFF k=3 verdict) |
| SM03 | Scale-up of the winning recipe (gate tier, ≤ 6 h) | Done 2026-10-15 (main session override: w64kd, 3 epochs): `scale-w64kd-dec` +5.28% / +5.99% vs fork w8+s8, beats w32 decay, not a good candidate; packaged | Superseded by SM10: the next scale-up uses long-context chunks (5.7) |
| SM04 | CPU inference cost of fork vs w32/w64 (recurrent step, 1 thread, `--timing`) | Pending; top-3 priority 2026-10-16 (user asked for the CPU saving; only the 8.3x matrix-parameter estimate exists) | Queue Q1: `scripts/small_model_cpu_timing.py` (Sonnet medium), batch 1 and batch 256, f32 and int8, analytic MACs; quiet-night `--timing` ticket |
| SM05 | KD only on fork-unseen training users (teacher targets on its own training users may be overfit) | Untested idea | Needs a per-user KD mask in the trainer |
| SM06 | w80 + KD, longer decay (1 epoch, Expertium's choice) | Untested; re-scoped 2026-10-16: at 128 chunks it is pointless | Only at 2,048 (or 4,096) chunks and only if SM13 says capacity-bound: queue Q7 |
| SM07 | Window-protocol and collection evaluation of w64 checkpoints (`fast_evaluation` and `accuracy_collection` only build the w32 model) | Gap | Add a variant-aware evaluator hook |
| SM08 | Fork runtime port of the exported format (section 4, "Fork changes needed") | Specified, not started | User or Codex in the fork; acceptance = Rust vs `small_model_numpy.py` within 1e-5 |
| SM09 | Teacher-target cache for KD (section 5.2) | In use since 2026-10-15. ctx2048 run (accuracy lane): epoch 1 80,842 reviews/s (online + writes), epoch 2 102,143 (hits), below the hoped-for no-KD rate | S06 (queue Q3) measures the no-KD rate at 2,048 chunks; if the gap is the cache read, S07 (cache I/O, cross-architecture sharing) |
| SM10 | Long-context training (A09): in the 128-window protocol w64kd matches the fork (+0.1% LogLoss); the full-history gap is long-context behaviour (section 5.1) | Confirmed 2026-10-15 (screen B, paired k = 3). `scale-w64kd-gate-cached-ctx2048` (2,048 x 16, 4 epochs): gates e1 +3.86% / +5.04%, e2 +3.24% / +3.73% vs fork on the 100-user subset (projection +2.45% / +2.31%); epochs 3–4, decay and 464-user scoring running 2026-10-16 | Final verdict (F) at ≈ 03:10; continuation per Priorities, queue Q4; chunk 4,096 probe Q6 |
| SM11 | KD targets from the teacher's full-history replay instead of its 128-chunk predictions | Idea; value now depends on SM13 (context-bound → justified) | Design (Opus) only if SM13 says context-bound and Q6 says chunk length is exhausted; replay cost ≈ 3.8 h at batch 1, less with S05 |
| SM12 | Head temperature (Platt) folded into the export | Measured offline: closes ≈ 8% of the instant RMSE(bins) gap, LogLoss −0.1% | Cheap add-on once the recipe is fixed; fit on training users, never on the 464 |
| SM13 | In-window check at 2,048: student@w2048 vs fork@w2048 on the 100 gate users (the 5.1 decisive check at the new context) | Proposed 2026-10-16 | Queue Q2 (runnable as-is, ≈ 15 min); sets G: capacity-bound → w80 / more epochs, context-bound → SM16/SM11, pruning can be tried |
| SM14 | Extend `scale-w64kd-gate-cached-ctx2048` to 6 epochs + decay from e6 | Proposed 2026-10-16 (log-slope: ≈ −0.35 pp curve / −0.75 pp instant) | Queue Q4, branch E; new `runbook-2026-10-16.sh extend` (Sonnet medium), never `runbook-next.sh` |
| SM15 | Pruning step 1: w48 student (3 heads x 16, 195,949 parameters, 14x fewer than the fork) with the same long-context KD recipe, non-inferiority gates vs the ctx2048 run | Proposed 2026-10-16; starts only when the bar is met (user: accuracy first, then prune) | Queue Q7-P (or Q4 if F ≤ bar); needs the `w48` preset, gate `--reference-prefix`, ARCH-generic runbook stage |
| SM16 | Context beyond the chunk: carry the recurrent state (detached) from a user's chunk k to chunk k+1 (truncated BPTT), so training sees deployment-length histories | Idea 2026-10-16 (design, Opus); conflicts with the per-batch purity the KD cache relies on, so it needs a batch-order redesign | Only if SM13 says context-bound and Q6 says longer chunks no longer pay |
| SM17 | Epoch scoring after every epoch on the 100 gate users; 464 users for finals and for gates near their boundary; all-464 gates once S05 lands | Proposed 2026-10-16 (answer to the user's 100-user question, Priorities) | Built into the new runbook stages (Q4, Q7) |
