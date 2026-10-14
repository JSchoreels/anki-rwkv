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

## Backlog

| ID | Idea | Status | Next step |
|---|---|---|---|
| SM01 | Full-history held-out frontier vs fork int8 (464 users) | Queued 2026-10-15 | Report after the runbook |
| SM02 | w64 vs w32 and w64+KD vs w64, paired k = 3 screens | Queued | Report; apply the HANDOFF threshold |
| SM03 | Scale-up of the winning recipe (gate tier, ≤ 6 h) | Planned, triggered by threshold | HANDOFF section "Scale-up" |
| SM04 | CPU inference cost of fork vs w32/w64 (recurrent step, 1 thread, `--timing`) | Pending | Small timing ticket with the RNN classes (`expertium.variants.build(..., rnn=True)`, fork `srs_model_rnn`) |
| SM05 | KD only on fork-unseen training users (teacher targets on its own training users may be overfit) | Untested idea | Needs a per-user KD mask in the trainer |
| SM06 | w80 + KD, longer decay (1 epoch, Expertium's choice) | Untested | After SM03 |
| SM07 | Window-protocol and collection evaluation of w64 checkpoints (`fast_evaluation` and `accuracy_collection` only build the w32 model) | Gap | Add a variant-aware evaluator hook |
| SM08 | Fork runtime port of the exported format (section 4, "Fork changes needed") | Specified, not started | User or Codex in the fork; acceptance = Rust vs `small_model_numpy.py` within 1e-5 |
