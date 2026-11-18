# Small model: first pruning step (design, 2026-10-17)

Design only: nothing enqueued, no code written. It starts when the w64 student meets bar P (SM15: "starts only when the bar is met"). It proposes replacing SM15's first step (w48 from scratch) with a depth cut of the context streams, which is cheaper. The user decides.

**Starting point.** Parent = the first w64 checkpoint that meets P (<= +1.0% curve AND <= +1.0% instant LogLoss vs fork w8+s8, 464 users, review-weighted). Notation: `P_dec` = its decayed model (the reference), `P_e` = the undecayed epoch it was decayed from (the pruning source). Today the best is `s-w64kd-gate-ctx16384c-dec9-L32k` at +1.01% / +1.01%. e10 at 32k is running (`dec10` result about 18:30). The parent path cost about 21 GPU h: e1-e6 at 2,048 (66 min each), e7-e8 at 8,192 (about 2 h each), e9 at 16,384 (4.5 h), e10 at 32,768 (about 6 h), plus decays and scoring.

**Where w64's 331,511 parameters sit** (CPU count of `expertium.variants.build("w64")`):
- 13 time-mixer layers, 254,775 in total (77%): card 2, deck 4, note 1, preset 3, user 3. Each layer has about 19.4-19.7k parameters: 4 x 64 x 64 projections plus norms and lerps, and LoRAs of 1.7-2.0k.
- 4 kept channel mixers (card:0, deck:0, deck:3, note:0), 8,384 each, 33,536 in total (10%).
- Input encoder (92 -> 128 -> 64): 20,416 (6%). Heads: 22,739 (7%).

Two points matter for the choice:
- **Batch-1 CPU inference is dispatch-bound** (SM04: w64 has 8.5x fewer MACs than the fork but is only 1.12x faster). Per-review latency in a plain runtime follows the number of layers (ops), not the width.
- **Training at long chunks is recurrence-bound** (S06), so epoch time scales with the number of sequential layers.

## 1. Options

| | Option | Params (vs parent) | Card / note state int8 | Code exists / to write | GPU h | Expected cost vs parent (curve / instant) |
|---|---|---:|---|---|---:|---|
| a | SM15 as written: w48 (3 heads x 16, mults 2) from scratch, full window curriculum | 195,949 (-41%) | 2,304 / 1,152 B (-25%) | w48 = `w80_config(heads=3, mult=2)`, not a preset yet; `cont`/`longdecay` stages hard-code `scale-w64kd` names and `expertium:w64:`; w48 kernel parity ticket (`check_wide_kernels.py`, about 3 min) | **22-26** through e10 + 32k decay + scoring (16-19 through e9 + 32k decay) | +0.1-0.3% / +0.3-0.8% (capacity), more if the curriculum is cut short |
| b1 | Width cut: slice w64 -> w48 (drop 1 head and 16 residual channels; heads/encoder 128 -> 96), then KD fine-tune at long windows | 195,949 (-41%) | 2,304 / 1,152 B | Slicing code (LN, GroupNorm, `rkvdag_lerp`, `bonus`, k/v scale, LoRA A/B, W_r/k/v/o, cmix, encoder, head_w/head_p, GRU weights), about 200 lines; w48 preset; kernel parity ticket | 6-9 (1 epoch at 16k or 32k + 32k decay + scoring) | Same capacity floor as (a); the init helps only partly, because every LayerNorm then normalises over 48 of 64 channels |
| b2 | **Depth cut of the context streams**: drop the *trailing* layers of deck / preset / user (D9 = d2 p2 u2, D8 = d2 p1 u2 or d2 p2 u1, D7 = d2 p1 u1), then KD fine-tune | D9 244,232 (-26%), D8 224,507 (-32%), D7 204,782 (-38%) | unchanged (3,072 / 1,536 B) | Preset grammar + key-drop converter (about 30 lines); no kernel change (d = 64, 4 x 16) | **2.5-4** (0.5 epoch + 32k decay + scoring) | D9 +0.1-0.4%, D7 +0.2-0.8% per head (no outside evidence; the CPU screen bounds it) |
| b3 | Strip the remaining channel mixers (card:0, deck:0, note:0) | -25k (-7.6%); D7 + this = 179,645 (-46%) | unchanged | `strip_cmix` field exists (exact branch removal); needs a named preset for the scorer | as b2 (combined) | small to moderate; Expertium already stripped 9 of 13 |
| b4 | LoRA ranks / 2, or head and encoder mults 2 -> 1 | -3.3% / -5.8% | unchanged | new preset fields | as b2 | small, but too little saving for a run |
| c | Distil a w48 from the best w64 (student as teacher; `teacher_architecture: "w64"` already works in `Teacher`), w64 slice init, long windows only | 195,949 | 2,304 / 1,152 B | b1's slicing code + teacher swap; new KD cache (new teacher sha, cold at every chunk) | 7-10 | b1 ± 0.1 points |

Reasoning per option:
- **(a)** Expertium's width ladder at matched training (his split): d80 -> d64 (1.7x fewer parameters) cost +0.01% / +0.15%, and d128 -> d64 (7.7x fewer) +0.17% / +0.65%. Each step costs more at the small end, and it lands on instant. Our instant gap is uniform across users (HANDOFF 21:40), i.e. a capacity/fit gap, so width is the head at risk.
  - Same 13 sequential layers, so long-chunk epochs are barely faster than w64's.
  - `architecture` is part of the KD-cache hash, so every chunk length starts cold (+15-30% on its first epoch).
  - Decided by: w48-dec vs `P_dec`, paired.
  - Risk: three or more GPU evenings for a +0.3-0.8% instant cost, which already fails P on its own.
- **(b1)** Pruning plus distillation usually recovers most of the slice's damage with a small share of the original training, but our recipe's gains came from 16-32k windows, and those are the slow epochs.
  - Decided by: b1-dec vs `P_dec`.
  - Risk: subtle slicing bugs (residual subset vs head subset), and a d = 48 kernel path that has never been exercised.
- **(b2)** The cut keeps the card and note streams intact. They hold the per-entity state and the curve gap (72% of it is on learning-step reviews, feature screen). Deck/preset/user layers already had their channel mixers stripped in iter 53.
  - Dropping *trailing* layers keeps the interleaved order an exact subsequence: `layer_steps(interleave_schedule(depths))` is round-major by layer index. The init is therefore the parent with residual branches removed, with no re-ordering or slicing.
  - It is the only option that cuts batch-1 CPU latency: 13 -> 7-9 layers.
  - It trains about 1.4-1.8x faster per review, because fewer layers run in sequence.
  - Risk: these streams carry the cross-card context that 16-32k windows bought (+0.13 points per head). The CPU screen measures that before any GPU.
  - Decided by: b2-dec vs `P_dec`.
- **(c)** Teacher-assistant KD can help at large capacity gaps (fork -> w48 is 14x). However, the w64 teacher caps the child at the parent (+1.0%) and lacks the fork's learning-step knowledge. It does make the targets free of review type.
  - Only worth it as a paired arm next to b1 (same init, schedule and users) if b1 lands in EXTEND (section 2).
- **(d) Also found:**
  - (d1) S07(b): share the w64 KD cache with pruned children. The pruned students build the same batches, and `zero_features`/interleave act inside the model. This removes the cold-cache cost (about 0.5-0.8 h per experiment) and a few GB of new cache per chunk configuration.
  - (d2) A zero-training CPU screen of the cuts (section 2.1) ranks the options for free.
  - (d3) int4 weight PTQ halves the file but does not cut parameters or CPU, so it comes after pruning, not instead of it.

## 2. Recommended first experiment: E1 = depth cut D7 (b2), with a CPU screen first

### 2.1 CPU screen (no GPU; CPU accuracy work runs freely, at most 8 threads while a GPU epoch is preparing)
- Convert `P_dec` with every cut and score zero-shot with `small_model_heldout.py --device cpu` on the 68 feature-screen users (`feature-screen/shard0.json` + `shard1.json`), w8+s8.
- Write to a scratch jsonl (`$SM/prune-screen/screen.jsonl`, never `heldout.jsonl`).
- Arms: parent, D9, D7, D7-nocmix, and single-stream cuts d2p3u3, d4p1u3, d4p3u1.
- Report with `small_model_report.py --jsonl <scratch> --reference <parent>/w8+s8`.
- Choice rule (pre-registered):
  - The GPU arm is D7.
  - If one single-stream cut accounts for > 60% of D7's zero-shot increase on either head, keep that stream one layer deeper (D8).
  - The zero-shot cost is a rough upper bound on the post-fine-tune cost, not a verdict.

### 2.2 GPU arm (one evening slot)
1. Prune `P_e` (undecayed, as the parent's own decays branch from it) to D7. Use a fresh AdamW (5-update warmup from `update_offset` 0).
2. Fine-tune 0.5 epoch (seed 2) at `P_e`'s chunk (32,768 x 1 after e10, 16,384 x 2 after e9). Constant 7e-4, KD alpha 0.9.
3. Run a 0.25-epoch `one_minus_sin` decay at 32,768 x 1, seed 1. These are the same users as the parent's 32k decays, checked by the `longdecay` paired-user assertion. Decay alpha 0.5.
4. Score on 464 users (w8+s8 + fp32).
- No undecayed gate: constant-LR checkpoints are too noisy (e9 undecayed was +2.93%, its decay +1.03%).
- **Budget:**

  | Step | Minutes |
  |---|---:|
  | 2 smokes | about 5 |
  | Fine-tune | 100-150 |
  | Decay | 40-65 |
  | Scoring | 15-25 |
  | **Total** | **about 2.5-4 h** (cap 4.5 h) |

  - Low end = warm cache with S07(b), high end = cold.
  - Stop rule: cancel if the fine-tune runs below 15k reviews/s.

### 2.3 Non-inferiority rule (pre-registered; paired user bootstrap as in HANDOFF / `small_model_report.py`)
- **Statistic.** Δ%_h = (LL_child - LL_P_dec) / LL_P_dec per head (curve, instant).
  - Review-weighted benchmark LogLoss, w8+s8 labels, on the users every label has.
  - 95% CI by resampling users with replacement (2,000 draws, review-weighted).
- **PASS** (adopt the cut): the upper 95% bound of Δ% is <= **+0.25%** on **both** heads. That is about +0.00075 curve / +0.00069 instant LogLoss, so the child stays within about +1.25% of fork w8+s8.
  - Why 0.25%: a quarter of bar P's 1-point budget for a >= 30% parameter cut.
  - Paired CIs of related checkpoints were about ±0.0003 curve / ±0.00005 instant (HANDOFF 08:07, 09:30), so one run resolves this margin.
- **EXTEND**: the point estimate is <= +0.25% on both heads but an upper bound is above it. Run one more 0.5-epoch segment from the fine-tuned checkpoint plus a new decay (about +2.5 h), then re-apply the rule once.
- **FAIL**: the point estimate is > +0.25% on either head. Stop D7 and go to section 4.
- **Margin is cumulative.** Later cuts are judged against the same `P_dec`, never against the previous child (no salami slicing).
- **Also reported:**
  - Child vs fork Δ% with CI, RMSE(bins), light/mid/heavy tertiles.
  - int8 cost (w8+s8 - fp32 <= +0.02 points, else investigate).
  - reviews/s for every ticket.
- **Caveat.** The child gets 0.75 epoch more training than `P_dec`, so Δ is the shipping cost, not the pure pruning cost. If Δ < 0, the parent is under-trained: run the matched parent control (same fine-tune + decay on w64, about 4 h) before drawing conclusions.

### 2.4 Code to write (Sonnet medium, about 3 h)
Work on a separate branch/worktree, then merge into `small-model` only **between** GPU tickets (SourceWatch kills a running epoch). Use absolute paths in every queued command (SM24 failed on a relative one).

1. **`expertium/variants.py`**: preset grammar `w64-d{D}p{P}u{U}[-nocmix]` (D 1-4, P 1-3, U 1-3).
   - Built from `preset("w64")` with `replace`: per-stream `n_layers`/`total_layers` for deck/preset/user, and `strip_cmix` filtered to kept layers (`-nocmix` adds card:0, deck:0, note:0).
   - Width, heads, mults, LoRAs, GRU head, interleave and `zero_features` stay unchanged. `w64` itself is unchanged (331,511).
   - Asserted counts: d2p2u2 244,232; d2p1u2 / d2p2u1 224,507; d2p1u1 204,782; d2p1u1-nocmix 179,645.
   - The name has no `:`, so `small_model_heldout.py --model L=expertium:<arch>:<pth>` works as is.
2. **`scripts/small_model_prune.py`** (new):
   - `depth --src --arch --out`: drop `rwkv_modules.<s>.blocks.<k>.*` for k >= the new depth, strict-load into `build(arch)`, and check every kept tensor is bit-identical.
   - `make-run --parent-exp --epoch --arch --name`: a synthetic experiment under `artifacts/accuracy/experiments/<name>/`. It holds:
     - run.json/experiment.json copied from the parent, with `options.architecture = arch` and `pruned_from`;
     - `run/epoch-NN/expertium/E89-F65e-F77a/`: the pruned `master.pth`, a fresh `optimizer.pth` (`CatchUp(options, {}, []).optimizer(model).state_dict()` on CPU) and `result.json` (`status complete`, `update_offset_end` 0);
     - a pointer to the parent's label-db if `run_case` needs it (verify in the dry run).
     The parent is only read; the command refuses an existing name.
   - `ft-prepare --run --epoch --out --fraction 0.5 --seed 2`: call `expertium_decay.prepare`, then delete `full_run.lr_schedule` from protocol.json. Result: constant 7e-4 after warmup and KD alpha 0.9.
     **Trap**: `--final 1.0` would also give a constant LR, but `LearningRate.decaying()` is then true for the whole run, so KD would use `decay_alpha` 0.5.
   - `promote --ft <dir> --run <run> --epoch NN+1`: copy master/optimizer/result into the run as epoch NN+1, so `expertium_decay.py prepare --source-run` can decay from it.
   - `--self-test` (CPU, < 5 min), five checks:
     - child forward == parent forward with the dropped blocks replaced by identity (<= 1e-6, one user <= 3k reviews, fp32, eval);
     - strict loads and parameter counts;
     - the fresh `optimizer.pth` loads and its first step equals a fresh AdamW's;
     - `ft-prepare` output has no `lr_schedule` and `CatchUp.loss` sees `decaying` false;
     - `zero_features == (22,)` and the input layer is `Linear(92, 128)`.
3. **`artifacts/small-model/runbook-prune-depth.sh`**, modelled on `runbook-2026-10-16d.sh` (`q()`, `need()`, `DRYRUN=1` with a stubbed queue).
   - Env: `ARCH`, `PARENT_EXP`, `PARENT_EPOCH`, `PARENT_LBL`, `FT_FRACTION=0.5`.
   - Stage `screen` (CPU, not queued): section 2.1, writes `$SM/prune-screen/report.md`.
   - Stage `e1`:
     1. make-run, then ft-prepare;
     2. ticket smoke && run (`--expected-minutes 150`);
     3. promote, then decay prepare at `{"chunk_reviews": 32768, "batch_chunks": 1}` seed 1, with the paired-user assertion against the parent's 32k decay;
     4. ticket smoke && decay (65);
     5. 464-user scoring of `s-w64kd-gate-prune-<arch>-dec` (w8+s8 and fp32) (25), plus a resume ticket only if users remain;
     6. decision helper: writes `$SM/prune-decision-<arch>.json` with PASS / EXTEND / FAIL and both heads' Δ% and CIs.
4. **Optional S07(b)** (Sonnet + Opus review, because a wrong key would silently mix targets): `distillation.cache_as_architecture: "w64"`.
   - When set, `Teacher` hashes `batch_options` with `architecture` replaced by that value. `distillation` is TEACHER_BLIND, so the existing w64 hashes do not change.
   - Add a check in `check_kd_cache.py`: w64 and w64-d2p1u1 build bit-identical PreparedBatches for 2 users.

## 3. What must stay true for deployment
- **Precision.** Decide on **w8+s8** only: per-output-channel int8 weights and int8 card/note WKV state, as the fork. fp32 is scored only to measure the int8 cost. Same 464 users, same `fork/w8+s8` reference, review-weighted.
- **Review type stays out.** Every pruned preset inherits `zero_features=(22,)` (the self-test asserts it). The exported `architecture.json` keeps `"zero_features": [22]` and the NumPy reference zeroes it. No substitute that encodes SM-2 state (learning/relearning flags) may be added.
  - The fork teacher still reads review type, but only for training targets, never as a deployed input.
- **Same 92-input layout.** `features2card` stays `Linear(92, 128)`. No SM27 columns in the same step (separate track, separate comparison).
- **Export parity.** `small_model_export.py package --arch <arch>` (generic: per-stream `n_layers`, stripped layers, interleaved `layer_steps`) must pass NumPy parity (about 1e-5 fp32), with file sizes recorded.
- **Fork side.** It needs only items 1-3 of "Fork changes needed" (data-driven `MODULE_LAYERS`). No new head size or width, unlike w48. Card/note state stays 3,072 / 1,536 B int8.
- **CPU claim.** Measure it with `scripts/small_model_cpu_timing.py` in a quiet window. Expectation: batch-1 latency falls roughly with the layer count (13 -> 7).

## 4. After E1
- **D7 PASS**: E2 = D7-nocmix (-46%) from the D7 child, about 2.5 h, same rule against `P_dec`. Then export, parity, CPU timing.
- **D7 FAIL**: E2 = D9 (or the D8 the screen favoured), about 3 h.
- **All depth arms FAIL**: width cut b1 (slice + 1 epoch at 16k + 32k decay, 6-9 h, plus the kernel parity ticket). Run (c) as its paired teacher arm only if b1 is EXTEND.
- **(a)** only if b1 cannot recover: 22-26 GPU h, which is several evenings, and needs the user's go.
