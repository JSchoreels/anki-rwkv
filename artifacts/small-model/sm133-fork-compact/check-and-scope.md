# Fork compact state (SM133): overflow check of the stored card state, and the implementation scope

Date: Fri 2026-11-20. Read-only check of the fork (`JSchoreels/anki`, `origin/main` at 4ae68fa86, before the history
rewrite). Throwaway worktree `wt/fork-overflow-check`, removed afterwards; nothing committed or pushed in the fork.
All test inputs were synthetic single-card histories. No dataset rows and no `testdata/*-vectors.json.gz` content were
used.

## 1. Verdict

- **The overflow is real in the fork.** It happens with the fork's default stored state (`cards_int4_notes_int8` =
  `w8+s4c8n`). The checks used both Big models, the show-time big model (SM91s2) and the show-time instant-only big
  model (SM99), through the fork's own runtime (`RwkvRuntime::review`). The fed-back card's int4 row scale starts at
  about 0.04. On the synthetic patterns it passes 1.0 after 184 to 557 answers and 65,504 after 517 to 1,621 answers.
  It then grows geometrically until float32 overflows. **At answer 2,226 to 11,201, depending on the pattern, the next
  answer fails.** The error is `invalid RWKV compact state: non-finite shift`, or for one Small run `non-finite or
  non-positive matrix scale`. The bulk warm-up replay does not return an error at all: it **panics** (`expect("packed
  state was validated on load")` in `ModuleState::into_decoded`).
- **It is not only Big.** With 1-day patterns the default format also fails on Small and Tiny. The failing models are
  the show-time retrain (SM92), the show-time instant-only model, stage 2 (SM98), the Tiny two-head model (SM106) and
  the smaller instant-only model (SM101). They fail at answer 4,618 to 10,566. On the mixed pattern, Small and Tiny
  stay finite up to 20,000 answers.
- **The mechanism is the same feedback loop as in the Python mirror.** The int4 rounding is fed back into the next step
  after every answer, so the error compounds. Float32 state stays bounded on the same inputs (max |value| <= 34).
  Int8 rows grow much more slowly or stay flat.
- **The compact format `w8+s4c8nhx8` removes the failure, but not the drift.** Its float16 row scales clamp at
  65,504, so a drifting card saturates: row values reach at most 7 x 65,504 = 458,528, and shift vectors use int8 codes
  with one float16 scale. Every run stays finite and answers to the end. But its predictions drift from the float32
  reference exactly as much as the default's did before it failed. Capping the row scale at 0.5 or 2 does not help
  either.

## 2. How the fork carries a card's state (code reading)

- Policy: the app default is `cards_int4_notes_int8` (`qt/aqt/rwkv_precision.py`). `RwkvInference::from_model` in
  `engine.rs` resolves it to `StatePrecision::Int4` for module 0 (card) and `Int8` for module 2 (note). Deck, preset and
  global stay float32.
- Per answer (`review_features` in `engine.rs`): the stored bytes are validated on input
  (`deserialize_module_state` calls `precision::decode`). Each layer then decodes them (`ModuleState::decoded`), runs
  in float32, and the new state goes straight through `ModuleState::compact(precision)` → `encode_matrix` (abs-max / 7,
  round half to even, clamp to ±7, float32 scale per row of `HEAD_SIZE` values) plus `write_shift` (float16). **So the
  state is stored quantized after every answer, and the quantized form is the only one the next answer sees.** This
  holds in-process too: `warm_up_states` keeps only the packed bytes (`layers` is empty). That answer's own output
  still uses the float32 update.
- Bulk replay (`bulk.rs`): the matrix stays float32 inside a chunk, but `round_matrix` (encode + decode) and
  `round_shift` run after every answer step. The result is bit-identical to the sequential path; the test
  `rwkv_state_precision_bulk_matches_answers_across_chunk_boundaries` checks this. Queries (`is_query`) read the state
  and never store it.
- Python mirror (`scripts/compress_quant.py` `reference_rwkv7_q`, `_quantize_rows` and `_round_shift`;
  `scripts/state_vq.py` `_reference_wkv_q`; branch `sm131`): the answer's output comes from the float32 update, the
  stored state is `_quantize_rows(next)`, and skip rows keep the state. This is the same arithmetic as the fork. The
  only difference is the fork's scale floor `f32::from_bits(1)`, which does not matter here. SM131's "9,465 module
  steps / 4,733 updates" is one query row plus one answer row per review, so it equals about 4,733 fork answers.
- Failure path: an overflowing card first produces an infinite float16 shift (the residual input of the next layer
  exceeds 65,504). Sometimes it produces an infinite matrix scale instead. `decode` rejects either one, so every later
  `review` or prediction for that card returns an error. Replaying the same history reaches the same state again.

## 3. Probe (throwaway test, uncommitted, in the removed worktree)

- `rslib/src/rwkv/overflow_probe.rs`, mounted in both engines: the big engine is `rwkv::alternate` under the default
  features, the small engine is `rwkv`. It loads the bundled weights through `RwkvRuntime::load_with_precision`. It
  answers one synthetic card N times through `RwkvRuntime::review` and feeds back all five stored states, as the app
  does. It parses the stored bytes to get the max row scale and the max |value| per layer, ignoring all-zero rows. In
  parallel it runs a float32 runtime on the same inputs, as the reference for |Δ retrievability|. The bulk path was
  also checked (`warm_up_reviews` in chunks of 250).
- Compact emulation: a test-only flag in `precision.rs` switches `encode_matrix`, `decode_matrix`, `row_bytes`,
  `write_shift`, `read_shift` and `round_shift` to float16 row scales and int8 shifts. The scale rule matches the Python
  `round_scale_f16`: clamp to 65,504, round to float16, and use 1.0 when the result is 0. The stored sizes match SM131
  exactly: Tiny 803 / 1,315 B, Small 1,521 / 1,315 B, Big 7,747 / 9,267 B per card / note. The default format measured
  1,055 / 1,567, 1,963 / 1,567 and 9,271 / 10,283 B.
- Patterns (deterministic, synthetic):
  - `mixed`: 15% Again, 10% Hard, 65% Good, 10% Easy, plus a 40-Again streak every 250 answers. Intervals are mostly 1
    to 3 days, some 1 h, some 7 to 10 days, and 10 min after an Again.
  - `again1d`: all Again at 1-day intervals.
  - `good1d`: all Good at 1-day intervals.
  - `again10m`: all Again at 10-minute intervals.
  - Every run had N = 20,000 answers. Release build, CPU, about 2 s to 60 s per run.

### Big, per-answer path, card int4 row scale (start about 0.04)

The scale columns give the answer at which the card's max row scale first passed that value.

| model | pattern | > 1 | > 65,504 | > 1e30 | default: fails at answer | compact: finite? | mean \|Δr\| vs float32: default (before failure) / compact / int8 |
|---|---|---:|---:|---:|---:|---|---|
| the show-time big model (SM91s2) | mixed | 374 | 1,621 | 9,303 | **11,201** | yes, saturated from 1,622 | 0.054 / 0.041 / 0.015 |
| the show-time big model (SM91s2) | again1d | 496 | 1,080 | 2,378 | **2,722** | yes, from 1,140 | 0.003 / 0.003 / 0.003 |
| the show-time big model (SM91s2) | good1d | 186 | 597 | 1,841 | **2,242** | yes, from 646 | 0.001 / 0.002 / 0.003 |
| the show-time big model (SM91s2) | again10m | 184 | 517 | 2,188 | **2,584** | yes, from 559 | 0.005 / 0.002 / 0.001 |
| the show-time instant-only big model (SM99) | mixed | 380 | 1,595 | 4,819 | **6,135** | yes, from 1,438 | 0.037 / 0.049 / 0.006 |
| the show-time instant-only big model (SM99) | again1d | 557 | 1,434 | 4,045 | **4,761** | yes, from 1,285 | 0.002 / 0.007 / 0.010 |
| the show-time instant-only big model (SM99) | good1d | 276 | 544 | 1,717 | **2,226** | yes, from 556 | 0.000 / 0.000 / 0.000 |
| the show-time instant-only big model (SM99) | again10m | 261 | 565 | 2,002 | **2,436** | yes, from 671 | 0.007 / 0.004 / 0.002 |

- Example, the show-time big model (SM91s2) on `mixed`, max row scale of card layers 1 / 2 / 3:
  - review 1: 0.04 / 0.04 / 0.04
  - review 500: 0.15 / 0.16 / 3.5
  - review 1,000: 0.18 / 4.3 / 722
  - review 2,000: 0.17 / 4,699 / 374,500
  - review 3,000: 1.1 / 2.4e6 / 6.5e8
  - review 5,000: 0.26 / 3.3e13 / 5.0e11
  - review 10,000: 1.1 / 2.3e32 / 1.0e24
  - review 11,201: fails

  Over the same reviews the float32 state stays between 0.7 and 4.9.
- Mean |Δr| vs float32 on `mixed` for this model, per window:
  - int4 default: 0.006 (reviews 1-100), 0.008 (101-200), 0.056 (201-300), then 0.035 to 0.059 for every window
  - int8: 0.0005, 0.0007 and 0.002, rising to 0.020 by 20,000
  - compact: tracks the default
- Note rows (int8) stay small in the default format: a peak note scale of 0.013 or less for both Big models, on all
  patterns.
- Int8 card rows (report only) never fail in 20,000 answers. On extreme 1-day / 10-minute patterns they still grow
  slowly. For the show-time big model (SM91s2): mixed 0.30, again1d 164, good1d 1,499, again10m 66,870 at answer
  20,000. For the show-time instant-only big model (SM99): mixed 0.034, again1d 29,000, good1d 168,000, again10m 13.5.
- Float32 (report only): the max |state value| is 5.4 (mixed) and at most 34 (good1d) after 20,000 answers. It never
  fails.
- Bulk warm-up path, the show-time big model (SM91s2), `good1d`, default format: it grows the same way (layer 3 is at
  5e32 by answer 2,000) and **panics** in the chunk that ends at answer 2,500. The compact emulation runs the same
  3,000 answers saturated and does not fail.

### Small and Tiny, default format, per-answer path, 20,000 answers

| model | mixed | again1d | good1d |
|---|---|---|---|
| the show-time retrain (SM92), Small two-head | finite (peak 0.58) | fails at 7,321 | fails at 10,566 |
| the show-time instant-only model, stage 2 (SM98), Small Instant-Only | finite (peak 0.84) | fails at 5,590 | fails at 9,267 |
| the Tiny two-head model (SM106) | finite (peak 0.34) | fails at 10,371 | finite, but 1.5e9 at 20,000 |
| the smaller instant-only model (SM101), Tiny Instant-Only | finite, but 37.5 at 20,000 | fails at 4,618 | fails at 6,898 |

- Int8 card rows on Small and Tiny: bounded on every pattern (peak scale 0.14 or less).
- Compact on Small and Tiny: always finite, and it saturates where the default would fail.
- Mean |Δr| vs float32 on `mixed`, default / compact / int8:
  - the show-time retrain (SM92): 0.013 / 0.018 / 0.002
  - the show-time instant-only model, stage 2 (SM98): 0.017 / 0.017 / 0.003
  - the Tiny two-head model (SM106): 0.014 / 0.014 / 0.003
  - the smaller instant-only model (SM101): 0.029 / 0.035 / 0.005

### Scale cap (report only)

Compact with the card row scale capped at 0.5 or 2, Big: always finite, and the card never exceeds 3.5 or 14. But the
mean |Δr| vs float32 on `mixed` was 0.038 (cap 0.5) and 0.036 (cap 2) for the show-time big model (SM91s2), against
0.041 uncapped. For the show-time instant-only big model (SM99) it was 0.033 and 0.031, against 0.049 uncapped. So a
cap only makes the state look tidy; the trajectory stays wrong.

## 4. Mechanism

- **Verified:** the quantize → dequantize feedback is the cause.
  - The fork stores, and the next answer reads, only the int4 reconstruction.
  - Float32 state stays bounded on the same inputs.
  - Growth is geometric at a roughly constant rate once it starts: about 0.8% per answer on `mixed` and about 4% per
    answer on `good1d` for the show-time big model (SM91s2).
  - The per-answer and the bulk paths behave the same, and the Python mirror shows the same numbers.
- **Inferred, not proven by an eigen-analysis:** the WKV update is
  `S' = S·diag(w) − (S·k̂)(a⊙k̂)ᵀ + v·kᵀ` (`update_recurrence_head_scalar`).
  - Each answer adds a rounding error of up to scale / 2 = abs-max / 14 per entry. That error is proportional to the
    row's own magnitude.
  - Directions with decay w close to 1, which the removal term does not touch, integrate this error. Because the error
    scales with the state, the integration compounds instead of averaging out.
  - Int8's error is 18 times smaller (abs-max / 254). That is below the decay for most patterns, so int8 drifts slowly
    or not at all.
- The deeper card layers (layers 2 and 3 on Big) grow first. Layer 1 grows later or barely.

## 5. Implementation scope: compact `w8+s4c8nhx8` as the fork's stored state for Tiny, Small and Big

The format was approved for the 11-28 release candidate. Bytes per card / note:

| size | default today | compact |
|---|---:|---:|
| Tiny | 1,055 / 1,567 | 803 / 1,315 |
| Small | 1,963 / 1,567 | 1,521 / 1,315 |
| Big | 9,271 / 10,283 | 7,747 / 9,267 |

Section 3 confirmed all three sizes in Rust. Compact fixes the hard failure, not the int4 drift on long cards. That
drift needs its own experiment (section 7).

### 5.1 Format (SM47 "h" + "x8", the SM129 / SM131 emulation)

- Card rows: int4 codes, 2 per byte (as today), with **one float16 scale per row** of `HEAD_SIZE` values.
  - The scale is `abs-max / 7`, clamped to 65,504, then rounded to float16 (round to nearest even).
  - If the float16 value is 0, the scale is 1.0. This covers all-zero rows and float16 underflow.
  - Codes are `clamp(round_half_even(x / scale), ±7)`.
- Note rows: the same with int8 codes and `abs-max / 127`.
- Shift vectors (time shift and channel shift, `D_MODEL` values): **int8 codes with one float16 scale per vector**
  (`abs-max / 127`, same scale rule), for cards and notes.
- Layout: unchanged apart from the scale width and the shift blocks (u32 count + 2 B scale + count B codes; an empty
  shift block keeps count 0).
  - Magic `ARWKVMODSTATE2`, then one precision byte, then u32 layers.
  - Per layer: the time-shift block, then u32 matrix count + rows, then the channel-shift block.
  - Proposal: keep the magic and add new precision-byte codes, e.g. 0x14 = compact int4 and 0x18 = compact int8
    (today 4 / 8). `deserialize_module_state` (`starts_with(precision::MAGIC)`) does not change, and old states stay
    decodable.

### 5.2 Fork files and functions that change

- `rslib/src/rwkv/precision.rs`:
  - `StatePrecision`: add the module precisions `Int4Compact` and `Int8Compact`, plus the policy `Compact` (cards
    `Int4Compact`, notes `Int8Compact`); update `parse`.
  - Update `limit`, `row_bytes`, `round_matrix` (unchanged API), `round_shift` (int8 per vector for compact, float16
    otherwise), `ModuleState::compact` (precision byte), and `write_shift` / `read_shift` (take the precision).
  - `decode`: precision-byte table, scale width, the code-validity checks with the new offsets, and the finite / > 0
    scale check, which stays.
  - `encode_matrix` / `decode_matrix`: scale width and the float16 rule.
  - Hardening that belongs in the same change: `decoded` / `into_decoded` panic through `expect`. The bulk path
    should return an `io::Error`, not crash.
- `rslib/src/rwkv/engine.rs`: `from_model` (policy → module precisions); the precision lists of the tests near
  lines 11127, 11252 and 12866.
- `rslib/src/rwkv/runtime.rs`: the `StatePrecision` mapping in `load_alternate` (the two engines each compile their own
  copy of `precision.rs`); the test `rwkv_runtime_restores_each_model_at_each_precision`.
- `rslib/src/rwkv/bulk.rs`:
  - `compact_shift` and the two inline `round_shift` sites (about lines 1463 and 1680). They already take
    `plan.precision`; the bit-identity with the sequential path must hold for the new shift rule.
  - The `compact` calls near lines 437 and 726 need no change.
- Benches that parse precision names: `app_speed_bench.rs`, `model_bench.rs`, `model_size_bench.rs`,
  `checkpoint_compression_bench.rs`, `runtime_precision_bench.rs`. Add the new value; `model_size_bench` hard-codes
  `Int4CardsInt8Notes`.
- `pylib/rsbridge/lib.rs`: passes the string through `StatePrecision::parse`; no logic change.
- `qt/aqt/rwkv_precision.py`: `RWKV_STATE_PRECISIONS` and the default. `qt/aqt/mediasrv.py` validates against that
  tuple.
- `ts/routes/deck-options/rwkv-model-estimates.ts`: the bytes table, which today holds 9,271 / 10,283, 1,963 / 1,567
  and 1,055 / 1,567. Also `RwkvModelOptions.svelte`, `lib.ts` and their tests (`lib.test.ts`,
  `rwkv-model-estimates.test.ts`, `ts/tests/e2e/deck-options.test.ts`), plus any ftl label.
- `proto/anki/deck_config.proto` (`state_precision` is a string): no change.
- Docs: the state-precision part of `docs/rwkv.MD` and `RELEASE.md`.

### 5.3 Versioning and existing stored states: rebuild, do not convert

- Stored RWKV states live only in the per-profile state cache (`rwkv-state-cache/`: snapshot, SQLite delta store,
  checkpoints). That cache is derived data that can always be rebuilt from the revlog. Its replay key
  (`_rwkv_replay_semantics_key` in `qt/aqt/rwkv_scheduler.py`) hashes the `statePrecision` string.
- Recommendation: introduce the profile value `compact` and make it the default.
  - `state_precision_for_profile` maps a saved `cards_int4_notes_int8` to `compact`, so no profile keeps the
    overflowing writer. `StatePrecision::parse("cards_int4_notes_int8")` can stay as an alias for old callers.
  - The new string changes the replay key, so every cache rebuilds on first start (one full replay). As a belt and
    braces measure, also bump `_RWKV_STATE_CACHE_VERSION` 13 → 14.
- Why no conversion:
  - Re-quantizing a stored state (float32 → float16 scale, float16 → int8 shift) does not give a fresh replay bit for
    bit.
  - A card that has already drifted would keep its drift.
  - A card that has already overflowed cannot be decoded at all.
- The decoder still accepts precision bytes 4 / 8, so in-session undo states, spills (`ARWKVSTASH1`) and old
  checkpoints read until the rebuild replaces them.
- Keep `f32` and `int8` as options, unchanged.

### 5.4 Tests that need new (synthetic) vectors

- `precision.rs` unit tests:
  - the codes / round-half-to-even test with compact rows;
  - the float16 clamp (row max 1e6 → scale 65,504, codes clamped) and the underflow case (row max 1e-9 → scale 1,
    codes 0);
  - the corrupt-input rejection test, whose `scale_start` offset changes to `MAGIC + 1 + 4 + 4 + 2 + D_MODEL + 4`;
  - "loading does not re-quantize".
- `precision_tests.rs`:
  - add `Compact` to `PRECISIONS`, which covers bulk vs sequential, queries / undo, the queue-scoring invalid state,
    and checkpoints / spills;
  - update `stored_layer_bytes` and `rwkv_state_precision_stored_sizes_of_the_small_and_tiny_models`: compact card +
    note = Small 2,836 B, Tiny 2,118 B;
  - add a **Big size test** on the big engine: 7,747 / 9,267 B;
  - update the precision-byte asserts of `rwkv_state_precision_mixed_packs_cards_as_int4_and_notes_as_int8`.
- New long-card regression test, synthetic, one per size: for example 5,000 answers at 1-day Good and 1-day Again
  (section 3). Assert that it runs without an error or panic and that every output is finite. Document the saturation
  by asserting that row scales stay at or below 65,504. Pick lengths so the test stays a few seconds in debug, or mark
  it `#[ignore]` and run it in release.
- New parity file, for example `rslib/src/rwkv/testdata/compact-state-parity.json.gz`, made only from synthetic
  histories (5.5).
- Python tests: the precision parameter lists in `qt/tests/test_rwkv_model.py` and
  `qt/tests/test_rwkv_inference_process.py`. The TypeScript tests are listed in 5.2.
- The existing PyTorch vector tests (`legacy_tests.rs`, `sm40_tests.rs` and the `testdata/sm*-vectors.json.gz` files)
  compare float32 outputs, so this change does not touch them. They hold real-user histories that the history rewrite
  purges, so **do not extend them**. The new compact vectors must not reuse their inputs.

### 5.5 Parity with the Python mirror

- Python reference, **on branch `sm131` (worktree `wt/sm131`), not yet on anki-rwkv `main`** (`sm129` and `sm131` are
  not merged):
  - `scripts/state_vq.py::int_row_quant(bits | F16_SCALE)`, with `scripts/small_model_wkv.py::round_scale_f16` and
    the constants `F16_SCALE = 256` and `HEAD_SCALE = 512`;
  - `scripts/compress_quant.py::_quantize_rows` / `_round_shift`, the fork-kind path;
  - `scripts/small_model_heldout.py::parse_variant`: `w8+s4c8nhx8` is cards 4|F16, notes 8|F16, shifts 8|F16.
- Format text: `artifacts/small-model/sm47/sm47-summary.md` ("h" = float16 row scales, "x8" = int8 shift codes with
  one scale per vector, float16 with "h").
- Byte-layout reference: `artifacts/small-model/sm131/sm131_bytes.py`. Its block layout is a model of the fork: it
  writes three u32 byte lengths per layer up front, while the fork writes u32 element counts before each block. The
  totals agree, but the bytes do not, so update it to the final Rust layout before using it for a byte-exact check.
- Step 1, codec, bit-exact both ways:
  - The Python side generates synthetic float32 states: normal, zero, tiny and huge rows, as in `random_layers`, with
    a non-symmetric matrix so a row/column transposition shows up. It quantizes them with `int_row_quant`, encodes
    them and writes `(input, reconstruction, bytes)`.
  - A Rust test re-encodes the inputs and must produce the same bytes. It decodes the bytes and must get the same
    values bit for bit.
  - This also pins the row orientation. A fork row is `HEAD_SIZE` contiguous values (value index = row); it must
    match the last axis of the Python `[B, H, K, K]` state.
- Step 2, model:
  - Take about 3 synthetic cards × 300 answers with mixed grades and intervals, plus a shared note. Run the Python CPU
    scorer on the Big, Small and Tiny fork models with `fp32+s4c8nhx8`. Use float32 weights so the weight
    quantization is not a confound; the fork's `weights.f32.bin` is float32.
  - Store per-answer probabilities and the final card / note bytes.
  - Rust gate: probabilities within 1e-4 per answer. Report the maximum and the count above 1e-5; a rare int4 code
    flip from float32 summation order is allowed. Card / note codes must be equal in at least 99.9% of entries.
  - Do not use a drifting long card for strict parity, because it is chaotic. Use it only for the finite / saturated
    check.
- Step 3 (optional): rerun one scored user (not stored) in both stacks and compare the LogLoss within 0.01%.

### 5.6 History rewrite constraint

The fork's history rewrite is prepared but not yet force-pushed. The implementation branch must be cut from, or rebased
onto, the **rewritten** `main` after the force-push, and must **not be pushed before then**. Pushing earlier would
reintroduce purged objects or fork the history. This check ran on pre-rewrite `origin/main` 4ae68fa86. No new file may
contain real-user review rows; the parity vectors in 5.5 are synthetic.

### 5.7 Effort estimate

| part | effort |
|---|---|
| Rust codec + plumbing (`precision.rs`, `engine.rs`, `runtime.rs`, `bulk.rs`, benches, panic → error) | 0.5-1 day |
| Rust tests: sizes, the Big size test, codec edge cases, long-card regression; both feature sets (`cargo test -p anki --lib rwkv::` with the default features and `--no-default-features`) | 1 day |
| Python parity: synthetic generator, layout-exact reference encoder, vector file, Rust gate | 0.5-1 day |
| App / UI / strings / docs, cache-version bump, Python and TypeScript tests | 0.5 day |
| Review, CI, a speed check (`app_speed_bench` `state` op) | 0.5 day |
| **Total** | **about 3-4 agent-days**, about 2 calendar days if the parity part runs in parallel |

This fits the RC checks (~11-25) if it starts by 11-21 / 11-22. It is gated on the rewrite force-push (5.6).

## 6. Evidence files

The probe source and the throwaway `precision.rs` patch were saved outside the repositories, in the session scratchpad:
`C:/Users/jonat/AppData/Local/Temp/claude/eefeeb6d2df69e2b/41a2947b-8da0-4ee6-a033-bc96eb42a697/scratchpad/overflow_probe.rs`
and `.../scratchpad/probe-precision.diff`, with the logs in `.../scratchpad/probe2`, `probe4`. The scratchpad can be
cleared. The worktree was removed.

## 7. Follow-up, not part of this scope

Int4 card rows drift on long cards on every size: the mean |Δr| vs float32 reaches 0.02 to 0.06 from about 200 to 500
answers on these synthetic cards, against 0.002 to 0.015 for int8 rows. Compact keeps that drift. SM131's "free" result
compares compact with the default, and both share the drift. Proposed experiment: score the subset of reviews on cards
with at least 300 prior answers in the Python mirror, comparing float32, int8-card and compact. The result would decide
whether Big, or every size, should store int8 card rows (Big card 15,415 B in the published format) or another
long-card fix.
