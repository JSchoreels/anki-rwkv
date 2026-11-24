# HANDOFF: Rank-2 to rank-4 low-rank states in the fork (SM151)

Branch `sm151` (this worktree, `C:/Users/jonat/workspace/wt/sm151`, from origin/main). Fork branch `rc-lowrank`
(worktree `C:/Users/jonat/workspace/wt/fork-lowrank`, from origin/rc-2026-11 at aecc5b09d). No GPU jobs in this
experiment. Read `artifacts/small-model/sm151/design.md` first, then the fork's
`docs/rwkv-lowrank-state-2026-11-23.MD` and `rslib/src/rwkv/lowrank.rs`.

## User decisions in force (11-23)

- Default for Big, Small and Tiny: T24 = rank 2 below 16 answers, rank 4 from 16, int8 factors (`w8+lr2.16.4q8x8`).
  The schedule stays a parameter; plain rank 2 (`w8+lr2q8x8`) keeps a parity test.
- "ok let's integrate in anki and push when finished": once everything below is done and green, merge `rc-lowrank`
  into `rc-2026-11` (fetch `origin rc-2026-11` explicitly, merge, re-run the full Rust suite on the merged tree,
  push). Never fork `main` or a release branch, no tags or releases, never force-push, never bypass hooks. If a check
  fails or a decision is needed: do not merge, stop and report.
- Speed runs: before each timing run check `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
  C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status` (a GPU job "Smaller shift vectors and factors on
  T24 states (SM152)" may run tonight); time the int8 base and T24 back to back; record whether a ticket ran; if so
  give absolute reviews/s as "under load" and list a clean re-time as a follow-up. Do not wait for the GPU job.
- The factored-update idea (ideas-1123 "Idea 1"): not adopted, reason in design.md ("Note on a factored update");
  repeat it in RESULT.md with the measured µs.

## Build setup on this PC (already done in fork-lowrank)

- Toolchain: the rustup shim `C:/Users/jonat/.cargo/bin` first in PATH (1.97.1 MSVC, from rust-toolchain.toml); the
  GNU 1.99 in Program Files cannot build windows-sys (no assembler for dlltool).
- `out/` was seeded from `wt/fork-rc-state/out` (copied with timestamps, `cp -a`), protoc copied to
  `out/extracted/protoc` from `wt/anki-tiny/out/extracted/protoc`, and the `ftl/core-repo` and `ftl/qt-repo`
  submodule contents copied (without `.git`) from `wt/fork-rc-state/ftl/` (same commits). None of this is tracked.
- Commands (Git Bash):

      cd C:/Users/jonat/workspace/wt/fork-lowrank
      export PATH="/c/Users/jonat/.cargo/bin:$PATH" CARGO_TARGET_DIR=$PWD/out/rust
      cargo test -p anki --lib rwkv::                       # rwkv module, default features
      cargo test -p anki --lib --no-default-features rwkv:: # the other engine as primary
      cargo test -p anki --lib                              # full rslib suite (base: 402/402 as the brief says)

## State at this handoff

See the "Status" section at the end (updated at each milestone).

## Remaining work (in order)

1. **End to end, Big** (benchmark per-user files only, never the user's own collection; outputs under
   `C:/Users/jonat/workspace/wt/sm151/work/` (never committed) except aggregates): adapt
   `fork-lowrank/qt/tools/rwkv_compact_state_big_vectors.py` into a scratch driver that, for a few users of
   `wt/sm148/artifacts/small-model/sm148/users-smoke.json` (2388, 4676, 1866), builds the scorer's show-time inputs
   (`e2s-query`, deleted cards neighbour) from `cc.data_path()` instead of the synthetic folder, and writes features +
   ids + labels per row plus the scorer's outputs at fp32 / compact int8 / T24 / Q2. For T24 / Q2 use the scorer's
   fork-model low-rank path: `PYTORCH_JIT=0`, `state_lowrank.install_fork(model)`, a `LowRankCodec` from
   `state_lowrank.parse_sm148("lr2.16.4q8x8")` with `rows_bits` / `shift_bits` = compact int8, `state_lowrank.set_fork`,
   and `cq.set_state_streams` with compact int8 for card and note (the shifts). Then an ignored Rust test (pattern:
   `legacy_tests.rs` `rwkv_legacy_compact_state_matches_the_scorer`, env var for the vector file) runs the release
   Big models (show-time big model (SM91s2) and show-time instant-only big model (SM99)) at `compact_lowrank_int8`,
   `compact_lowrank_int8:2` and `compact_cards_int8` and writes outputs; compute per-user LogLoss (curve and instant)
   on the scored rows and compare with the SM148 rows of the same user, label `w8+lr2.16.4q8x8` / `w8+lr2q8x8` / REF
   (`wt/sm148/artifacts/small-model/sm148/cpu/smoke-cpu-rows.jsonl`, `gpu/sm148-rows-732.jsonl`). Gate: Rust vs the
   scorer's per-row outputs within the discrete-codec rule (per user |delta LogLoss| <= 3e-4, mean signed delta <=
   1e-4), and Rust T24 minus Rust int8 matching the SM148 T24 minus REF per user within the same rule.
2. **End to end, Small and Tiny:** the scorer path is the eager training model with `state_vq` hooks
   (`state_lowrank.configure_expertium_sm148`); features from the same data pipeline as the package
   `make_testvectors.py` (`artifacts/models/sm72e25/testvectors/make_testvectors.py`). If this cannot be done within
   budget, say so in RESULT.md and rely on the codec parity (both head sizes) plus the bulk / per-answer identity.
3. **Speed:** single thread, release build, per-review time of the state write + read per size, int8 base
   (`compact_cards_int8`) vs T24 back to back: report reviews/s and µs per review. A suitable harness:
   `rslib/src/rwkv/model_size_bench.rs` (`ANKI_RWKV_SIZE_BENCH_PRECISION`, `ANKI_RWKV_SIZE_BENCH_RUNTIMES`), with
   `RAYON_NUM_THREADS=1`; also a micro-benchmark of `lowrank::encode_layer` + `decode_block` vs the int8 row codec.
4. **Long cards:** `cargo test --release -p anki --lib rwkv_long_card_full -- --ignored --nocapture`
   (`RWKV_LONG_CARD_THREADS=8`, about 2 to 5 minutes per policy) now includes `compact_lowrank_int8`; record the
   drift table rows for it in RESULT.md (long run: detach via WMI if > 10 min).
5. `cargo clippy -p anki --lib --tests` with and without default features: no warnings.
6. RESULT.md (`artifacts/small-model/sm151/RESULT.md`): what was built, parity numbers, end to end, speed, bytes,
   open risks, Mac tests (below). Update this file. Commit and push `sm151` (`git add -f` for artifacts).
7. Merge into `rc-2026-11` as approved (above), full Rust suite on the merged tree, push.

## Tests that must run on the Mac (they need a full `just build`)

- Python: `qt/tests/test_rwkv_precision.py`, `qt/tests/test_rwkv_model.py`, `qt/tests/test_rwkv_inference_process.py`,
  `qt/tests/test_rwkv_scheduler.py`, `qt/tests/test_rwkv_srs_benchmark.py` (updated for the new default, the moved
  `compact_cards_int8` profile value, cache version 15 and the new policy string).
- TypeScript: `ts/routes/deck-options/rwkv-model-estimates.test.ts` (new default bytes), and the deck-options
  end-to-end test `ts/tests/e2e/deck-options.test.ts` (unchanged, still uses an explicit `compact_cards_int8`).

## Status (updated 11-24, second agent)

- Items 1 to 6 done; see `artifacts/small-model/sm151/RESULT.md`. Fork `rc-lowrank` at 316111fc1 (end-to-end tests,
  Jacobi stop, speed, Deck Options range and replay savings). End to end: every gate passes (Big vs the scorer, all six
  release models vs the low-rank int8 states experiment (SM148) deltas). Long-card full run: 180 runs green.
  Clippy clean, fmt clean. Full suite on the branch: 1,560 passed, 46 ignored.
- Item 7 done: merge commit 514f3a41d pushed to `rc-2026-11` (merged-tree suite 1,560 passed, 46 ignored; base aecc5b09d: 1,546 passed, 42 ignored).
- Follow-ups: clean re-time without a GPU job; cheaper exact codec code (allocations); Mac tests (list above, plus
  the updated deck-options end-to-end test and the new tooltip string).
