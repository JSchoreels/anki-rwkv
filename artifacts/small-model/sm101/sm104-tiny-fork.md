# Big / Small / Tiny model sizes in the Anki fork, Tiny = the smaller instant-only model (SM101) (SM104), 2026-11-11

## Commits

- Fork `main`: `094910d73` "RWKV model sizes Big / Small / Tiny: Tiny runs RWKV-Instant on the smaller instant-only model (SM101)" (built on worktree branch `model-sizes`, fast-forwarded into `main` from `cea5304b9`, pushed). Report in the fork: `docs/rwkv-sm101-2026-11-11.MD`.
- anki-rwkv `sm101-package` (cut from `sm101`): `4b71dabc` package `artifacts/models/sm101` + INT8/INT4 state scoring (`artifacts/small-model/sm101/sm104/`), then this report and the speed-bench logs (`sm104/bench/`). No release, no GitHub comments.

## What the user gets

Deck Options → RWKV → Advanced: three radio choices labelled exactly **Big**, **Small**, **Tiny**, each with a one-line description without model IDs (Tiny's says "Tiny only shrinks RWKV-Instant's state; RWKV-Curve uses the same model as Small"). Default Small.

| Presets use | Big (`legacy`) | Small (`sm72e25`) | Tiny (`tiny`) |
| --- | --- | --- | --- |
| RWKV-Instant only (opt-out ignored) | big model alone | the show-time instant-only model, stage 2 (SM98) alone | the smaller instant-only model (SM101) alone |
| both, opt-out off | big model alone | the default small model (SM63) + SM98 | the default small model (SM63) + SM101 |
| both, opt-out on | big model alone | SM63 alone | SM63 alone |
| RWKV-Curve only / neither | big model alone | SM63 alone | SM63 alone |

## Design choices

- **Engine:** no third compiled specialization. The small engine reads the card layers from the weights: a file with no `rwkv_modules.0.blocks.1.` tensor gets `TINY_LAYER_PLAN` (layers 1/4/1/3/3, the 13 small-model steps without card layer 1), otherwise the engine's plan; `SrsModel` stores the plan and the five layer-step loops iterate its `&'static` slice instead of the constant. Widths, kernels and state codecs are untouched (states already carried their layer count). `ModelKind::Sm101` (leak-free, instant-only) plus a card-layer check per key: a key loading weights of the other plan is refused, alone or paired; `load_paired` takes SM101 only as the instant side. Predictions on `origin/main`'s models are bit-identical (benchmark checksums equal).
- **Profile value / migration:** the size reuses the profile key `rwkvModel`; Big = `legacy` and Small = `sm72e25` are the values earlier builds stored, so nothing is rewritten; Tiny is the new value `tiny` (an older build falls back to Small). Unset or unknown values (incl. `sm57`/`sm98`/`sm101`) → Small. Old small keys `sm40, sm36, sm55, sm56, sm72, sm72e20`: unchanged path (they keep running until the next Deck Options save, which stores Small; the UI shows them as Small), so an app update never triggers a rebuild by itself. `sm92` (the show-time retrain of the release (SM92)): preserved advanced value, shown as a fourth radio only for profiles that hold it (the user's "SM92 option only" decision of 11-05).
- **State keys:** SM101 alone (`model` `sm101`) and the SM101 pair (`instantModel` `sm101`) have their own replay keys; the model cache key also hashes the main model's weights. Small and Tiny without RWKV-Instant run the same model (SM63 alone), so that switch correctly keeps the state. A custom `ANKI_RWKV_MODEL_PATH` replaces the size's full model (Tiny → SM63).

## INT8 / INT4 checks for Tiny

1. **Parity vs PyTorch** (package export, 3 users): fp32 max |dp| instant 1.2e-6 / rating 1.3e-6; INT8 weights 1.0e-6 / 1.0e-6 (tol 1e-5). Show-time test vectors (users 4676, 1745; inputs identical to sm98's): NumPy checker PASS, max 6.0e-7. Native (fork) parity on the full 14,403-row vectors: f32 1.1e-6, INT8 1.2e-6; fixture (500 rows/user) 9.5e-7 / 1.1e-6; show-time feature builder 7.5e-7 over 6,543 queries.
2. **Every state precision runs Tiny:** all `precision_tests` cases (bulk vs sequential replay at chunk sizes 1/9/51, packed queries and undo, session checkpoints and spills, the int4-card / int8-note split, int8 shared scopes) now run the small test model and SM101; the paired and instant-only runtime tests run SM101 at f32, int8 and int4 cards + int8 notes; no panics. Exact stored sizes are asserted (below). The fork's shipped options are f32, int8 and int4 cards + int8 notes (full int4 is not parseable).
3. **Accuracy cost** (CPU, f32 weights, leak-free, show-time inputs, the 80 random users of the 732 set from the int4 drift check (SM78), 1,019,305 benchmark reviews; instant LogLoss in % of the same model's f32-state LogLoss; paired user bootstrap 2,000, seed 0, 95% CI; `sm104/sm104-precision.md`):

| RWKV-Instant model | f32 state LogLoss | int8 state | int4 cards + int8 notes (default) |
| --- | --- | --- | --- |
| the smaller instant-only model (SM101), Tiny | 0.294372 | -0.001% [-0.003, +0.001] | +0.013% [+0.003, +0.023] |
| the show-time instant-only model, stage 2 (SM98), Small | 0.293658 | -0.000% [-0.001, +0.000] | +0.010% [+0.001, +0.018] |
| Tiny's cost minus Small's | | -0.000% [-0.002, +0.001] | +0.003% [-0.005, +0.013] |

Tiny's INT4 cost is **not clearly larger** than SM98's (the difference's CI includes 0); both are about 0.01%, below the 0.05% minimum effect. Caveat: the research scorer keeps token-shift vectors at f32 where the fork rounds them to f16 (a much smaller rounding than the int8/int4 WKV rows).

## Memory per card

Exact stored state of one card plus one note:

| Precision | Small's RWKV-Instant model (SM98) / SM63 | Tiny's (SM101) | change |
| --- | --- | --- | --- |
| int4 cards + int8 notes (default) | 3,530 B | 2,622 B | -25.7% |
| int8 | 4,554 B | 3,134 B | -31.2% |
| f32 | 13,640 B | 9,276 B | -32.0% |

Measured in the benchmark (4,000 cards, 2,000 notes, default precision, per card + note): Small pair 7,098 B vs Tiny pair 6,190 B (-12.8%); SM98 alone 3,530 B vs SM101 alone 2,622 B; Big 19,554 B. Weights: 311,786 parameters, `weights.f32.bin` 1,270,749 B (SM98 1,351,553 B).

## Speed: Small before/after (reviews/s and CPU)

New ignored bench `rwkv_model_size_benchmark` (fork `rslib/src/rwkv/model_size_bench.rs`): synthetic 4,000 cards / 32,000 answers, int4 cards + int8 notes, one process per (tree, runtime), median of 5 timed samples per process, 3 interleaved processes per side against `origin/main` `cea5304b9` (the same bench file dropped into a scratch worktree), harness release build, Windows; a GPU training job ran meanwhile (its CPU load is shared by both sides). Logs `sm104/bench/`.

| Runtime | threads | main reviews/s | new reviews/s | main → new process CPU |
| --- | --- | --- | --- | --- |
| SM63 alone (Small, curve only) | 8 | 45,378 | 45,776 (+0.9%) | 58.8 → 59.4 s |
| SM63 + SM98 pair (Small, both) | 8 | 26,530 | 26,473 (-0.2%) | 104.6 → 103.5 s |
| SM98 alone (Small, instant only) | 8 | 46,469 | 46,416 (-0.1%) | 52.6 → 52.7 s |
| SM63 alone | 1 | 5,848 (7 runs) | 5,772 (-1.3%; best warm-up 5,350 vs 5,343 ms) | 38.4 → 38.5 s |
| SM63 + SM98 pair | 1 | 2,872 | 2,865 (-0.2%) | 69.6 → 69.8 s |
| SM98 alone | 1 | 5,692 | 5,777 (+1.5%) | 35.8 → 36.1 s |
| Big | 8 | 14,950 | 14,923 (-0.2%) | 194.6 → 196.4 s |
| SM63 + SM101 pair (Tiny, both) | 8 / 1 | — | 27,699 / 3,039 | 100.1 / 66.7 s |
| SM101 alone (Tiny, instant only) | 8 / 1 | — | 49,847 / 6,332 | 48.1 / 32.9 s |

Verdict: **no measurable speed or CPU loss on Small.** All differences are within ±1.5%, SM63 alone and SM98 alone (identical code path, different weights) move in opposite directions on one thread, and the best warm-up per process is equal. Tiny is faster than Small: +7% (8 threads) / +10% (1 thread) replay with RWKV-Instant only, +5% / +6% with both schedulers, queue scoring about 5-10% faster.

## Tests run

- Rust, native harness (`rwkvharness`, both feature sets): 237 passed, 4 failed, both with default features and `--no-default-features`; the 4 failures are the `fused_packed` bit-exactness tests, which fail identically on `origin/main` in the harness. Full fork `check:rust_test` (cargo nextest over the workspace): passed. New tests: `rwkv_sm101_matches_published_vectors`, `rwkv_sm101_ignores_missing_id_flags_in_every_inference_path`, `rwkv_sm101_feature_builder_matches_show_time_intervals`, `rwkv_small_engine_reads_card_layers_from_the_weights`, `rwkv_runtime_refuses_weights_of_another_layer_plan`, `rwkv_state_precision_stored_sizes_of_the_small_and_tiny_models`, the spec check against the tiny plan, and SM101 in every precision / paired / instant-only test.
- Python: `qt/tests/test_rwkv_model.py` 934 passed (new: three sizes × instant only / both / curve only / neither / opt-out, migration of every stored value incl. unset/unknown/old keys/sm92, custom model path, backend wiring for `sm101`, 8 distinct replay keys, instant-only refresh, workload availability); `test_rwkv_scheduler.py` 794 passed; `test_main.py` passed; `test_rwkv_inference_process.py` 80 passed / 4 skipped when pointed at the built `_rsbridge.pyd` (the committed test looks for `_rsbridge.so`, so on Windows it skips; run locally with a temporary path change, not committed).
- TS: `lib.test.ts` (vitest, all passed), Playwright `deck-options.test.ts` 26 passed (new: radios, no IDs in descriptions, Big → Small → Tiny → Big saved and reloaded, opt-out only for Small/Tiny, SM92 as a fourth choice, older models saved as Small).
- Lints: `just lint` (clippy, mypy, ruff, eslint, svelte, typescript) pass; `just minilints` (copyright, licenses) pass; `just fmt` passes except the pre-existing `AGENTS.md` symlink formatting.

## Left open

- **Fork build on this machine:** `n2` can no longer start `out/rust/release/runner.exe` (CreateProcess rejects the forward-slash program path, also from Python), so `just build` fails in a fresh tree. The checks above ran with an uncommitted local workaround in `build/ninja_gen/src/render.rs` (backslash runner path on Windows), and `just` was driven with `--shell powershell.exe` because `pwsh` is not installed. Worth a proper fix (or a check of what changed on Windows) before the next local release build.
- `test_rwkv_inference_process.py` skips on Windows (looks for `_rsbridge.so`); a `.pyd` lookup would make it run there.
- GPU-side checks: none needed (all CPU).
