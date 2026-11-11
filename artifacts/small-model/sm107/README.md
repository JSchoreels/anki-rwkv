# The Big size on the show-time big model (SM107)

The Anki fork's **Big** model size (profile key `legacy`, file `qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.bin`) now runs the show-time big model (SM91s2) instead of the big model (1 leak-free epoch) (SM72 big). User decision 2026-11-11: on show-time inputs the show-time big model (SM91s2) is −0.06% curve (tie) and −0.35% instant against the big model (1 leak-free epoch). Fork report: `docs/rwkv-sm107-2026-11-11.MD` (fork commit in `parity.json`).

## Source

- Checkpoint: `wt/sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth` (branch `sm91s2`, not committed), SHA-256 `67ee8e9b0d0169733242c40646e0d2e5637845a25775762002ffc4e6396f9590`, 504 tensors, 2,762,884 parameters, f32 state dict.
- Previous Big weights: the big model (1 leak-free epoch) master `ac7f0d8a…` → `.bin` `0e07004d…`.

## Steps (CPU, about 2 minutes)

```bash
cd C:/Users/jonat/workspace/wt/sm107/artifacts/small-model/sm107
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
F=C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference
M=C:/Users/jonat/workspace/wt/sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth
export CUDA_VISIBLE_DEVICES= RWKV_DELETED_CARDS=neighbour
# 1. writer check: the previous .pth reproduces the previous .bin byte for byte
$PY -X utf8 sm107_export.py $F/RWKV_trained_on_5000_10000.pth export/reexport-sm72big.bin   # 0e07004d..., cmp identical (before the swap)
# 2. export
$PY -X utf8 sm107_export.py $M export/sm91s2-big.bin                                         # b53a8972...
# 3. vectors (cwd = wt/sm91s2, which has scripts/sm31_inputs.py)
cd C:/Users/jonat/workspace/wt/sm91s2
$PY -X utf8 $OLDPWD/sm107_vectors.py $M $OLDPWD/export/sm91s2-showtime-full.json 0 e2s-query 4676 1745
$PY -X utf8 $OLDPWD/sm107_vectors.py $M $OLDPWD/export/sm91s2-normal-full.json 0 normal 4676 1745
cd -
$PY -X utf8 sm107_fixture.py export/sm91s2-showtime-full.json export/legacy-vectors.json.gz   # -> fork rslib/src/rwkv/testdata/
# 4. native parity in the fork (cargo-env: n2 + rustup cargo on PATH)
cargo test -p anki --lib legacy_tests -- --nocapture
ANKI_RWKV_LEGACY_VECTORS=<export>/sm91s2-showtime-full.json cargo test -p anki --lib legacy_tests -- --nocapture
ANKI_RWKV_LEGACY_VECTORS=<export>/sm91s2-normal-full.json cargo test -p anki --lib legacy_tests::rwkv_legacy_matches_published_vectors -- --nocapture
```

Method check: `sm107_vectors.py ... 500 normal 4676 1745` on the previous `.pth` reproduces the fork's previous `legacy-vectors.json.gz` exactly (features, IDs and skip flags identical, outputs equal to 0), so the new vectors use the 2026-11-10 method; show-time inputs change every one of the 6,543 query rows' inputs and no answer row.

Results: `parity.json`; logs in `export/` (`vectors-*.log`, `rust-*.log`). Not committed: the `.bin` files (in the fork) and the full 19 MB vector JSONs (regenerate in ~35 s with step 3).

## Later (not done)

When the show-time instant-only big model (SM99) finishes (~Thursday 2026-11-12) and passes, Big gets an RWKV-Instant partner the way Small and Tiny pair their curve model with an instant-only model: an entry in `RWKV_CHOICE_INSTANT_MODELS` (`qt/aqt/rwkv_model.py`), a model kind with bundled weights, and a big-engine pair in the runtime (`RwkvRuntime::load_paired` accepts only small models today).
