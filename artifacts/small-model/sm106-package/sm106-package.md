# The Tiny two-head model (SM106): family package `artifacts/models/sm106` (2026-11-13)

The Tiny two-head model (SM106) passed its rules (verdict go, Fri 11-13 01:32; `sm106-decision.md`, copied here from branch `sm106`). Show-time scores, % of the big model (1 leak-free epoch)'s show-time LogLoss: curve 0.300676, +0.366%; instant 0.277369, +0.076%. This folder packages it for the RWKV family release (SM108) and the Anki fork (the Tiny one-model choice, SM112). CPU only.

## How it was made

From the worktree root (branch `tiny-package`, cut from `origin/sm101-package`), `PY` = the main checkout's venv python:

1. `RWKV_DELETED_CARDS=neighbour CUDA_VISIBLE_DEVICES= $PY -X utf8 scripts/small_model_export.py package --checkpoint <workspace>/wt/sm106/artifacts/small-model/sm106-c1-e29st-st-dec-1ep32k-muon/run/E89-F65e-F77a/master.pth --arch w64-c1 --label sm106-c1-e29st-st-dec1ep32k --out artifacts/models/sm106` (the exporter and options of the sm92 / sm98 / sm101 packages).
2. `reference/`, `testvectors/check_testvectors.py` and `testvectors/make_testvectors.py` copied from `artifacts/models/sm101` (identical to sm92's except that `make_testvectors.py` builds the model from `architecture.json`); `INTEGRATION.md` = sm101's manual with a package sm106 note (both heads usable, no answer-time head).
3. `RWKV_DELETED_CARDS=neighbour CUDA_VISIBLE_DEVICES= $PY -X utf8 artifacts/models/sm106/testvectors/make_testvectors.py artifacts/models/sm106 artifacts/models/sm106/testvectors 4676 1745` (show-time `e2s-query` rows), then `check_testvectors.py` (`make-and-check-testvectors.log`).
4. `$PY -X utf8 artifacts/small-model/sm106-package/sm106_package.py`: neutral names, manifest `evaluation` from `sm106-decision.json`, `use` = curve + instant + rating heads (p_curve, intervals and S90 allowed, as for the show-time retrain (SM92)), `inputs` = show-time, files re-hashed.

## Checks

| Check | Result |
| --- | --- |
| Parameters | **311,786** (the w64-c1 tiny plan; `architecture.json` byte-identical to the Tiny Instant-Only model (SM101)'s) |
| Exporter parity vs PyTorch, 3 users (4676, 1745, 516; 12,461 reviews), fp32 | max \|Δp\| curve 8.9e−7, instant 8.3e−7, rating 1.8e−6: PASS (1e−5) |
| Exporter parity, INT8 weights | curve 8.3e−7, instant 1.1e−6, rating 1.4e−6: PASS |
| Show-time test vectors vs PyTorch while making them (users 4676 and 1745, 6,554 + 7,849 = 14,403 rows) | fp32 max 1.9e−6, int8 max 1.4e−6 |
| `check_testvectors.py` (NumPy, recurrent and parallel forms, fp32 + int8) | **RESULT: PASS (tolerance 1e−05)**, max 1.9e−6 |
| Vector inputs vs the Tiny Instant-Only model (SM101)'s vectors | identical (every non-output array equal) |
| Native parity in the Anki fork (SM112), fixture 500 rows/user / full 14,403 rows | f32 2.3e−6 / 3.2e−6, INT8 1.9e−6 / 3.2e−6 |
| Memory per card + note (fork, exact stored state) | int4 cards + int8 notes **2,622 B** (as SM101); int8 3,134 B; f32 9,276 B |

## Files (sha256)

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `weights.f32.bin` | 1,270,749 | `5b346308bbab6ff32241742544af762b714a16334f6a979423b40dfec8897ec3` |
| `weights.int8.bin` | 409,483 | `0a150164c2c6d014748eb9ee7f98cec5d307236e30018bc64895a34177352919` |
| `master.pth` | 1,398,567 | `e9d0d0a0b3368a920666c5cf89fdde402ca3cfd2465bd3688085bac90c8e3b4f` |
| `architecture.json` | 15,038 | `49650457329d43ff7f3f04f68926e371d49060699142cfec9497cb759ac7a75f` |
| `parity.json` | 3,163 | `88e2479c55a47eeb0c055a8592eb650cf3a1f410a0f27be26f4412d42fd2182f` |
| `testvectors/user_4676.npz` | 1,345,843 | `bc06c0ecbd7e94b80ccd66ee03f2230616746a11c718287c770a10ef6a55bf04` |
| `testvectors/user_1745.npz` | 1,651,723 | `e5ce5dda2a09a96d6adad63a355c4082162a042e933b20901bb9337e3d54e86b` |

`manifest.json` lists the first five with the same hashes. Unlike the sm101 package, `master.pth` is committed, so the package folder is complete on every computer.
