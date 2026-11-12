# RWKV family release (SM108): package inventory (2026-11-11, 19:1x; Big rows 2026-11-12)

Packages are NOT copied into this branch (they sit on their own branches; `build_release.sh` reads them from the worktrees). The earlier release package (sm72e25) did commit `master.pth` and the weight files (small, 1.4 MB each); the new Small and Tiny packages are the same size, so they can be committed the same way on their own branches if wanted. Show-time inputs (`e2s-query`, leak-free) are expected by all of them: the interval runs up to the moment the card is shown (INTEGRATION.md sections 6.9 and 6.10).

Check run: `python testvectors/check_testvectors.py` (NumPy only, tolerance 1e-5, fp32 + int8, recurrent + parallel form, 2 users), run 2026-11-11 about 19:10 with the repo venv.

| # | Model | Package source (path / branch / commit) | Package exists | check_testvectors.py | Parameters |
|---|---|---|---|---|---|
| 1 | Big two-head: the show-time big model (SM91s2) | `wt/big-package/artifacts/models/sm91s2`, branch `big-package` (cut from `sm107-big-showtime`), package commit 9dc96c91; label `sm91s2-big-showtime`; architecture `fork` | YES (with `master.pth`) | RESULT: PASS (largest difference 9.2e-7) | 2,762,884 (fork RWKV size) |
| 2 | Big Instant-Only: the show-time instant-only big model (SM99) | `wt/big-package/artifacts/models/sm99`, branch `big-package`, package commit 9dc96c91; label `sm99-big-inst-e2` (final epoch 2, branch `sm99` at 5a480ed8); architecture `fork`; manifest `use` = instant only | YES (with `master.pth`) | RESULT: PASS (largest difference 9.5e-7) | 2,762,884 |
| 3 | Small two-head: the show-time retrain (SM92) | `wt/sm101-package/artifacts/models/sm92`, branch `sm101-package` (also on `sm98-package`), package commit 876c68a2 (head-of-branch 740c12fd); label `sm92-e26-dec1ep32k` | YES (no `master.pth` in the folder; its sha256 2091f8a8...100d7f is in the manifest, file at `wt/sm92/artifacts/small-model/sm92-e26-dec-1ep32k-muon/run/E89-F65e-F77a/master.pth`, 1,489,805 B) | RESULT: PASS | 331,511 |
| 4 | Small Instant-Only: the Small Instant-Only model (SM98) | `wt/sm98-package/artifacts/models/sm98`, branch `sm98-package`, package commit dd885124; label `sm98-inst-e35st-st-dec1ep32k` | YES (with `master.pth`) | RESULT: PASS | 331,511 |
| 5 | Tiny two-head: the Tiny two-head model (SM106) | planned `wt/sm106/artifacts/models/sm106`; not trained yet (only the cut `sm106-cut-e26` exists) | NO (TODO) | not run | TODO (w64-c1 architecture, expected about 311,786 like SM101, to confirm) |
| 6 | Tiny Instant-Only: the Tiny Instant-Only model (SM101) | `wt/sm101-package/artifacts/models/sm101`, branch `sm101-package`, package commit 4b71dabc; label `sm101-c1-e38st-st-dec1ep32k` | YES (with `master.pth`) | RESULT: PASS | 311,786 |

## sha256 of the weight files and checkpoints

| Model | weights.f32.bin | weights.int8.bin | master.pth |
|---|---|---|---|
| Small two-head (SM92) | 4fbc5f963b6ef18d23dbc0f11358fed8a9c7c97c271d6eda95df3b8ed6fbeb6e (1,351,553 B) | d17c619c7a3197c6055e9e78e974a729bbee07dd1cc24279bb1685424c47711c (436,391 B) | not in package; 2091f8a879a56cf96a3df3b820fdacfa64c6467e8502a3a5eb17436524100d7f |
| Small Instant-Only (SM98) | a006dcc039d19142dc95d257d9c238fb93c7efa12b78e18ec6ae881ac9009a24 (1,351,553 B) | b2f335bd712b9d602150201de41ab5a0f0a4c291644705e2aa1ed2a1bf1592c3 (436,391 B) | 565b105e1e1674a535bc8ea8efcbeb848a6d0a6dc5abd86323274388282c37b9 (1,489,805 B) |
| Tiny Instant-Only (SM101) | 2faacb5b913bcfbdc0e363532d533e3bcff6872c8358f1ef753ca87f9d2a60fd (1,270,749 B) | 77703dd2f2a21a1bcadaedfed1cec93ee0fd6fd6d3c840be221e72b5e1c05066 (409,483 B) | d1408ae9c883e894f98bd8fe85aeb804aa5f66db7d11b02e4da5a7b834d7a149 (1,398,567 B) |
| Big two-head (SM91s2) | b53a89729d518c2db970833b7159904368fe9e882298e0f715f4d302f7cf4ed5 (11,082,637 B; = the fork's `RWKV_trained_on_5000_10000.bin`) | a3dc5df4db4cbb801c8eb816e12fc3c7f48cd5943a04a928fc4e23de00921255 (3,027,096 B) | 67ee8e9b0d0169733242c40646e0d2e5637845a25775762002ffc4e6396f9590 (11,251,089 B) |
| Big Instant-Only (SM99) | 682ba1ff207c51af7de5e2abeb25593362aa706b292cbcc338a1591fad0d935c (11,082,637 B; = the fork's `sm99/weights.f32.bin`) | 0076501c230db7d1a1cd6444b1cabd59118c473261e0bf3c148c77cf99b1f8b8 (3,027,096 B) | c49e45d698194da033caa1231f598f875922a4a2a3e921714a3fd433d37e5768 (11,251,089 B) |
| Tiny two-head (SM106) | TODO | TODO | TODO |

Parity (NumPy reference vs PyTorch, from each `parity.json`): SM92 1.1e-6 (fp32) / 1.2e-6 (int8); SM98 and SM101 about 1.3e-6 / 1.0e-6; SM91s2 2.4e-6 / 1.6e-6; SM99 1.2e-6 / 1.5e-6; all PASS at 1e-5. The Big packages' test vectors were also checked against the original big-model class (fp32, max 1.2e-6).

## Memory at the fork default (int4 cards + int8 notes), exact stored state of one card plus one note

| Model | per card + note, int4 cards + int8 notes | int8 | f32 |
|---|---|---|---|
| Small two-head (SM92) and Small Instant-Only (SM98), same architecture | 3,530 B | 4,554 B | 13,640 B |
| Tiny Instant-Only (SM101) (one card layer) | 2,622 B | 3,134 B | 9,276 B |
| Big (fork RWKV size): the show-time big model (SM91s2) and the show-time instant-only big model (SM99), same architecture | 19,554 B (card 9,271 + note 10,283; confirmed in the fork, `rwkv_big_pair_stores_two_big_states_per_card_and_note`) | 25,698 B | 87,136 B serialized (87,040 B of raw f32 values) |
| Tiny two-head (SM106) | TODO (same w64-c1 state as SM101 expected: 2,622 B) | TODO | TODO |

Source: `wt/sm101-package/artifacts/small-model/sm101/sm104-tiny-fork.md` (SM104) and INTEGRATION.md section 8. A pair of two models in the fork (curve model plus instant model): Small 7,098 B, Tiny 6,190 B, Big 39,146 B (int8: 51,434 B; f32: 174,310 B), each the two states plus a 19-byte pair header per stream (the Big pair in the fork (SM109), fork main 72e03b263, `docs/rwkv-sm109-2026-11-12.MD`).

## Notes

- Instant-Only models (SM98, SM101, SM99) have an untrained curve head (`curve_loss_weight` 0): only `p_imm` and `p_rating` may be used, never `p_curve`, intervals or the 90% stability. Their manifest says so under `use`.
- Big packages (SM91s2, SM99): the fork schedules with the raw forgetting curve and ignores the ahead residual; the package test vectors hold the training code's `p_curve` with the residual (their INTEGRATION.md section 5). Their INTEGRATION.md is a shorter manual (sections 1-10) that points to the small packages' section 6 for the full feature list.
- Package SM92 has a companion folder `sm92-time` (answer-time head, 69,056 parameters, same branch); it is not shipped in this release.
- The SM98 and SM101 manifests/INTEGRATION.md still say "in the Anki fork it runs beside the release (sm72e25)" in `use`; the release notes describe the correct pairing instead (package branches left untouched).
- Numbers for Big two-head (SM91s2) in RELEASE_NOTES.md are the step-2 show-time scores from `artifacts/small-model/showtime/showtime-report.md` (0.299415 curve / 0.276183 instant), not final cool-down numbers.
