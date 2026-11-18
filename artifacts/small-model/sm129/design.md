# Compact state for Tiny (SM129): design (2026-11-18, pre-registered before any GPU ticket)

## Question

The user wants Tiny clearly smaller and accepts up to +1% LogLoss against Small on each head, judged on the 95% CI upper bound (direction approved Wed 11-18 ~22:30). The outside review (`artifacts/small-model/tiny-smaller-codex/analysis.md` on main) ranks a compact stored state second and puts it into its first experiment. This run is the training-free half: **does the compact integer state of the compact int state test (SM47) keep the current Tiny candidates within +1% of Small, and what does it cost against the unmodified Tiny?** A second arm tests the more aggressive card-only format of the second integer-state test (SM49).

| plain name (ID) | what it is |
|---|---|
| the show-time two-head cool-down for Tiny (SM124) | the Tiny two-head candidate (w64-c1, 311,786 parameters); curve 0.300369, instant 0.277105 (published w8+s8) |
| the longer cool-down for Tiny (SM115) | the Tiny Instant-Only candidate (w64-c1); instant 0.276446 |
| the show-time two-head cool-down for Small (SM118) | the two-head reference: curve 0.300016, instant 0.276599 |
| the longer Small Instant-Only model (SM113) | the Instant-Only reference: instant 0.275933 |
| the compact int state test (SM47) | float16 row scales (h) and int8 token-shift vectors (x8) were free on the Muon-trained Small model (SM36): less than 0.002% per head over int4 cards + int8 notes |
| the second integer-state test (SM49) | its arm C' (per-head float16 scales and int4 shift vectors on cards, SM47's notes) cost about +0.08% curve / +0.13% instant more on SM36 |
| six-layer Tiny (SM130) | the companion experiment (trained); it scores its students in this run's compact format if this run goes |

## Arms (no training; scorer `scripts/small_model_heldout.py`, w8 weights)

| arm | variant | card state | note state |
|---|---|---|---|
| published (reference, rows reused) | `w8+s8` | int8 rows, float32 row scales, float shift vectors | same |
| fork default (report only) | `w8+s4c8n` | int4 rows, float32 scales | int8 rows, float32 scales |
| **compact (arm 1)** | `w8+s4c8nhx8` | int4 rows, float16 row scales, int8 shift vectors (float16 scale) | int8 rows, float16 scales, int8 shift vectors |
| **card-only (arm 2)** | `w8+s4hgx4c8hx8n` | int4 rows, one float16 scale per head, int4 shift vectors | as compact |

Both Tiny candidates are scored in all three new formats (one ticket per model, the three formats in one pass). The Small references are also scored at the fork's `w8+s4c8n` (report only: "verify against Small's actual fork precision", outside review step 3).

## Serialized bytes (measured on CPU now, `sm129_bytes.py` -> `sm129-bytes.json`)

A reference encoder / decoder of the fork's stored entity state (`ModuleState::compact`: 14-byte magic, precision byte, u32 layer count; per layer three u32 lengths, the time-shift block, 64 matrix rows, the channel-shift block where the layer keeps its channel mixer; layout from `wt/sm73s2/artifacts/small-model/sm73s2/DESIGN-ADDENDUM.md`). It reproduces the known fork totals (Small 4,554 B int8 and 1,963 B int4 card; Tiny 1,055 / 1,567 B), and decoding gives back the scorer's own quantized values bit for bit (`state_vq.int_row_quant`, random states incl. zero rows, float16-subnormal and float16-clamped scales). The six-layer Tiny (SM130) keeps Tiny's card and note streams, so its bytes are the same.

| Tiny format | card | note | card + note | vs published | vs fork default |
|---|---:|---:|---:|---:|---:|
| published w8+s8 | 1,567 | 1,567 | 3,134 | 0% | +20% |
| fork default w8+s4c8n | 1,055 | 1,567 | 2,622 | -16% | 0% |
| **compact w8+s4c8nhx8** | **803** | **1,315** | **2,118** | -32% | **-19%** |
| **card-only w8+s4hgx4c8hx8n** | **619** | **1,315** | **1,934** | -38% | **-26%** |
| (Small, fork default, for scale) | 1,963 | 1,567 | 3,530 | | |

Notes are shared by sibling cards, so per card in a collection the card number matters most: compact cards are 24% smaller than today's fork cards, card-only cards 41%.

## Pool, labels, rule (`sm129_rule.py`, fixed now)

- Pool: the 723 fit users of the 732 set (`wt/sm72s/.../eval-users-732-fit.json`); rule users = the 722 the big model's rows have (`showtime_report.py pools()`, sha256 pinned 3cdfed03afd4). Show-time inputs (`--inputs e2s-query`), leak-free (`--split-existing --deleted-cards neighbour`), GPU cap 0.8 (0.92 only for a retry without progress).
- Labels (gpu/sm129-rows.jsonl): `sm124-c1-e29st-dec2ep32k-t91s2/{w8+s4c8nhx8,w8+s4hgx4c8hx8n,w8+s4c8n}@e2s-query`, `sm115-c1-e38st-st-dec2ep32k-t99/{...}@e2s-query`, `sm118-e26-dec2ep32k-t91s2/w8+s4c8n@e2s-query`, `sm113-inst-e35st-st-dec2ep32k-t99/w8+s4c8n@e2s-query`. References from committed rows, sha256 pinned (SM124, SM115, SM118 in this tree; SM113 read from `wt/sm113`).
- Statistic: per head, 100 x (Tiny arm / Small - 1) of the review-weighted LogLoss, paired user bootstrap 2,000, seed 0, 95% CI.
- **Rule: an arm passes for a model if every head's CI upper bound is <= +1.00%** (two-head: curve and instant vs the show-time two-head cool-down for Small (SM118); Instant-Only: instant vs the longer Small Instant-Only model (SM113)), on all 722 rule users with finite rows. **Verdict per arm: go if both models pass**, "go for <model> only" if one does, no go otherwise.
- Reported, no verdict: the cost against the unmodified Tiny (arm minus published, arm minus fork default, % of Tiny's own LogLoss with CIs), the arm against Small at the fork's precision, and the bytes above.
- Selftest (CPU, passes now): the outside review's Tiny-vs-Small numbers reproduce exactly through this code (+0.118% [+0.057, +0.174] curve, +0.183% [+0.164, +0.204] instant; Instant-Only +0.186% [+0.166, +0.208]); relabelled Tiny rows as an arm give cost 0 and go; rows with instant scaled by 1.009 fail on the two-head instant CI.

## Expected result

Remaining budget at the published format: 0.882% curve, 0.817% two-head instant, 0.814% Instant-Only. SM47 measured the compact extras as free on SM36, and int4 cards (the fork default) cost SM36 about +0.017% per head on the whole 732 pool (+0.00005 LogLoss), with up to +0.129% instant in the 32-63-review bucket (card-history drift check (SM78)). Central guess: compact +0.02% to +0.06% over published, i.e. about +0.15% / +0.24% / +0.24% vs Small: **go, with a wide margin (~95%)**. Card-only: +0.08% to +0.15% more on the older model, so about +0.25% to +0.40% vs Small: likely go too (~85%). Either way the result also tells the six-layer Tiny (SM130) how much of its budget the state format takes.

## Cost and ETA

| step | GPU | ends about |
|---|---|---|
| launcher waits for the break to end (Wed 20:00 -> Thu 01:00) | | 01:00 Thu 19 Nov |
| CPU proofs `sm129_check.py` (bytes, parity with SM124's tree, GPU path on CPU, rule selftest; ~3-8 min, memory guard 5.9 GB) | | 01:08 Thu |
| SM124 in three formats (723 users, one pass) | 0.4-0.6 h | ~01:45 Thu |
| SM115 in three formats | 0.4-0.6 h | ~02:20 Thu |
| SM118 and SM113 at w8+s4c8n (report only) | 0.3-0.4 h | ~02:45 Thu |
| rule `decide` (CPU, seconds) | | **~02:45 Thursday 19 November 2026** (range 02:15-03:15) |

**Total about 1.2-1.6 GPU h** (the request said "< 1 GPU h"; the fork-default arm and Small's fork-precision rows add ~0.5 h, kept because they separate the int4-card cost from the compact extras and check the rule against the fork's real precision). CPU: < 10 min. Each scoring ticket fits easily before the next break (20:00 Thu).

## Code and branch

- Branch `sm129` is cut from `origin/sm124` (7cbf3d6f, the training and scoring tree of SM115 / SM118 / SM124; main lacks that chain's code) plus the second integer-state test (SM49)'s scorer commit 639e7ba1 applied to four files (`scripts/small_model_heldout.py`, `small_model_wkv.py`, `state_vq.py`, `state_vq_report.py`: the per-head scale flag HEAD_SCALE, int4 shift vectors, per-stream variant specs). The w8+s8 / s4c8n / SM47 paths stay bit for bit (CPU proof K2 against SM124's own tree, preflight on 1 user passed 11-18 22:50; the launcher repeats it on 2 users).
- The row field `state_entity_bits` of the scorer assumes the Small model's two card layers (SM36); it is wrong for Tiny and is not used. The bytes come from `sm129_bytes.py`.
- Files: `sm129_rule.py`, `sm129_bytes.py`, `sm129_check.py`, `sm129_block.py` (byte copy of SM124's), `memguard.ps1`, `launch-sm129.sh`.

## Risks

1. **GPU numerics of the card-only kernel flag:** SM49's GPU check stopped once on float-noise rounding flips for int4-shift variants (not a bug, rule re-set with the user's approval). K3 checks the GPU dispatch on CPU tensors against the CPU scorer (exact on the preflight user).
2. **Maturity-dependent cost** (card-state drift check (SM78): int4 cards cost +0.129% instant in the 32-63-review bucket): the whole-pool mean can hide it; the rule is on the whole pool, as pre-registered.
3. The 3 long users outside the fit pool and the fork's float16 shift rounding are not scored (the scorer keeps float shift vectors in the published and fork-default formats); both as in every published pair.
4. This is the scorer's emulation; the fork's Rust state encoder is not changed (no fork work in this run).
