# Compact state for Small and Big (SM131): design (2026-11-19, pre-registered before any GPU ticket)

## Question

The compact state for Tiny (SM129) found that the compact integer state of the compact int state test (SM47) (int4 cards + int8 notes, float16 row scales, int8 token-shift vectors; variant `w8+s4c8nhx8`) costs essentially nothing against the fork's default stored state (`w8+s4c8n`: int4 cards + int8 notes, float32 row scales, float16 shift vectors): +0.000% curve [-0.001, +0.001] / +0.001% instant for the show-time two-head cool-down for Tiny (SM124), +0.001% for the longer cool-down for Tiny (SM115), while cutting Tiny's card + note state from 2,622 to 2,118 bytes. The user (Thu 11-19 ~14:00) wants to know if the same free saving holds for the bigger models, whose purpose is accuracy per size: **is the compact state free against the fork default for Small and for Big?**

| plain name (ID) | role here |
|---|---|
| the show-time two-head cool-down for Small (SM118) | Small two-head candidate (`w64`, expertium kind), curve + instant |
| the longer Small Instant-Only model (SM113) | Small Instant-Only candidate (`w64`), instant |
| the show-time big model (SM91s2) | Big two-head (the fork architecture, `fork::`, 2,762,884 parameters), curve + instant |
| the show-time instant-only big model (SM99) | Big Instant-Only (the fork architecture), instant |
| the compact state for Tiny (SM129) | the Tiny run this one extends; its rows give Small's fork-default rows |
| the second integer-state test (SM49) | its arm C' = the card-only format (`w8+s4hgx4c8hx8n`), report only here |

## Architectures and the fork's default stored format

- Small: 2 card layers (the second without channel mixer), 1 note layer, 4 heads x 16, 64-wide shift vectors (as SM129's bytes script).
- Big (checked from `artifacts/hf-family/repo-new/rwkv-big-*/architecture.json`, both packages identical, sha256 5c15decc492f): d_model 128, every stream 4 heads x 32; **card 3 layers, note 2 layers, all with a channel mixer**; 128-wide shift vectors. The checkpoints are the packages' `master.pth` (sha256 67ee8e9b... for the show-time big model (SM91s2), c49e45d6... for the show-time instant-only big model (SM99)), read from `wt/sm91s2` / `wt/sm99`.
- **The fork's default for Big is the same as for Small and Tiny: int4 cards + int8 notes** (package INTEGRATION.md section 7 and the fork speed test of 11-14, `wt/speed-test/docs/rwkv-speed-test-2026-11-14.MD`: big-one 9,271 B per card, 10,283 B per note). So the fork-default variant is `w8+s4c8n` for every size, and the compact variant string `w8+s4c8nhx8` needs no adaptation; only the scorer's fork-kind path did (below).

## Formats (no training; scorer `scripts/small_model_heldout.py`, w8 weights, show-time inputs, leak-free)

| format | variant | card state | note state | scored here |
|---|---|---|---|---|
| published (report only) | `w8+s8` | int8 rows, float32 scales, float16 shifts on Big (float on Small) | same | no, committed rows |
| fork default (reference of the rule) | `w8+s4c8n` | int4 rows, float32 scales | int8 rows, float32 scales | Big yes; Small: SM129's rows |
| **compact (tested)** | `w8+s4c8nhx8` | int4 rows, float16 row scales, int8 shift vectors (float16 scale) | int8 rows, float16 scales, int8 shifts | all four models |
| card-only (report only) | `w8+s4hgx4c8hx8n` | int4, one float16 scale per head, int4 shift vectors | as compact | all four models |

Shift vectors of the fork default: on Big (fork path) float16 as in the fork; on Small (expertium path) the scorer keeps float shifts in the fork-default format (as in SM129), so Small's compact-vs-fork-default difference also contains float -> int8 shift rounding.

## Serialized bytes (`sm131_bytes.py` -> `sm131-bytes.json`, CPU, round trip bit for bit)

SM129's reference encoder / decoder of the fork's `ModuleState::compact` layout with the dimensions per architecture. It reproduces every known fork total (Big 9,271 B int4 card, 15,415 B int8 card, 10,283 B int8 note; Small 4,554 / 1,963 / 1,567; Tiny 1,055 / 803 / 619 / 1,315), and decoding gives back the scorer's own quantized values bit for bit (random, zero, float16-subnormal and float16-clamped rows).

| size | format | card | note | card + note | card vs fork default | card + note vs fork default |
|---|---|---:|---:|---:|---:|---:|
| Small | published w8+s8 | 2,987 | 1,567 | 4,554 | +52.2% | +29.0% |
| Small | fork default w8+s4c8n | 1,963 | 1,567 | 3,530 | 0 | 0 |
| Small | **compact w8+s4c8nhx8** | **1,521** | **1,315** | **2,836** | **-22.5%** | **-19.7%** |
| Small | card-only w8+s4hgx4c8hx8n | 1,185 | 1,315 | 2,500 | -39.6% | -29.2% |
| Big | published w8+s8 | 15,415 | 10,283 | 25,698 | +66.3% | +31.4% |
| Big | fork default w8+s4c8n | 9,271 | 10,283 | 19,554 | 0 | 0 |
| Big | **compact w8+s4c8nhx8** | **7,747** | **9,267** | **17,014** | **-16.4%** | **-13.0%** |
| Big | card-only w8+s4hgx4c8hx8n | 6,619 | 9,267 | 15,886 | -28.6% | -18.8% |

Big saves less in relative terms because its rows have 32 values per scale (a float32 -> float16 scale saves 2 of 20 / 36 bytes) and its 128-wide shift vectors were already float16 (int8 saves 126 of 256 bytes per vector). The Big pair (both Big models) doubles every Big number.

## Code change (branch `sm131` = `origin/sm129` + this run)

Before this run the scorer refused per-stream or SM47 / SM49 state bits on fork-kind models (`per-stream state bits are for expertium-kind models only`). Changes:

1. `scripts/compress_quant.py`: the patched fork model's row rounding `_quantize_rows` takes the flags of `scripts/small_model_wkv` (256 = float16 scales, 512 = one scale per head); a new `_round_shift` (shift bits 0 = the fork's float16 shifts, unchanged; 8 / 4 (| 256) = int codes with one scale per vector); a `shift_bits` attribute on both mixers; `set_state_streams(model, card, note)` (per stream, modules 0 and 2 only); `set_state_bits` also resets `shift_bits` (a model is reused across formats in one pass).
2. `scripts/compress_wkv_cuda.py`: the `compress::wkv_q` CUDA op sends flagged bits to `small_model_wkv`'s flag kernel (K = 32 works there; SM47 / SM49's kernel). Plain 0 / 8 / 4 still run the old kernel.
3. `scripts/small_model_heldout.py`: fork-kind models accept tuple / IntState state bits through `fork_streams()`; the rows file is read skipping unparseable lines (a torn or NUL-filled line from a crash no longer crashes a resume), and rows are written as complete lines in one write with flush + fsync, after closing a torn last line.

The row field `state_entity_bits` still assumes Small's layout (wrong for Big, not used); bytes come from `sm131_bytes.py`.

## Proofs

CPU (`sm131_check.py` -> `check-cpu-sm131.json`, ALL OK 11-19 ~14:00, 16 s; the launcher repeats it):

- K1 bytes: the table above, round trip and known totals.
- K2 helpers: `_quantize_rows` / `_round_shift` bit for bit the old formulas for plain 8 / 4 and float16 shifts, and bit for bit `state_vq.int_row_quant` for 8|F16, 4|F16, 4|F16|HEAD (on [B, 4, 32, 32]) and shift 8|F16, 4|F16 (on [B, T, 128]).
- K3 streams: the four variants map to the intended (card, note) bits; `set_state_streams` touches modules 0 and 2 only (layer counts 3, 4, 2, 3, 4); `set_state_bits` resets shift bits.
- K4 CPU smoke rows (`cpu/smoke-cpu-rows.jsonl`, users 2388 and 4676, the two smallest of the pool, the show-time big model (SM91s2), four formats, scored on CPU by this tree): w8+s8 equals the published GPU rows within 1e-6; every format finite; compact differs from fork default. Also tested: the Small two-model command (expertium kind) on CPU.
- K5 rule selftest (below).

GPU smoke (first step of the GPU ticket; a mismatch = `STOP`): the same 2 users and four formats on the GPU; w8+s8 within 1e-5 per head of the CPU rows and of the published rows, every other format within 1e-4 per head of its CPU row (the flag kernel vs the CPU reference: summation order differs, so a rare rounding flip is allowed).

## Pool, rule (`sm131_rule.py`, fixed now)

- Pool: 723 fit users of the 732 set (`wt/sm72s/.../eval-users-732-fit.json`); rule pool = the 722 users the big model's rows have (`showtime_report.py pools()`, sha256 pinned 3cdfed03afd4). Show-time inputs (`--inputs e2s-query`), leak-free (`--split-existing --deleted-cards neighbour`), GPU cap 0.8 (0.92 only for a retry without progress).
- Labels (`gpu/sm131-rows.jsonl`): `sm91s2-big-showtime/{w8+s4c8n,w8+s4c8nhx8,w8+s4hgx4c8hx8n}@e2s-query`, `sm99-big-inst-e2/{...}@e2s-query`, `sm118-e26-dec2ep32k-t91s2/{w8+s4c8nhx8,w8+s4hgx4c8hx8n}@e2s-query`, `sm113-inst-e35st-st-dec2ep32k-t99/{...}@e2s-query`. Small fork default: SM129's rows (`artifacts/small-model/sm129/gpu/sm129-rows.jsonl` in this tree, the same labels and pool, the same scorer code on the expertium path); published rows of SM118 (this tree), SM113 (`wt/sm113`), SM91s2 (`wt/sm91s2/.../sm91s2-inputs.jsonl`), SM99 (`wt/sm99`). All reference files sha256 pinned.
- Statistic: per head, 100 x (compact / fork default - 1) of the review-weighted LogLoss (pct_of_ref), paired user bootstrap 2,000, seed 0, 95% CI.
- **Rule: compact is "free" for a model iff every head's 95% CI upper bound < +0.05%** (two-head: curve and instant; Instant-Only: instant). **Verdict per size: free if both models of that size are free**, otherwise "not free" with the model(s) named.
- Users: the rule users that have finite rows in both formats. The 4 Big users that ran out of GPU memory in the per-user recalibration for Big (SM128) (481, 2267, 3735, 4852) may fail again (that was a per-review dump; plain Big scoring had all 727, and user 3735 hit one out-of-memory error per pass in SM129 at cap 0.8 that a fresh process fixed). Handling: an in-ticket retry in a fresh process at cap 0.8, then 0.92, then further tickets while there is progress; the rule then runs on the users both formats have and reports the count and the missing users; **at least 718 of the 722** are required, otherwise `decide` exits 2 (pending, `STOP` line) and the main session decides.
- Reported, no verdict: compact and fork default vs published w8+s8, card-only vs fork default and vs published (also with the +0.05% test, report only), bytes per card and per note and the saving %.
- Rows are read through a filter that skips and counts unparseable lines (reported as `unparseable_lines`).
- Selftest (CPU, passes now): the statistic reproduces SM129's compact-vs-fork-default numbers for the show-time two-head cool-down for Tiny (SM124) exactly; relabelled rows give 0 cost and free / free; the show-time instant-only big model (SM99)'s compact instant x 1.001 gives Big not free and Small free; 3 users dropped from one Big model gives a rule on 719 users with the missing users listed; 10 dropped gives exit 2; a NUL-filled and a torn line are skipped and counted.

## Expected result

On Tiny the compact extras cost +0.000% / +0.001% (CI half-width ~0.001%). The float16 scale changes each row's step by at most 2^-11 relative; int8 shift vectors with one scale per 128 values are coarser than float16 but were free on the Muon-trained Small model (SM36) in SM47 and on Tiny. Central guess: **free for both sizes (~85-90%)**, with Big's instant head the most likely to show a measurable cost (bigger state, three card layers whose shifts all become int8). Card-only (report only): on Tiny +0.16% / +0.21%; Big's per-head scale covers 1,024 values instead of 256, so likely +0.2% to +0.6%, not free.

## Cost and ETA

No daily break any more (user 11-19 ~14:10): the launcher honours only quiet.json windows of scope all / gpu, re-read before every ticket.

| step | GPU | ends about |
|---|---|---|
| launch, CPU proofs (`sm131_check.py`, < 1 min, memory guard 5.9 GB) | | 14:20 Thu 19 Nov |
| wait in the queue behind the six-layer Tiny (SM130)'s Instant-Only epoch 39 (started 13:49, ~110 min) | | ~15:40 Thu |
| ONE chained ticket: GPU smoke (2 users, ~3 min); Big, both models x 3 formats (published Big scoring: ~13 model-minutes per format, so ~80-90 min); Small, both models x 2 formats (~25-30 min); in-ticket retries | ~1.8-2.2 h | ~17:45 Thu |
| retry tickets only if users are still missing (each waits behind one SM130 ticket, ~110 min) | minutes | (+~2 h if needed) |
| rule `decide` (CPU, seconds) | | **~17:45 Thursday 19 November 2026** (range 17:15-20:00) |

**Total about 2 GPU h** (1.7-2.5 h). CPU: < 5 min. The chained ticket delays the six-layer Tiny (SM130)'s next ticket by the same ~2 h it would cost as separate tickets, but avoids two ~110-min waits for this run.

## Deviations from the brief

1. The scorer's fork-kind path had no per-stream or compact state support; it was added (above) and proven on the published format (CPU rows equal the published GPU rows within 1e-6, helpers bit for bit, GPU smoke gate).
2. Small's fork-default rows are reused from the compact state for Tiny (SM129) (identical labels, pool and scorer code), not rescored; published w8+s8 rows are reused for all four models.
3. One chained GPU ticket (smoke + Big + Small) with in-ticket out-of-memory retries instead of one ticket per model.
4. The rule's coverage floor (718 of 722) is fixed here.
5. The card-only format is a report-only arm for all four models.

## Risks

1. GPU vs CPU rounding of the flag kernel: K2 proves the CPU helpers bit for bit; the GPU smoke bounds the GPU path at 1e-4 per user (SM49 once saw float-noise flips on int4-shift variants).
2. Big users out of GPU memory: handled by retries and the coverage floor.
3. Shift rounding of Small's fork default is float (scorer convention), Big's is float16 (fork convention); each comparison is internally consistent with the rows it is compared to, and this is stated with the result.
4. This is the scorer's emulation; the fork's Rust state encoder is not changed (no fork work in this run).

## Amendment 2026-11-19 ~15:45 (after the first GPU ticket; no change to the rule, the pool, the formats or any finite number)

What happened: the first chained ticket (started 14:43:17) passed the GPU smoke and scored 37 users in the Big step (all six labels; user 2450 was the 37th and finished, its GPU rows match the CPU rows within 7e-6), then stopped at the 38th user, 3876 (58,826 reviews), with `Non-finite evaluation loss`.

Cause (the format, not the GPU): the fork default (`w8+s4c8n`, int4 card rows with one float32 scale per 32 values, the fork's own `precision.rs` encoding) is non-finite for user 3876 **on the CPU as well**, for both Big models (the show-time big model (SM91s2) and the show-time instant-only big model (SM99)). The card module has a row of 9,465 steps (4,733 state updates of one card). With int4 card rows, the second card layer's largest row scale grows from 0.04 at the start to 2.3 after 1,000 steps, 29,000 after 2,000, 1.5e10 after 4,000 and 4e33 at the end (that layer alone, the layers below it in float); with all three card layers in int4 the state overflows float32 and the loss becomes NaN. Int8 rows stay at 0.03 on the same row. Float16 row scales clamp at 65,504, so the compact format (`w8+s4c8nhx8`) and card-only (`w8+s4hgx4c8hx8n`) stay finite for this user (the card's state saturates; compact vs published w8+s8 +0.2% curve for 3876 on CPU). The Small models' fork-default rows (the compact state for Tiny (SM129)) are finite for 3876. The emulation follows `precision.rs` (float32 absmax / 7 scale, round half to even, clamp to +-7), so the fork's own runtime should overflow the same way on such a card: **a finding for the fork's default card state with the big models, reported with the result.**

Handling:
1. Scorer flag `--nonfinite-rows` (used by `sm131_ticket.sh`): a non-finite evaluation becomes a row with `nonfinite` and no `benchmark` (counted as done, not retried) and scoring goes on; without the flag the scorer still stops. `missing` (ticket) and `min_users` (launcher) count these rows as scored, so no retry tickets are spent on a deterministic NaN.
2. The rule already runs on "the rule users that have finite rows in both formats" (above), so 3876 is a missing user for both Big models; `decide` lists the non-finite rows (`nonfinite_rows`). The floor stays **718 of 722**: 3876 uses one of the four allowed missing users; if the four SM128 out-of-memory users also all fail, `decide` exits 2 (pending) and the main session decides.
3. Smoke gate: users 2450 and 3876 are added (`smoke-users.json`, `SMOKE_USERS`), CPU rows from this tree in `cpu/smoke-cpu-rows.jsonl`. For 3876 the fork default must be non-finite on the GPU too, w8+s8 must match the published GPU rows within 1e-5 and the CPU rows within 1e-4 for the two added users (3876's CPU w8+s8 differs from its published GPU row by 1.8e-5, summation order on 58,826 reviews; 2388 and 4676 keep 1e-5 against both), compact and card-only must be finite on the GPU and their GPU-vs-CPU difference is reported only (the user is outside the rule; a saturated int4 state need not agree to 1e-4). 2450 has the normal tolerances.
4. The 37 users already in `gpu/sm131-rows.jsonl` (scored at 078d3f37) are kept: the change touches no arithmetic of a finite evaluation.
