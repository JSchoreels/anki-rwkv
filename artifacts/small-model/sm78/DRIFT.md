# int4 drift check (SM78): does the int4 card state lose more on cards with long review histories?

Date 2026-10-29. CPU only (GPU untouched). Format tested: int4 card + int8 note state (w8+s4c8n, 3,530 B per card + note;
the format the smaller model (SM73) passed with) against int8 state (w8+s8, 4,554 B), same weights: the leak-free small model
(SM72s, cool-down from epoch 20), leak-free scoring (`--deleted-cards neighbour`).

## Does the scorer re-quantize after every review?

Yes. The scorer (`scripts/small_model_heldout.py` in branch sm73) scores the whole history as one sequence per stream.
Both the CUDA kernel (`scripts/small_model_wkv.py`) and the CPU path (`scripts/state_vq.py`, per-row int quantizer) round the
stored state after every recurrence step, and the next step reads the rounded state. For the card stream (one sequence
per card) that is one re-quantization per review of the card, as in the fork. The int8 vs int4 difference in
w8+s4c8n lies only in the card stream (notes are int8 in both), so the card's review index is the right axis.
Not saved by the smaller model (SM73) run: it stored per-user aggregates only (no `--dump`), so there were no per-review outputs.

## What was computed

CPU rescoring with `--dump` of 110 users of the 732 set (fit users, at most 40,000 reviews each): 80 random (seed 0) plus
the 30 with the most reviews of cards with 64+ reviews (`sample-users.json`). 1.52M labelled reviews x 2 formats; 4 processes
x 2 threads, about 17 min wall. Check: per-user curve LogLoss from the dumps matches the scorer's own number to 5e-7.
Per-review LogLoss is bucketed by the card's review index (n-th review of that card in the revlog; the first review of a card
has no label, so there is no bucket "1"). Gap = int4 minus int8 (positive = int4 worse); 95% CI from a bootstrap over users
(2,000, seed 0). Benchmark users only. Files: `drift_buckets.py`, `drift-buckets.json` (table), `cpu-scores.jsonl` (per-user scorer rows).

## Result (gap in 1e-5 LogLoss)

| card review | reviews | users | curve int8 | curve gap [95% CI] | instant int8 | instant gap [95% CI] | instant gap, % |
|---|---:|---:|---:|---|---:|---|---:|
| 2-3 | 429,261 | 110 | 0.2917 | -0.4 [-1.8, +0.9] | 0.2664 | +0.9 [-0.4, +2.2] | 0.003% |
| 4-7 | 488,255 | 110 | 0.3490 | +0.5 [-1.5, +2.4] | 0.3175 | +2.1 [-0.1, +4.1] | 0.006% |
| 8-15 | 340,967 | 110 | 0.4058 | +5.6 [+1.7, +9.7] | 0.3743 | +7.4 [+3.3, +11.4] | 0.020% |
| 16-31 | 181,399 | 105 | 0.4407 | +9.8 [+1.6, +18.0] | 0.4130 | +17.1 [+8.7, +26.2] | 0.041% |
| 32-63 | 66,377 | 77 | 0.4525 | +16.6 [-11.0, +39.8] | 0.4230 | +54.7 [+23.2, +82.1] | 0.129% |
| 64+ | 11,236 | 45 | 0.4083 | -105 [-214, +15] | 0.3789 | +1.5 [-104, +95] | 0.004% |
| all | 1,517,495 | 110 | 0.3615 | +2.4 [+0.2, +4.7] | 0.3323 | +7.0 [+4.1, +10.2] | 0.021% |

16+ minus 1-15 gap: curve +5.0e-5 [-4.6e-5, +15.3e-5] (not significant); instant +23.0e-5 [+11.6e-5, +34.1e-5] (significant).

## Verdict

There is a mild, real drift on the instant head and a weaker, not significant one on the curve head. The gap grows
about monotonically with the card's review count up to 32-63 reviews (instant 0.9, 2.1, 7.4, 17.1, 54.7 e-5; curve
-0.4 to +16.6 e-5), which fits error compounding. The 64+ bucket has a CI too wide to say anything (45 users, 11k
reviews) and its point estimates are back near zero. In relative terms the worst bucket costs 0.13% of instant LogLoss and
cards with 32+ reviews are 5% of reviews, so the format's gap stays small (+0.00007 instant on this sample, in line with the
+0.00005 on the 722 users). Cards with 32+ reviews carry about a third of the total instant gap on this sample. Not a
blocker for the int4 card + int8 note format. If the drift is to be removed, the fixes in the literature note are cheap
(finer scale per row; see the smaller int state, round 2 (SM49)).

Caveats: the long-history buckets are thin; part of the growth could be the card state mattering more late rather than
compounding (a paired fp32-state reference arm would separate them, ~same CPU cost again). Full int4 (w8+s4) is a different
case: it diverged on user 3876 through the note stream (9,539 steps), a clear compounding failure that int8 notes avoid.
