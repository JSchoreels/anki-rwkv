# Smaller model (SM73): design, 2026-10-29

Goal: a smaller small model on the leak-free recipe, within the bar **+0.5% curve / +0.4% instant vs the big model
(1 leak-free epoch) on the 732 set (722 users)** (user, 10-29; it was +1.0%).

Reference: the leak-free small model, cool-down from epoch 20 (SM72s), released as SM72: 331,511 parameters,
4,554 B of int8 state per card + its note, **+0.41% / +0.32%** (0.30019 / 0.27664 vs 0.29897 / 0.27576).
Room under the bar: **0.09 points curve (~0.00027 LogLoss) and 0.08 points instant (~0.00022)**.

**Bottom line.** No architecture shrink is likely to fit in that room; past cuts cost 4 to 5 times more. One shrink
should fit: storing the same model's card/note state at 4 bits instead of 8. That is stage 1 (scoring only, ~15-30
GPU min, launcher ready). A narrower or shallower network needs accuracy bought back first (section 4).

## 1. What "smaller" should mean

| Axis | Small model today | Does it limit a phone? |
|---|---|---|
| Card/note state | 4,554 B per card + note at int8 (62 MB for 14k cards; 0.46 GB per 100k) | **Yes**: it grows with the collection and has to fit in RAM (INTEGRATION.md 11.2) |
| Weights | 0.44 MB int8 | No: fixed size, under 1% of the state at 14k cards |
| CPU per review | 0.51 ms NumPy step, 13 layer steps | No: a full rebuild is already fast; per-review cost follows the layer count, not the width |

So "smaller" here means **fewer bytes of state per card and per note**. A cut in parameters only counts if it also cuts
that state. The per-card state is the card stream (2 layers) plus the note stream (1 layer). Each layer stores a
4 x 16 x 16 matrix (64 rows) plus 16-bit token-shift vectors. The deck, preset and user streams are stored once per
deck, preset or user, so cutting them saves almost nothing on a phone. The 7-layer depth cut (SM15) cut 38% of the
parameters and 0 bytes of card state.

## 2. Candidates

Bytes are for the fork's serialized layout (INTEGRATION.md 11.1 formula; it reproduces the published 4,554 / 3,018 B).
"Cost" is the expected change vs the reference, in points of % vs the big model.

| | Candidate | Params | Bytes per card + note | Training | Expected cost curve / instant | Fits +0.5 / +0.4? |
|---|---|---:|---:|---|---|---|
| **A1** | Same model, **int4 card + note state** (the fork's existing int4 format) | 331,511 | **3,018 (-34%)** | none (~15 min scoring) | **+0.03 / +0.06** | likely (instant margin ~0.02) |
| A2 | Same model, int4 card state, int8 note state | 331,511 | 3,530 (-22%) | none (~15 min) | +0.02 / +0.02 | very likely |
| B | Card stream 1 layer instead of 2 (12 layers) | 311,786 | 3,146 (-31%); 2,122 (-53%) with int4 | warm start + 3 epochs + cool-down, ~14 h | +0.1 to +0.3 per head (estimate) | unlikely |
| C | Width 48 (3 heads x 16) everywhere | 195,949 | 3,434 (-25%) | from scratch, full recipe ~1.5-2 days | +0.15 to +0.5, mostly instant | no |
| (D) | Depth cut of the deck / preset / user streams (as in the pruning test (SM15)) | 204,782 | 4,554 (0%) | | +0.36 / +0.42 measured | no, and it does not shrink the state |

Reasoning:
- **A1 / A2: measured on the earlier Muon best model (SM36), 723 users of the 732 set, paired.** Int4 cards + int8
  notes cost +0.00005 [+0.00003, +0.00006] curve / +0.00005 [+0.00004, +0.00006] instant (SM45). Int4 everywhere cost
  +0.00010 / +0.00017 (SM45). Applied to the reference that gives +0.44% / +0.38% (A1) and +0.43% / +0.34% (A2).
  Weights and state format are independent of training, so these costs should carry over to the new model within
  about ±0.00003. A1's instant margin (0.02 points, ~0.00005) is the one real risk. A CPU smoke on 1 user (2388,
  10-29) gives int4 vs int8 +0.00006 / +0.00002, the same size. A1 needs no fork change (int4 state already exists
  there). A2 needs the fork to pick the precision per entity kind (card vs note).
  - Smaller still, not tested here: 16-bit row scales and int8 token shifts cost ±0.00000 on the earlier model (SM47);
    4-bit notes and shifts +0.00012 / +0.00030 (SM49 B), which would not fit the instant room. Those formats are not in
    the fork's layout, so they are left out of stage 1.
- **B: the only architecture cut that removes per-card state at a small parameter cost.** There is no direct
  measurement. The pruning test (SM15) is the closest one: on the then-best recipe, cuts after fine-tuning and decay
  kept about 1/10 of their zero-shot cost (7-layer cut +4.7% / +3.5% zero-shot, then +0.36% / +0.42% vs the full-size
  model of the same recipe (SM40v2) on 459 users). The cheapest single-stream cut there (preset stream 3 -> 1 layers,
  zero-shot +0.00% / +0.25%) was in a context stream. The card stream holds the per-card memory, which carries most of
  the curve gap (72% of it is on learning-step reviews). So B should cost at least as much as a context-stream cut:
  +0.1 to +0.3 points per head after retraining. That is 1 to 3 times the room.
- **C: width.** Outside research's width ladder at matched training: 80 -> 64 channels cost +0.01 / +0.15, and
  128 -> 64 cost +0.17 / +0.65. Each step costs more at the small end, and it lands on instant, where the room is
  0.08. C also saves less state than A1 or B, and it needs the full from-scratch recipe (more than the budget).
- **D** is listed only because it is the one shrink actually measured on this model family. It fails the bar by
  4 to 5 times and leaves the per-card state unchanged.

## 3. Recommendation: stage 1 = A1, then A2 if A1 fails (prepared, not launched)

Launcher `wt/sm73/artifacts/small-model/sm73/launch-sm73.sh` (branch `sm73`, cut from `sm72s`; the scorer is
exactly the one that produced the reference rows). It scores the released checkpoint
(`wt/sm72s/.../scale-w64kd-final-sc-lanes-leakfree-dec20-1ep32k-muon/.../master.pth`, read only) at `w8+s4` on the
723 fit users of the 732 set, with leak-free scoring (`--split-existing --deleted-cards neighbour`). That is one queue
ticket, ~12-15 min at cap 0.8, with a second pass for gaps. It then runs `sm73_rule.py decide`.

**Decision rule (pre-registered).** Users = the 732 fit users that every series has (722, the SM72s readout).
A format **passes** if its review-weighted LogLoss is ≤ +0.50% curve AND ≤ +0.40% instant vs the big model
(1 leak-free epoch). The rule also reports the paired delta vs the int8 rows (user bootstrap 2,000, seed 0, 95% CI).
**Stop rule:** if A1 passes, stop: A2 is larger and not needed. If A1 fails, score A2 (one more ticket) and ship the
format that passes, or neither.

**Resolution.** Same weights and same users, only the stored state changes, so the paired delta is very tight. On the
earlier model the 95% CI was ±0.00001-0.00002 on 723 users, i.e. ±0.01 points: 20 to 50 times finer than the 0.08-0.09
room. The 732 set (722 users) is the standard decision readout, and it resolves this easily. The weak link is not
sampling noise. It is the point estimate landing right at the instant bar (+0.38% expected vs +0.40%).

GPU time: 15 min (A1 passes) to 30 min (A2 needed), plus ~5 min CPU. Self-checks done 10-29: the rule reproduces the
reference on its own int8 rows (722 users, 0.30019 / 0.27664, +0.41% / +0.32%); the dry run passes; a CPU smoke of the
int4 scorer path with the leak-free flags runs.

## 4. What could buy the accuracy back (needed before any architecture shrink)

A shrink of type B needs about +0.1 to +0.3 points bought back first, on the full-size small model, against the same
fixed reference (the bar is vs the big model (1 leak-free epoch), so a better teacher does not move the bar):
- **More epochs: small.** Cool-down from epoch 20 vs 15 bought 0.00020 / 0.00023 (0.07 / 0.08 points) for 5 epochs
  (~9 GPU h). The constant-LR read-outs are flat from epoch 14 to 20 (464 set: curve +1.18 to +1.89%, instant +1.19 to
  +1.33% vs big), so 5 more epochs would likely buy under 0.05 points.
- **A better teacher (SM65): the only lever of the right size, still untested.** One leak-free epoch made the big model
  0.23 / 0.16 points better. The small model trained on it gained 0.03 / 0.10 points at equal recipe (SM72s vs the
  full retrain (SM56), cool-downs from epoch 15). So about a third to half of a teacher gain reaches the student. To
  buy 0.2 points for the student, the teacher has to get ~0.4-0.6 points better (an ensemble or more users), which is
  days of big-model GPU.
- Stacking shrink + buy-back in one run makes the shrink cost impossible to read. Buy back first, on the full-size
  model, then shrink.

**Stage 2 (only if a buy-back lever first moves the full-size small model to ≤ +0.25% / +0.20%):** B, warm-started
from the epoch-20 checkpoint of the leak-free small model (SM72s). Drop the card stream's second layer: it is one
residual branch, and the other 12 layer steps keep their order. Then 3 constant-LR epochs (~100 min each) and the
1-epoch 32k Muon cool-down (~8 h), ~14 GPU h in all.
- Early check: a paired cool-down vs cool-down comparison, never per-epoch constant-LR read-outs. Those swing by
  0.27 points (curve) and 0.06 (instant) from one epoch to the next (epochs 14-20), more than the room. So both B after
  1 epoch and the reference epoch 20 get a quarter-epoch 32k cool-down (~2 h each).
- Resolution: architecture pairs are noisier than format pairs. On 459 users, the pruning test's (SM15) CI was ±0.19
  points curve and ±0.045 instant. The instant head therefore resolves 0.1 points on the 464 set (458 users). The
  curve head needs about 4 times as many users: the 2,225 set (±0.09 points). That also needs the big model
  (1 leak-free epoch) scored there (~1 h).
- Stop if the paired delta (point estimate) exceeds +0.00020 curve or +0.00015 instant, about 3/4 of the room.
- Budget: ~4.5 h of check + ~10 h to finish: inside the 1.5-day budget, but not started now.

## 5. Budget and order

| Step | GPU | When |
|---|---|---|
| Stage 1: A1 (int4 state) on the 732 set; A2 only if A1 fails | 15-30 min | ready; launch when the GPU is free |
| Stage 2: card stream 1 layer (B), early check then finish | ~14.5 h | only after a buy-back lever lands (better teacher (SM65)) |
| Unused budget | ~1 day | goes to the buy-back lever, not to shrinking |
