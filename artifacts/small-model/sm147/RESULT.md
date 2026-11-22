# Low-rank structure of the card and note states (SM147): result

CPU only, no GPU ticket, no training. A pointer on 8 public benchmark users (about 100k reviews), not a gate: no verdict rules, and every uncertainty below is wide. Source commit 0a128ce9 (clean tree). The run ended 11-23 00:46 (`SM147 RUN DONE`); the main scoring took about 30 minutes wall in two processes (the two Big models at about 558 review-evaluations per second, the Small and Tiny models at about 1,858 per second). Full tables: `sm147-result.md` and `sm147-result.json`; per-user aggregate rows in `cpu/sm147-rows.jsonl` (no card ids, no review rows).

Checks, all passed (`check-sm147.json`, `all_ok: true`): C1 the truncation selftest, C2 the float32 probe equals plain int8-weights scoring on 3 users x 6 models (18 of 18 pairs), C3 the int8 reference equals the committed CPU smoke rows of the int3 young tier (SM145) (18 of 18), C4 all 240 rows finite and the spectrum write counts consistent.

Labels: F32 is the float32 state, REF the int8 reference format (int8 cards and notes, float16 row scales, int8 shift vectors), R1 / R2 / R4 the best rank-1 / 2 / 4 approximation of each head's matrix at every write (float32 factors, no quantization of the factors). The int8 format's own cost on these users against the float state is within 0.006% for every model and head, so REF and F32 are interchangeable below.

## 1. Energy kept by rank (stored float32 matrices, every write)

An entity with n answers has a state of rank at most n, since each write adds a rank-1 term. So the buckets 1-2 are exactly low-rank by construction (rank 2 keeps 100.0%) and the buckets 3-4 are exact at rank 4 (100.0%). The informative buckets start at 5-8. Mean share kept, rank 1 / 2 / 4 (rank 8 is 99.8% to 99.99% overall for every model and stream):

- Big two-head (SM91s2) and Big Instant-Only (SM99) (head width 32), card: 5-8 answers 74% / 91% / 99%; 9-15 72% / 88.5% / 97.5%; 16-31 71% / 88% / 97%; 32-127 71% / 87.5% / 97%; 128+ 77% / 90% / 98% (few writes). Note: 5-8 79% / 93% / 99%; 9-15 75.5% / 90% / 98%; 16-31 74% / 89% / 97%; 32-127 70% to 72% / 86% to 87% / 96%; 128+ 62% / 81% / 94%. Over all writes: card 78% / 92.6% / 98.7%, note 79% / 92% / 98.3%.
- Small two-head (SM118) and Small Instant-Only (SM113) (width 16), card: 5-8 77% to 80% / 93% to 94% / 99.4%; 9-15 76% to 79% / 93% / 99%; 16-31 75% to 78% / 93% to 94% / 99%; 32-127 75% to 76% / 93% / 99%; 128+ 84% / 95% / 99%. Note: 5-8 76% to 80% / 94% to 95% / 99.5%; 9-15 74% to 76% / 91% to 92% / 98.7% to 98.9%; 16-31 72% to 74% / 90% to 91% / 98.3%; 32-127 71% / 89% / 98%; 128+ 69% to 72% / 85% to 87% / 96%.
- Six-layer Tiny two-head (SM130) and six-layer Tiny Instant-Only (SM130), card: 5-8 75% to 79% / 92.5% to 93.5% / 99.4% to 99.5%; 9-15 73% to 78% / 92% to 93.5% / 99%; 16-31 70% to 75% / 91% to 93% / 98.8% to 99.1%; 32-127 68% to 73% / 91% to 93% / 98.8% to 99.1%; 128+ 79% / 93.5% / 99.1% to 99.3%. Note: 5-8 78% to 81% / 94% to 95% / 99.5%; 9-15 74% to 78.5% / 92% to 93% / 99%; 16-31 72% to 76% / 91% to 92% / 98.4% to 98.8%; 32-127 71% to 74% / 89% to 91% / 97.8% to 98.3%; 128+ 72% to 73% / 87% / 96% to 97%.

So past the young cards the pattern is stable: rank 1 keeps roughly 70% to 80%, rank 2 about 87% to 95%, rank 4 about 97% to 99.5%. The notes drift lower with the answer count (the 128+ note bucket is the least low-rank, 94% at rank 4 for Big), the cards stay flat. The width-16 models are more compact than the width-32 Big models at equal rank (rank 4 is a quarter of the full rank for them, an eighth for Big). Per layer and head, the first layer is the most low-rank (Big card layer 0 rank 2: 96% to 98.7%) and some later heads are the least (Big card layer 2 head 1 rank 2: 84%, rank 4: 96%); the rank-1 share of the weakest heads is 62%.

## 2. Post-training rank truncation, benchmark LogLoss (review-weighted, % vs the int8 reference REF)

Paired user bootstrap 95% interval in brackets (8 users: rough width only). Rank 4 is free within noise everywhere; rank 2 is free for the width-16 models and costs a little for Big; rank 1 is visible everywhere.

| model | head | rank 1 | rank 2 | rank 4 |
|---|---|---|---|---|
| Big two-head (SM91s2) | curve | +0.16% [+0.01, +0.30] | +0.04% [+0.01, +0.07] | +0.01% [-0.01, +0.03] |
| Big two-head (SM91s2) | instant | +0.34% [+0.30, +0.41] | +0.08% [+0.04, +0.13] | +0.00% [-0.02, +0.03] |
| Big Instant-Only (SM99) | instant | +0.42% [+0.34, +0.54] | +0.06% [+0.04, +0.08] | -0.01% [-0.04, +0.02] |
| Small two-head (SM118) | curve | +0.21% [+0.10, +0.28] | +0.01% [-0.01, +0.05] | +0.01% [-0.01, +0.03] |
| Small two-head (SM118) | instant | +0.22% [+0.19, +0.31] | -0.02% [-0.07, +0.05] | +0.01% [-0.01, +0.03] |
| Small Instant-Only (SM113) | instant | +0.26% [+0.11, +0.35] | +0.01% [-0.02, +0.06] | -0.01% [-0.02, +0.01] |
| six-layer Tiny two-head (SM130) | curve | +0.21% [+0.08, +0.31] | +0.02% [-0.01, +0.05] | +0.00% [-0.02, +0.02] |
| six-layer Tiny two-head (SM130) | instant | +0.37% [+0.25, +0.63] | +0.07% [+0.01, +0.21] | -0.01% [-0.01, +0.01] |
| six-layer Tiny Instant-Only (SM130) | instant | +0.37% [+0.19, +0.63] | +0.07% [-0.01, +0.17] | -0.00% [-0.02, +0.03] |

Against the float state the numbers are the same to 0.005%. Instant is hurt more than curve at rank 1 (the Instant-Only models and the instant heads +0.22% to +0.42%, the curve heads +0.16% to +0.21%). By answer count of the card's earlier answers (all scope, % vs REF, read heads): rank 1 starts near zero for 1-4 earlier answers (-0.16% to +0.44%, mixed signs), reaches about +0.2% to +0.5% for 5-15, about +0.3% to +0.6% for 16-31 and about +0.6% to +1.2% for 32-127, so it grows with the answer count as expected; rank 2 stays within +0.1% up to 31 and reaches +0.1% to +0.4% at 32-127 (largest for the six-layer Tiny models, +0.29% to +0.38% instant); rank 4 is within +/-0.03% up to 31 and at most +0.09% at 32-127. The 128+ bucket (209 predicted reviews, all scope only) swings in both directions by up to a few percent and is noise.

## 3. Reading (a pointer on 8 users, not a gate)

- On these users rank 4 loses nothing measurable for any of the six models, on either head, and by bucket only a +0.05% to +0.09% hint at 32-127 earlier answers. Rank 2 is free for Small two-head (SM118), Small Instant-Only (SM113) and the curve heads, and costs +0.04% to +0.08% for Big two-head (SM91s2) and Big Instant-Only (SM99) and up to +0.07% for the six-layer Tiny instant heads (intervals touch zero for Tiny). Rank 1 costs +0.16% to +0.42% and is the only one clearly visible. These are post-training truncations with no recovery; a short cool-down trained with the compressed state would very likely close part of the rank-1 and rank-2 gaps, as outside research reports for its own models, but nothing here measures that.
- Young cards are the easy case by construction: with at most 4 answers rank 4 is exact, and for 5-8 answers rank 4 keeps 99% or more at width 16 and 98.9% to 99.2% at width 32. A young-card-only rank-r form (exact below r+1 answers, switching to the int8 matrix after) is therefore free, but it only saves bytes while the entity is young, so it helps the entity-count-weighted average rather than the long-lived cards that dominate the stored bytes.
- Worth a proper GPU test? Yes for rank 4, as a cheap one for the Big models (SM91s2, SM99) where the byte saving is large, and maybe for rank 2 with a short cool-down trained with the compressed state; rank 1 is not worth it without a cool-down. It combines with the int4 / int8 tiers in the obvious way (the factors are the int8 / int4 codes). For Tiny the saving is small in absolute terms and would add to the aggressive state tiers for Tiny (SM146) rather than replace them. The sample is 8 users and the 128+ bucket has 209 reviews, so a GPU test on the 732-user set would decide, not this probe.

Bytes per entity if the WKV part were stored as rank-r int8 factors with one float16 scale per factor (a pair for one head is 2 r K codes plus scales, so 2 r K + 4 B per head instead of K x K + 2 K at int8 rows; shift vectors and the rest unchanged), against the int8 rows:

| size | card int8 | card rank 1 / 2 / 4 | note int8 | note rank 1 / 2 / 4 |
|---|---:|---|---:|---|
| Big (SM91s2, SM99) | 13,891 | 1,651 (-88%) / 2,419 (-83%) / 3,955 (-72%) | 9,267 | 1,107 (-88%) / 1,619 (-83%) / 2,643 (-71%) |
| Small (SM118, SM113) | 2,545 | 529 (-79%) / 785 (-69%) / 1,297 (-49%) | 1,315 | 307 (-77%) / 435 (-67%) / 691 (-47%) |
| six-layer Tiny (SM130) | 1,315 | 307 (-77%) / 435 (-67%) / 691 (-47%) | 1,315 | 307 (-77%) / 435 (-67%) / 691 (-47%) |

Big at rank 4 would be about 3.5 times smaller than int8 rows at no measured cost; Small and Tiny at rank 4 about 1.9 times smaller.
