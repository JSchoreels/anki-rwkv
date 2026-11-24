# Smaller shift vectors and factors on T24 states (SM152): result

## Question

On top of the T24 state format (rank 2 then 4 from 16 answers, int8 factors; the user's 11-23 choice), can the shift vectors go to int4 (S4), the factors to int6 (F6), or both (F6S4), and still stay inside the cost budget? Budget, as a 95% CI upper bound per head on the 722 rule users of the 732 set: Big at most +0.15% vs its own int8; Small at most +0.5% and Tiny at most +1.0% vs the matching Big model at int8. The GPU part is done (723 users scored per label, 718 for Big; the 5 largest users never fit).

## Result per size (go pick)

- Big (Big two-head (SM91s2) and Big Instant-Only (SM99)): **F6S4**, state bytes -87.4% vs int8 and -30.9% vs T24 alone. All four labels pass. Worst upper bound is +0.125% (Big Instant-Only (SM99), instant head) against +0.15%.
- Small (Small two-head (SM118) and Small Instant-Only (SM113)): **F6**, bytes -71.4% vs int8 and -15.3% vs T24 alone. S4 and F6S4 fail for Small two-head (SM118): upper bounds vs Big two-head (SM91s2) at int8 are about +0.51% (curve) and +0.38% (instant) for F6S4, +0.51% and +0.36% for S4; the curve head of F6S4 is the one over +0.5%. F6 passes with +0.409% (curve) and +0.238% (instant) for Small two-head (SM118), and +0.106% for Small Instant-Only (SM113). T24 alone has +0.405% / +0.227% / +0.096%.
- Tiny (six-layer Tiny two-head (SM130) and six-layer Tiny Instant-Only (SM130)): **F6**, bytes -69.7% vs int8 and -14.4% vs T24 alone. S4 and F6S4 fail (upper bounds +1.01% to +1.11% vs Big at int8, over +1.0%). F6 passes with +0.911% (curve) and +0.826% (instant) for six-layer Tiny two-head (SM130), and +0.739% for six-layer Tiny Instant-Only (SM130). T24 alone has +0.899% / +0.802% / +0.717%.

## What each new step costs (vs T24 alone, point estimates, per head)

| model | F6 curve / instant | S4 curve / instant | F6S4 curve / instant |
|---|---|---|---|
| Big two-head (SM91s2) | +0.010% / +0.015% | +0.031% / +0.040% | +0.038% / +0.056% |
| Big Instant-Only (SM99) | - / +0.015% | - / +0.041% | - / +0.056% |
| Small two-head (SM118) | +0.005% / +0.011% | +0.091% / +0.135% | +0.095% / +0.145% |
| Small Instant-Only (SM113) | - / +0.010% | - / +0.251% | - / +0.260% |
| six-layer Tiny two-head (SM130) | +0.011% / +0.023% | +0.110% / +0.199% | +0.121% / +0.219% |
| six-layer Tiny Instant-Only (SM130) | - / +0.023% | - / +0.363% | - / +0.385% |

Int6 factors are almost free (at most +0.023%). Int4 shift vectors are cheap for Big and clearly costly for Small and Tiny, which is why they fail the budget there. The fused GPU kernel is slower than the plain REF path for all labels, about 0.36x for Big, 0.37x to 0.46x for Small, 0.22x to 0.28x for Tiny; F6 and S4 run at the same speed as T24 or slightly below (Small two-head (SM118): 47.3k reviews/s for F6 vs 56.9k for T24, 124.6k for REF).

## State bytes vs int8 (card mean)

| size | T24 | S4 | F6 | F6S4 |
|---|---|---|---|---|
| Big | -81.72% | -84.49% | -84.60% | -87.36% |
| Small | -66.26% | -70.38% | -71.44% | -75.56% |
| Tiny | -64.65% | -69.52% | -69.74% | -74.60% |

## Where the cost sits

By answer count, the F6 cost vs own int8 rises slowly with history (for example Small two-head (SM118) instant: +0.017% at 1-2 answers, +0.098% at 9-15, +0.070% at 128+). The only visible outlier is F6S4 on the 128+ bucket of Small two-head (SM118), curve head: +0.579% vs own int8 (+0.696% vs T24), but this bucket is small and F6S4 already fails there for other reasons. Full bucket tables are in `sm152-result.md`.

## Integrity

CPU: REF, T24 and Q2 rescored on the 3 smoke users equal the earlier T24 test (SM148) rows (largest difference 0). GPU: REF and T24 equal SM148's GPU smoke rows within 2.0e-6. Coverage 723 users per label (718 for Big), at least 715 for every comparison.

## Long-card and per-card warnings (CPU, finished 11-24 10:18)

These are warnings only and do not change the go rule. They were scored on the CPU for the 50 long-card users, for Small and Tiny only (Big was not run): 600 rows, 0 killed, 0 failed. A long card is one with 128 or more answers before the review. Flag rule for the long-card cost vs own int8: point estimate above +0.5% or 95% CI upper bound above +1.0%. Flag rule per card: the share of cards with at least one review whose |dR| vs int8 is above 0.10 must stay at 1% or less.

- **Int6 factors (F6), the go pick for Small and Tiny, have one flagged cell.** It is the curve head of six-layer Tiny two-head (SM130) on long cards: +0.205% [-0.129, +1.852] vs own int8, where T24 alone has +0.002% [-0.085, +0.371]. The point estimate is small but the upper bound is over +1.0% (the 50 long-card users are few, so the interval is wide). Every other F6 cell is unflagged: Small two-head (SM118) curve -0.158% [-1.112, +0.186] and instant +0.016% [-0.122, +0.196]; Small Instant-Only (SM113) +0.061% [-0.002, +0.315]; six-layer Tiny two-head (SM130) instant -0.034% [-0.121, +0.246]; six-layer Tiny Instant-Only (SM130) +0.071% [+0.010, +0.302].
- **Per-card shares for F6 are tiny**: at most 0.107% of cards have a review with |dR| above 0.10 (six-layer Tiny Instant-Only (SM130)); the others are 0.004% to 0.083%. None is near the 1% flag.
- **Int4 shift vectors (S4 and F6S4) are flagged per card** for Small Instant-Only (SM113) (S4 1.199%, F6S4 1.273%), six-layer Tiny two-head (SM130) (F6S4 instant 1.063%; S4 instant 0.943% is just under) and six-layer Tiny Instant-Only (SM130) (S4 3.247%, F6S4 3.459%). Small two-head (SM118) stays under 1% (at most 0.298%). These labels already fail the budget for Small two-head (SM118) and for both Tiny models, so this only confirms that int4 shift vectors are not a good idea below Big.
- **Long-card flags for int4 shift vectors**: F6S4 for Small two-head (SM118) curve is +1.297% [-0.033, +7.789] and for six-layer Tiny two-head (SM130) curve +0.460% [-0.199, +3.083], both flagged by the upper bound. S4 is unflagged on long cards.
- Median per-review |dR| is 0.002 to 0.008 and the 99th percentile 0.023 to 0.079 for every label; the largest single review shift is 0.76 (six-layer Tiny two-head (SM130), S4 curve). The full table is in `sm152-result.md`.

## What it means for the fork

Big can take F6S4 (int6 factors and int4 shift vectors) on top of T24 for -87.4% state bytes. Small and Tiny should take F6 (int6 factors only, T24 shift vectors kept at int8) for -71.4% and -69.7% bytes. Int4 shift vectors are not worth it below Big. The one long-card warning for F6 is the curve head of six-layer Tiny two-head (SM130) on 128+ answer cards (upper bound +1.852%, point +0.205%); worth a look but not a reason to drop F6, since T24 alone is +0.371% there.

## GPU time

Total 436 minutes over 11 tickets: smoke 1 min; Tiny 80 (77 + 2 + 1); Small 107 (105 + 1 + 1); Big 248 (156 + 4 + 81 + 7).
