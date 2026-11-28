# Tiny cool-down with session inputs (SM183): result

Finished Sat 2026-11-28 17:08 (`SM183 DONE`). Tables: `sm183-result.md`, `sm183-result.json`. Branch `sm183`.

## Question

The session inputs screen (SM179) and its confirmation (SM181) found, after a short fine-tune of about 2,550 updates, an instant LogLoss change of about -0.065% for six-layer Tiny two-head (SM130), with the curve unchanged. This run asks how large the gain is when the six session inputs are trained through a full cool-down of the same length and recipe as the release cool-down of six-layer Tiny (SM130), and whether that makes a deployable candidate.

## Caveat on the scoring format

All scoring used the reference format REF with int8 states (w8+s8hx8: int8 weights, int8 card and note state, int8 shift vectors), on show-time inputs. The shipped T24 and 6-bit formats have not been checked for these models and remain to be checked.

## Control

The reproduction check G0 passed (`check-ctl-sm183.json`): SM130's own pass 1, trained unchanged in the run tree, matched SM130's pass 1 at its 400-user snapshot bit for bit (210 weight tensors, 368 optimizer tensors, and all 1,189 per-update log records). So the release models, six-layer Tiny two-head (SM130) and six-layer Tiny Instant-Only (SM130), are the controls, and the only difference between an arm and its control is the session inputs.

## Rules (pre-registered in `design.md`)

A model passes on a set if its instant LogLoss change is at most -0.05% with the upper end of the 95% confidence interval below zero and, for the two-head model, the curve change is at most +0.05%. The verdict is GO if it passes on the 732 set and on the decisive users, PARTIAL if on only one, and NO GO if on neither. The Instant-Only model's curve head is not trained, so its curve is reported only. If the decisive users were not scored, the verdict is the 732 set's and is marked "732 set only".

## Verdicts

- Six-layer Tiny two-head (SM130): **GO**, passing on both sets.
- Six-layer Tiny Instant-Only (SM130): **GO (732 set only)**. Its decisive scoring was skipped by the budget gate (682 GPU minutes used, more than the 680 limit).

## Change versus the control (paired user bootstrap, 95% CI)

| model | set | users | curve | instant |
|---|---|---|---|---|
| six-layer Tiny two-head (SM130) | 732 set | 723 | -0.029% [-0.047, -0.008] | -0.499% [-0.578, -0.420] |
| six-layer Tiny two-head (SM130) | decisive users | 761 | -0.040% [-0.051, -0.030] | -0.474% [-0.573, -0.396] |
| six-layer Tiny Instant-Only (SM130) | 732 set | 723 | +0.001% [+0.001, +0.001] (not trained) | -0.589% [-0.699, -0.484] |

Control LogLoss to arm LogLoss: two-head, 732 set, curve 0.30339 to 0.30330, instant 0.28007 to 0.27867; two-head, decisive users, curve 0.30359 to 0.30347, instant 0.28115 to 0.27982; Instant-Only, 732 set, instant 0.27951 to 0.27787.

The instant gain is about -0.5%, roughly seven to eight times the -0.065% of the short fine-tune (SM179 / SM181), while the curve stays essentially unchanged (a very small improvement, well inside the rule's bound). The full cool-down therefore learns much more from the inputs than the short fine-tune did.

## Answer-count buckets (report only)

For two-head on the 732 set, the instant gain is steady at about -0.47% to -0.51% for cards with 1 to 32 prior answers, and smaller (-0.26%) for the 128 bucket; the curve gain is -0.02% to -0.04% throughout. The 300 and 1000 buckets hold very few reviews (2,723 and 5,997) and are noisy; on the decisive users the same buckets stay within about -0.7% to +0.2%. Instant-Only shows the same pattern (about -0.53% to -0.64% for 1 to 32 prior answers). Full per-bucket numbers are in `sm183-result.json`.

## AUC (report only, not part of the rule)

Within-user instant AUC rises by +0.250% (0.8418 to 0.8439) for two-head on the 732 set, +0.283% on the decisive users, and +0.272% for Instant-Only on the 732 set. Overall instant AUC rises by about +0.2%. Curve AUC is essentially unchanged (within-user +0.020% and +0.042% for two-head).

## Training cost

Each arm pass processed 581,942,724 reviews. Reviews per second and minutes per pass against the release passes:

| pass | arm | release (SM130) |
|---|---|---|
| two-head pass 1 | 64,866 /s, 149.5 min | 64,547 /s, 150.3 min |
| two-head pass 2 | 64,479 /s, 150.4 min | 64,303 /s, 150.8 min |
| Instant-Only pass 1 | 64,833 /s, 149.6 min | 65,967 /s, 147.0 min |
| Instant-Only pass 2 | 64,853 /s, 149.6 min | 65,706 /s, 147.6 min |

Peak allocated memory was 4,224 and 4,226 MiB, the same as the release passes (4,209 and 4,211 MiB). No pass was resumed. Total GPU time used was 682 minutes (about 11.4 hours).

## Next steps

Check the shipped T24 and 6-bit formats on these models, score the Instant-Only model on the decisive users, write the fork's feature builder for the six columns with test vectors, and run the same cool-down for Small two-head (SM118).
