# Six-layer Tiny (SM130): result

Ended 01:21:03 Fri 11-20. Both chains go, compact format go (pre-registered rule: every trained head's 95% CI upper bound <= +1.00% vs Small, show-time inputs, 722 users of the 732 set, w8+s8).

## Two-head (vs the show-time two-head cool-down for Small (SM118); vs the show-time two-head cool-down for Tiny (SM124))
| format | head | % vs SM118 [CI] | % vs SM124 [CI] |
|---|---|---|---|
| published w8+s8 | curve | +0.467% [+0.398, +0.546] | +0.349% [+0.252, +0.450] |
| published w8+s8 | instant | +0.524% [+0.484, +0.567] | +0.340% [+0.310, +0.372] |
| compact w8+s4c8nhx8 | curve | +0.483% [+0.415, +0.562] | |
| compact w8+s4c8nhx8 | instant | +0.545% [+0.505, +0.589] | |

## Instant-Only (vs the longer Small Instant-Only model (SM113); vs the longer cool-down for Tiny (SM115))
| format | % vs SM113 [CI] | % vs SM115 [CI] |
|---|---|---|
| published w8+s8 | +0.564% [+0.522, +0.607] | +0.378% [+0.347, +0.412] |
| compact w8+s4c8nhx8 | +0.603% [+0.560, +0.647] | |

## Early checks (six-layer epoch minus the uncut epoch, continue every time)
- Two-head after epoch 30 (stop above +1.50%): curve +1.378% [+1.085, +1.657], instant +0.918% [+0.852, +0.990]. After epoch 32 (stop above +1.20%): curve +1.022% [+0.710, +1.288], instant +0.637% [+0.578, +0.701].
- Instant-Only after epoch 39 (+1.50%): +0.999% [+0.873, +1.135]. After epoch 41 (+1.20%): +0.286% [+0.242, +0.333].

## Size and cost
- Parameters 185,057 (Tiny 311,786). Scorer weight_bytes 226,191 (w8+s8 and compact). State bytes same as Tiny: 1,055 / 1,567 B fork default, 803 / 1,315 B compact.
- CPU review step (PyTorch, 1 thread): batch 1 1.062 vs 2.062 ms (x0.515); batch 256 33.4 vs 64.5 us per review.

## Throughput (581,942,724 reviews per pass)
- Recovery epochs: 177,800-182,500 reviews/s (53-55 min each).
- Cool-down passes: two-head 64,547 / 64,303 reviews/s; Instant-Only 65,967 / 65,706 reviews/s.
- 8k / 2k 10-epoch run at ~65,000 reviews/s: about 4.6 h.

## GPU time
Real run minutes ~1,000 (25 tickets). segments.jsonl sums 1,014 min but the Instant-Only epoch 41 segment (143.5 min) and its scoring ticket (117 min) include waiting; the log shows 53.2 min of training and ~10 min of scoring for them.
