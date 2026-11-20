# Six-layer Tiny (SM130), io: decision

Verdict: **go** (pre-registered: every trained head's 95% CI upper bound of 100 x (six-layer / Small - 1) <= +1.00%, published w8+s8 format, show-time inputs, 722 users of the 732 set). Compact format (w8+s4c8nhx8): **go**.

| format | head | six-layer | vs the longer Small Instant-Only model (SM113) % [CI] | vs the longer cool-down for Tiny (SM115) % [CI] |
|---|---|---|---|---|
| published w8+s8 | instant | 0.27749 | +0.564% [+0.522, +0.607] | +0.378% [+0.347, +0.412] |
| compact w8+s4c8nhx8 | instant | 0.277597 | +0.603% [+0.560, +0.647] | (vs six-layer published +0.039% [+0.032, +0.046]) |
