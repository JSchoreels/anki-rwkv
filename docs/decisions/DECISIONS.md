# Decision log (short)

One line per decision, newest first. **You** = decided by Jonathan; **Agent** = obvious call taken by the agent
and logged here for review. Details, evidence and how to undo each one: [DECISIONS-full.md](DECISIONS-full.md).

## 2026-11-24

- **You** - Fork work goes straight to fork **main** (your Mac follows main only). The release candidate
  (rc-2026-11) is merged into fork main: fb9f59a18, Rust tests 1,596 passed. Release build and tags still need you.
- **Agent** - Old Tiny weights dropped from the weekend packages. The six-layer Tiny (SM130) replaces the
  13 Nov Tiny (SM106 two-head, SM101 Instant-Only); the 13 Nov release stays downloadable as it is. (You confirmed.)
- **You** - Per-user recalibration of the timed predictions (SM125, SM128, SM140) waits until after the
  state-minimization work; not in the weekend release.
- **Agent** - Training is done for this family; everything left before the release is fork work, checks and
  packaging. Training ideas (state-energy penalty, deleted-card note id fix, flatter cool-down) are for the next family.
- **Agent** - Clean speed re-measure of state building (SM163) after the format changes, on a quiet machine tonight.
- **You** - Big and Small move to T24 with 6-bit factors (−84.6% / −71.4% of int8 bytes, ≤ +0.023% beyond
  T24); Tiny stays on T24 because of a long-card warning. Fork work started (SM162).
- **You** - Learning-time fix merged into the fork release candidate (found by fork input parity, SM161):
  Learning cards shown on a later day got a whole-day interval instead of the real time (8% of show-time
  predictions, about +0.05% instant LogLoss for Small).
- **You** - Faster bit-identical low-rank codec (SM156) merged into the fork release candidate.
- **Agent** - Output-aware truncation (SM158) not adopted: −0.008% to −0.027% on instant heads, under the
  0.05% bar (pre-registered rule).
- **Agent** - Energy-triggered rank growth (SM159) not adopted: no better than T24 at equal bytes
  (pre-registered rule).
- **Agent** - 4-bit shift vectors not used: they break the Small and Tiny budgets (SM152).
- **Agent** - Started the instant-prediction ensemble test (SM160) and the fork input-parity check (SM161) to
  keep the GPU and CPU busy.

## 2026-11-23

- **You** - T24 (rank 2 below 16 answers, rank 4 after; int8 factors) for every size, post-training, no retrain.
  Merged into the fork release candidate.
- **You** - Rank 2 is acceptable for Anki (rank 1 failed every size, SM150).
- **You** - State-format budgets: Small ≤ +0.5% and Tiny ≤ +1.0% total vs the matching Big model at int8;
  Big ≤ +0.15% vs its own int8 (CI upper bound per head). Long-card and per-card checks are warnings only.
- **You** - No down time: explore rather than wait for answers.

## Earlier (backfilled)

- 11-22 **You** - Cards with 128+ answers may lose about 0.5% (point) / 1% (CI upper) under compressed tiers.
- 11-21 **You** - Weekend release date is soft: do things properly.
- 11-21 **You** - Release-candidate state format: int8 card rows + compact notes, for robustness (no user
  without a valid prediction).
- 11-21 **You** - Six-layer Tiny (SM130) replaces the previous Tiny in the family.
- 11-18 **You** - Tiny may cost up to +1% vs Small to become clearly smaller.
- 11-13 **You** - Family release `rwkv-family-2026-11-13` published.
