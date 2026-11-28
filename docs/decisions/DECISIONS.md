# Decision log (short)

One line per decision, newest first. **You** = decided by Jonathan; **Agent** = obvious call taken by the agent
and logged here for review. Details, evidence and how to undo each one: [DECISIONS-full.md](DECISIONS-full.md).

## 2026-11-28

- **Agent** - Tiny two-head full cool-down with session inputs (SM183): go, -0.499% instant / -0.029% curve vs the
  release Tiny on the 732 set (-0.474% / -0.040% on 761 more users); within-user instant AUC +0.25%.
- **Agent** - Session inputs confirmed (SM181): -0.063% instant on 761 more users, -0.065% for Small, AUC up slightly;
  they join the next family's recipe.
- **Agent** - AUC ceiling (SM182): ranking-only training gains at most +0.12% within-user AUC; LogLoss training leaves
  almost nothing; ranking losses dropped.
- **Agent** - 4-bit weights (SM180): free only for Small after a short retraining; not adopted now (weights are
  already small next to card states).
- **You** - Measure AUC too; run an AUC-ceiling screen (SM182, ranking loss).
- **Agent** - Queued the session-inputs confirmation (SM181, 765 more users + Small, AUC reported) and SM182.

## 2026-11-27

- **Agent** - Session inputs (SM179): go, -0.065% instant LogLoss on a short Tiny fine-tune; card state no go.
  Next: confirm on 1,500 users and Small; candidate input for the next family.
- **You** - Keep exploring features for LogLoss and model size while donors arrive (supersedes the idle-GPU note above).
- **Agent** - Started session inputs (SM179, LogLoss) and 4-bit weights (SM180, size).
- **You** - Keep the GPU idle until more donors arrive; then a first MIT Tiny run (Small 2,000-user rerun parked).
- **Agent** - Release models on donor data (SM178): donors about 20% / 27% harder (curve / instant), family ranking holds.
- **Agent** - Donor data: reviews dated before 2006 (a wrong clock, a zeroed id) are dropped by the converter; they
  crashed the scoring of the release models on donor data (SM178), which keeps its 218-donor snapshot, fixed in place.
- **You** - Fushi joins the data campaign as a fourth credited source; their first batch (3 donors, real data) is
  included with their OK. Format accepted (fetch_fushi_batch.py).
- **Agent** - GPU idle: started scoring the release models on all donor users (SM178), as recommended.
- **Agent** - Small data scaling (SM176): at 4,000 users without a teacher, +0.21% vs the same-budget teacher model;
  Milestone 2 (5,000 training users) is enough for an MIT Small.

## 2026-11-26

- **Agent** - GPU idle after SM174: started the Small data-scaling test (SM176) and the donations-to-training converter (SM177).
- **Agent** - Data scaling (SM174): a no-teacher Tiny comes within 1% at about 1,200 users and 0.5% at about 3,200;
  campaign milestones now show about 99.2% / 99.6% / 99.7% of today's accuracy.
- **You** - LMSherlock's shared collections (D:/Dropbox/Jarrett User Collections) join the data campaign, credited
  as author of FSRS; he is OK with MIT-licensed weights trained on them.
- **Agent** - Timed recalibration (SM175) merged into fork main 36c4f39d7 (1,614 Rust tests, parity 4.5e-9); Mac checks pending.
- **You** - Implement the per-user recalibration of the timed predictions in the fork now (SM175); recall predictions
  stay as they are (per-user recalibration gave nothing there: SM143, SM66, SM12).
- **You** - Goal: MIT-licensed weights trained only on donated data (from scratch, no teacher). Data campaign page
  with goals 732-user eval set, 7k users (5k/2k), 10k users (8k/2k); refreshed every 4 h.
- **Agent** - Started the data-scaling test (SM174): Tiny from scratch, no teacher, 500 to 4,000 users, to set the
  real donor goal.

## 2026-11-25

- **Agent** - Deleted-card note ids (SM173): harmless now (fork cost about +0.01% on existing cards); next family
  trains with correct 64-bit placeholder ids. No retrain screen.
- **Agent** - Started the deleted-card note ids check (SM173): measure first, retrain screen only if it matters.
- **Agent** - Sharpness-aware cool-down (SM172): no go (rank-2 cost shrinks at most 0.006%); dropped.
- **Agent** - Started the sharpness-aware cool-down screen (SM172, GPU), reusing SM171's control arm.
- **Agent** - State-energy penalty (SM171): no go; rank 2 gets cheaper only by losing about 0.3% at int8. Dropped.
- **You** - Published rwkv-family-2026-11-28 (13:14): six models, new weights notice (weights stay under Anki's terms);
  fork main 6cb164606 has the Tiny ensemble and the batched codec.
- **Agent** - Fork main 6cb164606: Tiny ensemble (SM170), batched codec (SM168) and the new weights notice merged
  (1,607 Rust tests each way); rc-2026-11 fast-forwarded. Release packages (SM166) rebuilt with the exact notice and
  the Tiny ensemble in the notes; all six pass from a clean extract; notes made publish-ready, nothing published.
- **You** - Tiny ensemble ships; everything goes to fork main (Mac retest later); release approved once the weights
  notice says clearly that the weights stay under Anki's terms.
- **Agent** - GPU idle since 10:06 with the release waiting on you: started the first next-family training screen, the
  state-energy penalty (SM171), which aims to make rank 2 nearly free.
- **Agent** - Tiny ensemble fork change ready on branch tiny-ensemble (62f10ed42, 1,605 Rust tests pass); not merged,
  waiting for your yes.
- **Agent** - Preparing the Tiny ensemble fork change on its own branch (SM170) so it is ready if you say yes; not merged.
- **Agent** - Tiny ensemble confirmed on 1,385 users (SM169): -0.062% [-0.066, -0.057], 82.8% of users improve;
  shipping it is still your call.
- **Agent** - Started the decisive 1,500-user check of the Tiny ensemble (SM169, GPU) so your ship decision has it.
- **Agent** - Batched codec (SM168) works: one-thread long-history gap Small -17.9% -> -3.2%, Tiny -24.2% -> -5.0%,
  bit-identical; fork branch codec-batch 39e8b308f, merged into fork main after the weekend release (AVX2 only).
- **Agent** - Started the batched low-rank codec (SM168) on its own fork branch; not merged before the release.
- **Agent** - Links between cards (SM167): no go; correction gains -0.0017% LogLoss (bar 0.05%); not pursued.
- **Agent** - Big long cards with 6-bit factors (SM165): inside the tolerance (at most +0.022%, CI upper +0.299%);
  no open accuracy check left before the release.
- **Agent** - Started the links-between-cards study (SM167, CPU only, reuses SM160 predictions) to answer your
  question on card connections; next family only.
- **Agent** - Weekend release packages (SM166) built, nothing published: six zips plus `INTEGRATION.md`, `WEIGHTS-NOTICE.md` and `SHA256SUMS`;
  answer-time heads included; weights identical to fork main; every zip passes its checks from a clean extract.
- **Agent** - Speed (SM163) is no release blocker: rebuild -3.9% vs int8 rows for Small and Tiny, Big +38.8%; codec
  batching (one-thread long cards trail 17-24%) waits until after the release.
- **Agent** - Started the Big long-card check with 6-bit factors (SM165, GPU) and the release packages (SM166, nothing published).
- **Agent** - Release-candidate re-check (SM164): all three final formats pass on both user sets; tightest
  Small two-head curve +0.412% (of +0.5%) and Tiny two-head curve +0.908% (of +1.0%).
- **Agent** - Hourly check found the GPU idle at 03:28 (re-check SM164 finished 02:40): started its read-out and the
  clean speed re-measure (SM163), as planned.
- **Agent** - 6-bit factors (SM162) merged into fork main (eda153ddb): Big and Small states -84.6% / -71.3% of int8 bytes,
  Tiny unchanged; 1,603 Rust tests pass. Mac Python/TS tests and the clean speed re-measure (SM163) still to do.

## 2026-11-24

- **Agent** - Instant ensemble (SM160): go for six-layer Tiny only (-0.067%, weight 0.69); Small and Big gain under 0.05%.
  The Tiny fork change waits for the 6-bit factor work (SM162); weekend inclusion is your call.
- **You** - Resume the 6-bit factor fork work (SM162).
- **Agent** - Hourly idle check (`scripts/factory_idle_check.py`, cron at :17) after the GPU sat idle 8 hours
  when a watcher died; started the ensemble test (SM160) read-out and the release-candidate re-check (SM164).
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
