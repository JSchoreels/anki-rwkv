# Decision log (complete)

The working record behind [DECISIONS.md](DECISIONS.md). Newest first. Each entry: who decided, the decision, why,
evidence (experiment, branch, commit), scope, how to undo it, and status. "You" = Jonathan; "Agent" = an obvious
call the agent took alone (allowed since 11-24: obvious decisions need no approval, but every one is logged here).

What still needs Jonathan, always: releases, tags, release assets, Hugging Face uploads, the fork release build,
any software install, anything touching the private Anki profile, and any real trade-off
(accuracy vs size or speed beyond the agreed budgets, scope changes to a release).

Conventions: every model as "plain name (ID)"; costs are LogLoss changes in %, CI upper bounds are 95% paired user
bootstrap; "732 set" = the standard 732-user evaluation pool (722 to 723 rule users).

---

## 2026-11-27

### D-1127-01 - Data-scaling test on the Small (SM176) read-out (Agent)
- Decision: log the result; no model is shipped (restricted training data).
- Result: vs the same-budget teacher run (SM176 n8000-kd, +1.19% / +1.34% vs the release Small, SM118): 1,000 users +1.36% / +1.01%, 2,000 users +2.08% / +1.93% (stopped after 2 epochs, outlier), 4,000 users +0.21% / +0.21% (curve / instant). Users needed: ~2,982 for +1.0%, ~3,590 for +0.5% (interpolation 2,000 to 4,000); ignoring n2000 about 1,550 / 2,830. Tiny (SM174): ~1,182 / ~3,194.
- Evidence: `artifacts/small-model/sm176/RESULT.md` on branch `sm176`.
- Undo: nothing to undo. Follow-up candidate: rerun n2000 with longer patience.

## 2026-11-26

### D-1126-01 - Data-scaling test (SM174) design (Agent)
- Decision: six-layer Tiny (SM130 architecture) from scratch, no teacher, nested random subsets of 500, 1,000, 2,000,
  4,000 and 8,000 of the 8,000 training users, each under the same review budget (4 x 581.9M reviews); constant LR
  with early stopping on a 190-user held-out slice, then a one_minus_sin decay; a control with 8,000 users and the
  usual replay teacher under the same budget; scored on the 732 set vs the release six-layer Tiny two-head (SM130).
- Why: the 8,000-user arm and the control both fit the 1-2 day budget (~27 GPU h upper bound); the release model
  trained far longer, so only an equal-budget control separates the teacher's share from the budget's. Small
  (SM118 architecture) would not fit at about half the speed.
- Evidence: `artifacts/small-model/sm174/design.md` on branch `sm174`.
- Undo: stop the launcher (`gpu/STOP`); nothing is shipped from this test (restricted training data).
- Status: running.

## 2026-11-24

### D-1125-08 - Family release rwkv-family-2026-11-28 published (You)
- **Decision:** user "Yes" (publish, notice precise) and "yes perfect" on the notice wording.
- **Done:** GitHub release rwkv-family-2026-11-28 (Latest) on JSchoreels/anki-rwkv, 13:14 11-25: 6 zips + INTEGRATION.md +
  WEIGHTS-NOTICE.md + SHA256SUMS; downloaded back and all checksums OK. Fork main 6cb164606 (Tiny ensemble SM170,
  batched codec SM168, new notice; 1,607 Rust tests both suites); rc-2026-11 = 6cb164606; anki-rwkv 24978a40.
- **Still open:** Mac Python/TS tests + just check (user retests later); fork release build (needs the user); Anki
  permission for shipping weights to other users (anki.tenderapp.com, user's call); HF upload not done.

### D-1125-06 - Fork main merges and release packages rebuilt with the final notice (Agent, under D-1125-05 "You")
- **Done (fork):** `tiny-ensemble` (62f10ed42, the Tiny instant ensemble (SM170)) and `codec-batch` (39e8b308f, the
  batched low-rank codec (SM168), same bytes) merged into fork main; `WEIGHTS-NOTICE.md` replaced by the exact text of
  `artifacts/small-model/sm166/WEIGHTS-NOTICE.next.md`. Fork main = 6cb164606; `cargo test -p anki --lib` 1,607 passed
  with default features and with `--no-default-features`. `rc-2026-11` fast-forwarded to the same commit.
- **Done (packages):** `build_release.py` ships `WEIGHTS-NOTICE.next.md` verbatim (no build-time patching) and stops
  unless fork main holds the same bytes; `verify_release.py` checks every copy. The six-layer Tiny manuals (section
  A.2) and the notes describe the ensemble; the Big long-card read-out (SM165) replaces "running"; the speed section
  mentions the batched codec (SM168) as a fork speed change. All six zips pass from a clean extract; packaged weights
  and heads match fork main 6cb164606 byte for byte. The repository's own notice copies (top level and
  `artifacts/hf-family/repo-new/`) carry the same text.
- **Call taken:** the DRAFT heading and "publishing needs the user's OK" line were removed from `RELEASE_NOTES.md` so
  the file can be used as the release body as is; the status lives in `artifacts/small-model/sm166/README.md`.
- **Open for you:** confirm the notice wording; then the main session publishes.
- **Undo:** revert the fork merge commits / notice commit; rebuild with the old `run_all.sh`.

### D-1125-07 - Weekend release packages (SM166) built, not published (Agent)
- **Done:** six packages in the 13 November layout in `work/release-2026-11-28/dist` (uncommitted), built by
  `artifacts/small-model/sm166/build_release.py`. The packages are Big two-head (SM91s2), Big Instant-Only (SM99),
  Small two-head (SM118), Small Instant-Only (SM113), six-layer Tiny two-head (SM130) and six-layer Tiny Instant-Only
  (SM130). Every zip also has `WEIGHTS-NOTICE.md`, the answer-time head files of fork main eda153ddb and synthetic
  answer-time test vectors; `SHA256SUMS` covers all 8 assets.
- **Calls taken:**
  - The answer-time heads ship, since fork main uses one for every model.
  - Each `INTEGRATION.md` is a new Part A (this release: scores, F6/T24 state format, head, files, speed) plus the
    13 November manual of the same slot as Part B. This follows the 13 November precedent of a note on top of the
    sm92 manual.
  - `master.pth` is now in every package.
- **Evidence:** all weights and head files match fork main byte for byte (Big `weights.int8.bin` is not in the fork).
  The answer-time vectors equal the fork's own fixtures (largest difference 0). Every zip passes both checks from a
  clean extract (`artifacts/small-model/sm166/logs/verify-summary.json`).
- **Open for you:** publishing (tag, title, notes). The Big long-card check with 6-bit factors (SM165) is pending.
  The Tiny ensemble (SM160) is left out until you decide. The weights notice does not name `time_head.*.bin`.
- **Undo:** delete `work/release-2026-11-28`; nothing was published.

### D-1125-03 - Speed is no release blocker; codec batching after the release (Agent)
- **Result:** SM163 (anki-rwkv 97128eb8), quiet machine, median of 3 ABAB runs: full rebuild with the new default Big 14,346
  reviews/s (+38.8% vs int8 rows), Small 45,156 (-3.9%), Tiny 80,569 (-3.9%); 6-bit factors cost <= ~2% end to end.
- **Decision:** the pre-registered 15% rule triggers only on the one-thread long-history replay (Small -17.2%, Tiny -23.9%
  vs int8 rows); on the realistic rebuild the gap is 3.9%. Batching the codec across cards goes to the backlog for after
  the release instead of a late hot-path change now.
- **Undo:** start the batching now if the user wants it in the release.

### D-1125-05 - Tiny ensemble ships; everything to fork main; release approved pending notice wording (You)
- **Decisions (user ~13:00):** (1) the Tiny ensemble (SM170) goes in; (2) "push everything to main": tiny-ensemble and
  codec-batch (SM168) merge into fork main, the user retests on the Mac later and said to assume it is OK; (3) publish
  the release, provided the weights notice is precise that the weights stay under Anki's terms.
- **Notice:** the dataset LICENSE names no owner; it says the data was provided to the FSRS project for memory research
  and sends other uses to anki.tenderapp.com. New wording (artifacts/small-model/sm166/WEIGHTS-NOTICE.next.md): this
  project grants no license to the weights; they stay under the data's terms; other uses need Anki's permission via
  anki.tenderapp.com. Not worded as legal ownership by Anki because the license does not say that; user confirms.
- **Status:** fork merges + package rebuild running; gh release create after the user confirms the wording.

### D-1125-04 - Big long-card check with 6-bit factors (SM165) and release packages (SM166) started (Agent)
- **Why:** Big F6 was the only final format without a long-card read-out (warning only); packages are the next plan step
  (prepared only; publishing needs the user). The Tiny ensemble (SM160) is not in the packages until the user decides.

### D-1125-02 - Final state formats pass the release-candidate re-check (Agent, pre-registered rule)
- **Result:** SM164 (branch sm164, c30a0070): Big F6, Small F6, six-layer Tiny T24 all GO, 9 of 9 cells pass on the
  732 set and the 1,500 rule. Tightest: Small two-head curve +0.412% (ceiling +0.5%), Tiny two-head curve +0.908%
  (ceiling +1.0%), Big Instant-Only instant +0.084% (ceiling +0.15%).
- **Rule note:** on the 1,500 set Small/Tiny use upper(own int8 vs Big int8, 732) + upper(final vs own int8, 1,500),
  because ids >= 5000 are Big's training users (direct comparison there: Small two-head curve +1.6%, in-sample effect).
- **Open:** Big F6 has no long-card read-out (Big T24 passed, SM155). Next: SM163 speed, packages.

### D-1125-01 - 6-bit factors (SM162) merged into fork main (Agent, under D-1124-10 and the user's 11-24 choice)
- **Done:** fork main eda153ddb (merge of rc-int6 6bbd23b18; rc-2026-11 fast-forwarded to 6bbd23b18). Precision byte 0x26,
  default profile `compact_lowrank` (factor width per size from the card layers: Tiny int8, Big/Small int6),
  `compact_lowrank_int8`/`_int6` stay valid; cache version 16; old 0x28 states decode; downgrade rebuilds with int8.
- **Evidence:** both Rust suites 1,603 passed; 504/504 synthetic head writes bit-identical to the scorer; SM91s2 end to end
  on 2 synthetic users within 1.0e-4 per-user LogLoss (limit 3e-4); live vs rebuild identical bytes on every size.
  The one-ulp floor gate for low-rank formats was replaced by the LogLoss rule (int6 failed the floor, passes the rule).
- **Open:** Small has no end-to-end model check against the scorer (codec is bit-exact); Mac Python/TS tests + just check;
  clean speed re-measure SM163 (int6 decode 23-33% slower in a noisy sanity run).
- **Undo:** revert merge eda153ddb.

### D-1124-13 - Ensemble of the two instant predictions: go for Tiny only (Agent, pre-registered rule)
- **Result:** the deploy-time ensemble test (SM160), branch sm160 commit 3a5530cc. Logit average with a fixed weight on the
  Instant-Only model, against the Instant-Only model alone, instant head, chosen format (T24), the 732 set:
  six-layer Tiny (SM130) -0.067% [-0.072, -0.061] w 0.69 = go; Small (SM118 + SM113) -0.022% w 0.78 = no go;
  Big (SM91s2 + SM99) -0.012% w 0.80 = no go (also 5 Big users never fit in GPU memory, as in SM148).
- **Next:** a small fork change for Tiny's default two-model mode (one weight, w = 0.69); it touches the same fork files
  as the 6-bit factor work (SM162), so it waits until SM162 lands. Whether it ships in the weekend release is the user's call.
- **Undo:** nothing changed yet.

### D-1124-12 - 6-bit factor fork work (SM162) resumed (You)
- **Decision:** "yep implement it" (23:40). Agent resumes from the partial edits in wt/fork-int6 (branch rc-int6) and lands
  on fork main + rc-2026-11 when both Rust suites, the vectors and the end-to-end check pass.

### D-1124-11 - Hourly factory idle check; SM160 read-out and release-candidate re-check started (Agent, asked by You)
- **Why:** the deploy-time ensemble test (SM160) finished its GPU work at 15:26; the watcher that should have woken
  the main session died at the interrupt, so the GPU sat idle about 8 hours. User: "we should have a simple 1h/2h check".
- **Done:** `scripts/factory_idle_check.py` (78fcebe7) prints BUSY / QUIET / IDLE (exit 0 / 3 / 4) from the queue,
  nvidia-smi, live launcher/scorer processes and quiet.json. Session cron at :17 every hour runs it; IDLE means read
  out finished work and start the next backlog item. The cron is session-only: recreate it after any restart.
- **Also started (23:20):** SM160 read-out (CPU, Sonnet agent) and the release-candidate re-check of the final
  formats (SM164: Big and Small T24 + 6-bit factors, Tiny T24, 732 + 1,500 users, GPU).
- **Undo:** CronDelete the job; the script is harmless on its own.

### D-1124-10 - Fork work goes to fork main; release candidate merged into main (You)
- **Decision:** "For Anki you can just push straight to main, otherwise the other Mac won't see it". Finished,
  tested fork work is merged into fork `main` and pushed; `rc-2026-11` was only visible as a side branch.
- **Done:** fork main afa298cbe..fb9f59a18 (merge of rc-2026-11 b04e3d117 into main, which had 21 commits from the
  other machine). Only conflict: `RELEASE.md` bullet lists, resolved as the union of both sides; the older
  "Explore recall by answer time" bullet (still said estimates need the Small Curve model) dropped for the
  release-candidate version. Rust `cargo test -p anki --lib` 1,596 passed, `--no-default-features` 1,596 passed
  (run in `wt/fork-lowrank-speed` detached at fb9f59a18; the main checkout lacks the extracted `protoc`).
- **Still needs the user:** fork release build, releases, tags. Python/TS tests and `just check` on the Mac.
- **Undo:** revert merge fb9f59a18 (first parent afa298cbe).

### D-1124-09 - Old Tiny weights dropped from the weekend packages (Agent, confirmed by You)
- **Decision:** the weekend packages ship only the six-layer Tiny two-head and Instant-Only models (SM130). The
  older Tiny two-head (SM106), Tiny Instant-Only (SM101) and the Tiny Instant-Only candidate (SM115) are not packaged.
- **Why:** six-layer Tiny (SM130) replaced them in the family on 11-21; it has its own answer-time heads (SM138);
  the fork runs each size only with its own models, so nothing falls back to old weights. No real trade-off.
- **Scope:** weekend release packages and release notes ("six-layer Tiny replaces the 13 Nov Tiny"). The
  `rwkv-family-2026-11-13` release assets are not touched.
- **Undo:** add the old files back to the package list before upload.
- **Status:** decided; applies when packages are built.
- **Lesson:** this was asked as a question although it had no trade-off; obvious calls are now taken and logged.

### D-1124-08 - Per-user recalibration waits (You)
- **Decision:** rolling per-user Platt recalibration of the timed predictions ("Good or Easy answered within S s,
  reading the back included") is not in the weekend release; its fork work comes after the state-minimization work.
- **Evidence:** go for every model: rolling recalibration on the 732 set (SM125), for the Big models (SM128),
  for the six-layer Tiny (SM140, branch sm140 f7a32b48): all 21 timed cells (2 to 10 s) reliable after
  recalibration (raw 13 and 14 of 21 for the six-layer Tiny two-head and Instant-Only (SM130)); "Good or Easy
  within 4 s" ECE 0.0143 → 0.0016 and 0.0131 → 0.0022, LogLoss −1.24% and −1.09% vs raw.
- **Fork work needed later:** per-user Platt parameters, refit every 7 days on the latest 30,000 earlier reviews,
  raw model below 1,000 reviews, repair step as in SM125.
- **Status:** deferred to the next release.

### D-1124-07 - Training is done for this family (Agent)
- **Decision:** no more training for this family release. Release-candidate models: Big two-head (SM91s2),
  Big Instant-Only (SM99), Small two-head (SM118), Small Instant-Only (SM113), six-layer Tiny two-head and
  Instant-Only (SM130), with their answer-time heads.
- **Why:** all state-format work since 11-23 is post-training; the short rank-2 cool-down on Small (SM149) gave no
  useful effect, so no retrain is needed for T24 or 6-bit factors.
- **Next-family training ideas (parked):** training with a penalty on the state energy the truncation drops (the
  outside-research recipe that makes rank 2 nearly free), fixing the collapsed placeholder note ids of deleted cards
  in our pipeline (found by SM161: 485 cards on 35 notes; changes training inputs), a sharpness-aware cool-down,
  session inputs for the answer-time head.
- **Status:** in effect.

### D-1124-06 - Clean re-measure of state-building speed (Agent)
- **Decision:** run SM163 on a quiet machine (no GPU job) after SM160 ends and SM162 merges: `rc-2026-11` before
  and after, int8 / T24 / T24 + 6-bit factors per size: full first-launch rebuild of a large collection, per-review
  answer and show-time latency (Learning cards included), Deck Options replay with four threads, plus a query
  batch-size sweep (32 to 1,024).
- **Why:** every T24 speed number so far was measured with a GPU job running; since then the codec (SM156), the
  Learning fix (SM161) and 6-bit factors (SM162) changed the hot paths. Last under-load numbers (int8 / T24 before
  SM156 / T24 with SM156, reviews/s): Big ~3,960 / 3,316 / 4,219; Small ~16,900 / 11,075 / 13,804; Tiny ~32,200 /
  18,334 / 25,541.
- **Follow-up rule:** if Small or Tiny still trail int8 by more than about 15% on the clean run, batch the codec
  across several cards during the rebuild (SM156 write-up).
- **Status:** planned for the night of 11-24.

### D-1124-05 - 6-bit factors for Big and Small, Tiny stays on T24 (You)
- **Decision:** Big and Small: `w8+lr2.16.4q6x8` (T24 with int6 factors). Tiny: `w8+lr2.16.4q8x8` (T24).
- **Evidence:** smaller shift vectors and factors on T24 states (SM152, branch sm152 8f4604c7), 732 set:
  - Big two-head (SM91s2) 6-bit factors: curve / instant CI upper +0.056% / +0.076% vs own int8 (budget +0.15%);
    bytes −84.6% vs int8 (T24 −81.7%).
  - Small two-head (SM118): +0.409% / +0.238% vs Big two-head (SM91s2) at int8 (budget +0.5%); Small
    Instant-Only (SM113) +0.106%; bytes −71.4% (T24 −66.3%).
  - Six-layer Tiny two-head (SM130): +0.911% / +0.826% (budget +1.0%), Instant-Only +0.739%; bytes −69.7%
    (T24 −64.7%). Long-card warning: curve on 128+ cards +0.205% [−0.129, +1.852] vs T24 +0.002% [−0.085, +0.371].
  - Extra cost of 6-bit factors beyond T24: at most +0.023% on every model.
- **Fork work:** SM162 (fork branch `rc-int6` from rc-2026-11, worktree `wt/fork-int6`; write-up branch sm162).
  Approved to merge into `rc-2026-11` and push once every check passes (rc-2026-11 only).
- **Undo:** per-size default back to T24 int8 factors (the format keeps reading old layouts).
- **Status:** fork work running.

### D-1124-04 - Learning-time fix merged (You)
- **Problem (fork input parity, SM161, branch sm161 e8ab44d4):** the fork's history rebuild matches the training
  rows exactly (24 columns + 28 day inputs, 62,758 reviews of 8 benchmark users, max diff 8.3e-7), but the live
  path gave a Learning card shown on a later day Anki's whole-day learning interval instead of the real seconds since
  its last review. 4,258 of 51,970 show-time queries (8.2%); also the state written by the live answer.
  Cost: instant LogLoss +0.054% Small two-head (SM118), +0.059% Small Instant-Only (SM113) on 12 users.
- **Fix:** fork branch `rc-learning-elapsed` dcd0f3fcf: Learning added to the states that use the real elapsed time
  (`qt/aqt/rwkv_scheduler.py`), batch path calendar days for Learning (`rslib/src/scheduler/rwkv.rs`), stats-graph
  twin. 6 synthetic tests. Parity re-run with the fixed rules: live drift 4,258 → 0, batch 2,674 → 57 (the 57 are
  shown before the rollover, answered after).
- **Merged:** rc-2026-11 b04e3d117 (merge commit), pushed 11-24 ~13:30. Rust `cargo test -p anki --lib` 1,565
  passed, `--no-default-features` 1,565 passed.
- **Open:** fork Python/TS tests must run on the Mac (this PC's fork Python env is broken).
- **Undo:** revert merge b04e3d117's second parent.

### D-1124-03 - Faster low-rank codec merged (You)
- **Decision:** SM156 (fork 47163a997, bit-identical to the frozen reference: 9,900 layer writes + 1,332 solves)
  fast-forwarded into rc-2026-11, then the Learning fix merged on top (b04e3d117), pushed.
- **Undo:** revert 47163a997 (format unchanged).

### D-1124-02 - Not adopted: output-aware truncation, energy-triggered rank growth, 4-bit shifts (Agent)
- **Output-aware T24 on the 732 set (SM158, branch sm158 821f4845):** no go under the pre-registered rule.
  Weighted minus plain T24: Big two-head (SM91s2) instant −0.008%, Big Instant-Only (SM99) −0.012% (bar −0.05%);
  other heads −0.027% to +0.006%; half the GPU scoring speed; worse on 128+ cards on 8 of 9 heads.
- **Energy-triggered rank growth (SM159, branch sm159 2d758502):** no go; at matched bytes no better than T24,
  worse on the instant heads of Big two-head (SM91s2) +0.022%, six-layer Tiny two-head (SM130) +0.050%,
  six-layer Tiny Instant-Only (SM130) +0.043% (8 probe users). First run failed on a CRLF bug in the launcher,
  fixed in 283c5422.
- **4-bit shift vectors (SM152):** break the Small and Tiny budgets (Small two-head (SM118) curve upper +0.507%;
  Tiny +1.006% to +1.114%); per-card flags above 1% for Small Instant-Only (SM113) and both Tiny models.
- **Answer-time spread shrink (SM153):** no go (held-out −0.015% to −0.084%, every CI includes 0).

### D-1124-01 - Experiments started to avoid idle time (Agent)
- Instant-prediction ensemble (SM160, branch sm160): logit average of the two instant predictions the fork already
  computes, one weight per size fitted on 100 users outside the pools; go = CI below 0 and ≤ −0.05% vs
  Instant-Only alone. GPU, verdict ~14:00 to 15:00 11-24. If go, hold for the next release unless the fork change
  is trivial.
- Fork input parity (SM161): done, see D-1124-04.
- Bounded-memory scorer first use (SM155, branch sm155 cdb38316): Big long-card coverage 37 → 50 of 50 users;
  T24 on 128+ cards −0.016% / +0.026% (Big two-head (SM91s2)), +0.006% (Big Instant-Only (SM99)), uppers ≤ +0.11%.
  Launcher hung in its final `wait` on its own memory guard; released by hand (bug noted in its RESULT.md).
- SM152 long-card CPU part launched 06:46 once the CPU was free; done 10:18.

---

## 2026-11-23

### D-1123-04 - T24 for every size (You)
- "Yeah let's go T24": `w8+lr2.16.4q8x8` for all sizes, post-training ("do we need to retrain?" → no: QAT SM149
  did not help). Low-rank int8 states (SM148) T24 vs Big int8 (CI upper): Big two-head (SM91s2) +0.043% / +0.059%,
  Big Instant-Only (SM99) +0.065%, Small two-head (SM118) +0.405% / +0.227%, Small Instant-Only (SM113) +0.096%,
  six-layer Tiny two-head (SM130) +0.899% / +0.802%, Instant-Only +0.717%.
- "ok let's integrate in anki and push when finished": fork SM151 `rc-lowrank` merged into rc-2026-11 514f3a41d,
  pushed (rc-2026-11 only). Cache version 14 → 15 (one rebuild).

### D-1123-03 - Rank 2 acceptable; rank 1 rejected (You / evidence SM150)
- "I think we can implement Rank 2 for anki"; rank 1 (R1) and rank 1 then 2 (R12) fail every size (SM150).

### D-1123-02 - State-format budgets vs Big (You)
- Small ≤ +0.5%, Tiny ≤ +1.0% total vs the matching Big model at int8 (two-head vs Big two-head (SM91s2),
  Instant-Only vs Big Instant-Only (SM99)); Big ≤ +0.15% vs its own int8; CI upper per head. Long-card and
  per-card |dR| are warnings only. Replaces the +0.05% per-step bar for formats.

### D-1123-01 - No down time (You)
- "better explore stuff than to have to wait for me": start the next backlog item, tell the user, let them redirect.

---

## Earlier (backfilled from notes; see the factory page for evidence)

- 11-22 (You): cards with 128+ answers may lose about 0.5% (point) / 1% (CI upper) under int4/int8 tiers.
- 11-21 (You): release date 11-28 soft ("do things properly"); state format int8 card rows + compact notes for all
  sizes (robustness: no user without a valid prediction); six-layer Tiny (SM130) replaces the previous Tiny;
  simulator bias fix + full run approved.
- 11-20 (You): compact state format approved for the release candidate (superseded 11-21 by int8 card rows).
- 11-18 (You): Tiny may cost up to +1% vs Small (CI upper, both heads) to be clearly smaller.
- 11-16 (You): new models in the fork ahead of the release, with an answer-time head for every model.
- 11-13 (You): `rwkv-family-2026-11-13` published; 11-14: no new release soon.
