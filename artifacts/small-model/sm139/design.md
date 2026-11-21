# Release-candidate checks for the family (SM139): design

Pre-registered Sat 11-21 (about 11:45) before any GPU ticket. Branch `sm139`, worktree `C:/Users/jonat/workspace/wt/sm139`, cut from `origin/sm137` (934b6bf2), which holds the scorer with the formats of the compact state for Small and Big (SM131) and the long-card state drift on real users (SM136) (fork-kind per-stream state, `w8+s8hx8`, `--nonfinite-rows`) and the six-layer architecture of the six-layer Tiny (SM130) (`w64-c1-d2p1u1`). `origin/main` has no small-model code. No source file is changed by this run; everything new is under `artifacts/small-model/sm139/`.

## Goal

The next RWKV family release follows `rwkv-family-2026-11-13` (target about Sat 11-28, soft). Before packaging, check every release-candidate model in the release-candidate state format on more users than any single experiment used, and report what users get from the update against the published 11-13 models. User decisions 11-21: the state format is int8 card rows with float16 row scales and int8 shifts, plus the compact note format (scorer variant `w8+s8hx8`), for all three sizes; the six-layer Tiny (SM130), both chains, replaces the current Tiny candidates (the show-time two-head cool-down for Tiny (SM124), the longer cool-down for Tiny (SM115)). Every model ships with an answer-time head (SM117, SM119, SM138). The checks were planned for about Wed 11-25 and are pulled forward because the GPU is free.

## The release-candidate models (verified 11-21: master files hashed now)

| size | kind | model | scorer kind:arch | master (sha256) | source of the choice |
|---|---|---|---|---|---|
| Big | two-head | the show-time big model (SM91s2) | `fork:` | `wt/sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth` (`67ee8e9b0d0169733242c40646e0d2e5637845a25775762002ffc4e6396f9590`) | published 11-13 (manifest `checkpoint.sha256` equal); no newer Big candidate: the show-time retrain step 3 (SM91s3) had no gain |
| Big | Instant-Only | the show-time instant-only big model (SM99) | `fork:` | `wt/sm99/artifacts/small-model/sm99-big-inst-e2/expertium/big-inst/master.pth` (`c49e45d698194da033caa1231f598f875922a4a2a3e921714a3fd433d37e5768`) | published 11-13 (manifest equal); the big instant-only teacher (SM110) was no go; the fork branch `rwkv-time-heads-family` maps Big Instant-Only to `sm99` |
| Small | two-head | the show-time two-head cool-down for Small (SM118) | `expertium:w64` | `wt/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/E89-F65e-F77a/master.pth` (`b4fe1175d25a7e6abff70e87ac0af1f4d76db9622d54f7cb53196fbf35117eb6`) | factory plan row: go 11-17 (-0.114% curve, -0.100% instant vs SM92), held for this release; SM131 / SM136 used the same file |
| Small | Instant-Only | the longer Small Instant-Only model (SM113) | `expertium:w64` | `wt/sm113/artifacts/small-model/sm113-inst-e35st-st-dec2ep32k-t99-p2/expertium/E89-F65e-F77a/master.pth` (`e4574e2aa8b2cfe6360dbd86f2a75b71d23ea81677b00ffcc9c3abce6b4cb736`) | go 11-14 (-0.094% vs SM98), held for this release; already the fork branch's Small Instant-Only (`sm113`) |
| Tiny | two-head | the six-layer Tiny two-head model (SM130) | `expertium:w64-c1-d2p1u1` | `wt/sm130/artifacts/small-model/sm130-th-dec2ep32k-p2/expertium/E89-F65e-F77a/master.pth` (`48a5e724cf38b53a32d570d4d034791f1da2cfcbe878572ff03ad434d7fb75af`) | user decision 11-21; pass 2 master, the same file as the answer-time heads for the six-layer Tiny (SM138) |
| Tiny | Instant-Only | the six-layer Tiny Instant-Only model (SM130) | `expertium:w64-c1-d2p1u1` | `wt/sm130/artifacts/small-model/sm130-io-dec2ep32k-p2/expertium/E89-F65e-F77a/master.pth` (`23f037282379199bd113a5297737c920fcbefd702b7b76eecc0095b8a31186a2`) | user decision 11-21; as SM138 |

The published 11-13 models (README of `artifacts/hf-family/repo-new`, manifests; scored here at their published format `w8+s8`): Big = the same two models as above; Small = the show-time retrain of the release (SM92) (`wt/sm92/.../sm92-e26-dec-1ep32k-muon/.../master.pth`, `2091f8a8...`) and the Small Instant-Only model (SM98) (`565b105e...`); Tiny = the Tiny two-head model (SM106) (`e9d0d0a0...`) and the Tiny Instant-Only model (SM101) (`d1408ae9...`). Every published sha256 equals its manifest's `checkpoint.sha256` (the Small two-head package has no `master.pth`; SM92's training master matches the manifest). The full table with paths is `models.sh` (read by the launcher and by `sm139_report.py check`).

### Open questions (not guessed; nothing in this run depends on them)

1. **Which Tiny is "published".** The brief named the show-time two-head cool-down for Tiny (SM124) and the longer cool-down for Tiny (SM115) as the published Tiny models. They were the Tiny candidates for this release; the published 11-13 Tiny models are the Tiny two-head model (SM106) and the Tiny Instant-Only model (SM101) (README "The family", manifests), and Small's are SM92 and SM98. The update comparison uses the published 11-13 models. SM130 vs SM124 / SM115 is already in SM130's results (+0.349% curve / +0.340% instant two-head, +0.378% Instant-Only on the 732 set, w8+s8).
2. **Rolling recalibration.** Rolling recalibration on the 732 set (SM125) and for the Big models (SM128) were go for the release candidates of that time. Whether a recalibration ships in this release (and for the six-layer Tiny, which it never saw) is not decided here; these checks score the raw models.
3. **The fork mapping.** The fork branch `rwkv-time-heads-family` still maps Small two-head to `sm92` and Tiny to `sm106` / `sm115`; the six-layer Tiny needs fork engine support (layer counts compiled in) and 346-input heads (SM138). Packaging is the fork agent's step; not checked here.

## Users

- **The 732 set:** `wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json` (723 users = the 732 set minus the 9 never-fit). The rule users are the 722 of `showtime_report.py pools()` (sha256 pinned), as in every 732-set readout.
- **The extension:** `wt/sm57/artifacts/small-model/sm57/stage4/users-sm93-score.json` (771 users, 56.0M reviews): the 768 validation users of the 1,500 pool (`scripts/eval_pools.py --pool 1500`) outside the 732 set, minus the 3 above 650,000 reviews, plus 6 users of the 732 fit list as a row check. This is the pool of every earlier "1,500 validation users" check (the instant model check (SM93), SM113, SM115, SM118, SM124: 1,487 users scored). The 2,225 pool is not used: it contains training users of every final-tier model. The extension users (ids >= 5000) were never seen by the Small and Tiny models, but they are training users of the two Big models, so Big's extension numbers are a format and robustness check only (stated with every Big number).
- Analysis users: "732" = 722, "ext" = the 765 extension users outside the 732 fit list, "all" = 1,487. Both users files are sha256 pinned (`design-pins.json`).

## Scoring (GPU, no training)

`scripts/small_model_heldout.py --pool 732|1500 --users-file F --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows` (`--pool 732` for the 732 set and the smoke, `--pool 1500` for the extension, as the SM124 extension tickets), w8 weights, labels `<tag>/<variant>@e2s-query`: the six release candidates at `w8+s8` (published format) and `w8+s8hx8` (release-candidate format), the four published Small / Tiny models at `w8+s8`. One ticket per size and pool (all labels of a size in one process), Tiny, Small, Big on the 732 set first, then the extension. GPU memory cap 0.8; a pass without progress switches to 0.92 (memory only, same numerics), as in the trained smaller state for Tiny (SM137). Rows: `gpu/sm139-rows-732.jsonl`, `gpu/sm139-rows-ext.jsonl`.

**Smoke (first ticket, STOP on failure):** the 2 smallest users of the 732 fit list (2388, 4676, `users-smoke.json`), all 16 labels on the GPU vs the CPU rows of this tree (`cpu/smoke-cpu-rows.jsonl`, run 11-21 before the launch, 32 rows, all finite, 96 s): every (user, label, head) benchmark LogLoss within 1e-5 (the strict rule of AGENT.md for the int8 variants; SM136's GPU smoke had `w8+s8hx8` within 5e-6 on long-card users).

## Checks (`sm139_report.py`, rules fixed before any GPU ticket)

Statistic: review-weighted LogLoss (weights = the reference rows' benchmark sizes), % = 100 x (new / reference - 1) of the reference's own LogLoss, paired user bootstrap 2,000, seed 0, 95% CI. Trained heads: two-head = curve + instant; Instant-Only = instant (its curve head is untrained and never reported).

- **C1 every user finite:** per release candidate, no `nonfinite` row and every benchmark LogLoss finite at `w8+s8hx8` on both pools.
- **C2 format cost:** cost_h = 100 x (the model at `w8+s8hx8` / the same model at `w8+s8` - 1), per trained head, on the 732 set and on the extension separately, over the users with rows in both formats. **Pass iff the 95% CI upper bound is <= +0.05% on both pools for every trained head.**
  - Why +0.05%: the earlier format experiments measured costs of about +0.001%: compact vs fork default for Small and Big (SM131) between -0.001% and +0.001%, every CI upper bound <= +0.002%; the trained smaller state for Tiny (SM137) at `w8+s8hx8` vs `w8+s8` +0.001% (0.277522 vs 0.277518); int8 card rows on the reviews of cards with 128+ prior answers (about 0.8% of reviews) at most +0.02% mean cost, every CI upper bound <= +0.12% (SM136), which is <= +0.001% dataset-wide. +0.05% is 25 times the largest measured upper bound and equals the repository's minimum effect size, so a pass means "no cost a user could notice", and a fail means something differs from the earlier measurements (a bug or a model-specific overflow) and must be investigated before packaging.
- **C3 coverage:** >= 720 of the 722 rule users and >= 760 of the 765 extension users with rows in both formats (earlier runs scored 722 and 761: 767 of 771 including the 6 check users; fewer users = GPU memory, not a model property).
- **Status:** pass iff every model passes C1 and C2 (and C3); incomplete if only C3 is missed; fail otherwise. A fail is reported to the user; it does not stop the release by itself.

## Report only (no go / no-go): what users get from the update

Per size and kind: the release candidate at `w8+s8hx8` vs the published 11-13 model at `w8+s8`, per trained head, on 732 / ext / all, with LogLoss levels and users better / worse; plus the model-only part (both at `w8+s8`). For Big the two are the same models, so the update is the format only. Row checks (report only): the 6 check users' extension rows vs their 732 rows; this tree's rows vs recorded rows of earlier runs with the same label (SM91s2, SM118, SM113, SM130, SM137, SM92, SM106, SM124 row files).

## Time heads (item 5)

Not rescored. The answer-time heads (SM117, SM119, SM138) were trained and scored on the fp32 trunk with fp32 state (`sm117_head.py`: `set_state_bits(model, 0)`); scoring P(recall within S) at `w8+s8hx8` would need the state codec inside the head's feature capture, a code change and a new check, not a cheap rescoring. The heads read the trunk's residual features and the instant head's hidden layer; the effect of the state format on those features is what C2 measures on the grade heads (+0.001% level), so the head numbers carry over within that noise. A rescoring at the release format stays an idea for after the release (recorded in HANDOFF.md).

## GPU time

Scoring speed from earlier tickets (723 users of the 732 set, 42.8M scored reviews): a Small or Tiny model 11-13 min per process with 1-2 labels (SM118, SM124, SM137 tickets); Small labels cost about 5.6 model-minutes each, Tiny 2.4; Big about 9.2 model-minutes per label (SM131 rows) and about 17 min elapsed per label. The extension has 56.0M reviews (x 1.31; SM118 / SM92 / SM124 extension tickets: 15-16 min + 3 short retry passes).

| ticket | labels | estimate |
|---|---|---|
| smoke (2 users, 16 labels) | 16 | ~5 min |
| Tiny, 732 set | 6 | ~25-30 min |
| Small, 732 set | 6 | ~40-50 min |
| Big, 732 set | 4 | ~70-80 min |
| Tiny, extension | 6 | ~35-40 min |
| Small, extension | 6 | ~55-65 min |
| Big, extension | 4 | ~90-105 min |
| retry passes (~2 per ticket, ~2 min each) | | ~25 min |
| **total** | | **~6-7 h of GPU** |

Launched about 12:00 Sat 11-21: result about 18:30-19:30 Sat 11-21, then the CPU report (seconds).

## Files

`models.sh` (model table), `users-smoke.json`, `design-pins.json`, `cpu-proof.sh` + `cpu/smoke-cpu-rows.jsonl` (CPU proof), `sm139_report.py` (checks and report; `selftest` passes on synthetic rows: a 0.0001% cost passes, a 0.2% cost and a non-finite user fail), `launch-sm139.sh` (DRYRUN=1, STOP file, resume, CR stripped from every Python output bash reads, quiet.json windows only, `done (status ...)` / `STOP: ...`, a final check that every output exists). Outputs: `gpu/` rows, `gpu/launch-sm139.log`, `gpu/tickets.jsonl`, `gpu/smoke-check.txt`, `sm139-result.{json,md}`.
