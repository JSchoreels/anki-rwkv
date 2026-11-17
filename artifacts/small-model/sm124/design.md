# The show-time two-head cool-down for Tiny (SM124): design (2026-11-17)

## Question

The show-time two-head cool-down for Small (SM118) went go on 11-17: one learning-rate decay 7e-4 -> 0 stretched over two passes of the 8,000 users, from the show-time retrain of the release (SM92)'s constant-LR epoch 26, both heads trained, with the show-time big model (SM91s2)'s show-time curve and grade targets, beat the show-time retrain of the release (SM92) by -0.114% [-0.177, -0.062] on curve and -0.100% [-0.110, -0.091] on instant (722 users of the 732 set, show-time inputs). The same recipe went go for both instant-only models before it (the longer big instant-only teacher (SM113), the longer cool-down for Tiny (SM115)).

This run asks: does the same recipe lift the Tiny two-head model (SM106) on BOTH heads, with the same teacher and the same targets?

Unit everywhere ("%"): % of the big model (1 leak-free epoch)'s normal-input LogLoss of the same head on the same 722 users of the 732 set (curve 0.298966, instant 0.275757), as for the show-time two-head cool-down for Small (SM118). The Tiny two-head model (SM106)'s own decision used the big model's show-time LogLoss as the unit (curve 0.29958, instant 0.277157), so its published numbers are 0.2% (curve) to 0.5% (instant) larger in relative terms; the selftest reproduces them after rescaling. All runs are leak-free (deleted cards: neighbour), scored w8+s8, on show-time inputs (`e2s-query`). Whatever the result, it is held for the 11-28 release decision (no release, tag, fork or Hugging Face change).

| plain name (ID) | what it is |
|---|---|
| the show-time two-head cool-down for Tiny (SM124) | this design: the two-pass cool-down of the show-time two-head cool-down for Small (SM118) applied to the Tiny two-head model (SM106)'s pre-cool-down checkpoint, both heads trained, the show-time big model (SM91s2)'s targets |
| the Tiny two-head model (SM106) | the show-time retrain of the release (SM92)'s epoch 26 with one card layer cut (w64-c1), epochs 27-29 on show-time inputs (both heads), then the release's one-epoch 32k Muon cool-down on show-time inputs with the old targets of the big model (1 leak-free epoch); the Tiny two-head model of the family release; **the reference** |
| the show-time big model (SM91s2) | the big model (1 leak-free epoch) + one epoch on show-time inputs; **the teacher** (curve and grade targets) |
| the show-time two-head cool-down for Small (SM118) | the recipe copied here (go), on the Small architecture (w64) |
| the longer cool-down for Tiny (SM115) / the Tiny Instant-Only model (SM101) | the same schedule on the Tiny instant-only chain (go); instant read-outs only |
| the show-time retrain of the release (SM92) / the release (SM63) | read-outs only |

## Chosen design: the two-epoch cool-down from the Tiny two-head model (SM106)'s epoch 29

- **Start checkpoint:** the Tiny two-head model (SM106)'s constant-LR epoch 29, `wt/sm106/artifacts/small-model/sm106-c1-e29-st/expertium/E89-F65e-F77a/master.pth` (sha256 b3a694ca14cb, read only, w64-c1, both heads). It is the start of SM106's own cool-down (`sm106-c1-e29st-st-dec-1ep32k-muon`, decay.json `start_master_sha256`), update offset 640,262.
- **Teacher targets (reused, no export):** `anki-rwkv/artifacts/cache/kd-replay/sm91s2-big-showtime-final8k-neighbour-e2s` (5.5 GB), exported by the show-time two-head cool-down for Small (SM118) on 11-16 for exactly these 8,000 users (the cool-down users of SM106's recipe are the same: `sm77_warm.py cooldown --epoch 25`). Read only. C5 checks it is complete (`sm11_replay_export.py status`), that its manifest equals the one SM118 committed, that no file changed after SM118's pass 1 started, and re-runs SM118's `targets` check on all 8,000 users. The extra options file is byte for byte SM118's (= SM106's = SM92's except the teacher and the replay folder).
- **Pass 1** `sm124-c1-e29st-dec2ep32k-t91s2-p1`, prepared before the launch: `sm77_warm.py cooldown --epoch 25 --start sm106-c1-e29-st --arch w64-c1 --fraction 1.0` with this folder's `extra-options.json`, the same label-window trim (user 5599 x 14), then `sm124_prepare.py p1`: `lr_resume = {"total_reviews": 2R, "reviews_offset": 0, "segment": true}`, R = 581,942,724. Learning rate 7e-4 -> 0.293 x 7e-4. KD alpha 0.9 / decay alpha 0.5, both heads (no curve loss weight).
- **Pass 2** `sm124-c1-e29st-dec2ep32k-t91s2-p2`, prepared when pass 1 is complete: pass 1's final master.pth AND optimizer.pth (Muon state and update offset continue), the same rows in a new order (`random.Random(2).shuffle`), `lr_resume = {"total_reviews": 2R, "reviews_offset": R}`. Learning rate 0.293 x 7e-4 -> 0.
- **Pair proofs:** `sm124_pair.py p1`: pass 1 == the Tiny two-head model (SM106)'s cool-down (as prepared) except lr_resume, the targets and the folder name; AND == the show-time two-head cool-down for Small (SM118)'s pass 1 except the start (init_master, update offset 640,262 vs 574,028, start sha256 / source paths / the sm77 note) and the architecture (w64-c1 vs w64), each of which must equal SM106's cool-down's. So the users, the order, the seeds, alpha 0.9 / decay alpha 0.5, show-time inputs, 32k windows, 1 lane, strided carry, fresh Muon (`expertium-iter53`), snapshots every 400 users and the trim are SM106's, and the schedule and targets are SM118's. `sm124_pair.py recipe`: this tree prepares SM106's cool-down exactly (old targets; only the folder name and the creation time differ). `sm124_pair.py p2` = SM118's p2 check.
- **No source change.** The code is the longer cool-down for Tiny (SM115)'s launch tree (112c4082), the tree the show-time two-head cool-down for Small (SM118) trained with. Against the tree the Tiny two-head model (SM106) trained and scored with (c99330a7) it differs only by the 2R segment schedule of the longer big instant-only teacher (SM113) (12 lines in `expertium_decay.py` / `expertium_options.py`) and the history-rewrite commit-id strings (`sm91s2.py`, `sm91s3.py`, not imported by the trainer or the scorer); C2 re-proves the old paths on CPU against SM106's own worktree, bit for bit, including the KD replay loss under the cool-down's switch.
- **Branch:** `sm124` is cut from `origin/sm118` (c8f5abf7), not from main: main lacks the chain's trainer code (`sm77_warm.py`, `sm11_replay_export.py`, the `w64-c1` preset, the segment schedule), as for the Tiny two-head model (SM106), which was cut from `origin/sm101`. The branch also holds byte-for-byte copies of SM106's committed rows (`artifacts/small-model/sm106/gpu/sm106-rows.jsonl`, `sm106-rows-1500.jsonl`, sha256 pinned in the rule).

**What differs from the show-time two-head cool-down for Small (SM118):** the student (the Tiny two-head model (SM106)'s epoch 29, w64-c1), the reference (SM106), the early-stop reference (SM106's snapshot) and no export (the targets are SM118's). The final model: the end of pass 2, label `sm124-c1-e29st-dec2ep32k-t91s2/w8+s8`.

## Early-stop reference: the Tiny two-head model (SM106)'s own snapshot near the matching learning rate

Pass 1 ends at 0.2929 x peak (half of the 2R reviews). The Tiny two-head model (SM106)'s cool-down ran in one attempt (no `attempts/`, no `resume.json`), with snapshots every 400 users. The nearest to the matching point is **users-04000**:
- 4,000 of 8,000 users done, 284,944,410 of 581,942,724 reviews (fraction 0.4896);
- its last update ran at learning rate **0.3045 x peak** (the formula gives 0.3045); pass 1 ends at 0.2929 x peak, a slightly lower learning rate (users-04400 sits at 0.2536, further away). The gap is a bit larger than for SM118's reference (0.2977) and SM113's (0.299), in the same direction: the reference is slightly less cooled than pass 1;
- it lies on the path to SM106's final model; master sha256 957b9118e944 (pinned in C4 and the launcher).

It is scored by this run (the first GPU ticket, show-time inputs, 723 users, label `sm106-dec-u4000`), with the same scorer and architecture (w64-c1); C9 shows on CPU that this tree's scorer reproduces the committed rows of the Tiny two-head model (SM106) (within 1e-5 LogLoss, the per-user evaluation tolerance of AGENT.md; the trial run saw 1.0e-6, one rounding step of the 6-decimal committed GPU rows, on 2 users, both heads) and that the snapshot scores.

## Pre-registered rules (`sm124_rule.py`, fixed before launch)

All numbers: 722 users of the 732 set, review-weighted LogLoss, show-time inputs (`e2s-query`), paired user bootstrap (2,000 resamples, seed 0, 95% CI), % of the big model (1 leak-free epoch)'s normal-input LogLoss of the same head; the same `showtime_report.py` loaders (sha256 pinned) as the show-time instant-only small model (SM96) ... the show-time two-head cool-down for Small (SM118). References are read from their own committed rows (sha256 pinned), never rescored.

Per head h (curve, instant), x_h = the new model minus the Tiny two-head model (SM106), show-time inputs:
- **gain(h)** = the 95% CI of x_h entirely below 0 AND x_h <= -0.05%;
- **regression(h)** = x_h >= +0.05% OR the 95% CI of x_h entirely above 0.

**Early stop after pass 1 (`sm124_rule.py mid`).** y_h = pass 1's final model minus the Tiny two-head model (SM106)'s cool-down snapshot after 4,000 users (both heads). **Continue** to pass 2 if at least one head's point estimate is below 0 AND neither head's point estimate is +0.05% or more. **Stop** otherwise: verdict "no go (stopped after pass 1)". Read-out (no verdict): pass 1 minus SM106's final model.

**Final go rule (`sm124_rule.py decide`).**
- **Go** = a gain on at least one head AND a regression on neither head. **No go** otherwise. The same rule as the show-time two-head cool-down for Small (SM118), with SM106 in place of SM92.
- **Reported with it (no verdict):** per head, the share of the teacher gap closed, (new - SM106) / (SM91s2 - SM106) (the show-time big model (SM91s2) minus the Tiny two-head model (SM106): curve -0.422% [-0.597, -0.257], instant -0.430% [-0.467, -0.394]); minus the show-time big model (SM91s2), minus the release (SM63), minus the show-time retrain of the release (SM92) and minus the show-time two-head cool-down for Small (SM118) on show-time inputs (and minus SM118 on normal inputs); the smaller-model bar against the big model (1 leak-free epoch) on show-time and on normal inputs (curve <= +0.5%, instant <= +0.4%, point estimates as the 10-29 pruning bar, CIs reported); the show-time cost; both early-stop pairs; instant only: minus the Tiny Instant-Only model (SM101) and minus the longer cool-down for Tiny (SM115); the 1,500 validation users on show-time inputs (% of the release's own LogLoss): minus SM106 (its committed 1,500-pool rows, 767 users), minus SM91s2 (ids >= 5000 are its training users), minus SM118, minus the release, by id part, row checks. The Tiny two-head model (SM106) has no normal-input rows, so there is no normal-input pair against it.
- **Selftest** (`sm124_rule.py selftest`, CPU proof C8): SM106 against itself 0 / no go / stop; the show-time two-head cool-down for Small (SM118) minus SM92 reproduced exactly through this code (-0.114% [-0.177, -0.062] / -0.100% [-0.110, -0.091], go on both); SM106's published pairs vs SM92 and vs the release reproduced after rescaling to this unit (within 0.003%); the release minus SM106 -> no go (instant regression), stop; SM92 minus SM106 -> go on both heads; the show-time big model (SM91s2) minus SM106 -> go on both heads, gap closed 1.0; synthetic: 0.04% better instant -> no go but continue; 0.1% better instant with the curve 0.06% worse -> no go, stop; 0.1% better curve only -> go; the edges; an end-to-end mid + decide on relabelled SM106 rows.

## Expected gain

- The teacher gap is much larger than for Small: the show-time big model (SM91s2) minus the Tiny two-head model (SM106) is -0.422% curve and -0.430% instant (Small: -0.315% / -0.251%). The show-time two-head cool-down for Small (SM118) closed 0.36 (curve) and 0.40 (instant) of its gap; the longer cool-down for Tiny (SM115) closed 0.33 of its instant gap. A third of SM106's gap would be about -0.14% on each head.
- Against it: a smaller model may absorb less of the teacher, the curve head is noisy (CI half-width ~0.09%), and a go mixes the longer cool-down and the new teacher (as for SM113 and SM118).
- Central guess: curve -0.06% to -0.16%, instant -0.10% to -0.17%; chance of a go ~75%. A go would also widen SM106's margin under the smaller-model bar (it sat at +0.366% curve / +0.076% instant in the show-time unit of its own decision).

## GPU hours and timeline (launch ~11:10 Tue 2026-11-17, queue empty)

Breaks (`sm124_block.py` = quiet.json + daily 20:00-01:00, Sunday from 16:00): Tue 20:00 -> Wed 01:00, Wed 20:00 -> Thu 01:00.

Speed basis: the Tiny two-head model (SM106)'s one-epoch cool-down ran at 20,941 reviews/s (w64-c1, old targets) and the longer cool-down for Tiny (SM115)'s passes at 20,960-20,983 (w64-c1), but the show-time two-head cool-down for Small (SM118)'s passes ran at 28,690-28,915 reviews/s on 11-16 (w64; SM92's cool-down had run at 20,377). The cause of SM118's speed-up is not known, so both are planned: one pass = 463 min at 21k, 336 min at 29k.

| step | GPU | ends about (29k / 21k reviews/s) |
|---|---|---|
| pre-launch GPU smoke of pass 1 (smoke copy; queue ticket 001303, 2 users x 512 reviews) | 0.01 h | done before the launch |
| early-stop reference: SM106's snapshot after 4,000 users scored on show-time (723 users) | 0.2 h | ~11:25 Tue |
| CPU: targets status + manifest, pass-1 Muon check + pair; smoke of pass 1 | 0.02 h | ~11:30 Tue |
| **pass 1** (snapshot stop <= 2 min before 20:00 if not finished) | 5.6 / 7.7 h | ~17:10 / ~19:15 Tue |
| early stop: pass 1 scored on show-time (723 users), `mid` (needs 28 min before 20:00, else after 01:00) | 0.2 h | ~17:25 / ~19:30 Tue |
| (if stop: done, verdict no go; ~6 / 8 GPU h in all) | | |
| pass 2 prepare + pair (CPU), smoke | 0.02 h | ~17:30 / ~19:35 Tue |
| **pass 2** (29k: ~2.4 h before the break, the rest after 01:00; 21k: starts 01:00, a start needs 33 min before a break) | 5.6 / 7.7 h | ~04:15 / ~08:45 Wed |
| lr check; final scoring: show-time + normal (723 + 3 long users each), the 1,500 pool for the new model, `decide` | 0.8 h | **~05:05 / ~09:40 Wed 2026-11-18** |

**Total: about 12.5 GPU h at 29k reviews/s, 16.5 GPU h at 21k** (about 6 / 8 GPU h if the early stop fires). No export (SM118's targets are reused) and no reference scoring on the 1,500 pool (SM106 already has those rows), so it is ~5 GPU h shorter than SM118. Margin to the Wednesday 20:00 break: ~10 h. Anything that slips into a break pauses and resumes by itself. 10-epoch ETA (8k tier = 581.94M reviews per epoch, 2k tier = 147.6M): at 21k reviews/s 77.0 h (8k) / 19.5 h (2k); at 29k 55.7 h (8k) / 14.1 h (2k).

## Risks

1. **The effect is below the bar on both heads**, or the curve gets worse while the instant head gains: the early stop caps the clear failures at ~6-8 GPU h.
2. **A go mixes length and teacher** (as for SM113 and SM118).
3. **The reused targets:** checked by C5 (status complete, manifest == SM118's, no file newer than SM118's pass 1, SM118's full `targets` check again) and again by the launcher (status + manifest) before any training. Any failure stops the launcher.
4. **Interruptions:** every step resumes (a pass from its last 400-user snapshot). Detached launchers die when the Claude app restarts: check after restarts.
5. **References outside this worktree (read only; none may be cleaned up while this runs):** `wt/sm106` (epoch 29, its cool-down folder with the reference snapshot and final model, its code tree for C2), `wt/sm91s2` (the teacher), `wt/sm118` (its pass-1 folder for the pair proof), `wt/sm55` (the 8k run file), `wt/sm72s` / `wt/sm57` (user files), the targets caches (SM118's 8k show-time targets; SM106's old targets for C2 and the recipe).
6. **CPU load:** this GPU job runs beside the retrieval-effort probe (SM122) of another agent (three CPU processes until about 14:00, `wt/sm122` untouched); if training runs closer to 21k than 29k reviews/s, pass 1 crosses the 20:00 break (planned above).

## What is built

- `sm124_prepare.py` (SM118's code, names only), `sm124_pair.py` (recipe and p1 against SM106's cool-down and SM118's pass 1, p2 = SM118's), `sm124_rule.py` (mid, decide, selftest; two heads; SM118's code with SM106 as the reference), `sm124_check.py` (CPU proofs C1-C11), `sm124_probe.py` (SM118's trainer probe, code byte for byte), `sm124_block.py` (code byte for byte SM118's), `launch-sm124.sh` (SM118's launcher without the export steps; the early-stop reference and the pass-1 checks first), `extra-options.json` (byte for byte SM118's).
- `sm124-recipe-e29-dec1ep32k` (never trained, not committed): SM106's cool-down prepared by this tree, == SM106's.
- `sm124-c1-e29st-dec2ep32k-t91s2-p1` (not committed): pass 1, prepared, pair OK, Muon fresh-state check OK.
- `sm124-smoke0-c1-e29st-dec2ep32k-t91s2` (never trained beyond the smoke, not committed): pass 1 with another control folder; its GPU smoke (2 users x 512 reviews) passed 11:00 (`gpu/smoke0.out`, queue ticket 001303).
- CPU proofs `check-cpu-sm124.json`; dry run `gpu/launch-sm124-dryrun.log`.
