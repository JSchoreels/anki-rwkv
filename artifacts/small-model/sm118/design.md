# The show-time two-head cool-down for Small (SM118): design (2026-11-16)

## Question

The longer cool-down (one learning-rate decay 7e-4 -> 0 stretched over two passes of the 8,000 users, with show-time
teacher targets) went go twice for instant-only students: the longer big instant-only teacher (SM113) gained -0.094%
vs the Small Instant-Only model (SM98), and the longer cool-down for Tiny (SM115) gained -0.099% [-0.108, -0.091] vs
the Tiny Instant-Only model (SM101) (722 users, show-time instant; the teacher gap closed 0.331).

This run asks: does the same recipe lift the Small two-head model, the show-time retrain of the release (SM92), on
BOTH heads, with the show-time big model (SM91s2) (the big two-head model) as the teacher of both the curve and the
instant targets?

Unit everywhere ("%"): % of the big model (1 leak-free epoch)'s normal-input LogLoss of the same head on the same 722
users of the 732 set (curve 0.298966, instant 0.275757). All runs are leak-free (deleted cards: neighbour), scored
w8+s8, on show-time inputs (`e2s-query`). Whatever the result, it is held for the next release (no release, tag, fork
or Hugging Face change; user 11-14: no new release soon).

| plain name (ID) | what it is |
|---|---|
| the show-time two-head cool-down for Small (SM118) | this design: the two-pass cool-down of the longer cool-down for Tiny (SM115) applied to the show-time retrain of the release (SM92)'s pre-cool-down checkpoint, both heads trained, the show-time big model (SM91s2)'s targets |
| the show-time retrain of the release (SM92) | the leak-free small model (SM72s) epoch 25 + epoch 26 on show-time inputs + the release's one-epoch 32k Muon cool-down on show-time inputs with the old targets of the big model (1 leak-free epoch); the Small two-head model of the family release; **the reference** |
| the show-time big model (SM91s2) | the big model (1 leak-free epoch) + one epoch on show-time inputs; the Big two-head model of the family release; **the teacher** (curve and grade targets) |
| the longer cool-down for Tiny (SM115) / the longer big instant-only teacher (SM113) | the recipe copied here (go for both) |
| the release (SM63) | the released small model (cool-down from epoch 25); read-out only |

## Chosen design: the two-epoch cool-down from the show-time retrain of the release (SM92)'s epoch 26

- **Start checkpoint:** the show-time retrain of the release (SM92)'s constant-LR epoch 26,
  `wt/sm92/artifacts/small-model/sm92-e26/expertium/E89-F65e-F77a/master.pth` (sha256 5a629e4e932a, read only, w64, both
  heads). It is the start of the show-time retrain of the release (SM92)'s own cool-down (`sm92-e26-dec-1ep32k-muon`,
  decay.json `start_master_sha256`), update offset 574,028.
- **Teacher targets (new, the first long GPU step):** the show-time big model (SM91s2) (sha256 67ee8e9b0d01) replays the
  full histories of the 8,000 cool-down users on show-time inputs (`scripts/sm11_replay_export.py --inputs e2s-query`,
  deleted cards neighbour), curve logit + 4 grade probabilities per review, into a NEW folder
  `anki-rwkv/artifacts/cache/kd-replay/sm91s2-big-showtime-final8k-neighbour-e2s` (~5.5 GB). Only 1,991 users of these
  targets exist today (the show-time interval retrain (SM91) step 3, `big-showtime-cool1991-neighbour-e2s`, the first
  1,991 users of the same cool-down order); they are re-exported with the rest and serve as a GPU-to-GPU check of the
  new export. Export ~4.2 GPU h at ~39k reviews/s (the 8k export of the show-time instant-only big model (SM99) on 11-13:
  39,288 reviews/s), per-user resumable, tickets cut before the breaks.
- **Pass 1** `sm118-e26-dec2ep32k-t91s2-p1`: prepared after the export like the show-time retrain of the release
  (SM92)'s cool-down (`sm77_warm.py cooldown --epoch 25 --start sm92-e26 --arch w64 --fraction 1.0` with this folder's
  `extra-options.json` = SM92's except the teacher and replay folder), the same label-window trim (user 5599 x 14),
  then `sm118_prepare.py p1`: `lr_resume = {"total_reviews": 2R, "reviews_offset": 0, "segment": true}`,
  R = 581,942,724. Learning rate 7e-4 -> 0.293 x 7e-4. KD alpha 0.9 / decay alpha 0.5 (the cool-down uses 0.5), both
  heads (no curve loss weight).
- **Pass 2** `sm118-e26-dec2ep32k-t91s2-p2`, prepared when pass 1 is complete: pass 1's final master.pth AND
  optimizer.pth (Muon state and update offset continue), the same rows in a new order (`random.Random(2).shuffle`),
  `lr_resume = {"total_reviews": 2R, "reviews_offset": R}`. Learning rate 0.293 x 7e-4 -> 0.
- **Pair proofs:** `sm118_pair.py p1` (in the launcher after the export): pass 1 == the show-time retrain of the
  release (SM92)'s cool-down (as prepared) except lr_resume, the targets and the folder name; AND == the longer big
  instant-only teacher (SM113)'s pass 1 except the start, the curve loss weight (0.0 there, absent here) and the
  targets. So the users, the order, the seeds, alpha 0.9 / decay alpha 0.5, show-time inputs, 32k windows, 1 lane,
  strided carry, fresh Muon (`expertium-iter53`), snapshots every 400 users and the trim are the show-time retrain of
  the release (SM92)'s, and the schedule is the longer big instant-only teacher (SM113)'s. `sm118_pair.py recipe`: this
  tree prepares the show-time retrain of the release (SM92)'s cool-down exactly (old targets; only the folder name and
  the creation time differ). `sm118_pair.py p2` = SM115's p2 check.
- **No source change.** The code is the longer cool-down for Tiny (SM115)'s launch tree (112c4082). Against the tree the
  show-time retrain of the release (SM92) trained with (9c125341) it differs only by the curve-loss-weight option of the
  instant-only small model (SM57) (default 1.0, bit for bit), the 2R segment schedule of the longer big instant-only
  teacher (SM113), the 1,500 pool option and the history-rewrite commit-id strings; C2 re-proves the old paths on CPU
  against SM92's own worktree, bit for bit, including the KD replay loss under the cool-down's switch.

**What differs from the longer cool-down for Tiny (SM115):** the student (the show-time retrain of the release (SM92)'s
epoch 26, w64, both heads trained), the teacher (the show-time big model (SM91s2), both heads; its targets exported by
this run) and a two-head rule. The final model: the end of pass 2, label `sm118-e26-dec2ep32k-t91s2/w8+s8`.

## Early-stop reference: the show-time retrain of the release (SM92)'s own snapshot near the matching learning rate

Pass 1 ends at 0.2929 x peak (half of the 2R reviews). The show-time retrain of the release (SM92)'s cool-down was
interrupted once (a stop after 3,256 users, attempt 1; the final attempt resumed from it), so its snapshots of the final
attempt sit at 3,256 + 400 k users. The nearest is **users-00800 of the final attempt**:
- 4,056 of 8,000 users done (3,256 + 800), 288,456,043 of 581,942,724 reviews (fraction 0.4957);
- its last update ran at learning rate **0.2977 x peak** (the formula gives 0.2977); pass 1 ends at 0.2929 x peak, a
  slightly lower learning rate (the same gap as the longer big instant-only teacher (SM113)'s reference, 0.299);
- it lies on the path to the show-time retrain of the release (SM92)'s final model (the final attempt is the only one
  after attempt 1); master sha256 e12a7ae7bbe5 (pinned in C4 and the launcher).

It is scored by this run (the first GPU ticket, show-time inputs, 723 users, label `sm92-dec-u4056`), with the same
scorer and architecture; C9 shows on CPU that this tree's scorer reproduces the committed rows of the show-time retrain
of the release (SM92) exactly (0.0 difference on 2 users, both heads) and that the snapshot scores.

## Pre-registered rules (`sm118_rule.py`, fixed before launch)

All numbers: 722 users of the 732 set, review-weighted LogLoss, show-time inputs (`e2s-query`), paired user bootstrap
(2,000 resamples, seed 0, 95% CI), % of the big model (1 leak-free epoch)'s normal-input LogLoss of the same head; the
same showtime_report.py loaders (sha256 pinned) as the show-time instant-only small model (SM96) ... the longer cool-down
for Tiny (SM115). References are read from their own committed rows (sha256 pinned), never rescored.

Per head h (curve, instant), x_h = the new model minus the show-time retrain of the release (SM92), show-time inputs:
- **gain(h)** = the 95% CI of x_h entirely below 0 AND x_h <= -0.05%;
- **regression(h)** = x_h >= +0.05% OR the 95% CI of x_h entirely above 0.

**Early stop after pass 1 (`sm118_rule.py mid`).** y_h = pass 1's final model minus the show-time retrain of the release
(SM92)'s cool-down snapshot after 4,056 users (both heads). **Continue** to pass 2 if at least one head's point estimate is
below 0 AND neither head's point estimate is +0.05% or more. **Stop** otherwise: verdict "no go (stopped after pass 1)",
saving ~10 GPU h. Read-out (no verdict): pass 1 minus the show-time retrain of the release (SM92)'s final model.

**Final go rule (`sm118_rule.py decide`).**
- **Go** = a gain on at least one head AND a regression on neither head. **No go** otherwise.
- **The curve rule, explicitly:** the curve point estimate must stay below +0.05% and its CI must not lie entirely above
  0, even when the instant head gains (and vice versa). This is stricter than the show-time retrain of the release
  (SM92)'s own ship rule (which needed both a +0.05% worsening AND a CI above 0 to block) and in the spirit of the
  pruning bar (point estimates; a smaller model may cost curve accuracy only within the bar, a cool-down may not cost
  any beyond the 0.05% minimum effect).
- **Reported with it (no verdict):** per head, the share of the teacher gap closed, (new - SM92) / (SM91s2 - SM92)
  (the show-time big model (SM91s2) minus the show-time retrain of the release (SM92): curve -0.315% [-0.491, -0.149],
  instant -0.251% [-0.280, -0.222]); minus the show-time big model (SM91s2) and minus the release (SM63) on show-time
  inputs; minus the show-time retrain of the release (SM92) on normal inputs; the smaller-model bar against the big
  model (1 leak-free epoch) on normal and on show-time inputs (curve <= +0.5%, instant <= +0.4%, point estimates as the
  10-29 pruning bar, CIs reported); the show-time cost; both early-stop pairs; instant only: minus the Small
  Instant-Only model (SM98) and minus the longer big instant-only teacher (SM113); the 1,500 validation users on
  show-time inputs (% of the release's own LogLoss): minus the show-time retrain of the release (SM92) (scored by this
  run on the 1,500 pool, which it never was), minus the show-time big model (SM91s2) (ids >= 5000 are its training
  users), minus the release, by id part, row checks.
- **Selftest** (`sm118_rule.py selftest`, CPU proof C8): the show-time retrain of the release (SM92) against itself 0 /
  no go / stop; its published rule pair vs the release (SM63) reproduced exactly (curve -0.019% [-0.092, +0.085],
  instant -0.305% [-0.405, -0.232]) -> go on instant; the release minus SM92 -> no go (instant regression); the show-time
  big model (SM91s2) minus SM92 reproduced (-0.315% / -0.251%) -> go on both heads, gap closed 1.0; synthetic: 0.04%
  better instant -> no go but continue; 0.1% better instant with the curve 0.06% worse -> no go, stop; 0.1% better curve
  only -> go; the edges; an end-to-end mid + decide on relabelled SM92 rows.

## Expected gain

- The instant-only students closed 0.33-0.87 of their teacher gap with this recipe. Here the teacher gap is -0.251%
  instant and -0.315% curve, so a third of it would be ~-0.08% instant and ~-0.10% curve.
- Against it: the show-time interval retrain (SM91) step 3 found that these same targets added nothing in a quarter-epoch
  cool-down, and the one-pass cool-down with new targets of the big instant-only teacher (SM110) gave only -0.023%; the
  gain came with the longer schedule. The curve head is noisier (CI half-width ~0.09%), so a curve gain needs to be
  ~0.1% to count, and a +0.05% curve point estimate blocks a go (about one chance in eight if the true curve effect is 0).
- Central guess: instant -0.06% to -0.10%, curve -0.03% to -0.12%; chance of a go ~60%. As for the longer big
  instant-only teacher (SM113), a go mixes the longer cool-down and the new teacher.

## GPU hours and timeline (launch ~06:50 Mon 2026-11-16, queue empty)

Breaks (`sm118_block.py` = quiet.json + daily 20:00-01:00, Sunday from 16:00): Mon 20:00 -> Tue 01:00, Tue 20:00 -> Wed 01:00.

| step | GPU | ends about |
|---|---|---|
| pre-launch GPU smoke of pass 1 with the 1,991-user targets of the same teacher (done 06:18) | 0.01 h | done |
| early-stop reference: the show-time retrain of the release (SM92)'s snapshot scored on show-time (723 users) | 0.35 h | 07:15 Mon |
| teacher targets: the show-time big model (SM91s2), 8,000 users, show-time inputs | 4.2 h | ~11:35 Mon |
| CPU: export status, verify (8 users on CPU), all 8,000 files checked, pass 1 prepared + pair + Muon check; smoke | 0.05 h | ~11:55 Mon |
| **pass 1**, first segment (snapshot stop <= 2 min before 20:00 if not finished) | ~8.0 h | ~19:55 Mon |
| break | | 01:00 Tue |
| pass 1 rest (if any); lr check; early stop: pass 1 scored on show-time (723 users), `mid` | 0.5 h | ~01:40 Tue |
| (if stop: done, verdict no go; ~13 GPU h in all) | | (~01:40 Tue) |
| pass 2 prepare + pair (CPU), smoke | 0.05 h | ~01:50 Tue |
| **pass 2** (no break in the way) | 8.0 h | ~09:50 Tue |
| lr check; final scoring: show-time + normal (723 + 3 long users each), the 1,500 pool for the new model and for SM92, `decide` | 2.5 h | **~12:30 Tue 2026-11-17** |

**Total: about 23.5 GPU h** (the plan's ~17 GPU h did not include the 4.2 h teacher export and the 2.5 h of scoring;
about 13 GPU h if the early stop fires). Margin to the Tuesday 20:00 break: ~7 h; anything that slips pauses and resumes
by itself after the break. Speed basis: the show-time retrain of the release (SM92)'s cool-down ran at 20,377
reviews/s (581.9M reviews = 476 min per pass), process-tree RSS 7.4 GB; the exports at 39-41k reviews/s. 10-epoch ETA
at 20.4k reviews/s (8k tier = 581.94M reviews per epoch, 2k tier = 147.6M): 79.3 h (8k) / 20.1 h (2k).

## Risks

1. **The effect is below the bar on both heads**, or the curve gets worse while the instant head gains: the early stop
   caps the clear failures at ~13 GPU h.
2. **A go mixes length and teacher** (as for the longer big instant-only teacher (SM113)).
3. **The export:** a new 8,000-user export of a big model (the third one); checked by status, CPU verify of 8 users,
   the shape / NaN pattern / finiteness of all 8,000 files against the old targets, and the 1,991 users of the same
   teacher's earlier export (exporter tolerance). Any failure stops the launcher before any training.
4. **Interruptions:** every step resumes (the export per user, a pass from its last 400-user snapshot). Detached
   launchers die when the Claude app restarts: check after restarts.
5. **References outside this worktree (read only; none may be cleaned up while this runs):** `wt/sm92` (epoch 26, its
   cool-down folder with the reference snapshot and final model, its code tree for C2), `wt/sm91s2` (the teacher),
   `wt/sm113` (its pass-1 folder for the pair proof), `wt/sm55` (the 8k run file), `wt/sm72s` / `wt/sm57` (user files),
   the targets caches (old 8k targets, SM91 step 3's 1,991 users).

## What is built

- `sm118_prepare.py` (SM115's code, names only), `sm118_pair.py` (recipe, p1 against SM92's cool-down and SM113's pass
  1, p2 = SM115's), `sm118_rule.py` (mid, decide, selftest; two heads), `sm118_check.py` (CPU proofs C1-C11 and
  `targets` after the export), `sm118_probe.py` (SM57's trainer probe + the KD teacher), `sm118_block.py` (code byte for
  byte SM115's), `launch-sm118.sh` (SM115's launcher + the export steps of the big instant-only teacher (SM110)),
  `extra-options.json` (SM92's with the new teacher and replay folder).
- `sm118-recipe-e26-dec1ep32k` (never trained, not committed): SM92's cool-down prepared by this tree, == SM92's.
- `sm118-smoke0-e26-dec2ep32k-st91` (never trained, not committed): pass 1 as it will be prepared, with the 1,991-user
  targets of the same teacher; its GPU smoke (2 users x 512 reviews) passed 06:18 (`gpu/smoke0.out`, queue ticket 001255).
- CPU proofs `check-cpu-sm118.json`: ALL OK. Dry run `gpu/launch-sm118-dryrun.log`: `DRYRUN OK`.
