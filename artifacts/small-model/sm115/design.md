# The longer cool-down for Tiny (SM115): design (2026-11-14)

## Question

The longer big instant-only teacher (SM113) stretched the Small Instant-Only model (SM98)'s cool-down over two passes of the 8,000 users (one learning-rate decay 7e-4 -> 0 over twice the reviews), from SM98's constant-LR epoch 35, with the show-time instant-only big model (SM99)'s show-time targets. It was a go: -0.094% [-0.103, -0.085] vs the Small Instant-Only model (SM98) on show-time instant, 722 users of the 732 set (the one-pass version, the big instant-only teacher (SM110), gave only -0.023%).

This run asks: does the same recipe pass the go rule for Tiny, against the Tiny Instant-Only model (SM101)?

Unit everywhere: % of the big model (1 leak-free epoch)'s normal-input instant LogLoss, 0.275757 on 722 users of the 732 set. All runs are leak-free (deleted cards: neighbour), scored w8+s8, on show-time inputs. Whatever the result, it is held for the next release (no release, tag, fork or Hugging Face change).

| plain name (ID) | what it is |
|---|---|
| the longer cool-down for Tiny (SM115) | this design: SM113's two-pass cool-down applied to the Tiny Instant-Only model (SM101)'s pre-cool-down checkpoint |
| the Tiny Instant-Only model (SM101) | the Small Instant-Only model (SM98)'s epoch 35 with one card layer cut (w64-c1), epochs 36-38 instant-only on show-time inputs, then the one-epoch 32k Muon cool-down from epoch 38 with the old targets; the reference |
| the longer big instant-only teacher (SM113) | the same recipe on the Small architecture (w64), from SM98's epoch 35; go, -0.094% vs SM98 |
| the show-time instant-only big model (SM99) | the teacher (big architecture, 2 instant-only epochs on show-time inputs) |
| the Small Instant-Only model (SM98) | 35 epochs + the one-epoch cool-down, w64 (read-out only) |

## Chosen design: SM113's two-epoch cool-down from the Tiny Instant-Only model (SM101)'s epoch 38

- **Start checkpoint:** SM101's constant-LR epoch 38, `wt/sm101/artifacts/small-model/sm101-c1-e38-st/expertium/E89-F65e-F77a/master.pth` (sha256 e6ea69124af0, read only). It is the start of SM101's own cool-down (`sm101-c1-e38st-st-dec-1ep32k-muon`, decay.json `start_master_sha256`), i.e. exactly the checkpoint just before SM101's cool-down (SM101's DESIGN.md: cut epoch 35 -> epochs 36, 37, 38 -> cool-down).
- **Pass 1** `sm115-c1-e38st-st-dec2ep32k-t99-p1`: prepared like SM101's cool-down (`sm77_warm.py cooldown --epoch 25 --start <epoch 38> --arch w64-c1 --fraction 1.0`), with SM113's extra options (== SM101's except the teacher + replay folder: SM99's show-time targets, `anki-rwkv/artifacts/cache/kd-replay/sm99-big-inst-final8k-neighbour-e2s`, read only), the same label-window trim (user 5599 x 14), then `sm115_prepare.py p1`: `lr_resume = {"total_reviews": 2R, "reviews_offset": 0, "segment": true}`, R = 581,942,724. Learning rate 7e-4 -> 0.293 x 7e-4.
- **Pass 2** `sm115-c1-e38st-st-dec2ep32k-t99-p2`, prepared when pass 1 is complete: pass 1's final master.pth AND optimizer.pth (Muon state and update offset continue), the same rows in a new order (`random.Random(2).shuffle`), `lr_resume = {"total_reviews": 2R, "reviews_offset": R}`. Learning rate 0.293 x 7e-4 -> 0.
- **Pair proof** (`sm115_pair.py p1`, CPU proof C5): pass 1 == SM101's cool-down (as prepared) except lr_resume, the targets and the folder name; AND == SM113's pass 1 except the start (init_master, update offset 838,964 vs 772,730, start sha256 / source paths / the sm77 note) and the architecture (w64-c1), each of which must equal SM101's cool-down's. So the users, the order, the seeds, alpha 0.9 / decay alpha 0.5, curve loss weight 0, show-time inputs, 32k windows, 1 lane, strided carry, fresh Muon (`expertium-iter53`), snapshots every 400 users and the trim are all SM101's and SM113's.
- **No source change.** The code is SM113's launch tree (d99fe356, which holds SM113's 12-line segment change); the only difference in `scripts/ expertium/ rwkv/` is the commit-id strings that the history rewrite follow-up (f561e1c0) changed in `scripts/sm91s2.py` / `scripts/sm91s3.py` (not imported by the trainer or the scorer). The old paths are A/B-identical to c99330a7, the tree SM101 trained and scored with (C2, now also on SM101's own twice-resumed cool-down protocol).

**What differs from SM113:** only the start checkpoint (SM101's epoch 38 instead of SM98's epoch 35) and the architecture (w64-c1). The final model: the end of pass 2, label `sm115-c1-e38st-st-dec2ep32k-t99/w8+s8`.

## Early-stop reference: SM101's own snapshot at the matching learning rate (exists)

SM113 compared its pass 1 (learning rate 29.29% of its peak) with SM98's cool-down snapshot after 4,000 users (29.9%). SM101's cool-down was interrupted twice (a break stop after 2,894 users, then a reboot after 1,200 more), so its snapshots sit at other user counts. The one at the matching point is **attempt-2 `users-01200`** of SM101's cool-down folder:
- 4,094 of 8,000 users done (2,894 in attempt 1 + 1,200 in attempt 2), 290,914,508 of 581,942,724 reviews (fraction 0.49990);
- its last update ran at learning rate 2.0514e-4 = **0.2931 x peak** (the formula at that fraction gives 0.2930); pass 1 ends at 0.2929 x peak. This is a closer match than SM113's reference (0.299);
- it lies on the path to SM101's final model: the last attempt resumed from exactly this snapshot (`resume.json`); master sha256 3ab9e629e7d7 (pinned in C4 and the launcher).

It is scored by this run (the first GPU ticket, show-time inputs, 723 users, label `sm101-dec-u4094`), with the same scorer and architecture SM101 used; it had never been scored before.

## Pre-registered rules (`sm115_rule.py`, fixed before launch)

All numbers: 722 users of the 732 set, review-weighted instant LogLoss, show-time inputs (`e2s-query`), paired user bootstrap (2,000 resamples, seed 0, 95% CI), as % of the big model (1 leak-free epoch)'s normal-input instant LogLoss; the same showtime_report.py loaders (sha256 pinned) as SM96 ... SM113. References are read from their own committed rows (sha256 pinned), never rescored: SM101, SM98, SM99 rows of this branch (pins updated for the history rewrite, which changed only the `provenance_commit` field) and SM113's final and pass-1 rows.

**Early stop after pass 1 (`sm115_rule.py mid`).**
- x = SM115's pass-1 final model minus SM101's cool-down snapshot after 4,094 users (above).
- **Continue** to pass 2 if the point estimate x is below 0. **Stop** otherwise: verdict "no go (stopped after pass 1)", saving ~8 GPU h.
- Read-out (no verdict): pass 1 minus SM113's pass 1 (Small, the same recipe at the same point). For scale: SM101's final model is +0.132% behind SM113's pass 1.

**Final go rule (`sm115_rule.py decide`).**
- **Primary:** the longer cool-down for Tiny (SM115) minus the Tiny Instant-Only model (SM101). **Go** if its 95% CI lies entirely below 0 AND the gain is at least 0.05% (points <= -0.05). **No go** otherwise.
- **Reported with it (no verdict):** minus the longer big instant-only teacher (SM113) (Tiny vs Small, the same recipe); the share of the teacher gap closed, (SM115 - SM101) / (SM99 - SM101) (SM99 - SM101 = -0.299% [-0.328, -0.272]); minus SM99, SM98, SM92, the release (SM63) on show-time inputs; minus SM101 on normal inputs; normal inputs minus the big model (1 leak-free epoch) (the smaller-model instant bar: CI upper end <= +0.40%; SM101 was +0.298%); the show-time cost; both early-stop pairs; the 1,500 validation users (minus SM101, SM113, SM99, the release, by id part, row check).
- **Selftest** (`sm115_rule.py selftest`, CPU proof C7): SM101 against itself 0 / no go / stop; SM98 - SM101 -0.192% [-0.215, -0.171] (SM101's decision, negated) go / continue; SM113 - SM98 -0.094% [-0.103, -0.085] through this code (SM113's decision); SM99 - SM101 -0.299% go, gap closed 1.0; SM113 - SM101 -0.285% go; a synthetic 0.04% gain no go but continue; the edges; an end-to-end mid + decide on relabelled SM101 rows.

## Expected gain

- SM113 gained -0.094% on Small (gap closed 0.87 of SM99 - SM98). Tiny's teacher gap is almost three times larger (SM99 - SM101 = -0.299%), so the same recipe has more room: the teacher part alone could give more than on Small.
- Against it: a smaller model may absorb less of the teacher, and SM101's own cool-down already recovered part of the cut.
- Central guess -0.08% to -0.12%; chance of passing the go rule ~65-70%. The same attribution caveat as SM113: a go mixes the longer cool-down and the teacher.

## GPU hours and timeline (launch ~16:00 Sat 2026-11-14, queue empty)

Breaks (`sm115_block.py` = quiet.json + daily 20:00-01:00, Sunday from 16:00): Sat 20:00 -> Sun 01:00, Sun 16:00 -> Mon 01:00.

| step | GPU | ends about |
|---|---|---|
| early-stop reference: SM101's snapshot scored on show-time (723 users) | 0.2 h | 16:15 Sat |
| smoke of pass 1 (2 users x 512 reviews) | 0.05 h | 16:20 Sat |
| **pass 1**, first segment (snapshot stop <= 2 min before 20:00) | 3.6 h | 19:58 Sat |
| break | | 01:00 Sun |
| **pass 1**, rest (resumes from its last snapshot) | 4.2 h | ~05:15 Sun |
| lr check; early stop: pass 1 scored on show-time (723 users, ~12 min), `mid` | 0.2 h | ~05:30 Sun |
| (if stop: done, verdict no go; ~8.3 GPU h in all) | | (~05:30 Sun) |
| pass 2 prepare + pair (CPU), smoke | 0.05 h | ~05:35 Sun |
| **pass 2** (7.8 h, no break in the way) | 7.8 h | ~13:25 Sun |
| lr check; final scoring: show-time + normal (723 + 3 long users each), the 1,500 pool, `decide` | 0.8 h | **~14:15 Sun 2026-11-15** |

**Total: about 16.9 GPU h** (about 8.3 GPU h if the early stop fires). Margin to the Sunday 16:00 break: ~1.7 h. If pass 2 or the scoring slips past 16:00 it snapshot-stops / waits and resumes at Mon 01:00 by itself. Speed basis: SM101's cool-down ran at 20,839 reviews/s (clean segment), nvidia-smi max 15,297 MiB; SM113's passes (w64, SM99 targets) at 20.2-20.3k, process-tree RSS ~11.2 GB. 10-epoch ETA at 20.8k reviews/s (8k tier = 581.94M reviews per epoch, 2k tier = 147.6M): 77.7 h (8k) / 19.7 h (2k).

## Risks

1. **The effect is below the bar.** The early stop caps the clear failures at ~8.3 GPU h.
2. **A go mixes length and teacher** (as for SM113).
3. **The reference snapshot was never scored**: if its scoring fails, the launcher stops before any training (the first ticket).
4. **Interruptions:** every step resumes (a pass from its last 400-user snapshot). Detached launchers die when the Claude app restarts: check after restarts.
5. **References outside this worktree (read only, none may be cleaned up while this runs):** `wt/sm101` (epoch 38, its cool-down folder with the reference snapshot), `wt/sm113` (its pass-1 folder for the pair proof), `wt/sm110` (targets check, replay manifest copy), `wt/sm99` (the teacher), the targets cache.

## What is built

- `sm115_prepare.py` (SM113's code, names only), `sm115_pair.py` (p1 against SM101's cool-down and SM113's pass 1; p2 = SM113's), `sm115_rule.py` (mid, decide, selftest), `sm115_check.py` (CPU proofs C1-C9), `sm115_block.py` (byte for byte SM113's), `launch-sm115.sh` (SM113's launcher + the reference scoring step), `extra-options.json` (SM113's).
- CPU proofs `check-cpu-sm115.json`: ALL OK. Dry run `gpu/launch-sm115-dryrun.log`: `DRYRUN OK`.
- Pass 1 prepared: `gpu/prepare.out`, `gpu/trim.out`, `gpu/check-replay-label-windows-p1.json` (ok, trim user 5599 x 14), `gpu/check-muon-fresh-p1.json` (passed).
