# How many users for an MIT Small (SM176): design

Written Thu 2026-11-26 ~19:45, before any GPU ticket (pre-registered; only CPU checks ran: `sm176_chain.py check`
ALL OK, `sm176_report.py selftest` OK). Branch `sm176` (from `origin/sm174` 1b5222e8), worktree
`C:/Users/jonat/workspace/wt/sm176`. Track: small model.

**This is How many users for MIT weights (SM174) (`../sm174/design.md`) on the Small architecture. Everything not
listed below is SM174's design unchanged**: the same users file (same nested subsets, same 190-user held-out slice),
the same row orders (`random.Random(174000 + epoch)`), the same fresh init (training_seed 2025), the same 2.33B-review
budget rule, the same early stopping (slice every k epochs, 0.05% minimum gain, decay share 0.15), the same 2048-review
one_minus_sin decay, the same 732-set scoring (723 users, show-time inputs, `w8+s8hx8`, exactly SM174's flags), the
same bootstrap and the same "users needed" interpolation (log2(users), worse head, point and CI upper bound).

## Question

How many users does a from-scratch, no-teacher **Small** need to come within +1% and +0.5% LogLoss of the same-budget
teacher run (and of the Small two-head release model (SM118)), on both heads? SM174 answered it for the six-layer Tiny
(about 1,182 users for +1%, about 3,194 for +0.5%, vs its same-budget teacher run). Weights stay experiment-only.

## Diff vs SM174

| item | SM174 (six-layer Tiny) | SM176 (Small) |
|---|---|---|
| architecture | `w64-c1-d2p1u1`, 185,057 parameters | `w64`, 331,511 parameters: the Small two-head model (SM118) |
| per-folder options | the six-layer Tiny two-head (SM130) epoch-30 folder | the same `template.json` (SM174's file, byte-identical) with `architecture = "w64"`; `check` proves the options then equal the Small chain's own constant-LR epoch 26 (the show-time retrain of the release (SM92) `sm92-e26`, the start of SM118's cool-down) exactly for the control and minus the teacher for the no-teacher arms |
| reference for "% vs release" | the six-layer Tiny two-head release model (SM130), its SM148 rows | the Small two-head release model (SM118) `sm118-e26-dec2ep32k-t91s2` (recipe dec2ep32k, sha256 b4fe1175), its SM148 rows (725 users, same scorer tree cea0f931) |
| integrity ticket | SM130 REF on SM148's 3 smoke users == SM148 rows | SM118 REF on the same 3 smoke users == SM148 rows (max abs LogLoss difference <= 1e-6) |
| arms | 500 / 1k / 2k / 4k / 8k no teacher + 8k teacher | **8,000 users with the usual teacher (SM176 n8000-kd) first** (the same-budget reference), then **1,000 / 2,000 / 4,000 users, no teacher (SM176 n1000 / n2000 / n4000)** |
| order | smallest first, control last | the reference first, then smallest to largest, so partial results are usable |
| speed basis | 178k reviews/s | **87k reviews/s** (Small 2048-window epochs: SM92 e26 86.9k, SM98 e32-e34 87.1k-87.4k), about 2.0x the Tiny time |
| run tree | `wt/sm174-run` (detached cea0f931, unedited) | the same tree, reused (still unedited, idle since SM174 ended 18:26); folders under `artifacts/small-model/sm176/` |
| queue purpose prefix | `How many users for MIT weights (SM174): ` | `How many users for an MIT Small (SM176): ` |

**Why 500 and 8,000 without a teacher are dropped.** Upper bound (no early stop) per arm is ~7.4-7.8 h of training at
87k reviews/s, plus slice evals and scoring, about 8.0-8.5 h. Six arms: ~50 h; five (without 500): ~41 h upper bound,
~28 h with SM174's early-stop ratios. That is above the ~30 GPU h cap, so the 8,000-user no-teacher arm is dropped too
(and 500, as asked). Consequences: the "teacher cost at 8,000 users" line of SM174's decomposition is not measured here
(SM174 found +0.24% curve / +0.27% instant for the Tiny); "users needed" interpolates over 1,000 / 2,000 / 4,000, and
if 4,000 is still above a threshold the answer is "> 4,000" with SM174's indicative power-law fit (three points).

| arm | users | epochs E (C_max + D_max) | slice every | eval points | training (upper bound) |
|---|---:|---|---:|---|---:|
| 8,000 users, with the usual teacher (SM176 n8000-kd) | 8,000 | 4 (3 + 1) | 1 | 1, 2, 3 | ~7.5 h |
| 1,000 users, no teacher (SM176 n1000) | 1,000 | 31 (26 + 5) | 4 | 4, 8, ..., 24, 26 | ~7.8 h |
| 2,000 users, no teacher (SM176 n2000) | 2,000 | 16 (14 + 2) | 2 | 2, 4, ..., 14 | ~7.6 h |
| 4,000 users, no teacher (SM176 n4000) | 4,000 | 8 (7 + 1) | 1 | 1 ... 7 | ~7.4 h |

## Budget and ETA

| step | GPU |
|---|---:|
| smoke (two arms) + integrity | ~0.3 h |
| four arms, training (upper bound, no early stop) | ~30.3 h |
| slice evals (24 x ~3 min) | ~1.2 h |
| 732 scoring (4 x ~20 min) | ~1.3 h |
| **total** | **~33 h upper bound; ~22 h expected with SM174's early stops (its arms used 48-86% of the bound)** |

Start ~20:00 Thu 26 Nov. Arm ends, expected (upper bound): n8000-kd ~03:30 Fri (~04:20), n1000 ~09:30 Fri (~12:50),
n2000 ~14:00 Fri (~21:10), n4000 **~18:00 Fri 27 Nov (~05:15 Sat 28 Nov)**, the end.

## Risks (in addition to SM174's)

1. **Small may need more data than Tiny** (more parameters, more overfitting of 1,000 users): the slice stops it; if
   even 4,000 is above +1% vs the control, the answer is "> 4,000" and a follow-up adds 8,000 without a teacher.
2. **Memory**: Small training reserves ~12.9-13.0 GB at fraction 0.8 (SM92 / SM98 epochs), process tree RSS ~8-11 GB
   (the trainer, as every Small epoch; CPU-only steps here are seconds and small).
3. Read-only inputs outside this worktree, as SM174 plus `wt/sm92` (`sm92-e26/decay.json`, check only) and `wt/sm118`
   (the reference checkpoint, integrity only).

## What is built

`sm176_chain.py`, `sm176_report.py`, `launch-sm176.sh`: SM174's three files with the changes above (diff them against
`../sm174/`), `users.json` and `template.json` (SM174's files, byte-identical, checked).
