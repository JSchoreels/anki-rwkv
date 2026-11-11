# The show-time instant-only big model (SM99): design (2026-11-11)

## Question

How good can RWKV-Instant get with the big architecture? This run measures that ceiling.

- The show-time instant-only small model (SM96) beats both big models on show-time instant LogLoss. It is still
  +0.04% [+0.02, +0.06] behind the show-time big model (SM91 step 2).
- The show-time instant-only model, stage 2 (SM98) is -0.035% vs the show-time instant-only small model (SM96).
- So the show-time instant-only model, stage 2 (SM98) and the show-time big model (SM91 step 2) are tied today: on
  show-time instant, 722 users, the show-time big model (SM91 step 2) minus SM98 is **-0.003% [-0.026, +0.019]**
  (the rule's selftest reproduces it).
- The show-time big model (SM91 step 2) was never trained instant-only. Instant-only training gave the small model
  about 0.2%. If the big model gains anywhere near that, the ceiling is clearly above the show-time instant-only model,
  stage 2 (SM98), and the big model is a candidate teacher for a future small instant-only model.

Unit: % of the big model (1 leak-free epoch)'s normal-input instant LogLoss, 0.275757 on 722 users of the 732 set.

Models named here:

| plain name (ID) | what it is |
|---|---|
| the show-time instant-only big model (SM99) | this design: the show-time big model (SM91 step 2) + instant-only big-model epochs on show-time inputs |
| the show-time big model (SM91 step 2) | the big model (1 leak-free epoch) + one big-model epoch on show-time inputs (both heads); the start |
| the show-time instant-only model, stage 2 (SM98) | the best small instant-only model: 35 epochs + the 32k Muon cool-down, show-time inputs; the primary reference |
| the show-time instant-only small model (SM96) | 31 epochs + cool-down; what the show-time instant-only model, stage 2 (SM98) was built on |
| the show-time retrain of the release (SM92) | what the fork's RWKV-Instant runs today |
| the big model (1 leak-free epoch) | the unit's denominator (normal inputs) |

## The recipe

| step | what | about |
|---|---|---|
| start | the show-time big model (SM91 step 2): `wt/sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth` (sha256 67ee8e9b0d01, trained parameters 4f31860d3cde), read only | |
| **epoch 1** | the big model's own epoch recipe (`scripts/sm72_big_epoch.py`, the one that made the show-time big model (SM91 step 2)) with two changes: `"curve_loss_weight": 0.0` and shuffle seed 2. Folder `artifacts/small-model/sm99-big-inst-e1`, track `big-inst` | ~545 min |
| early rule | epoch 1 scored on show-time inputs (723 users); `sm99_rule.py early` (below) | ~17 min |
| **epoch 2** (if the early rule says continue) | the same recipe from epoch 1, shuffle seed 3. Folder `sm99-big-inst-e2` | ~545 min |
| final scoring | the final model on show-time and normal inputs (732 set + 3 long users), the 1,500 pool on show-time inputs for the final model and for the show-time big model (SM91 step 2); `sm99_rule.py decide` | ~80 min |

The epoch recipe, unchanged from the show-time big model (SM91 step 2):
- the fork's training users 5000-10000 (5,001 users, 370,292,439 reviews after the same 27 label-window trims), whole
  histories;
- 2 strided lanes x 8,192-review windows per update, with carried state (budgeted carry table 768 MiB);
- a fresh AdamW (upstream groups), warmup 200 updates, LR 3.5e-4 -> 0 on a one_minus_sin schedule over the epoch;
- leak-free inputs (`RWKV_DELETED_CARDS=neighbour`), `show_time_intervals` true;
- no teacher; GPU cap 0.8; snapshots every 50 users.

**Why the big model's own recipe, and not epochs + the 32k Muon cool-down.**
- The small chain's recipe is constant-LR epochs, then one cool-down epoch. The big model has no such split: each of
  its epochs is a full warmup-and-decay cycle that ends at LR 0. So one big-model epoch already is "an instant-only
  epoch plus a cool-down", and every epoch ends in a finished model.
- The release's cool-down uses 32k-review windows with one lane. The small model runs it 4.4x slower than its 2k
  epochs (20.8k vs 91.9k reviews/s for the smaller instant-only model (SM101)). The big model trains at 11.4k reviews/s
  on 8k windows. At 32k windows it would run at roughly 2.5k reviews/s, so about 65 GPU h for one cool-down, and it
  likely does not fit the 0.8 memory cap (8k windows already peak at 11.1 GiB allocated, 13.0 GiB reserved).
- So the big model's own cool-down shape is the only one that fits the budget.

**Why two epochs, with an early rule between them.**
- One epoch is about 9.1 GPU h (the show-time big model (SM91 step 2): 542 min at 11,387 reviews/s, the same recipe
  on the same users). Two epochs plus scoring is about 20 GPU h, the budget's limit.
- A second epoch of the same recipe on the same users is expected to add little: the better teacher (SM65)'s added
  big-model epoch gave -0.03% instant on normal inputs, and SM98's four added small-model epochs gave -0.035%. So the
  early rule runs epoch 2 only when epoch 1 already beats the show-time instant-only model, stage 2 (SM98).
- Epoch 1 is a finished model (its LR has decayed to 0). If the early rule stops, the final read-out runs on epoch 1,
  so the GPU time is never wasted on a half-trained checkpoint.

**Why a new user order per epoch (seeds 2, 3).** The big model (1 leak-free epoch) and the show-time big model (SM91
step 2) both used shuffle seed 1. A third and fourth pass in the same order would end every epoch, at the lowest LR, on
the same users. Seeds 2 and 3 change only the order: the same users, train ranges and trims (`sm99_epoch.py pair`
checks it).

**KD: off.** The big model has no teacher above it.
- The big model's own recipe has no teacher. The show-time big model (SM91 step 2) was trained on labels only.
- Self-KD would mean distilling toward the show-time big model (SM91 step 2) itself. With the small chain's alpha 0.9,
  90% of the instant target would be the start's own predictions, so it would hold the instant head where it is. But
  moving the instant head is exactly what this run measures.
- Self-KD would also need replay targets for 370M reviews on show-time inputs (an export of ~2.6 GPU h at ~39k
  reviews/s, as for the show-time teacher targets (SM91 step 3)) and a KD path never run with this recipe.
- With curve loss weight 0, KD would only act on the instant term anyway.

**Instant-only.** `curve_loss_weight` 0 scales every curve term of the loss by 0. It is the option the instant-only
small model (SM57) introduced and SM96 / SM98 / SM101 used. The instant head and the shared trunk get all the
gradient. The curve-only weights get zero gradients (K3c below names the 12 tensors). The curve head is therefore
untrained, and it is never part of a verdict.

## Early rule (`sm99_rule.py early`, fixed before launch)

x = epoch 1 minus the show-time instant-only model, stage 2 (SM98), instant head, show-time inputs, 722 users, paired.

- **Continue** to epoch 2 if the point estimate of x is below 0 (epoch 1 already beats the show-time instant-only
  model, stage 2 (SM98)).
- **Stop** otherwise (x >= 0). There is no epoch 2, and the final read-out runs on epoch 1. That saves ~9.3 GPU h.

**Why 0.** A verdict "ceiling above" needs <= -0.05% with the CI below 0. From x >= 0, a second epoch would have to gain
more than 0.05% and also clear the CI, while the evidence above says a repeated epoch adds about 0.03%. Epoch 1 starts
where SM98 already is (-0.003%), so any gain from instant-only training shows up in x directly.

## Final read-out (`sm99_rule.py decide`, pre-registered)

All numbers use 722 users of the 732 set, review-weighted LogLoss, the instant head, show-time inputs (`e2s-query`, what
RWKV-Instant sees in the fork), leak-free w8+s8 scoring, a paired user bootstrap (2,000 resamples, seed 0, 95% CI), as %
of the big model (1 leak-free epoch)'s normal-input instant LogLoss 0.275757.

The final model is fixed in advance: epoch 2 if the early rule said continue, otherwise epoch 1. It is never "the
better of the two".

- **Primary, the verdict (a measurement, not a go):** the show-time instant-only big model (SM99) minus the show-time
  instant-only model, stage 2 (SM98).
  - **"Ceiling above SM98"** if it is <= **-0.05%** AND its 95% CI is entirely below 0. Then a teacher experiment for a
    future small instant-only model is worth running next.
  - **"At the ceiling"** otherwise: the small instant-only model is at the big-model ceiling.
- **With it (reported, no verdict):** the show-time instant-only big model (SM99) minus the show-time big model (SM91
  step 2), on show-time inputs (and on normal inputs). This is the gain of instant-only training on the big
  architecture.
- Also reported:
  - minus the show-time instant-only small model (SM96), the show-time retrain of the release (SM92) and the release
    (SM63), all on show-time inputs;
  - minus the show-time instant-only model, stage 2 (SM98) on normal inputs;
  - the final model on normal inputs minus the big model (1 leak-free epoch);
  - the final model's own show-time cost (show-time minus normal);
  - if epoch 2 ran: epoch 1 minus SM98 and epoch 2 minus epoch 1, on show-time inputs.
- **The 1,500 validation users** (outside the 732 set), on show-time inputs. Their unit is % of the release (SM63)'s
  own LogLoss, as in SM96 / SM98 / SM101, because the big model (1 leak-free epoch) has no normal rows there. Reported:
  minus SM98, minus the show-time big model (SM91 step 2), minus SM96 and minus the release (SM63), plus the row
  check on the 6 check users. The show-time big model (SM91 step 2) is scored on that pool by this run (~20 min).
- References are read from their own rows (sha256 pinned), never rescored: the rows of SM98, SM96 and the show-time big
  model (SM91 step 2) carried in this branch, and those of SM92 and the release from the show-time report.

**The rule's selftest** (`sm99_rule.py selftest`, part of the CPU proofs):
- SM98 against itself gives 0, "at the ceiling", early stop;
- SM96 minus SM98 gives +0.035% [+0.029, +0.041] (as in SM98's decision), "at the ceiling", early stop;
- the show-time big model (SM91 step 2) minus SM98 gives -0.003% [-0.026, +0.019], "at the ceiling", early continue;
- a synthetic model 0.3% better everywhere gives "ceiling above SM98";
- the edges of both rules, and an end-to-end run of `early` and `decide` on SM98's rows relabelled as epoch 1.

## GPU memory, CPU memory

- **Training:** cap 0.8 (the recipe's `gpu_memory_fraction`). The show-time big model (SM91 step 2) peaked at 11.1 GiB
  allocated and 13.0 GiB reserved with this recipe.
- **Scoring:** `HELDOUT_GPU_MEMORY_FRACTION=0.8`. The cap is 0.92 only for the 3 long users, or for a retry pass that
  made no progress (memory only, same numbers).
- **CPU proofs:** each CPU process stays well under 6 GB (the trainer slice uses 1,024-review windows).
- **Trainer process tree:** it reached 11.9 GB RSS (main + 5 preparation workers) in the show-time big model (SM91
  step 2)'s epoch, with the same recipe. The machine has 32 GB. The launcher holds a segment while less than 3 GB is
  free.

## Breaks

`sm99_block.py` is SM101's `sm101_block.py`: the quiet.json windows (scope all / gpu) plus the daily 20:00-01:00
(Sunday from 16:00), unless a quiet.json window overlaps that night. It is re-read at every check, and CPU steps wait
too.
- An epoch segment starts only when 30 min fit before the next break.
- It snapshot-stops <= 2 min before a break (snapshots every 50 users, ~1 min for a stop request) and resumes after it
  from the snapshot. The user in flight at the stop is retrained from its start.
- Every scoring ticket must end before the next break. No long non-resumable job starts into a break.

## Checks before launch

- **CPU proofs** (`sm99_check.py` -> `check-cpu-sm99.json`, required ALL OK):
  - K1: the code trees (scripts/ expertium/ rwkv/) == 1e22b62c, the trees SM96 / SM98 / SM101 trained and scored with.
    The 8 files changed since the show-time big model (SM91 step 2)'s launch commit 72ba7cef are listed.
  - K2: epoch 1's folder == the show-time big model (SM91 step 2)'s epoch folder, except curve loss weight 0, the start,
    the seed (row order only) and paths. 11 mutated copies are refused.
  - K3: the trainer slice on CPU (`sm99_probe.py` = `sm57_probe.py`), the fork architecture from the start weights,
    2 users at 1,024 x 2 carried lanes, with AdamW:
    - K3a: the tree of 72ba7cef == this tree without the option, bit for bit (loss, instant average, every gradient,
      final weights);
    - K3b: weight 1.0 == no option, bit for bit;
    - K3c: weight 0 trains, every loss == its instant average bit for bit, and only the 12 curve-only tensors get newly
      zero gradients.
  - K4: the show-time samples with the epoch's options: only the query intervals and curve query times move, and the
    trainer rows == the scorer's e2s-query rows, bitwise (`scripts/sm91.py t1_t2`).
  - K5: this tree's scorer == the scorer of 72ba7cef on CPU, for the show-time big model (SM91 step 2), on show-time
    and normal inputs, so its committed rows are comparable with this run's.
  - K6: the rule selftest. K7: the breaks on fixed times.
- **GPU pair check** (`sm99_gpu_check.py` -> `gpu/check-gpu-sm99.json`, one queue ticket):
  - G1: a trainer smoke of the show-time big model (SM91 step 2)'s own epoch folder, on its tree (wt/sm91s2) and on
    this tree: the trained weights are bit for bit equal.
  - G2: the epoch-1 smoke records curve_loss_weight 0.0, and its weights differ from the same folder without the weight.
  - G3: this tree's scorer on the GPU reproduces the committed show-time and normal rows of the show-time big model
    (SM91 step 2) for 2 users, number for number.
- **Label windows:** `scripts/sm72_label_windows.py --trim` is the big-model version of the replay-label windows check:
  there is no replay KD, so the label mask comes from the data. It runs at every prepare, and it trims the same 27
  unlabelled tails as the show-time big model (SM91 step 2).
- **Trainer smoke** of each epoch folder (2 users x 512 reviews), as a ticket before the epoch.

## GPU budget and timeline

| step | GPU | launched ~07:10 Wed 2026-11-11, queue empty |
|---|---|---|
| GPU pair check (before launch) | ~0.2 h | done 07:00 |
| epoch 1 smoke + epoch 1 (370.3M reviews at ~11.4k reviews/s) | ~9.2 h | ~16:25 Wed |
| epoch 1 show-time scoring (723 users) + early rule | ~0.3 h | ~16:45 |
| epoch 2 prepare + smoke + first segment (~190 min, ~130M reviews) to the 19:58 snapshot stop | | 19:58 |
| break 20:00-01:00 | | |
| epoch 2 resumed (~350 min left + ~10 min restart and the user in flight) | ~9.3 h for the whole epoch | ~07:05 Thu |
| final scoring: show-time and normal (732 + long), the 1,500 pool x 2, rules | ~1.3 h | **~08:30 Thu 2026-11-12** |

- **Total: about 20.3 GPU h.** If the early rule stops: about 10.6 GPU h, and the run ends ~18:10 Wed (the final scoring
  of epoch 1 fits before the break).
- Speeds assumed: 11,387 reviews/s, the show-time big model (SM91 step 2)'s epoch on the same users with the same
  recipe. Instant-only training does not skip any forward work, so the speed should be about the same.
- 10-epoch ETA at that speed: 8k tier (581.94M reviews per epoch) 142 h; 2k tier (147.6M) 36 h. This is the big model,
  about 8x slower per review than the small model's constant-LR epochs (~88-92k reviews/s).

## Branch

`sm99` is cut from `origin/sm101`, not from main, as SM101 was cut from SM98. Main lacks this chain's trainer code:
there is no `curve_loss_weight`, no `show_time_intervals`, and no `scripts/sm72_big_epoch.py --extra-options`.
`scripts/ expertium/ rwkv/` are bitwise the trees SM96 / SM98 / SM101 trained with (K1). No source file is changed for
this run. Everything new is under `artifacts/small-model/sm99/`.

## Files

- `launch-sm99.sh` (launcher);
- `sm99_epoch.py` (prepare / pair / negatives);
- `sm99_check.py` (CPU proofs) and `sm99_probe.py` (the trainer slice);
- `sm99_gpu_check.py` (GPU pair check);
- `sm99_rule.py` (early, decide, selftest);
- `sm99_block.py` (breaks).
