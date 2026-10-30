# HANDOFF: data augmentation (SM67) stage 1

Branch `sm67` (worktree `C:/Users/jonat/workspace/wt/sm67`), cut from `origin/sm72s` (9441d7db). Design:
`artifacts/small-model/sm67/DESIGN.md` on main (section "Implementation"). Code and CPU checks only: **not launched,
no GPU used**. The user launches it; the launcher itself waits for the tiny architecture additions' (SM77) launcher.

## What the arm is

The leak-free small model's (SM72s) 1-epoch 32k Muon cool-down from constant-LR epoch K (`START_EPOCH`, default 25),
exactly as the SM72s launcher prepares it (same `$EXTRA`, users, seed-1 order, LR 7e-4 -> 0 over the same 581.9M
reviews, fresh Muon, replay KD from the big model (1 leak-free epoch), lanes carry, cap 0.8,
`RWKV_DELETED_CARDS=neighbour`), plus
`"augment": {"kind": "note_split", "min_reviews": 65536, "share": 0.5, "seed": 67}`:
- candidates = training users with >= 65,536 reviews that are in no evaluation set. The 437 large users of the 2,225
  pool (final-tier training users) stay whole, as in the control, and are never split. 959 of the 1,918 candidates
  are split (hash of seed and user), each into two pseudo-users (one random half of its notes; a deleted card counts
  as its own note), placed where the user was in the order.
- every card keeps its whole history: own inputs, ids and labels identical; only the context columns change.
- teacher targets = the original ones (the real user's replay file, `label_review_th` mapped through the half's kept
  original review numbers).
- pseudo ids `10,000,000 + 4 x user + part`; the item seed (id codes, day baseline, dropout) stays the real user's.
- control = the existing SM72s cool-down from the same epoch K, scored on the 732 set by the SM72s launcher (free).

Code (flag off = today, bit for bit): `rwkv/data_processing.get_rwkv_data(..., keep=None)`, `scripts/augment.py` (new),
`scripts/gpu_batch1.py` (`row_frame`, `seed_user`), `scripts/gpu_r01_prefetch.py`, `scripts/gpu_preprocessing_stage.py`,
`scripts/gpu_r01_training.py` (primer), `scripts/expertium_options.py` (`ReplayTargets` remap),
`scripts/expertium_decay.py prepare` (row expansion, recorded in decay.json `augment`),
`scripts/check_replay_label_windows.py` (pseudo rows), `scripts/check_augment.py` (new), `scripts/sm67_rule.py` (new).

## Steps of `artifacts/small-model/sm67/launch-sm67.sh`

0. Waits until the last line of `wt/sm77/artifacts/small-model/sm77/gpu/launch-sm77.log` is `] done` or a failure /
   STOP line (checked every 2 min, logged hourly); `touch artifacts/small-model/sm67/gpu/NO-WAIT` skips it.
1. CPU (~5 min): `expertium_decay.py prepare` (SM72s's command + augment) -> `artifacts/small-model/sm67-aug-decK-1ep32k-muon`;
   label-window trim pass + check; Muon fresh-state CPU check; `check_augment.py pair` (= the control's decay.json /
   protocol except `augment` and the expanded rows; same labelled reviews; no held-out user).
2. Ticket 0 (~15 min): `expertium_decay.py smoke` on a copy whose first rows are the two halves of one split user
   (pseudo-user path on GPU: half frames, real-user seeds, replay remap). Writes `gpu/gpu-smoke-ok`.
3. The cool-down in segments (`MIN_DECAY` 540 min; snapshot stop <= 2 min before a quiet window, resume after).
4. 732 scoring: 723 fit users at the 0.8 cap + the 3 long users at 0.92, leak-free inputs, w8+s8, label
   `sm67-aug-decK-1ep32k-muon/w8+s8` -> `gpu/sm67-neighbour.jsonl`.
5. Waits for the control's 732 rows (`s-w64kd-final-sc-lanes-leakfree-decK-1ep32k-muon/w8+s8` in
   `wt/sm72s/.../sm72s/gpu/small-lf-neighbour.jsonl`), then `scripts/sm67_rule.py decide` -> `sm67-decision.{json,md}`,
   one log line GO / STOP / GREY + ship check, then `done`.

GPU: ~8.7 h (smoke 0.2 h + cool-down ~8.2 h + scoring 0.3 h). Breaks (quiet.json + daily 21:00-02:00) add wall time,
not GPU time.

## Launch (PowerShell, hidden; start epoch 25)

```
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; START_EPOCH=25 bash C:/Users/jonat/workspace/wt/sm67/artifacts/small-model/sm67/launch-sm67.sh >> C:/Users/jonat/workspace/wt/sm67/artifacts/small-model/sm67/gpu/launch-sm67.nohup 2>&1"'; ProcessStartupInformation=$si}
```

From epoch 20 instead (if the epoch-25 cool-down loses and epoch 20 stays the release): `START_EPOCH=20` in the command
line; the control is then SM72s's cool-down from epoch 20 (already scored).

- Log `artifacts/small-model/sm67/gpu/launch-sm67.log`; `prepare.out`, `check.out`, `decay.out`, `score.out` next to it.
- Stop: `touch artifacts/small-model/sm67/gpu/STOP` (a running segment is snapshot-stopped).
- Relaunch after a session restart is safe: the prepared folder is re-checked, an interrupted cool-down resumes from
  its snapshot, scoring skips scored users. Check the launcher is alive after a Claude app update.
- Dry run: `DRYRUN=1 bash artifacts/small-model/sm67/launch-sm67.sh` (log `gpu/launch-sm67-dryrun.log`; runs the real
  CPU prepare and checks into `sm67-dryrun-*` and deletes it; GPU tickets only logged).

## Follow-up commands

```
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; WT=C:/Users/jonat/workspace/wt/sm67
tail -20 $WT/artifacts/small-model/sm67/gpu/launch-sm67.log
$PY -X utf8 $WT/scripts/sm67_rule.py decide --arm-label sm67-aug-dec25-1ep32k-muon/w8+s8 \
    --ref-label s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/w8+s8
cat $WT/artifacts/small-model/sm67/sm67-decision.md
```
- GO: stage 2 (augmented constant-LR epochs 21'-25' with re-drawn splits + re-seed, ~18 GPU h; DESIGN.md section 4).
- Ship check true: the arm is a release candidate after the fork's default-state check (int4 cards + int8 notes, ~13 min).
- Grey: report and ask. STOP: close data augmentation (SM67) (the note split is the only valid augmentation).

## Checks done (CPU, 10-30; no GPU)

`scripts/check_augment.py cpu` -> `artifacts/small-model/sm67/check-augment-cpu.json`, **ALL OK** (87 s, 2-4 threads):
- A0 flag off, loader: frames of 2 users bitwise equal to the `wt/sm72s` tree (run read-only, no bytecode written).
- T0 flag off, trainer path (SM72s cool-down options, replay KD, lanes carry, 2 lanes x 128, epoch-25 weights,
  2 users, 8 updates): prepared batch bytes, teacher targets, loss, outputs and every gradient bitwise equal to the
  `wt/sm72s` tree. T1: the same bits with the option on (users that are not split train exactly as in the control).
- B 3 split users (65,560 / 115,853 / 248,965 reviews): every kept card's own inputs, ids and labels identical on
  100% of rows, each review in exactly one half, labelled reviews conserved (59,434 = 30,557 + 28,877, ...), label
  review numbers map back to the whole user's. Context columns change (cum_reviews_today 97-100% of rows,
  diff_reviews 82-89%, diff_new_cards 62-71%, cum_new_cards_today 45-71%, day_offset_diff 0.2-1.3%).
- C teacher targets: on the trainer path (2 halves x 768 reviews, 6 updates) the remapped targets equal the direct
  lookup of the real user's file on every labelled row; index remap exact on every labelled row of the 3 users.
- E pseudo item seeds = training_seed + real user + offset; id codes of a half = the whole user's.
- D held-out: 0 training rows in the 464 / 732 / validation / final-test sets; 0 pseudo-users from any evaluation
  user (the 2,225 pool included); a held-out row is refused; order and reviews kept; the scorer never reads `augment`.
- Expansion (the arm's rows): 959 of 1,918 candidates split -> 8,959 rows, 187.1M reviews (32.2%) in halves, smallest
  half 31,826 (13 halves under 32,768), 581,942,738 reviews; 32k updates 22,079 -> 22,642 (+2.5%).

Dry run OK (`gpu/launch-sm67-dryrun.log`, START_EPOCH=25): the real prepare + trims (only user 5599, as the control) +
Muon fresh-state check (7 PASS) + pair check vs the control (`check-pair-sm67-dryrun-dec25.json`: same options except
augment, same start / schedule / users / order / reviews, labelled reviews 505,739,554 = 505,739,554); the smoke folder
built (halves of user 9409); GPU tickets only logged.

Not done (optional pre-flight from the design): the CPU proxy (score the released model on whole users vs their
halves); it does not gate the run.
