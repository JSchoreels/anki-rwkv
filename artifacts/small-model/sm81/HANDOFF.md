# Scheduling-time vs review-time accuracy (SM81): handoff

Question (user): FSRS7-Instant is much better at review time but worse at scheduling time (shared-5k: -9.5% LogLoss at review time, -1.9% at the start of the day, +3.8% at scheduling time vs FSRS-7). Does the small model's advantage over FSRS have the same nuance? This measures, on the same reviews and with the same metric, FSRS-7 (default and per-user fit), FSRS7-Instant, the released small model (the leak-free small model (SM72s), cool-down from epoch 25; label `s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/w8+s8`) and the big model (1 leak-free epoch) (label `big-leakfree-1ep/w8+s8`) at scheduling time, at the start of the day and at review time.

Branch `sm81`, worktree `C:/Users/jonat/workspace/wt/sm81`, cut from `origin/sm77` (origin/main has no held-out scorer; sm77 carries it with the detach fix bf938aa4).

## Protocol

- **RWKV points** (scorer `scripts/small_model_heldout.py --start-of-day`, `scripts/sm81_sod.py`). Scheduling time = curve head `p_curve(t)` from the card's previous answer at the true elapsed time. Review time = instant head `p_imm` on the query row just before the answer. Start of day = `p_imm` from the state after every review before the review's day only: every query row of day D is moved in front of day D's first update row (query rows commit no state), with the row's own inputs unchanged (the start tensor of the normal pass, permuted). Day = the dataset's `day_offset` (anki-revlogs-10k: the Anki day with the user's rollover). Entities first seen on day D (a new card asked again that day, the user stream on the first day) have no start-of-day state: zero state, own input as token shift. Leak-free inputs (`--deleted-cards neighbour`, `env -u RWKV_DELETED_CARDS`), w8+s8, 723 users at the 0.8 cap + the 3 long users at 0.92 = 726 users (`users-726.json`).
- **FSRS side** (`scripts/sm81_fsrs.py`, srs-benchmark's own code and venv, read only, on the RWKV scorer's dataset copy `data/anki-revlogs-10k-original-20261011`, so review_th = row number in the same revlog file on both sides). Test reviews = the published FSRS-7 protocol's 5 time-series test folds (`--short --secs --equalize_test_with_non_secs`: reviews whose card was last reviewed on an earlier day). FSRS-7 per-user fit = per fold, fitted on the user's reviews before that fold (fsrs-rs `tools/fit-v7`, recency weight 0.0667 + 0.9333 (i/N)^11.25 = the published FSRS-7 recency row). FSRS-7 default = default parameters, no fit. FSRS7-Instant = the shared-5k protocol (one shared start fit on 100 training users, online updates over the whole stream, config "FSRS7-Instant"), its instant / scheduling / start-of-day modes.
- **Common row set**: per user, FSRS-7 test reviews that are also RWKV benchmark reviews with all three RWKV predictions, for both RWKV models (= answered reviews with a previous review of the card, on a later day). Every model is scored on exactly these reviews. `sm81_table.py` exits 1 if the two sides' y labels differ on any common review, if a review_th exceeds the user's review count, or if a user has fewer than half of its FSRS-7 test reviews in common.
- **Metric**: LogLoss, review-weighted (all common reviews pooled, our standard) and per-user mean (the benchmark's), % vs FSRS-7 per-user fit at scheduling time, paired user-bootstrap 95% CI of the difference (2,000 resamples, seed 0). Plus per model: the share of the review-time gain (scheduling time -> review time) already known at the start of the day, overall and split into the user's first review of a day (start of day = review time by construction) vs the later reviews of the day.

## CPU proof (done, 10-31 ~20:50, `cpu-proof/`, `cpu-proof.sh`)

3 small users of the 732 set (2388, 4676, 1712), both models, w8+s8 and fp32: PASS.
- (a) off = bit-identical: the patched scorer without the option writes the same rows as the scorer at the branch point (every field but timings / provenance), and the normal pass of a `--start-of-day` run is identical too.
- (b) start of day = review time on the user's first review of each day, up to float rounding: fp32 max |diff| 1.8e-7 (most bitwise equal); w8+s8 max 1.2e-4 (int8 state rounding flips); the other reviews differ on 100% of them (median |diff| 0.002-0.07).
- (c) the curve head of the start-of-day pass equals the normal pass: fp32 max |diff| 5e-7, w8+s8 3e-4, LogLoss within 1e-6. Moving rows changes their positions in the batched matmuls, so equality is to float rounding, not bitwise.
- FSRS side, same 3 users: y equal on all common reviews; common rows 555/555, 1,435/1,435, 928/1,080 of the FSRS-7 test reviews (the RWKV label set's first fold starts later for user 1712); FSRS7-Instant LogLoss within 1e-4 of the published shared-5k rows in all three modes (those used the other dataset copy).
- Seen on the CPU users: on same-day repeats (not in the FSRS-7 test set, so not in the table) the RWKV start-of-day state lacks the card's own earlier answer of that day and is much worse (user 1712: 0.81 vs 0.39 at scheduling time); on the cross-day reviews it sits between scheduling and review time.

## Timeline (launched 10-31 ~21:00, waiting)

| Step | Expected |
| --- | --- |
| break until 01:00 Sun | waiting |
| FSRS-7 fits + FSRS7-Instant replay, CPU, 6 processes (alongside the data-order re-seed's (SM80) cool-down) | 01:00 - ~01:45 |
| wait for the data-order re-seed (SM80) launcher ("] done" / STOP) | ~03:30 |
| small model 723 users (start-of-day pass + dump) | ~30 min |
| small model 3 long users (0.92) | ~10 min |
| big model 723 users | ~35 min |
| big model 3 long users (0.92) | ~10 min |
| table (CPU) | ~05:10 Sun |

Log: `artifacts/small-model/sm81/gpu/launch-sm81.log` (the waiter key is `sm81`). Rows: `gpu/sm81-neighbour.jsonl` (normal-pass fields = our standard rows, plus `start_of_day`), per-review dumps `gpu/dump/` (not committed, ~1 GB), FSRS per-user files `gpu/fsrs/`, scorer output `gpu/score.out`, FSRS output `gpu/fsrs.out`.

## How to read the result

- `sm81-result.md` / `.json`: one table (rows = FSRS-7 default, FSRS-7 per-user fit, FSRS7-Instant, small model, big model; columns = scheduling time / start of day / review time, review-weighted and per user, % vs FSRS-7 per-user fit at scheduling time with CIs), row counts, and the "share known at the start of the day" line per model. JSON `checks`: start-of-day pass curve agreement, first-of-day agreement, and the normal pass vs the existing rows (`wt/sm72s/.../small-lf-neighbour.jsonl`, `wt/sm72/.../bigleak-neighbour.jsonl`), which should agree to GPU rounding.
- The scheduling-time column is the honest comparison for scheduling; the RWKV scheduling -> review-time gap mixes the head difference (curve vs instant head) with the information arriving between the card's previous answer and the review.
- Re-run the table only: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 scripts/sm81_table.py --rows artifacts/small-model/sm81/gpu/sm81-neighbour.jsonl --dump artifacts/small-model/sm81/gpu/dump --fsrs artifacts/small-model/sm81/gpu/fsrs --label-small s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/w8+s8 --label-big big-leakfree-1ep/w8+s8 --existing-small C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/gpu/small-lf-neighbour.jsonl --existing-big C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl --out-md artifacts/small-model/sm81/sm81-result.md --out-json artifacts/small-model/sm81/sm81-result.json` (from the sm81 worktree).
- Then commit `sm81-result.md`, `sm81-result.json`, `gpu/sm81-neighbour.jsonl`, `gpu/fsrs/users.jsonl`, `gpu/launch-sm81.log` (benchmark users only; `git add -f`), push `sm81`.

## Relaunch / stop

- Relaunch (finished users and steps are skipped), from PowerShell: `$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm81/artifacts/small-model/sm81/launch-sm81.sh >> C:/Users/jonat/workspace/wt/sm81/artifacts/small-model/sm81/gpu/launch-sm81.nohup 2>&1"'; ProcessStartupInformation=$si}`
- Dry run: `DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm81/artifacts/small-model/sm81/launch-sm81.sh` (log `gpu/launch-sm81-dryrun.log`).
- Stop: `touch C:/Users/jonat/workspace/wt/sm81/artifacts/small-model/sm81/gpu/STOP`; skip the SM80 wait: `touch .../sm81/gpu/NO-WAIT`.
- Do not edit `scripts/` of the sm81 worktree while it runs (the launcher refuses uncommitted source changes at start; a running scorer would mix revisions).
