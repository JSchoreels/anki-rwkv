# SM43 full: handoff (launched 10-20 13:59)

Data diversity vs epochs at equal reviews seen, full size, with an LR decay at the end. The design is in
`sm43_full_prepare.py` (docstring). The mini (10-19) found B +0.36% curve / +0.46% instant worse (FAIL).

| arm | experiment | users | epochs | reviews/epoch | reviews total |
|---|---|---|---|---|---|
| A | `sm43full-a-1250u-11ep` | 1,250 (125 per length decile, seed 4343) | 10 constant + 1 decay | 96,446,289 | 1,060,909,179 |
| B | `sm43full-b-125u-110ep` | 125 of A's (13/12 per decile, pick seed 25) | 100 constant + 10 decay | 9,618,195 | 1,058,001,450 |

B/A total reviews = 0.9973 (within 0.5%). Users: `sm43-full-users.json`. Recipe: options of
`scale-w64kd-gate-cached-ctx2048` (2,048-review windows, cached KD, gpu_memory_fraction 0.8), seed 2025, order seed
20261013, from scratch.

## Decay mechanism
`scripts/lr_tail_decay.py`, an accuracy-loop `epoch_runner` hook (`experiment.json`: `"epoch_runner":
"scripts.lr_tail_decay:run_epoch"`, `"lr_tail_decay": {"epochs": N}`). Epochs 1..E-N use the accepted constant R01 LR
(7e-4). Decay epoch j of N gets the trainer's own `full_run.lr_schedule` (`expertium_options.LearningRate`) with
peak = 7e-4 (N-j+1)/N, final = (N-j)/(N-j+1), linear, so the LR is one continuous linear ramp 7e-4 -> 0 over the
last N epochs (A: N = 1, B: N = 10). Each epoch continues from the previous weights and AdamW state, as usual. As in
every lr_schedule run, KD uses decay_alpha 0.5 (not 0.9) during the decay epochs. Both arms use this mechanism.
Why not run.json `final_lr_schedule`: that one only covers the last epoch, which fits A but not B's 10 decay epochs.
The CPU dry run checked the protocol through accuracy_loop's own wrapper. Epoch 100 has no schedule. Epoch 101 is
7e-4 -> 6.3e-4 and epoch 110 is 7e-5 -> 0. The ramp is continuous across epochs.

## Running
- Launcher: `artifacts/small-model/sm43/launch-sm43-full.sh`. The bash pid is 1075 in MSYS and 19464 in Windows.
- Log: `artifacts/small-model/sm43/launch-sm43-full.log`
- Queue ticket for training A: 000555-small-model (wrapper pid 38864, command pid 240), started 13:59:26.
  Training B, then the two 732-pool scoring passes, get their own tickets later.
  The launcher waits while `artifacts/small-model/HOLD-chain` exists before each ticket, and retries a failed
  training ticket once.
- Measured rate in the mini: about 101.9k reviews/s, plus about 17 s of process start per epoch.
- Estimate: A 11 x ~16 min, about 2.9 h, ending around 16:55. B 110 x ~112 s, about 3.4 h, ending around 20:20.
  Scoring: pass 1 about 27 min, pass 2 about 2 min. Pairing on CPU takes seconds.
  **Expected end: about 20:50 (10-20)**, so about 6.9 h in all. That is longer than the 5.5 h first guess: with 110
  short epochs, B pays about 30 min of process start alone.
- **Do not edit `scripts/` in wt/small-model until the log says `done`**: SourceWatch kills the epoch.

## Outputs
- Checkpoints:
  - A: `artifacts/accuracy/experiments/sm43full-a-1250u-11ep/run/epoch-11/expertium/E89-F65e-F77a/master.pth`
  - B: `artifacts/accuracy/experiments/sm43full-b-125u-110ep/run/epoch-110/expertium/E89-F65e-F77a/master.pth`
  - Constant-LR ends, kept: epoch-10 for A and epoch-100 for B.
- Per-epoch timings: `artifacts/accuracy/experiments/<exp>/phases.jsonl`
- 732-pool rows: `artifacts/small-model/sm43/heldout-sm43full-pools.jsonl`
  - labels `s-w64kd-sm43full-a-1250u-11ep/w8+s8` and `s-w64kd-sm43full-b-125u-110ep/w8+s8`
- Paired result:
  - `artifacts/small-model/sm43/sm43-full-decision.json` and `sm43-full-summary.md`.
  - B vs A, review-weighted, user bootstrap with 2,000 resamples, seed 0, 2.5/97.5%.
  - Both arms are also compared with the fork's w8+s8 rows from `heldout.jsonl` + `heldout-pools.jsonl`.
  - The math reproduces the mini decision exactly on the mini's rows.

## Follow-up (when the log's last line is `[sm43full ...] done`)
```bash
cd C:/Users/jonat/workspace/wt/small-model
grep -aE '^\[sm43full|\[queue\]' artifacts/small-model/sm43/launch-sm43-full.log | tail -20
cat artifacts/small-model/sm43/sm43-full-summary.md
# if the summary says INCOMPLETE (< 700 common users): one more scoring pass, then re-pair
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py run --owner small-model \
  --purpose "SM43 full scoring A and B, 732 pool (pass 3)" --claim gpu --expected-minutes 15 -- \
  C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/small-model/scripts/small_model_heldout.py --pool 732 \
  --model "s-w64kd-sm43full-a-1250u-11ep=expertium:w64:C:/Users/jonat/workspace/wt/small-model/artifacts/accuracy/experiments/sm43full-a-1250u-11ep/run/epoch-11/expertium/E89-F65e-F77a/master.pth" \
  --model "s-w64kd-sm43full-b-125u-110ep=expertium:w64:C:/Users/jonat/workspace/wt/small-model/artifacts/accuracy/experiments/sm43full-b-125u-110ep/run/epoch-110/expertium/E89-F65e-F77a/master.pth" \
  --only "s-w64kd-sm43full-a-1250u-11ep/w8+s8" --only "s-w64kd-sm43full-b-125u-110ep/w8+s8" --time-limit 3000 \
  --out C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/sm43/heldout-sm43full-pools.jsonl
# pairing (CPU, also the launcher's last step)
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm43/sm43_full_summary.py
# commit + push the results
git add -f artifacts/small-model/sm43/heldout-sm43full-pools.jsonl artifacts/small-model/sm43/sm43-full-decision.json \
  artifacts/small-model/sm43/sm43-full-summary.md artifacts/small-model/sm43/launch-sm43-full.log \
  artifacts/accuracy/experiments/sm43full-a-1250u-11ep/phases.jsonl artifacts/accuracy/experiments/sm43full-b-125u-110ep/phases.jsonl
git -c user.name="Jonathan Schoreels" -c user.email=jonathan.schoreels@gmail.com commit -m "SM43 full: 732-pool rows and paired result" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```
If the log ends with `A failed` or `B failed`, read the last `[train]` lines and the epoch folder's `result.json`
`error`. `accuracy_loop.py train <exp>` resumes at the first incomplete epoch, through the queue as above with
`--expected-minutes` scaled to the remaining epochs. Then run the two scoring passes and the pairing by hand.
After the result: update the factory page (pages/factory.html via factory-curated.json). Report reviews/s per arm,
plus absolute LogLoss and the eval size.
