
## Update 2026-10-17 21:50 (evening results, night plan)
- Results (464 users, fork 0.29766 / 0.27216):
  - epoch 10 + decay at 32k (`s-w64kd-gate-ctx32768c-dec10`): 0.30096 / 0.27471 (+1.11% / +0.94%);
  - one-epoch decay at 2,048 (`s-w64kd-gate-ctx2048c-dec8-1ep`, SM35): 0.30225 / 0.27649. That is −0.00033 / −0.00019 vs the quarter-epoch decay, significant on both heads.
- Collection: dec10 0.4423 / 0.4375 (+0.14% / +0.18%).
- SM24 by stage (dec10): same-day +1.63% / +1.16%, after-fail +3.35% / +0.96%, spaced +0.79% / +0.84%.
- SM25 relabelled fork: +0.46% / +0.45% vs the real fork.
- Speed profile: `C:/Users/jonat/workspace/wt/fork-reference/artifacts/fork-reference/speed-20261018/report.txt`.
- Break is 22:00-02:00 (runbook f2 replaced f at 19:28). f2 is done.
- SM26 was skipped. `check_instant_loss_weight.py` takes `git merge-base HEAD small-model` as "before SM26", which equals HEAD once the code is cherry-picked (aa5ed3a, pushed). Fix the check: use the parent of the first commit adding instant_loss_weight. Then run SM26 (prepare is in runbook f2).
- SM30 started 21:27 into the break; I stopped it (ticket 000430 cancelled, wrapper killed, epoch 7 lost at 24%).
  - `wt/rep-flags/artifacts/small-model/launch-at-0200.sh` restarts it at 02:00 (log appended to `runbook-2026-10-18-repflags.log`).
  - Then `wt/state-vq/artifacts/small-model/launch-after-repflags-qat.sh` (running) starts the SM32b QAT decay. The GPU bench is correct: 85 min projected, MINUTES 110, gate 180. It writes `runbook-2026-10-18-qat.log`, then CPU scoring on users34.
