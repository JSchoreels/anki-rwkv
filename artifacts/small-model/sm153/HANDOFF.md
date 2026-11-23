# Handoff: answer-time spread shrink (SM153)

Design (pre-registered, committed 507031ff before any shrunk score): `design.md`. Code: `scripts/sm153_shrink.py` (imports SM142 / SM125 code read only from `wt/sm142`, commit c8764a2e). Checks C1, C3, C4 PASS (`check-cpu.json`).

## Running (detached via WMI, launched 23:30 Mon 11-23)

`run-grid.sh`: the grid pass, 2 processes at a time, 1 thread each, about 10 min per model. The six-layer Tiny two-head model (SM130) and the longer Small Instant-Only model (SM113) are done (23:39). Then the show-time two-head cool-down for Small (SM118) and the six-layer Tiny Instant-Only model (SM130), then the show-time instant-only big model (SM99). Expected end about 00:00. Log: `logs/chain.log` (the run is over when it shows `grid done`), per model `logs/grid-<key>.out`.

## Follow-up (fresh agent, Sonnet medium is enough)

```bash
cd C:/Users/jonat/workspace/wt/sm153; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; X="$PY -X utf8 scripts/sm153_shrink.py"
grep "grid done" artifacts/small-model/sm153/logs/chain.log
$X fit                                   # rewrite sm153-fit.json on the complete grids (the 23:40 fit below was partial for sm118/sm130io)
# held-out users, 2 at a time (about 10 to 15 min each; three SM142 analyses per user):
$X apply --model sm99 & $X apply --model sm113 & wait
$X apply --model sm118 & $X apply --model sm130th & wait
$X apply --model sm130io
# after-recalibration arm (combined shrink, held-out users, only the 21 cells fitted): time one model on --limit 20 first;
# at about 7,200 rows per second it is about 20 min per model; if the 5 models need over 1 h of CPU, run them detached (WMI) and hand off again
$X recal --model sm113 --arm combined    # ... per model
$X report                                # -> sm153-result.json; then write RESULT.md
```

C2 is in the apply rows (`c2_same_rows`, `c2_max_abs`): the raw events rebuilt from the SM142 distributions must equal the SM125 / SM128 / SM140 dumps. The recal baseline is the recorded per-user `cal` sums of SM125 / SM128 / SM140 (identical cells, see `lean_valid`); check this by running `recal --arm raw --limit 3` on one model and comparing with the recorded rows.

## Preliminary read (fit users only, 23:40; NOT the held-out result)

The fitted location shrink factor is beta = 1.0 (no shrink) for every model; the shift alone gains nothing. The mixture with the user's running per-grade distribution picks alpha = 0.85 to 0.93 and lowers fit-user time LogLoss by -0.17% (the longer Small Instant-Only model (SM113)) and -0.30% (the six-layer Tiny two-head model (SM130)); the partial grids for the other two gave -0.33% to -0.53%. That looks like tail widening (a defensive mixture) rather than a fix of the between-card spread. Whether it holds on held-out users, what it does to AUC and the number of distinct shapes, and whether the rolling recalibration already absorbs it are still open.

## Update 23:57 Mon 11-23 (agent 2): apply + recal running detached

Grid done; `fit` rerun on the complete grids (`sm153-fit.json`, combined alpha 0.90 to 0.94, beta 1.0 to 1.05, fit-user gain -0.12% to -0.30%; shift alone gains nothing). The remaining CPU work (apply for 5 models plus recal combined for 5 models) is well over 1 h, so it runs detached (WMI, PID 39908 at launch): `run-apply.sh`, two lanes (sm99, sm118, sm130io | sm113, sm130th then `recal --arm raw --limit 3` on sm113 as check C2), then `report`. Progress: `logs/apply-chain.log` (done when it shows `all done`), per step `logs/apply-<key>.out`, `logs/recal-<key>.out`. Estimate 1.5 to 3 h. Scripts resume (rows are appended, finished users skipped), so if it died (check with `tasklist`), relaunch `run-apply.sh` the same way.

Next (fresh Sonnet medium agent): when `all done`, read `rows/` and `sm153-result.json`, check C1/C2 (apply rows `c2_same_rows`, `c2_max_abs`; the raw recal check on 3 users against the recorded sums), write `RESULT.md` (question, rule, verdict per model before and after recalibration, time LogLoss change with CI, within-user AUC, between-card slope, range and distinct shapes, expressiveness warning, CPU minutes, meaning for the fork's answer-time estimates), commit, push.
