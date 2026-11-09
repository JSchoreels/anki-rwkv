# The show-time error sensitivity test (SM100): handoff (2026-11-09 16:30)

## Question (outside research; user 11-09 "let's try it")

The show-time query interval is rebuilt as the card's previous answer -> this review's answer minus this review's recorded
duration (`e2s-query`, `scripts/sm31_inputs.py` `show_time_seconds`; in training `show_time_query_rows`). When the recorded
duration is too short (undo + regrade, a timeout cap, a timer that does not tick) the rebuilt show time is too late and the
interval a few seconds to minutes too long. Undo and timeouts go with failing, so the error could in principle carry the
outcome into the evaluation. The model has no seconds-level collection gap input (the KEEP list of `gpu_r01_samples.py`;
`day_offset_diff` is day-level), so it can only use the error through its sensitivity to the query interval. We measure
that sensitivity: if moving every show-time interval by +10 s or +60 s barely moves instant LogLoss, the error carries no
usable signal.

## Design

- Model: the show-time instant-only small model (SM96), `sm96-inst-e31st-st-dec1ep32k/w8+s8`, checkpoint
  `wt/sm96/artifacts/small-model/sm96-inst-e31st-st-dec-1ep32k-muon/expertium/E89-F65e-F77a/master.pth` (read only,
  sha256 5bc83259645a, checked by the launcher).
- Inputs (new in this branch, `scripts/sm31_inputs.py`): `e2s-query+Ns` / `e2s-query-Ns` = `e2s-query` with every show-time
  interval + N s, with its cumulative, sin / cos and scaled columns and the curve query label moved consistently, floored at
  0, the -1 sentinel kept. The scorer accepts them through `--inputs` (labels `...@e2s-query+60s`).
- One GPU queue ticket (`sm100_score.py`), five passes of the SM96 scorer command (same path: `--pool 732`, 723 users of
  `eval-users-732-fit.json`, `--only .../w8+s8`, `--split-existing`, `--deleted-cards neighbour`, cap 0.8, memory-only
  retry at 0.92) + `--dump gpu/dump` (per-review instant predictions): shift 0 (re-scored for the dumps), +60 s, +10 s,
  +300 s, normal. Negative shifts skipped: a too-short recorded duration only makes the interval too long, and five passes
  already take ~65 min. The second model (the show-time retrain of the release (SM92)) is skipped for the same GPU budget.
- Rows `gpu/sm100-rows.jsonl`; dumps `gpu/dump/` (~5 GB, never committed).

## Pre-registered rule (`sm100_rule.py`, fixed before any GPU ticket)

- **Primary (verdict):** shifted - unshifted, instant head, the 722 rule users of the 732 set, review-weighted, paired user
  bootstrap 2,000, seed 0, 95% CI (the show-time report's `compare`, sha256 pinned), unit points = % of the big model
  (1 leak-free epoch) normal-input instant LogLoss 0.275757. **negligible** = the CI of BOTH +10 s and +60 s lies inside
  (-0.05, +0.05) points; **not negligible** otherwise (then report the size and look closer).
- **Secondary (reported):** +300 s the same way; mean |dR| of the instant predictions vs shift 0, its 99th percentile and the
  share with |dR| > 0.01; |dR| and the LogLoss change by show-time interval bucket (< 10 min, 10 min - 1 day, >= 1 day);
  reproduction of SM96's own rows (shift 0 and normal); the model's own show-time cost for scale (SM96: +0.055
  [+0.042, +0.072] points).
- **Timeout subset (reported):** cap = the user's max recorded duration, "capped" if >= 10,000 ms, >= 20 times and >= 0.5% of
  the user's reviews; timeout reviews = their benchmark reviews at the cap. Share, failure rate vs other reviews, instant
  LogLoss normal vs show-time (shift 0) on them (paired user bootstrap), the capped users' other reviews, and +60 s vs shift 0
  on them. Note: on a 60,000 ms timeout review, +60 s rebuilds exactly the normal (answer-to-answer) interval.

## CPU proofs (before launch)

- `sm100_check.py` -> `check-cpu-sm100.json` ALL OK, on users 2388, 4676, 1712, 1286: whole scorer batch of `e2s-query` in
  this branch == SM96's code (`git show 4b3d96c0:scripts/sm31_inputs.py`) bit for bit; `e2s-query+0s` == `e2s-query` bit
  for bit; for +10 / +60 / +300 / -60 s only the 8 query interval columns of query rows move, by exactly the shift, the
  derived columns follow the formulas, the curve label moves the same; +60 s changes only the `start` (inputs) and `labels`
  tensors; malformed variant names refused.
- `sm100_rule.py selftest` OK (SM96's show-time cost +0.055 [+0.042, +0.072] reproduced; e2s - e2s = 0).
- `cpu-smoke.txt`: the scorer on CPU for 2 users x 5 variants with dumps; shift 0 and normal reproduce SM96's GPU rows to
  6 decimals; the dump LogLoss equals the row LogLoss; the timeout code path runs.

## What runs

- Launcher `artifacts/small-model/sm100/launch-sm100.sh`, started 2026-11-09 16:20 hidden via WMI from commit 29975fac:
  wrapper bash pid **56180**, launcher bash pid **34440**; queue ticket **001087-sm100** (waiting behind stage 2 (SM98)'s
  epoch 34 ticket 001086 at 16:20, max wait 141 min). Log `gpu/launch-sm100.log`
  (ends `done (verdict ...)` or a `STOP:` line); scorer output `gpu/score.out`; tickets `gpu/tickets.jsonl`; dry run
  `gpu/launch-sm100-dryrun.log` (DRYRUN OK).
- Queue owner `sm100`; purpose starts "the show-time error sensitivity test (SM100): ". One ticket, expected 75 min (SM96:
  12 min per 723-user pass). Breaks: `sm100_block.py` (quiet.json + daily 20:00-01:00, Sunday from 16:00); the ticket
  starts only when 78 min fit before the break; if the queue wait would reach the break it re-plans after the break.
- Expected: stage 2 (SM98)'s epoch 34 holds the GPU since 15:56 (~125 min, to ~18:00); this ticket is next in the queue,
  so ~18:00-19:10, then the rule (CPU, a few minutes). Expected end ~19:15 Mon 2026-11-09. If epoch 34 runs past ~18:45,
  the ticket waits for the break and runs from 01:00 (end ~02:20 Tue).

Alive check: `tail -5 gpu/launch-sm100.log`;
`powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | ? { $_.CommandLine -like '*launch-sm100*' } | select ProcessId,CommandLine"`.
Stop: `touch gpu/STOP` (between scorer passes; or `r01_queue.py cancel` the ticket). Relaunch (only if the process is gone and
the log has no `done`; scored users are skipped):

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash /c/Users/jonat/workspace/wt/sm100/artifacts/small-model/sm100/launch-sm100.sh >> /c/Users/jonat/workspace/wt/sm100/artifacts/small-model/sm100/gpu/launch-sm100.nohup 2>&1"'; ProcessStartupInformation=$si}
```

Do not edit `scripts/` or the sm100 `*.py` while it runs (the launcher refuses uncommitted changes; the scorer records
provenance). Never touch `wt/sm96`, `wt/sm98`, `wt/sm57`.

## Follow-up (fresh Sonnet agent, medium effort; do not wait on the GPU)

1. `git -C C:/Users/jonat/workspace/wt/sm100 fetch` and fast-forward `sm100` if needed. `tail -8 artifacts/small-model/sm100/gpu/launch-sm100.log`.
   If it ends with `done (verdict ...)`, go on. If `STOP:`, read that line and the tail of `gpu/score.out`, fix only a
   clear transient (relaunch with the command above) and report otherwise.
2. If the rule did not run: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm100/sm100_rule.py decide`
   (CPU, a few minutes; exit 2 = rows missing).
3. Write `artifacts/small-model/sm100/sm100-result.md` from `sm100-decision.md/json`: verdict; the shift table (+10 s, +60 s,
   +300 s: instant LogLoss, points [95% CI], mean |dR|, share |dR| > 0.01); the bucket table; the timeout subset (capped users,
   share, failure rates, LogLoss show-time vs normal and +60 s vs show-time); the reproduction check; the ticket's wall
   minutes (`gpu/tickets.jsonl`) and the scorer's review-evals/s (last `review-evals/s` line per pass in `gpu/score.out`).
   One plain-sentence interpretation: if negligible, "the rebuilt show-time interval's error cannot carry a usable signal";
   if not, give the size and which bucket carries it (expected: reviews with show-time intervals under 10 minutes).
4. Commit with explicit paths and `git add -f`: `sm100-decision.json`, `sm100-decision.md`, `sm100-result.md`,
   `gpu/sm100-rows.jsonl`, `gpu/launch-sm100.log`, `gpu/launch-sm100.nohup`, `gpu/tickets.jsonl`, `gpu/score.out` (if
   under ~5 MB; else its last 200 lines as `gpu/score-tail.out`). Never `gpu/dump/`, checkpoints, snapshots, `.claude/`,
   `docs/GPU_OPTIMIZATIONS_LOG.md`, quiet.json. As
   `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line
   + `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Right before the push: `git fetch`, merge `origin/sm100`,
   push `sm100`. Do not merge into main; do not touch the Anki fork.
5. Report in plain names with IDs in parentheses (e.g. "the show-time instant-only small model (SM96)"), units stated
   (points = % of the big model (1 leak-free epoch) normal-input instant LogLoss), never "Expertium" (say "outside research").
