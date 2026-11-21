# Handoff: answer-time expressiveness (SM142)

Design: `design.md` (pre-registered 11-21 before any GPU ticket; a description, not a gate). Code: `scripts/sm142_expr.py` (dump / verify / analyse / report / check). CPU proof PASS (`check-cpu.json`: the CPU dump of 2 users equals SM138's recorded GPU sums within 1.5e-7 relative; the baselines equal a brute force; quantiles equal `median_log_s`; AUC equals brute pairs). Dry run OK (`gpu/launch-sm142-dryrun.log`). A pipeline test on 30 small users (CPU dump, analyse, report into `work/pipe/`, gitignored) ran end to end; its numbers are a pipeline check, not a result.

## The running launcher

- Launched 17:31 Sat 11-21 from `8185c5ba` (branch `sm142`), hidden via WMI: wrapper bash PID 22060 (`bash .../launch-sm142.sh`). First queue ticket **001444** (dump of the six-layer Tiny two-head model (SM130)), started 17:31 at once (queue was empty).
- Steps: one GPU dump ticket per model (cap 0.8; a second pass at 0.92 only for missing users), each followed by a CPU `verify` against the model's recorded scoring rows (same review sha256, counts exact, LogLoss and |ln error| sums of the head and the ablation within 1e-6 relative; at least 700 of 722 users must match). Order: sm130th, sm130io, sm113, sm118, sm99. Expected 14 to 20 min each (SM125 / SM128 dumps), **all dumps done about 19:00 Sat 11-21** (up to 19:30 if the Big model is slower).
- Log: `C:/Users/jonat/workspace/wt/sm142/artifacts/small-model/sm142/gpu/launch-sm142.log` (lines `[sm142 MM-DD HH:MM:SS]`); stdout/stderr `gpu/launch-sm142.nohup`; per model `gpu/dump-<key>.out`, index `gpu/dump-<key>.jsonl`, `gpu/verify-<key>.json`, marker `gpu/dump-ok-<key>`. Dumps: `work/dump-<key>/<user>.npz` (gitignored, about 12 GB per model; never commit).
- Done: `grep -E '\] done$' .../gpu/launch-sm142.log`. Failure: `grep -E '\] STOP' .../gpu/launch-sm142.log` (the one before `STOP: ended with exit 1` names the step). Report; do not relaunch blindly.
- Alive (PowerShell): `Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -like '*launch-sm142*' } | Select ProcessId, CommandLine`

## Stop / relaunch

- Stop between tickets: `touch C:/Users/jonat/workspace/wt/sm142/artifacts/small-model/sm142/gpu/STOP`. A running dump: cancel its ticket (`C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py cancel <ticket>`) and kill the launcher; a relaunch skips dumped users and verified models.
- Relaunch (remove STOP first):

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm142/artifacts/small-model/sm142/launch-sm142.sh >> C:/Users/jonat/workspace/wt/sm142/artifacts/small-model/sm142/gpu/launch-sm142.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- Do not edit `scripts/`, `expertium/`, `rwkv/` of the sm142 worktree, and do not merge into it, while it runs (the launcher refuses uncommitted source at start). Edits under `artifacts/` are fine.

## Follow-up after `done` (CPU, fresh agent; Sonnet medium is enough)

The analysis is CPU only: one process per model, 1 thread, well under 6 GB RAM (the pipeline test: 85,000 rows in 2 s plus the baselines); at most 2 processes at a time (SM140's CPU recalibrations may still be running tonight). The first model also fills the model-independent baseline cache `work/base/<user>.npz`, so run it alone, then the others two at a time. Each run is resumable (analysed users are skipped).

```bash
cd C:/Users/jonat/workspace/wt/sm142
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; D=artifacts/small-model/sm142; X="$PY -X utf8 scripts/sm142_expr.py"
grep -E 'dumped users|verify exit|done|STOP' $D/gpu/launch-sm142.log | cut -c1-300
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
$X analyse --model sm130th > $D/gpu/analyse-sm130th.out 2>&1          # alone first (fills work/base/)
$X analyse --model sm130io > $D/gpu/analyse-sm130io.out 2>&1 & $X analyse --model sm113 > $D/gpu/analyse-sm113.out 2>&1 & wait
$X analyse --model sm118 > $D/gpu/analyse-sm118.out 2>&1 & $X analyse --model sm99 > $D/gpu/analyse-sm99.out 2>&1 & wait
tail -1 $D/gpu/analyse-*.out
$X report                                                              # -> $D/sm142-result.{md,json}
$PY C:/Users/jonat/workspace/anki-rwkv/scripts/r01_inspect.py $D/sm142-result.md --lines 40
```

1. Write-up: add a short paragraph at the top of `sm142-result.md`, in full plain sentences: per model the label (template-like / partly expressive / card-expressive / shape-expressive) and the headline numbers: within-user, within-grade AUC of the head vs the card's running median (a) at 2 / 4 / 8 s with the CI of the difference; the review-level slope and the out-of-sample between-card slope; the between-card explained share (head vs (a) vs the ablation); the per-user effective number of distinct curves (location included and normalised, 0.1 grid) and the pooled PCA k95; the card-level shape calibration (head vs the card's own earlier spread). Answer the user's question plainly: how many genuinely different time curves a user gets, and whether the head knows which card is slow better than the card's own history. Answer time = front + back ("answered within S s, reading the back included"); models always as "plain name (ID)"; "%" never "points"; never "Expertium" ("outside research"). Then mark this HANDOFF closed.
2. Commit on `sm142` with `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit` and `git add -f`: `sm142-result.{md,json}`, `rows/sm142-*-users.jsonl` (per-user aggregates only, allowed), `gpu/*.log`, `gpu/*.out`, `gpu/*.json`, `gpu/dump-*.jsonl`, this HANDOFF; never `work/`, `.npz`, `.pth`. Message ends with a blank line + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. `git fetch` + merge `origin/sm142`, push `sm142`; never force-push.
3. Factory page: do NOT edit `pages/data/factory-curated.json` yourself; hand the results-row text (below, filled in) to the main session.

## Factory rows (for the main session)

Plan row (kind `gpu`, minutes 100, match `"Answer-time expressiveness (SM142): "`, state `running`, started_at `2026-11-21T17:31+01:00`):

- what: "Answer-time expressiveness (SM142): does the answer-time head give each card its own predicted time curve, in location and in shape, or only a few templates around the user&rsquo;s average? One GPU dump per model of the full predicted time distributions (4 grades &times; 16 bins) with card ids, for the longer Small Instant-Only model (SM113), the show-time two-head cool-down for Small (SM118), the six-layer Tiny two-head and Instant-Only models (SM130) and the show-time instant-only big model (SM99); then on CPU, within each user and grade: the AUC of P(answered within S s, reading the back included) against the card&rsquo;s running median, the card&rsquo;s last time and the user&rsquo;s running median; calibration slopes and the out-of-sample spread between cards; the share of the persistent card-to-card differences explained; the number of genuinely different curves per user (PCA, 10% grid) and whether wider predicted curves really vary more; and how the prediction moves over a card&rsquo;s life. A description, not a gate."
- when: "Started 17:31 Sat 11-21 (queue ticket 001444 onwards): GPU about 1.5 h (five dumps), dumps done about 19:00; CPU analysis afterwards (about 1 h), read-out Sat evening or Sun morning"
- decides: "Whether the answer-time predictions are exploitable card by card (which card is slow, which card is erratic), not only right on average; it shapes what the time features may promise in the client"

Results row (fill in after `report`; kind `cpu-analysis` or as the SM137-style rows, inputs `show-time`, tickets 001444 onwards, gpu_minutes from the log): takeaway = the plain-language read-out of `sm142-result.md` section 6 (per model: curves per user, AUC vs the card median at 2 / 4 / 8 s, between-card share explained vs the card median and the ablation, the label); detail.verdict = the labels; detail.table = the summary table.

## CLOSED 11-21 (evening)

All five dumps verified (the show-time instant-only big model (SM99): 719 of 722 users), `analyse` run for all models (about 7 min each, `gpu/analyse-*.out`), `report` written, read-out added at the top of `sm142-result.md`. Nothing remains here; factory page rows are written by the analysis agent.
