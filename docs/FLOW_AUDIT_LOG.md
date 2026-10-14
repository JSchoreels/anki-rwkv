# Flow audit log

One entry per finished queue ticket. See [FLOW_AUDIT.md](FLOW_AUDIT.md) for the checklist.

## 2026-10-14 18:28 — 000309-compression
- verdict: clean (0 violations; items 1, 2, 3 predate the rule)
- 1 lane: predates rule (no timing flag, but the log prints review-evals/s, about 38k/s) · 2 quiet: predates rule (ran 16:58–17:52 local, outside 21:00–01:00 anyway) · 3 expected: predates rule (no expected_minutes; real 53.8 min, purpose says <57 min)
- 4 exit: ok (0) · 5 handoff: ok (HANDOFF.md names 000309, logs/heldout-b-run1.log, heldout-b.jsonl) · 6 pushed: ok (compression = origin/compression at e59ac71; main in sync)
- 7 docs: pending (speed row not processed yet) · 8 no waiting agent: not verifiable from the record · 9 outside work: ok now (nvidia-smi shows no GPU compute processes; no non-queue GPU jobs in handoff)
- notes: the job hit its own 3,200 s limit after 300 of 1,000 users (exit 0), so chunk 2 or a re-run is needed. HANDOFF.md says "(not pushed)", which is stale. It also says "Speed: unmeasured", but the log reports 38k review-evals/s.

## 2026-10-14 18:47 — 000310-compression
- verdict: 1 violation
- 1 lane: ok (accuracy job, no timing flag needed; log prints review-evals/s, not a benchmark) · 2 quiet: n/a (not timing; ran 17:51–18:46 local, outside 21:00–01:00 anyway) · 3 expected: VIOLATION (no expected_minutes in record; real 54.4 min, purpose says <57 min) · 4 exit: ok (0)
- 5 handoff: ok (artifacts/compression/HANDOFF.md at b8a2408 names 000310, logs/heldout-b-run2.log, heldout-b.jsonl) · 6 pushed: ok (compression and main in sync with origin) · 7 docs: pending (result not processed yet) · 8 no waiting agent: not verifiable from the record · 9 outside work: ok (nvidia-smi shows no GPU compute processes)
- notes: HANDOFF.md is under artifacts/compression/, not the track root. Its later commit adds 000321/000322, so item 5 was checked against the current file as instructed.

## 2026-10-14 19:29 — 000311-compression
- verdict: 1 violation (lane), 1 push gap fixed by this push
- 1 lane: VIOLATION (summary.json reports reviews_per_second 269.0 and HANDOFF says to report it, but the record has no timing flag; purpose is accuracy, so the lane is arguable) · 2 quiet: n/a (not timing; ran 18:46–19:28 local, outside 21:00–01:00 anyway) · 3 expected: predates rule (no expected_minutes; real 42.2 min, purpose ~45 min)
- 4 exit: ok (0) · 5 handoff: ok (wt artifacts/compression/HANDOFF.md at 4d6a768 names 000311 and qat/w4pc-run1 outputs) · 6 pushed: VIOLATION at audit time (main ahead 1: a13bcb1, main-session r01_queue_wait.py commit; pushed with this entry) · wt compression in sync
- 7 docs: pending (QAT result not processed; separate agent processing) · 8 no waiting agent: not verifiable from the record · 9 outside work: ok (nvidia-smi compute list has no queue job; one codex-runtime python.exe, pid 24128, is listed, not a queue job)
- notes: 000311 finished normally (exit 0, 24 steps, 648,441 reviews, peak 23,160 MiB, 0 OOM skips). Stale copy in queue/stale/ and the missing expected_minutes are known context, not flagged. Whether 269 reviews/s counts as a measured speed for a QAT job needs a main-session decision.

## 2026-10-14 19:35 — 000312-eval-speed
- verdict: 3 violations
- 1 lane: VIOLATION (purpose is a parity smoke, fast_evaluation false/true, so no timing flag is needed; but docs/GPU_OPTIMIZATIONS.md on eval-speed e458927 quotes "Final evaluation 232.3 → 37.2 s" as a speed result) · 2 quiet: n/a (not timing; started 19:28 local, outside 21:00–01:00 anyway) · 3 expected: predates rule (no expected_minutes; real 5.5 min vs ~15 min purpose, not an overrun)
- 4 exit: ok (0) · 5 handoff: VIOLATION (no committed HANDOFF.md in wt/eval-speed names 000312) · 6 pushed: VIOLATION (wt eval-speed ahead 1: e458927; main clean) · 7 docs: pending (per main session)
- 8 no waiting agent: not verifiable from the record · 9 outside work: ok now (nvidia-smi lists no compute processes; not checked during the run)
- notes: eval-speed is a parity smoke in the accuracy lane; its logged reviews/s (167k) is incidental. Main-session decision needed on the docs quote and the missing handoff.

## 2026-10-14 20:03 — 000316-accuracy-loop
- verdict: clean
- 1 lane: ok (accuracy job, A15 order-noise screens; no timing flag needed; purpose is not a benchmark) · 2 quiet: n/a (not timing; ran 19:34–20:00 local, outside 21:00–01:00 anyway) · 3 expected: ok (no expected_minutes, predates flag; real 26.6 min vs 2 x 13 = 26 min purpose) · 4 exit: ok (0)
- 5 handoff: ok (wt accuracy-loop artifacts/accuracy/HANDOFF.md at c1010b4 names 000316 and its outputs) · 6 pushed: ok (accuracy-loop and main in sync with origin) · 7 docs: pending (result processed only when runbook finishes) · 8 no waiting agent: not verifiable from the record · 9 outside work: nvidia-smi now lists only desktop processes; not checked during the run
- notes: 000324-accuracy-loop (seed 2027) registered through the queue as designed and is waiting behind 000317, 000321, 000322, 000323. Shared eval is queued by runbook 14b after 000324 finishes, not yet a ticket. Runbook log is in wt artifacts/accuracy/runbook-2026-10-14b-detached.log.

## 2026-10-14 20:09 — 000317-accuracy-loop, 000321-compression, 000322-compression
- verdict: 0 violations (000321/000322 exit 1, handled by re-queue; recurring risk, see notes)
- 000317: 1 lane: ok (accuracy eval; its log prints 197,571 reviews/s, not quoted as speed) · 2 quiet: n/a (not timing; ran 20:00–20:06 local) · 3 expected: ok (no expected_minutes, predates field; real 6.1 min vs ~7 min purpose) · 4 exit: ok (0) · 5 handoff: ok (accuracy-loop artifacts/accuracy/HANDOFF.md at c1010b4 names 000317 and experiments/decay-e10-q25) · 6 pushed: ok (accuracy-loop and main in sync with origin) · 7 docs: n/a (not timing; runbook 14c eval done, collection/report/compare still pending) · 8 no waiting agent: not verifiable from the record (handoff says detached) · 9 outside work: ok (nvidia-smi lists only desktop processes)
- 000321/000322: 1 lane: ok (held-out scoring, no timing flag needed) · 2 quiet: n/a · 3 expected: ok (57 min expected; failed after 0.7 s and 0.8 s) · 4 exit: 1 each, handled (re-queued as 000325/000326 with Git Bash's absolute path, expected 57, both waiting) · 5 handoff: ok (compression HANDOFF names 000321/000322 and logs run3/run4, but not the failure or the re-queue) · 6 pushed: ok (compression in sync) · 7 docs: n/a · 8 no waiting agent: not verifiable from the record · 9 outside work: n/a (not timing)
- notes: bare "bash" resolved to C:\Windows\System32\bash.exe (WSL, no distro); fixed at f046f7b. Proposed rule: queued commands use an absolute interpreter path (000323 already does: C:\Program Files\Git\usr\bin\bash.exe). 000323 (wide-kernels, timing, expected 45 min) started 20:06 local, so its expected end is 20:51 local, before the 21:00 window. The plan complies, but it was still running at 20:09, so the actual finish is unverified. Quote no reviews/s from it if it runs past 21:00. nvidia-smi compute list showed no CUDA process at 20:09, which may be a WDDM listing limit.

## 2026-10-14 20:25 — 000323-wide-kernels
- verdict: 1 violation (possible contamination, item 9)
- 1 lane: ok (timing: true; speed run) · 2 quiet: ok (20:06:46–20:24:55 local, before 21:00) · 3 expected: ok (45 min expected, 18.2 min real) · 4 exit: ok (ticket 0; inside speed.sh, w32 control exit 1, rerun not yet queued) · 5 handoff: ok (wt HANDOFF committed, names 000323 and output paths) · 6 pushed: ok (main and wide-kernels in sync with origin; wt has uncommitted queue-console.log) · 7 docs: pending (W01–W04 still "pending"; summary.md has no reviews/s yet) · 8 no waiting agent: not verifiable from the record · 9 outside work: VIOLATION (possible: accuracy collection decay-e10-q25 --threads 2 ran on CPU 20:06:46–20:15:10 during the timing job, runbook-2026-10-14c-detached.log; nvidia-smi lists only NVIDIA Overlay, no CUDA process)
- notes: HANDOFF expected an overnight run (~01:00), but it started 20:06 local. Proposed rule: CPU-heavy steps of accuracy runbooks should not run while a --timing ticket runs; timing tickets check for heavy processes at start. Control w32 failed, so W04 needs a rerun of w32 before the W04 cells are read.

## 2026-10-14 20:39 — 000324-accuracy-loop
- verdict: 2 violations
- 1 lane: ok (accuracy job, seed 2027 screen training; no timing flag needed) · 2 quiet: n/a (not timing) · 3 expected: VIOLATION (no expected_minutes on record; runbook q() does not pass it, so the default 60 applied; real 13.2 min vs 13 min purpose, no overrun) · 4 exit: ok (0)
- 5 handoff: VIOLATION (wt accuracy-loop HANDOFF at bf4bba0 names 000316/000317 only; 000324 appears only as step 2 of the 000316 row, with no ticket id) · 6 pushed: ok (accuracy-loop and main in sync with origin) · 7 docs: n/a (not timing) · 8 no waiting agent: not verifiable (detached bash -c runbook, no agent listed) · 9 outside work: n/a (not timing); nvidia-smi now lists no CUDA process
- notes: 000328 (fast eval, runbook step 3) is still "waiting" with no expected_minutes, so the default 60 applies against its ~19 min purpose. Runbook q() never passes --expected-minutes, so later tickets default too. HANDOFF "Update" section calls 000324 the -o1/-o2 eval; the -o1/-o2 trainings are 000316, and 000324 is the seed 2027 training.

## 2026-10-14 21:33 — 000325-compression
- verdict: clean
- 1 lane: ok (accuracy: held-out B chunk 3, no timing flag) · 2 quiet: n/a (not timing) · 3 expected: ok (57 min expected, 54.5 min real: 18:38:09–19:32:36 UTC) · 4 exit: ok (0) · 5 handoff: ok (wt HANDOFF in da54177 names 000325, run3 log, heldout-b.jsonl) · 6 pushed: ok (compression and main in sync with origin) · 7 docs: n/a (not timing; final results pending until 000329 is processed) · 8 no waiting agent: not verifiable from the record · 9 outside work: n/a (not timing; nvidia-smi shows no CUDA process, but only after the run)
- notes: record claims only gpu, which fits a GPU scoring job; heldout-b.sh runs a small CPU parity check, which is not CPU-heavy. 000329 (chunk 4, waiting, expected 57) has heartbeat 19:33:36 UTC and expires 19:51:57 UTC; if it has not started by then, it may need re-queuing. 21:00 local quiet window is timing-only, so accuracy work may run; no timing ticket is queued.

## 2026-10-15 00:11 — 000329-compression
- verdict: clean
- 1 lane: ok (accuracy: held-out B chunk 4, claims gpu, no timing flag; handoff quotes no speed result from it) · 2 quiet: n/a (not timing; quiet.json is []) · 3 expected: ok (57 min expected, 49.2 min real: 21:20:53–22:10:04 UTC) · 4 exit: ok (0) · 5 handoff: ok (wt HANDOFF in da54177 names 000329, logs/heldout-b-run4.log, heldout-b.jsonl) · 6 pushed: ok (compression and main in sync with origin; wt has uncommitted heldout-b.jsonl, the approved per-user data commit, not yet made) · 7 docs: n/a (not timing; chunk 4 results pending processing) · 8 no waiting agent: not verifiable from the record · 9 outside work: n/a (not timing; no cpu-heavy steps listed; no nvidia-smi check run; timing ticket 000327 started right after, not a violation for this ticket)
- notes: 000329 replaced cancelled 000326 after 000321/000322 failed on WSL bash (fixed via Git bash). Handoff expects the aggregation step after this ticket, then committing heldout-b.jsonl; the report agent still pending, so item 7 is pending, not violated.

## 2026-10-14 22:38 — 000327-wide-kernels
- verdict: 1 violation
- 1 lane: ok (timing true on record) · 2 quiet: ok (quiet.json is []; ran 00:10–00:36 local) · 3 expected: ok (20 min expected, 26.2 min real, 1.31x: 22:10:04–22:36:16 UTC) · 4 exit: ok (0) · 5 handoff: VIOLATION (wide-kernels HANDOFF at 1a38ef1 gives the rerun command and output paths but never names ticket 000327) · 6 pushed: ok (wide-kernels and main have no ahead commits; wide-kernels has uncommitted rerun output in speed-20261014/all.log, summary.json, summary.md) · 7 docs: pending (W04 cells not yet filled from the clean rerun) · 8 no waiting agent: not verifiable (detached speed.sh, child 26908; no agent listed) · 9 outside work: ok as far as verifiable (no queue ticket overlapped; history has 0 cpu-heavy claims; nvidia-smi now lists only display processes; the 00:15 compress_report_b.py run leaves no timestamped output, so unverified; small-model files written from 00:36, after the job ended)
- notes: Recurring with 000324 and 000328: handoffs name purposes or paths but not ticket ids. Proposed rule for the main session: handoffs must cite each ticket id. small-model accuracy logs carry no timestamps, so their CPU/GPU timing in the window can't be checked. Item 7 stays pending per the main session; the anomalous speed numbers are not judged here.

## 2026-10-15 01:30 — 000330/000331-accuracy-loop
- verdict: clean
- 1 lane: ok (accuracy: A08 decay seeds 2 and 3, no timing flag; handoff quotes no reviews/s from them) · 2 quiet: n/a (not timing; quiet.json is []) · 3 expected: ok (20 min expected; 26.3 and 27.0 min real, 1.32x and 1.35x, under 1.5x) · 4 exit: ok (0 and 0) · 5 handoff: ok (accuracy-loop HANDOFF at b7717a2 names 000330 and 000331 with output paths) · 6 pushed: ok (main and accuracy-loop have no ahead commits) · 7 docs: n/a (not timing) · 8 no waiting agent: not verifiable from the record (plain python command, no agent listed) · 9 outside work: ok (0 cpu-heavy claims in history; no other ticket overlapped 22:36–23:30 UTC; nvidia-smi shows no CUDA process now)
- notes: 000331 started at the second the 000330 child finished (23:02:35 UTC); the queue ran them back to back as intended. Real time 1.3x the 20 min estimate is a minor overrun, not a violation.

## 2026-10-15 01:45 — 000332-small-model
- verdict: clean
- 1 lane: ok (accuracy: screen-w64-s2025 train, claims gpu, no timing flag; handoff quotes no reviews/s) · 2 quiet: n/a (not timing; quiet.json is []) · 3 expected: ok (28 expected, 15.0 min real: 23:29:33–23:44:30 UTC, 0.53x, fast) · 4 exit: ok (0) · 5 handoff: ok (small-model HANDOFF at fddbbce names 000332 in the runbook table with output paths) · 6 pushed: ok (small-model and main have no ahead commits; main has only the excluded GPU_OPTIMIZATIONS_LOG.md and .claude/) · 7 docs: n/a (not timing) · 8 no waiting agent: not verifiable (runbook detached per HANDOFF; no agent listed in record) · 9 outside work: ok (0 cpu-heavy claims; no timing ticket overlapped 23:29–23:44 UTC; nvidia-smi shows only display processes now)
- notes: Runbook steps 2–10 still pending; each ticket gets its own audit. Real time under the 28 min estimate is not a violation.
