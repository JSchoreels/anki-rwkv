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
