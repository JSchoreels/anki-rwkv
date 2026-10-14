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
