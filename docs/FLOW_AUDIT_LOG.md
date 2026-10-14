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
