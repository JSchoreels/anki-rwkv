# Handoff: RWKV Memorised swings replay (2026-10-26)

The CPU replay was stopped at 20:54 for the user's 21:00-02:00 break (CPU quiet). It is resumable: the last checkpoint is after 2026-03-24 (day 795 of 989; query window started 2025-11-25).

- Scratch folder (per-card values, never commit): `C:\Users\jonat\AppData\Local\Temp\claude\C--Users-jonat-workspace-anki-rwkv\41a2947b-8da0-4ee6-a033-bc96eb42a697\scratchpad\mem2`
  - `ckpt.pt`: model states and counters after the last checkpointed day (about 1.2 GB), `replay.npz`: per-card values up to that day (`completed_through`), `replay.log`: progress.
- Resume (CPU, 4 threads, below-normal priority, about 7-10 s per query day; nothing else needed):
  `.venv/Scripts/python.exe artifacts/collection-analysis/rwkv-memorised-swings/replay_memorised.py --scratch <scratch> --query-start 2025-11-25 --ckpt-days 15 --resume`
- Then aggregates and charts (about 15 s): `.venv/Scripts/python.exe artifacts/collection-analysis/rwkv-memorised-swings/analyse_swings.py --scratch <scratch>`; it rewrites `daily.csv`, `buckets.csv`, `logit_shift_daily.csv`, `summary.json` and the two PNGs here, then update the numbers in `README.md`.
- Commit only this folder's scripts, README and aggregate outputs (git add -f), never the scratch files.
- Speed idea if a rerun is needed: for non-reviewed cards the q0/qg/qs passes share the card-module output (same features, same card state), and qg differs from q0 only in the global module, so about half of the query work can be reused.
