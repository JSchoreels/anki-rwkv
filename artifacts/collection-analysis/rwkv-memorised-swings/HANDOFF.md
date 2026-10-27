# Handoff: RWKV Memorised swings replay (2026-10-26, completed 2026-10-27)

Done. The CPU replay stopped on 2026-10-26 at 20:54 for the break, resumed from its checkpoint on 2026-10-27 00:25 and finished at 00:59 (day 989/989, last review in the copy 2026-10-04, exit 0). `analyse_swings.py` rewrote the aggregates and charts and README.md has the full-period numbers.

- Scratch folder (per-card values, never commit): `C:\Users\jonat\AppData\Local\Temp\claude\C--Users-jonat-workspace-anki-rwkv\41a2947b-8da0-4ee6-a033-bc96eb42a697\scratchpad\mem2` (`ckpt.pt`, `replay.npz`, `replay.log`).
- Re-run the aggregates only (about 15 s): `.venv/Scripts/python.exe artifacts/collection-analysis/rwkv-memorised-swings/analyse_swings.py --scratch <scratch>`.
- Full rerun from scratch: `replay_memorised.py --scratch <scratch> --query-start 2025-11-25 --ckpt-days 15` (about 1 h CPU in two parts). Speed idea: for non-reviewed cards the q0/qg/qs passes share the card-module output, and qg differs from q0 only in the global module, so about half of the query work can be reused.
- Commit only this folder's scripts, README and aggregate outputs (git add -f), never the scratch files.
