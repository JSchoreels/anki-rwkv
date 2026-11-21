# Time heads for the new candidates (SM119): design

User 2026-11-16: the same answer-time heads as the time heads for the family (SM117), for the new candidates.
Branch `sm119` (cut from `sm117`), worktree `C:/Users/jonat/workspace/wt/sm119`. Code: `scripts/sm117_head.py` (SM117's, with three entries added to its model table: master, pinned sha256, label, recorded rows); the SM117 inputs (`../sm117/plan.json`, `priors.json`, `time-baselines.jsonl`, user lists) are read only. Launcher: `launch-sm119.sh`.

## Models

| model | size, kind | scorer kind | master (read only) |
|---|---|---|---|
| the longer Small Instant-Only model (SM113) | Small, instant-only | `expertium:w64` | `wt/sm113/artifacts/small-model/sm113-inst-e35st-st-dec2ep32k-t99-p2/expertium/E89-F65e-F77a/master.pth` (sha256 `e4574e2a...`), verdict go |
| the longer cool-down for Tiny (SM115) | Tiny, instant-only | `expertium:w64-c1` | `wt/sm115/artifacts/small-model/sm115-c1-e38st-st-dec2ep32k-t99-p2/expertium/E89-F65e-F77a/master.pth` (sha256 `c59d7049...`), verdict go |
| the show-time two-head cool-down for Small (SM118), only if its verdict is go | Small, two-head | `expertium:w64` | `wt/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/E89-F65e-F77a/master.pth` (sha256 read by the launcher once SM118 has ended) |

Small and Tiny models take 474 head inputs (5 snapshots of 64, the instant head's hidden layer of 128, 26 explicit features), as in SM117. The launcher does nothing until the launcher of SM118 has ended (a "done" or "STOP" line in its log), then reads SM118's decision file: verdict go adds SM118's final model to the list, anything else leaves it out. It never touches the sm118 worktree, only reads its log, decision file and (on go) the final master.

## Rules

Exactly the rules of the time heads for the family (SM117) (`../sm117/design.md`): same recipe (MLP -> 128 -> 4 grades x 16 log-time bins, zero-initialised output, trained alone on the frozen model, explicit-only ablation in the same pass, 1,991 training users, first 400,000 reviews each, one pass, AdamW lr 2e-3, batch 8,192, seed 0), show-time inputs (`e2s-query`), same-day reviews included, same gate per head, same S grid (3, 5, 8, 10, 15, 20, 30 s), same baselines, 732 set (722 users), leak-free. Steps per model: smoke, grade-unchanged check (bitwise, release and trunk in separate scorer processes), head training, gate scoring, scoring of P(recall within S seconds). The CPU proof of SM117 (22 checks) is not rerun: only the model table differs.

**Answer time** = front + back together, from the moment the question is shown to the moment a grade button is pressed, capped at the preset's maximum answer seconds. **Recall within S s** = recalled (Hard, Good or Easy) and answered within S s, reading the back included. The moment "Show answer" is pressed is not stored, so front-only time cannot be computed.

Outputs: `<key>-gate.json`, `sm119-result.md`, `sm119-result.json` here; heads in `artifacts/small-model/sm119-<key>-head/`; logs and rows in `gpu/`.
