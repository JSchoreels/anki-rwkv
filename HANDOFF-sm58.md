# HANDOFF SM58: disagreement test, small model (SM36) vs big model (UM+ and slope)

Worktree `C:/Users/jonat/workspace/wt/sm58` (branch `sm58`, from `small-model`). Files in `artifacts/small-model/sm58/` (see its README.md).

## Launch (detached; done by the preparing agent 2026-10-22)
```
cd C:/Users/jonat/workspace/wt/sm58 && nohup bash artifacts/small-model/sm58/launch-sm58.sh > artifacts/small-model/sm58/launch-sm58.log 2>&1 & disown
```
Log: `artifacts/small-model/sm58/launch-sm58.log`. Watch for `[sm58 ...] done` (success) or `STOP: <reason>`.
Dry run: `DRYRUN=1 bash artifacts/small-model/sm58/launch-sm58.sh` (log `dryrun-launch-sm58.log`, ends `DRYRUN OK`).
STOP file `artifacts/small-model/sm58/STOP` makes the launcher exit before enqueuing.

## What the launcher does
1. Waits for `all arms finished` or `STOP:` in `wt/sm59/artifacts/small-model/sm59/launch-sm59.log` (SM59 check D needs ~17 GB RAM).
   If SM59 only logs `deferred:` and exits, the launcher keeps waiting (main session decides).
2. CPU (not in a quiet window): SM36 own-collection predictions (full history, float32, ~2-3 min) -> `collection-cache/` (uncommitted).
3. GPU through the queue (owner `sm58`, claim gpu, cap 0.8, honest minutes; the launcher itself waits out quiet windows):
   | Ticket | Users / reviews | Minutes |
   |---|---|---:|
   | big model (fork) dump, 265 users not in sm24-dump (464 reused) | 265 / 15.2 M | 16 |
   | SM36 dump | 729 / 47.0 M | 19 |
   | SM55 dump (only if `artifacts/small-model/sm58/INCLUDE-SM55` exists when the launcher reaches it; also its collection preds) | 729 / 47.0 M | 19 |
   Up to 2 passes per dump (second pass only for users a time limit left over); users that fail (OOM at 0.8) are listed in `failed-<tag>.json` and skipped.
4. CPU (not in a quiet window): `sm58_umplus.py --models sm36[,sm55]` -> `summary.json`, `summary.md`, `graphs.html`, `um-plus-{benchmark,collection}-<model>-{curve,instant}.png`.

Total GPU nominal: 35 min (54 with SM55). Create `INCLUDE-SM55` any time before the launcher finishes the SM36 dump to include the 8k model.

## After `done` (fresh Sonnet medium agent)
Read `summary.md`; commit (explicit paths, `git add -f`): `summary.json summary.md graphs.html um-plus-*.png users.json heldout-sm58.jsonl README.md`
(plus `failed-*.json`). Do NOT commit `dump/` (per-user npz, large) or `collection-cache/` (per-review rows of the own collection).
Own-collection results in `summary.*` are aggregates only. Report UM+ / slope per model/head/sample, the paired-bootstrap CI of the UM+ difference
(723-ish users), and the users skipped. Reading: lower UM+ = better; tie slope = +0.5 small / -0.5 big.

## Checks done
- `sm58_umplus.py` reproduces `graphs/summary-epoch-01.json` exactly (UM+, slope, LogLoss of A epoch 1 vs big, all 4 head/sample cells).
- Dump LogLoss of a reused big-model dump equals its benchmark row (user 4676: 0.05666).
- `small_model_heldout.py --device cpu` smoke wrote `<user>_sm36_w8+s8.npz`; SM36 own-collection CPU scoring works (127 s).
- Do not edit `scripts/ expertium/ rwkv/` in wt/sm58 while a ticket runs (SourceWatch).
