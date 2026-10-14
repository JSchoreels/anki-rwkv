# a08-decay handoff (2026-10-14)

A08 promoted (provisional until A15 order band, ticket 000328): `scripts/run_full_training.py prepare` writes `final_decay_epochs: 0.25` to run.json by default; `run` appends the cool-down (`scripts/expertium_decay.py auto`, output `<run>/decay-final/`) after the last epoch and its validation + final-test scores are the final result. `--final-decay-epochs 0` disables it; old run.json files without the key are unaffected.

- CPU check: `.venv/Scripts/python.exe scripts/check_full_run_decay.py` (main-epoch protocols identical absent/0/0.25; stage no-op when off; launch arguments).
- No GPU smoke queued: the stage reuses `expertium_decay.py auto`, already run on GPU as `decay-8k-e10-q25` (expertium-catch-up). First real exercise is the next `run_full_training.py run`.
- No best-checkpoint registry exists in the repo; the accuracy loop's record is on branch `accuracy-loop` (docs/ACCURACY_IDEAS.md, ACCURACY_LOG.md, artifacts/accuracy/HANDOFF.md).
- If A15 shows the gain inside the order band: run prepare with `--final-decay-epochs 0` or revert the default (`FINAL_DECAY_EPOCHS` in run_full_training.py).
