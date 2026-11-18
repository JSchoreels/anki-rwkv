"""accuracy_loop epoch_runner hook: constant LR (R01 schedule), then a linear decay to 0 over the last N epochs.

Set on an experiment with `accuracy_loop.py prepare ... --epoch-runner scripts.lr_tail_decay:run_epoch` and the key
"lr_tail_decay": {"epochs": N} in its experiment.json. Epochs 1..E-N run untouched (the accepted constant 7e-4 after
the 5-update warmup). Each decay epoch j = 1..N gets the trainer's own warmup-stable-decay schedule
(scripts/expertium_options.LearningRate, full_run.lr_schedule) with

    peak = 7e-4 (N - j + 1) / N,  final = (N - j) / (N - j + 1),  decay_start 0, decay_end 1, shape linear

so the LR falls linearly within the epoch from 7e-4 (N - j + 1) / N to 7e-4 (N - j) / N, measured in the epoch's
training reviews: one continuous linear ramp 7e-4 -> 0 over the N epochs, continuing from the previous epoch's
weights and AdamW state like every other epoch. N = 1 equals run.json "final_lr_schedule"
{"decay_start": 0, "decay_end": 1, "shape": "linear"}. As in every lr_schedule run, distillation uses
decay_alpha (not alpha) while the schedule is decaying, i.e. during all N decay epochs.
"""
import json
import os
from pathlib import Path

PEAK_LR = 7e-4


def schedule_for(epoch, epochs, decay_epochs):
    """The lr_schedule of `epoch` (1-based) or None while the LR is constant."""
    j = epoch - (epochs - decay_epochs)
    if j < 1:
        return None
    n = decay_epochs
    return {"peak": PEAK_LR * (n - j + 1) / n, "final": (n - j) / (n - j + 1), "decay_start": 0.0, "decay_end": 1.0,
            "shape": "linear"}


def run_epoch(run, epoch, experiment, rft):
    spec = experiment["lr_tail_decay"]
    decay_epochs = int(spec["epochs"])
    if not 1 <= decay_epochs <= experiment["epochs"]:
        raise ValueError(f"lr_tail_decay epochs {decay_epochs} outside 1..{experiment['epochs']}")
    schedule = schedule_for(epoch, experiment["epochs"], decay_epochs)
    if schedule is not None:
        inner = rft.write_protocol  # already wrapped by accuracy_loop (evaluation off, init hook)

        def write_protocol(run_dir, ep, config):
            folder = inner(run_dir, ep, config)
            path = Path(folder) / "protocol.json"
            protocol = json.loads(path.read_text(encoding="utf-8"))
            if protocol["full_run"].get("resumed"):
                # A mid-epoch resume trains only the remaining users, so progress (reviews of this case) would restart.
                raise RuntimeError("lr_tail_decay does not support mid-epoch resume of a decay epoch; rerun the epoch")
            protocol["full_run"]["lr_schedule"] = schedule
            protocol["lr_tail_decay"] = {"decay_epochs": decay_epochs, "decay_epoch": epoch - (experiment["epochs"] - decay_epochs),
                                         "lr_start": schedule["peak"], "lr_end": schedule["peak"] * schedule["final"]}
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(protocol, indent=1) + "\n", encoding="utf-8")
            os.replace(tmp, path)
            return folder

        rft.write_protocol = write_protocol
    return rft.run_epoch(run, epoch)
