"""CPU-only check of the A08 full-run cool-down option (no GPU, no data): parsing, default, off switch and unchanged main epochs."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_full_training as rft  # noqa: E402


def config(**extra):
    users = list(range(1, 41))
    return {"model": "expertium", "track": "T", "epochs": 3, "split_seed": 7, "dataset_path": "d", "dataset_revision": "r",
            "train_users": users[:20], "validation_users": users[20:30], "final_test_users": users[30:40],
            "quick_validation_users": users[20:23], "review_counts": {str(u): 10 + u for u in users}, **extra}


def main():
    assert rft.FINAL_DECAY_EPOCHS == 0.25
    # CLI default and off switch parse.
    sys.argv = ["x", "latest", "--run", "artifacts/none"]
    import inspect
    assert "--final-decay-epochs" in inspect.getsource(rft.main) and "final_decay_epochs=FINAL_DECAY_EPOCHS" in inspect.getsource(rft.prepare)
    with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as tmp:
        run = Path(tmp) / "run"
        run.mkdir()
        (run / "label-db").mkdir()
        # Main-epoch protocols are identical whether or not the cool-down is configured.
        protocols = {}
        for name, extra in (("absent", {}), ("off", {"final_decay_epochs": 0}), ("on", {"final_decay_epochs": 0.25})):
            cfg = config(**extra)
            for epoch in (1, 3):
                if epoch > 1:
                    # fake previous epoch result so write_protocol can chain
                    prev = rft.epoch_dir(run, epoch - 1) / "expertium" / "T"
                    prev.mkdir(parents=True, exist_ok=True)
                    (prev / "result.json").write_text(json.dumps({"status": "complete", "update_offset_end": 5}))
                folder = rft.write_protocol(run, epoch, cfg)
                protocols[(name, epoch)] = (folder / "protocol.json").read_text()
        for epoch in (1, 3):
            assert protocols[("absent", epoch)] == protocols[("off", epoch)] == protocols[("on", epoch)], epoch
        # Stage is a no-op when off, and launches expertium_decay with the validated recipe when on.
        calls = []
        real = subprocess.call
        subprocess.call = lambda cmd, **kw: calls.append(cmd) or 1
        try:
            assert rft.decay_stage(run, config()) == 0 and rft.decay_stage(run, config(final_decay_epochs=0)) == 0 and not calls
            assert rft.decay_stage(run, config(final_decay_epochs=0.25)) == 1
        finally:
            subprocess.call = real
        cmd = calls[0]
        assert cmd[3].endswith("expertium_decay.py") and cmd[4] == "auto"
        for flag, value in (("--epoch", "3"), ("--fraction", "0.25"), ("--shape", "one_minus_sin"), ("--evaluate", "validation+final_test")):
            assert cmd[cmd.index(flag) + 1] == value, flag
    print("check_full_run_decay: ok")


if __name__ == "__main__":
    main()
