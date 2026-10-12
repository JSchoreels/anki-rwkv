"""CPU contract checks for ordered, bounded R01 F01 preparation."""

from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import gpu_r01_prefetch as prefetch
from scripts.experiment_provenance import process_provenance
from scripts.run_gpu_batch1 import atomic_json
from scripts.gpu_r01_policy import check_inputs, reusable_check


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat-reason", help="Concrete reason to repeat unchanged scheduling checks")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a fresh artifact directory")
    inputs = check_inputs(ROOT, ("scripts/check_gpu_r01_prefetch.py", "scripts/gpu_r01_prefetch.py",
                                 "scripts/gpu_batch1.py", "scripts/trial_batches.py"))
    previous = reusable_check(ROOT, "R01 ordered prefetch", inputs)
    if previous and not (args.repeat_reason or "").strip():
        print(f"Reuse passed ordered-prefetch checks: {previous.relative_to(ROOT).as_posix()}")
        return
    if output.exists():
        parser.error("Preserve the previous checks; use a fresh artifact directory")
    import pandas as pd
    rows = [{"user_id": user, "available_reviews": length, "train_range": [0, length]}
            for user, length in ((4, 512), (17, 257), (29, 128))]
    loads, checks = [], []
    def frames(selected, prefetch):
        assert prefetch
        for row in selected:
            loads.append(row["user_id"])
            yield row, pd.DataFrame({"review": list(range(row["available_reviews"]))})
    prefetch.frame_stream = frames
    prefetch.split_frame = lambda frame, row, split: frame

    class Pool:
        outstanding = 0
        maximum = 0
        cancellations = 0
        def submit(self, function, *args):
            self.outstanding += 1
            self.maximum = max(self.maximum, self.outstanding)
            owner = self
            class Future:
                def result(self):
                    owner.outstanding -= 1
                    row, offset, end, seed, _, section, _ = args
                    return SimpleNamespace(offset=offset, seed=seed, reviews=end-offset,
                                           inputs=section["review"].tolist())
                def cancel(self):
                    owner.outstanding -= 1
                    owner.cancellations += 1
            return Future()

    pool = Pool()
    actual = [(row["user_id"], item.offset, item.seed, item.inputs)
              for row, item in prefetch.training_batches(rows, 256, 2025, pool, {"prefetch_users": True}, depth=3)]
    expected = [(row["user_id"], offset, 2025 + row["user_id"] + offset,
                 list(range(offset, min(row["available_reviews"], offset+256))))
                for row in rows for offset in range(0, row["available_reviews"], 256)]
    assert actual == expected, "Prefetch changed input coverage, update order or seeds"
    checks.append("all inputs, offsets and seeds match the serial schedule across user boundaries")
    assert pool.maximum == 3 and pool.outstanding == 0
    checks.append("queued preparations are bounded and fully drained")
    assert loads == [4, 17, 29]
    checks.append("each history loaded once, in the declared order")
    pool = Pool()
    iterator = prefetch.training_batches(rows, 256, 2025, pool, {"prefetch_users": True}, depth=3)
    next(iterator)
    iterator.close()
    assert pool.cancellations == 3 and pool.outstanding == 0
    checks.append("early consumer shutdown cancels every queued preparation")
    try:
        next(prefetch.training_batches(rows, 256, 2025, Pool(), {"prefetch_users": True, "sample_cache": True}))
    except ValueError:
        checks.append("unsupported optimization combinations rejected")
    else:
        raise AssertionError("Unexpected options accepted")
    output.mkdir(parents=True)
    atomic_json(output / "result.json", {"status": "passed", "checks": checks, "provenance": process_provenance(),
                                         "check_suite": "R01 ordered prefetch", "check_inputs": inputs, "repeat_reason": args.repeat_reason,
                                         "scope": "CPU scheduling contracts; full-run numerical parity is measured separately"})
    print(f"Passed {len(checks)} R01 F01 contract checks")


if __name__ == "__main__":
    main()
