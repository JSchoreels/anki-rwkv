"""S05 CPU test of the batched scorer's out-of-memory fallback (no GPU): evaluate_rows is patched to raise a CUDA
"out of memory" error for groups above N users; the rows must still match the batch-1 rows within the AGENT.md
tolerance, with the fallback recorded. Also checks is_oom on the exception types seen on the GPU (ticket 000392:
torch.AcceleratorError "CUDA error: out of memory" from torch.gather, which is not a torch.OutOfMemoryError).

usage: check_s05_fallback.py --reference s05-cpu-b1.jsonl --label TAG/VARIANT --model TAG=expertium:ARCH:PATH
       --users-file gate-users.json [--max-reviews 7000] [--workdir DIR]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts import small_model_heldout as sm  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--users-file", required=True)
    parser.add_argument("--max-reviews", default="7000")
    parser.add_argument("--workdir", default=tempfile.gettempdir())
    args = parser.parse_args()

    assert sm.is_oom(torch.AcceleratorError("CUDA error: out of memory"))
    assert sm.is_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert sm.is_oom(torch.OutOfMemoryError("x"))
    assert not sm.is_oom(RuntimeError("device-side assert"))
    assert not sm.is_oom(ValueError("out of memory"))
    groups = sm.length_groups([1, 2, 3, 4], {"1": 100, "2": 100, "3": 100, "4": 600_000}, 8, 800_000)
    assert sorted(map(sorted, groups)) == [[1, 2, 3], [4]], groups

    original = sm.evaluate_rows
    for limit in (2, 0):  # 0: every batched attempt fails, so every user ends on the batch-1 path
        calls = {"oom": 0}

        def patched(model, batch, rows, limit=limit):
            if len(rows) > limit:
                calls["oom"] += 1
                raise torch.AcceleratorError("CUDA error: out of memory")
            return original(model, batch, rows)

        out = Path(args.workdir) / f"s05-fallback-limit{limit}.jsonl"
        out.unlink(missing_ok=True)
        sm.evaluate_rows = patched
        sys.argv = ["small_model_heldout.py", "--device", "cpu", "--users-file", args.users_file, "--max-reviews",
                    args.max_reviews, "--batch-users", "8", "--input-sha", "--model", args.model, "--only",
                    args.label, "--workers", "0", "--out", str(out)]
        try:
            sm.main()
        finally:
            sm.evaluate_rows = original
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        fallbacks = [r for r in rows if r.get("batch_fallback")]
        assert calls["oom"] > 0 and fallbacks, "the simulated OOM never triggered a fallback"
        assert all(r["batch_users"] <= max(limit, 1) for r in rows if "error" not in r), "a group was not split"
        assert not [r for r in rows if "error" in r], [r for r in rows if "error" in r]
        result = out.with_suffix(".compare.json")
        subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts" / "small_model_batch_compare.py"),
                        "--reference", args.reference, "--candidate", str(out), "--label", args.label,
                        "--allow-missing-input-sha", "--out", str(result)], check=True, stdout=subprocess.DEVNULL)
        tolerance = json.loads(result.read_text(encoding="utf-8"))["tolerance"]
        assert tolerance["passes"], tolerance
        print(f"limit {limit}: {len(rows)} rows, {len(fallbacks)} with batch_fallback, batch_users "
              f"{sorted({r['batch_users'] for r in rows})}, {calls['oom']} simulated OOMs, tolerance passes "
              f"(max aggregate {tolerance.get("max_aggregate_abs")})")
    print("S05 fallback test passed")


if __name__ == "__main__":
    main()
