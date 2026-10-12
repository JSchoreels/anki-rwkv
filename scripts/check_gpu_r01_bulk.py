"""CUDA check: F20 bulk staging delivers bitwise-identical batches to per-view staging."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Use a new workspace artifact directory")
    cohort = json.loads((ROOT / "artifacts/retraining/gpu-retrain-r01/cohorts-user-holdout-v1.json").read_text(encoding="utf-8"))
    os.environ["GPU_BATCH1_DATA_PATH"] = str(ROOT / cohort["dataset_path"])
    import torch
    from rwkv.data_processing import get_rwkv_data
    from scripts.experiment_provenance import process_provenance
    from scripts.gpu_batch1 import split_frame
    from scripts.gpu_b01 import batch_tensors, prepare_job, transfer_batches
    options = {"prefetch_users": True, "preparation_workers": 8, "gpu_packed_features": True,
               "packed_ipc": True, "async_transfer": True}
    rows = sorted(cohort["stages"]["100"]["rows"], key=lambda row: row["available_reviews"])
    cases = [(rows[-1], 0, 32768), (rows[-1], 32768, 65536), (rows[len(rows) // 2], 0, 32768), (rows[0], 0, rows[0]["available_reviews"])]
    result = {"provenance": process_provenance(), "cases": [], "status": "running",
              "scope": "real packed Expertium batches; per-view versus bulk pinned staging, then GPU feature materialization"}
    try:
        def staged(bulk):
            jobs = []
            for row, offset, end in cases:
                frame = split_frame(get_rwkv_data(Path(os.environ["GPU_BATCH1_DATA_PATH"]), row["user_id"]), row, "train")
                end = min(end, len(frame))
                jobs.append((row, prepare_job(row, offset, end, 2025 + row["user_id"] + offset, options, frame.iloc[offset:end])))
            return [[tensor.cpu() for tensor in batch_tensors(batch)]
                    for _, _, batch in transfer_batches(iter(jobs), torch.bfloat16, True, bulk=bulk)]
        reference, bulk = staged(False), staged(True)
        for (row, offset, end), left, right in zip(cases, reference, bulk):
            exact = len(left) == len(right) and all(
                a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.uint8) if a.dtype != torch.bool else a,
                                                                          b.view(torch.uint8) if b.dtype != torch.bool else b)
                for a, b in zip(left, right))
            result["cases"].append({"user_id": row["user_id"], "offset": offset, "end": end, "tensors": len(left), "bitwise_equal": exact})
        result["status"] = "passed" if result["cases"] and all(case["bitwise_equal"] for case in result["cases"]) else "failed"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        output.mkdir(parents=True)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "cases": result["cases"]}))


if __name__ == "__main__":
    main()
