"""CPU check: F30 create_sample_fast equals create_sample bit for bit on real chunks."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def same(a, b):
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor):
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        bits = {torch.float32: torch.int32, torch.float64: torch.int64, torch.bfloat16: torch.int16}.get(a.dtype)
        return torch.equal(a.view(bits), b.view(bits)) if bits else torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    return a == b


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviews-per-user", type=int, default=4096)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Use a new workspace artifact directory")
    cohort = json.loads((ROOT / "artifacts/retraining/gpu-retrain-r01/cohorts-user-holdout-v1.json").read_text(encoding="utf-8"))
    os.environ["GPU_BATCH1_DATA_PATH"] = str(ROOT / cohort["dataset_path"])
    import torch
    torch.set_num_threads(1)
    from rwkv.data_processing import get_rwkv_data
    from scripts.experiment_provenance import process_provenance
    from scripts.gpu_batch1 import make_samples, split_frame
    from scripts.gpu_r01_samples import make_samples_fast
    result = {"provenance": process_provenance(), "status": "running", "users": [], "mismatches": [],
              "scope": f"all 100-cohort training users, first {args.reviews_per_user} training reviews each, 128-review chunks; every RWKVSample field compared bitwise; one CPU thread"}
    original_seconds = fast_seconds = 0.0
    chunks = 0
    try:
        for row in cohort["stages"]["100"]["rows"]:
            frame = split_frame(get_rwkv_data(Path(os.environ["GPU_BATCH1_DATA_PATH"]), row["user_id"]), row, "train")
            section = frame.iloc[:args.reviews_per_user]
            started = time.perf_counter()
            reference = make_samples(section, row["user_id"], 128)
            original_seconds += time.perf_counter() - started
            started = time.perf_counter()
            candidate = make_samples_fast(section, row["user_id"], 128)
            fast_seconds += time.perf_counter() - started
            chunks += len(reference)
            for index, (a, b) in enumerate(zip(reference, candidate)):
                fields = ["user_id", "start_th", "end_th", "length", "card_features", "global_labels", "review_ths",
                          "label_review_ths", "day_offsets", "day_offsets_first", "skips"]
                bad = [f for f in fields if not same(getattr(a, f), getattr(b, f))]
                bad += [f"ids.{k}" for k in a.ids if not same(a.ids[k], b.ids[k])]
                for k, m in a.modules.items():
                    n = b.modules[k]
                    bad += [f"modules.{k}.{f}" for f in ("split_len", "split_B", "from_perm", "to_perm") if not same(getattr(m, f), getattr(n, f))]
                if bad:
                    result["mismatches"].append({"user_id": row["user_id"], "chunk": index, "fields": bad})
            result["users"].append({"user_id": row["user_id"], "chunks": len(reference), "equal": len(reference) == len(candidate)})
        result["summary"] = {"users": len(result["users"]), "chunks": chunks, "mismatched_chunks": len(result["mismatches"]),
                             "original_seconds": original_seconds, "fast_seconds": fast_seconds,
                             "speedup": original_seconds / fast_seconds if fast_seconds else None}
        result["status"] = "passed" if chunks and not result["mismatches"] and all(u["equal"] for u in result["users"]) else "failed"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        output.mkdir(parents=True)
        (output / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "summary": result.get("summary"), "first_mismatches": result["mismatches"][:3],
                          "error": result.get("error")}))


if __name__ == "__main__":
    main()
