"""F44/F45: compare every sample, packed batch byte and post-preparation RNG."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if output.exists() or not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a fresh artifacts output")
    cohort = json.loads((ROOT / "artifacts/retraining/gpu-retrain-r01/cohorts-user-holdout-v1.json").read_text())
    os.environ["GPU_BATCH1_DATA_PATH"] = str(ROOT / cohort["dataset_path"])
    import torch
    torch.set_num_threads(1)
    from rwkv.data_processing import get_rwkv_data
    from scripts.gpu_batch1 import split_frame
    from scripts.gpu_preprocessing_stage import prepare_job
    from scripts.gpu_r01_samples import samples_for
    from scripts.check_r01_worker_counts import digest
    from scripts.experiment_provenance import process_provenance
    options = json.loads((ROOT / "artifacts/retraining/gpu-retrain-r01/codex-round-0034-0036-handoff.json").read_text())["accepted_options"]
    variants = {"E33": {}, "F44": {"batch_segment_features": True},
                "F45": {"direct_sample_matrix": True},
                "F44-F45": {"batch_segment_features": True, "direct_sample_matrix": True}}
    rows = sorted(cohort["stages"]["100"]["rows"], key=lambda row: row["available_reviews"])
    picks = [(rows[-1], 0, 65536), (rows[-3], 32768, 65536),
             (rows[len(rows)//2], 0, 32769), (rows[0], 0, 127), (rows[10], 0, 129)]
    result = {"status": "running", "provenance": process_provenance(), "windows": [],
              "scope": "five real edge windows including 127/129 and 32769 reviews; every sample field, packed bytes and RNG; E33/F44/F45/combined; CPU one thread"}
    try:
        for row, offset, end in picks:
            frame = split_frame(get_rwkv_data(ROOT / cohort["dataset_path"], row["user_id"]), row, "train")
            end = min(len(frame), end)
            section = frame.iloc[offset:end]
            records = {}
            for name, extra in variants.items():
                active = {**options, **extra}
                started = time.perf_counter()
                samples = samples_for(section, row["user_id"], active)
                seconds = time.perf_counter() - started
                sample_digest = digest(samples)
                del samples
                item = prepare_job(row, offset, end, 2025 + row["user_id"] + offset, active, section)
                records[name] = {"sample_sha256": sample_digest, "batch_sha256": digest(item.batch),
                                 "rng_sha256": digest(torch.random.get_rng_state()), "sample_seconds": seconds}
                del item
            control = records["E33"]
            equal = all(all(value[key] == control[key] for key in ("sample_sha256", "batch_sha256", "rng_sha256")) for value in records.values())
            result["windows"].append({"user_id": row["user_id"], "offset": offset, "reviews": end-offset, "equal": equal, "variants": records})
            print(f"[prepare] {len(result['windows'])}/5 windows; all sample/packed/RNG bytes equal {equal}", flush=True)
        result["status"] = "passed" if all(window["equal"] for window in result["windows"]) else "failed"
        result["summary"] = {"windows": len(result["windows"]), "all_equal": result["status"] == "passed"}
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "summary": result.get("summary"), "error": result.get("error")}))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
