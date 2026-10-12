"""Summarize recorded CUDA kernels without launching a competing GPU workload."""

from argparse import ArgumentParser
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance


def summarize(path):
    raw = Path(path).read_bytes()
    trace = json.loads(raw)
    kernels = defaultdict(list)
    events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    for event in events:
        if event.get("cat") == "kernel" and event.get("ph") == "X":
            kernels[event["name"]].append(event)
    total = sum(event.get("dur", 0) for values in kernels.values() for event in values)
    rows = []
    for name, events in kernels.items():
        times = [event.get("dur", 0) for event in events]
        metadata = {}
        for field in ("block", "grid", "registers per thread", "shared memory", "est. achieved occupancy %"):
            values = {json.dumps(event.get("args", {}).get(field)) for event in events}
            metadata[field] = [json.loads(value) for value in sorted(values)]
        rows.append({"name": name, "calls": len(events), "total_ms": sum(times) / 1000,
                     "median_us": statistics.median(times), "kernel_time_fraction": sum(times) / total if total else None,
                     **metadata})
    return {"trace": str(Path(path).relative_to(ROOT)), "trace_sha256": hashlib.sha256(raw).hexdigest(),
            "original_experiment_provenance": "not recorded in this historical trace; analysis provenance is separate",
            "timing_scope": "sum of kernel durations in recorded trace, not end-to-end training time",
            "device_properties": trace.get("deviceProperties", []) if isinstance(trace, dict) else [],
            "total_kernel_ms": total / 1000, "kernels": sorted(rows, key=lambda row: -row["total_ms"])}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmarks/gpu-future/kernel-profile.json")
    args = parser.parse_args()
    provenance = process_provenance()
    report = summarize(args.trace.resolve())
    report["analysis_provenance"] = provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["kernels"]:
        if "wkv" in row["name"]:
            print(json.dumps(row))


if __name__ == "__main__":
    main()
