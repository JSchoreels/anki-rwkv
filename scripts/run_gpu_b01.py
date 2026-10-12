"""User-authorized B01 comparison on the preserved 500-user temporal cohort.

Reuse the recorded training/optimizer/evaluation loop with explicit isolated
staging callbacks. Historical runner source and historical results stay intact.
"""

from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts import run_gpu_batch1 as original

EXECUTION_SCRIPTS = ("scripts/run_gpu_b01.py", "scripts/gpu_b01.py", "scripts/report_gpu_b01.py",
                     "scripts/gpu_resource_monitor.py", "scripts/run_gpu_b01.ps1")


class B01SourceWatch(original.SourceWatch):
    def __init__(self, provenance):
        super().__init__(provenance)
        with ZipFile(ROOT / provenance["source_archive"]) as archive:
            manifest = json.loads(archive.read("source-manifest.json"))["files"]
        for name in EXECUTION_SCRIPTS:
            self.hashes[name] = manifest[name]
        self.implementation_sha256 = hashlib.sha256(json.dumps(self.hashes, sort_keys=True).encode()).hexdigest()
        self.check()


def case(args, protocol):
    expected_workspace = protocol["execution_mode"].get("CUBLAS_WORKSPACE_CONFIG")
    if expected_workspace is not None and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != expected_workspace:
        raise RuntimeError("Fresh comparison process must inherit its frozen CUBLAS workspace setting")
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("Every comparison case requires a fresh CUDA process")
    torch._C._set_graph_executor_optimize(protocol["execution_mode"]["torch_graph_executor_optimize"])
    torch.use_deterministic_algorithms(protocol["execution_mode"]["deterministic_algorithms"], warn_only=False)
    os.environ["GPU_BATCH1_DATA_PATH"] = protocol["dataset_path"]
    from scripts import gpu_batch1, gpu_b01, gpu_power
    from scripts.gpu_resource_monitor import ResourceMonitor
    folder = ROOT / args.output / args.model / args.track
    if (folder / "result.json").exists():
        raise FileExistsError("Preserve an existing case; use a new suite for a changed implementation")
    folder.mkdir(parents=True, exist_ok=True)
    # The shared loop imports these names at case entry. Both controls use the
    # same new staging module with all switches off; B01 uses the frozen options.
    gpu_batch1.TRACKS = {"control": {}, "control-repeat": {}, "B01": protocol["options"]}
    gpu_batch1.training_batches = lambda rows, span, seed, pool, options, cache_paths, depth: gpu_b01.training_batches(rows, span, seed, pool, options, depth)
    gpu_batch1.transfer_batches = gpu_b01.transfer_batches
    original.SourceWatch = B01SourceWatch
    saves = []
    previous_save = torch.save

    def measured_save(value, destination, *values, **kwargs):
        started = time.perf_counter()
        succeeded = False
        try:
            result = previous_save(value, destination, *values, **kwargs)
            succeeded = True
            return result
        finally:
            target = Path(destination) if isinstance(destination, (str, Path)) else None
            if target is not None and target.resolve() == (folder / "master.pth").resolve():
                saves.append({"path": target.relative_to(ROOT).as_posix(), "elapsed_seconds": time.perf_counter() - started,
                              "succeeded": succeeded, "bytes": target.stat().st_size if target.exists() else None})

    with ResourceMonitor(folder) as monitor:
        previous_power = gpu_power.limited_gpu
        gpu_power.limited_gpu = lambda *values: monitor.training_phase(previous_power, *values)
        torch.save = measured_save
        try:
            code = original.run_case(args)
        finally:
            gpu_power.limited_gpu = previous_power
            torch.save = previous_save
    raw = folder / "result.json"
    if raw.exists():
        result = json.loads(raw.read_text())
        result.update(bundle_id="B01", execution_mode=protocol["execution_mode"],
                      actual_execution_mode={"torch_graph_executor_optimize": torch._C._get_graph_executor_optimize(),
                                             "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                             "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
                      checkpoint_saves=saves,
                      selected_components=protocol["selected_components"] if args.track == "B01" else [],
                      protocol_sha256=hashlib.sha256((ROOT / args.output / "protocol.json").read_bytes()).hexdigest(),
                      runtime_substitution="shared run_case loop; B01SourceWatch, gpu_b01 ordered prepare/transfer callbacks, resource monitor; no model/optimizer/evaluation replacements")
        original.atomic_json(raw, result)
    return code


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/benchmarks/gpu-baseline-b01")
    parser.add_argument("--case", action="store_true")
    parser.add_argument("--model", choices=("baseline", "expertium"))
    parser.add_argument("--track", choices=("control", "control-repeat", "B01"))
    parser.add_argument("--expected-source")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Output must remain below workspace artifacts")
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["protocol_status"] != "frozen before comparison" or not protocol["selected_components"]:
        raise ValueError("Freeze the complete requested B01 protocol before execution")
    if args.case:
        return case(args, protocol)
    if args.smoke:
        parser.error("Functional smoke is per-case only and uses a separate artifact directory")
    if (output / "suite.json").exists():
        raise FileExistsError("Preserve an existing suite; no implicit restart")
    if json.loads((ROOT / "artifacts/benchmarks/gpu-batch1-500/process.json").read_text()).get("status") == "running":
        raise RuntimeError("Original batch is still active")
    provenance = process_provenance()
    watch = B01SourceWatch(provenance)
    if watch.implementation_sha256 != protocol["implementation_sha256"]:
        raise ValueError("Implementation changed since the comparison protocol was frozen")
    lock = ROOT / "artifacts/gpu-batch1.lock"
    with lock.open("x") as file:
        json.dump({"pid": os.getpid(), "output": output.relative_to(ROOT).as_posix(), "bundle": "B01"}, file)
    from scripts.report_gpu_b01 import controls_stable, write_report
    suite = {"status": "running", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
             "protocol": protocol, "provenance": provenance, "implementation_sha256": watch.implementation_sha256, "results": []}
    try:
        for model in protocol["models"]:
            # Controls first: establish repeated-control stability before spending
            # the model's B01 candidate run. No warm-start checkpoint reuse.
            for track in ("control", "control-repeat", "B01"):
                # STOP is also consumed inside the shared training loop. This
                # separate marker is controller-only: finish the child normally,
                # preserve evaluation/saves, then stop before the next case.
                if (output / "STOP").exists() or (output / "STOP_AFTER_CASE").exists():
                    suite["status"] = "stopped"
                    suite["stop_reason"] = "STOP_AFTER_CASE" if (output / "STOP_AFTER_CASE").exists() else "STOP"
                    suite["stopped_before_case"] = {"model": model, "track": track}
                    return 1
                watch.check()
                if track == "B01":
                    controls = [r for r in suite["results"] if r["model"] == model and r["track"] in ("control", "control-repeat")]
                    if not controls_stable(controls):
                        suite["status"] = "inconclusive: controls disagree"
                        return 1
                suite["active_case"] = {"model": model, "track": track}
                original.atomic_json(output / "suite.json", suite)
                write_report(output, suite)
                folder = output / model / track
                folder.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--case", "--model", model, "--track", track,
                           "--output", args.output, "--expected-source", watch.implementation_sha256]
                started = time.perf_counter()
                with (folder / "console.log").open("x", encoding="utf-8") as log:
                    status = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
                raw = folder / "result.json"
                result = json.loads(raw.read_text()) if raw.exists() else {"model": model, "track": track, "status": "failed", "error": f"child exit {status} before result creation"}
                result["process_wall_seconds"] = time.perf_counter() - started
                original.atomic_json(raw, result)
                suite["results"].append(result)
                suite.pop("active_case", None)
                original.atomic_json(output / "suite.json", suite)
                write_report(output, suite)
                print(f"{model}/{track}: {result['status']}", flush=True)
                if status or result["status"] != "complete":
                    suite["status"] = "failed"
                    return 1
        suite["status"] = "complete"
        return 0
    finally:
        original.atomic_json(output / "suite.json", suite)
        write_report(output, suite)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
