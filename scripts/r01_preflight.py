"""Preflight R01 runtime once, or run one focused check with bounded logging.

Default preflight is CPU-only. --cuda explicitly probes CUDA and NVRTC without
training. --run accepts a workspace check_*.py script and its arguments; it never
retries, changes privileges, freezes a training protocol, or launches training.
"""
from argparse import ArgumentParser, REMAINDER
from datetime import datetime, timezone
from importlib import metadata
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.r01_workflow import configure_runtime, read_json, run_logged, training_inputs, write_json, StepFailure


def runtime_info(root, main_threads):
    info = configure_runtime(root, main_threads)
    info["toolkit_environment"] = {key: os.environ.get(key) for key in ("CUDA_PATH", "CUDA_HOME", "CUPY_CACHE_DIR", "CUBLAS_WORKSPACE_CONFIG")}
    info["helper_sha256"] = {name: hashlib.sha256((Path(root) / "scripts" / name).read_bytes()).hexdigest()
                             for name in ("r01_preflight.py", "r01_workflow.py")}
    info["versions"] = {}
    for package in ("torch", "cupy-cuda12x", "cupy-cuda13x", "cupy", "numpy", "nvidia-cuda-nvrtc-cu12", "nvidia-cuda-nvrtc-cu13"):
        try:
            info["versions"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5)
        info["gpu_driver"] = driver.stdout.strip() if driver.returncode == 0 else "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        info["gpu_driver"] = "unavailable"
    return info


def preflight(root, output, main_threads=1):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root / "artifacts"):
        raise ValueError("Preflight must remain under workspace artifacts")
    current = runtime_info(root, main_threads)
    if output.exists():
        previous = read_json(output)
        if previous.get("runtime") != current or previous.get("status") != "passed":
            raise ValueError("Preflight environment changed or failed; preserve it and use a new output")
        return previous, True
    record = {"status": "passed", "scope": "CPU configuration and writable temp probe; CUDA untested",
              "recorded_utc": datetime.now(timezone.utc).isoformat(), "runtime": current}
    write_json(output, record)
    return record, False


CUDA_PROBE = """
import json, os, time
from pathlib import Path
import torch
torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
from scripts.experiment_provenance import process_provenance
provenance = process_provenance()
started = time.monotonic()
assert float(torch.ones(1, device='cuda').sum().item()) == 1.0
import cupy as cp
x = cp.zeros(1, dtype=cp.float32)
kernel = cp.RawKernel('extern "C" __global__ void probe(float* x) { x[0] = 1.0f; }', 'probe')
kernel((1,), (1,), (x,))
cp.cuda.get_current_stream().synchronize()
assert float(x.item()) == 1.0
print(json.dumps({'status':'passed', 'scope':'CUDA/NVRTC setup only; no training',
                 'torch':torch.__version__, 'cuda':torch.version.cuda,
                 'threads':torch.get_num_threads(), 'elapsed_seconds':time.monotonic()-started,
                 'cpu_seconds':{'user':os.times().user, 'system':os.times().system},
                 'provenance':provenance}))
"""


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Reusable preflight JSON under artifacts")
    parser.add_argument("--main-threads", type=int, default=1)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--cuda", action="store_true", help="Explicit CUDA/NVRTC setup probe; no training")
    parser.add_argument("--probe-timeout", type=int, default=60)
    parser.add_argument("--repeat-reason", help="Required to repeat an earlier failed CUDA probe")
    parser.add_argument("--run", nargs=REMAINDER, help="Focused scripts/check_*.py and arguments; finish edits first")
    args = parser.parse_args()
    if not 30 <= args.heartbeat_seconds <= 60 or args.probe_timeout < 1:
        parser.error("Use a 30–60 second heartbeat and positive probe timeout")
    output = (ROOT / args.output).resolve()
    record, reused = preflight(ROOT, output, args.main_threads)
    print(f"[preflight] {'reused' if reused else 'passed'}; {args.main_threads} main threads; temp {record['runtime']['temporary_directory']}", flush=True)
    try:
        if args.cuda:
            # Probe reuse requires the same runtime identity; its log retains provenance.
            if record.get("cuda_probe"):
                if record["runtime"].get("gpu_driver") in (None, "unavailable"):
                    raise ValueError("GPU driver identity unavailable; cannot establish cached CUDA probe compatibility")
                log = (ROOT / record["cuda_probe"]).resolve()
                if not log.is_relative_to(ROOT / "artifacts") or not log.is_file() or hashlib.sha256(log.read_bytes()).hexdigest() != record.get("cuda_probe_sha256"):
                    raise ValueError("Cached CUDA probe log changed or is missing; preserve the record and use a new output")
                print(f"[cuda] reuse compatible probe {record['cuda_probe']}", flush=True)
            else:
                if output.with_suffix(".cuda.log").exists() and not (args.repeat_reason or "").strip():
                    raise ValueError("CUDA probe attempt exists; supply --repeat-reason for a repaired failure or changed setup")
                log = run_logged("cuda", [sys.executable, "-c", CUDA_PROBE], output.with_suffix(".cuda.log"),
                                 root=ROOT, heartbeat_seconds=args.heartbeat_seconds, timeout_seconds=args.probe_timeout)
                probe = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
                if probe.get("status") != "passed":
                    raise ValueError("CUDA probe did not pass")
                record.update(cuda_probe=log.relative_to(ROOT).as_posix(), cuda_probe_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                              scope="CPU configuration and CUDA/NVRTC setup; no training")
                write_json(output, record)
                print(f"[cuda] passed; Torch {probe['torch']}, CUDA {probe['cuda']}; log {record['cuda_probe']}", flush=True)
        if args.run:
            script = (ROOT / args.run[0]).resolve()
            if script.parent != ROOT / "scripts" or not script.name.startswith("check_") or script.suffix != ".py":
                raise ValueError("--run accepts only workspace scripts/check_*.py; launch training through its controller")
            result_path = None
            if "--output" in args.run:
                target = (ROOT / args.run[args.run.index("--output") + 1]).resolve()
                if not target.is_relative_to(ROOT / "artifacts"):
                    raise ValueError("Check outputs must stay under workspace artifacts")
                result_path = target
            before = training_inputs(ROOT)
            script_digest = hashlib.sha256(script.read_bytes()).hexdigest()
            log = run_logged("check", [sys.executable, str(script), *args.run[1:]], output.with_suffix(".check.log"),
                             root=ROOT, heartbeat_seconds=args.heartbeat_seconds)
            unchanged = training_inputs(ROOT) == before and hashlib.sha256(script.read_bytes()).hexdigest() == script_digest
            if result_path is not None:
                result_path = result_path if result_path.is_file() else result_path / "result.json"
                if result_path.is_file():
                    write_json(result_path.with_name(result_path.name + ".execution.json"),
                               {"status": "source unchanged" if unchanged else "source changed; incompatible evidence",
                                "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                                "check_script": script.relative_to(ROOT).as_posix(), "check_script_sha256": script_digest,
                                "preflight": output.relative_to(ROOT).as_posix(), "log": log.relative_to(ROOT).as_posix()})
            if not unchanged:
                sidecar = read_json(log.with_suffix(log.suffix + ".run.json"))
                sidecar["status"] = "source changed during check; evidence is incompatible"
                write_json(log.with_suffix(log.suffix + ".run.json"), sidecar)
                raise ValueError("Training source changed during the check; preserve this attempt, finish edits and rerun")
            from scripts.r01_workflow import tail
            print(f"[check] complete; source unchanged; log {log.relative_to(ROOT)}; {tail(log, lines=1, chars=350)}", flush=True)
    except (StepFailure, ValueError) as error:
        print(f"[stop] {error}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        print(f"[stop] {error}", flush=True)
        raise SystemExit(1)
