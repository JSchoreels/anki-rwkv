"""Apply live R01 guards, reuse evidence, then run isolated source if needed."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.gpu_r01_policy import launch_plan
from scripts.run_gpu_batch1 import atomic_json


def launch_case(args, output, *, root=ROOT, runtime=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    decision = launch_plan(root, output, args)
    if decision["action"] == "reuse":
        records_path = output / "reused-results.json"
        records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.exists() else {}
        records.setdefault(args.model, {})[args.track + (" smoke" if args.smoke else "")] = decision
        atomic_json(records_path, records)
        print(json.dumps(decision))
        return 0
    for path in (root / "artifacts/gpu-r01.lock", root / "artifacts/gpu-batch1.lock"):
        if path.exists() and psutil.pid_exists(json.loads(path.read_text())["pid"]):
            raise RuntimeError(f"GPU experiment is already active: {path}")
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if runtime is None:
        origin = protocol["preparation_provenance"]
        archive = root / origin["source_archive"]
        if not archive.is_relative_to(root / "artifacts"):
            raise ValueError("Frozen source archive must remain under workspace artifacts")
        runtime = root / "artifacts/retraining/gpu-retrain-r01" / ("runtime-" + protocol["iteration"] + "-" + hashlib.sha256(archive.read_bytes()).hexdigest()[:12])
        if not runtime.exists():
            from scripts import isolate_gpu_r01
            if root != isolate_gpu_r01.ROOT:
                raise ValueError("Automatic runtime creation requires the controller workspace")
            isolate_gpu_r01.build(archive, runtime, protocol["iteration"])
    else:
        runtime = (root / runtime).resolve()
    if not runtime.is_relative_to(root / "artifacts") or not (runtime / "isolation.json").exists():
        raise ValueError("Use a recorded isolated runtime under workspace artifacts")
    relative = output.relative_to(root)
    isolated_output = runtime / relative
    # Synchronize only immutable inputs needed for this request. Never overwrite results.
    for name in ("protocol.json", "selection.json", "reference-reuse-amendment.json", "loop-definition.md", "loop-definition-update-001.md"):
        source = output / name
        if source.exists():
            target = isolated_output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError(f"Frozen runtime input differs: {name}; use a new runtime")
            if not target.exists():
                target.write_bytes(source.read_bytes())
    decision = launch_plan(root, output, args)  # Recheck after isolation/preparation work.
    if decision["action"] != "run":
        raise RuntimeError("Recorded work changed during preparation; reuse it before launching")
    lock = root / "artifacts/gpu-r01.lock"
    for path in (lock, root / "artifacts/gpu-batch1.lock"):
        if path.exists() and psutil.pid_exists(json.loads(path.read_text())["pid"]):
            raise RuntimeError(f"GPU experiment is already active: {path}")
    if lock.exists():
        lock.unlink()  # Stale PID checked above; remove only this lock file.
    with lock.open("x", encoding="utf-8") as file:
        json.dump({"pid": os.getpid(), "isolated_runtime": runtime.relative_to(root).as_posix()}, file)
    ticket = output / "launches" / (uuid4().hex + ".json")
    decision.update(recorded_utc=datetime.now(timezone.utc).isoformat(), output=relative.as_posix(),
                    runtime=runtime.relative_to(root).as_posix(), controller_root=str(root))
    try:
        atomic_json(ticket, decision)
        temporary = runtime / "artifacts/tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, TEMP=str(temporary), TMP=str(temporary),
                           GPU_R01_LAUNCH_TICKET=str(ticket), GPU_R01_CONTROLLER_ROOT=str(root),
                           GPU_R01_SHARED_DATA_PATH=str(root / protocol["dataset_path"]),
                           GPU_R01_SHARED_CACHE_PATH=str(root / "artifacts/cache/r01-preprocessing-v1"))
        command = [sys.executable, str(runtime / "scripts/run_gpu_r01.py"), "--case", "--output", relative.as_posix(),
                   "--model", args.model, "--track", args.track]
        if args.smoke:
            command.append("--smoke")
        code = subprocess.run(command, cwd=runtime, env=environment).returncode
        case_result = isolated_output / ("functional-smoke-v1" if args.smoke else "") / args.model / args.track / "result.json"
        if code == 0 and not case_result.exists():
            raise RuntimeError("Isolated worker exited without recording its result")
        from scripts import isolate_gpu_r01
        if root != isolate_gpu_r01.ROOT:
            raise ValueError("Export requires the controller workspace")
        isolate_gpu_r01.export(runtime, protocol["iteration"])
        return code
    finally:
        if lock.exists() and json.loads(lock.read_text()).get("pid") == os.getpid():
            lock.unlink()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, help='Optional existing frozen runtime; otherwise isolate automatically')
    parser.add_argument('--case', action='store_true', help='Compatibility with the original isolated launcher')
    parser.add_argument('--output', type=Path, default=Path('artifacts/retraining/gpu-retrain-r01/iteration-0002'))
    parser.add_argument('--model', choices=('baseline', 'expertium'), required=True)
    parser.add_argument('--track', default='control')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--repeat-reason', help='Concrete incompatibility or repaired failure requiring a repeat')
    parser.add_argument('--check-reason', help='Changed behaviour this focused smoke verifies')
    args = parser.parse_args()
    return launch_case(args, ROOT / args.output, runtime=args.runtime)


if __name__ == '__main__':
    raise SystemExit(main())
