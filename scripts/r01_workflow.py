"""Small, CPU-only I/O and process helpers for R01 orchestration."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from uuid import uuid4


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def tail(path, lines=3, chars=600):
    """Read only the end of a log, even when it contains enormous lines."""
    with Path(path).open("rb") as stream:
        stream.seek(max(0, Path(path).stat().st_size - 4096))
        text = stream.read(4096).decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])[-chars:]


def configure_runtime(root, main_threads=1):
    """Choose writable compiler temp storage before importing Torch or CuPy."""
    if type(main_threads) is not int or main_threads < 1:
        raise ValueError("main_threads must be a positive integer")
    root = Path(root).resolve()
    temporary = (root / "artifacts/tmp/r01").resolve()
    if not temporary.is_relative_to(root / "artifacts"):
        raise ValueError("R01 temporary storage must stay under workspace artifacts")
    temporary.mkdir(parents=True, exist_ok=True)
    cache = (root / "artifacts/cache/cupy-cache").resolve()
    if not cache.is_relative_to(root / "artifacts"):
        raise ValueError("CuPy cache must stay under workspace artifacts")
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=temporary) as probe:
        probe.write(b"R01 writable temp probe")
        probe.flush()
    os.environ.update(TEMP=str(temporary), TMP=str(temporary),
                      CUPY_CACHE_DIR=str(cache), OMP_NUM_THREADS=str(main_threads), MKL_NUM_THREADS=str(main_threads), PYTHONUTF8="1")
    tempfile.tempdir = str(temporary)
    return {"temporary_directory": str(temporary), "main_threads": main_threads,
            "cupy_cache_directory": str(cache), "python_executable": sys.executable, "python_prefix": sys.prefix}


class StepFailure(RuntimeError):
    def __init__(self, step, code, log):
        self.returncode, self.log = code, Path(log)
        super().__init__(f"{step} failed (exit {code}); log {log}; tail: " + tail(log).replace("\n", " | "))


def stop_process(process):
    """Stop only this helper's owned subprocess and its preparation children."""
    try:
        import psutil
    except ImportError:
        process.kill()
    else:
        try:
            children = psutil.Process(process.pid).children(recursive=True)
        except psutil.Error:
            children = []
        process.kill()
        for child in children:
            try:
                child.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(children, timeout=2)
    process.wait()


def run_logged(step, command, log, *, root, heartbeat_seconds=60, timeout_seconds=None):
    """Wait inside the helper; emit at most one status line per heartbeat.

    No retries or privilege changes. Keep earlier logs on resume, and record the
    command, timings and exit code beside the actual log used for this attempt.
    """
    if heartbeat_seconds < 30 or heartbeat_seconds > 60:
        raise ValueError("Use a 30–60 second heartbeat; default 60")
    root, log = Path(root).resolve(), Path(log)
    if not log.is_absolute():
        log = root / log
    log = log.resolve()
    if not log.is_relative_to(root / "artifacts"):
        raise ValueError("Logs must remain under workspace artifacts")
    log.parent.mkdir(parents=True, exist_ok=True)
    requested = log
    attempt = 1
    while True:
        try:
            handle = log.open("x", encoding="utf-8", newline="\n")
            break
        except FileExistsError:
            attempt += 1
            log = requested.with_name(f"{requested.stem}-attempt-{attempt:04d}{requested.suffix}")
    started = time.monotonic()
    record = {"step": step, "command": [str(arg) for arg in command], "log": log.relative_to(root).as_posix(),
              "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "attempt": attempt}
    record_path = log.with_suffix(log.suffix + ".run.json")
    process = None
    try:
        with handle:
            process = subprocess.Popen(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT)
            record["pid"] = process.pid
            write_json(record_path, record)
            while True:
                elapsed = time.monotonic() - started
                remaining = None if timeout_seconds is None else timeout_seconds - elapsed
                if remaining is not None and remaining <= 0:
                    stop_process(process)
                    record["status"] = "timed out; no automatic retry"
                    raise StepFailure(step, process.returncode, log)
                try:
                    code = process.wait(timeout=heartbeat_seconds if remaining is None else min(heartbeat_seconds, remaining))
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - started
                    record["elapsed_seconds"] = elapsed
                    try:
                        import psutil
                    except ImportError:
                        pass
                    else:
                        try:
                            cpu = psutil.Process(process.pid).cpu_times()
                            record["cpu_seconds"] = {"user": cpu.user, "system": cpu.system}
                        except (psutil.Error, OSError):
                            pass
                    write_json(record_path, record)
                    print(f"[{step}] running {elapsed:.0f}s; log {record['log']} (resource counters in sidecar)", flush=True)
            record.update(returncode=code, status="complete" if code == 0 else "failed; no automatic retry")
        if code:
            raise StepFailure(step, code, log)
        return log
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        if record["status"] == "running":
            record["status"] = "failed or interrupted; no automatic retry"
        raise
    finally:
        if process is not None and process.poll() is None:
            stop_process(process)
            record["status"] = "interrupted; no automatic retry"
        record["elapsed_seconds"] = time.monotonic() - started
        write_json(record_path, record)


def training_inputs(root):
    """Same numerical/source scope as the R01 watches, without importing Torch."""
    import ast
    root = Path(root)
    names = {str(path.relative_to(root).as_posix()) for folder in ("rwkv", "expertium")
              for path in (root / folder).rglob("*") if path.is_file() and path.suffix in (".py", ".cu", ".cpp", ".h")}
    for filename, class_name in (("scripts/run_gpu_batch1.py", "SourceWatch"), ("scripts/run_gpu_r01.py", "R01SourceWatch")):
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        for node in ast.walk(cls):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                name = node.value
                if Path(name).suffix in (".py", ".cu", ".cpp", ".h", ".toml", ".lock"):
                    names.add(name if "/" in name or (root / name).is_file() else "scripts/" + name)
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sorted(names)}


def check_evidence(root, path):
    """Check success and source compatibility; never waive runtime or parity gates."""
    from zipfile import ZipFile
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root / "artifacts"):
        raise ValueError("Check evidence must be under workspace artifacts")
    result = read_json(path)
    passed = result.get("status") == "passed" or (isinstance(result.get("passed"), dict)
                                                and bool(result["passed"]) and all(result["passed"].values()))
    if not passed:
        raise ValueError(f"Check did not pass: {path}")
    execution_path = path.with_name(path.name + ".execution.json")
    if execution_path.exists():
        execution = read_json(execution_path)
        if execution["status"] != "source unchanged":
            raise ValueError("Source changed during recorded check; evidence is incompatible")
        if hashlib.sha256(path.read_bytes()).hexdigest() != execution["result_sha256"]:
            raise ValueError("Check result changed since its recorded execution")
        script = (root / execution["check_script"]).resolve()
        if not script.is_relative_to(root / "scripts") or hashlib.sha256(script.read_bytes()).hexdigest() != execution["check_script_sha256"]:
            raise ValueError("Checker changed since recorded execution; rerun its affected checks")
    if result.get("torch"):
        from importlib.metadata import version
        if result["torch"] != version("torch"):
            raise ValueError("Torch build differs from recorded check; rerun its affected kernel checks")
    archive = (root / result["provenance"]["source_archive"]).resolve()
    if not archive.is_relative_to(root / "artifacts"):
        raise ValueError("Check source archive must remain under workspace artifacts")
    with ZipFile(archive) as source:
        manifest = json.loads(source.read("source-manifest.json"))["files"]
    changed = [name for name, digest in training_inputs(root).items() if manifest.get(name) != digest]
    if changed:
        raise ValueError("Training source changed since check; finish edits and rerun the affected check: " + ", ".join(changed[:3]))
    return {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "status": "passed, source compatible; runtime compatibility must also be reviewed"}
