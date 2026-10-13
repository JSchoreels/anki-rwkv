"""Content-addressed index of recorded R01 runs and checks, consulted before repeating identical work (F74 T3).

Human request 2026-10-13: "make sure iteration can be as short as possible tho, avoiding multiple time the
same work, for example through storing prior result to reuse later". Each entry lives in
`result-store/<kind>/<key[:2]>/<key>/<value sha256[:16]>.json`; the key hashes everything that decides the
work (implementation sha256, options, cohort/protocol context, seed, smoke flag and the runtime manifest)
and the value records the result path and its sha256, hashes, scores and timing. Writes are one new file per
record, so parallel sessions never rewrite each other's entries.

Reuse fails closed: a lookup returns a record only when the result file still has its recorded sha256, the
work completed or passed, every key field is present, and the record carries a complete runtime manifest
equal to the current one. Historical records lacking a driver/NVRTC manifest are indexed but never reused.
The caller must still apply its own gate (`matching_case` for cases, the in-process F67 proof for
evaluation reuse) and label the reuse in its report.

  r01_result_store.py backfill                       index every recorded case and check (idempotent)
  r01_result_store.py record --result <result.json>  index one case or check result
  r01_result_store.py lookup-case --protocol <protocol.json> --track T [--smoke] [--options JSON]
  r01_result_store.py lookup-check -- scripts/check_x.py <arguments>
  r01_result_store.py verify | stats
"""
from argparse import ArgumentParser, REMAINDER
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.r01_workflow import read_json, write_json  # noqa: E402
from scripts.r01_evaluation_reuse import complete_runtime, runtime_manifest  # noqa: E402

R = Path("artifacts/retraining/gpu-retrain-r01")
STORE = R / "result-store"
# Protocol fields that decide training exposure and evaluation, as in gpu_r01_policy.matching_case.
PROTOCOL_CONTEXT = ("cohort_sha256", "dataset_revision", "dataset_path", "seeds", "training", "training_reviews",
                    "validation_reviews", "rows", "evaluation_rows", "execution_mode")
TIMING = ("training_seconds", "reviews_per_second", "validation_seconds", "discarded_primer_seconds",
          "model_setup_seconds", "process_wall_seconds", "case_elapsed_seconds")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path, root=ROOT):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def case_key(protocol, model, options, smoke, runtime):
    """Everything that decides a run, computable before launch from the frozen protocol."""
    return {"kind": "case", "model": model, "options": options, "smoke": bool(smoke),
            "implementation_sha256": protocol.get("implementation_sha256"),
            "seed": options.get("training_seed", (protocol.get("seeds") or [None])[0]),
            "protocol_context_sha256": digest({key: protocol.get(key) for key in PROTOCOL_CONTEXT}),
            "cohort_sha256": protocol.get("cohort_sha256"), "runtime": runtime}


def check_key(root, script, arguments, inputs, runtime):
    """A check repeats identical work when its script, arguments, training inputs and runtime match."""
    return {"kind": "check", "script": script, "script_sha256": file_sha256(Path(root) / script),
            "arguments": normalized_arguments(root, arguments), "training_inputs_sha256": digest(inputs) if inputs else None,
            "runtime": runtime}


def normalized_arguments(root, arguments, *, file_contents=True):
    """Drop the fresh --output directory; file arguments (plans, option files) count by content."""
    normalized, skip = [], False
    for argument in arguments:
        if skip:
            skip = False
            continue
        if argument == "--output":
            normalized.append("--output <fresh directory>")
            skip = True
            continue
        path = Path(root) / argument
        if not argument.startswith("-") and path.is_file():
            # A backfilled check cannot prove the file content it read, so its key stays unreachable.
            argument += "#" + (file_sha256(path) if file_contents else "content at check time unknown")
        normalized.append(argument)
    return normalized


def entry_folder(root, key):
    name = digest(key)
    return Path(root) / STORE / key["kind"] / name[:2] / name, name


def case_record(root, path):
    """Index value for one recorded case result; None for unreadable or non-case JSON."""
    root, path = Path(root).resolve(), Path(path).resolve()
    try:
        case = read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(case, dict) or "options" not in case or "model" not in case:
        return None
    protocol_path = path.parents[2] / "protocol.json"
    if not protocol_path.exists():
        return None
    protocol = read_json(protocol_path)
    context = case.get("evaluation_context") or {}
    runtime = context.get("runtime") if isinstance(context.get("runtime"), dict) else None
    key = case_key(protocol, case["model"], case["options"], case.get("smoke_only"), runtime)
    key["implementation_sha256"] = case.get("implementation_sha256")  # what actually ran
    reuse = case.get("evaluation_reuse") or {}
    value = {"result": relative(path, root), "result_sha256": file_sha256(path), "protocol": relative(protocol_path, root),
             "protocol_sha256": file_sha256(protocol_path), "track": case.get("track"), "status": case.get("status"),
             "smoke": bool(case.get("smoke_only")), "config": case.get("config"),
             "trained_parameter_sha256": case.get("trained_parameter_sha256"),
             "initial_parameter_sha256": case.get("initial_parameter_sha256"),
             "validation_input_sha256": case.get("validation_input_sha256"),
             "evaluation": case.get("evaluation"), "checkpoint": (path.parent / "master.pth").is_file(),
             "evaluation_context": bool(context), "evaluation_reused": reuse.get("status") == "reused",
             "evaluation_origin": (reuse.get("loss_provenance") or {}).get("origin_result"),
             "timing": {name: case.get(name) for name in TIMING},
             "git_commit": (case.get("provenance") or {}).get("git_commit"), "runtime_complete": complete_runtime(runtime)}
    return key, value


def check_record(root, path):
    """Index value for one recorded check result with an execution sidecar from r01_preflight --run."""
    from zipfile import ZipFile
    from scripts.r01_workflow import training_inputs
    root, path = Path(root).resolve(), Path(path).resolve()
    execution_path = path.with_name(path.name + ".execution.json")
    if not execution_path.exists():
        return None
    execution, result = read_json(execution_path), read_json(path)
    run_path = (root / execution["log"]).with_name(Path(execution["log"]).name + ".run.json")
    if not run_path.exists():
        return None
    command = read_json(run_path)["command"]
    script = execution["check_script"]
    arguments = command[2:] if len(command) > 2 else []
    runtime, inputs = None, None
    preflight = root / execution.get("preflight", "")
    if preflight.is_file():
        recorded = read_json(preflight).get("runtime", {})
        runtime = runtime_from_preflight(recorded)
    archive = (result.get("provenance") or {}).get("source_archive")
    if archive and (root / archive).is_file():
        with ZipFile(root / archive) as source:
            manifest = json.loads(source.read("source-manifest.json"))["files"]
        inputs = {name: manifest.get(name) for name in training_inputs(root)}
    # Live checks record their key when they run (r01_preflight --run); backfilled keys hash what is provable.
    key = execution.get("store_key") or {"kind": "check", "script": script, "script_sha256": execution["check_script_sha256"],
                                         "arguments": normalized_arguments(root, arguments, file_contents=False),
                                         "training_inputs_sha256": digest(inputs) if inputs else None, "runtime": runtime}
    passed = result.get("status") == "passed" or (isinstance(result.get("passed"), dict) and bool(result["passed"])
                                                 and all(result["passed"].values()))
    value = {"result": relative(path, root), "result_sha256": file_sha256(path), "status": "passed" if passed else "failed",
             "execution": relative(execution_path, root), "execution_status": execution["status"], "log": execution["log"],
             "elapsed_seconds": read_json(run_path).get("elapsed_seconds"), "runtime_complete": complete_runtime(runtime)}
    return key, value


def runtime_from_preflight(info):
    """Map the preflight runtime record (versions, nvidia-smi driver) onto the manifest fields."""
    versions = info.get("versions") or {}
    driver, _, gpu = (info.get("gpu_driver") or "").partition(",")
    toolkit = info.get("toolkit_environment") or {}
    return {"torch": versions.get("torch"), "numpy": versions.get("numpy"),
            "cupy": next((versions[name] for name in ("cupy-cuda13x", "cupy-cuda12x", "cupy") if name in versions), None),
            "nvrtc_package": next((versions[name] for name in ("nvidia-cuda-nvrtc-cu13", "nvidia-cuda-nvrtc-cu12") if name in versions), "none"),
            "cuda_path": toolkit.get("CUDA_PATH") or toolkit.get("CUDA_HOME") or "none",
            "driver_version": driver.strip() or None, "gpu_name": gpu.strip() or None}


def add(root, key, value):
    folder, name = entry_folder(root, key)
    target = folder / (value["result_sha256"][:16] + ".json")
    if target.exists():
        return target, False
    write_json(target, {"key": key, "key_sha256": name, "value": value, "indexed_utc": datetime.now(timezone.utc).isoformat()})
    return target, True


def record(root, path):
    path = Path(path)
    made = case_record(root, path) if path.name == "result.json" and not (path.with_name(path.name + ".execution.json")).exists() else None
    made = made or check_record(root, path)
    if made is None:
        raise ValueError(f"Not an indexable case or check result: {path}")
    return add(root, *made)


def recorded_case_paths(root):
    base = Path(root) / R
    for output in sorted(base.glob("iteration-*")):
        for pattern in ("*/*/result.json", "replicates/*/*/*/result.json", "functional-smoke-*/*/*/result.json"):
            yield from sorted(path for path in output.glob(pattern) if path.parts[-3] in ("baseline", "expertium"))


def backfill(root):
    counts = {"case": 0, "check": 0, "new": 0, "skipped": 0}
    for path in recorded_case_paths(root):
        made = case_record(root, path)
        if made is None:
            counts["skipped"] += 1
            continue
        counts["case"] += 1
        counts["new"] += add(root, *made)[1]
    for execution in sorted((Path(root) / "artifacts").rglob("*.execution.json")):
        made = check_record(root, execution.with_name(execution.name.removesuffix(".execution.json")))
        if made is None:
            counts["skipped"] += 1
            continue
        counts["check"] += 1
        counts["new"] += add(root, *made)[1]
    return counts


def entries(root, kind=None):
    for path in sorted((Path(root) / STORE).glob(f"{kind or '*'}/*/*/*.json")):
        yield path, read_json(path)


def valid(root, entry):
    """Reasons a stored record may not be reused now; empty means reusable."""
    value, reasons = entry["value"], []
    path = Path(root) / value["result"]
    if not path.is_file():
        return ["result file missing"]
    if file_sha256(path) != value["result_sha256"]:
        reasons.append("result file changed since it was indexed")
    if value["status"] not in ("complete", "passed"):
        reasons.append(f"status {value['status']}")
    if not value.get("runtime_complete"):
        reasons.append("record lacks a complete runtime manifest (driver/NVRTC)")
    if entry["key"]["kind"] == "check" and value.get("execution_status") != "source unchanged":
        reasons.append("source changed during the recorded check")
    if any(value is None for field, value in entry["key"].items() if field != "runtime"):  # runtime: manifest reason above
        reasons.append("key field missing")
    return reasons


def lookup(root, key, *, require=None):
    """Return (reusable entries, refusals); every candidate is rechecked, none is trusted blindly."""
    folder, _ = entry_folder(root, key)
    found, refused = [], []
    if not complete_runtime(key.get("runtime")):
        return [], [{"reason": "current runtime manifest incomplete; reuse refused"}]
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        entry = read_json(path)
        if entry["key"] != key:
            refused.append({"entry": relative(path, root), "reasons": ["key collision or tampered key"]})
            continue
        reasons = valid(root, entry) + [reason for reason in (require(entry) if require else [])]
        (refused if reasons else found).append({"entry": relative(path, root), "reasons": reasons} if reasons else entry)
    return found, refused


def lookup_case(root, protocol_path, model, track, smoke=False, options=None, runtime=None):
    """Recorded identical case for a frozen protocol; verified again with the live matching_case guard."""
    from scripts.gpu_r01_policy import matching_case
    protocol_path = Path(protocol_path)
    protocol = read_json(protocol_path)
    options = options if options is not None else protocol["case_options_by_model"][model][track]
    key = case_key(protocol, model, options, smoke, runtime or runtime_manifest())

    def require(entry):
        case = read_json(Path(root) / entry["value"]["result"])
        recorded = read_json(Path(root) / entry["value"]["protocol"])
        recorded["_file_sha256"] = file_sha256(Path(root) / entry["value"]["protocol"])
        return [] if matching_case(case, recorded, protocol, model, options, smoke=smoke, source=True) else ["matching_case refused"]
    return lookup(root, key, require=require)


def evaluation_origin(root, protocol_path, model, track, *, iteration_dir=None, runtime=None, smoke=False):
    """An actually evaluated full run that an exact replica may reuse (T1); the training process re-proves it."""
    def evaluated(entry):
        value = entry["value"]
        checkpoint = (Path(root) / value["result"]).parent / "master.pth"  # checked live: files can disappear
        reasons = [] if checkpoint.is_file() and value["evaluation_context"] and not value["evaluation_reused"] \
            and value["evaluation"] and value["smoke"] is bool(smoke) else ["origin lacks checkpoint, manifest or its own evaluation"]
        return reasons
    found, refused = lookup_case(root, protocol_path, model, track, smoke, runtime=runtime)
    found = [entry for entry in found if not evaluated(entry)]
    if iteration_dir is not None:  # Prefer an origin from the same iteration (same frozen runtime).
        prefix = relative(iteration_dir, root) + "/"
        found.sort(key=lambda entry: not entry["value"]["result"].startswith(prefix))
    return (found[0] if found else None), refused


def lookup_check(root, script, arguments, runtime=None):
    from scripts.r01_workflow import training_inputs
    if runtime is None:
        from scripts.r01_preflight import runtime_info
        runtime = runtime_from_preflight(runtime_info(root, 1))
    key = check_key(root, script, arguments, training_inputs(root), runtime)
    return lookup(root, key, require=lambda entry: [] if entry["value"]["status"] == "passed" else ["check failed"])


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("backfill")
    r = sub.add_parser("record")
    r.add_argument("--result", type=Path, required=True)
    c = sub.add_parser("lookup-case")
    c.add_argument("--protocol", type=Path, required=True)
    c.add_argument("--model", default="expertium")
    c.add_argument("--track", required=True)
    c.add_argument("--smoke", action="store_true")
    c.add_argument("--options", help="Options JSON overriding the frozen track options (e.g. a training seed)")
    k = sub.add_parser("lookup-check")
    k.add_argument("command_line", nargs=REMAINDER)
    sub.add_parser("verify")
    sub.add_parser("stats")
    args = parser.parse_args()
    if args.command == "backfill":
        print(json.dumps(backfill(ROOT)))
    elif args.command == "record":
        target, new = record(ROOT, ROOT / args.result)
        print(json.dumps({"entry": relative(target), "new": new}))
    elif args.command == "lookup-case":
        options = json.loads(args.options) if args.options else None
        found, refused = lookup_case(ROOT, ROOT / args.protocol, args.model, args.track, args.smoke, options)
        print(json.dumps({"reusable": [entry["value"]["result"] for entry in found], "refused": refused[:5]}))
    elif args.command == "lookup-check":
        line = [part for part in args.command_line if part != "--"]
        found, refused = lookup_check(ROOT, line[0], line[1:])
        print(json.dumps({"reusable": [entry["value"]["result"] for entry in found], "refused": refused[:5]}))
    else:
        rows = list(entries(ROOT))
        if args.command == "verify":
            bad = [{"entry": relative(path), "reasons": reasons} for path, entry in rows
                   if (reasons := [reason for reason in valid(ROOT, entry) if "manifest" not in reason and not reason.startswith("status")])]
            print(json.dumps({"entries": len(rows), "inconsistent": bad[:10], "inconsistent_count": len(bad)}))
            if bad:
                raise SystemExit(1)
        else:
            kinds = {}
            for _, entry in rows:
                kind = kinds.setdefault(entry["key"]["kind"], {"entries": 0, "keys": set(), "reusable_runtime": 0})
                kind["entries"] += 1
                kind["keys"].add(entry["key_sha256"])
                kind["reusable_runtime"] += bool(entry["value"].get("runtime_complete"))
            print(json.dumps({kind: {**value, "keys": len(value["keys"])} for kind, value in kinds.items()}))


if __name__ == "__main__":
    main()
