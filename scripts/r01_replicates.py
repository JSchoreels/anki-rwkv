"""Run uncounted R01 timing replicates in one frozen diagnostic iteration (R01-M01).

Human direction 2026-10-12: "better do more runs and have a better statistical understanding of our
results than just merely doing 7% from the best we ever saw". Each replicate is a complete,
isolated full-history run in `iteration-NNNN/replicates/rK/`, launched only when background CPU is
low, in the requested order (for example A,A,A,A for calibration or A,B,A,B for interleaved blocks).

  r01_replicates.py prepare --source <iteration dir> --output <new iteration dir> --authorization "<quote>"
  r01_replicates.py run --output <iteration dir> --sequence TRACK[,TRACK...] --reason "<repeat reason>"
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.r01_workflow import configure_runtime, run_logged, write_json
from scripts.gpu_r01_policy import launch_plan, result_metadata


def gpu_state():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,clocks.sm,pstate,utilization.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout.strip()
    used, clock, pstate, util = [part.strip() for part in out.split(",")]
    return {"memory_used_mib": float(used), "clocks_sm_mhz": float(clock), "pstate": pstate, "utilization_gpu": float(util)}


def wait_for_quiet(limit, max_wait, sample_seconds=5):
    """Sample system CPU until it is at most `limit` percent; never launch on a busy machine."""
    import psutil
    samples, started = [], time.monotonic()
    while True:
        value = psutil.cpu_percent(interval=sample_seconds)
        samples.append(round(value, 1))
        if value <= limit:
            return {"total_cpu_percent": value, "samples": samples, "waited_seconds": round(time.monotonic() - started, 1)}
        if time.monotonic() - started > max_wait:
            raise RuntimeError(f"Background CPU stayed above {limit}% for {max_wait:.0f} s: {samples[-6:]}; no replicate launched")
        print(f"[replicates] background CPU {value:.1f}% > {limit}%; waiting", flush=True)
        time.sleep(25)


def prepare(args):
    from scripts.experiment_provenance import process_provenance
    from scripts.run_gpu_r01 import R01SourceWatch
    source, output = (ROOT / args.source).resolve(), (ROOT / args.output).resolve()
    if output.exists():
        raise FileExistsError("Fresh evidence directory required")
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    provenance = process_provenance()
    if provenance.get("git_dirty") is not False:
        raise RuntimeError("Commit first; replicates must start from a clean tree")
    iteration = output.name.removeprefix("iteration-")
    protocol.update(iteration=iteration, authorization=args.authorization, round_authorization=args.authorization,
                    round_start_iteration=iteration, preparation_provenance=provenance,
                    implementation_sha256=R01SourceWatch(provenance).implementation_sha256,
                    diagnostic_replicates={"scope": "uncounted full-history timing replicates; no optimization decision, no broader validation",
                                           "authorization": args.authorization,
                                           "source_protocol": result_metadata(source / "protocol.json", ROOT)},
                    frozen_utc=datetime.now(timezone.utc).isoformat())
    protocol.pop("diagnostic_repeat", None)
    loop = (ROOT / "docs/GPU_RETRAIN_LOOP.md").read_bytes()
    protocol["loop_sha256"] = hashlib.sha256(loop).hexdigest()
    output.mkdir(parents=True)
    (output / "loop-definition.md").write_bytes(loop)
    write_json(output / "protocol.json", protocol)
    print(json.dumps({"prepared": output.relative_to(ROOT).as_posix(), "commit": provenance.get("git_commit"),
                      "tracks": list(protocol["case_options_by_model"]["expertium"])}))


def run(args):
    configure_runtime(ROOT, 1)
    output = (ROOT / args.output).resolve()
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    sequence = [track.strip() for track in args.sequence.split(",") if track.strip()]
    unknown = [track for track in sequence if track not in protocol["case_options_by_model"]["expertium"]]
    if unknown:
        raise ValueError(f"Tracks outside the frozen protocol: {unknown}")
    replicates = output / "replicates"
    existing = sorted(int(p.name[1:]) for p in replicates.glob("r*") if p.name[1:].isdigit()) if replicates.exists() else []
    # The record stays beside the iteration folder: the isolated runtime copies that folder once,
    # so a record updated inside it would differ from the runtime copy at the next export.
    log_path = output.parent / f"{output.name}-replicates.json"
    record = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {"runs": []}
    for offset, track in enumerate(sequence, start=(existing[-1] if existing else 0) + 1):
        try:
            quiet = wait_for_quiet(args.max_background_cpu, args.max_wait_minutes * 60)
        except RuntimeError as refusal:
            # Record the refusal; nothing was launched, so no replicate folder exists for it.
            record.setdefault("refusals", []).append({"track": track, "reason": str(refusal),
                                                      "recorded_utc": datetime.now(timezone.utc).isoformat()})
            write_json(log_path, record)
            raise
        folder = replicates / f"r{offset}"
        folder.mkdir(parents=True)
        for name in ("protocol.json", "loop-definition.md", "selection.json", "reference-reuse-amendment.json"):
            if (output / name).exists():  # selection.json lets the launch guard accept the selected candidate
                (folder / name).write_bytes((output / name).read_bytes())
        state = {"background": quiet, "gpu_state": gpu_state(), "recorded_utc": datetime.now(timezone.utc).isoformat()}
        write_json(folder / "background-load.json", {"total_cpu_percent": quiet["total_cpu_percent"], **state})
        launch = SimpleArgs(track, args.reason)
        print(f"[replicates] r{offset} {track}: " + json.dumps(launch_plan(ROOT, folder, launch))[:200], flush=True)
        command = [sys.executable, "-X", "utf8", str(ROOT / "scripts/run_isolated_gpu_r01.py"), "--case",
                   "--output", str(folder.relative_to(ROOT)), "--model", "expertium", "--track", track,
                   "--repeat-reason", args.reason]
        # Logs stay outside the iteration folder: the isolated runtime copies it, and a live log would differ at export.
        log = output.parent / f"{output.name}-replicate-r{offset}-console.log"
        run_logged(f"replicate-r{offset}", command, log, root=ROOT, heartbeat_seconds=60, timeout_seconds=1200)
        result = json.loads((folder / "expertium" / track / "result.json").read_text(encoding="utf-8"))
        record["runs"].append({"replicate": f"r{offset}", "track": track, "reviews_per_second": result["reviews_per_second"],
                               "training_seconds": result["training_seconds"], "trained_parameter_sha256": result["trained_parameter_sha256"],
                               **state})
        write_json(log_path, record)
        print(f"[replicates] r{offset} {track}: {result['reviews_per_second']:,.0f} reviews/s", flush=True)


class SimpleArgs:
    def __init__(self, track, reason):
        self.model, self.track, self.smoke, self.check_reason = "expertium", track, False, None
        self.repeat_reason = reason


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--authorization", required=True)
    r = sub.add_parser("run")
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--sequence", required=True)
    r.add_argument("--reason", required=True)
    r.add_argument("--max-background-cpu", type=float, default=10.0)
    r.add_argument("--max-wait-minutes", type=float, default=20.0)
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.command](args)


if __name__ == "__main__":
    main()
