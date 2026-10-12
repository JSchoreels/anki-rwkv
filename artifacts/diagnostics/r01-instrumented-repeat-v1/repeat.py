"""One human-authorized, uncounted repeat of accepted 0042 with R01-M01 per-update timing.

Purpose: the first instrumented reference for paired per-update comparisons, and an exactness check
that the accounting-only instrumentation leaves every update, weight, input and score unchanged.
No optimization selection; the counted round starts after this diagnostic.
Usage: repeat.py prepare | smoke | run
"""
import hashlib
import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.r01_workflow import configure_runtime, run_logged, write_json
from scripts.gpu_r01_policy import result_metadata, launch_plan
from scripts.r01_iterate import load_snapshot

R = Path("artifacts/retraining/gpu-retrain-r01")
OUT = ROOT / R / "iteration-0046"
TRACK = "E33-F48-F50"
REFERENCE = ROOT / R / f"iteration-0042/expertium/{TRACK}/result.json"
REASON = ("Human authorized one uncounted, instrumented repeat of accepted 0042 (2026-10-12, decision 4: \"Sure\") "
          "as the paired per-update timing reference for the next round; the source adds accounting-only per-update "
          "timing and GPU-state telemetry (commit e3b15e8), so historical 0042 results cannot provide per-update timing.")
CHECK = ("Accounting-only R01-M01 instrumentation: per-update CUDA events/host timestamps read with deferred values, "
         "allocator counters and device memory at start must leave smoke updates, losses and norms bitwise equal to 0042.")


def gpu_memory_used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,clocks.sm,pstate,utilization.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout.strip()
    used, clock, pstate, util = [part.strip() for part in out.split(",")]
    return {"memory_used_mib": float(used), "clocks_sm_mhz": float(clock), "pstate": pstate, "utilization_gpu": float(util)}


def prepare():
    from scripts.experiment_provenance import process_provenance
    from scripts.run_gpu_r01 import R01SourceWatch
    if OUT.exists():
        raise FileExistsError("Fresh evidence directory required")
    original = ROOT / R / "iteration-0042/protocol.json"
    protocol = json.loads(original.read_text(encoding="utf-8"))
    provenance = process_provenance()
    if provenance.get("git_dirty") is not False:
        raise RuntimeError("Commit the instrumented harness first; the repeat must start from a clean tree")
    reference = result_metadata(REFERENCE, ROOT)
    protocol.update(
        iteration="0046", authorization=REASON, round_authorization=REASON, round_start_iteration="0046",
        prior_no_20_percent_gain_streak={"expertium": 0}, control_tracks={"expertium": TRACK},
        candidate_tracks={"expertium": []},
        case_options_by_model={"expertium": {TRACK: protocol["case_options_by_model"]["expertium"][TRACK]}},
        resource_retuning_pending={"expertium": False}, retuning_candidates={}, models=["expertium"],
        paused_models=["baseline"], previous_iteration="0042",
        references={"expertium": {"previous": reference,
                                  "original_baseline": protocol["references"]["expertium"]["original_baseline"]}},
        preparation_provenance=provenance, implementation_sha256=R01SourceWatch(provenance).implementation_sha256,
        diagnostic_repeat={"scope": "one full repeat of accepted 0042 configuration with instrumented source; uncounted; no optimization round or broader validation",
                           "reason": REASON, "original_protocol": result_metadata(original, ROOT),
                           "changed_source": "accounting-only per-update timing, allocator counters, device memory at start, GPU clocks/P-state telemetry",
                           "band_note": "speed_noise_band takes max/min over identical-configuration repeats, so this repeat can only keep or widen the frozen band"},
        frozen_utc=datetime.now(timezone.utc).isoformat())
    loop = (ROOT / "docs/GPU_RETRAIN_LOOP.md").read_bytes()
    protocol["loop_sha256"] = hashlib.sha256(loop).hexdigest()
    OUT.mkdir(parents=True)
    (OUT / "loop-definition.md").write_bytes(loop)
    write_json(OUT / "protocol.json", protocol)
    print(json.dumps({"prepared": str(OUT.relative_to(ROOT)), "commit": provenance.get("git_commit"),
                      "implementation_sha256": protocol["implementation_sha256"]}))


def launch(smoke):
    configure_runtime(ROOT, 1)
    args = Namespace(model="expertium", track=TRACK, smoke=smoke, repeat_reason=REASON, check_reason=CHECK if smoke else None)
    print(json.dumps(launch_plan(ROOT, OUT, args))[:400], flush=True)
    load = load_snapshot()
    load["gpu_state"] = gpu_memory_used()
    load["recorded_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(OUT / ("background-load-smoke.json" if smoke else "background-load.json"), load)
    print("[repeat] Background before timing: " + json.dumps(load), flush=True)
    command = [sys.executable, "-X", "utf8", str(ROOT / "scripts/run_isolated_gpu_r01.py"), "--case",
               "--output", str(OUT.relative_to(ROOT)), "--model", "expertium", "--track", TRACK, "--repeat-reason", REASON]
    if smoke:
        command += ["--smoke", "--check-reason", CHECK]
    name = "instrumented-smoke" if smoke else "instrumented-repeat"
    # Logs stay beside the iteration folder: the isolated runtime copies that folder, and a live log would then differ at export.
    run_logged(name, command, ROOT / R / f"iteration-0046-{name}-console.log", root=ROOT, heartbeat_seconds=60, timeout_seconds=900)
    print(f"[repeat] Completed {'smoke' if smoke else 'full run'}.", flush=True)


if __name__ == "__main__":
    {"prepare": prepare, "smoke": lambda: launch(True), "run": lambda: launch(False)}[sys.argv[1]]()
