"""Run one R01 iteration end to end and print one line per step.

Writes the plan from the previous iteration's accepted reference plus option
overrides, then runs prepare, select, smoke (compared bitwise with the accepted
reference's smoke), the full case, the resource profile and the brief report,
all through the existing controllers. Console output goes to log files; only
status lines are printed. Use --from-step to resume after a failure.
"""

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.r01_metrics import case_metrics, drift_line  # noqa: E402
from scripts.r01_workflow import check_evidence, read_json, run_logged, tail, write_json, StepFailure  # noqa: E402
from scripts.r01_preflight import preflight  # noqa: E402
R = Path("artifacts/retraining/gpu-retrain-r01")
STEPS = ("plan", "prepare", "select", "smoke", "full", "profile", "report")
UPDATE_KEYS = ("step", "user_id", "offset", "reviews", "seed", "loss", "gradient_norm", "curve_labels", "instant_labels", "lr")
RESULT_KEYS = ("status", "updates", "trained_parameter_sha256", "validation_input_sha256", "evaluation")
HEARTBEAT_SECONDS = 60


def say(step, text):
    print(f"[{step}] {text}", flush=True)


def run(step, arguments, log):
    try:
        return run_logged(step, [sys.executable, *arguments], log, root=ROOT, heartbeat_seconds=HEARTBEAT_SECONDS)
    except StepFailure as error:
        say(step, str(error))
        raise SystemExit(error.returncode or 1)


def accepted_reference(previous, model):
    comparison = read_json(ROOT / R / f"iteration-{previous}/comparison.json")["comparisons"][model]
    result_path = ROOT / comparison["accepted_reference"]["path"]
    return result_path.parent, read_json(result_path)


def smoke_view(case):
    rows = [json.loads(line) for line in (case / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
    updates = [{key: row[key] for key in UPDATE_KEYS} for row in rows]
    result = read_json(case / "result.json")
    return updates, {key: result.get(key) for key in RESULT_KEYS}


def load_snapshot():
    import psutil
    from scripts.r01_replicates import wait_for_quiet
    quiet = wait_for_quiet(10.0, 20 * 60)
    query = ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"]
    try:
        result = subprocess.run(query, capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("GPU process query timed out; resolve background-load visibility before timing") from error
    if result.returncode:
        raise RuntimeError("GPU process query failed; resolve background-load visibility before timing")
    apps = result.stdout.lower()
    return {"total_cpu_percent": quiet["total_cpu_percent"], "quiet_gate": quiet, "python_gpu_processes": apps.count("python")}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", required=True, help="New iteration number, e.g. 0028")
    parser.add_argument("--previous", required=True, help="Previous iteration number, e.g. 0027")
    parser.add_argument("--model", default="expertium", choices=("expertium", "baseline"))
    parser.add_argument("--candidate", required=True, help="Candidate track ID, e.g. E27-F38")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="Option override on top of the control options")
    parser.add_argument("--explanation", help="1-3 sentence case explanation (run-log 'What was tried')")
    parser.add_argument("--evidence", help="selection_evidence text")
    parser.add_argument("--check-reason", help="Changed behaviour the smoke verifies")
    parser.add_argument("--round-authorization", help="Quoted human request; only at the start of a newly authorized round")
    parser.add_argument("--round-iterations", type=int, help="Explicit human iteration count; complete this count despite misses")
    parser.add_argument("--loss-ceiling", type=float, help="Round relative loss ceiling fraction (0.001 = 0.1%%); only with --round-authorization")
    parser.add_argument("--agent", default="claude")
    parser.add_argument("--from-step", choices=STEPS, default="plan")
    parser.add_argument("--allow-smoke-mismatch", action="store_true", help="For candidates without exact parity")
    parser.add_argument("--preflight", type=Path, help="Reusable preflight JSON under artifacts")
    parser.add_argument("--check", type=Path, action="append", default=[], help="Successful source-compatible prerequisite; review runtime compatibility too")
    parser.add_argument("--heartbeat-seconds", type=int, default=60, help="30–60 seconds; avoid short polling")
    args = parser.parse_args()
    if args.loss_ceiling is not None and not args.round_authorization:
        parser.error("--loss-ceiling is frozen at round start; pass it with --round-authorization")
    if args.round_iterations is not None and (args.round_iterations < 1 or not args.round_authorization):
        parser.error("--round-iterations requires a positive count and new --round-authorization")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", args.agent):
        parser.error("Use an agent name without path separators")
    if not args.iteration.isdigit() or not args.previous.isdigit() or not 0 < int(args.previous) < int(args.iteration):
        parser.error("Use positive iteration numbers, with the new iteration after the previous")
    args.iteration, args.previous = f"{int(args.iteration):04d}", f"{int(args.previous):04d}"
    global HEARTBEAT_SECONDS
    HEARTBEAT_SECONDS = args.heartbeat_seconds
    if not 30 <= HEARTBEAT_SECONDS <= 60:
        parser.error("--heartbeat-seconds must be between 30 and 60")

    out = R / f"iteration-{args.iteration}"
    plan_path = R / "plans" / f"iteration-{args.iteration}-{args.agent}.json"
    reference_dir, reference = accepted_reference(args.previous, args.model)
    control = reference_dir.name
    steps = STEPS[STEPS.index(args.from_step):]
    if args.check and "plan" not in steps:
        parser.error("Record --check evidence when creating the plan; do not add checks to an existing frozen plan")

    if "plan" in steps:
        if not args.explanation or not args.evidence:
            parser.error("--explanation and --evidence are required to write the plan")
        if (ROOT / plan_path).exists():
            parser.error(f"{plan_path} exists; resume with --from-step prepare")
        overrides = {}
        for item in args.set:
            key, _, value = item.partition("=")
            overrides[key] = json.loads(value)
        allowlist = (ROOT / "scripts/gpu_r01_prefetch.py").read_text(encoding="utf-8")
        missing = [key for key in overrides if f'"{key}"' not in allowlist]
        if missing:
            parser.error(f"Add {missing} to the allowlist in scripts/gpu_r01_prefetch.py before freezing a protocol")
        names = (ROOT / "scripts/report_gpu_r01_next.py").read_text(encoding="utf-8")
        unnamed = [part for part in args.candidate.split("-") if f'"{part}":' not in names]
        if unnamed:
            parser.error(f"Add {unnamed} to describe() in scripts/report_gpu_r01_next.py")
        options = reference["options"]
        previous_plans = sorted((ROOT / R / "plans").glob(f"iteration-{reference_dir.parent.parent.name.split('-')[-1]}-*.json"))
        control_text = next((read_json(path)["case_explanations"][args.model].get(control)
                             for path in previous_plans), None) or f"Accepted reference {control}."
        plan = {"models": [args.model],
                "model_scope_reason": "Continuation of the Expertium-only R01 work; Alex remains paused by earlier human request.",
                "control_tracks": {args.model: control}, "candidate_tracks": {args.model: args.candidate},
                "case_options_by_model": {args.model: {control: options, args.candidate: {**options, **overrides}}},
                "case_explanations": {args.model: {control: control_text, args.candidate: args.explanation}},
                "selection_evidence": args.evidence,
                "prerequisite_checks": [check_evidence(ROOT, ROOT / path) for path in args.check]}
        write_json(ROOT / plan_path, plan)
        say("plan", f"{plan_path}: control {control}, candidate {args.candidate}, overrides {overrides}")

    plan = read_json(ROOT / plan_path)
    if any(step in steps for step in ("prepare", "smoke", "full")):
        options = plan["case_options_by_model"][args.model][args.candidate]
        if "prepare" not in steps:
            protocol = read_json(ROOT / out / "protocol.json")
            if protocol.get("plan_sha256") and hashlib.sha256((ROOT / plan_path).read_bytes()).hexdigest() != protocol["plan_sha256"]:
                parser.error("Plan changed after freeze; preserve this iteration and use a new one")
            options = protocol.get("case_options_by_model", {}).get(args.model, {}).get(args.candidate, options)
        for record in plan.get("prerequisite_checks", []):
            digest = hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest()
            if digest != record["sha256"]:
                parser.error("Prerequisite result changed after planning; preserve it and use a new iteration")
            if "prepare" in steps:
                check_evidence(ROOT, ROOT / record["path"])
        threads = options.get("main_threads", reference.get("config", {}).get("CPU_THREADS", 4))
        _, reused = preflight(ROOT, ROOT / (args.preflight or Path(f"{out}-preflight.json")), threads)
        say("preflight", f"{'reused' if reused else 'passed'}; {threads} main threads, writable workspace temp")

    if "prepare" in steps:
        command = ["scripts/run_gpu_r01.py", "--prepare", "--output", str(out), "--iteration", args.iteration,
                   "--previous", str(R / f"iteration-{args.previous}"), "--plan", str(plan_path)]
        if args.round_authorization:
            command += ["--round-authorization", args.round_authorization]
        if args.round_iterations is not None:
            command += ["--round-iterations", str(args.round_iterations)]
        if args.loss_ceiling is not None:
            command += ["--loss-ceiling", repr(args.loss_ceiling)]
        run("prepare", command, f"{out}-prepare-console.log")
        say("prepare", f"protocol frozen in {out}/protocol.json")

    if "select" in steps:
        run("select", ["scripts/select_gpu_r01_next.py", "--model", args.model, "--output", str(out)], f"{out}-select-console.log")
        track = read_json(ROOT / out / "selection.json")[args.model]["track"]
        say("select", f"selected {track}")
        if track != args.candidate:
            say("select", f"STOP: selector chose {track}, not {args.candidate}")
            sys.exit(1)

    case = f"{args.model}/{args.candidate}"
    if "smoke" in steps:
        if not args.check_reason:
            parser.error("--check-reason is required for the smoke")
        run("smoke", ["scripts/run_gpu_r01.py", "--case", "--output", str(out), "--model", args.model, "--track", args.candidate,
                      "--smoke", "--check-reason", args.check_reason], f"{out}-{args.model}-smoke-console.log")
        reference_smoke = reference_dir.parents[1] / "functional-smoke-v1" / args.model / control
        new, old = smoke_view(ROOT / out / "functional-smoke-v1" / args.model / args.candidate), smoke_view(reference_smoke)
        same_updates, same_results = new[0] == old[0], new[1] == old[1]
        differing = [key for key in RESULT_KEYS if new[1][key] != old[1][key]]
        say("smoke", f"status {new[1]['status']}; updates equal {same_updates}; results equal {same_results}"
                     + (f" (differ: {differing})" if differing else "") + f"; reference {reference_smoke.relative_to(ROOT)}")
        if not (same_updates and same_results) and not args.allow_smoke_mismatch:
            say("smoke", "STOP: smoke differs from the accepted reference smoke")
            sys.exit(1)

    if "full" in steps:
        load = load_snapshot()
        write_json(ROOT / out / "background-load.json", load)
        say("full", f"background before timing: {load['total_cpu_percent']}% total CPU, {load['python_gpu_processes']} other Python GPU processes")
        if load["python_gpu_processes"]:
            say("full", "STOP: another Python process is on the GPU")
            sys.exit(1)
        run("full", ["scripts/run_gpu_r01.py", "--case", "--output", str(out), "--model", args.model, "--track", args.candidate],
            f"{out}-{args.model}-console.log")
        result = read_json(ROOT / out / case / "result.json")
        say("full", f"status {result['status']}; training {result.get('training_seconds', float('nan')):.3f} s; "
                    f"{result.get('reviews_per_second', float('nan')):,.0f} reviews/s; process wall {result.get('process_wall_seconds', float('nan')):.3f} s")
        say("drift", drift_line(case_metrics(ROOT / out / case), case_metrics(reference_dir)))

    if "profile" in steps:
        run("profile", ["scripts/profile_gpu_resources.py", "--case", str(out / case), "--protocol", str(out / "protocol.json"),
                        "--output", str(out / "profiles" / f"{args.model}-{args.candidate}")], f"{out}-profile-console.log")
        say("profile", f"{out}/profiles/{args.model}-{args.candidate}/profile.md")
        new, old = case_metrics(ROOT / out / case), case_metrics(reference_dir)
        say("profile", f"training worker CPU {new.get('worker_cpu_percent') or float('nan'):.1f}% ({old.get('worker_cpu_percent') or float('nan'):.1f}%); "
                       f"GPU {new.get('gpu_percent') or float('nan'):.1f}% ({old.get('gpu_percent') or float('nan'):.1f}%) [new (reference)]")

    if "report" in steps:
        log = f"{out}-report-console.log"
        actual_log = run("report", ["scripts/report_gpu_r01_next.py", "--output", str(out), "--brief"], log)
        for line in tail(actual_log, lines=2).splitlines():
            say("report", line)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"[stop] {error}", flush=True)
        raise SystemExit(1)
