"""Run one R01 iteration end to end and print one line per step.

Writes the plan from the previous iteration's accepted reference plus option
overrides, then runs prepare, select, smoke (compared bitwise with the accepted
reference's smoke), the full case, the resource profile and the brief report,
all through the existing controllers. Console output goes to log files; only
status lines are printed. Use --from-step to resume after a failure.

F74 turnaround (opt-in, frozen in protocol["turnaround"] and inherited within a round): --fast-turnaround
enables the reuse-only items T1 (an exact replica reuses an evaluated origin's held-out scores after the F67
proof) and T2 (the full run is block 1's candidate, launched right after reference replicate r1). Identical
smokes found in the result store are reused instead of relaunched. T4-T6 and P-D budgets need --turnaround
with a quoted human approval.
"""

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import os
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


def index(path):
    """Record a finished case in the F74 result store; indexing never stops an iteration."""
    try:
        from scripts.r01_result_store import record
        record(ROOT, path)
    except Exception as error:  # noqa: BLE001
        say("store", f"not indexed: {error}")


def stored_smoke(out, model, track):
    """A recorded smoke with the same key, verified again by the live matching_case guard."""
    try:
        from scripts.r01_result_store import lookup_case
        found, _ = lookup_case(ROOT, ROOT / out / "protocol.json", model, track, smoke=True)
    except Exception as error:  # noqa: BLE001 - a store failure means launching, never reusing
        say("store", f"lookup failed; launching: {error}")
        return None
    return found[0] if found else None


def round_iterations(previous_protocol, previous):
    start = previous_protocol.get("round_start_iteration")
    if not start:
        return []
    return [R / f"iteration-{number:04d}" for number in range(int(start), int(previous) + 1)
            if (ROOT / R / f"iteration-{number:04d}/protocol.json").exists()]


def track_family(track):
    return re.sub(r"[a-z]+$", "", track.split("-")[-1])


def budget_guard(args, turnaround, previous_protocol):
    """P-D (approval required, off by default): per-track variant cap and round GPU-time budget."""
    folders = round_iterations(previous_protocol, args.previous)
    cap = turnaround.get("variant_cap_per_track")
    if cap:
        family = track_family(args.candidate)
        used = [folder.name for folder in folders if track_family(
            (read_json(ROOT / folder / "protocol.json").get("candidate_tracks") or {}).get(args.model, "")) == family]
        if len(used) >= cap and not args.fresh_selection:
            raise ValueError(f"P-D: {family} already has {len(used)} variants this round ({', '.join(used)}); "
                             "select from a fresh profile and pass --fresh-selection <profile evidence>")
    minutes = turnaround.get("gpu_budget_minutes")
    if minutes:
        from scripts.r01_close_round import iteration_gpu_time
        spent = sum(iteration_gpu_time(ROOT / folder)["gpu_process_wall_seconds"] for folder in folders) / 60
        if spent >= minutes:
            raise ValueError(f"P-D: round GPU budget spent ({spent:.1f} of {minutes} min); a new authorization is required")
        say("plan", f"P-D GPU time this round {spent:.1f} of {minutes} min")


def prescreen_guard(args, turnaround, out):
    """T6 (approval required, off by default): skip blocks only for a strongly negative real-batch prescreen."""
    settings = turnaround.get("strong_negative_prescreen")
    if not settings:
        return
    if not args.prescreen:
        raise ValueError("T6 is enabled; pass --prescreen <paired real-batch result with gain_percent>")
    gain = read_json(ROOT / args.prescreen)["gain_percent"]
    if gain < settings["threshold_percent"]:
        write_json(ROOT / out / "prescreen-stop.json", {"gain_percent": gain, "threshold_percent": settings["threshold_percent"],
                                                        "evidence": str(args.prescreen), "decision": "reject without blocks (T6)"})
        say("full", f"STOP: prescreen gain {gain:+.2f}% below {settings['threshold_percent']}% (T6); no blocks run")
        sys.exit(0)


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
    parser.add_argument("--fast-turnaround", action="store_true", help="F74 T1+T2 reuse-only settings for this plan")
    parser.add_argument("--turnaround", help="F74/P-D settings JSON; otherwise the previous protocol's settings are inherited")
    parser.add_argument("--fresh-selection", type=Path, help="P-D: profile evidence for a track past its per-round variant cap")
    parser.add_argument("--prescreen", type=Path, help="T6 (approval required): paired real-batch prescreen result with gain_percent")
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
        previous_protocol = read_json(ROOT / R / f"iteration-{args.previous}/protocol.json")
        turnaround = json.loads(args.turnaround) if args.turnaround else (
            {} if args.round_authorization else dict(previous_protocol.get("turnaround") or {}))
        if args.fast_turnaround:
            turnaround.update(evaluation_reuse="exact_replica", candidate_full_run="first_block")
        if turnaround:
            from scripts.r01_replicates import turnaround_settings
            turnaround = turnaround_settings(turnaround)
            budget_guard(args, turnaround, previous_protocol)
        plan = {"models": [args.model],
                "model_scope_reason": "Continuation of the Expertium-only R01 work; Alex remains paused by earlier human request.",
                "control_tracks": {args.model: control}, "candidate_tracks": {args.model: args.candidate},
                "case_options_by_model": {args.model: {control: options, args.candidate: {**options, **overrides}}},
                "case_explanations": {args.model: {control: control_text, args.candidate: args.explanation}},
                "selection_evidence": args.evidence,
                "prerequisite_checks": [check_evidence(ROOT, ROOT / path) for path in args.check]}
        if turnaround:
            plan["turnaround"] = turnaround
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
    turnaround = (read_json(ROOT / out / "protocol.json").get("turnaround") or {}) if (ROOT / out / "protocol.json").exists() else {}
    if "smoke" in steps:
        if not args.check_reason:
            parser.error("--check-reason is required for the smoke")
        smoke_case = ROOT / out / "functional-smoke-v1" / args.model / args.candidate
        stored = stored_smoke(out, args.model, args.candidate)
        if stored is not None:
            # F74 T3: an identical smoke (same implementation, options, protocol context, runtime) is already recorded.
            smoke_case = ROOT / Path(stored["value"]["result"]).parent
            write_json(ROOT / out / "functional-smoke-v1-reused.json", {"reused": stored["value"], "key_sha256": stored["key_sha256"],
                                                                        "label": "smoke reused from the result store; not relaunched"})
            say("smoke", f"reused identical recorded smoke {smoke_case.relative_to(ROOT)} (result store)")
        else:
            run("smoke", ["scripts/run_gpu_r01.py", "--case", "--output", str(out), "--model", args.model, "--track", args.candidate,
                          "--smoke", "--check-reason", args.check_reason], f"{out}-{args.model}-smoke-console.log")
            if not (smoke_case / "result.json").exists() and (ROOT / out / "reused-results.json").exists():
                reused = read_json(ROOT / out / "reused-results.json")[args.model][args.candidate + " smoke"]
                smoke_case = ROOT / Path(reused["result"]["path"]).parent
                say("smoke", f"launch guard reused recorded {smoke_case.relative_to(ROOT)}")
            index(smoke_case / "result.json")
        reference_smoke = reference_dir.parents[1] / "functional-smoke-v1" / args.model / control
        new, old = smoke_view(smoke_case), smoke_view(reference_smoke)
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
        prescreen_guard(args, turnaround, out)
        if turnaround.get("candidate_full_run") == "first_block":
            # F74 T2: reference replicate r1, then this full run as block 1's candidate (R C), so no full run repeats.
            first = ROOT / out / "replicates/r1" / args.model / control / "result.json"
            if not first.exists():
                if (ROOT / out / "replicates").exists():
                    say("full", "STOP: replicates exist without reference r1; T2 needs r1 to be the reference")
                    sys.exit(1)
                run("full", ["scripts/r01_replicates.py", "run", "--output", str(out), "--sequence", control,
                             "--reason", "F74 T2: block 1 reference run, launched immediately before the candidate full run"],
                    f"{out}-t2-reference-console.log")
            say("full", f"T2 block 1 reference {first.parent.relative_to(ROOT)}")
        origin = None
        if turnaround.get("evaluation_reuse") == "exact_replica":
            from scripts.r01_replicates import evaluation_origin
            origin, _ = evaluation_origin(ROOT / out / "protocol.json", args.candidate, ROOT / out)
            say("full", f"evaluation origin {origin or 'none (evaluates)'}")
        from scripts.r01_evaluation_reuse import ORIGIN_ENVIRONMENT
        if origin:
            os.environ[ORIGIN_ENVIRONMENT] = origin
        try:
            run("full", ["scripts/run_gpu_r01.py", "--case", "--output", str(out), "--model", args.model, "--track", args.candidate],
                f"{out}-{args.model}-console.log")
        finally:
            os.environ.pop(ORIGIN_ENVIRONMENT, None)
        index(ROOT / out / case / "result.json")
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
