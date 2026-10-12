"""Run one R01 iteration end to end and print one line per step.

Writes the plan from the previous iteration's accepted reference plus option
overrides, then runs prepare, select, smoke (compared bitwise with the accepted
reference's smoke), the full case, the resource profile and the brief report,
all through the existing controllers. Console output goes to log files; only
status lines are printed. Use --from-step to resume after a failure.
"""

from argparse import ArgumentParser
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
R = Path("artifacts/retraining/gpu-retrain-r01")
STEPS = ("plan", "prepare", "select", "smoke", "full", "profile", "report")
UPDATE_KEYS = ("step", "user_id", "offset", "reviews", "seed", "loss", "gradient_norm", "curve_labels", "instant_labels", "lr")
RESULT_KEYS = ("status", "updates", "trained_parameter_sha256", "validation_input_sha256", "evaluation")


def say(step, text):
    print(f"[{step}] {text}", flush=True)


def run(step, arguments, log):
    with (ROOT / log).open("w", encoding="utf-8") as handle:
        code = subprocess.run([sys.executable, *arguments], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
    if code:
        tail = (ROOT / log).read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
        say(step, f"FAILED exit {code}; log {log}; last lines: " + " | ".join(line[:200] for line in tail))
        sys.exit(code)


def accepted_reference(previous, model):
    comparison = json.loads((ROOT / R / f"iteration-{previous}" / "comparison.json").read_text())["comparisons"][model]
    result_path = ROOT / comparison["accepted_reference"]["path"]
    return result_path.parent, json.loads(result_path.read_text())


def smoke_view(case):
    updates = [{key: json.loads(line)[key] for key in UPDATE_KEYS}
               for line in (case / "updates.jsonl").read_text().splitlines()]
    result = json.loads((case / "result.json").read_text())
    return updates, {key: result.get(key) for key in RESULT_KEYS}


def load_snapshot():
    import psutil
    query = ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"]
    apps = subprocess.run(query, capture_output=True, text=True).stdout.lower()
    return {"total_cpu_percent": psutil.cpu_percent(interval=2.0), "python_gpu_processes": apps.count("python")}


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
    parser.add_argument("--agent", default="claude")
    parser.add_argument("--from-step", choices=STEPS, default="plan")
    parser.add_argument("--allow-smoke-mismatch", action="store_true", help="For candidates without exact parity")
    args = parser.parse_args()

    out = R / f"iteration-{args.iteration}"
    plan_path = R / "plans" / f"iteration-{args.iteration}-{args.agent}.json"
    reference_dir, reference = accepted_reference(args.previous, args.model)
    control = reference_dir.name
    steps = STEPS[STEPS.index(args.from_step):]

    if "plan" in steps:
        if not args.explanation or not args.evidence:
            parser.error("--explanation and --evidence are required to write the plan")
        if (ROOT / plan_path).exists():
            parser.error(f"{plan_path} exists; resume with --from-step prepare")
        overrides = {}
        for item in args.set:
            key, _, value = item.partition("=")
            overrides[key] = json.loads(value)
        allowlist = (ROOT / "scripts/gpu_r01_prefetch.py").read_text()
        missing = [key for key in overrides if f'"{key}"' not in allowlist]
        if missing:
            parser.error(f"Add {missing} to the allowlist in scripts/gpu_r01_prefetch.py before freezing a protocol")
        names = (ROOT / "scripts/report_gpu_r01_next.py").read_text()
        unnamed = [part for part in args.candidate.split("-") if f'"{part}":' not in names]
        if unnamed:
            parser.error(f"Add {unnamed} to describe() in scripts/report_gpu_r01_next.py")
        options = reference["options"]
        previous_plans = sorted((ROOT / R / "plans").glob(f"iteration-{reference_dir.parent.parent.name.split('-')[-1]}-*.json"))
        control_text = next((json.loads(path.read_text())["case_explanations"][args.model].get(control)
                             for path in previous_plans), None) or f"Accepted reference {control}."
        plan = {"models": [args.model],
                "model_scope_reason": "Continuation of the Expertium-only R01 work; Alex remains paused by earlier human request.",
                "control_tracks": {args.model: control}, "candidate_tracks": {args.model: args.candidate},
                "case_options_by_model": {args.model: {control: options, args.candidate: {**options, **overrides}}},
                "case_explanations": {args.model: {control: control_text, args.candidate: args.explanation}},
                "selection_evidence": args.evidence}
        (ROOT / plan_path).write_text(json.dumps(plan, indent=2), encoding="utf-8")
        say("plan", f"{plan_path}: control {control}, candidate {args.candidate}, overrides {overrides}")

    if "prepare" in steps:
        command = ["scripts/run_gpu_r01.py", "--prepare", "--output", str(out), "--iteration", args.iteration,
                   "--previous", str(R / f"iteration-{args.previous}"), "--plan", str(plan_path)]
        if args.round_authorization:
            command += ["--round-authorization", args.round_authorization]
        run("prepare", command, f"{out}-prepare-console.log")
        say("prepare", f"protocol frozen in {out}/protocol.json")

    if "select" in steps:
        run("select", ["scripts/select_gpu_r01_next.py", "--model", args.model, "--output", str(out)], f"{out}-select-console.log")
        track = json.loads((ROOT / out / "selection.json").read_text())[args.model]["track"]
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
        (ROOT / out / "background-load.json").write_text(json.dumps(load), encoding="utf-8")
        say("full", f"background before timing: {load['total_cpu_percent']}% total CPU, {load['python_gpu_processes']} other Python GPU processes")
        if load["python_gpu_processes"]:
            say("full", "STOP: another Python process is on the GPU")
            sys.exit(1)
        run("full", ["scripts/run_gpu_r01.py", "--case", "--output", str(out), "--model", args.model, "--track", args.candidate],
            f"{out}-{args.model}-console.log")
        result = json.loads((ROOT / out / case / "result.json").read_text())
        say("full", f"status {result['status']}; training {result.get('training_seconds', float('nan')):.3f} s; "
                    f"{result.get('reviews_per_second', float('nan')):,.0f} reviews/s; process wall {result.get('process_wall_seconds', float('nan')):.3f} s")

    if "profile" in steps:
        run("profile", ["scripts/profile_gpu_resources.py", "--case", str(out / case), "--protocol", str(out / "protocol.json"),
                        "--output", str(out / "profiles" / f"{args.model}-{args.candidate}")], f"{out}-profile-console.log")
        say("profile", f"{out}/profiles/{args.model}-{args.candidate}/profile.md")

    if "report" in steps:
        log = f"{out}-report-console.log"
        run("report", ["scripts/report_gpu_r01_next.py", "--output", str(out), "--brief"], log)
        for line in (ROOT / log).read_text(encoding="utf-8").strip().splitlines()[-2:]:
            say("report", line)


if __name__ == "__main__":
    main()
