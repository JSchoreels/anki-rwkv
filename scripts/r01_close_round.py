"""Close an R01 round with complete summaries and compact handoffs.

--preview renders into a new artifacts directory without changing cases, run
logs or the ledger. Existing implementation records and edited summaries are
preserved by default. No training or new quality evaluation is launched.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.r01_round_report import entry, safe_cell, write_documents
from scripts.r01_workflow import configure_runtime, read_json, run_logged, tail, write_json, write_text, StepFailure

R = Path("artifacts/retraining/gpu-retrain-r01")
LEDGER = ROOT / "docs/GPU_OPTIMIZATIONS.md"


def iteration_gpu_time(directory):
    """P-D/F74: wall seconds of every GPU case process recorded in one iteration (smokes, full run, replicates)."""
    runs = []
    for pattern in ("*/*/result.json", "functional-smoke-*/*/*/result.json", "replicates/*/*/*/result.json",
                    "replicates/*/functional-smoke-*/*/*/result.json", "s*/functional-smoke-*/*/*/result.json"):
        for path in Path(directory).glob(pattern):
            if path.parts[-3] not in ("baseline", "expertium"):
                continue
            result = read_json(path)
            reuse = result.get("evaluation_reuse") or {}
            saved = None
            if reuse.get("status") == "reused" and result.get("validation_seconds") is not None:
                saved = reuse["loss_provenance"]["origin_validation_seconds"] - result["validation_seconds"]
            runs.append({"wall": result.get("process_wall_seconds") or 0.0, "smoke": bool(result.get("smoke_only")),
                         "reused": reuse.get("status") == "reused", "saved": saved})
    return {"gpu_process_wall_seconds": round(sum(run["wall"] for run in runs), 1), "runs": len(runs),
            "smoke_runs": sum(run["smoke"] for run in runs), "evaluations_reused": sum(run["reused"] for run in runs),
            "evaluation_seconds_saved": round(sum(run["saved"] or 0.0 for run in runs), 1),
            "scope": "process wall of recorded GPU case processes; excludes waits for a quiet machine, profiles and reports"}


def iteration_numbers(text):
    if "-" in text:
        first, last = map(int, text.split("-"))
        values = list(range(first, last + 1))
    else:
        values = list(map(int, text.split(",")))
    if not values or min(values) < 1 or len(values) != len(set(values)):
        raise ValueError("Use an ascending range or distinct positive iteration numbers")
    return [f"{number:04d}" for number in sorted(values)]


def record_commit(item, commit, directory, overwrite=False):
    path = Path(directory) / "implementation-commit.json"
    if path.exists():
        previous = read_json(path)
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", previous["git_commit"]):
            raise ValueError("Existing implementation record has an invalid commit ID")
        if not overwrite:
            if previous["git_commit"] != commit:
                raise ValueError("Implementation record has another commit; keep it or explicitly --overwrite")
            return False
        archive = path.with_name(f"implementation-commit-before-{previous['git_commit']}.json")
        if archive.exists() and archive.read_bytes() != path.read_bytes():
            raise ValueError("Previous record archive differs; preserve it")
        if not archive.exists():
            archive.write_bytes(path.read_bytes())
    role = "accepted" if item["accepted"] else ("exact, not accepted" if item["exact"] else "not exact, not accepted")
    write_json(path, {"git_commit": commit, "recorded_utc": datetime.now(timezone.utc).isoformat(),
                     "timing": "implementation commit after completed experiments; not the revision tested by any full run",
                     "tracks": item["track"], "source_archive": item["metrics"]["source_archive"],
                     "evidence": (R / f"iteration-{item['iteration']}/summary.md").as_posix(),
                     "scope": f"{item['track']} ({role}); {item['explanation']}; origin {item['origin']}",
                     "experiment_git_commit": item["metrics"]["experiment_git_commit"]})
    return True


def append_ledger(path, root, summary, entries, commit, change, tracks):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start = lines.index("## Implementation commits")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    if any(commit in line for line in lines[start:end]):
        print("[ledger] implementation row already present")
        return
    index = max(i for i in range(start, end) if lines[i].startswith("|")) + 1
    accepted = [item for item in entries if item["accepted"]]
    status = ("Accepted " + ", ".join(item["iteration"] for item in accepted) if accepted else "Not accepted") + "; preliminary only"
    records = ", ".join(f"[{item['iteration']}](../{R.as_posix()}/iteration-{item['iteration']}/implementation-commit.json)" for item in entries)
    experiments = "; ".join(f"{item['iteration']} `{item['metrics']['experiment_git_commit']}` (origin `{item['origin']}`)" for item in entries)
    lines.insert(index, f"| {safe_cell(change)} | {safe_cell(tracks or ', '.join(item['track'] for item in entries))}; [round summary](../{summary.relative_to(root).as_posix()}), commit records for {records} | `{commit}` | {experiments} | {status} |")
    write_text(path, "\n".join(lines) + "\n")
    print("[ledger] implementation row appended")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", required=True)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--model", default="expertium", choices=("expertium", "baseline"))
    parser.add_argument("--agent", default="claude")
    parser.add_argument("--overwrite", action="store_true", help="Replace commit records, preserving earlier records")
    parser.add_argument("--refresh-summary", action="store_true", help="Explicitly regenerate an existing summary/handoff")
    parser.add_argument("--preview", type=Path, help="New artifacts directory; leave all recorded cases untouched")
    parser.add_argument("--next-step", action="append", default=[])
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--ledger-change")
    parser.add_argument("--ledger-tracks")
    args = parser.parse_args()
    try:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", args.agent):
            raise ValueError("Use an agent name without path separators")
        iterations = iteration_numbers(args.iterations)
        commit = subprocess.run(["git", "rev-parse", "--verify", args.commit + "^{commit}"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        entries = [entry(ROOT, R, iteration, args.model) for iteration in iterations]
        configure_runtime(ROOT, 1)
        destination = ROOT / R
        if args.preview:
            destination = (ROOT / args.preview).resolve()
            if not destination.is_relative_to(ROOT / "artifacts") or destination.exists():
                raise ValueError("Preview requires a new directory under artifacts")
            destination.mkdir(parents=True)
        else:
            for item in entries:
                path = item["case"].parents[1] / "implementation-commit.json"
                if path.exists() and not args.overwrite and read_json(path)["git_commit"] != commit:
                    raise ValueError("Existing implementation commit differs; use its revision or explicitly --overwrite")
            for i, item in enumerate(entries):
                directory = item["case"].parents[1]
                written = record_commit(item, commit, directory, args.overwrite)
                log = run_logged("report", [sys.executable, "scripts/report_gpu_r01_next.py", "--output", str(R / directory.name), "--brief"],
                                 directory / "close-report-console.log", root=ROOT, heartbeat_seconds=args.heartbeat_seconds)
                entries[i] = entry(ROOT, R, item["iteration"], args.model)
                if written:
                    path = directory / "implementation-commit.json"
                    record = read_json(path)
                    record["scope"] = f"{entries[i]['track']}; {entries[i]['decision']}; origin {entries[i]['origin']}"
                    write_json(path, record)
                print(f"[{item['iteration']}] record {'written' if written else 'kept'}; {tail(log, lines=1, chars=600)}", flush=True)
        summary = destination / f"{args.agent}-round-{iterations[0]}-{iterations[-1]}-summary.md"
        write_documents(ROOT, R, entries, commit, args.agent, summary, next_steps=args.next_step,
                        refresh=args.refresh_summary, update_index=not args.preview)
        handoff_path = summary.with_name(summary.stem.removesuffix("-summary") + "-handoff.json")
        if handoff_path.exists():
            handoff = read_json(handoff_path)
            if "gpu_time_by_iteration" not in handoff or args.refresh_summary:
                # P-D: rounds are budgeted in GPU time as well as candidate count.
                handoff["gpu_time_by_iteration"] = {item["iteration"]: iteration_gpu_time(item["case"].parents[1]) for item in entries}
                handoff["gpu_process_wall_seconds_total"] = round(sum(value["gpu_process_wall_seconds"]
                                                                      for value in handoff["gpu_time_by_iteration"].values()), 1)
                write_json(handoff_path, handoff)
        if args.ledger_change and not args.preview:
            append_ledger(LEDGER, ROOT, summary, entries, commit, args.ledger_change, args.ledger_tracks)
        print(f"[done] {'preview' if args.preview else 'closed'} {len(entries)} iterations; no training launched", flush=True)
    except (ValueError, StepFailure, subprocess.CalledProcessError) as error:
        print(f"[stop] {error}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
