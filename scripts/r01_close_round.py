"""Close an R01 round after the implementation commit, printing only status lines.

For each iteration: write implementation-commit.json (kept if present unless
--overwrite), rerun scripts/report_gpu_r01_next.py --brief and print its line.
Then write a draft round summary with the filled results table (only if the
summary does not exist yet; narrative sections stay as TODO lines) and, with
--ledger-change, append the implementation-commit row to docs/GPU_OPTIMIZATIONS.md.

Example:
  scripts/r01_close_round.py --iterations 0028-0030 --commit HEAD --ledger-change "Shared shift orders ..." --ledger-tracks "F35, F38"
"""

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.r01_metrics import case_metrics  # noqa: E402
from scripts.report_gpu_r01_next import describe  # noqa: E402

R = Path("artifacts/retraining/gpu-retrain-r01")
LEDGER = ROOT / "docs/GPU_OPTIMIZATIONS.md"


def iteration_numbers(text):
    if "-" in text:
        first, last = text.split("-")
        return [f"{number:04d}" for number in range(int(first), int(last) + 1)]
    return [f"{int(number):04d}" for number in text.split(",")]


def entry(iteration, model):
    comparison = json.loads((ROOT / R / f"iteration-{iteration}" / "comparison.json").read_text(encoding="utf-8"))
    item = comparison["comparisons"][model]
    exact = all(item["parity"]["fields"].values())
    case = ROOT / R / f"iteration-{iteration}" / model / item["track"]
    previous = ROOT / Path(item["references"]["previous"]["path"]).parent
    isolation = json.loads((ROOT / R / f"iteration-{iteration}" / "source-isolation.json").read_text(encoding="utf-8"))
    return {"iteration": iteration, "track": item["track"], "exact": exact, "accepted": item["accepted_main"],
            "decision": item["decision"], "gain": item["references"]["previous"]["throughput_gain_percent"],
            "case": case, "previous": previous, "origin": isolation["origin_git_commit"], "metrics": case_metrics(case)}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", required=True, help="Range like 0028-0030 or a comma list")
    parser.add_argument("--commit", default="HEAD", help="Implementation commit (resolved with git rev-parse)")
    parser.add_argument("--model", default="expertium")
    parser.add_argument("--agent", default="claude")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing implementation-commit.json files")
    parser.add_argument("--ledger-change", help="Change/bundle text for a new implementation-commits ledger row")
    parser.add_argument("--ledger-tracks", help="Related track IDs for that row, e.g. 'F35, F38'")
    args = parser.parse_args()
    commit = subprocess.run(["git", "rev-parse", args.commit], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    iterations = iteration_numbers(args.iterations)
    entries = [entry(iteration, args.model) for iteration in iterations]

    for item in entries:
        directory = ROOT / R / f"iteration-{item['iteration']}"
        record_path = directory / "implementation-commit.json"
        write = args.overwrite or not record_path.exists()
        if write:
            role = "accepted" if item["accepted"] else ("exact, not accepted" if item["exact"] else "not exact, not accepted")
            record = {"git_commit": commit, "recorded_utc": datetime.now(timezone.utc).isoformat(),
                      "timing": "implementation commit after completed experiments; not the revision tested by any full run",
                      "tracks": item["track"], "source_archive": item["metrics"]["source_archive"],
                      "evidence": str((R / f"iteration-{item['iteration']}" / "summary.md").as_posix()),
                      "scope": f"{item['track']} ({describe(item['track'])}; {role}); the tested isolated runtime is a dirty snapshot over origin {item['origin'][:7]}",
                      "experiment_git_commit": item["metrics"]["experiment_git_commit"]}
            record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        report = subprocess.run([sys.executable, "scripts/report_gpu_r01_next.py", "--output", str(R / f"iteration-{item['iteration']}"), "--brief"],
                                cwd=ROOT, capture_output=True, text=True)
        lines = report.stdout.strip().splitlines()
        print(f"[{item['iteration']}] record {'written' if write else 'kept'}; "
              + (lines[-1] if report.returncode == 0 and lines else f"REPORT FAILED: {report.stderr.strip()[-300:]}"))

    first, last = iterations[0], iterations[-1]
    reference = case_metrics(entries[0]["previous"])
    reference_iteration = entries[0]["previous"].parents[1].name.split("-")[-1]
    summary = ROOT / R / f"{args.agent}-round-{first}-{last}-summary.md"
    accepted = [item for item in entries if item["accepted"]]
    if not summary.exists():
        def cells(metrics):
            return (f"{metrics['training_seconds']:.3f} | {metrics['reviews_per_second']:,.0f}", f"{metrics['process_wall_seconds']:.3f}",
                    f"{metrics.get('queue_wait_seconds', float('nan')):.2f}", f"{metrics.get('preparation_seconds', float('nan')):.1f}")
        speed, wall, wait, prep = cells(reference)
        rows = [f"| [{reference_iteration}](iteration-{reference_iteration}/summary.md) | {entries[0]['previous'].name} | {speed} | — | {wall} | {wait} | {prep} | Previous reference |"]
        for item in entries:
            speed, wall, wait, prep = cells(item["metrics"])
            decision = "Accepted" if item["accepted"] else ("Exact; speed target missed" if item["exact"] else item["decision"])
            rows.append(f"| [{item['iteration']}](iteration-{item['iteration']}/summary.md) | {item['track']} | {speed} | {item['gain']:+.2f}% | {wall} | {wait} | {prep} | {decision} |")
        headline = (f"The accepted preliminary reference is now **{accepted[-1]['iteration']}** ({accepted[-1]['track']})." if accepted
                    else f"No iteration was accepted; the reference stays {reference_iteration}.")
        summary.write_text("\n".join([
            f"# {args.agent.capitalize()} continuation: iterations {first}–{last}", "",
            f"{headline} TODO: gain versus the round's reference and the original baseline, parity statement, authorization quote, implementation commit `{commit[:7]}`.", "",
            "## Full-history results", "",
            "TODO: cohort, seed, precision, chunks and updates line (copy from the previous round summary).", "",
            "| Iteration | Bundle | Training seconds | Reviews/s | Gain vs reference | Complete process seconds | Queue wait s | Summed worker preparation s | Decision |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|", *rows, "",
            *[f"**{item['iteration']}, {describe(item['track'])}.** TODO: hypothesis, change and measured outcome in 2–4 sentences.\n" for item in entries],
            "## Checks and caveats", "", "- TODO: bitwise checks, background load, machine-state drift.", "",
            "## Next ideas (ledger)", "", "TODO", ""]), encoding="utf-8")
        print(f"[summary] draft written: {summary.relative_to(ROOT)}")
    else:
        print(f"[summary] kept existing {summary.relative_to(ROOT)}")

    if args.ledger_change:
        records = ", ".join(f"[{item['iteration']}](../{R.as_posix()}/iteration-{item['iteration']}/implementation-commit.json)" for item in entries)
        experiments = "; ".join(f"{item['iteration']} `{item['metrics']['experiment_git_commit']}`" for item in entries)
        status = (f"Accepted {', '.join(item['iteration'] for item in accepted)} under the {'exact 2%' if all(item['exact'] for item in accepted) else 'applicable'} gate"
                  if accepted else "Not accepted") + "; preliminary only"
        row = (f"| {args.ledger_change} | {args.ledger_tracks or ', '.join(item['track'] for item in entries)}; "
               f"[round summary](../{summary.relative_to(ROOT).as_posix()}), commit records for {records} | `{commit}` | "
               f"{experiments} (dirty isolated snapshots over `{entries[0]['origin'][:7]}`) | {status} |")
        lines = LEDGER.read_text(encoding="utf-8").split("\n")
        start = lines.index("## Implementation commits")
        end = max(index for index in range(start, len(lines)) if lines[index].startswith("|"))
        if any(commit in line for line in lines[start:end + 1]):
            print("[ledger] row with this commit already present; unchanged")
        else:
            lines.insert(end + 1, row)
            LEDGER.write_text("\n".join(lines), encoding="utf-8")
            print("[ledger] implementation-commit row appended")


if __name__ == "__main__":
    main()
