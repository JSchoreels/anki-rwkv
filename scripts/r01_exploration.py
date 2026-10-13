"""P-A exploration lane: record an uncounted R01 prototype screen (opt-in, pending approval).

Proposal P-A in docs/R01_NEXT_ROUND_PLAN.md lets a round carry bounded, uncounted
prototypes for structural tracks (F05, F69, F22, bucketed compilation), screened with
kernel/real-batch microbenchmarks and paired real-batch A/B timing before any
full-run blocks are spent. This tool only writes a record under
artifacts/retraining/gpu-retrain-r01/exploration/<track>-<n>/. It launches nothing and
never reads or writes protocols, plans, run logs, round budgets, ledgers or promotion
decisions: a screen that passes still enters a counted slot under the normal gates.

  record  --track F05 --options <json|file> --source <file>... --microbenchmark <path>...
          --paired-ab <path>... --outcome passed|failed|inconclusive --decision "<text>"
  list    one line per record
"""

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "artifacts/retraining/gpu-retrain-r01/exploration"
TRACK = re.compile(r"^[A-Z][0-9]{2,3}[a-z]{0,2}$")
OUTCOMES = ("passed", "failed", "inconclusive")
SCOPE = ("Uncounted P-A exploration screen. Not a counted iteration, not a promotion, not broader validation; "
         "it does not consume or change a round budget. A passed screen only makes the prototype eligible "
         "for a counted slot under the normal exactness, speed and quality gates.")


def repository_file(value, kind):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"{kind} must be inside the repository: {value}")
    if not path.is_file():
        raise ValueError(f"{kind} does not exist or is not a file: {value}")
    data = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def parse_options(value):
    candidate = Path(value)
    if not value.lstrip().startswith("{") and (candidate if candidate.is_absolute() else ROOT / candidate).is_file():
        source = repository_file(value, "options file")
        options = json.loads((ROOT / source["path"]).read_text(encoding="utf-8"))
    else:
        source, options = None, json.loads(value)
    if not isinstance(options, dict):
        raise ValueError("Options must be a JSON object")
    return options, source


def record(*, track, options, sources, microbenchmarks, paired_ab, outcome, decision, owner,
           lane_authorization=None, notes=None, base=BASE, provenance=None):
    if not TRACK.fullmatch(track):
        raise ValueError(f"Track ID must look like a ledger ID (F05, F69a): {track}")
    if outcome not in OUTCOMES:
        raise ValueError(f"Outcome must be one of {OUTCOMES}")
    if not decision.strip() or not owner.strip():
        raise ValueError("Decision text and owner are required")
    if not microbenchmarks and not paired_ab:
        raise ValueError("At least one microbenchmark or paired real-batch A/B evidence path is required")
    options, options_source = parse_options(options)
    evidence = {"microbenchmarks": [repository_file(p, "microbenchmark evidence") for p in microbenchmarks],
                "paired_real_batch_ab": [repository_file(p, "paired A/B evidence") for p in paired_ab]}
    source_files = [repository_file(p, "source file") for p in sources]
    if provenance is None:
        from scripts.experiment_provenance import process_provenance
        provenance = process_provenance()
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    taken = [int(m.group(1)) for d in base.iterdir() if (m := re.fullmatch(re.escape(track) + r"-(\d+)", d.name))]
    number = max(taken, default=0) + 1
    while True:
        folder = base / f"{track}-{number}"
        try:
            folder.mkdir()  # Atomic: concurrent recorders never share or overwrite a folder.
            break
        except FileExistsError:
            number += 1
    entry = {"kind": "r01-exploration-screen", "track": track, "number": number, "created_utc": datetime.now(timezone.utc).isoformat(),
             "owner": owner, "counted": False, "promotion": None, "scope": SCOPE,
             "lane_status": ("P-A approved: " + lane_authorization) if lane_authorization else "P-A pending human approval (preparation record)",
             "options": options, "options_source": options_source, "source_files": source_files, "evidence": evidence,
             "outcome": outcome, "decision": decision.strip(), "notes": notes, "command": sys.argv, "provenance": provenance}
    (folder / "record.json").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    links = [f"- {kind}: `{item['path']}` (sha256 {item['sha256'][:12]})" for kind, items in evidence.items() for item in items]
    (folder / "README.md").write_text("\n".join([
        f"# {track} exploration screen {number}", "",
        f"{SCOPE} Lane: {entry['lane_status']}.", "",
        f"Outcome: **{outcome}**. Decision: {entry['decision']}", "",
        f"Owner `{owner}`; source commit `{provenance.get('git_commit')}` (dirty: {provenance.get('git_dirty')}), snapshot `{provenance.get('source_archive')}`.", "",
        "Evidence:", *links, "", "Full record: [record.json](record.json)", ""]), encoding="utf-8")
    return folder


def listing(base=BASE):
    lines = []
    for path in sorted(Path(base).glob("*/record.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        lines.append(f"{entry['track']}-{entry['number']} {entry['outcome']} {entry['created_utc'][:19]} {entry['decision'][:100]}")
    return lines


def main():
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("record")
    add.add_argument("--track", required=True)
    add.add_argument("--options", required=True, help="prototype options as a JSON object or a JSON file path")
    add.add_argument("--source", action="append", default=[], help="prototype source file (repeat)")
    add.add_argument("--microbenchmark", action="append", default=[], help="kernel or real-batch microbenchmark evidence (repeat)")
    add.add_argument("--paired-ab", action="append", default=[], help="paired real-batch A/B timing evidence, P3b (repeat)")
    add.add_argument("--outcome", required=True, choices=OUTCOMES)
    add.add_argument("--decision", required=True, help="decision text: what the screen shows and the next step")
    add.add_argument("--owner", required=True)
    add.add_argument("--lane-authorization", help="quoted human approval of P-A, once given")
    add.add_argument("--notes")
    commands.add_parser("list")
    args = parser.parse_args()
    if args.command == "list":
        print("\n".join(listing()) or "no exploration records")
        return
    folder = record(track=args.track, options=args.options, sources=args.source, microbenchmarks=args.microbenchmark,
                    paired_ab=args.paired_ab, outcome=args.outcome, decision=args.decision, owner=args.owner,
                    lane_authorization=args.lane_authorization, notes=args.notes)
    print(f"recorded {folder.relative_to(ROOT).as_posix()} (uncounted)")


if __name__ == "__main__":
    main()
