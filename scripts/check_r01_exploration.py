"""CPU checks for scripts/r01_exploration.py (P-A, pending approval); writes only to a check folder."""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts import r01_exploration as lane


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Preserve evidence; use a new artifact directory")
    output.mkdir(parents=True)
    provenance = process_provenance()
    result = {"provenance": provenance, "checks": [], "status": "running"}

    def check(name, value):
        if not value:
            raise AssertionError(name)
        result["checks"].append(name)

    def rejects(name, function):
        try:
            function()
        except ValueError:
            result["checks"].append(name)
            return
        raise AssertionError(name)

    base = output / "exploration"
    evidence = output / "evidence"
    evidence.mkdir()
    bench = evidence / "bench.json"
    bench.write_text('{"median_ms": 1.0}\n', encoding="utf-8")
    ab = evidence / "paired-ab.json"
    ab.write_text('{"paired_gain_percent": 2.0}\n', encoding="utf-8")
    options_file = evidence / "options.json"
    options_file.write_text('{"cuda_graphs": true}\n', encoding="utf-8")
    real_bench = bench.relative_to(ROOT).as_posix()
    common = dict(sources=["scripts/r01_exploration.py"], owner="r01-harness-check", base=base, provenance=provenance)
    budget_files = [ROOT / "docs/GPU_OPTIMIZATIONS_LOG.md", ROOT / "docs/R01_NEXT_ROUND_PLAN.md"]
    before = {path: path.stat().st_mtime_ns for path in budget_files}
    lane.record(track="F05", options='{"cuda_graphs": true}', microbenchmarks=[real_bench], paired_ab=[str(ab)],
                outcome="passed", decision="synthetic check record", **common)
    second = lane.record(track="F05", options=str(options_file), microbenchmarks=[], paired_ab=[str(ab)],
                         outcome="inconclusive", decision="second synthetic record", lane_authorization="quoted approval", **common)
    lane.record(track="F69a", options="{}", microbenchmarks=[real_bench], paired_ab=[], outcome="failed", decision="third", **common)
    names = sorted(p.name for p in base.iterdir())
    check("per-track numbering increments", names == ["F05-1", "F05-2", "F69a-1"])
    entry = json.loads((second / "record.json").read_text(encoding="utf-8"))
    check("record is uncounted with no promotion", entry["counted"] is False and entry["promotion"] is None)
    check("options read from a file with its hash", entry["options"] == {"cuda_graphs": True} and entry["options_source"]["path"].endswith("options.json"))
    check("evidence hashed", entry["evidence"]["paired_real_batch_ab"][0]["sha256"] and entry["evidence"]["microbenchmarks"] == [])
    check("provenance recorded", entry["provenance"]["git_commit"] == provenance["git_commit"])
    check("lane approval text recorded only when given", entry["lane_status"].startswith("P-A approved")
          and json.loads((base / "F05-1/record.json").read_text(encoding="utf-8"))["lane_status"].startswith("P-A pending"))
    check("README written", (second / "README.md").read_text(encoding="utf-8").startswith("# F05 exploration screen 2"))
    check("listing has one line per record", len(lane.listing(base)) == 3)
    rejects("bad track ID rejected", lambda: lane.record(track="cuda graphs", options="{}", microbenchmarks=[real_bench], paired_ab=[],
                                                         outcome="passed", decision="x", **common))
    rejects("missing evidence rejected", lambda: lane.record(track="F05", options="{}", microbenchmarks=[], paired_ab=[],
                                                             outcome="passed", decision="x", **common))
    rejects("nonexistent evidence rejected", lambda: lane.record(track="F05", options="{}", microbenchmarks=["artifacts/does-not-exist.json"],
                                                                 paired_ab=[], outcome="passed", decision="x", **common))
    with tempfile.TemporaryDirectory() as outside:
        stray = Path(outside) / "bench.json"
        stray.write_text("{}", encoding="utf-8")
        check("temporary folder is outside the repository", not stray.resolve().is_relative_to(ROOT))
        rejects("evidence outside the repository rejected", lambda: lane.record(track="F05", options="{}", microbenchmarks=[str(stray)],
                                                                                paired_ab=[], outcome="passed", decision="x", **common))
    rejects("non-object options rejected", lambda: lane.record(track="F05", options="[1]", microbenchmarks=[real_bench], paired_ab=[],
                                                               outcome="passed", decision="x", **common))
    rejects("unknown outcome rejected", lambda: lane.record(track="F05", options="{}", microbenchmarks=[real_bench], paired_ab=[],
                                                            outcome="promoted", decision="x", **common))
    check("rejected calls created no folders", sorted(p.name for p in base.iterdir()) == names)
    check("ledger and plan untouched", all(path.stat().st_mtime_ns == before[path] for path in budget_files))
    source = (ROOT / "scripts/r01_exploration.py").read_text(encoding="utf-8")
    check("tool never imports iteration, protocol or ledger writers",
          not any(name in source for name in ("import r01_iterate", "run_gpu_r01", "r01_close_round", "GPU_OPTIMIZATIONS")))
    result["status"] = "passed"
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"passed {len(result['checks'])} checks; evidence {output.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
