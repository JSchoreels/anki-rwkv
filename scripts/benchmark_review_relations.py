"""Matched CPU-only relation audits with stage timing and immutable sources."""
from argparse import ArgumentParser, Namespace
from contextlib import redirect_stdout
import cProfile
from hashlib import sha256
import json
from pathlib import Path
import pstats
import sys
from threading import Event, Thread, Lock
from time import perf_counter
from types import ModuleType
from uuid import uuid4

import psutil
import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", type=Path, default=ROOT / "scripts/review_relations.py")
    parser.add_argument("--label", required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/anki-revlogs-10k")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/data-quality/history-cache")
    parser.add_argument("--limit-users", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    output = ROOT / "artifacts/data-quality/relations-performance" / (args.label + "-" + uuid4().hex[:6])
    output.mkdir(parents=True)
    source = args.implementation.read_bytes()
    (output / "implementation.py").write_bytes(source)
    provenance = process_provenance()
    module = ModuleType("measured_review_relations")
    # Keep normal repository paths/assets while executing the immutable source copy.
    module.__file__ = str(ROOT / "scripts/review_relations.py")
    exec(compile(source, str(output / "implementation.py"), "exec"), module.__dict__)
    stages = {}
    stage_lock = Lock()
    for name in ("prepare_history", "equal_groups", "subset_candidates", "segment_candidates", "compare_histories", "write_graph"):
        function = getattr(module, name)
        def measured(*items, _name=name, _function=function, **kwargs):
            start = perf_counter()
            try:
                return _function(*items, **kwargs)
            finally:
                elapsed = perf_counter() - start
                with stage_lock:
                    stages[_name] = stages.get(_name, 0.0) + elapsed
        setattr(module, name, measured)
    if hasattr(module, "_ordered_map"):
        ordered_map = module._ordered_map
        def measured_map(function, items, **kwargs):
            start = perf_counter()
            try:
                yield from ordered_map(function, items, **kwargs)
            finally:
                stages[function.__name__ + "_phase_elapsed"] = perf_counter() - start
        module._ordered_map = measured_map
    report = {"label": args.label, "implementation_sha256": sha256(source).hexdigest(),
              "implementation_source": str((output / "implementation.py").relative_to(ROOT)),
              "provenance": provenance, "profiled": args.profile, "results": [],
              "stage_scope": "Function-call durations are summed, with synchronized counters. Concurrent call durations overlap; their sum is not phase wall time. Ordered-map phase elapsed includes main-thread result consumption. End-to-end elapsed is the acceptance metric.",
              "environment": {"python": sys.version, "numpy": np.__version__, "pyarrow": pa.__version__,
                  "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
                  "system_ram_bytes": psutil.virtual_memory().total},
              "scope": "CPU data audit, no training; end-to-end includes graph writing; OS cache and other system work are uncontrolled."}
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Benchmark output: {output}", flush=True)
    for repeat in range(args.repeats):
        stages.clear()
        folder = output / f"repeat-{repeat + 1:02d}"
        stopped, memory = Event(), [0]
        process = psutil.Process()
        def sample():
            while not stopped.wait(0.05):
                memory[0] = max(memory[0], process.memory_info().rss)
        sampler = Thread(target=sample, daemon=True)
        sampler.start()
        profile = cProfile.Profile() if args.profile else None
        start = perf_counter()
        try:
            with (output / f"repeat-{repeat + 1:02d}.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
                if profile:
                    profile.enable()
                summary = module.run(Namespace(data=args.data, output=folder, cache=args.cache,
                    users=None, limit_users=args.limit_users, time_kind="show", min_segment=20,
                    partitions=256, index_memory_mib=256, graph_edge_limit=5000))
                if profile:
                    profile.disable()
        except Exception as error:
            report.update(status="failed", error=repr(error), failed_repeat=repeat + 1)
            (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
            raise
        finally:
            elapsed = perf_counter() - start
            stopped.set()
            sampler.join()
        if profile:
            profile.dump_stats(str(output / "profile.prof"))
            with (output / "profile.txt").open("w") as stream:
                pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(55)
        result = {"repeat": repeat + 1, "elapsed_seconds": elapsed, "stage_seconds": dict(stages),
                  "peak_rss_bytes_sampled": memory[0],
                  **{key: summary[key] for key in ("users", "distinct_reviews", "cache_hits", "candidate_pairs", "minimizer_postings", "equal_groups", "subset_edges", "common_segment_edges")},
                  "relations_sha256": sha256((folder / "relations.jsonl").read_bytes()).hexdigest(),
                  "nodes_sha256": sha256((folder / "nodes.json").read_bytes()).hexdigest(),
                  "canonical_sha256": sha256(json.dumps([(h["user_id"], h["event_sha256"]) for h in json.loads((folder / "input_manifest.json").read_text())["histories"]]).encode()).hexdigest()}
        report["results"].append(result)
        report["status"] = "complete" if repeat + 1 == args.repeats else "running"
        (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
