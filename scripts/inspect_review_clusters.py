"""Inspect connected components under the relation viewer's exact filters."""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", type=float, default=95)
    parser.add_argument("--mode", choices=("either", "both"), default="either")
    args = parser.parse_args()
    if not 0 <= args.coverage <= 100:
        parser.error("coverage must be between 0 and 100")
    provenance = process_provenance()
    nodes = {n["id"]: n for n in json.loads((args.audit / "nodes.json").read_text())}
    with sqlite3.connect(args.audit / "relations.sqlite") as db:
        edges = [dict(zip(("a", "b", "shared", "coverage_a", "coverage_b", "longest"), row))
                 for row in db.execute("SELECT a,b,shared_reviews,coverage_a,coverage_b,longest_segment FROM relations WHERE kind='common_segment' AND longest_segment>=20")]
    metric = max if args.mode == "either" else min
    edges = [e for e in edges if metric(e["coverage_a"], e["coverage_b"]) >= args.coverage / 100]
    parents = {}
    def find(x):
        parents.setdefault(x, x)
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    for e in edges:
        parents[find(e["b"])] = find(e["a"])
    groups = defaultdict(list)
    for x in parents:
        groups[find(x)].append(x)
    components = []
    for members in groups.values():
        ids = set(members)
        selected = [e for e in edges if e["a"] in ids]
        components.append({"groups": sorted(members), "users": sorted(u for x in members for u in nodes[x]["users"]),
                           "group_count": len(members), "edge_count": len(selected), "edges": selected,
                           "node_summaries": [nodes[x] for x in sorted(members)]})
    components.sort(key=lambda c: (-c["group_count"], -c["edge_count"]))
    report = {"provenance": provenance, "source_audit": str(args.audit.resolve()),
              "relations_sqlite_sha256": sha256((args.audit / "relations.sqlite").read_bytes()).hexdigest(),
              "nodes_sha256": sha256((args.audit / "nodes.json").read_bytes()).hexdigest(),
              "filter": {"relation": "common_segment", "minimum_block": 20, "coverage_percent": args.coverage, "coverage_mode": args.mode},
              "related_groups": len(parents), "links": len(edges), "components": components}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"groups": len(parents), "links": len(edges), "components": [{k: c[k] for k in ("group_count", "edge_count", "groups")} for c in components]}, indent=2))


if __name__ == "__main__":
    main()
