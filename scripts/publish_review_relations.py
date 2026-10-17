"""Build a public D3 research-results viewer without source review data."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/data-quality/index.html")
    args = parser.parse_args()
    summary = json.loads((args.audit / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Only completed audits can be published")
    public_summary = {key: summary[key] for key in (
        "status", "users", "distinct_reviews", "distinct_groups", "equal_groups",
        "subset_edges", "common_segment_edges", "related_groups", "min_segment")}
    if args.benchmark:
        benchmark = json.loads(args.benchmark.read_text())
        result = next(item for item in benchmark["results"]
                      if item["relations_sha256"] == sha256((args.audit / "relations.jsonl").read_bytes()).hexdigest())
        if result["nodes_sha256"] != sha256((args.audit / "nodes.json").read_bytes()).hexdigest():
            raise ValueError("Benchmark does not describe these graph nodes")
        public_summary["benchmark_seconds"] = result["elapsed_seconds"]
        public_summary["benchmark_cache_hits"] = result["cache_hits"]
    nodes = [{key: node[key] for key in ("id", "users", "reviews", "rows")}
             for node in json.loads((args.audit / "nodes.json").read_text())]
    edges = []
    with (args.audit / "relations.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            relation = json.loads(line)
            edge = {key: relation[key] for key in (
                "a", "b", "kind", "subset_id", "superset_id", "shared_reviews",
                "coverage_a", "coverage_b", "longest_segment")}
            edge.update(segment_count=len(relation["segments"]), segments=[])
            edges.append(edge)
    edges.sort(key=lambda edge: (-edge["longest_segment"], -edge["shared_reviews"], edge["a"], edge["b"]))
    if len(edges) != summary["subset_edges"] + summary["common_segment_edges"]:
        raise ValueError("Graph relation count disagrees with the completed audit")
    graph = {"summary": public_summary, "nodes": nodes, "edges": edges,
             "edges_truncated": False, "aggregate_only": True}
    encoded = json.dumps(graph, separators=(",", ":")).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    vendor = ROOT / "scripts/vendor"
    d3 = ("/*\n" + (vendor / "D3-LICENSE.txt").read_text() + "\n*/\n" +
          (vendor / "d3-7.9.0.min.js").read_text()).replace("</script", "<\\/script")
    template = (ROOT / "scripts/review_relations_graph.html").read_text(encoding="utf-8")
    template = template.replace('<p id="scope" class="lede"></p>',
                                '<p id="scope" class="lede"></p>\n'
                                '        <p class="lede"><a href="cluster-95.html">Why the large 95% cluster shares reviews</a></p>')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace("__REVIEW_RELATIONS_DATA__", encoded).replace("__D3_SOURCE__", d3), encoding="utf-8")
    print(f"Published {len(nodes):,} groups and {len(edges):,} relations: {args.output}")


if __name__ == "__main__":
    main()
