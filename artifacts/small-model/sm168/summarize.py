"""SM168: medians of 3 ABAB passes (before = eda153ddb, after = codec-batch)."""
import json
import re
import statistics
import sys
from collections import defaultdict

base = sys.argv[1]
replay = defaultdict(list)  # (bench, runtime, binary, precision) -> [reviews/s]
section = None
for line in open(f"{base}/speed-run.log", encoding="utf-8"):
    m = re.match(r"--- one-thread (\S+) rep=(\d) binary=(\w+) precision=(\S+)", line)
    if m:
        section = (m.group(1), m.group(3), m.group(4))
        continue
    if line.startswith("==="):
        section = None
    m = re.match(r"(big|small|tiny): .*reviews_per_s=(\d+)", line)
    if m and section:
        bench, binary, precision = section
        replay[(bench, m.group(1), binary, precision)].append(int(m.group(2)))

app = defaultdict(lambda: defaultdict(list))
for line in open(f"{base}/app-rebuild-latency.jsonl", encoding="utf-8"):
    r = json.loads(line)
    binary = r["tag"].split("/")[1]
    key = (r["runtime"], binary, r["precision"])
    app[key]["rebuild"].append(r["rebuild"]["replay_reviews_per_s"])
    app[key]["total_s"].append(r["rebuild"]["total_ms"] / 1000)
    for kind in ("review", "learning"):
        for f in ("show_p50_us", "answer_p50_us"):
            app[key][f"{kind}_{f}"].append(r["latency"][kind][f])


def med(values):
    return statistics.median(values)


def spread(values):
    return (max(values) - min(values)) / med(values) * 100


def pct(a, b):
    return (a / b - 1) * 100


out = {"replay": {}, "app": {}}
I8, LR = "compact_cards_int8", "compact_lowrank"
print("## one-thread replay (reviews/s)")
for bench in ("1000x24", "4000x8"):
    for rt in ("big", "small", "tiny"):
        if (bench, rt, "before", I8) not in replay:
            continue
        v = {(b, p): replay[(bench, rt, b, p)] for b in ("before", "after") for p in (I8, LR)}
        m = {k: med(x) for k, x in v.items()}
        out["replay"][f"{bench} {rt}"] = {f"{b} {p}": {"median": m[(b, p)], "runs": v[(b, p)]} for b, p in v}
        print(
            f"{bench} {rt}: int8 rows {m[('before', I8)]:.0f} ({spread(v[('before', I8)]):.1f}%) / "
            f"{m[('after', I8)]:.0f} ({spread(v[('after', I8)]):.1f}%) [{pct(m[('after', I8)], m[('before', I8)]):+.1f}%]; "
            f"default {m[('before', LR)]:.0f} ({spread(v[('before', LR)]):.1f}%) / {m[('after', LR)]:.0f} ({spread(v[('after', LR)]):.1f}%) "
            f"[{pct(m[('after', LR)], m[('before', LR)]):+.1f}%]; vs int8 rows before {pct(m[('before', LR)], m[('before', I8)]):+.1f}% "
            f"after {pct(m[('after', LR)], m[('after', I8)]):+.1f}%"
        )
print("## rebuild (replay reviews/s, all threads) and latency p50 (us)")
for rt in ("big-pair", "small-pair", "tiny-pair"):
    rows = {}
    for b in ("before", "after"):
        for p in (I8, LR):
            d = app[(rt, b, p)]
            rows[(b, p)] = {k: med(x) for k, x in d.items()}
            rows[(b, p)]["rebuild_spread"] = spread(d["rebuild"])
            rows[(b, p)]["runs"] = d["rebuild"]
    out["app"][rt] = {f"{b} {p}": r for (b, p), r in rows.items()}
    r = rows
    print(
        f"{rt}: int8 rows {r[('before', I8)]['rebuild']:.0f} / {r[('after', I8)]['rebuild']:.0f} "
        f"[{pct(r[('after', I8)]['rebuild'], r[('before', I8)]['rebuild']):+.1f}%]; default "
        f"{r[('before', LR)]['rebuild']:.0f} ({r[('before', LR)]['rebuild_spread']:.1f}%) / {r[('after', LR)]['rebuild']:.0f} "
        f"({r[('after', LR)]['rebuild_spread']:.1f}%) [{pct(r[('after', LR)]['rebuild'], r[('before', LR)]['rebuild']):+.1f}%]; "
        f"vs int8 rows before {pct(r[('before', LR)]['rebuild'], r[('before', I8)]['rebuild']):+.1f}% after "
        f"{pct(r[('after', LR)]['rebuild'], r[('after', I8)]['rebuild']):+.1f}%; whole rebuild "
        f"{r[('before', LR)]['total_s']:.2f} -> {r[('after', LR)]['total_s']:.2f} s (int8 rows {r[('after', I8)]['total_s']:.2f} s)"
    )
    for b, p in r:
        x = r[(b, p)]
        print(
            f"   latency {b} {p}: review show {x['review_show_p50_us']:.0f} answer {x['review_answer_p50_us']:.0f}; "
            f"learning show {x['learning_show_p50_us']:.0f} answer {x['learning_answer_p50_us']:.0f}"
        )
json.dump(out, open(f"{base}/summary.json", "w"), indent=1)
