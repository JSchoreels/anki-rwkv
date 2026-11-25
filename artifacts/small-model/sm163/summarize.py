"""SM163 summary: parse run.log (model_size_bench lines) and app.jsonl (rebuild / latency / sweep).

Writes summary.json and replay.csv / app.csv next to the output folder given as argv[1] (dest dir).
"""
import collections
import csv
import json
import re
import statistics as st
import sys

src = sys.argv[1]  # folder with run.log and out/app.jsonl
dest = sys.argv[2]

FORMATS = ["compact_cards_int8", "compact_lowrank_int8", "compact_lowrank", "compact_lowrank_int6"]
SHORT = {"compact_cards_int8": "int8 rows", "compact_lowrank_int8": "T24 int8 factors",
         "compact_lowrank": "default (by size)", "compact_lowrank_int6": "T24 int6 factors"}

# ---- model_size_bench lines
replay = collections.defaultdict(list)  # (shape, bin, prec, runtime) -> [reviews/s]
loads = collections.defaultdict(list)
cur = None
section = "app"
for line in open(f"{src}/run.log", encoding="utf-8"):
    m = re.match(r"=== (\S+ \S+) rep=(\d) binary=(\w+) precision=(\w+)", line)
    if m:
        cur = (m.group(2), m.group(3), m.group(4)); section = "app"; continue
    m = re.match(r"--- (one-thread 1000x24|one-thread 4000x8|four-thread 4000x8) ", line)
    if m:
        section = m.group(1); continue
    m = re.match(r"cpu_load_percent=(\d+)", line)
    if m and cur:
        loads[section].append(int(m.group(1)))
    m = re.match(r"([a-z-]+): warm_up_ms=([\d.]+) reviews_per_s=(\d+)", line)
    if m and cur and section != "app":
        replay[(section, cur[1], cur[2], m.group(1))].append(int(m.group(3)))

# ---- app.jsonl
app = collections.defaultdict(list)
for line in open(f"{src}/out/app.jsonl", encoding="utf-8"):
    r = json.loads(line)
    rep, binary = r["tag"].split("/")
    app[(binary, r["precision"], r["runtime"])].append(r)


def med(v):
    return st.median(v)


def spread(v):
    m = med(v)
    return (max(v) - min(v)) / m * 100 if m else 0.0


summary = {"loads_percent": {k: {"min": min(v), "median": med(v), "max": max(v)} for k, v in loads.items()}}

# replay tables
rows = []
for (section, binary, prec, rt), v in sorted(replay.items()):
    rows.append({"shape": section, "binary": binary, "precision": prec, "runtime": rt,
                 "median_reviews_per_s": med(v), "min": min(v), "max": max(v),
                 "spread_percent": round(spread(v), 1), "runs": v})
summary["replay"] = rows

# app tables
arows = []
for (binary, prec, rt), rs in sorted(app.items()):
    g = lambda f: [f(r) for r in rs]
    row = {"binary": binary, "precision": prec, "runtime": rt, "runs": len(rs), "reviews": rs[0]["reviews"],
           "rebuild_replay_rps": g(lambda r: r["rebuild"]["replay_reviews_per_s"]),
           "rebuild_total_rps": g(lambda r: r["rebuild"]["total_reviews_per_s"]),
           "rebuild_total_ms": g(lambda r: r["rebuild"]["total_ms"]),
           "checkpoint_ms": g(lambda r: r["rebuild"]["checkpoint_ms"]),
           "store_bytes": rs[0]["rebuild"]["store_bytes"]}
    for cat in ["review", "learning"]:
        for k in ["show_p50_us", "show_p95_us", "answer_p50_us", "answer_p95_us"]:
            row[f"{cat}_{k}"] = g(lambda r: r["latency"][cat][k])
    for b in ["32", "64", "128", "256", "512", "1024"]:
        row[f"sweep_{b}_qps"] = g(lambda r: r["sweep"][b]["queries_per_s"])
    arows.append(row)
summary["app"] = arows
json.dump(summary, open(f"{dest}/summary.json", "w"), indent=1)

with open(f"{dest}/replay.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["shape", "binary", "precision", "runtime", "median_reviews_per_s", "min", "max", "spread_percent", "runs"])
    for r in rows:
        w.writerow([r["shape"], r["binary"], r["precision"], r["runtime"], r["median_reviews_per_s"], r["min"], r["max"],
                    r["spread_percent"], " ".join(map(str, r["runs"]))])
with open(f"{dest}/app.csv", "w", newline="") as f:
    keys = [k for k in arows[0].keys()]
    w = csv.writer(f)
    w.writerow(keys)
    for r in arows:
        w.writerow([(" ".join(f"{x:g}" for x in r[k]) if isinstance(r[k], list) else r[k]) for k in keys])


# ---- printed tables (markdown fragments)
def pct(a, b):
    return (a / b - 1) * 100


A = {(r["binary"], r["precision"], r["runtime"]): r for r in arows}
R = {(r["shape"], r["binary"], r["precision"], r["runtime"]): r for r in rows}

print("loads", summary["loads_percent"])
print("\n## Rebuild (app path, pairs, all threads)")
print("| Size | Format | before: reviews/s (spread) | after: reviews/s (spread) | after vs before | after vs int8 rows (after) | total incl. load+checkpoints: after |")
for size in ["big-pair", "small-pair", "tiny-pair"]:
    base = med(A[("after", "compact_cards_int8", size)]["rebuild_replay_rps"])
    for p in FORMATS:
        b = A.get(("before", p, size)); a = A.get(("after", p, size))
        bs = f"{med(b['rebuild_replay_rps']):,.0f} ({spread(b['rebuild_replay_rps']):.1f}%)" if b else "-"
        as_ = f"{med(a['rebuild_replay_rps']):,.0f} ({spread(a['rebuild_replay_rps']):.1f}%)" if a else "-"
        ab = f"{pct(med(a['rebuild_replay_rps']), med(b['rebuild_replay_rps'])):+.1f}%" if a and b else "-"
        ai = f"{pct(med(a['rebuild_replay_rps']), base):+.1f}%" if a else "-"
        tot = f"{med(a['rebuild_total_rps']):,.0f} ({med(a['rebuild_total_ms'])/1000:.2f} s)" if a else "-"
        print(f"| {size} | {SHORT[p]} | {bs} | {as_} | {ab} | {ai} | {tot} |")

for shape in ["one-thread 1000x24", "one-thread 4000x8", "four-thread 4000x8"]:
    print(f"\n## Replay {shape}")
    print("| Runtime | Format | before | after | after vs before | after vs int8 rows (after) |")
    rts = sorted({k[3] for k in R if k[0] == shape}, key=lambda x: ["big", "big-pair", "small", "small-pair", "tiny", "tiny-pair"].index(x))
    for rt in rts:
        base = R[(shape, "after", "compact_cards_int8", rt)]["median_reviews_per_s"]
        for p in FORMATS:
            b = R.get((shape, "before", p, rt)); a = R.get((shape, "after", p, rt))
            f = lambda r: f"{r['median_reviews_per_s']:,.0f} ({r['spread_percent']:.1f}%)" if r else "-"
            ab = f"{pct(a['median_reviews_per_s'], b['median_reviews_per_s']):+.1f}%" if a and b else "-"
            ai = f"{pct(a['median_reviews_per_s'], base):+.1f}%" if a else "-"
            print(f"| {rt} | {SHORT[p]} | {f(b)} | {f(a)} | {ab} | {ai} |")
    if shape.startswith("four"):
        print("\nDeck Options savings vs Big (replay time, same format and layout):")
        for binary in ["before", "after"]:
            for p in FORMATS:
                out = []
                for lay, big in [("pair", "big-pair"), ("alone", "big")]:
                    if (shape, binary, p, big) not in R:
                        continue
                    bt = 1 / R[(shape, binary, p, big)]["median_reviews_per_s"]
                    for s in ["small", "tiny"]:
                        rt = s + ("-pair" if lay == "pair" else "")
                        st_ = 1 / R[(shape, binary, p, rt)]["median_reviews_per_s"]
                        out.append(f"{s} {lay} {(1 - st_ / bt) * 100:.0f}%")
                if out:
                    print(f"  {binary} {SHORT[p]}: " + ", ".join(out))

print("\n## Latency (pairs, median over runs of per-run p50 / p95, µs)")
print("| Size | Format | binary | Review show p50/p95 | Review answer p50/p95 | Learning show p50/p95 | Learning answer p50/p95 |")
for size in ["big-pair", "small-pair", "tiny-pair"]:
    for p in FORMATS:
        for binary in ["before", "after"]:
            a = A.get((binary, p, size))
            if not a:
                continue
            c = lambda k: f"{med(a[k + '_p50_us']):.0f} / {med(a[k + '_p95_us']):.0f}"
            print(f"| {size} | {SHORT[p]} | {binary} | {c('review_show')} | {c('review_answer')} | {c('learning_show')} | {c('learning_answer')} |")

print("\n## Batch sweep (queries/s, median of runs)")
print("| Size | Format | binary | 32 | 64 | 128 | 256 | 512 | 1024 |")
for size in ["big-pair", "small-pair", "tiny-pair"]:
    for p in FORMATS:
        for binary in ["before", "after"]:
            a = A.get((binary, p, size))
            if not a:
                continue
            print(f"| {size} | {SHORT[p]} | {binary} | " + " | ".join(f"{med(a[f'sweep_{b}_qps']):,.0f}" for b in ["32", "64", "128", "256", "512", "1024"]) + " |")
