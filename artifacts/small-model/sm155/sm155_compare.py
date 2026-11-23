"""Bounded-memory long-card scorer (SM155): exactness of the stream engine against the stock CPU scorer.

usage: sm155_compare.py DIR [DIR ...] [--json OUT]
Each DIR holds parallel.jsonl / stream.jsonl (scorer rows) and dump-parallel / dump-stream (--dump-scope all: every
predicted review per head, float32). Per (user, label): review keys equal per head, max |dp| per head (curve =
ahead, instant), benchmark LogLoss / RMSE(bins) / AUC differences, model seconds, and the engine's storage notes.
Rule (design.md, pre-registered): max |dp| <= 1e-5 on every review of both heads, keys equal. Aggregates and per-user
maxima only (benchmark users).
"""
import json
import sys
from pathlib import Path

import numpy as np

TOL = 1e-5


def rows(path):
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if "benchmark" in r:
                    out[(r["user_id"], r["label"])] = r
    return out


def compare(folder):
    folder = Path(folder)
    old, new = rows(folder / "parallel.jsonl"), rows(folder / "stream.jsonl")
    result = []
    for key in sorted(set(old) & set(new)):
        user, label = key
        name = f"{user}_{label.replace('/', '_')}.npz"
        a, b = np.load(folder / "dump-parallel" / name), np.load(folder / "dump-stream" / name)
        item = {"user": user, "label": label, "reviews": old[key]["reviews"]}
        for head in ("ahead", "instant"):
            ka, kb = a[f"{head}_th"], b[f"{head}_th"]
            same = bool(np.array_equal(ka, kb))
            item[f"{head}_keys_equal"] = same
            item[f"{head}_n"] = int(len(ka))
            if same and len(ka):
                d = np.abs(a[f"{head}_p"].astype(np.float64) - b[f"{head}_p"].astype(np.float64))
                item[f"{head}_max_abs"] = float(d.max())
                item[f"{head}_bitwise_share"] = float((d == 0).mean())
        bo, bn = old[key]["benchmark"], new[key]["benchmark"]
        item["benchmark_max_abs"] = max((abs(bo[k] - bn[k]) for k in bo if isinstance(bo.get(k), (int, float))
                                         and isinstance(bn.get(k), (int, float))), default=0.0)
        item["seconds_parallel"] = old[key]["model_seconds"]
        item["seconds_stream"] = new[key]["model_seconds"]
        item["stream_engine"] = new[key].get("stream_engine", {}).get("state_tables")
        worst = max(item.get("ahead_max_abs", 0.0), item.get("instant_max_abs", 0.0))
        item["max_abs"] = worst
        item["pass"] = bool(item["ahead_keys_equal"] and item["instant_keys_equal"] and worst <= TOL)
        result.append(item)
    return result


def main():
    args = sys.argv[1:]
    out = None
    if "--json" in args:
        i = args.index("--json")
        out = args[i + 1]
        args = args[:i] + args[i + 2:]
    allrows = []
    for folder in args:
        for item in compare(folder):
            item["dir"] = Path(folder).name
            allrows.append(item)
            print(f"{item['dir']:>14} {item['user']:>5} {item['label']:<44} n={item['reviews']:>6} "
                  f"max|dp| curve {item.get('ahead_max_abs', float('nan')):.2e} instant "
                  f"{item.get('instant_max_abs', float('nan')):.2e} bitwise {item.get('ahead_bitwise_share', 0):.3f}/"
                  f"{item.get('instant_bitwise_share', 0):.3f} bench {item['benchmark_max_abs']:.1e} "
                  f"s {item['seconds_parallel']:.0f}/{item['seconds_stream']:.0f} {'PASS' if item['pass'] else 'FAIL'}")
    print(f"{sum(r['pass'] for r in allrows)}/{len(allrows)} pass; worst max|dp| "
          f"{max((r['max_abs'] for r in allrows), default=0):.2e}")
    if out:
        Path(out).write_text(json.dumps(allrows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
