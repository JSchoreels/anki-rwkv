"""SM58 follow-up 2 (user 2026-10-23): percentiles p50 / p90 / p95 / p99 of |R_small - R_big| per review (w8+s8), plus the
mean signed difference (bias), per model / head / sample. Reuses sm58_disagreement_size.py's loaders on the same SM58 dumps
(722 benchmark users) and the own collection (aggregate only).
  python sm58_disagreement_pct.py  -> disagreement-pct.{json,md}"""
import json
from pathlib import Path

import numpy as np

import sm58_disagreement_size as S

HERE = Path(__file__).resolve().parent
S.PCTS = (50, 90, 95, 99)
_stats = S.stats
S.stats = lambda x: {**_stats(x), "mean": float(np.mean(x)), "mean_abs": float(np.mean(np.abs(x)))}


def main():
    users = json.loads((HERE / "users.json").read_text())["users"]
    big_dirs = [str(S.U.SM24_DUMP), str(HERE / "dump")]
    res = {}
    for tag in ("sm36", "sm55"):
        n, b = S.benchmark(tag, users, big_dirs)
        res[tag] = {"benchmark_users": n, "benchmark": b, "collection": S.collection(tag)}
        print(tag, n, "users", flush=True)
    (HERE / "disagreement-pct.json").write_text(json.dumps(res, indent=1) + "\n")
    names = {"sm36": "Muon version (SM36)", "sm55": "8k model (SM55)"}
    lines = ["# SM58 follow-up 2: percentiles of the R difference between the small and big model", "",
             "x = R_small - R_big per review (w8+s8, R = predicted recall probability). Percentiles are of |x| (0.01 = 1 point); "
             "mean x > 0 = the small model predicts higher recall. Own collection: aggregate only.", ""]
    for where, title in (("benchmark", f"Benchmark users ({res['sm36']['benchmark_users']} users)"), ("collection", "Own collection")):
        lines += [f"## {title}", "", "| Model | Head | Sample | Reviews | p50 | p90 | p95 | p99 | mean abs | mean x |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for tag in res:
            for key, s in res[tag][where].items():
                h, smp = key.split("/")
                lines.append(f"| {names[tag]} | {h} | {smp} | {s['reviews']:,} | {s['p50']:.4f} | {s['p90']:.4f} | {s['p95']:.4f} | "
                             f"{s['p99']:.4f} | {s['mean_abs']:.4f} | {s['mean']:+.4f} |")
        lines.append("")
    (HERE / "disagreement-pct.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
