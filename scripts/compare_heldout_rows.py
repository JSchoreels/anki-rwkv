"""SM48: compare two small_model_heldout.py row files (same users and labels) for scorer A/B checks.

Per (user, label): the losses (curve/instant log loss, rating cross-entropy, label counts) and the benchmark dict
(size, coverage, LogLoss / RMSE(bins) / AUC / RMSE per head). Reports how many pairs are bitwise equal, the max |delta|
per field, and the evaluation-tolerance verdict of AGENT.md (per-user losses <= 1e-5, AUC <= 1e-4, counts/sizes equal)
plus the review-weighted benchmark LogLoss of each file (aggregate |delta| <= 1e-6). Exit 1 on a failed check.

usage: compare_heldout_rows.py A.jsonl B.jsonl [--out result.json] [--bitwise]
"""
import argparse
import json
import sys
from pathlib import Path

LOSSES = ("curve_log_loss", "curve_raw_log_loss", "instant_log_loss", "instant_rating_cross_entropy")
COUNTS = ("curve_labels", "instant_labels", "reviews")
BENCH = ("LogLoss", "RMSE(bins)", "RMSE", "AUC")


def load(path):
    rows = {}
    for row in map(json.loads, Path(path).read_text(encoding="utf-8").splitlines()):
        if "benchmark" in row:
            rows[(row["user_id"], row["label"])] = row
    return rows


def weighted_logloss(rows, head):
    pairs = [(r["benchmark"][head]["LogLoss"], r["benchmark"][head]["size"]) for r in rows if head in r["benchmark"]]
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total if total else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--out", default="")
    parser.add_argument("--bitwise", action="store_true", help="fail unless every compared value is bitwise equal")
    args = parser.parse_args()
    a, b = load(args.a), load(args.b)
    keys = sorted(set(a) & set(b))
    worst = {}
    bitwise = 0
    failures = []

    def note(name, x, y, tol):
        if x is None or y is None:
            ok = x is None and y is None
            delta = 0.0 if ok else float("inf")
        else:
            delta = abs(x - y)
            ok = delta <= tol
        worst[name] = max(worst.get(name, 0.0), delta)
        return ok, x == y

    for key in keys:
        ra, rb = a[key], b[key]
        same = True
        for name in COUNTS:
            ok, eq = note(name, ra.get(name), rb.get(name), 0)
            same &= eq
            if not ok:
                failures.append((key, name))
        for name in LOSSES:
            ok, eq = note(name, ra[name], rb[name], 1e-5)
            same &= eq
            if not ok:
                failures.append((key, name))
        ba, bb = ra["benchmark"], rb["benchmark"]
        for name in ("size", "ahead_coverage", "instant_coverage"):
            ok, eq = note(f"benchmark.{name}", ba.get(name), bb.get(name), 0)
            same &= eq
            if not ok:
                failures.append((key, name))
        for head in ("ahead", "instant"):
            if (head in ba) != (head in bb):
                failures.append((key, head))
                same = False
                continue
            if head not in ba:
                continue
            for name in BENCH:
                ok, eq = note(f"{head}.{name}", ba[head].get(name), bb[head].get(name), 1e-4 if name == "AUC" else 1e-5)
                same &= eq
                if not ok:
                    failures.append((key, f"{head}.{name}"))
        bitwise += same
    aggregate = {}
    for head in ("ahead", "instant"):
        wa = weighted_logloss([a[k] for k in keys], head)
        wb = weighted_logloss([b[k] for k in keys], head)
        aggregate[head] = {"a": wa, "b": wb, "abs_delta": abs(wa - wb) if wa is not None and wb is not None else None}
        if aggregate[head]["abs_delta"] is not None and aggregate[head]["abs_delta"] > 1e-6:
            failures.append(("aggregate", head))
    missing = {"only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a))}
    passed = bool(keys) and not failures and not missing["only_a"] and not missing["only_b"]
    if args.bitwise:
        passed &= bitwise == len(keys)
    result = {"a": args.a, "b": args.b, "pairs": len(keys), "bitwise_equal_pairs": bitwise, **missing,
              "max_abs_delta": worst, "weighted_benchmark_logloss": aggregate, "failures": failures[:20],
              "passed": passed}
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("pairs", "bitwise_equal_pairs", "only_a", "only_b", "passed")}),
          "max deltas:", {k: v for k, v in worst.items() if v})
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
