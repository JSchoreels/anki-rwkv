"""S05 check: batched full-history scoring rows (small_model_heldout.py --batch-users N) against one-user rows of the
same label, with the AGENT.md "Evaluation tolerance" thresholds (scripts/check_fast_evaluation.py TOLERANCE).

Passes when, on the same users: input_sha256 equal per user (when both sides recorded it, --input-sha; reference rows
written before S05 have none, which needs --allow-missing-input-sha and is reported), label counts, benchmark sizes
and coverage identical; every aggregate (the four R01 losses label-weighted and user-averaged; benchmark LogLoss,
RMSE(bins), AUC, RMSE per head, review-weighted and user-averaged) within 1e-6; every per-user loss within 1e-5;
per-user AUC within 1e-4. Also reports model seconds (sum over users) on both sides.

usage: small_model_batch_compare.py --reference heldout.jsonl --candidate s05-check.jsonl --label TAG/VARIANT
       [--reference-label L] [--out result.json] [--allow-missing-input-sha]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_fast_evaluation import COUNT_FIELDS, LOSS_FIELDS, TOLERANCE, _flatten  # noqa: E402

R01_LOSSES = {"curve_log_loss": "curve_labels", "curve_raw_log_loss": "curve_labels",
              "instant_log_loss": "instant_labels", "instant_rating_cross_entropy": "instant_labels"}
BENCH = [(h, m) for h in ("ahead", "instant") for m in ("LogLoss", "RMSE(bins)", "AUC", "RMSE")]
AUC_FIELDS = ("benchmark.ahead.AUC", "benchmark.instant.AUC")


def load(path, label):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("label") == label and "error" not in row:
            rows[row["user_id"]] = row  # last row per user wins (resumed files)
    return rows


def aggregates(rows):
    out = {}
    for loss, count in R01_LOSSES.items():
        n = sum(r[count] for r in rows)
        out[f"{loss}/label_weighted"] = sum(r[loss] * r[count] for r in rows) / n
        out[f"{loss}/user_averaged"] = sum(r[loss] for r in rows) / len(rows)
    for head, metric in BENCH:
        kept = [r["benchmark"][head] for r in rows if head in r["benchmark"]]
        size = sum(b["size"] for b in kept)
        out[f"benchmark.{head}.{metric}/review_weighted"] = sum(b[metric] * b["size"] for b in kept) / size
        out[f"benchmark.{head}.{metric}/user_averaged"] = sum(b[metric] for b in kept) / len(kept)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--label", required=True, help="candidate label (TAG/VARIANT)")
    parser.add_argument("--reference-label", default="", help="reference label if different (default: --label)")
    parser.add_argument("--allow-missing-input-sha", action="store_true",
                        help="reference rows predate input_sha256; input identity then rests on the CPU check")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    ref = load(args.reference, args.reference_label or args.label)
    cand = load(args.candidate, args.label)
    users = sorted(set(ref) & set(cand))
    missing = sorted(set(cand) - set(ref))
    if not users:
        raise SystemExit("no common users")
    a = [ref[u] for u in users]
    b = [cand[u] for u in users]
    stats = {}
    for field in LOSS_FIELDS + list(AUC_FIELDS) + COUNT_FIELDS:
        deltas, exact = [], 0
        for x, y in zip(a, b):
            fx, fy = _flatten(x).get(field), _flatten(y).get(field)
            if fx is None or fy is None:
                continue
            deltas.append(abs(fx - fy))
            exact += fx == fy
        stats[field] = {"values": len(deltas), "exact": exact, "max_abs": max(deltas, default=0.0)}
    agg_a, agg_b = aggregates(a), aggregates(b)
    agg = {k: {"reference": agg_a[k], "candidate": agg_b[k], "abs": abs(agg_a[k] - agg_b[k])} for k in agg_a}
    shas = [(x.get("input_sha256"), y.get("input_sha256")) for x, y in zip(a, b)]
    if all(p and q for p, q in shas):
        input_equal = all(p == q for p, q in shas)
    else:
        input_equal = None
    counts_exact = all(s["exact"] == s["values"] for f, s in stats.items() if f in COUNT_FIELDS)
    reviews_equal = all(x["reviews"] == y["reviews"] for x, y in zip(a, b))
    max_aggregate = max(v["abs"] for v in agg.values())
    max_loss = max(stats[f]["max_abs"] for f in LOSS_FIELDS)
    max_auc = max(stats[f]["max_abs"] for f in AUC_FIELDS)
    input_ok = input_equal is True or (input_equal is None and args.allow_missing_input_sha)
    seconds = (sum(x["model_seconds"] for x in a), sum(y["model_seconds"] for y in b))
    reviews = sum(x["reviews"] for x in a)
    result = {
        "reference": args.reference, "candidate": args.candidate, "label": args.label,
        "reference_label": args.reference_label or args.label, "users": len(users),
        "candidate_users_missing_in_reference": missing,
        "candidate_batch_users": sorted({y.get("batch_users", 1) for y in b}),
        "candidate_fallbacks": sum("batch_fallback" in y for y in b),
        "tolerance": {"input_sha_equal": input_equal, "input_sha_missing_allowed": args.allow_missing_input_sha,
                      "counts_exact": counts_exact, "reviews_equal": reviews_equal,
                      "max_aggregate_abs": max_aggregate, "max_per_user_loss_abs": max_loss,
                      "max_per_user_auc_abs": max_auc, "thresholds": TOLERANCE},
        "timing": {"reference_model_seconds": round(seconds[0], 3), "candidate_model_seconds": round(seconds[1], 3),
                   "speedup": round(seconds[0] / max(seconds[1], 1e-9), 3), "reviews": reviews,
                   "reference_reviews_per_second": round(reviews / max(seconds[0], 1e-9)),
                   "candidate_reviews_per_second": round(reviews / max(seconds[1], 1e-9))},
        "per_user_fields": stats, "aggregates": agg,
    }
    result["tolerance"]["passes"] = bool(input_ok and counts_exact and reviews_equal and not missing
                                         and max_aggregate <= TOLERANCE["aggregate"]
                                         and max_loss <= TOLERANCE["per_user_loss"]
                                         and max_auc <= TOLERANCE["per_user_auc"])
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"users": len(users), **result["tolerance"], **result["timing"]}, indent=1))


if __name__ == "__main__":
    main()
