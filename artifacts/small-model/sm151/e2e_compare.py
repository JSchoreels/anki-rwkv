"""Low-rank states in the fork (SM151): per-user LogLoss of the fork runtime vs the scorer and vs the SM148 rows.

    python e2e_compare.py <vectors.json> <rust-out.json> <sm148 tag> <rows.jsonl>... > summary.json

Curve LogLoss is the raw forgetting curve (the fork ignores the big model's ahead residual), on the scorer's curve
rows (non-query rows with a label); instant LogLoss is 1 - P(again) on the query rows with a label (the scorer's
`curve_raw_log_loss` and `instant_log_loss`). Only aggregates are printed (no rows).
"""

import json
import sys

import numpy as np

FORMATS = ["fp32", "compact_int8", "t24", "q2"]
SM148_LABEL = {"compact_int8": "w8+s8hx8", "t24": "w8+lr2.16.4q8x8", "q2": "w8+lr2q8x8"}


def bce(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def main():
    vectors = json.load(open(sys.argv[1], encoding="utf-8"))
    rust = json.load(open(sys.argv[2], encoding="utf-8"))
    tag = sys.argv[3]
    sm148 = {}
    for path in sys.argv[4:]:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            if row.get("tag") == tag and row.get("inputs") == "e2s-query":
                sm148[(row["user_id"], row["variant"])] = row
    users = []
    for user in vectors:
        labels = np.array([r["label"] for r in user["rows"]])
        y, has, query = labels[:, 0], labels[:, 1] > 0, labels[:, 2] > 0
        curve, inst = has & ~query, has & query
        result = {"user": user["user"], "curve_labels": int(curve.sum()), "instant_labels": int(inst.sum())}
        for fmt in FORMATS:
            scorer = np.array([r[fmt] for r in user["rows"]])
            ours = np.array(rust[user["name"]][fmt])
            result[fmt] = {
                "scorer_curve": bce(scorer[curve, 1], y[curve]),
                "scorer_curve_residual": bce(scorer[curve, 2], y[curve]),
                "scorer_instant": bce(scorer[inst, 0], y[inst]),
                "rust_curve": bce(ours[curve, 1], y[curve]),
                "rust_instant": bce(ours[inst, 0], y[inst]),
                "max_abs_row_curve": float(np.abs(ours[curve, 1] - scorer[curve, 1]).max()),
                "max_abs_row_instant": float(np.abs(ours[inst, 0] - scorer[inst, 0]).max()),
                "mean_abs_row": float(np.abs(ours[:, :2] - scorer[:, :2]).mean()),
            }
            if fmt in SM148_LABEL and (user["user"], SM148_LABEL[fmt]) in sm148:
                row = sm148[(user["user"], SM148_LABEL[fmt])]
                result[fmt]["sm148_curve"] = row["curve_raw_log_loss"]
                result[fmt]["sm148_curve_residual"] = row["curve_log_loss"]
                result[fmt]["sm148_instant"] = row["instant_log_loss"]
                result[fmt]["sm148_curve_labels"] = row["curve_labels"]
        users.append(result)
    gates = {}
    for fmt in FORMATS:
        for head in ("curve", "instant"):
            d = [u[fmt][f"rust_{head}"] - u[fmt][f"scorer_{head}"] for u in users]
            gates[f"{fmt} rust-scorer {head}"] = {"max_abs": max(map(abs, d)), "mean_signed": float(np.mean(d)),
                                                 "pass": bool(max(map(abs, d)) <= 3e-4 and abs(np.mean(d)) <= 1e-4)}
    for fmt in ("t24", "q2"):
        for head in ("curve", "instant"):
            d = []
            for u in users:
                if "sm148_curve" not in u[fmt] or "sm148_curve" not in u["compact_int8"]:
                    continue
                rust_delta = u[fmt][f"rust_{head}"] - u["compact_int8"][f"rust_{head}"]
                sm148_delta = u[fmt][f"sm148_{head}"] - u["compact_int8"][f"sm148_{head}"]
                u[fmt][f"delta_{head}"] = {"rust": rust_delta, "sm148": sm148_delta}
                d.append(rust_delta - sm148_delta)
            if d:
                gates[f"{fmt}-int8 rust vs sm148 {head}"] = {
                    "max_abs": max(map(abs, d)), "mean_signed": float(np.mean(d)),
                    "pass": bool(max(map(abs, d)) <= 3e-4 and abs(np.mean(d)) <= 1e-4)}
    json.dump({"tag": tag, "users": users, "gates": gates}, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
