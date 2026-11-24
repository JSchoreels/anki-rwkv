"""Low-rank states in the fork (SM151), Small and Tiny: fork-runtime LogLoss deltas vs the SM148 rows.

    python e2e_compare_small.py <vectors.json> <rust-small.json> <rows.jsonl>... > summary.json

There is no per-row scorer output for the small engine's models here (their scorer path is the eager training model
with the state hooks); the check is the fork runtime's T24 (and Q2) minus compact int8 LogLoss per user against the
same difference in the low-rank int8 states experiment (SM148) rows, same users and show-time inputs. The fork runs
float32 weights; SM148 scored int8 weights (`w8`), so absolute values differ slightly and only deltas are gated.
Curve LogLoss on non-query labelled rows, instant (1 - P(again)) on query labelled rows. Aggregates only.
"""

import json
import sys

import numpy as np

TAGS = {"sm118": "sm118-e26-dec2ep32k-t91s2", "sm113": "sm113-inst-e35st-st-dec2ep32k-t99",
        "sm130th": "sm130-th-dec2ep32k-t91s2", "sm130io": "sm130-io-dec2ep32k-t99"}
SM148_LABEL = {"compact_int8": "w8+s8hx8", "t24": "w8+lr2.16.4q8x8", "q2": "w8+lr2q8x8"}


def bce(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def main():
    vectors = json.load(open(sys.argv[1], encoding="utf-8"))
    rust = json.load(open(sys.argv[2], encoding="utf-8"))
    sm148 = {}
    for path in sys.argv[3:]:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            if row.get("inputs") == "e2s-query":
                sm148[(row["tag"], row["user_id"], row["variant"])] = row
    out = {"packages": {}, "gates": {}}
    for package, by_user in rust.items():
        tag = TAGS[package]
        users = []
        for user in vectors:
            labels = np.array([r["label"] for r in user["rows"]])
            y, has, query = labels[:, 0], labels[:, 1] > 0, labels[:, 2] > 0
            masks = {"curve": has & ~query, "instant": has & query}
            result = {"user": user["user"]}
            for fmt in ("fp32", "compact_int8", "t24", "q2"):
                ours = np.array(by_user[user["name"]][fmt])
                result[fmt] = {"rust_curve": bce(ours[masks["curve"], 1], y[masks["curve"]]),
                               "rust_instant": bce(ours[masks["instant"], 0], y[masks["instant"]])}
                key = (tag, user["user"], SM148_LABEL.get(fmt))
                if key in sm148:
                    result[fmt]["sm148_curve"] = sm148[key]["curve_raw_log_loss"]
                    result[fmt]["sm148_instant"] = sm148[key]["instant_log_loss"]
            users.append(result)
        out["packages"][package] = users
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
                    out["gates"][f"{package} {fmt}-int8 rust vs sm148 {head}"] = {
                        "max_abs": max(map(abs, d)), "mean_signed": float(np.mean(d)),
                        "pass": bool(max(map(abs, d)) <= 3e-4 and abs(np.mean(d)) <= 1e-4)}
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
