"""The grade order by construction (SM95): check of the GPU scorer smoke (2 users, reference checkpoint, w8+s8).

The raw label must reproduce the reference's raw rows (the SM77 scoring) and the rectified label must keep the instant
head (probe rows are skip rows without labels, so only the curve probability may change) and score the same reviews;
the curve LogLoss differs. Prints one line starting with [sm95-smoke] PASS or FAIL, plus reviews/s.

  python sm95_smoke.py --rows gpu/smoke-rect.jsonl --ref-rows <sm77 rows>
"""
import argparse
import json


def rows(path, label):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("label") == label and "error" not in r:
            out[r["user_id"]] = r
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", required=True)
    p.add_argument("--ref-rows", required=True)
    args = p.parse_args()
    raw, rect = rows(args.rows, "sm77-ref-e21-q32k/w8+s8"), rows(args.rows, "sm77-ref-e21-q32k-rect/w8+s8")
    ref = rows(args.ref_rows, "sm77-ref-e21-q32k/w8+s8")
    users = sorted(raw)
    ok = len(users) == 2 and set(rect) == set(raw) and all(u in ref for u in users)
    detail = []
    for u in users if ok else []:
        a, b, c = raw[u]["benchmark_existing"], rect[u]["benchmark_existing"], ref[u]["benchmark_existing"]
        same_ref = all(abs(a[h]["LogLoss"] - c[h]["LogLoss"]) <= 1e-5 for h in ("ahead", "instant"))
        inst = abs(a["instant"]["LogLoss"] - b["instant"]["LogLoss"]) <= 1e-4 and a["ahead"]["size"] == b["ahead"]["size"]
        curve_diff = a["ahead"]["LogLoss"] != b["ahead"]["LogLoss"]
        ok &= same_ref and inst and rect[u].get("grade_rectify") is True and curve_diff
        detail.append(f"user {u} ({raw[u]['reviews']} reviews): curve raw {a['ahead']['LogLoss']:.6f} rect {b['ahead']['LogLoss']:.6f} "
                      f"(ref rows {c['ahead']['LogLoss']:.6f}), instant raw {a['instant']['LogLoss']:.6f} rect {b['instant']['LogLoss']:.6f}; "
                      f"{raw[u]['reviews'] / raw[u]['model_seconds']:,.0f} / {rect[u]['reviews'] / rect[u]['model_seconds']:,.0f} reviews/s (raw / rect)")
    print("[sm95-smoke] " + ("PASS " if ok else "FAIL ") + "; ".join(detail) if detail else f"[sm95-smoke] FAIL: users {users}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
