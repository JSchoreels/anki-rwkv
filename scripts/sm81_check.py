"""Scheduling-time vs review-time accuracy (SM81): CPU proof of the start-of-day scorer option.

  sm81_check.py compare --orig A.jsonl --off B.jsonl --sod C.jsonl [--dump DIR] --out check.json

(a) off: the patched scorer without --start-of-day writes the same rows as the scorer before the change (every field
    except timings and provenance), and with --start-of-day the normal-pass fields are still the same;
(b) start of day == review time on the user's first review of each day, and differs on the other reviews;
(c) the curve head of the start-of-day pass equals the normal pass (query rows commit no state).
"Equal" in (b) and (c) is up to float rounding (--tol): moving rows changes their positions in the batched matmuls, so
fp32 agrees to ~1-2 ulp (tol 1e-6), and int8 state (+s8) can flip a state rounding step (tol 1e-3, LogLoss within
2e-6); the other reviews of a day differ by far more (median and p90 |difference| are reported).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

VOLATILE = {"prepare_seconds", "model_seconds", "provenance_commit", "provenance_dirty", "start_of_day"}


def rows(path):
    return {(r["user_id"], r["label"]): r for r in map(json.loads, Path(path).read_text().splitlines())}


def stable(row):
    return {k: v for k, v in row.items() if k not in VOLATILE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compare"])
    ap.add_argument("--orig", required=True)
    ap.add_argument("--off", required=True)
    ap.add_argument("--sod", required=True)
    ap.add_argument("--dump", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1e-6, help="max |difference| counted as equal (float rounding)")
    a = ap.parse_args()
    orig, off, sod = rows(a.orig), rows(a.off), rows(a.sod)
    fail, users = [], []
    if set(orig) != set(off) or set(orig) != set(sod):
        fail.append(f"row keys differ: {sorted(orig)} / {sorted(off)} / {sorted(sod)}")
    for key in sorted(orig):
        o = stable(orig[key])
        if key in off and stable(off[key]) != o:
            fail.append(f"{key}: off rows differ from the scorer before the change")
        if key not in sod:
            continue
        if stable(sod[key]) != o:
            fail.append(f"{key}: the normal pass of the --start-of-day run differs")
        s = sod[key]["start_of_day"]
        u = {"user": key[0], "label": key[1], **{k: v for k, v in s.items() if k != "benchmark"},
             "logloss_review_time": o["benchmark"]["instant"]["LogLoss"],
             "logloss_start_of_day": s["benchmark"]["instant"]["LogLoss"],
             "logloss_scheduling_time": o["benchmark"]["ahead"]["LogLoss"],
             "sod_pass_curve_logloss": s["benchmark"]["ahead"]["LogLoss"]}
        if not (s["curve_keys_equal"] and s["curve_max_abs_diff"] <= a.tol
                and abs(u["sod_pass_curve_logloss"] - u["logloss_scheduling_time"]) <= 2e-6):
            fail.append(f"{key}: start-of-day pass curve differs (max abs {s['curve_max_abs_diff']})")
        if not s.get("instant_keys_equal"):
            fail.append(f"{key}: instant keys differ")
        else:
            if s["first_of_day_max_abs_diff"] > a.tol:
                fail.append(f"{key}: start of day != review time on {s['first_of_day_reviews'] - s['first_of_day_equal']}"
                            f" first-of-day reviews (max abs {s['first_of_day_max_abs_diff']})")
            if s["other_reviews"] and s["other_equal"] > 0.01 * s["other_reviews"]:   # bitwise equal
                fail.append(f"{key}: start of day == review time on {s['other_equal']} of {s['other_reviews']} other reviews")
        if a.dump:
            f = Path(a.dump) / f"{key[0]}_{key[1].replace('/', '_')}.npz"
            d = np.load(f)
            u["dump_reviews"] = int(len(d["review_th"]))
            u["dump_curve_sod_bitwise_equal"] = bool(d["curve_sod_equal"])
            y = d["y"].astype(float)
            ll = lambda p: float(-np.mean(y * np.log(np.clip(p.astype(np.float64), 1e-5, 1 - 1e-5)) + (1 - y) * np.log(np.clip(1 - p.astype(np.float64), 1e-5, 1 - 1e-5))))   # noqa: E731
            u["dump_logloss"] = {"curve": ll(d["p_curve"]), "instant": ll(d["p_instant"]), "sod": ll(d["p_sod"])}
            diff = np.abs(d["p_sod"].astype(np.float64) - d["p_instant"])
            u["sod_vs_review_abs_diff"] = {"median": float(np.median(diff)), "p90": float(np.quantile(diff, 0.9))}
        users.append(u)
    out = {"ok": not fail, "tol": a.tol, "failures": fail, "users": users}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for u in users:
        print(f"[sm81-check] user {u['user']} {u['label']}: first-of-day {u.get('first_of_day_equal')}/"
              f"{u.get('first_of_day_reviews')} bitwise equal (max abs {u.get('first_of_day_max_abs_diff')}), other {u.get('other_equal')}/{u.get('other_reviews')} equal, "
              f"curve diff {u['curve_max_abs_diff']}, leading-query sequences {u['leading_query_sequences']}; LogLoss "
              f"sched {u['logloss_scheduling_time']:.5f} sod {u['logloss_start_of_day']:.5f} review {u['logloss_review_time']:.5f}")
    print("[sm81-check] " + ("PASS" if not fail else "FAIL: " + "; ".join(fail)))
    sys.exit(0 if not fail else 1)


if __name__ == "__main__":
    main()
