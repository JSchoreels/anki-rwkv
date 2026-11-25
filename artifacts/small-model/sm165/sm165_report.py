#!/usr/bin/env python
"""Big long cards with 6-bit factors (SM165): long-card read-out of F6 (w8+lr2.16.4q6x8) for Big two-head (SM91s2) and
Big Instant-Only (SM99) on the 50 long-card users, against int8 REF (w8+s8hx8) and T24 (w8+lr2.16.4q8x8).

Same statistic and inputs as the bounded-memory long-card scorer (SM155) first-use read-out (sm155_firstuse_report.py):
sm148_report.py of wt/sm148 (imported, read-only), REF / T24 rows of SM148 (37 users) + SM155 first use (13 users).
% = 100 x (sum LogLoss new / sum LogLoss base - 1) per bucket of answers before the review, paired user bootstrap 2,000
draws, seed 0, 95% CI. Aggregates only.

  sm165_report.py gate-reuse   gate: T24 stream rows of cpu/gate equal the reused T24 rows (benchmark, 1e-6)
  sm165_report.py selftest     T24 vs REF reproduces SM155's 50-user numbers; the code path with F6 := T24 gives 0
  sm165_report.py report       -> sm165-result.json / sm165-result.md; exit 0 decided, 2 pending (coverage < 48)
"""
import json, sys
from pathlib import Path
sys.dont_write_bytecode = True
S148 = Path("C:/Users/jonat/workspace/wt/sm148/artifacts/small-model/sm148")
sys.path.insert(0, str(S148))
import sm148_report as R

HERE = Path(__file__).resolve().parent
S155 = HERE.parent / "sm155" / "firstuse"
OLD = S148 / "cpu/sm148-longcard-rows.jsonl"
FIRST = S155 / "sm155-big-longcard-rows.jsonl"
S155_RES = S155 / "sm155-firstuse-result.json"
NEW = HERE / "cpu/sm165-f6-rows.jsonl"
GATE = HERE / "cpu/gate/stream.jsonl"
PIN = {OLD: "feb431e65c9a7292e3cd1029636b329a61d5d146a2181b831fc25e5619d6431f",
       S148 / "sm148_report.py": "2abf58383b0875e9a722be992429dac5d20082740728793427fd1cdbadf52af8"}
LAB = {"REF": "w8+s8hx8", "T24": "w8+lr2.16.4q8x8", "F6": "w8+lr2.16.4q6x8"}
BK = [(1, 3), (3, 5), (5, 9), (9, 16), (16, 32), (32, 128), (128, None)]
HEADS = {"sm91s2": ("Big two-head (SM91s2)", ("curve", "instant")), "sm99": ("Big Instant-Only (SM99)", ("instant",))}
PAIRS = (("F6", "REF"), ("F6", "T24"), ("T24", "REF"))
MIN_LONG = 48
LONG_POINT, LONG_UPPER = 0.5, 1.0   # user 11-22 long-card tolerance vs int8 (warning only)


def bname(lo, hi):
    return f"{lo}-{hi - 1 if hi else ''}"


def load(f6_path):
    for p, h in PIN.items():
        assert R.sha(p) == h, f"pinned file changed: {p}"
    old, _, _ = R.read_rows(OLD)
    first, _, _ = R.read_rows(FIRST)
    new, nf, torn = R.read_rows(f6_path) if f6_path else ({}, {}, 0)
    rows = {L: dict(old.get(L, {})) for L in set(old) | set(first) | set(new)}
    for L, us in first.items():
        for u, r in us.items():
            assert u not in old.get(L, {}), (L, u)
            rows[L][u] = r
    for L, us in new.items():
        assert L not in old and L not in first, L   # only F6 labels are new
        rows[L] = dict(us)
    return rows, nf, torn


def readout(rows, users, f6_label_of=None):
    res = {"coverage": {}, "cost": {}}
    for key, (name, hs) in HEADS.items():
        tag = R.MODELS[key][0]
        lab = {k: f"{tag}/{v}{R.E2S}" for k, v in LAB.items()}
        if f6_label_of:
            lab["F6"] = lab[f6_label_of]
        sel = {k: R.sub(rows, L, users) for k, L in lab.items()}
        for k in LAB:
            res["coverage"][f"{name} {k}"] = len(sel[k])
        for new, base in PAIRS:
            for h in hs:
                d = {bname(lo, hi): R.bucket_cmp(sel[new], sel[base], R.KEY[h], lo, hi) for lo, hi in BK}
                d["users"] = len(set(sel[new]) & set(sel[base]))
                res["cost"].setdefault(f"{name} | {h}", {})[f"{new} vs {base}"] = d
    return res


def selfcheck(res):
    """T24 vs REF must equal SM155's 50-user ('after') numbers exactly."""
    ref = json.loads(S155_RES.read_text())["cost"]
    bad = []
    for k, v in ref.items():
        for b, c in v["after"].items():
            if b == "users" or b == "0-0":
                continue
            mine = res["cost"][k]["T24 vs REF"].get(b)
            if (c is None) != (mine is None) or (c and any(abs(c[f] - mine[f]) > 1e-12 for f in ("value",)) or
                                                 c and (c["ci"] != mine["ci"] or c["reviews"] != mine["reviews"])):
                bad.append((k, b, c, mine))
    return bad


def gate_reuse():
    gate, _, _ = R.read_rows(GATE)
    old, _, _ = R.read_rows(OLD)
    g155, _, _ = R.read_rows(S155 / "gate/stream.jsonl")
    L = f"sm91s2-big-showtime/{LAB['T24']}{R.E2S}"
    worst, n = 0.0, 0
    for u, src in ((2388, g155), (2883, old)):
        a, b = gate[L][u]["benchmark"], src[L][u]["benchmark"]
        for h in ("ahead", "instant"):
            for m in ("LogLoss", "RMSE(bins)", "AUC"):
                worst = max(worst, abs(a[h][m] - b[h][m])); n += 1
    ok = worst <= 1e-6
    print(f"gate reuse: T24 stream rows vs reused rows, {n} metrics, worst {worst:.1e}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def write_md(res, path):
    out = ["# Big long cards with 6-bit factors (SM165): read-out", "",
           "Coverage (users of 50): " + "; ".join(f"{k} {v}" for k, v in res["coverage"].items()), "",
           "% = 100 x (LogLoss new / LogLoss base - 1), review-weighted, paired user bootstrap 2,000 draws, 95% CI. "
           "Buckets: answers of the card before the review. Tolerance on 128+ vs int8 (warning only): point 0.5%, CI upper 1%.", ""]
    for k, v in res["cost"].items():
        out += [f"## {k}", "", "| answers before | " + " | ".join(f"{p} ({v[p]['users']} users)" for p in v) + " |",
                "|---|" + "---|" * len(v)]
        for lo, hi in BK:
            b = bname(lo, hi)
            out.append(f"| {b} | " + " | ".join(R.fmt(v[p][b]) for p in v) + " |")
        out.append("")
    out += ["## Tolerance (128+ cards, F6 vs REF)", ""] + [f"- {w}" for w in res["tolerance"]] + [""]
    path.write_text("\n".join(out), encoding="utf-8")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    longusers = set(json.loads((S148 / "users-longcard.json").read_text())["users"])
    if mode == "gate-reuse":
        return gate_reuse()
    if mode == "selftest":
        rows, _, _ = load(None)
        res = readout(rows, longusers, f6_label_of="T24")
        bad = selfcheck(res)
        zero = all(c is None or (c["value"] == 0 and c["ci"] == [0, 0]) for v in res["cost"].values()
                   for b, c in v["F6 vs T24"].items() if b != "users")
        print(f"selftest: T24 vs REF vs SM155 mismatches {len(bad)}; F6:=T24 gives zero cost: {zero}")
        for x in bad[:5]:
            print("  ", x)
        return 0 if not bad and zero else 1
    rows, nf, torn = load(NEW)
    res = readout(rows, longusers)
    bad = selfcheck(res)
    res["selfcheck_t24_vs_ref_mismatches"] = len(bad)
    res["f6_nonfinite"] = {L: sorted(us) for L, us in nf.items()}
    res["torn_lines"] = torn
    tol = []
    for k, v in res["cost"].items():
        c = v["F6 vs REF"]["128-"]
        if c is None:
            tol.append(f"{k}: no 128+ reviews"); continue
        flag = c["value"] > LONG_POINT or c["ci"][1] > LONG_UPPER
        tol.append(f"{k}: F6 vs REF {R.fmt(c)} on {c['reviews']} reviews, {c['users']} users: "
                   f"{'WARNING (over the tolerance)' if flag else 'inside the tolerance'}")
    res["tolerance"] = tol
    cov = min(res["coverage"][f"{n} F6"] for n, _ in HEADS.values())
    res["state"] = "decided" if cov >= MIN_LONG and not bad else ("invalid" if bad else "pending")
    json.dump(res, open(HERE / "sm165-result.json", "w"), indent=1, default=str)
    write_md(res, HERE / "sm165-result.md")
    print(json.dumps(res["coverage"]))
    for t in tol:
        print("[sm165]", t)
    print(f"[sm165] self-check T24 vs REF = SM155: {'OK' if not bad else f'{len(bad)} MISMATCHES'}; state {res['state']}")
    return {"decided": 0, "pending": 2, "invalid": 3}[res["state"]]


if __name__ == "__main__":
    sys.exit(main())
