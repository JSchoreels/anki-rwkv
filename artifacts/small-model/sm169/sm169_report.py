#!/usr/bin/env python
"""Tiny ensemble on the 1,500 users (SM169): users, GPU smoke check and the pre-registered read-out (design.md "Rules",
fixed 11-25 before any row or dump of this run existed). CPU only.

Decisive confirmation of the deploy-time ensemble (SM160) six-layer Tiny go, with the weight fixed (NOT refitted):
E = sigmoid(w * logit(p_IO) + (1 - w) * logit(p_TH)), w = 0.69 (SM160's `gpu/wfit-T24.json`, Tiny), p_IO the six-layer
Tiny Instant-Only (SM130) instant prediction, p_TH the six-layer Tiny two-head (SM130) instant prediction, at T24
(`w8+lr2.16.4q8x8@e2s-query`). The loss, logit, dump and bootstrap code is SM160's (`../sm160/sm160_report.py`, imported).

Rule users: the 1,500 pool minus SM160's 100 fit users = two parts
  e732  the 732 fit list (723 users): SM160's rows (`../sm160/gpu/sm160-rows-732.jsonl`) and SM160's Tiny T24 dumps
        copied to `work/sm169-dump/e732/` (never committed)
  new   the 1,500 pool's users with id >= 5000 and at most 650,000 reviews, minus users-wfit.json (665 users,
        `users-1500-new.json`, equal to SM164's users-new-rest.json): this run's rows `gpu/sm169-rows-1500.jsonl`,
        dumps `work/sm169-dump/new/`

  python sm169_report.py users      -> users-1500-new.json, users-smoke.json (or checks the committed files)
  python sm169_report.py smoke      smoke users' rows vs SM164's rows + dumps reproduce the rows; exit 0 = smoke OK
  python sm169_report.py report     -> sm169-result.{json,md}; exit 0 decided, 2 pending, 3 invalid
  python sm169_report.py selftest   synthetic dumps
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G = HERE / "gpu"
WT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "sm160"))
import sm160_report as s160  # noqa: E402  (SM160's loss / logit / dump / bootstrap code, unchanged)

DUMP = WT / "work" / "sm169-dump"
PARTS = {   # part: (dump folder, rows file, users file)
    "e732": (DUMP / "e732", HERE.parent / "sm160" / "gpu" / "sm160-rows-732.jsonl", s160.USERS732),
    "new": (DUMP / "new", G / "sm169-rows-1500.jsonl", HERE / "users-1500-new.json"),
}
WFIT = HERE.parent / "sm160" / "gpu" / "wfit-T24.json"
W_FIXED = 0.69
SM164_ROWS = Path("C:/Users/jonat/workspace/wt/sm164/artifacts/small-model/sm164/gpu/sm164-rows-new.jsonl")
SM164_USERS = Path("C:/Users/jonat/workspace/wt/sm164/artifacts/small-model/sm164/users-new-rest.json")
USERS_NEW = HERE / "users-1500-new.json"
USERS_SMOKE = HERE / "users-smoke.json"
NAME, SIZE = "T24", "Tiny"
TH, IO = s160.SIZES[SIZE]
MIN_USERS = 1375      # rule users (both parts) with a finite row and a dump for both models
MIN_GAIN = s160.MIN_GAIN
TOL_ROW = s160.TOL_ROW
TOL_SMOKE = 1e-5
MAX_REVIEWS = 650_000
NEW_RULE = ("the 1,500 pool's users with id >= 5000 and at most 650,000 reviews (SM93's rule, as SM164), minus the 100 "
            "SM160 fit users (users-wfit.json)")


def k_of(w):
    return int(round(w * 100))


def gather(part, users=None, folder=None, rows_path=None):
    """Per-user sums of one part (SM160's collect), with out-of-memory error rows split off (excluded and listed)."""
    f0, r0, u0 = PARTS[part]
    folder, rows_path = folder or f0, rows_path or r0
    users = users if users is not None else json.loads(Path(u0).read_text())["users"]
    rows = s160.read_rows(rows_path)
    rt, ri = rows.get(s160.lab(NAME, TH), {}), rows.get(s160.lab(NAME, IO), {})
    oom = sorted(u for u in users if any(u in r and "error" in r[u] for r in (rt, ri)))
    missing_row = sorted(u for u in users if u not in oom and (u not in rt or u not in ri))
    ok = [u for u in users if u not in oom and u not in missing_row]
    sums, worst, miss, nonf = s160.collect(folder, rows, ok, NAME, SIZE)
    return {"sums": sums, "max_dump_vs_row": worst, "missing_dump": miss, "nonfinite": nonf, "oom": oom,
            "missing_row": missing_row, "listed": len(users)}


def stats(sums, w):
    us = sorted(sums)
    io = np.array([sums[u]["io"] for u in us])
    th = np.array([sums[u]["th"] for u in us])
    L = np.array([sums[u]["logit"] for u in us])
    k, ko = k_of(w), int(np.argmin(L.sum(0)))
    better = L[:, k] < io
    return {"users": len(us), "reviews": int(sum(sums[u]["n"] for u in us)),
            "D": s160.boot(L[:, k], io), "w_oracle": float(s160.GRID[ko]), "D_oracle": s160.boot(L[:, ko], io),
            "share_users_improve": float(better.mean()) if len(us) else None, "users_improve": int(better.sum()),
            "TH_vs_IO": s160.boot(th, io), "E_vs_TH": s160.boot(L[:, k], th),
            "curve_logit_pct_vs_io": [round(float((s / io.sum() - 1) * 100), 5) for s in L.sum(0)[::10]]}


def users_cmd():
    sys.path.insert(0, str(WT))
    from scripts.eval_pools import pool_users, split_info
    _, split, _ = split_info()
    counts = {int(k): v for k, v in split["review_counts"].items()}
    wfit = set(json.loads(s160.USERS_WFIT.read_text())["users"])
    e732 = set(json.loads(s160.USERS732.read_text())["users"])
    p1500 = set(pool_users(1500))
    new = sorted(u for u in p1500 if u >= 5000 and counts[u] <= MAX_REVIEWS and u not in wfit)
    assert not (set(new) & (e732 | wfit | set(split["final_test_users"])))
    assert e732 <= p1500 and wfit <= p1500
    smoke = sorted(sorted(new, key=lambda u: (counts[u], u))[:2])
    if SM164_USERS.exists():
        assert json.loads(SM164_USERS.read_text())["users"] == new, "differs from SM164's users-new-rest.json"
    docs = {USERS_NEW: {"note": "Tiny ensemble on the 1,500 users (SM169): the users with id >= 5000 to score. Public "
                                "benchmark user ids only.", "rule": NEW_RULE, "users_count": len(new),
                        "reviews_total": int(sum(counts[u] for u in new)), "users": new},
            USERS_SMOKE: {"note": "SM169 GPU smoke users: the 2 smallest of users-1500-new.json (rows vs SM164's rows). "
                                  "Public benchmark user ids only.", "users": smoke}}
    rc = 0
    for path, doc in docs.items():
        if path.exists():
            same = json.loads(path.read_text())["users"] == doc["users"]
            print(path.name, "OK (unchanged)" if same else "DIFFERS")
            rc |= not same
        else:
            path.write_text(json.dumps(doc, indent=1) + "\n")
            print(f"wrote {path.name}: {len(doc['users'])} users" + (f", {doc['reviews_total']} reviews" if "reviews_total" in doc else ""))
    print(f"732 part: {len(e732)} users; new part: {len(new)} users; together {len(e732) + len(new)} "
          f"(the 1,500 pool has {len(p1500)}; minus the 100 fit users {len(p1500 - wfit)}; "
          f"{len(p1500 - wfit) - len(e732) - len(new)} not scorable: 732-set users outside the fit list or > 650,000 reviews)")
    return rc


def smoke_cmd():
    new, old = s160.read_rows(PARTS["new"][1]), s160.read_rows(SM164_ROWS)
    users = json.loads(USERS_SMOKE.read_text())["users"]
    lines, ok, worst_old, worst_dump = [], True, 0.0, 0.0
    for tag in (TH, IO):
        L = s160.lab(NAME, tag)
        for u in users:
            r, o = new.get(L, {}).get(u), old.get(L, {}).get(u)
            if r is None or "benchmark" not in r or o is None or "benchmark" not in o:
                ok = False; lines.append(f"missing row {u} {L} (this run or SM164)"); continue
            heads = ("instant", "ahead") if tag == TH else ("instant",)
            for h in heads:
                worst_old = max(worst_old, abs(r["benchmark"][h]["LogLoss"] - o["benchmark"][h]["LogLoss"]))
            f = s160.dump_file(PARTS["new"][0], u, L)
            if not f.exists():
                ok = False; lines.append(f"missing dump {f.name}"); continue
            z = np.load(f)
            y = np.asarray(z["y"], dtype=np.float64)
            for h, k in ((h, "p_" + h) for h in heads):
                b = r["benchmark"][h]
                if b["size"] != len(y):
                    ok = False; lines.append(f"dump size {len(y)} != row {b['size']}: {u} {L} {h}")
                worst_dump = max(worst_dump, abs(s160.loss_vec(y, z[k]).mean() - b["LogLoss"]))
    for u in users:
        try:
            if s160.load_pair(PARTS["new"][0], u, NAME, SIZE) is None:
                ok = False; lines.append(f"no dump pair {u}")
        except SystemExit as e:
            ok = False; lines.append(str(e))
    ok = bool(ok and worst_old <= TOL_SMOKE and worst_dump <= TOL_ROW)
    worst_old, worst_dump = float(worst_old), float(worst_dump)
    lines.append(f"max |GPU smoke - SM164 row| LogLoss {worst_old:.2e} (tolerance {TOL_SMOKE:g}); "
                 f"max |dump - row| LogLoss {worst_dump:.2e} (tolerance {TOL_ROW:g})")
    lines.append("smoke OK" if ok else "smoke FAILED")
    G.mkdir(parents=True, exist_ok=True)
    (G / "smoke-check.json").write_text(json.dumps({"ok": ok, "max_vs_sm164": worst_old, "max_dump_vs_row": worst_dump,
                                                    "problems": lines[:-2]}, indent=1) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


def sm164_crosscheck():
    """Report only: this run's new-part rows vs SM164's rows of the same users (no dumps there), instant and ahead."""
    if not SM164_ROWS.exists():
        return None
    new, old = s160.read_rows(PARTS["new"][1]), s160.read_rows(SM164_ROWS)
    worst, n = {}, 0
    for tag in (TH, IO):
        L = s160.lab(NAME, tag)
        for u, r in new.get(L, {}).items():
            o = old.get(L, {}).get(u)
            if "benchmark" not in r or o is None or "benchmark" not in o:
                continue
            n += 1
            for h in ("instant", "ahead"):
                if isinstance(r["benchmark"].get(h), dict) and isinstance(o["benchmark"].get(h), dict):
                    d = abs(r["benchmark"][h]["LogLoss"] - o["benchmark"][h]["LogLoss"])
                    if d > worst.get(f"{tag}/{h}", (0.0, None))[0]:
                        worst[f"{tag}/{h}"] = (d, u)
    return {"pairs": n, "max_abs_diff": {k: {"diff": v[0], "user": v[1]} for k, v in worst.items()}}


def evaluate(parts, w):
    res, allsums = {"parts": {}}, {}
    for part, g in parts.items():
        x = {k: v for k, v in g.items() if k != "sums"}
        if g["sums"]:
            x |= stats(g["sums"], w)
        else:
            x["users"] = 0
        res["parts"][part] = x
        allsums |= g["sums"]
    worst = max(g["max_dump_vs_row"] for g in parts.values())
    nonf = sum((g["nonfinite"] for g in parts.values()), [])
    res["all"] = stats(allsums, w) if allsums else {"users": 0}
    res["rows_ok"] = worst <= TOL_ROW
    res["coverage_ok"] = len(allsums) >= MIN_USERS and not nonf
    res["max_dump_vs_row"] = worst
    d = res["all"].get("D")
    res["go"] = bool(res["rows_ok"] and res["coverage_ok"] and d and d["ci"][1] < 0 and d["value"] <= -MIN_GAIN)
    return res


def write_md(doc):
    a = doc["result"]["all"]
    f = lambda c: f"{c['value']:+.3f}% [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]"
    L = ["# Tiny ensemble on the 1,500 users (SM169): read-out", "",
         "Generated by `sm169_report.py report` (design.md \"Rules\"). % = review-weighted instant-head LogLoss change of "
         "the logit ensemble (w = 0.69, fixed from SM160) against the six-layer Tiny Instant-Only (SM130) alone, at T24, "
         "paired user bootstrap 2,000 draws, seed 0, 95% CI.", "",
         "| Users | Reviews | D (w = 0.69) | Go | oracle w (report only): D | share of users that improve | two-head alone vs Instant-Only | ensemble vs two-head |",
         "|---|---|---|---|---|---|---|---|"]
    rows = [("all (decision)", a)] + [(p, x) for p, x in doc["result"]["parts"].items()]
    for name, x in rows:
        if "D" not in x:
            L.append(f"| {name}: {x['users']} users | | | pending | | | | |")
            continue
        go = ("YES" if doc["result"]["go"] else "NO") if name.startswith("all") else "(part)"
        L.append(f"| {name}: {x['users']} | {x['reviews']:,} | {f(x['D'])} | {go} | {x['w_oracle']:.2f}: "
                 f"{x['D_oracle']['value']:+.3f}% | {100 * x['share_users_improve']:.1f}% ({x['users_improve']}) | "
                 f"{f(x['TH_vs_IO'])} | {f(x['E_vs_TH'])} |")
    L += ["", f"Excluded (out of memory on the GPU at cap 0.92): {doc['result']['parts']['new'].get('oom', [])} (new part), "
          f"{doc['result']['parts']['e732'].get('oom', [])} (732 part). Largest |dump - row| instant LogLoss: "
          f"{doc['result']['max_dump_vs_row']:.1e} (limit {TOL_ROW:g}).", "",
          f"Cross-check vs SM164's rows (report only): {json.dumps(doc.get('sm164_crosscheck'))}", "",
          f"Verdict: {doc['verdict']}", ""]
    (HERE / "sm169-result.md").write_text("\n".join(L) + "\n")


def report_cmd():
    wf = json.loads(WFIT.read_text())["sizes"][SIZE]["w_logit"]
    assert abs(wf - W_FIXED) < 1e-9, f"SM160's fitted Tiny w is {wf}, design says {W_FIXED}"
    parts = {p: gather(p) for p in PARTS}
    res = evaluate(parts, W_FIXED)
    doc = {"rules": {"w_fixed": W_FIXED, "min_gain_pct": MIN_GAIN, "min_users": MIN_USERS, "draws": s160.DRAWS,
                     "seed": 0, "label": s160.lab(NAME, IO).split("/", 1)[1]},
           "models": {"two_head": s160.NAMES[TH], "instant_only": s160.NAMES[IO]}, "result": res,
           "sm164_crosscheck": sm164_crosscheck()}
    if not res["rows_ok"]:
        status, doc["verdict"] = 3, "invalid (a dump does not reproduce its row)"
    elif not res["coverage_ok"]:
        status, doc["verdict"] = 2, (f"pending ({res['all']['users']} rule users, need {MIN_USERS}; non-finite rows: "
                                     f"{sum((g['nonfinite'] for g in parts.values()), [])})")
    else:
        status = 0
        doc["verdict"] = ("GO: the six-layer Tiny ensemble (w = 0.69) holds on the 1,500 users" if res["go"] else
                          "NO GO: the six-layer Tiny ensemble (w = 0.69) does not hold on the 1,500 users")
    (HERE / "sm169-result.json").write_text(json.dumps(doc, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    write_md(doc)
    print(f"[sm169] {doc['verdict']}")
    a = res["all"]
    if "D" in a:
        print(f"[sm169] all: {a['users']} users, D {a['D']['value']:+.4f}% CI [{a['D']['ci'][0]:+.4f}, {a['D']['ci'][1]:+.4f}], "
              f"oracle w {a['w_oracle']:.2f}, share improving {100 * a['share_users_improve']:.1f}%")
    for p, x in res["parts"].items():
        if "D" in x:
            print(f"[sm169] {p}: {x['users']} users, D {x['D']['value']:+.4f}% CI [{x['D']['ci'][0]:+.4f}, {x['D']['ci'][1]:+.4f}]")
    return status


def selftest():
    from sklearn.metrics import log_loss
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rows = []
        users = list(range(100, 1500))
        for u in users:
            n = int(rng.integers(200, 1500))
            z = rng.normal(0.8, 1.5, n)
            yy = (rng.uniform(0, 1, n) < s160.sigmoid(z)).astype(np.int64)
            pt, pi = s160.sigmoid(z + rng.normal(0, 0.9, n)), s160.sigmoid(z + rng.normal(0, 0.7, n))
            for tag, pp in ((TH, pt), (IO, pi)):
                L = s160.lab(NAME, tag)
                if u == 101 and tag == IO:
                    rows.append({"user_id": u, "label": L, "error": "OOM: synthetic"})
                    continue
                np.savez_compressed(s160.dump_file(tmp, u, L), review_th=np.arange(n), bin=np.zeros(n), y=yy,
                                    p_ahead=pp, p_instant=pp)
                rows.append({"user_id": u, "label": L,
                             "benchmark": {"instant": {"LogLoss": round(log_loss(yy, pp, labels=[0, 1]), 6), "size": n}}})
        rp = tmp / "rows.jsonl"
        rp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        g = gather("new", users, tmp, rp)
        assert g["oom"] == [101] and not g["nonfinite"] and len(g["sums"]) == len(users) - 1, g["oom"]
        res = evaluate({"new": g, "e732": {**g, "sums": {}}}, W_FIXED)
        assert res["coverage_ok"] and res["rows_ok"] and res["go"], res["all"]["D"]
        assert 0.5 <= res["all"]["w_oracle"] <= 0.8 and 0.5 < res["all"]["share_users_improve"] <= 1
        res1 = evaluate({"new": g}, 1.0)   # w = 1: the ensemble is the Instant-Only model alone
        assert abs(res1["all"]["D"]["value"]) < 1e-6 and not res1["go"]   # equal up to the logit clip
        few = evaluate({"new": gather("new", users[:900], tmp, rp)}, W_FIXED)
        assert not few["coverage_ok"] and not few["go"]
    print("[sm169] selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("users", "smoke", "report", "selftest"))
    a = ap.parse_args()
    sys.exit({"users": users_cmd, "smoke": smoke_cmd, "report": report_cmd, "selftest": selftest}[a.cmd]())


if __name__ == "__main__":
    main()
