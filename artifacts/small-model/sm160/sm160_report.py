#!/usr/bin/env python
"""Deploy-time ensemble of the two instant predictions (SM160): training users, GPU smoke check, the w fit and the
pre-registered read-out (design.md "Rules", fixed 11-24 before any row or dump of this run existed). CPU only.

Per size, the two-head model (TH) and the Instant-Only model (IO) both run on every review in the fork's default mode.
Ensemble E(w) = sigmoid(w * logit(p_IO) + (1 - w) * logit(p_TH)), p = the instant head's probability of each model,
logits of p clipped to [1e-7, 1 - 1e-7]. One w per size and label, fitted on the 100 training users (users-wfit.json)
only: the grid w = 0.00, 0.01, ..., 1.00, minimum of the summed per-review LogLoss (review-weighted).

  python sm160_report.py users             -> users-wfit.json (or checks the committed file)
  python sm160_report.py smoke             GPU smoke rows vs SM148's GPU smoke rows + the dumps; exit 0 = smoke OK
  python sm160_report.py fit --label T24   -> gpu/wfit-T24.json (training users only)
  python sm160_report.py report            -> sm160-result.{json,md}; exit 0 decided, 2 pending, 3 invalid
  python sm160_report.py selftest          synthetic dumps

Statistic: % = 100 x (sum over users of the ensemble's benchmark instant LogLoss x size / the same for IO alone - 1),
review-weighted, paired user bootstrap 2,000 draws, seed 0, 95% CI (the ratio-of-sums bootstrap of every 732 read-out),
on the rule users of the 732 fit list (both models of the size have a row and a dump; at least 715).
"""
import argparse
import json
import math
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.special import xlogy

HERE = Path(__file__).resolve().parent
G = HERE / "gpu"
WT = HERE.parents[2]
DUMP = WT / "work" / "sm160-dump"
OLDSMOKE = HERE.parent / "sm148" / "gpu" / "smoke-gpu-rows.jsonl"
OLD732 = HERE.parent / "sm148" / "gpu" / "sm148-rows-732.jsonl"
USERS732 = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json")
USERS_WFIT = HERE / "users-wfit.json"
USERS_SMOKE = HERE / "users-smoke.json"
ROWS = {"smoke": G / "smoke-gpu-rows.jsonl", "wfit": G / "sm160-rows-wfit.jsonl", "e732": G / "sm160-rows-732.jsonl"}
E2S = "@e2s-query"
LABELS = {"T24": "w8+lr2.16.4q8x8", "REF": "w8+s8hx8"}
DECISION = "T24"
SIZES = {   # size: (two-head tag, Instant-Only tag)
    "Big": ("sm91s2-big-showtime", "sm99-big-inst-e2"),
    "Small": ("sm118-e26-dec2ep32k-t91s2", "sm113-inst-e35st-st-dec2ep32k-t99"),
    "Tiny": ("sm130-th-dec2ep32k-t91s2", "sm130-io-dec2ep32k-t99"),
}
NAMES = {
    "sm91s2-big-showtime": "Big two-head (SM91s2)", "sm99-big-inst-e2": "Big Instant-Only (SM99)",
    "sm118-e26-dec2ep32k-t91s2": "Small two-head (SM118)", "sm113-inst-e35st-st-dec2ep32k-t99": "Small Instant-Only (SM113)",
    "sm130-th-dec2ep32k-t91s2": "six-layer Tiny two-head (SM130)", "sm130-io-dec2ep32k-t99": "six-layer Tiny Instant-Only (SM130)",
}
GRID = np.round(np.linspace(0.0, 1.0, 101), 2)
CLIP_LOGIT = 1e-7
MIN_GAIN = 0.05       # go: D <= -0.05% AND the CI upper bound below 0
MIN_732 = 715
MIN_WFIT = 90         # training users with both models' dumps needed for a fit
DRAWS = 2000
TOL_ROW = 2e-6        # dump-recomputed LogLoss vs the row (rows store 6 decimals)
TOL_SMOKE = 1e-5      # GPU smoke rows vs SM148's GPU smoke rows, per user and head
WFIT_RULE = ("validation users of the 8k split with id >= 5000 (the 1,500 pool minus the 732 set), none in the 732 or "
             "2,225 pools or the sealed final-test list, at most 300,000 reviews; random.Random(160).sample of 100")


def lab(name, tag):
    return f"{tag}/{LABELS[name]}{E2S}"


def dump_file(folder, user, label):
    return Path(folder) / f"{user}_{label.replace('/', '_')}.npz"


def loss_vec(y, p):
    """Per-review LogLoss as sklearn.metrics.log_loss (1.9: p clipped to float64 eps, xlogy)."""
    eps = np.finfo(np.float64).eps
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return -(xlogy(y, p) + xlogy(1 - y, 1 - p))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), CLIP_LOGIT, 1 - CLIP_LOGIT)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def read_rows(path):
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    for raw in open(p, "rb"):
        try:
            r = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            continue
        if "label" in r and "user_id" in r:
            out.setdefault(r["label"], {})[r["user_id"]] = r
    return out


def load_pair(folder, user, name, size):
    """y, p_TH, p_IO (instant heads) of one user; None if a dump is missing; raises if the review sets differ."""
    th, io = SIZES[size]
    f_th, f_io = dump_file(folder, user, lab(name, th)), dump_file(folder, user, lab(name, io))
    if not (f_th.exists() and f_io.exists()):
        return None
    a, b = np.load(f_th), np.load(f_io)
    if not (np.array_equal(a["review_th"], b["review_th"]) and np.array_equal(a["y"], b["y"])):
        raise SystemExit(f"[sm160] review sets differ: user {user}, {size}, {name}")
    return np.asarray(a["y"], dtype=np.float64), np.asarray(a["p_instant"]), np.asarray(b["p_instant"])


def user_sums(y, p_th, p_io):
    """Loss sums of one user: IO alone, TH alone, logit average and probability average for every grid w."""
    lt, li = logit(p_th), logit(p_io)
    s_logit = np.array([loss_vec(y, sigmoid(w * li + (1 - w) * lt)).sum() for w in GRID])
    s_prob = np.array([loss_vec(y, w * p_io + (1 - w) * p_th).sum() for w in GRID])
    return {"n": len(y), "io": loss_vec(y, p_io).sum(), "th": loss_vec(y, p_th).sum(), "logit": s_logit, "prob": s_prob}


def collect(folder, rows, users, name, size):
    """Per-user sums for the users with a row (benchmark, finite) and a dump for both models of the size.
    Also the largest |dump LogLoss - row LogLoss| (instant head, both models) as a pipeline check."""
    th, io = SIZES[size]
    rt, ri = rows.get(lab(name, th), {}), rows.get(lab(name, io), {})
    out, worst, missing_dump, nonfinite = {}, 0.0, [], []
    for u in users:
        if u not in rt or u not in ri:
            continue
        if "benchmark" not in rt[u] or "benchmark" not in ri[u]:
            nonfinite.append(u)
            continue
        pair = load_pair(folder, u, name, size)
        if pair is None:
            missing_dump.append(u)
            continue
        y, p_th, p_io = pair
        s = user_sums(y, p_th, p_io)
        if not all(math.isfinite(v) for v in (s["io"], s["th"])) or not np.isfinite(s["logit"]).all():
            nonfinite.append(u)
            continue
        for r, tot in ((ri[u], s["io"]), (rt[u], s["th"])):
            b = r["benchmark"]["instant"]
            if b["size"] != s["n"]:
                raise SystemExit(f"[sm160] dump size {s['n']} != row size {b['size']}: user {u}, {size}, {name}")
            worst = max(worst, abs(tot / s["n"] - b["LogLoss"]))
        out[u] = s
    return out, worst, missing_dump, nonfinite


def boot(a, b, draws=DRAWS):
    """% of sum(a) vs sum(b), paired user bootstrap (ratio of sums, seed 0)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    idx = np.random.default_rng(0).integers(0, len(a), size=(draws, len(a)))
    bs = (a[idx].sum(1) / b[idx].sum(1) - 1) * 100
    return {"value": float((a.sum() / b.sum() - 1) * 100),
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "users": int(len(a))}


def fit_w(sums):
    """Grid w minimising the summed training loss, logit average and probability average; aggregate curves."""
    us = sorted(sums)
    L = np.array([sums[u]["logit"] for u in us]).sum(0)
    P = np.array([sums[u]["prob"] for u in us]).sum(0)
    io = sum(sums[u]["io"] for u in us)
    th = sum(sums[u]["th"] for u in us)
    kl, kp = int(np.argmin(L)), int(np.argmin(P))
    return {"w_logit": float(GRID[kl]), "w_prob": float(GRID[kp]), "users": len(us),
            "reviews": int(sum(sums[u]["n"] for u in us)),
            "train_pct_vs_io": {"logit_fit": float((L[kl] / io - 1) * 100), "prob_fit": float((P[kp] / io - 1) * 100),
                                "logit_half": float((L[50] / io - 1) * 100), "th_alone": float((th / io - 1) * 100)},
            "train_curve_logit_pct_vs_io": [round(float((x / io - 1) * 100), 5) for x in L[::10]]}


# ----------------------------------------------------------------------------------------------------------- users
def users_cmd():
    sys.path.insert(0, str(WT))
    from scripts.eval_pools import pool_users, split_info
    _, split, _ = split_info()
    counts = {int(k): v for k, v in split["review_counts"].items()}
    p732, p2225 = set(pool_users(732)), set(pool_users(2225))
    sealed = set(split["final_test_users"])
    cands = sorted(u for u in pool_users(1500) if u >= 5000 and u not in p732 and u not in p2225 and u not in sealed
                   and counts[u] <= 300_000)
    users = sorted(random.Random(160).sample(cands, 100))
    assert not (set(users) & (p732 | p2225 | sealed | set(json.loads(USERS732.read_text())["users"])))
    assert set(users) <= set(split["validation_users"])
    doc = {"note": "Deploy-time ensemble of the two instant predictions (SM160): training users for the ensemble weight w. "
                   "Public benchmark user ids only.",
           "rule": WFIT_RULE, "candidates": len(cands), "users_count": len(users),
           "reviews_total": int(sum(counts[u] for u in users)), "users": users}
    if USERS_WFIT.exists():
        old = json.loads(USERS_WFIT.read_text())
        print("users-wfit.json", "OK (unchanged)" if old["users"] == users else "DIFFERS")
        return 0 if old["users"] == users else 1
    USERS_WFIT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {USERS_WFIT}: {len(users)} users of {len(cands)} candidates, {doc['reviews_total']} reviews")
    return 0


# ----------------------------------------------------------------------------------------------------------- smoke
def smoke_cmd():
    new, old = read_rows(ROWS["smoke"]), read_rows(OLDSMOKE)
    users = json.loads(USERS_SMOKE.read_text())["users"]
    lines, ok, worst_old, worst_dump = [], True, 0.0, 0.0
    for name in LABELS:
        for size, tags in SIZES.items():
            for tag in tags:
                L = lab(name, tag)
                for u in users:
                    r, o = new.get(L, {}).get(u), old.get(L, {}).get(u)
                    if r is None or "benchmark" not in r:
                        ok = False; lines.append(f"missing GPU smoke row {u} {L}"); continue
                    if o is None or "benchmark" not in o:
                        ok = False; lines.append(f"missing SM148 GPU smoke row {u} {L}"); continue
                    for h in ("ahead", "instant"):
                        if isinstance(r["benchmark"].get(h), dict) and isinstance(o["benchmark"].get(h), dict):
                            worst_old = max(worst_old, abs(r["benchmark"][h]["LogLoss"] - o["benchmark"][h]["LogLoss"]))
                    f = dump_file(DUMP / "smoke", u, L)
                    if not f.exists():
                        ok = False; lines.append(f"missing dump {f.name}"); continue
                    z = np.load(f)
                    y = np.asarray(z["y"], dtype=np.float64)
                    heads = (("instant", "p_instant"), ("ahead", "p_ahead")) if tag == tags[0] else (("instant", "p_instant"),)
                    for h, k in heads:
                        b = r["benchmark"][h]
                        if b["size"] != len(y):
                            ok = False; lines.append(f"dump size {len(y)} != row {b['size']}: {u} {L} {h}")
                        worst_dump = max(worst_dump, abs(loss_vec(y, z[k]).mean() - b["LogLoss"]))
                if tag == tags[1]:
                    for u in users:
                        try:
                            if load_pair(DUMP / "smoke", u, name, size) is None:
                                ok = False; lines.append(f"no dump pair {u} {size} {name}")
                        except SystemExit as e:
                            ok = False; lines.append(str(e))
    if worst_old > TOL_SMOKE:
        ok = False
    if worst_dump > TOL_ROW:
        ok = False
    lines.append(f"max |GPU smoke - SM148 GPU smoke| LogLoss {worst_old:.2e} (tolerance {TOL_SMOKE:g}); "
                 f"max |dump - row| LogLoss {worst_dump:.2e} (tolerance {TOL_ROW:g})")
    lines.append("smoke OK" if ok else "smoke FAILED")
    G.mkdir(parents=True, exist_ok=True)
    (G / "smoke-check.json").write_text(json.dumps({"ok": ok, "max_vs_sm148": worst_old, "max_dump_vs_row": worst_dump,
                                                    "problems": lines[:-2]}, indent=1) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


# ----------------------------------------------------------------------------------------------------------- fit
def fit_cmd(name, folder=None, rows_path=None, users=None, out=None):
    folder = folder or DUMP / "wfit"
    rows = read_rows(rows_path or ROWS["wfit"])
    users = users or json.loads(USERS_WFIT.read_text())["users"]
    doc = {"label": name, "variant": LABELS[name] + E2S, "rule": "grid w = 0.00..1.00 step 0.01, minimum summed per-review "
           "LogLoss on the training users (users-wfit.json); logit average decides, probability average reported",
           "sizes": {}}
    for size in SIZES:
        sums, worst, miss, nonf = collect(folder, rows, users, name, size)
        if len(sums) < MIN_WFIT:
            print(f"[sm160] fit {name} {size}: only {len(sums)} training users with both dumps (need {MIN_WFIT})")
            return 1
        if worst > TOL_ROW:
            print(f"[sm160] fit {name} {size}: dump vs row LogLoss {worst:.2e} > {TOL_ROW:g}")
            return 1
        doc["sizes"][size] = fit_w(sums) | {"missing_dump": miss, "nonfinite": nonf, "max_dump_vs_row": worst}
    out = out or G / f"wfit-{name}.json"
    Path(out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[sm160] w fit {name}: " + "; ".join(f"{s} w {d['w_logit']:.2f} (probability average {d['w_prob']:.2f}, "
                                                f"{d['users']} users)" for s, d in doc["sizes"].items()))
    return 0


# ----------------------------------------------------------------------------------------------------------- report
def evaluate(name, wfit, folder=None, rows_path=None, users=None):
    folder = folder or DUMP / "e732"
    rows = read_rows(rows_path or ROWS["e732"])
    users = users or json.loads(USERS732.read_text())["users"]
    res = {}
    for size in SIZES:
        sums, worst, miss, nonf = collect(folder, rows, users, name, size)
        us = sorted(sums)
        x = {"users": len(us), "reviews": int(sum(sums[u]["n"] for u in us)), "missing_dump": miss, "nonfinite": nonf,
             "max_dump_vs_row": worst, "coverage_ok": len(us) >= MIN_732 and not nonf, "rows_ok": worst <= TOL_ROW}
        if not us:
            res[size] = x
            continue
        io = [sums[u]["io"] for u in us]
        th = [sums[u]["th"] for u in us]
        kl = int(round(wfit[size]["w_logit"] * 100))
        kp = int(round(wfit[size]["w_prob"] * 100))
        L = np.array([sums[u]["logit"] for u in us])
        P = np.array([sums[u]["prob"] for u in us])
        ko = int(np.argmin(L.sum(0)))
        x |= {"w_fit": wfit[size]["w_logit"], "w_prob_fit": wfit[size]["w_prob"], "w_oracle": float(GRID[ko]),
              "D_fit": boot(L[:, kl], io), "D_half": boot(L[:, 50], io), "D_prob_fit": boot(P[:, kp], io),
              "D_oracle": boot(L[:, ko], io), "TH_vs_IO": boot(th, io), "E_fit_vs_TH": boot(L[:, kl], th),
              "curve_logit_pct_vs_io": [round(float((s / sum(io) - 1) * 100), 5) for s in L.sum(0)[::10]]}
        d = x["D_fit"]
        x["go"] = bool(x["coverage_ok"] and x["rows_ok"] and d["ci"][1] < 0 and d["value"] <= -MIN_GAIN)
        res[size] = x
    return res


def write_md(doc):
    L = ["# Deploy-time ensemble of the two instant predictions (SM160): read-out", "",
         "Generated by `sm160_report.py report` (design.md \"Rules\"). % = review-weighted LogLoss change of the instant "
         "head against the Instant-Only model alone, paired user bootstrap 2,000 draws, seed 0, 95% CI. w is the weight "
         "of the Instant-Only model's logit, fitted on the 100 training users only.", ""]
    for name, res in doc["labels"].items():
        if res is None:
            L += [f"## {name} ({LABELS[name]}): pending", ""]
            continue
        L += [f"## {name} ({LABELS[name]}{E2S})" + (" - the decision" if name == DECISION else " - information"), "",
              "| Size | Users | w fit | D (fit) | CI | Go | w = 0.5 | probability average (w) | oracle w (eval, report only) | two-head alone vs Instant-Only | ensemble vs two-head |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for size, x in res.items():
            if "D_fit" not in x:
                L.append(f"| {size} | {x['users']} | | | | pending | | | | | |")
                continue
            f = lambda c: f"{c['value']:+.3f}%"
            ci = lambda c: f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]"
            go = ("YES" if x["go"] else "NO") if name == DECISION else "(info)"
            L.append(f"| {size} | {x['users']} | {x['w_fit']:.2f} | {f(x['D_fit'])} | {ci(x['D_fit'])} | {go} | "
                     f"{f(x['D_half'])} {ci(x['D_half'])} | {f(x['D_prob_fit'])} ({x['w_prob_fit']:.2f}) | "
                     f"{x['w_oracle']:.2f}: {f(x['D_oracle'])} | {f(x['TH_vs_IO'])} {ci(x['TH_vs_IO'])} | "
                     f"{f(x['E_fit_vs_TH'])} {ci(x['E_fit_vs_TH'])} |")
        L.append("")
    L += [f"Verdict: {doc['verdict']}", ""]
    (HERE / "sm160-result.md").write_text("\n".join(L) + "\n")


def report_cmd():
    doc = {"rules": {"min_gain_pct": MIN_GAIN, "min_users": MIN_732, "draws": DRAWS, "seed": 0, "decision_label": DECISION},
           "models": {s: {"two_head": NAMES[t[0]], "instant_only": NAMES[t[1]]} for s, t in SIZES.items()}, "labels": {}}
    status = 0
    for name in LABELS:
        wf = G / f"wfit-{name}.json"
        if not wf.exists():
            doc["labels"][name] = None
            status = max(status, 2) if name == DECISION else status
            continue
        wfit = json.loads(wf.read_text())["sizes"]
        res = evaluate(name, wfit)
        doc["labels"][name] = res
        if name == DECISION:
            if any(not x["rows_ok"] for x in res.values()):
                status = 3
            elif any(not x["coverage_ok"] for x in res.values()):
                status = max(status, 2)
    dec = doc["labels"].get(DECISION)
    if status == 0:
        goes = [s for s, x in dec.items() if x["go"]]
        doc["verdict"] = (f"go for {', '.join(goes)} (the ensemble replaces the Instant-Only model's instant prediction "
                          f"there); no go for {', '.join(s for s in SIZES if s not in goes) or 'none'}") if goes else \
            "NO GO for every size: the Instant-Only model alone stays"
    else:
        doc["verdict"] = {2: "pending (missing fit or fewer than 715 users)", 3: "invalid (dump vs row mismatch)"}[status]
    (HERE / "sm160-result.json").write_text(json.dumps(doc, indent=1) + "\n")
    write_md(doc)
    print(f"[sm160] {doc['verdict']}")
    return status


# ----------------------------------------------------------------------------------------------------------- selftest
def selftest():
    from sklearn.metrics import log_loss
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500).astype(np.float64)
    p = rng.uniform(0, 1, 500)
    p[:3] = [0.0, 1.0, 1e-20]
    assert abs(loss_vec(y, p).mean() - log_loss(y, p, labels=[0, 1])) < 1e-12, "loss formula vs sklearn"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rows = []

        def make(folder, users, seed):
            r = np.random.default_rng(seed)
            for u in users:
                n = int(r.integers(200, 2000))
                z = r.normal(0.8, 1.5, n)
                yy = (r.uniform(0, 1, n) < sigmoid(z)).astype(np.int64)
                pt, pi = sigmoid(z + r.normal(0, 0.8, n)), sigmoid(z + r.normal(0, 0.8, n))
                th = np.arange(n, dtype=np.int64) * 3 + 1
                for size, (t, i) in SIZES.items():
                    for name in LABELS:
                        for tag, pp in ((t, pt), (i, pi)):
                            L = lab(name, tag)
                            folder.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(dump_file(folder, u, L), review_th=th, bin=np.zeros(n), y=yy,
                                                p_ahead=pp, p_instant=pp)
                            ll = log_loss(yy, pp, labels=[0, 1])
                            rows.append({"user_id": u, "label": L, "benchmark": {"instant": {"LogLoss": round(ll, 6), "size": n}}})
        make(tmp / "fit", list(range(9000, 9100)), 2)
        make(tmp / "ev", list(range(100, 820)), 3)
        rp = tmp / "rows.jsonl"
        rp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert fit_cmd("T24", tmp / "fit", rp, list(range(9000, 9100)), tmp / "wfit.json") == 0
        wf = json.loads((tmp / "wfit.json").read_text())["sizes"]
        assert all(0.4 <= wf[s]["w_logit"] <= 0.6 for s in SIZES), wf   # equal noise: w near 0.5
        res = evaluate("T24", wf, tmp / "ev", rp, list(range(100, 820)))
        for s, x in res.items():
            assert x["users"] == 720 and x["coverage_ok"] and x["rows_ok"], x
            assert x["go"] and x["D_fit"]["ci"][1] < 0, x["D_fit"]
        one = {s: {"w_logit": 1.0, "w_prob": 1.0} for s in SIZES}   # w = 1: the ensemble is the Instant-Only model
        res1 = evaluate("T24", one, tmp / "ev", rp, list(range(100, 820)))
        assert all(abs(x["D_fit"]["value"]) < 1e-6 and not x["go"] for x in res1.values()), res1
        few = evaluate("T24", wf, tmp / "ev", rp, list(range(100, 700)))   # 600 users: coverage fails, no go
        assert all(not x["coverage_ok"] and not x["go"] for x in few.values())
        assert boot([1, 2, 3], [1, 2, 3])["ci"] == [0.0, 0.0]
    print("[sm160] selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("users", "smoke", "fit", "report", "selftest"))
    ap.add_argument("--label", choices=tuple(LABELS), default=DECISION)
    a = ap.parse_args()
    sys.exit({"users": users_cmd, "smoke": smoke_cmd, "report": report_cmd, "selftest": selftest,
              "fit": lambda: fit_cmd(a.label)}[a.cmd]())


if __name__ == "__main__":
    main()
