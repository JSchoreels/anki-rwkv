#!/usr/bin/env python
"""How many users for an MIT Small (SM176): read-out (design.md). SM174's report with the Small two-head release model
(SM118) as the reference and the arms 1,000 / 2,000 / 4,000 (no teacher) + 8,000 (teacher). CPU only, numpy; seconds.

  python sm176_report.py report     -> sm176-result.json, sm176-result.md (partial while arms are pending)
                                       exit 0 all arms scored, 2 pending
  python sm176_report.py selftest   SM148's rows of the Small two-head (SM118) relabelled as every arm, scaled
                                    by known factors: the gaps, the interpolated user counts and the identity must come
                                    out exactly

Statistic (as SM148 / SM171): % = 100 x (sum of the arm's LogLoss x reviews / the same sum of the reference - 1) per
head, paired user bootstrap, 2,000 draws, numpy default_rng(0), 95% CI, on the 723 users of the 732 set that have rows
for both (at least 720). Reference = the Small two-head (SM118) release model, w8+s8hx8, show-time inputs, its
SM148 rows (same scorer tree cea0f931). Gap of an arm = max over the two heads.
User count for a threshold t (+1.0%, +0.5%): piecewise-linear interpolation of the gap in log2(users) over the
no-teacher arms (1,000 ... 4,000), first crossing from above; "<= 1,000" if the smallest arm is already under; "> 4,000"
if the largest is not, then an indicative power-law fit gap(n) = a + b n^-c (c on a grid 0.05..2, least squares in
a, b) gives the extrapolated count when its floor a is below t.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sm176_chain as chain  # noqa: E402

REF_ROWS = chain.RUN / "artifacts/small-model/sm148/gpu/sm148-rows-732.jsonl"
U723 = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json")
REF_LABEL = f"sm118-e26-dec2ep32k-t91s2/{chain.VARIANT}{chain.E2S}"
HEADS = {"curve": "ahead", "instant": "instant"}
NO_TEACHER = ["n1000", "n2000", "n4000"]
CONTROL = "n8000-kd"
NAMES = {a: f"{chain.ARMS[a][0]:,} users, {'with the usual teacher' if chain.ARMS[a][1] else 'no teacher'} (SM176 {a})"
         for a in chain.ORDER}
REF_NAME = "the Small two-head release model (SM118)"
MIN_USERS = 720
DRAWS = 2000
THRESHOLDS = (1.0, 0.5)


def label(arm):
    return f"sm176-{arm}/{chain.VARIANT}{chain.E2S}"


def sums(rows, users, head):
    h = HEADS[head]
    return np.array([rows[u]["benchmark"][h]["LogLoss"] * rows[u]["benchmark"][h]["size"] for u in users])


def cmp(new, ref, users, head):
    idx = np.random.default_rng(0).integers(0, len(users), size=(DRAWS, len(users)))
    a, b = sums(new, users, head), sums(ref, users, head)
    for u in users:
        if new[u]["benchmark"][HEADS[head]]["size"] != ref[u]["benchmark"][HEADS[head]]["size"]:
            raise SystemExit(f"user {u} {head}: review counts differ between the arm and the reference")
    value = (a.sum() / b.sum() - 1) * 100
    boot = (a[idx].sum(1) / b[idx].sum(1) - 1) * 100
    return {"value": float(value), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "users": len(users), "logloss": float(a.sum() / sum(new[u]["benchmark"][HEADS[head]]["size"] for u in users)),
            "ref_logloss": float(b.sum() / sum(ref[u]["benchmark"][HEADS[head]]["size"] for u in users))}


def pair(new, ref, u723):
    users = sorted(set(new) & set(ref) & u723)
    if len(users) < MIN_USERS:
        return None
    return {h: cmp(new, ref, users, h) for h in HEADS}


def crossing(points, t):
    """points: [(users, gap)] sorted by users. First crossing of gap <= t, interpolated in log2(users)."""
    if not points:
        return None
    if points[0][1] <= t:
        return {"users": f"<= {points[0][0]:,}", "value": points[0][0], "kind": "at or below the smallest arm"}
    for (n0, g0), (n1, g1) in zip(points, points[1:]):
        if g0 > t >= g1:
            x = math.log2(n0) + (g0 - t) / (g0 - g1) * (math.log2(n1) - math.log2(n0))
            n = 2 ** x
            return {"users": f"~{n:,.0f}", "value": n, "kind": f"interpolated between {n0:,} and {n1:,}"}
    if len(points) >= 3:
        n = np.array([p[0] for p in points], float)
        g = np.array([p[1] for p in points], float)
        best = None
        for c in np.arange(0.05, 2.0001, 0.05):
            X = np.stack([np.ones_like(n), n ** -c], 1)
            coef, *_ = np.linalg.lstsq(X, g, rcond=None)
            err = float(((X @ coef - g) ** 2).sum())
            if best is None or err < best[0]:
                best = (err, float(c), float(coef[0]), float(coef[1]))
        _, c, a, b = best
        fit = {"floor_a": a, "b": b, "c": c}
        if a < t and b > 0:
            n_t = ((t - a) / b) ** (-1 / c)
            return {"users": f"> {points[-1][0]:,} (extrapolated ~{n_t:,.0f}, indicative)", "value": None, "kind": "power-law fit", "fit": fit}
        return {"users": f"> {points[-1][0]:,} (not reached; fitted floor {a:+.2f}% at this budget)", "value": None,
                "kind": "power-law fit", "fit": fit}
    return {"users": f"> {points[-1][0]:,} (not reached)", "value": None, "kind": "not reached"}


def training_stats(arm):
    st_path = chain.state_path(arm)
    st = json.loads(st_path.read_text()) if st_path.exists() else None
    out = {"state": st and {k: st[k] for k in ("reason", "best_epoch", "decay_epochs", "slice_mean_loss")}}
    folders = sorted((chain.OUT / arm).glob("e[0-9][0-9]")) + sorted((chain.OUT / arm).glob("d[0-9][0-9]"))
    reviews = seconds = 0
    done = 0
    for f in folders:
        r = chain.result_of(f)
        if r and r.get("status") == "complete" and not r.get("smoke_only"):
            done += 1
            reviews += r.get("review_presentations", 0)
            seconds += r.get("training_seconds", 0)
    out.update(folders_complete=done, reviews_trained=reviews, training_hours=seconds / 3600,
               reviews_per_second=(reviews / seconds if seconds else None))
    return out


def compute(rows732, ref_rows, u723):
    ref = ref_rows.get(REF_LABEL, {})
    if not ref:
        raise SystemExit(f"reference rows lack {REF_LABEL}")
    res = {"reference": REF_LABEL, "arms": {}, "pending": []}
    for arm in chain.ORDER:
        rows = rows732.get(label(arm), {})
        vs_ref = pair(rows, ref, u723)
        entry = {"name": NAMES[arm], "users_scored": len(rows), "vs_sm118": vs_ref}
        if vs_ref is None:
            res["pending"].append(arm)
        else:
            entry["gap_point"] = max(vs_ref[h]["value"] for h in HEADS)
            entry["gap_ci_upper"] = max(vs_ref[h]["ci"][1] for h in HEADS)
            for other in (CONTROL,):
                if other != arm and rows732.get(label(other)):
                    entry[f"vs_{other}"] = pair(rows, rows732[label(other)], u723)
        res["arms"][arm] = entry
    pts = [(chain.ARMS[a][0], res["arms"][a]) for a in NO_TEACHER if res["arms"][a]["vs_sm118"]]
    res["users_needed_vs_sm118"] = {
        f"+{t:.1f}%": {"point": crossing([(n, e["gap_point"]) for n, e in pts], t),
                       "ci_upper": crossing([(n, e["gap_ci_upper"]) for n, e in pts], t)} for t in THRESHOLDS}
    ctl = [(chain.ARMS[a][0], res["arms"][a].get(f"vs_{CONTROL}")) for a in NO_TEACHER]
    ctl = [(n, max(v[h]["value"] for h in HEADS)) for n, v in ctl if v]
    if ctl and res["arms"][CONTROL]["vs_sm118"]:
        res["users_needed_vs_equal_compute_teacher"] = {f"+{t:.1f}%": crossing(ctl, t) for t in THRESHOLDS}
    res["status"] = "complete" if not res["pending"] else "pending"
    return res


def fmt(c):
    return f"{c['value']:+.2f}% [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]"


def write_md(res, path):
    L = ["# How many users for an MIT Small (SM176): result", "",
         f"Status: **{res['status']}**" + (f" (pending: {', '.join(NAMES[a] for a in res['pending'])})" if res["pending"] else ""), "",
         f"Every row: Small architecture (`w64`, 331,511 parameters) trained from scratch under the same review budget "
         f"(four 8k epochs), scored on the 732 set (723 users), show-time inputs, w8+s8hx8, against {REF_NAME}. "
         "% = review-weighted LogLoss vs the reference, paired user bootstrap 95% CI.", "",
         "| model | curve vs SM118 | instant vs SM118 | curve LogLoss | instant LogLoss | vs the 8,000-user teacher control (SM176 n8000-kd), curve / instant | epochs (best constant epoch + decay) | reviews trained | GPU h | reviews/s |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in chain.ORDER:
        e = res["arms"][arm]
        t = e.get("training", {})
        st = t.get("state") or {}
        ep = f"{st.get('best_epoch')} + {st.get('decay_epochs')}" if st else "-"
        rs = f"{t['reviews_per_second']:,.0f}" if t.get("reviews_per_second") else "-"
        if e["vs_sm118"]:
            v = e["vs_sm118"]
            vc = e.get(f"vs_{CONTROL}")
            vcs = f"{fmt(vc['curve'])} / {fmt(vc['instant'])}" if vc else "-"
            L.append(f"| {e['name']} | {fmt(v['curve'])} | {fmt(v['instant'])} | {v['curve']['logloss']:.4f} | "
                     f"{v['instant']['logloss']:.4f} | {vcs} | {ep} | {t.get('reviews_trained', 0) / 1e9:.2f}B | "
                     f"{t.get('training_hours', 0):.1f} | {rs} |")
        else:
            L.append(f"| {e['name']} | pending | pending | | | | {ep} | {t.get('reviews_trained', 0) / 1e9:.2f}B | "
                     f"{t.get('training_hours', 0):.1f} | {rs} |")
    L += ["", "## Users needed (no-teacher arms, gap = the worse of the two heads)", "",
          "| threshold | vs SM118, point estimate | vs SM118, CI upper bound | vs the equal-compute teacher control (SM176 n8000-kd), point |",
          "|---|---|---|---|"]
    eq = res.get("users_needed_vs_equal_compute_teacher", {})
    for t, v in res["users_needed_vs_sm118"].items():
        p, u = v["point"], v["ci_upper"]
        L.append(f"| {t} | {p['users'] if p else 'pending'} | {u['users'] if u else 'pending'} | "
                 f"{eq[t]['users'] if eq.get(t) else 'pending'} |")
    L += ["", "## Held-out slice (early stopping; 190 validation users never trained on, not in the 732 set)", ""]
    for arm in chain.ORDER:
        st = (res["arms"][arm].get("training") or {}).get("state")
        if st:
            s = st["slice_mean_loss"]
            first = next(iter(s.values()))
            L.append(f"- {NAMES[arm]}: " + ", ".join(f"epoch {k}: {v:.4f} ({(v / first - 1) * 100:+.2f}%)" for k, v in s.items())
                     + f". {st['reason']}; decay {st['decay_epochs']} epoch(s) from epoch {st['best_epoch']}.")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def report(args):
    rows732 = chain.read_rows(chain.ROWS732)
    ref_rows = chain.read_rows(REF_ROWS)
    u723 = set(json.loads(U723.read_text())["users"])
    res = compute(rows732, ref_rows, u723)
    for arm in chain.ORDER:
        res["arms"][arm]["training"] = training_stats(arm)
    (HERE / "sm176-result.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    write_md(res, HERE / "sm176-result.md")
    print((HERE / "sm176-result.md").read_text(encoding="utf-8"))
    return 0 if res["status"] == "complete" else 2


def selftest(args):
    ref_rows = chain.read_rows(REF_ROWS)
    u723 = set(json.loads(U723.read_text())["users"])
    ref = ref_rows[REF_LABEL]
    scale = {"n1000": 1.016, "n2000": 1.008, "n4000": 1.004, CONTROL: 1.0}
    rows = {}
    for arm, s in scale.items():
        rows[label(arm)] = {}
        for u, r in ref.items():
            r2 = json.loads(json.dumps(r))
            for h in HEADS.values():
                r2["benchmark"][h]["LogLoss"] *= s
            rows[label(arm)][u] = r2
    res = compute(rows, ref_rows, u723)
    ok = True
    for arm, s in scale.items():
        for h in HEADS:
            v = res["arms"][arm]["vs_sm118"][h]
            good = abs(v["value"] - (s - 1) * 100) < 1e-9 and abs(v["ci"][0] - v["ci"][1]) < 1e-9
            ok &= good
    n1 = res["users_needed_vs_sm118"]["+1.0%"]["point"]["value"]
    n05 = res["users_needed_vs_sm118"]["+0.5%"]["point"]["value"]
    want1 = 2 ** (math.log2(1000) + (1.6 - 1.0) / (1.6 - 0.8))
    want05 = 2 ** (math.log2(2000) + (0.8 - 0.5) / (0.8 - 0.4))
    ok &= abs(n1 - want1) < 1e-6 and abs(n05 - want05) < 1e-6
    eq = res["users_needed_vs_equal_compute_teacher"]["+1.0%"]["value"]
    ok &= abs(eq - want1) < 1e-6
    far = crossing([(500, 3.0), (1000, 2.4), (2000, 2.0), (4000, 1.8), (8000, 1.7)], 1.0)
    print(json.dumps({"n_for_1pct": n1, "want": want1, "n_for_05pct": n05, "want05": want05, "not_reached": far["users"]}))
    print("SELFTEST OK" if ok else "SELFTEST FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("report", "selftest"))
    args = ap.parse_args()
    return {"report": report, "selftest": selftest}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
