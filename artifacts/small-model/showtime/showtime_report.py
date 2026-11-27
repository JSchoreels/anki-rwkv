#!/usr/bin/env python
"""Show-time scoring of the headline models: report (CPU, seconds; numpy only). Reads rows, writes showtime-report.{json,md}.

Per model, on the 732 set (722 users = the 732 set that the big model (1 leak-free epoch) normal-input rows have):
  curve and instant LogLoss on normal and on show-time (e2s-query) inputs; points = % of the big model (1 leak-free epoch)
  normal-input LogLoss on the same users (curve 0.298966, instant 0.275757) for both input kinds; the show-time cost per
  model (show-time minus normal, points, paired user bootstrap CI).
Pairs, paired user bootstrap 2,000 resamples (seed 0, 95% CI; the SM92 `compare`), review-weighted:
  732 set (722 users), unit points (% of the big model normal-input LogLoss): stage 4 - stage 1 and stage 4 - the release
    of the instant-only small model (SM57), on show-time inputs (and on normal inputs for context);
  1,500 pool, unit % of the release's own LogLoss (SM93 convention): the same pairs on show-time inputs, plus a row check
    (the 6 check users scored by the 1,500 ticket vs their 732-set rows).
The instant-only models never trained the curve head: their curve numbers are printed but flagged and left out of the pairs'
verdicts.

Rows (all read only except the two new files):
  new       gpu/showtime-rows.jsonl (732 set + 3 long users), gpu/showtime-rows-1500.jsonl (1,500 pool)   [launch-showtime.sh]
  reused    release e2s (SM31), big model (1 leak-free epoch) e2s (SM31), show-time big model (SM91 step 2),
            show-time retrain of the release (SM92) e2s + normal, all normal rows of the other models.

  python artifacts/small-model/showtime/showtime_report.py           -> showtime-report.{json,md}
  python artifacts/small-model/showtime/showtime_report.py --check   -> re-derives the SM92 decision numbers from the old rows
                                                                       (release / SM92 / big e2s and normal); exit 0 if equal
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
G = HERE / "gpu"
W = Path("C:/Users/jonat/workspace/wt")
HEADS = {"ahead": "curve", "instant": "instant"}
NEW732, NEW1500 = G / "showtime-rows.jsonl", G / "showtime-rows-1500.jsonl"
E2S = "@e2s-query"
SM31 = W / "sm31/artifacts/small-model/sm31/gpu/sm31-inputs.jsonl"
S72 = W / "sm72s/artifacts/small-model/sm72s"
LF = S72 / "gpu/small-lf-neighbour.jsonl"
BIG_NORMAL = (W / "sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl", "big-leakfree-1ep/w8+s8", None)
U732, ULONG = S72 / "eval-users-732-fit.json", S72 / "users-732-long.json"
POOL1500 = W / "sm57/artifacts/small-model/eval-users-1500.json"
USERS1500 = W / "sm57/artifacts/small-model/sm57/stage4/users-sm93-score.json"
S57 = W / "sm57/artifacts/small-model/sm57"
SM92 = W / "sm92/artifacts/small-model/sm92"
REL = "s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/w8+s8"
S1, S4 = "sm57-inst-dec1ep32k/w8+s8", "sm57s4-inst-e30-dec1ep32k/w8+s8"
BIG_DENOM = {"curve": 0.298966, "instant": 0.275757}   # the big model (1 leak-free epoch), normal inputs, 722 users

# key: name (plain name, ID in parentheses), curve trained?, normal source, e2s sources (first wins), 1,500-pool e2s sources
MODELS = {
    "big": dict(name="the big model (1 leak-free epoch)", curve=True, normal=[BIG_NORMAL],
                e2s=[(SM31, "big-leakfree-1ep/w8+s8" + E2S, None)]),
    "showbig": dict(name="the show-time big model (SM91 step 2)", curve=True, normal=[],
                    e2s=[(W / "sm91s2/artifacts/small-model/sm91s2/gpu/sm91s2-inputs.jsonl", "sm91s2-big-showtime/w8+s8" + E2S, None)]),
    "release": dict(name="the release (SM63)", curve=True, normal=[(LF, REL, None)],
                    e2s=[(SM31, REL + E2S, None), (NEW732, REL + E2S, None)],
                    e2s1500=[(SM31, REL + E2S, None), (NEW1500, REL + E2S, None)]),
    "sm92": dict(name="the show-time retrain of the release (SM92)", curve=True,
                 normal=[(SM92 / "gpu/sm92-inputs.jsonl", "sm92-e26-dec1ep32k/w8+s8", None)],
                 e2s=[(SM92 / "gpu/sm92-inputs.jsonl", "sm92-e26-dec1ep32k/w8+s8" + E2S, None)]),
    "s1": dict(name="the instant-only small model (SM57) stage 1", curve=False, normal=[(S57 / "gpu/sm57-rows.jsonl", S1, None)],
               e2s=[(NEW732, S1 + E2S, None)], e2s1500=[(NEW732, S1 + E2S, None), (NEW1500, S1 + E2S, None)]),
    "s4": dict(name="the instant-only small model (SM57) stage 4", curve=False, normal=[(S57 / "gpu/sm57s4-rows.jsonl", S4, None)],
               e2s=[(NEW732, S4 + E2S, None)], e2s1500=[(NEW732, S4 + E2S, None), (NEW1500, S4 + E2S, None)]),
    "e20": dict(name="the previous release, cool-down from epoch 20 (SM72e20)", curve=True,
                normal=[(LF, "s-w64kd-final-sc-lanes-leakfree-dec20-1ep32k-muon/w8+s8", None)],
                e2s=[(NEW732, "s-w64kd-final-sc-lanes-leakfree-dec20-1ep32k-muon/w8+s8" + E2S, None)]),
    "e15": dict(name="the earlier release, cool-down from epoch 15 (SM72)", curve=True,
                normal=[(LF, "s-w64kd-final-sc-lanes-leakfree-dec15-1ep32k-muon/w8+s8", None)],
                e2s=[(NEW732, "s-w64kd-final-sc-lanes-leakfree-dec15-1ep32k-muon/w8+s8" + E2S, None)]),
    "sm56e15": dict(name="the small model, cool-down from epoch 15 (SM56)", curve=True,
                    normal=[(S72 / "gpu/small-dec15-neighbour.jsonl", "sm56dec15/w8+s8", None)],
                    e2s=[(NEW732, "sm56dec15/w8+s8" + E2S, None)]),
    "sm56e10": dict(name="the previous release, cool-down from epoch 10 (SM56 dec10)", curve=True,
                    normal=[(W / "sm71/artifacts/small-model/sm71/gpu/small-neighbour.jsonl", "sm56/w8+s8", "neighbour")],
                    e2s=[(NEW732, "sm56dec10/w8+s8" + E2S, None)]),
    "bigship": dict(name="the shipped big model (fork RWKV)", curve=True,
                    normal=[(W / "sm71/artifacts/small-model/sm71/gpu/big-neighbour.jsonl", "fork/w8+s8", "neighbour")],
                    e2s=[(NEW732, "fork-shipped-big/w8+s8" + E2S, None)]),
}
ORDER = ["release", "sm92", "s1", "s4", "e20", "e15", "sm56e15", "sm56e10", "bigship", "big", "showbig"]


def load(sources, users=None):
    """{user: {head: (LogLoss, size)}} on all reviews; the first source with a user wins; errors / other labels skipped."""
    out = {}
    for path, label, dc in sources:
        path = Path(path)
        if not path.exists():
            continue
        got = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("label") != label or "error" in r or "benchmark" not in r:
                    continue
                if dc is not None and r.get("deleted_cards") != dc:
                    continue
                b = r["benchmark"]
                got[r["user_id"]] = {h: ((b[h]["LogLoss"], b[h]["size"]) if h in b else (float("nan"), 0)) for h in HEADS}
        for u, v in got.items():
            if u not in out and (users is None or u in users):
                out[u] = v
    return out


def compare(new, ref, big, users):
    """new - ref per head over `users` (the SM92 rule's compare): raw review-weighted delta, points = % of `big` LogLoss,
    pct_of_ref = % of the reference's own LogLoss, paired user bootstrap 2,000, seed 0, 95% CI."""
    idx = np.random.default_rng(0).integers(0, len(users), size=(2000, len(users)))
    out = {"users": len(users)}
    for head, h in HEADS.items():
        a = np.array([new[u][head][0] for u in users]); b = np.array([ref[u][head][0] for u in users])
        c = np.array([big[u][head][0] for u in users])
        w = np.array([big[u][head][1] for u in users], float)
        a, b, c = (np.where(w > 0, x, 0.0) for x in (a, b, c))
        boot_raw = ((a - b)[idx] * w[idx]).sum(1) / w[idx].sum(1)
        boot = ((a - b)[idx] * w[idx]).sum(1) / (c[idx] * w[idx]).sum(1) * 100
        boot_pct = ((a - b)[idx] * w[idx]).sum(1) / (b[idx] * w[idx]).sum(1) * 100
        big_loss = (c * w).sum()
        out[h] = {"new": round(float((a * w).sum() / w.sum()), 6), "ref": round(float((b * w).sum() / w.sum()), 6),
                  "big": round(float(big_loss / w.sum()), 6), "delta": round(float(((a - b) * w).sum() / w.sum()), 6),
                  "delta_ci": [round(float(np.percentile(boot_raw, q)), 6) for q in (2.5, 97.5)],
                  "points": round(float(((a - b) * w).sum() / big_loss * 100), 3),
                  "points_ci": [round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)],
                  "pct_of_ref": round(float(((a - b) * w).sum() / (b * w).sum() * 100), 3),
                  "pct_of_ref_ci": [round(float(np.percentile(boot_pct, q)), 3) for q in (2.5, 97.5)],
                  "users_better": int((a < b).sum()), "users_worse": int((a > b).sum())}
    return out


def level(rows, big, users, head):
    """Review-weighted LogLoss of `rows` and points = % of the big model normal LogLoss on the same users."""
    w = np.array([big[u][head][1] for u in users], float)
    a = np.array([rows[u][head][0] for u in users])
    c = np.array([big[u][head][0] for u in users])
    ll = float((a * w).sum() / w.sum())
    return ll, (ll - float((c * w).sum() / w.sum())) / float((c * w).sum() / w.sum()) * 100


def pools():
    u732 = set(json.loads(U732.read_text())["users"])
    big = load([BIG_NORMAL], u732 | set(json.loads(ULONG.read_text())["users"]))
    return sorted(u732 & set(big)), big


def f3(x):
    return f"{x:+.3f}"


def fp(x):
    return f"{x['points']:+.3f} [{x['points_ci'][0]:+.3f}, {x['points_ci'][1]:+.3f}]"


def fpct(x):
    return f"{x['pct_of_ref']:+.3f} [{x['pct_of_ref_ci'][0]:+.3f}, {x['pct_of_ref_ci'][1]:+.3f}]"


def per_model(users, big):
    res = {}
    for k in ORDER:
        m = MODELS[k]
        have = set(users)
        e2s = load(m["e2s"], have)
        nor = load(m["normal"], have) if m["normal"] else {}
        e = {"name": m["name"], "curve_trained": m["curve"], "users_e2s": len(e2s), "users_normal": len(nor)}
        if k == "big":
            nor = load([BIG_NORMAL], have)
        us = sorted(set(users) & set(e2s) & set(nor)) if nor else sorted(set(users) & set(e2s))
        e["users"] = len(us)
        if not e2s:
            e["status"] = "show-time rows not scored yet"
            res[k] = e
            continue
        e["status"] = "ok" if len(us) == len(users) else f"{len(us)} of {len(users)} users"
        for head, h in HEADS.items():
            e[h] = {}
            ll, pt = level(e2s, big, us, head)
            e[h]["showtime"] = {"logloss": round(ll, 6), "points": round(pt, 3)}
            if nor and us:
                ll, pt = level(nor, big, us, head)
                e[h]["normal"] = {"logloss": round(ll, 6), "points": round(pt, 3)}
        if nor and us:
            cost = compare(e2s, nor, big, us)
            for head, h in HEADS.items():
                e[h]["cost"] = {k2: cost[h][k2] for k2 in ("delta", "delta_ci", "points", "points_ci", "users_better", "users_worse")}
        res[k] = e
    return res


def pairs732(users, big):
    out = {}
    sets = {k: {"e2s": load(MODELS[k]["e2s"], set(users)), "normal": load(MODELS[k]["normal"], set(users)) if MODELS[k]["normal"] else {}}
            for k in ("release", "s1", "s4")}
    for kind in ("e2s", "normal"):
        for a, b in (("s4", "s1"), ("s4", "release")):
            ra, rb = sets[a][kind], sets[b][kind]
            us = sorted(set(users) & set(ra) & set(rb))
            if us:
                out[f"{a}_minus_{b}_{kind}"] = compare(ra, rb, big, us)
    return out


def pool1500(big):
    """Show-time inputs on the 1,500 validation users (the SM93 convention: % of the release's own LogLoss)."""
    if not POOL1500.exists():
        return {}
    pool = set(json.loads(POOL1500.read_text())["users"])
    rel = load(MODELS["release"]["e2s1500"], pool)
    s1 = load(MODELS["s1"]["e2s1500"], pool)
    s4 = load(MODELS["s4"]["e2s1500"], pool)
    spec = json.loads(USERS1500.read_text())
    res = {"pool": "the 1,500 validation users (eval-users-1500.json), never-fit skipped", "never_fit": spec.get("never_fit")}
    users = sorted(set(s4) & set(s1) & set(rel))
    res["users"] = len(users)
    if not users:
        return res
    res["s4_minus_release"] = compare(s4, rel, rel, users)
    res["s4_minus_s1"] = compare(s4, s1, rel, users)
    res["s1_minus_release"] = compare(s1, rel, rel, users)
    old, new = [u for u in users if u < 5000], [u for u in users if u >= 5000]
    res["by_part"] = {"ids < 5000 (the 732 set)": compare(s4, rel, rel, old) if old else None,
                      "ids >= 5000": compare(s4, rel, rel, new) if new else None}
    # row check: the check users (732 set users inside the 1,500-ticket list) scored by that ticket vs their 732-set rows
    u732 = set(json.loads(U732.read_text())["users"]) | set(json.loads(ULONG.read_text())["users"])
    check = sorted(set(spec["users"] if isinstance(spec["users"], list) else []) & u732)
    rc = {}
    for k, lab_src in (("s1", S1), ("s4", S4), ("release", REL)):
        a = load([(NEW732, lab_src + E2S, None)], set(check)) if k != "release" else load([(SM31, REL + E2S, None)], set(check))
        b = load([(NEW1500, lab_src + E2S, None)], set(check))
        d = [max(abs(a[u][h][0] - b[u][h][0]) for h in HEADS) for u in check if u in a and u in b]
        rc[k] = {"users": len(d), "max_abs_logloss_diff": max(d) if d else None}
    res["row_check"] = rc
    return res


def write(res):
    (HERE / "showtime-report.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    md = ["# Show-time scoring of the headline models", "",
          "Leak-free scoring (deleted cards: neighbour), w8+s8, review-weighted LogLoss. Show-time inputs = `e2s-query` (what the "
          "Anki fork gives at a query). 732 set = 722 users. Points = % of the big model (1 leak-free epoch) normal-input "
          f"LogLoss on the same users (curve {BIG_DENOM['curve']}, instant {BIG_DENOM['instant']}), for both input kinds. Cost = "
          "show-time minus normal on the same model (points, paired user bootstrap 2,000, seed 0, 95% CI). The instant-only "
          "models never trained the curve head: their curve numbers are not meaningful.", ""]
    for h in ("curve", "instant"):
        md += [f"## {h.capitalize()} head, 732 set", "",
               "| model | users | normal LogLoss | show-time LogLoss | normal points | show-time points | show-time cost (points) [95% CI] |",
               "|---|---|---|---|---|---|---|"]
        for k in ORDER:
            e = res["models"][k]
            if "curve" not in e:
                md.append(f"| {e['name']} | {e['users']} | - | - | - | - | {e['status']} |")
                continue
            if h == "curve" and not e["curve_trained"]:
                md.append(f"| {e['name']} | {e['users']} | n/a (curve head untrained) | | | | |")
                continue
            x = e[h]
            n, s, c = x.get("normal"), x["showtime"], x.get("cost")
            md.append(f"| {e['name']} | {e['users']} | {n['logloss'] if n else '-'} | {s['logloss']} | {f3(n['points']) if n else '-'} | "
                      f"{f3(s['points'])} | {fp(c) if c else '-'} |")
        md.append("")
    p = res["pairs_732"]
    md += ["## Pairs on the 732 set (722 users), unit points (% of the big model normal-input LogLoss)", "",
           "| comparison | inputs | head | model | ref | points [95% CI] | users better / worse |", "|---|---|---|---|---|---|---|"]
    names = {"s4_minus_s1": "stage 4 - stage 1 of the instant-only small model (SM57)", "s4_minus_release": "stage 4 of the instant-only small model (SM57) - the release (SM63)"}
    for key, x in p.items():
        base, kind = key.rsplit("_", 1)
        for h in ("instant", "curve"):
            flag = " (curve head untrained)" if h == "curve" else ""
            md.append(f"| {names[base]} | {'show-time' if kind == 'e2s' else 'normal'} | {h}{flag} | {x[h]['new']} | {x[h]['ref']} | "
                      f"{fp(x[h])} | {x[h]['users_better']} / {x[h]['users_worse']} |")
    md.append("")
    q = res["pool_1500"]
    if q.get("users"):
        md += [f"## 1,500 validation users, show-time inputs ({q['users']} users), unit % of the release's own LogLoss", "",
               "| comparison | head | model | ref | % [95% CI] | users better / worse |", "|---|---|---|---|---|---|"]
        for key, nm in (("s4_minus_release", "stage 4 - the release"), ("s4_minus_s1", "stage 4 - stage 1"), ("s1_minus_release", "stage 1 - the release")):
            x = q[key]["instant"]
            md.append(f"| {nm} | instant | {x['new']} | {x['ref']} | {fpct(x)} | {x['users_better']} / {x['users_worse']} |")
        for part, x in q["by_part"].items():
            if x:
                y = x["instant"]
                md.append(f"| stage 4 - the release, {part} ({x['users']} users) | instant | {y['new']} | {y['ref']} | {fpct(y)} | {y['users_better']} / {y['users_worse']} |")
        md += ["", f"Row check (check users re-scored by the 1,500 ticket vs their 732-set rows): {json.dumps(q['row_check'])}", ""]
    else:
        md += ["## 1,500 validation users, show-time inputs", "", "Not scored yet.", ""]
    (HERE / "showtime-report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def check():
    """The SM92 decision numbers re-derived from the old rows: release / SM92 / big e2s and normal."""
    users, big = pools()
    d = json.loads((SM92 / "sm92-decision.json").read_text())["rule_pool"]
    new_e = load(MODELS["sm92"]["e2s"], set(users)); new_n = load(MODELS["sm92"]["normal"], set(users))
    rel_e = load([(SM31, REL + E2S, None)], set(users)); rel_n = load(MODELS["release"]["normal"], set(users))
    big_e = load(MODELS["big"]["e2s"], set(users))
    got = {"new_minus_release_e2s": compare(new_e, rel_e, big, users), "new_e2s_minus_new_normal": compare(new_e, new_n, big, users),
           "release_e2s_minus_release_normal": compare(rel_e, rel_n, big, users), "release_e2s_minus_big_e2s": compare(rel_e, big_e, big, users),
           "release_e2s_minus_big_normal": compare(rel_e, big, big, users), "new_normal_minus_release_normal": compare(new_n, rel_n, big, users)}
    ok = len(users) == d["users"]
    print(f"users {len(users)} (SM92: {d['users']}); big normal curve {big_denominator(big, users, 'ahead'):.6f} instant {big_denominator(big, users, 'instant'):.6f}")
    for k, x in got.items():
        for h in ("curve", "instant"):
            same = all(abs(x[h][f] - d[k][h][f]) < 1e-9 for f in ("new", "ref", "delta", "points", "pct_of_ref")) and \
                x[h]["points_ci"] == d[k][h]["points_ci"] and x[h]["users_better"] == d[k][h]["users_better"]
            ok &= same
            print(f"  {k} {h}: points {fp(x[h])}  SM92 {d[k][h]['points']:+.3f} [{d[k][h]['points_ci'][0]:+.3f}, {d[k][h]['points_ci'][1]:+.3f}]  {'OK' if same else 'DIFF'}")
    print("CHECK", "OK" if ok else "FAILED")
    return 0 if ok else 1


def big_denominator(big, users, head):
    w = np.array([big[u][head][1] for u in users], float)
    return float((np.array([big[u][head][0] for u in users]) * w).sum() / w.sum())


def main():
    ap = ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        return check()
    users, big = pools()
    res = {"pool_732": {"users": len(users), "big_curve": round(big_denominator(big, users, "ahead"), 6),
                        "big_instant": round(big_denominator(big, users, "instant"), 6)},
           "models": per_model(users, big), "pairs_732": pairs732(users, big), "pool_1500": pool1500(big)}
    write(res)
    print(f"[showtime-report] {len(users)} users; models with show-time rows: "
          f"{[k for k in ORDER if 'curve' in res['models'][k]]}; 1,500 pool users: {res['pool_1500'].get('users', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
