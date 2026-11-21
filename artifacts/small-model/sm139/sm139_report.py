#!/usr/bin/env python
"""Release-candidate checks for the family (SM139): pre-registered checks and report (fixed 2026-11-21 before any GPU
ticket; CPU, seconds; numpy only). Rules and their reasons: design.md.

Rows (this tree, GPU): gpu/sm139-rows-732.jsonl (the 732 set minus the 9 never-fit = 723 users) and
gpu/sm139-rows-ext.jsonl (users-sm93-score.json: the 765 validation users of the 1,500 pool outside the 732 set that fit
+ 6 check users of the 732 set). Labels `<tag>/<variant>@e2s-query`, show-time inputs, leak-free (neighbour), w8 weights.
Variants: w8+s8 = the published format (int8 card and note rows, float32 row scales, float shifts); w8+s8hx8 = the
release-candidate format (user decision 11-21: int8 card rows with float16 row scales and int8 shifts, plus the compact
note format, i.e. the same for notes).

Users: "732" = the 722 rule users of the 732 set (showtime_report.py pools(), sha256 pinned, as every 732-set readout);
"ext" = the users of users-sm93-score.json that are not in the 732 fit list (765); "all" = 732 + ext (1,487).
Statistic: review-weighted LogLoss (weights = the reference rows' benchmark sizes), % = 100 x (new / reference - 1) of
the reference's own LogLoss, paired user bootstrap 2,000 resamples, seed 0, 95% CI (showtime_report.compare).
Trained heads: two-head models curve + instant; Instant-Only models instant only (their curve head is untrained).

CHECKS (per release-candidate model; the overall status is "pass" iff every model passes C1 and C2, "incomplete" if C3
is not met, otherwise "fail"):
  C1 finite: no 'nonfinite' row and every benchmark LogLoss finite at w8+s8hx8, on both row files.
  C2 format cost: cost_h = 100 x (model at w8+s8hx8 / the same model at w8+s8 - 1), per trained head, on "732" and on
     "ext" separately, over the users with rows in both formats: pass iff the 95% CI upper bound <= +0.05% on both.
  C3 coverage: >= 720 of the 722 rule users and >= 760 of the 765 ext users with rows in both formats (fewer = GPU
     memory, not a model property: the result is reported as incomplete, not as a failure).
REPORT ONLY (no go / no-go): per size and kind, what users get from the update = the candidate at w8+s8hx8 vs the
  published 11-13 model at w8+s8 (and the model-only part: both at w8+s8), per trained head, on 732 / ext / all; the
  6 check users' ext rows vs their 732 rows; this tree's rows vs recorded rows of earlier runs (same label).

  python sm139_report.py selftest
  python sm139_report.py check                   pins (models, users files, showtime_report.py); exit 1 on a change
  python sm139_report.py smoke --gpu G --cpu C    every (user, label, head) benchmark LogLoss |GPU - CPU| <= 1e-5
  python sm139_report.py report                  -> sm139-result.{json,md}; exit 2 while rows are missing
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G = HERE / "gpu"
WT = HERE.parents[2]
SMG = WT / "artifacts/small-model"
W = Path("C:/Users/jonat/workspace/wt")
REPORT = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/showtime/showtime_report.py")
REPORT_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"
ROWS732, ROWSEXT = G / "sm139-rows-732.jsonl", G / "sm139-rows-ext.jsonl"
U732 = W / "sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json"
UEXT = W / "sm57/artifacts/small-model/sm57/stage4/users-sm93-score.json"
USERS_PINS = {U732: None, UEXT: None}   # filled from design-pins.json by `check` (written once by `pin`)
E2S = "@e2s-query"
PUB, RC = "w8+s8", "w8+s8hx8"
CEILING, TOL_SMOKE = 0.05, 1e-5
MIN_732, MIN_EXT = 720, 760
KEY = {"curve": "ahead", "instant": "instant"}
TWO, IO = ("curve", "instant"), ("instant",)
# key: (tag, plain name, heads, size, kind, published partner key)
RCM = {
    "sm91s2": ("sm91s2-big-showtime", "the show-time big model (SM91s2)", TWO, "Big", "two-head", "sm91s2"),
    "sm99": ("sm99-big-inst-e2", "the show-time instant-only big model (SM99)", IO, "Big", "Instant-Only", "sm99"),
    "sm118": ("sm118-e26-dec2ep32k-t91s2", "the show-time two-head cool-down for Small (SM118)", TWO, "Small", "two-head", "sm92"),
    "sm113": ("sm113-inst-e35st-st-dec2ep32k-t99", "the longer Small Instant-Only model (SM113)", IO, "Small", "Instant-Only", "sm98"),
    "sm130th": ("sm130-th-dec2ep32k-t91s2", "the six-layer Tiny two-head model (SM130)", TWO, "Tiny", "two-head", "sm106"),
    "sm130io": ("sm130-io-dec2ep32k-t99", "the six-layer Tiny Instant-Only model (SM130)", IO, "Tiny", "Instant-Only", "sm101"),
}
PUBM = {
    "sm91s2": ("sm91s2-big-showtime", "the show-time big model (SM91s2)"),
    "sm99": ("sm99-big-inst-e2", "the show-time instant-only big model (SM99)"),
    "sm92": ("sm92-e26-dec1ep32k", "the show-time retrain of the release (SM92)"),
    "sm98": ("sm98-inst-e35st-st-dec1ep32k", "the Small Instant-Only model (SM98)"),
    "sm106": ("sm106-c1-e29st-st-dec1ep32k", "the Tiny two-head model (SM106)"),
    "sm101": ("sm101-c1-e38st-st-dec1ep32k", "the Tiny Instant-Only model (SM101)"),
}
# recorded rows of earlier runs (report only; a label match on common users)
RECORDED = [W / "sm91s2/artifacts/small-model/sm91s2/gpu/sm91s2-inputs.jsonl", SMG / "sm118/gpu/sm118-rows.jsonl",
            W / "sm113/artifacts/small-model/sm113/gpu/sm113-rows.jsonl", SMG / "sm130/gpu/sm130-rows.jsonl",
            SMG / "sm137/gpu/sm137-rows.jsonl", W / "sm92/artifacts/small-model/sm92/gpu/sm92-inputs.jsonl",
            W / "sm106/artifacts/small-model/sm106/gpu/sm106-rows.jsonl", W / "sm106/artifacts/small-model/sm106/gpu/sm106-rows-1500.jsonl",
            W / "sm124/artifacts/small-model/sm124/gpu/sm124-rows-1500.jsonl"]
PINFILE = HERE / "design-pins.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def showtime():
    if sha(REPORT) != REPORT_SHA:
        raise SystemExit("[sm139] showtime_report.py changed: refusing")
    spec = importlib.util.spec_from_file_location("showtime_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def models_table():
    """The launcher's model table (models.sh): {tag: (kind:arch, path, sha256, role, size)}."""
    text = (HERE / "models.sh").read_text(encoding="utf-8")
    block = text.split('MODELS="', 1)[1].split('"', 1)[0]
    w = "C:/Users/jonat/workspace/wt"
    out = {}
    for line in block.splitlines():
        if line.strip():
            tag, ka, path, digest, role, size = line.strip().split("|")
            out[tag] = (ka, path.replace("$W", w).replace("$TRACK", "E89-F65e-F77a"), digest, role, size)
    return out


def labels():
    out = []
    for tag, (_, _, _, role, _) in models_table().items():
        out.append(f"{tag}/{PUB}{E2S}")
        if role == "rc":
            out.append(f"{tag}/{RC}{E2S}")
    return out


def pin():
    PINFILE.write_text(json.dumps({str(p): sha(p) for p in USERS_PINS} | {"showtime_report": REPORT_SHA}, indent=1) + "\n",
                       encoding="utf-8")
    print(f"[sm139] pinned {len(USERS_PINS)} users files -> {PINFILE.name}")


def check():
    bad = []
    for tag, (_, path, digest, _, _) in models_table().items():
        if not Path(path).exists() or sha(path) != digest:
            bad.append(f"model {tag}: {path}")
    pins = json.loads(PINFILE.read_text(encoding="utf-8"))
    for p in USERS_PINS:
        if sha(p) != pins[str(p)]:
            bad.append(f"users file {p}")
    if sha(REPORT) != REPORT_SHA:
        bad.append("showtime_report.py")
    tags = set(models_table())
    want = {v[0] for v in RCM.values()} | {v[0] for v in PUBM.values()}
    if tags != want:
        bad.append(f"model table tags {sorted(tags ^ want)}")
    print(f"[sm139] check {'OK' if not bad else 'FAILED'}: {len(models_table())} models, {len(USERS_PINS)} users files"
          + ("" if not bad else ": " + "; ".join(bad)))
    return 0 if not bad else 1


def read_rows(path):
    """{label: {user: row}} of benchmark rows and {label: set(users)} of non-finite rows."""
    rows, nonfinite = {}, {}
    if Path(path).exists():
        for raw in open(path, "rb"):
            try:
                r = json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if "error" in r:
                continue
            if "benchmark" in r:
                rows.setdefault(r["label"], {})[r["user_id"]] = r
            elif "nonfinite" in r:
                nonfinite.setdefault(r["label"], set()).add(r["user_id"])
    return rows, nonfinite


def as_loss(rows):
    """{user: {ahead|instant: (LogLoss, size)}} as showtime_report.load returns."""
    out = {}
    for u, r in rows.items():
        b = r["benchmark"]
        out[u] = {h: ((b[h]["LogLoss"], b[h]["size"]) if h in b else (float("nan"), 0)) for h in ("ahead", "instant")}
    return out


def pct(x, h):
    return {"value": x[h]["pct_of_ref"], "ci": x[h]["pct_of_ref_ci"], "new": x[h]["new"], "ref": x[h]["ref"],
            "better": x[h]["users_better"], "worse": x[h]["users_worse"], "users": x["users"]}


def fmt(x):
    return f"{x['value']:+.3f}% [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"


def smoke(gpu, cpu, tol=TOL_SMOKE):
    def vals(path):
        rows, nf = read_rows(path)
        out = {(u, lab, h): r["benchmark"][h]["LogLoss"] for lab, d in rows.items() for u, r in d.items() for h in ("ahead", "instant")}
        return out, {(u, lab) for lab, s in nf.items() for u in s}
    (g, gnf), (c, cnf) = vals(gpu), vals(cpu)
    want = {(u, lab, h) for u in json.loads((HERE / "users-smoke.json").read_text())["users"] for lab in labels() for h in ("ahead", "instant")}
    worst = max((abs(g[k] - c[k]) for k in c if k in g), default=float("inf"))
    ok = set(g) == set(c) == want and not gnf and not cnf and worst <= tol and all(math.isfinite(v) for v in g.values())
    print(f"[sm139] smoke {'OK' if ok else 'FAILED'}: {len(c)} CPU / {len(g)} GPU (user, label, head) values of {len(want)}, "
          f"max |GPU - CPU| {worst:.2e} (tol {tol:g}), non-finite GPU {len(gnf)} CPU {len(cnf)}")
    return 0 if ok else 1


def report(out_dir=HERE, rows732=ROWS732, rowsext=ROWSEXT):
    st = showtime()
    users732, _ = st.pools()
    fit732 = set(json.loads(U732.read_text())["users"])
    ext_all = json.loads(UEXT.read_text())["users"]
    users_ext = sorted(set(ext_all) - fit732)
    check_users = sorted(set(ext_all) & fit732)
    r732, nf732 = read_rows(rows732)
    rext, nfext = read_rows(rowsext)
    pools = {"732": (r732, nf732, users732), "ext": (rext, nfext, users_ext)}

    def get(pool, tag, var):
        rows, _, users = pools[pool]
        return as_loss({u: r for u, r in rows.get(f"{tag}/{var}{E2S}", {}).items() if u in set(users)})

    def both(pool, a, b):
        """users of the pool with rows in a and b, plus 'all' = 732 + ext merged."""
        if pool == "all":
            return {**a["732"], **a["ext"]}, {**b["732"], **b["ext"]}
        return a[pool], b[pool]

    def cmp(new, ref, heads):
        users = sorted(set(new) & set(ref))
        if not users:
            return None
        x = st.compare(new, ref, ref, users)
        return {h: pct(x, h) for h in heads}

    res = {"pools": {"732": len(users732), "ext": len(users_ext), "check_users": len(check_users)}, "models": {},
           "update": {}, "ceiling_pct": CEILING, "min_users": {"732": MIN_732, "ext": MIN_EXT}}
    missing = []
    status = "pass"
    for key, (tag, name, heads, size, kind, partner) in RCM.items():
        m = {"name": name, "size": size, "kind": kind, "heads": list(heads)}
        rc = {p: get(p, tag, RC) for p in ("732", "ext")}
        pb = {p: get(p, tag, PUB) for p in ("732", "ext")}
        nonfin = sorted((nf732.get(f"{tag}/{RC}{E2S}", set()) | nfext.get(f"{tag}/{RC}{E2S}", set())))
        nonfin_pub = sorted((nf732.get(f"{tag}/{PUB}{E2S}", set()) | nfext.get(f"{tag}/{PUB}{E2S}", set())))
        fin = not nonfin and all(math.isfinite(v[KEY[h]][0]) for p in rc for v in rc[p].values() for h in heads)
        cover = {p: len(set(rc[p]) & set(pb[p])) for p in rc}
        m.update({"finite": fin, "nonfinite_users": nonfin, "nonfinite_users_published_format": nonfin_pub,
                  "users_both_formats": cover, "users_rc_format": {p: len(rc[p]) for p in rc}})
        for p in ("732", "ext"):
            if cover[p] == 0:
                missing.append(f"{tag} {p}")
        m["format_cost"] = {p: cmp(*both(p, rc, pb), heads) for p in ("732", "ext", "all")}
        fc = m["format_cost"]
        m["C1"] = fin
        m["C2"] = bool(fc["732"] and fc["ext"] and all(fc[p][h]["ci"][1] <= CEILING for p in ("732", "ext") for h in heads))
        m["C3"] = cover["732"] >= MIN_732 and cover["ext"] >= MIN_EXT
        if not (m["C1"] and m["C2"]):
            status = "fail"
        elif not m["C3"] and status == "pass":
            status = "incomplete"
        res["models"][key] = m
        ptag, pname = PUBM[partner]
        pp = {p: get(p, ptag, PUB) for p in ("732", "ext")}
        res["update"][key] = {"candidate": name, "published": pname, "same_model": ptag == tag,
                              "rc_vs_published": {p: cmp(*both(p, rc, pp), heads) for p in ("732", "ext", "all")},
                              "model_only": {p: cmp(*both(p, pb, pp), heads) for p in ("732", "ext", "all")}}
    if missing:
        print(f"[sm139] report pending: no rows for {', '.join(missing)}")
        return 2
    # row checks (report only)
    rcheck = {}
    for lab in labels():
        a, b = r732.get(lab, {}), rext.get(lab, {})
        d = [max(abs(a[u]["benchmark"][h]["LogLoss"] - b[u]["benchmark"][h]["LogLoss"]) for h in ("ahead", "instant"))
             for u in check_users if u in a and u in b]
        rcheck[lab] = {"users": len(d), "max_abs": max(d) if d else None}
    res["check_users_ext_vs_732"] = rcheck
    recorded = {}
    for path in RECORDED:
        rows, _ = read_rows(path)
        for lab in labels():
            if lab in rows:
                for mine in (r732.get(lab, {}), rext.get(lab, {})):
                    common = [u for u in rows[lab] if u in mine]
                    if common:
                        d = max(abs(rows[lab][u]["benchmark"][h]["LogLoss"] - mine[u]["benchmark"][h]["LogLoss"])
                                for u in common for h in ("ahead", "instant") if h in rows[lab][u]["benchmark"])
                        e = recorded.setdefault(lab, {"files": [], "users": 0, "max_abs": 0.0})
                        e["files"].append(path.name); e["users"] += len(common); e["max_abs"] = max(e["max_abs"], d)
    res["recorded_rows_vs_this_tree"] = recorded
    res["status"] = status
    (out_dir / "sm139-result.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    write_md(res, out_dir)
    print(f"[sm139] report {status}: " + "; ".join(
        f"{k} C1 {m['C1']} C2 {m['C2']} C3 {m['C3']} ({m['users_both_formats']['732']}+{m['users_both_formats']['ext']} users)"
        for k, m in res["models"].items()))
    return 0


def write_md(res, out_dir):
    L = ["# Release-candidate checks for the family (SM139): result", "",
         f"Status: **{res['status']}** (C1 every user finite at the release-candidate format, C2 format cost CI upper bound "
         f"<= +{CEILING:.2f}% per trained head on the 732 set and on the extension, C3 coverage; design.md).", "",
         f"Users: the 732 set = {res['pools']['732']} rule users; the extension = {res['pools']['ext']} validation users of the "
         "1,500 pool outside the 732 set (ids >= 5000: never seen by the Small and Tiny models, but training users of the two "
         "Big models, so Big's extension numbers are a format and robustness check only); all = both. Show-time inputs, "
         "leak-free, w8 weights; % of the reference's own LogLoss, paired user bootstrap 95% CI.", "",
         "## Checks per release-candidate model (format cost = w8+s8hx8 vs the same model at w8+s8)", "",
         "| model | head | finite (C1) | cost on the 732 set | cost on the extension | cost on all | users 732 + ext (C3) | C2 |",
         "|---|---|---|---|---|---|---|---|"]
    for m in res["models"].values():
        for h in m["heads"]:
            fc = m["format_cost"]
            L.append(f"| {m['name']} | {h} | {m['C1']} | {fmt(fc['732'][h])} | {fmt(fc['ext'][h])} | {fmt(fc['all'][h])} | "
                     f"{m['users_both_formats']['732']} + {m['users_both_formats']['ext']} | {'pass' if m['C2'] else 'fail'} |")
    L += ["", "## What users get from the update (report only): the candidate at w8+s8hx8 vs the published 11-13 model at w8+s8", "",
          "| size | kind | candidate | published | head | 732 set | extension | all | LogLoss all (candidate / published) | model only, all (both w8+s8) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for key, u in res["update"].items():
        m = res["models"][key]
        for h in m["heads"]:
            a = u["rc_vs_published"]
            mo = u["model_only"]["all"][h]
            L.append(f"| {m['size']} | {m['kind']} | {u['candidate']} | {u['published']}{' (same model: format only)' if u['same_model'] else ''} | "
                     f"{h} | {fmt(a['732'][h])} | {fmt(a['ext'][h])} | {fmt(a['all'][h])} | {a['all'][h]['new']} / {a['all'][h]['ref']} | {fmt(mo)} |")
    L += ["", "## Row checks (report only)", "",
          "- The 6 check users, extension ticket vs 732 ticket, max |LogLoss difference| per label: "
          + ", ".join(f"{k.split('@')[0]} {v['max_abs']:.1e}" if v["max_abs"] is not None else f"{k.split('@')[0]} none"
                      for k, v in res["check_users_ext_vs_732"].items()),
          "- This tree vs recorded rows of earlier runs (same label, common users): "
          + (", ".join(f"{k.split('@')[0]} {v['users']} users max {v['max_abs']:.1e} ({'/'.join(sorted(set(v['files'])))})"
                       for k, v in res["recorded_rows_vs_this_tree"].items()) or "none found")]
    for m in res["models"].values():
        if m["nonfinite_users"] or m["nonfinite_users_published_format"]:
            L.append(f"- Non-finite users of {m['name']}: release-candidate format {m['nonfinite_users']}, published format "
                     f"{m['nonfinite_users_published_format']}")
    (out_dir / "sm139-result.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def selftest():
    """Synthetic rows through report(): a 1e-4 relative format cost passes C2; a 0.2% cost fails; a non-finite row fails C1."""
    import tempfile
    st = showtime()
    users732, _ = st.pools()
    fit732 = set(json.loads(U732.read_text())["users"])
    ext = sorted(set(json.loads(UEXT.read_text())["users"]) - fit732)
    rng = np.random.default_rng(1)
    results = {}
    for case, (cost, nonfin) in {"pass": (1e-6, False), "cost": (2e-3, False), "nonfinite": (1e-6, True)}.items():
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for path, us in ((td / "a.jsonl", users732), (td / "b.jsonl", ext)):
                with open(path, "w", encoding="utf-8") as f:
                    for u in us:
                        base = {"ahead": 0.3 + 0.05 * rng.standard_normal(), "instant": 0.27 + 0.05 * rng.standard_normal()}
                        n = int(rng.integers(100, 5000))
                        for tag, (_, _, _, role, _) in models_table().items():
                            for var in ((PUB, RC) if role == "rc" else (PUB,)):
                                k = 1 + (cost * (1 + 0.1 * rng.standard_normal()) if var == RC else 0)
                                if nonfin and var == RC and tag == "sm130-io-dec2ep32k-t99" and u == us[0]:
                                    f.write(json.dumps({"user_id": u, "label": f"{tag}/{var}{E2S}", "nonfinite": "x"}) + "\n")
                                    continue
                                b = {h: {"LogLoss": base[h] * k, "size": n} for h in ("ahead", "instant")}
                                f.write(json.dumps({"user_id": u, "label": f"{tag}/{var}{E2S}", "benchmark": b}) + "\n")
            rc = report(td, td / "a.jsonl", td / "b.jsonl")
            results[case] = (rc, json.loads((td / "sm139-result.json").read_text())["status"])
    ok = results == {"pass": (0, "pass"), "cost": (0, "fail"), "nonfinite": (0, "fail")}
    print(f"[sm139] selftest {'OK' if ok else 'FAILED'}: {results}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest"); sub.add_parser("check"); sub.add_parser("report"); sub.add_parser("pin")
    s = sub.add_parser("smoke"); s.add_argument("--gpu", required=True); s.add_argument("--cpu", required=True)
    a = ap.parse_args()
    if a.cmd == "smoke":
        return smoke(a.gpu, a.cpu)
    return {"selftest": selftest, "check": check, "report": report, "pin": pin}[a.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
