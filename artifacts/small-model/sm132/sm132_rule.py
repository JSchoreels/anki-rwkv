#!/usr/bin/env python
"""Again pressed as Hard (SM132): pre-registered readout (fixed 2026-11-19 before any GPU ticket; CPU, seconds).

Question (user, Thu 11-19 ~15:40): how much worse does RWKV get if people press Hard instead of Again with probability
p? The user's bet: as long as p <= 20%, LogLoss is at most 2% worse.

Corruption (scripts/sm31_inputs.py again_as_hard, scorer --again-as-hard P): every Again review is recorded as Hard in
the MODEL INPUTS with probability p (seed [132, user], nested in p); the card state follows what Anki would record
(state "anki"); targets stay the true outcomes. Model: the show-time two-head cool-down for Small (SM118), published
format w8+s8, show-time inputs (e2s-query), leak-free (deleted cards neighbour), 723 users of the 732 set; rule pool =
the 722 users the big model's rows have (showtime_report.pools(), sha256 pinned). Reference: p = 0 scored by this run.

Statistic: per head, % = 100 x (corrupted / clean - 1) of the review-weighted LogLoss (pct_of_ref of
artifacts/small-model/showtime/showtime_report.py compare()), paired user bootstrap 2,000, seed 0, 95% CI, same users.
READOUT (pre-registered):
  the bet holds for a head  iff  the 95% CI upper bound is <= +2.0% at every p in {5%, 10%, 20%};
  verdict = "holds" if it holds for both heads (curve and instant), else "does not hold (<heads>)".
Users: the rule users with finite rows at p = 0 and at every p of the model; at least MIN_USERS = 718 of the 722,
otherwise `decide` exits 2 (pending).
Reported, no rule: p = 30% and 50%; the grade-only arm (state "keep") at 20%; the show-time big model (SM91s2) at 10% and
20% against its published rows (the same test, and Big minus Small with a paired bootstrap); LogLoss against the
corrupted labels (Anki's own view of such a collection, dominated by label noise); the Again share and the share of
reviews changed; p = 0 of this run vs the published SM118 rows.
Rows files are read through a filter that skips and counts unparseable lines (a torn or NUL-filled last line).

  python sm132_rule.py selftest   -> the readout's code on synthetic rows
  python sm132_rule.py decide     -> sm132-decision.{json,md}; exit 2 while rows are missing
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
G = HERE / "gpu"
WT = HERE.parents[2]
SMG = WT / "artifacts/small-model"
W = Path("C:/Users/jonat/workspace/wt")
REPORT = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/showtime/showtime_report.py")
REPORT_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"   # the same pin as SM96 ... SM131
ROWS = G / "sm132-rows.jsonl"
PUB118 = SMG / "sm118/gpu/sm118-rows.jsonl"
PUB91 = W / "sm91s2/artifacts/small-model/sm91s2/gpu/sm91s2-inputs.jsonl"
PINS = {PUB118: "430ffe8a809c27079862acf0bfb57b12b279a8472aa5eff19dfb232db378befb",
        PUB91: "e156b6fd907857b22f96d64035c06417b063c88d416a959bc91231ed85075c80"}
T118, T91 = "sm118-e26-dec2ep32k-t91s2", "sm91s2-big-showtime"
NAMES = {"sm118": "the show-time two-head cool-down for Small (SM118)", "sm91s2": "the show-time big model (SM91s2)"}
SMALL_PS = (5, 10, 20, 30, 50)
BIG_PS = (10, 20)
BET_PS = (5, 10, 20)
BET = 2.0
MIN_USERS = 718
HEADS = {"curve": "ahead", "instant": "instant"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report():
    got = sha(REPORT)
    if got != REPORT_SHA:
        raise SystemExit(f"[sm132-rule] showtime_report.py changed (sha256 {got[:12]}): refusing")
    spec = importlib.util.spec_from_file_location("showtime_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_pins(pins=PINS):
    for path, want in pins.items():
        if sha(path) != want:
            raise SystemExit(f"[sm132-rule] reference file {path} changed: refusing")


def clean_copy(path, tmp, counts):
    """A copy of `path` with only parseable JSON object lines (showtime_report.load would crash on a torn line)."""
    out = Path(tmp) / f"clean-{len(counts)}-{Path(path).name}"
    bad = 0
    with open(path, "rb") as f, open(out, "w", encoding="utf-8") as g:
        for raw in f:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError
            except ValueError:
                bad += 1
                continue
            g.write(json.dumps(row) + "\n")
    counts[str(path)] = bad
    return out


def lab(tag, p=None, grade_only=False):
    return f"{tag}/w8+s8@e2s-query" + ("" if p is None else f"+aah{p:g}" + ("g" if grade_only else ""))


def load_extra(path, label, users):
    """{user: (corrupt-label benchmark {head: (LogLoss, size)}, again_as_hard counts)} for one label."""
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("label") != label or "benchmark_corrupt_labels" not in r or r.get("user_id") not in users:
            continue
        b = r["benchmark_corrupt_labels"]
        out[r["user_id"]] = ({h: ((b[h]["LogLoss"], b[h]["size"]) if h in b else (float("nan"), 0)) for h in HEADS.values()},
                             r["again_as_hard"])
    return out


def finite_users(rows, users):
    return {u for u in users if u in rows and all(math.isfinite(rows[u][k][0]) and rows[u][k][1] > 0
                                                  for k in HEADS.values())}


def pct(x, h):
    return {"value": x[h]["pct_of_ref"], "ci": x[h]["pct_of_ref_ci"], "new": x[h]["new"], "ref": x[h]["ref"],
            "delta": x[h]["delta"], "delta_ci": x[h]["delta_ci"], "better": x[h]["users_better"], "worse": x[h]["users_worse"]}


def diff_pct(new_a, ref_a, new_b, ref_b, big, users):
    """(pct of model A) - (pct of model B) on the same users, paired user bootstrap 2,000, seed 0 (report only)."""
    idx = np.random.default_rng(0).integers(0, len(users), size=(2000, len(users)))
    out = {}
    for h, k in HEADS.items():
        w = np.array([big[u][k][1] for u in users], float)
        arr = [np.where(w > 0, np.array([d[u][k][0] for u in users]), 0.0) for d in (new_a, ref_a, new_b, ref_b)]
        na, ra, nb, rb = arr

        def val(ix):
            ww = w[ix]
            return (((na - ra)[ix] * ww).sum(-1) / (ra[ix] * ww).sum(-1) - ((nb - rb)[ix] * ww).sum(-1) / (rb[ix] * ww).sum(-1)) * 100
        boot = val(idx)
        out[h] = {"value": round(float(val(np.arange(len(users)))), 3),
                  "ci": [round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)]}
    return out


def counts(extra, users):
    keys = ("reviews", "again", "flipped", "again_review_state", "flipped_review_state", "state_rows_changed",
            "benchmark_reviews", "benchmark_again", "benchmark_flipped")
    tot = {k: int(sum(extra[u][1][k] for u in users)) for k in keys}
    tot["again_share_of_reviews"] = round(tot["again"] / tot["reviews"], 5)
    tot["changed_share_of_reviews"] = round(tot["flipped"] / tot["reviews"], 5)
    tot["again_share_of_benchmark_reviews"] = round(tot["benchmark_again"] / tot["benchmark_reviews"], 5)
    tot["flipped_share_of_again"] = round(tot["flipped"] / max(tot["again"], 1), 5)
    return tot


def decide(rows=ROWS, out_dir=HERE, pins=PINS):
    r = report()
    check_pins(pins)
    users, big = r.pools()
    have = set(users)
    res = {"readout": f"the bet holds for a head iff the 95% CI upper bound of 100 x (corrupted / clean - 1) is <= "
                      f"+{BET:.1f}% at every p in {{{', '.join(f'{p}%' for p in BET_PS)}}}; verdict holds iff both "
                      f"heads hold; rule users with finite rows at p = 0 and every p (at least {MIN_USERS} of "
                      f"{len(users)})", "pool_users": len(users), "unparseable_lines": {}}
    pending = []
    with tempfile.TemporaryDirectory() as tmp:
        src = clean_copy(rows, tmp, res["unparseable_lines"]) if Path(rows).exists() else None

        def load(label, path=None):
            path = path or src
            return r.load([(path, label, None)], have) if path else {}

        # --- Small: the readout
        ref = load(lab(T118, 0))
        news = {p: load(lab(T118, p)) for p in SMALL_PS}
        grade = load(lab(T118, 20, True))
        both = finite_users(ref, users)
        for p in SMALL_PS:
            both &= finite_users(news[p], users)
        both = sorted(both)
        small = {"name": NAMES["sm118"], "users": len(both), "missing_users": sorted(have - set(both))}
        res["small"] = small
        if len(both) < MIN_USERS:
            pending.append(f"sm118: {len(both)} of {len(users)} users with rows at p = 0 and every p")
        else:
            small["by_p"] = {}
            for p in SMALL_PS:
                x = r.compare(news[p], ref, big, both)
                ex = load_extra(src, lab(T118, p), have)
                cl = {u: v[0] for u, v in ex.items()}
                cboth = sorted(set(both) & finite_users(cl, users))
                y = r.compare(cl, ref, big, cboth) if len(cboth) >= MIN_USERS else None
                small["by_p"][str(p)] = {
                    **{h: pct(x, h) for h in HEADS},
                    "counts": counts(ex, both),
                    "corrupt_labels_vs_clean_true_labels": ({h: pct(y, h) for h in HEADS} if y else "incomplete")}
            small["holds"] = {h: all(small["by_p"][str(p)][h]["ci"][1] <= BET for p in BET_PS) for h in HEADS}
            bad = [h for h, v in small["holds"].items() if not v]
            small["verdict"] = "holds" if not bad else f"does not hold ({', '.join(bad)})"
            small["first_p_point_above_bet"] = {h: next((p for p in SMALL_PS if small["by_p"][str(p)][h]["value"] > BET),
                                                        None) for h in HEADS}
            gboth = sorted(set(both) & finite_users(grade, users))
            if len(gboth) >= MIN_USERS:
                small["grade_only_20"] = {"users": len(gboth),
                                          "vs_clean": {h: pct(r.compare(grade, ref, big, gboth), h) for h in HEADS},
                                          "vs_anki_state_20": {h: pct(r.compare(grade, news[20], big, gboth), h)
                                                               for h in HEADS}}
            else:
                small["grade_only_20"] = {"users": len(gboth), "note": "incomplete, not reported"}
            pub = r.load([(PUB118, lab(T118), None)], have)
            pb = sorted(set(both) & finite_users(pub, users))
            small["p0_vs_published"] = {h: pct(r.compare(ref, pub, big, pb), h) for h in HEADS}
            small["p0_vs_published_max_abs_user_diff"] = {
                h: max(abs(ref[u][k][0] - pub[u][k][0]) for u in pb) for h, k in HEADS.items()}
        # --- Big: report only (the same test at 10% and 20%, reference = published rows)
        bref = r.load([(PUB91, lab(T91), None)], have)
        bnews = {p: load(lab(T91, p)) for p in BIG_PS}
        bb = finite_users(bref, users)
        for p in BIG_PS:
            bb &= finite_users(bnews[p], users)
        bb = sorted(bb)
        bigm = {"name": NAMES["sm91s2"], "users": len(bb), "missing_users": sorted(have - set(bb)),
                "reference": "published rows (sm91s2-inputs.jsonl, the GPU smoke checks p = 0 against them)"}
        res["big"] = bigm
        if len(bb) >= MIN_USERS:
            bigm["by_p"] = {}
            for p in BIG_PS:
                x = r.compare(bnews[p], bref, big, bb)
                ex = load_extra(src, lab(T91, p), have)
                cl = {u: v[0] for u, v in ex.items()}
                cboth = sorted(set(bb) & finite_users(cl, users))
                y = r.compare(cl, bref, big, cboth) if len(cboth) >= MIN_USERS else None
                entry = {**{h: pct(x, h) for h in HEADS}, "counts": counts(ex, bb),
                         "corrupt_labels_vs_clean_true_labels": ({h: pct(y, h) for h in HEADS} if y else "incomplete")}
                if "by_p" in small:
                    common = sorted(set(bb) & set(both))
                    entry["big_minus_small_pct"] = {"users": len(common),
                                                    **diff_pct(bnews[p], bref, news[p], ref, big, common)}
                bigm["by_p"][str(p)] = entry
            bigm["holds_report_only"] = {h: all(bigm["by_p"][str(p)][h]["ci"][1] <= BET for p in BIG_PS) for h in HEADS}
        else:
            bigm["note"] = "incomplete, not reported"
    if pending:
        print("[sm132-rule] decide pending: " + "; ".join(pending))
        return 2
    res["verdict"] = res["small"]["verdict"]
    share = json.loads((HERE / "again-share.json").read_text()) if (HERE / "again-share.json").exists() else None
    if share:
        res["again_share_723_users"] = share
    (out_dir / "sm132-decision.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    write_md(res, out_dir)
    s = res["small"]["by_p"]
    print(f"[sm132-rule] verdict: the bet {res['verdict']} for {NAMES['sm118']} ({res['small']['users']} users; "
          + "; ".join(f"p {p}%: curve {f(s[str(p)]['curve'])}, instant {f(s[str(p)]['instant'])}" for p in BET_PS)
          + f"; unparseable lines {sum(res['unparseable_lines'].values())})")
    return 0


def f(x):
    return f"{x['value']:+.3f}% [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"


def write_md(res, out_dir):
    s = res["small"]
    L = ["# Again pressed as Hard (SM132): decision", "",
         f"Readout (pre-registered): {res['readout']}. Show-time inputs, leak-free, the 732 set, published format "
         "w8+s8, paired user bootstrap 2,000, seed 0. % = 100 x (corrupted / clean - 1) of the review-weighted LogLoss, "
         "targets = the true outcomes.", "",
         f"- **The user's bet {res['verdict']}** for {s['name']} ({s['users']} users): "
         + ", ".join(f"{h} {'holds' if v else 'does not hold'}" for h, v in s["holds"].items()), ""]
    bad = {k: v for k, v in res["unparseable_lines"].items() if v}
    L += [f"Unparseable row lines skipped: {sum(bad.values())}" + (f" ({bad})" if bad else ""), "",
          "| p | share of all reviews changed | curve % [CI] | instant % [CI] | clean LogLoss curve / instant | "
          "corrupted LogLoss curve / instant | within the bet |", "|---|---|---|---|---|---|---|"]
    for p, v in s["by_p"].items():
        within = (("yes" if all(v[h]["ci"][1] <= BET for h in HEADS) else "no") if int(p) in BET_PS
                  else "(not in the bet)")
        L.append(f"| {p}% | {100 * v['counts']['changed_share_of_reviews']:.2f}% | {f(v['curve'])} | {f(v['instant'])} | "
                 f"{v['curve']['ref']} / {v['instant']['ref']} | {v['curve']['new']} / {v['instant']['new']} | {within} |")
    c = s["by_p"][str(SMALL_PS[0])]["counts"]
    L += ["", f"Again share: {100 * c['again_share_of_reviews']:.2f}% of all reviews, "
          f"{100 * c['again_share_of_benchmark_reviews']:.2f}% of the scored (benchmark) reviews of the rule users.", ""]
    g = s.get("grade_only_20", {})
    if "vs_clean" in g:
        L += [f"Grade-only arm at 20% (recorded card states kept; report only): curve {f(g['vs_clean']['curve'])}, "
              f"instant {f(g['vs_clean']['instant'])} vs clean; vs the Anki-state arm at 20%: curve "
              f"{f(g['vs_anki_state_20']['curve'])}, instant {f(g['vs_anki_state_20']['instant'])}.", ""]
    L += ["## LogLoss against the corrupted labels (secondary, no rule; dominated by label noise)", "",
          "What Anki's own metrics would show for a collection with this habit: the same predictions scored against "
          "labels where a flipped Again counts as passed, vs the clean model on the true labels.", "",
          "| p | curve % [CI] | instant % [CI] |", "|---|---|---|"]
    for p, v in s["by_p"].items():
        y = v["corrupt_labels_vs_clean_true_labels"]
        if isinstance(y, dict):
            L.append(f"| {p}% | {f(y['curve'])} | {f(y['instant'])} |")
    b = res["big"]
    if "by_p" in b:
        L += ["", f"## {b['name']} (report only, reference = its published rows, {b['users']} users)", "",
              "| p | curve % [CI] | instant % [CI] | Big minus Small, curve | Big minus Small, instant |",
              "|---|---|---|---|---|"]
        for p, v in b["by_p"].items():
            d = v.get("big_minus_small_pct", {})
            L.append(f"| {p}% | {f(v['curve'])} | {f(v['instant'])} | {f(d['curve']) if d else 'n/a'} | "
                     f"{f(d['instant']) if d else 'n/a'} |")
    L += ["", f"p = 0 of this run vs the published rows of {s['name']}: curve {f(s['p0_vs_published']['curve'])}, "
          f"instant {f(s['p0_vs_published']['instant'])} (largest per-user difference "
          f"{max(s['p0_vs_published_max_abs_user_diff'].values()):.1e})."]
    if s["missing_users"]:
        L.append(f"\n{s['name']}: readout on {s['users']} users; missing {s['missing_users']}")
    (out_dir / "sm132-decision.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def selftest():
    """Synthetic rows from the published rows of SM118 / SM91s2 relabelled as the run's labels:
    (1) every p = the clean rows -> 0.000% everywhere, holds; a torn + NUL line counted;
    (2) curve x 1.03 at p = 20% -> "does not hold (curve)", instant holds; first p above the bet = 20;
    (3) both heads x 1.019 at p <= 20% -> holds (CI upper bound ~ +1.9%); x 1.021 -> does not hold (curve, instant);
    (4) 10 users dropped at p = 10% -> exit 2 (pending); 3 dropped -> readout on 719 users, reported."""
    import io
    from contextlib import redirect_stdout
    r = report()
    check_pins()
    users, _ = r.pools()
    ok = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def build(name, scale=None, drop=None, junk=False):
            rows = tmp / f"{name}.jsonl"
            with open(rows, "w", encoding="utf-8") as g:
                for tag, src, ps in ((T118, PUB118, (0, *SMALL_PS)), (T91, PUB91, BIG_PS)):
                    with open(src, encoding="utf-8") as fi:
                        for line in fi:
                            y0 = json.loads(line)
                            if y0.get("label") != lab(tag) or "benchmark" not in y0:
                                continue
                            for p in ps + ((20.0001,) if tag == T118 else ()):
                                y = json.loads(line)
                                grade = p == 20.0001
                                pp = 20 if grade else p
                                y["label"] = lab(tag, pp, grade)
                                if drop and tag == T118 and pp == 10 and y["user_id"] in drop:
                                    continue
                                for h, fac in (scale or {}).get(pp, {}).items() if tag == T118 and not grade else ():
                                    y["benchmark"][h]["LogLoss"] *= fac
                                y["benchmark_corrupt_labels"] = y["benchmark"]
                                n = y["benchmark"]["size"]
                                y["again_as_hard"] = {"p": pp / 100, "reviews": 2 * n, "again": n // 3,
                                                      "flipped": int(n // 3 * pp / 100), "again_review_state": n // 6,
                                                      "flipped_review_state": int(n // 6 * pp / 100),
                                                      "state_rows_changed": 0, "benchmark_reviews": n,
                                                      "benchmark_again": n // 6, "benchmark_flipped": int(n // 6 * pp / 100)}
                                g.write(json.dumps(y) + "\n")
            if junk:
                with open(rows, "ab") as g:
                    g.write(b'{"user_id": 1, "label": "torn\n' + b"\x00" * 300)
            return rows

        def run(rows):
            with redirect_stdout(io.StringIO()) as s:
                code = decide(rows, tmp)
            d = json.loads((tmp / "sm132-decision.json").read_text()) if code == 0 else None
            if d is not None:
                (tmp / "sm132-decision.json").unlink()
            return code, d, s.getvalue()

        code, d, _ = run(build("zero", junk=True))
        ok["zero_cost_holds"] = code == 0 and d["verdict"] == "holds" and all(
            d["small"]["by_p"][str(p)][h]["value"] == 0 for p in SMALL_PS for h in HEADS) and all(
            d["big"]["by_p"][str(p)][h]["value"] == 0 for p in BIG_PS for h in HEADS)
        ok["unparseable_lines_counted"] = code == 0 and sum(d["unparseable_lines"].values()) == 2
        ok["p0_vs_published_zero"] = code == 0 and all(d["small"]["p0_vs_published"][h]["value"] == 0 for h in HEADS)
        code, d, _ = run(build("curve20", scale={20: {"ahead": 1.03}}))
        ok["curve_x1.03_at_20_fails_curve_only"] = code == 0 and d["verdict"] == "does not hold (curve)" and \
            d["small"]["holds"]["instant"] and d["small"]["first_p_point_above_bet"]["curve"] == 20
        code, d, _ = run(build("in", scale={p: {"ahead": 1.019, "instant": 1.019} for p in BET_PS}))
        ok["x1.019_holds"] = code == 0 and d["verdict"] == "holds"
        code, d, _ = run(build("out", scale={p: {"ahead": 1.021, "instant": 1.021} for p in BET_PS}))
        ok["x1.021_does_not_hold"] = code == 0 and d["verdict"] == "does not hold (curve, instant)"
        code, d, _ = run(build("drop3", drop=set(users[:3])))
        ok["three_users_missing_reported"] = code == 0 and d["small"]["users"] == len(users) - 3 and \
            d["small"]["missing_users"] == sorted(users[:3])
        code, d, out = run(build("drop10", drop=set(users[:10])))
        ok["ten_users_missing_pending"] = code == 2 and "pending" in out
        ok["markdown"] = (tmp / "sm132-decision.md").exists()
    good = all(ok.values())
    print(f"[sm132-rule] selftest {json.dumps(ok)} -> {'OK' if good else 'FAILED'}")
    return 0 if good else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "decide"
    raise SystemExit({"selftest": selftest, "decide": decide}[cmd]())
