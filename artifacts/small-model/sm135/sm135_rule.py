#!/usr/bin/env python
"""Six-layer Tiny with longer recovery (SM135): pre-registered rules (fixed 2026-11-20 before any GPU ticket; CPU, seconds).

Question: do more constant-LR recovery epochs before the two-pass cool-down close part of the gap between the six-layer
Tiny (SM130) and the unmodified Tiny? Chains: io = Instant-Only (SM130's epoch 41 -> epochs 42..45 -> the longer cool-down
for Tiny (SM115)'s two-pass cool-down with the show-time Instant-Only big model (SM99)'s targets); th = two-head (SM130's
epoch 32 -> epochs 33..36 -> the show-time two-head cool-down for Tiny (SM124)'s cool-down with the show-time big model
(SM91s2)'s targets). Trained heads: io instant; th curve + instant.
All numbers: show-time inputs (`e2s-query`), leak-free (neighbour), w8 weights, 722 rule users of the 732 set
(showtime_report.py pools(), sha256 pinned), review-weighted LogLoss, paired user bootstrap 2,000, seed 0, 95% CI.
"%" = 100 x (new / reference - 1) of the reference's own LogLoss (pct_of_ref).

EARLY RULES (`early --chain C --epoch N`, after every recovery epoch N, constant-LR checkpoints, w8+s8):
  x_h = 100 x (epoch N / epoch N - 1 - 1) per trained head (for N = FIRST, epoch N - 1 = SM130's last recovery epoch,
        its committed rows); y_h = 100 x (epoch N / SM130's last recovery epoch - 1).
  worse    = any trained head's 95% CI lower bound of x_h > 0 (the epoch made that head significantly worse).
  decision, in this order:
    non-finite rows                              -> stop
    worse, N = FIRST                             -> stop (the chain ends: "no go (early stop)", no cool-down)
    worse, N > FIRST                             -> cooldown_previous (recovery ends; the cool-down starts from N - 1),
                                                    except at N = FIRST + 1 when epoch FIRST itself has a trained head
                                                    with y_h > -0.05% -> stop
    N = FIRST + 1 and any trained head y_h > -0.05%  -> stop (two more epochs gained less than the minimum effect on a
                                                    head that the main rule needs to improve)
    N = LAST                                     -> cooldown (from N)
    otherwise                                    -> continue
MAIN RULE (`decide --chain C`, after the cool-down; published w8+s8 rows):
  d_h = 100 x (SM135 final / SM130 final - 1) per trained head, both w8+s8 (SM130 = its committed rows, sha256 pinned).
  go(C) iff for EVERY trained head: the 95% CI upper bound of d_h < 0 AND the point estimate d_h <= -0.05% (the repo's
  minimum-effect rule), on all 722 rule users with finite rows (a non-finite rule user -> no go; missing -> exit 2).
REPORTED (no verdict): vs Small (the show-time two-head cool-down for Small (SM118) / the longer Small Instant-Only model
  (SM113), published w8+s8 rows) for the published w8+s8, compact w8+s4c8nhx8 and card-only w8+s4hgx4c8hx8n rows, each
  with the budget check (CI upper bound <= +1.00%); the same format of SM130 (compact: SM130's rows; card-only: the
  six-layer Tiny with the card-only state (SM134)'s rows); vs the unmodified Tiny (the show-time two-head cool-down for
  Tiny (SM124) / the longer cool-down for Tiny (SM115)) at w8+s8; the early checks.

  python sm135_rule.py selftest
  python sm135_rule.py early --chain io --epoch 42   -> sm135-early-io-e42.{json,md}; exit 2 while rows are missing
  python sm135_rule.py decide --chain io             -> sm135-decision-io.{json,md}; exit 2 while rows are missing
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
G = HERE / "gpu"
WT = HERE.parents[2]
SMG = WT / "artifacts/small-model"
W = Path("C:/Users/jonat/workspace/wt")
REPORT = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/showtime/showtime_report.py")
REPORT_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"
EARLY_ROWS, ROWS = G / "sm135-early-rows.jsonl", G / "sm135-rows.jsonl"
R130E, R130, R134 = SMG / "sm130/gpu/sm130-early-rows.jsonl", SMG / "sm130/gpu/sm130-rows.jsonl", SMG / "sm134/gpu/sm134-rows.jsonl"
R124, R115, R118 = SMG / "sm124/gpu/sm124-rows.jsonl", SMG / "sm115/gpu/sm115-rows.jsonl", SMG / "sm118/gpu/sm118-rows.jsonl"
R113 = W / "sm113/artifacts/small-model/sm113/gpu/sm113-rows.jsonl"
PINS = {R130E: "f22c2f85ecab321880cee0cb6d4d8ff05b9771effabc4a0bede29d29095f6cf6",
        R130: "f4b38086f872762442d754752542e9d26606fc11fee10bd586c8e2e739f0a5b6",
        R134: "7b286b434f53e346dcf0618f764b71e6826bc54979413292aef3fb2e80eda44e",
        R124: "d68823daecc8d87a8197970961c4381f38f143c1f80c5701622cac3da7ae777d",
        R115: "992fc0b6a6c3c77d232067c9234592f7262d1a48ebb35c7eb07568524da3db0c",
        R118: "430ffe8a809c27079862acf0bfb57b12b279a8472aa5eff19dfb232db378befb",
        R113: "af104a0774b1aa7da7b26bc48ac667a9991d86efd8dda4ac8de7bb51a3fe281a"}
E2S = "@e2s-query"
PUBLISHED, COMPACT, CARD_ONLY = "w8+s8", "w8+s4c8nhx8", "w8+s4hgx4c8hx8n"
FORMATS = (PUBLISHED, COMPACT, CARD_ONLY)
CHAINS = {
    "io": {"heads": ("instant",), "first": 42, "last": 45, "sm130_last": "sm130-io-e41-st-const",
           "final": "sm135-io-dec2ep32k-t99", "sm130_final": "sm130-io-dec2ep32k-t99",
           "small": (R113, "sm113-inst-e35st-st-dec2ep32k-t99"), "tiny": (R115, "sm115-c1-e38st-st-dec2ep32k-t99"),
           "names": {"self": "the Instant-Only six-layer Tiny with longer recovery (SM135)",
                     "sm130": "the Instant-Only six-layer Tiny (SM130)", "small": "the longer Small Instant-Only model (SM113)",
                     "tiny": "the longer cool-down for Tiny (SM115)", "last": "the six-layer Tiny (SM130)'s epoch 41"}},
    "th": {"heads": ("curve", "instant"), "first": 33, "last": 36, "sm130_last": "sm130-th-e32-st-const",
           "final": "sm135-th-dec2ep32k-t91s2", "sm130_final": "sm130-th-dec2ep32k-t91s2",
           "small": (R118, "sm118-e26-dec2ep32k-t91s2"), "tiny": (R124, "sm124-c1-e29st-dec2ep32k-t91s2"),
           "names": {"self": "the two-head six-layer Tiny with longer recovery (SM135)",
                     "sm130": "the two-head six-layer Tiny (SM130)", "small": "the show-time two-head cool-down for Small (SM118)",
                     "tiny": "the show-time two-head cool-down for Tiny (SM124)", "last": "the six-layer Tiny (SM130)'s epoch 32"}},
}
MIN_EFFECT = 0.05    # % (main rule: point <= -0.05%; early no-gain bar: y_h > -0.05%)
CEILING = 1.0        # % (the Tiny-shrink budget vs Small, report only here)
KEY = {"curve": "ahead", "instant": "instant"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report():
    if sha(REPORT) != REPORT_SHA:
        raise SystemExit("[sm135-rule] showtime_report.py changed: refusing")
    spec = importlib.util.spec_from_file_location("showtime_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_pins():
    for path, want in PINS.items():
        if sha(path) != want:
            raise SystemExit(f"[sm135-rule] reference rows {path.name} changed: refusing")


def epoch_label(chain, n):
    return f"sm135-{chain}-e{n}-st-const"


def pct(x, h):
    return {"value": x[h]["pct_of_ref"], "ci": x[h]["pct_of_ref_ci"], "new": x[h]["new"], "ref": x[h]["ref"],
            "better": x[h]["users_better"], "worse": x[h]["users_worse"]}


def finite(rows, users, heads):
    return all(math.isfinite(rows[u][KEY[h]][0]) for u in users for h in heads)


def nonfinite_users(path, label):
    out = set()
    if Path(path).exists():
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("label") == label and "nonfinite" in r and "benchmark" not in r:
                out.add(r.get("user_id"))
    return out


def f(x):
    return f"{x['value']:+.3f}% [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"


def early_decision(chain, n, x, y, first_y=None, ok_finite=True):
    """x, y: {head: {"value", "ci"}}; first_y: epoch FIRST's y (needed when epoch FIRST + 1 is worse)."""
    c = CHAINS[chain]
    heads = c["heads"]
    if not ok_finite:
        return "stop"
    worse = any(x[h]["ci"][0] > 0 for h in heads)
    if worse:
        if n == c["first"]:
            return "stop"
        if n == c["first"] + 1 and (first_y is None or any(first_y[h]["value"] > -MIN_EFFECT for h in heads)):
            return "stop"
        return "cooldown_previous"
    if n == c["first"] + 1 and any(y[h]["value"] > -MIN_EFFECT for h in heads):
        return "stop"
    if n == c["last"]:
        return "cooldown"
    return "continue"


def early(chain, n, rows=EARLY_ROWS, out_dir=HERE, ref_rows=R130E):
    c = CHAINS[chain]
    if not c["first"] <= n <= c["last"]:
        raise SystemExit(f"early checks run after epochs {c['first']}..{c['last']}")
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    a = r.load([(rows, epoch_label(chain, n) + "/" + PUBLISHED + E2S, None)], have)
    last = r.load([(ref_rows, c["sm130_last"] + "/" + PUBLISHED + E2S, None)], have)
    prev = last if n == c["first"] else r.load([(rows, epoch_label(chain, n - 1) + "/" + PUBLISHED + E2S, None)], have)
    if not (have <= set(a) and have <= set(prev) and have <= set(last)):
        print(f"[sm135-rule] early {chain} e{n} pending: epoch {len(have & set(a))}, previous {len(have & set(prev))} of {len(users)}")
        return 2
    xr, yr = r.compare(a, prev, big, users), r.compare(a, last, big, users)
    x, y = {h: pct(xr, h) for h in c["heads"]}, {h: pct(yr, h) for h in c["heads"]}
    first_y = None
    if n == c["first"] + 1:
        p = out_dir / f"sm135-early-{chain}-e{c['first']}.json"
        if p.exists():
            first_y = json.loads(p.read_text())["vs_sm130_last"]
    ok = finite(a, users, c["heads"])
    decision = early_decision(chain, n, x, y, first_y, ok)
    start = {"cooldown": n, "cooldown_previous": n - 1}.get(decision)
    res = {"chain": chain, "epoch": n, "users": len(users), "heads": list(c["heads"]), "finite": ok,
           "vs_previous_epoch": x, "vs_sm130_last": y, "decision": decision, "cooldown_start_epoch": start,
           "previous": c["sm130_last"] if n == c["first"] else epoch_label(chain, n - 1)}
    name = f"sm135-early-{chain}-e{n}"
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    prev_name = c["names"]["last"] if n == c["first"] else f"epoch {n - 1}"
    L = [f"# Six-layer Tiny with longer recovery (SM135), {chain}: early check after epoch {n}", "",
         f"Decision: **{decision}**" + (f" (cool-down from epoch {start})" if start else "") +
         f". Pre-registered: stop / cool down from the previous epoch if any trained head is significantly worse than the "
         f"previous epoch (CI lower bound > 0); after epoch {c['first'] + 1} stop if any trained head gained less than "
         f"{MIN_EFFECT:.2f}% vs {c['names']['last']}. Constant-LR checkpoints, show-time inputs, {len(users)} users, w8+s8.", "",
         f"| head | epoch {n} | vs {prev_name} % [95% CI] | vs {c['names']['last']} % [95% CI] | better / worse (vs previous) |",
         "|---|---|---|---|---|"]
    for h in c["heads"]:
        L.append(f"| {h} | {x[h]['new']} | {f(x[h])} | {f(y[h])} | {x[h]['better']} / {x[h]['worse']} |")
    (out_dir / f"{name}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[sm135-rule] early {chain} e{n} {decision}" + (f" (cool-down from epoch {start})" if start else "") + ": " +
          ", ".join(f"{h} vs previous {f(x[h])}, vs SM130 last {f(y[h])}" for h in c["heads"]))
    return 0


def decide(chain, rows=ROWS, out_dir=HERE, r130=R130, r134=R134):
    c = CHAINS[chain]
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    lab = {fm: c["final"] + "/" + fm + E2S for fm in FORMATS}
    new = {fm: r.load([(rows, lab[fm], None)], have) for fm in FORMATS}
    bad = {fm: sorted(nonfinite_users(rows, lab[fm]) & have) for fm in FORMATS}
    missing = have - set(new[PUBLISHED]) - set(bad[PUBLISHED])
    if missing:
        print(f"[sm135-rule] decide {chain} pending: final w8+s8 rows {len(have & set(new[PUBLISHED]))} of {len(users)}")
        return 2
    small = r.load([(c["small"][0], c["small"][1] + "/" + PUBLISHED + E2S, None)], have)
    tiny = r.load([(c["tiny"][0], c["tiny"][1] + "/" + PUBLISHED + E2S, None)], have)
    old = {PUBLISHED: r.load([(r130, c["sm130_final"] + "/" + PUBLISHED + E2S, None)], have),
           COMPACT: r.load([(r130, c["sm130_final"] + "/" + COMPACT + E2S, None)], have),
           CARD_ONLY: r.load([(r134, c["sm130_final"] + "/" + CARD_ONLY + E2S, None)], have)}
    res = {"chain": chain, "users": len(users), "heads": list(c["heads"]), "min_effect_pct": MIN_EFFECT,
           "nonfinite_rule_users": bad, "formats": {}}
    pub_ok = not bad[PUBLISHED] and have <= set(new[PUBLISHED]) and finite(new[PUBLISHED], users, c["heads"])
    if pub_ok:
        m = r.compare(new[PUBLISHED], old[PUBLISHED], big, users)
        main = {h: pct(m, h) for h in c["heads"]}
        per = {h: main[h]["ci"][1] < 0 and main[h]["value"] <= -MIN_EFFECT for h in c["heads"]}
    else:
        main, per = {}, {h: False for h in c["heads"]}
    res["main"] = {"label": lab[PUBLISHED], "reference": c["sm130_final"] + "/" + PUBLISHED + E2S, "finite": pub_ok,
                   "vs_sm130": main, "head_pass": per}
    res["verdict"] = "go" if all(per.values()) else "no go"
    for fm in FORMATS:
        n = new[fm]
        if not (have <= set(n)) or not finite(n, users, c["heads"]):
            res["formats"][fm] = {"label": lab[fm], "complete_and_finite": False, "users": len(have & set(n))}
            continue
        e = {"label": lab[fm], "complete_and_finite": True,
             "vs_small_published": {h: pct(r.compare(n, small, big, users), h) for h in c["heads"]}}
        e["budget_pass"] = all(e["vs_small_published"][h]["ci"][1] <= CEILING for h in c["heads"])
        if have <= set(old[fm]):
            e["vs_sm130_same_format"] = {h: pct(r.compare(n, old[fm], big, users), h) for h in c["heads"]}
        if fm == PUBLISHED:
            e["vs_unmodified_tiny"] = {h: pct(r.compare(n, tiny, big, users), h) for h in c["heads"]}
        res["formats"][fm] = e
    res["early"] = {}
    for k in range(c["first"], c["last"] + 1):
        p = out_dir / f"sm135-early-{chain}-e{k}.json"
        if p.exists():
            res["early"][f"e{k}"] = json.loads(p.read_text())
    name = f"sm135-decision-{chain}"
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    N = c["names"]
    L = [f"# Six-layer Tiny with longer recovery (SM135), {chain}: decision", "",
         f"Verdict: **{res['verdict']}** (pre-registered: every trained head's 95% CI upper bound of 100 x ({N['self']} / "
         f"{N['sm130']} - 1) < 0 AND the point estimate <= -{MIN_EFFECT:.2f}%, published w8+s8, show-time inputs, "
         f"{len(users)} users of the 732 set).", "",
         f"| head | SM135 w8+s8 | vs {N['sm130']} % [CI] | head passes |", "|---|---|---|---|"]
    for h in c["heads"]:
        if main:
            L.append(f"| {h} | {main[h]['new']} | {f(main[h])} | {'yes' if per[h] else 'no'} |")
    L += ["", f"Reported (no verdict): vs {N['small']} (its published w8+s8; budget: CI upper bound <= +{CEILING:.2f}%), "
              f"vs {N['sm130']} in the same format, vs {N['tiny']} (w8+s8).", "",
          "| format | head | SM135 | vs Small % [CI] | budget | vs SM130 same format % [CI] | vs unmodified Tiny % [CI] |",
          "|---|---|---|---|---|---|---|"]
    for fm in FORMATS:
        e = res["formats"][fm]
        if not e["complete_and_finite"]:
            L.append(f"| {fm} | | not complete / not finite ({e['users']} users) | | | | |")
            continue
        for h in c["heads"]:
            v = e["vs_small_published"][h]
            s = e.get("vs_sm130_same_format", {}).get(h)
            t = e.get("vs_unmodified_tiny", {}).get(h)
            L.append(f"| {fm} | {h} | {v['new']} | {f(v)} | {'pass' if e['budget_pass'] else 'fail'} | "
                     f"{f(s) if s else ''} | {f(t) if t else ''} |")
    if any(bad.values()):
        L.append("\nNon-finite rule users: " + json.dumps(bad))
    (out_dir / f"{name}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[sm135-rule] decide {chain}: verdict {res['verdict']}; " +
          (", ".join(f"{h} vs SM130 {f(main[h])}" for h in c["heads"]) if main else "w8+s8 rows not finite") +
          "; vs Small " + ", ".join(f"{fm} " + "/".join(f(res['formats'][fm]['vs_small_published'][h]) for h in c["heads"])
                                     for fm in FORMATS if res["formats"][fm]["complete_and_finite"]))
    return 0


def selftest():
    import io
    import tempfile
    from contextlib import redirect_stdout
    r = report()
    check_pins()
    ok = {}

    def relabel(src, old, new, dst, factor=None, nonfinite_user=None, drop_user=None):
        with open(src, encoding="utf-8") as fi, open(dst, "a", encoding="utf-8") as g:
            for line in fi:
                x = json.loads(line)
                if x.get("label") != old or "benchmark" not in x:
                    continue
                if x.get("user_id") == drop_user:
                    continue
                x["label"] = new
                if factor:
                    for h, k in KEY.items():
                        if k in x["benchmark"] and h in factor:
                            x["benchmark"][k]["LogLoss"] *= factor[h]
                if x.get("user_id") == nonfinite_user:
                    x.pop("benchmark")
                    x["nonfinite"] = "selftest"
                g.write(json.dumps(x) + "\n")

    def run(fn, *a, **k):
        with redirect_stdout(io.StringIO()):
            return fn(*a, **k)

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        io41, th32 = CHAINS["io"]["sm130_last"] + "/w8+s8" + E2S, CHAINS["th"]["sm130_last"] + "/w8+s8" + E2S
        lab = lambda ch, n: epoch_label(ch, n) + "/w8+s8" + E2S   # noqa: E731

        def case(name, specs):
            d = t / name
            d.mkdir()
            rows = d / "early.jsonl"
            for ch, n, src_label, factor in specs:
                relabel(R130E, src_label, lab(ch, n), rows, factor)
            return d, rows

        def dec(d, ch, n):
            return json.loads((d / f"sm135-early-{ch}-e{n}.json").read_text())

        # 1. epoch 42 = SM130 epoch 41 (identical): x = 0 [0, 0] -> continue
        d, rows = case("same", [("io", 42, io41, None)])
        ok["e42_identical_continue"] = run(early, "io", 42, rows, d) == 0 and dec(d, "io", 42)["decision"] == "continue" \
            and dec(d, "io", 42)["vs_previous_epoch"]["instant"]["value"] == 0
        # 2. epoch 42 = SM130 epoch 39 (worse than 41 by ~0.7%) -> stop
        d, rows = case("worse42", [("io", 42, CHAINS["io"]["sm130_last"].replace("e41", "e39") + "/w8+s8" + E2S, None)])
        run(early, "io", 42, rows, d)
        x = dec(d, "io", 42)
        ok["e42_is_e39_worse_stop"] = x["decision"] == "stop" and x["vs_previous_epoch"]["instant"]["ci"][0] > 0
        # 3. e42 = e41, e43 = e41 x 0.999 (-0.1%) -> continue; e43 = e41 x 0.9998 (-0.02%) -> stop (no gain)
        d, rows = case("gain", [("io", 42, io41, None), ("io", 43, io41, {"instant": 0.999})])
        run(early, "io", 42, rows, d), run(early, "io", 43, rows, d)
        ok["e43_gain_0p1_continue"] = dec(d, "io", 43)["decision"] == "continue" and dec(d, "io", 43)["vs_sm130_last"]["instant"]["value"] == -0.1
        d, rows = case("nogain", [("io", 42, io41, None), ("io", 43, io41, {"instant": 0.9998})])
        run(early, "io", 42, rows, d), run(early, "io", 43, rows, d)
        ok["e43_gain_0p02_stop"] = dec(d, "io", 43)["decision"] == "stop"
        # 4. e44 worse than e43 -> cooldown_previous (from 43); e45 better than e44 -> cooldown (from 45)
        d, rows = case("prev", [("io", 42, io41, None), ("io", 43, io41, {"instant": 0.999}), ("io", 44, io41, {"instant": 1.002}),
                                ("io", 45, io41, {"instant": 0.998})])
        for n in (42, 43, 44, 45):
            run(early, "io", n, rows, d)
        ok["e44_worse_cooldown_from_43"] = dec(d, "io", 44)["decision"] == "cooldown_previous" and dec(d, "io", 44)["cooldown_start_epoch"] == 43
        ok["e45_last_cooldown_from_45"] = dec(d, "io", 45)["decision"] == "cooldown" and dec(d, "io", 45)["cooldown_start_epoch"] == 45
        # 5. e42 gained only -0.01%, e43 worse -> stop (the fallback epoch has no gain)
        d, rows = case("fallback", [("io", 42, io41, {"instant": 0.9999}), ("io", 43, io41, {"instant": 1.001})])
        run(early, "io", 42, rows, d), run(early, "io", 43, rows, d)
        ok["e43_worse_e42_no_gain_stop"] = dec(d, "io", 42)["decision"] == "continue" and dec(d, "io", 43)["decision"] == "stop"
        # 6. two-head: epoch 33 = SM130 epoch 30 (worse than 32) -> stop; e34 with only curve gaining -> stop
        d, rows = case("th", [("th", 33, CHAINS["th"]["sm130_last"].replace("e32", "e30") + "/w8+s8" + E2S, None)])
        run(early, "th", 33, rows, d)
        ok["th_e33_is_e30_stop"] = dec(d, "th", 33)["decision"] == "stop"
        d, rows = case("th2", [("th", 33, th32, None), ("th", 34, th32, {"curve": 0.995})])
        run(early, "th", 33, rows, d), run(early, "th", 34, rows, d)
        ok["th_e34_one_head_no_gain_stop"] = dec(d, "th", 34)["decision"] == "stop"
        ok["pending_exit_2"] = run(early, "io", 44, t / "same" / "early.jsonl", t / "same") == 2

        # final rule: SM130's final rows relabelled as SM135's
        def final(name, fac_io=None, fac_th=None, nonfinite=None, drop=None):
            d = t / ("final-" + name)
            d.mkdir()
            rows = d / "rows.jsonl"
            for ch, fac in (("io", fac_io), ("th", fac_th)):
                c = CHAINS[ch]
                for fm in (PUBLISHED, COMPACT):
                    relabel(R130, c["sm130_final"] + "/" + fm + E2S, c["final"] + "/" + fm + E2S, rows, fac,
                            nonfinite if fm == PUBLISHED else None, drop)
                relabel(R134, c["sm130_final"] + "/" + CARD_ONLY + E2S, c["final"] + "/" + CARD_ONLY + E2S, rows, fac)
            a, b = run(decide, "io", rows, d), run(decide, "th", rows, d)
            out = [json.loads((d / f"sm135-decision-{ch}.json").read_text()) if (d / f"sm135-decision-{ch}.json").exists() else None
                   for ch in ("io", "th")]
            return a, b, out[0], out[1]

        a, b, di, dt = final("same")
        vs = lambda dd, fm, h: (dd["formats"][fm]["vs_small_published"][h]["value"], dd["formats"][fm]["vs_small_published"][h]["ci"])  # noqa: E731
        ok["final_identical_no_go"] = a == 0 and b == 0 and di["verdict"] == "no go" and dt["verdict"] == "no go" \
            and di["main"]["vs_sm130"]["instant"]["value"] == 0
        ok["final_reproduces_sm130_vs_small"] = vs(di, PUBLISHED, "instant") == (0.564, [0.522, 0.607]) \
            and vs(dt, PUBLISHED, "curve") == (0.467, [0.398, 0.546]) and vs(dt, PUBLISHED, "instant") == (0.524, [0.484, 0.567]) \
            and vs(di, COMPACT, "instant") == (0.603, [0.56, 0.647]) and vs(dt, COMPACT, "curve")[0] == 0.483
        ok["final_reproduces_sm134_card_only"] = vs(di, CARD_ONLY, "instant") == (0.888, [0.832, 0.943]) \
            and vs(dt, CARD_ONLY, "curve") == (0.659, [0.585, 0.743]) and vs(dt, CARD_ONLY, "instant") == (0.78, [0.73, 0.833])
        ok["final_reproduces_tiny_gap"] = di["formats"][PUBLISHED]["vs_unmodified_tiny"]["instant"]["value"] == 0.378 \
            and dt["formats"][PUBLISHED]["vs_unmodified_tiny"]["curve"]["value"] == 0.349
        a, b, di, dt = final("better", {"instant": 0.995, "curve": 0.995}, {"instant": 0.995, "curve": 0.995})
        ok["final_minus_0p5_go"] = di["verdict"] == "go" and dt["verdict"] == "go" and di["main"]["vs_sm130"]["instant"]["value"] == -0.5 \
            and di["formats"][CARD_ONLY]["vs_sm130_same_format"]["instant"]["value"] == -0.5
        a, b, di, dt = final("tiny", {"instant": 0.9997}, {"instant": 0.995})
        ok["final_minus_0p03_no_go_min_effect"] = di["verdict"] == "no go" and di["main"]["vs_sm130"]["instant"]["ci"][1] < 0
        ok["final_th_curve_unchanged_no_go"] = dt["verdict"] == "no go" and dt["main"]["head_pass"] == {"curve": False, "instant": True}
        users, _ = r.pools()
        a, b, di, dt = final("nonfinite", {"instant": 0.995}, None, nonfinite=users[5])
        ok["final_nonfinite_no_go"] = a == 0 and di["verdict"] == "no go" and di["nonfinite_rule_users"][PUBLISHED] == [users[5]]
        a, b, di, dt = final("missing", drop=users[7])
        ok["final_missing_exit_2"] = a == 2 and b == 2
        ok["markdown"] = (t / "final-better" / "sm135-decision-io.md").exists() and (t / "same" / "sm135-early-io-e42.md").exists()
    good = all(ok.values())
    print(f"[sm135-rule] selftest {json.dumps(ok)} -> {'OK' if good else 'FAILED'}")
    return 0 if good else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["selftest", "early", "decide"])
    p.add_argument("--chain", choices=sorted(CHAINS), default="io")
    p.add_argument("--epoch", type=int, default=0)
    a = p.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "early":
        return early(a.chain, a.epoch)
    return decide(a.chain)


if __name__ == "__main__":
    raise SystemExit(main())
