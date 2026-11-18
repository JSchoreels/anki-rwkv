#!/usr/bin/env python
"""Six-layer Tiny (SM130): pre-registered rules (fixed 2026-11-18 before any GPU ticket; CPU, seconds).

Chains: th = two-head (from the Tiny two-head model (SM106)'s epoch 29, cool-down = the show-time two-head cool-down for
Tiny (SM124)'s recipe with the show-time big model (SM91s2)'s targets); io = Instant-Only (from the Tiny Instant-Only
model (SM101)'s epoch 38, cool-down = the longer cool-down for Tiny (SM115)'s recipe with the show-time Instant-Only big
model (SM99)'s targets). Trained heads: th curve + instant, io instant.
All numbers: show-time inputs (`e2s-query`), leak-free (neighbour), w8 weights, 722 rule users of the 732 set
(showtime_report.py pools(), sha256 pinned), review-weighted LogLoss, paired user bootstrap 2,000, seed 0, 95% CI.
"%" = 100 x (new / reference - 1) of the reference's own LogLoss (pct_of_ref).

EARLY STOPS (`early --chain C --epoch N`, constant-LR checkpoints, w8+s8):
  x_h = SM130's recovery epoch N - the uncut start epoch (SM106 epoch 29 / SM101 epoch 38), both constant-LR, per
  trained head.
  after the first recovery epoch (30 / 39): stop if any trained head's point estimate > +1.50%;
  after the last recovery epoch (32 / 41), before the cool-down: stop if any trained head's point estimate > +1.20%.
  Otherwise continue. (Remaining budget at the end: about +0.82% to +0.88% vs Small after Tiny's own +0.12% to
  +0.19%; the bars leave room for the cool-down to shrink the cut's cost by 30-45%, as the card cut of SM106 shrank
  its curve cost from +0.26% at epoch 27 to +0.11% after the cool-down.) A stop ends that chain with verdict
  "no go (early stop)"; the other chain still runs.
FINAL (`decide --chain C`):
  published format (w8+s8, the benchmark format of the family):
    go(C) = every trained head's 95% CI upper bound of 100 x (SM130 / Small - 1) <= +1.00%, on all 722 rule users with
            finite rows; Small = the show-time two-head cool-down for Small (SM118) (th) / the longer Small Instant-Only
            model (SM113) (io), their committed rows (sha256 pinned).
  compact format (w8+s4c8nhx8, only if the compact state for Tiny (SM129) went go for the matching model): the same
    rule on the compact rows -> "go in the compact format" (reported beside the verdict).
  Reported (no verdict): SM130 - the unmodified Tiny (SM124 / SM115) in % of Tiny's LogLoss; the early pairs.

  python sm130_rule.py selftest
  python sm130_rule.py early --chain th --epoch 30   -> sm130-early-th-e30.{json,md}; exit 2 while rows are missing
  python sm130_rule.py decide --chain th             -> sm130-decision-th.{json,md}; exit 2 while rows are missing
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
EARLY_ROWS, ROWS = G / "sm130-early-rows.jsonl", G / "sm130-rows.jsonl"
R124, R115, R118 = SMG / "sm124/gpu/sm124-rows.jsonl", SMG / "sm115/gpu/sm115-rows.jsonl", SMG / "sm118/gpu/sm118-rows.jsonl"
R113 = W / "sm113/artifacts/small-model/sm113/gpu/sm113-rows.jsonl"
R106E = W / "sm106/artifacts/small-model/sm106/gpu/sm106-early-rows.jsonl"
PINS = {R124: "d68823daecc8d87a8197970961c4381f38f143c1f80c5701622cac3da7ae777d",
        R115: "992fc0b6a6c3c77d232067c9234592f7262d1a48ebb35c7eb07568524da3db0c",
        R118: "430ffe8a809c27079862acf0bfb57b12b279a8472aa5eff19dfb232db378befb",
        R113: "af104a0774b1aa7da7b26bc48ac667a9991d86efd8dda4ac8de7bb51a3fe281a",
        R106E: "efce215f0ca4034baadff0cdd73d57dbf57b5e4bb63b4739f7f05fa87147a006"}
E2S = "@e2s-query"
CHAINS = {
    "th": {"heads": ("curve", "instant"), "first": 30, "last": 32, "start_label": "sm106-c1-e29-st-const",
           "final": "sm130-th-dec2ep32k-t91s2", "small": (R118, "sm118-e26-dec2ep32k-t91s2"),
           "tiny": (R124, "sm124-c1-e29st-dec2ep32k-t91s2"), "sm129_model": "two-head",
           "names": ("the show-time two-head cool-down for Small (SM118)", "the show-time two-head cool-down for Tiny (SM124)",
                     "the Tiny two-head model (SM106)'s epoch 29")},
    "io": {"heads": ("instant",), "first": 39, "last": 41, "start_label": "sm101-c1-e38-st-const",
           "final": "sm130-io-dec2ep32k-t99", "small": (R113, "sm113-inst-e35st-st-dec2ep32k-t99"),
           "tiny": (R115, "sm115-c1-e38st-st-dec2ep32k-t99"), "sm129_model": "instant-only",
           "names": ("the longer Small Instant-Only model (SM113)", "the longer cool-down for Tiny (SM115)",
                     "the Tiny Instant-Only model (SM101)'s epoch 38")},
}
BARS = {"first": 1.5, "last": 1.2}
CEILING = 1.0
PUBLISHED, COMPACT = "w8+s8", "w8+s4c8nhx8"
SM129_DECISION = W / "sm129/artifacts/small-model/sm129/sm129-decision.json"
KEY = {"curve": "ahead", "instant": "instant"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report():
    if sha(REPORT) != REPORT_SHA:
        raise SystemExit("[sm130-rule] showtime_report.py changed: refusing")
    spec = importlib.util.spec_from_file_location("showtime_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_pins():
    for path, want in PINS.items():
        if sha(path) != want:
            raise SystemExit(f"[sm130-rule] reference rows {path.name} changed: refusing")


def epoch_label(chain, n):
    return f"sm130-{chain}-e{n}-st-const"


def pct(x, h):
    return {"value": x[h]["pct_of_ref"], "ci": x[h]["pct_of_ref_ci"], "new": x[h]["new"], "ref": x[h]["ref"],
            "better": x[h]["users_better"], "worse": x[h]["users_worse"]}


def finite(rows, users, heads):
    return all(math.isfinite(rows[u][KEY[h]][0]) for u in users for h in heads)


def f(x):
    return f"{x['value']:+.3f}% [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"


def early_decision(x, heads, bar):
    return "stop" if any(x[h]["value"] > bar for h in heads) else "continue"


def early(chain, n, rows=EARLY_ROWS, out_dir=HERE, start=None):
    c = CHAINS[chain]
    if n not in (c["first"], c["last"]):
        raise SystemExit(f"early checks run after epochs {c['first']} and {c['last']} only")
    bar = BARS["first" if n == c["first"] else "last"]
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    a = r.load([(rows, epoch_label(chain, n) + "/" + PUBLISHED + E2S, None)], have)
    b = r.load([(start or rows, c["start_label"] + "/" + PUBLISHED + E2S, None)], have)
    if not (have <= set(a) and have <= set(b)):
        print(f"[sm130-rule] early {chain} e{n} pending: epoch {len(have & set(a))}, start {len(have & set(b))} of {len(users)}")
        return 2
    x = r.compare(a, b, big, users)
    res = {"chain": chain, "epoch": n, "bar_pct": bar, "users": len(users), "heads": list(c["heads"]),
           "new_minus_uncut_start": {h: pct(x, h) for h in c["heads"]}, "finite": finite(a, users, c["heads"])}
    res["decision"] = early_decision(res["new_minus_uncut_start"], c["heads"], bar) if res["finite"] else "stop"
    name = f"sm130-early-{chain}-e{n}"
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    L = [f"# Six-layer Tiny (SM130), {chain}: early check after epoch {n}", "",
         f"Decision: **{res['decision']}** (pre-registered: stop if any trained head is above +{bar:.2f}%). "
         f"Six-layer epoch {n} minus {c['names'][2]} (uncut, both constant LR), show-time inputs, {len(users)} users, "
         "% of the uncut epoch's LogLoss.", "", "| head | six-layer | uncut | % [95% CI] | better / worse |", "|---|---|---|---|---|"]
    for h, v in res["new_minus_uncut_start"].items():
        L.append(f"| {h} | {v['new']} | {v['ref']} | {f(v)} | {v['better']} / {v['worse']} |")
    (out_dir / f"{name}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[sm130-rule] early {chain} e{n} {res['decision']}: " +
          ", ".join(f"{h} {f(v)}" for h, v in res["new_minus_uncut_start"].items()))
    return 0


def sm129_compact_go(chain):
    if not SM129_DECISION.exists():
        return None
    d = json.loads(SM129_DECISION.read_text())
    return bool(d.get("arms", {}).get("compact", {}).get("pass", {}).get(CHAINS[chain]["sm129_model"]))


def decide(chain, rows=ROWS, out_dir=HERE):
    c = CHAINS[chain]
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    small = r.load([(c["small"][0], c["small"][1] + "/" + PUBLISHED + E2S, None)], have)
    tiny = r.load([(c["tiny"][0], c["tiny"][1] + "/" + PUBLISHED + E2S, None)], have)
    new = r.load([(rows, c["final"] + "/" + PUBLISHED + E2S, None)], have)
    if not (have <= set(new)):
        print(f"[sm130-rule] decide {chain} pending: final rows {len(have & set(new))} of {len(users)}")
        return 2
    vs_small = r.compare(new, small, big, users)
    vs_tiny = r.compare(new, tiny, big, users)
    res = {"chain": chain, "users": len(users), "heads": list(c["heads"]), "ceiling_pct": CEILING,
           "published": {"label": c["final"] + "/" + PUBLISHED + E2S, "finite": finite(new, users, c["heads"]),
                         "vs_small": {h: pct(vs_small, h) for h in c["heads"]},
                         "vs_unmodified_tiny": {h: pct(vs_tiny, h) for h in c["heads"]}}}
    res["published"]["pass"] = res["published"]["finite"] and all(res["published"]["vs_small"][h]["ci"][1] <= CEILING for h in c["heads"])
    res["verdict"] = "go" if res["published"]["pass"] else "no go"
    comp = r.load([(rows, c["final"] + "/" + COMPACT + E2S, None)], have)
    res["sm129_compact_go"] = sm129_compact_go(chain)
    if have <= set(comp):
        cs, ct = r.compare(comp, small, big, users), r.compare(comp, new, big, users)
        res["compact"] = {"label": c["final"] + "/" + COMPACT + E2S, "finite": finite(comp, users, c["heads"]),
                          "vs_small": {h: pct(cs, h) for h in c["heads"]}, "cost_vs_published": {h: pct(ct, h) for h in c["heads"]}}
        res["compact"]["pass"] = res["compact"]["finite"] and all(res["compact"]["vs_small"][h]["ci"][1] <= CEILING for h in c["heads"])
    for n in (c["first"], c["last"]):
        p = HERE / f"sm130-early-{chain}-e{n}.json"
        if p.exists():
            res[f"early_e{n}"] = json.loads(p.read_text())
    name = f"sm130-decision-{chain}"
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    L = [f"# Six-layer Tiny (SM130), {chain}: decision", "",
         f"Verdict: **{res['verdict']}** (pre-registered: every trained head's 95% CI upper bound of 100 x (six-layer / "
         f"Small - 1) <= +{CEILING:.2f}%, published w8+s8 format, show-time inputs, {len(users)} users of the 732 set)."
         + (f" Compact format (w8+s4c8nhx8): **{'go' if res['compact']['pass'] else 'no go'}**." if "compact" in res else ""), "",
         "| format | head | six-layer | vs " + c["names"][0] + " % [CI] | vs " + c["names"][1] + " % [CI] |", "|---|---|---|---|---|"]
    for h in c["heads"]:
        a, b = res["published"]["vs_small"][h], res["published"]["vs_unmodified_tiny"][h]
        L.append(f"| published w8+s8 | {h} | {a['new']} | {f(a)} | {f(b)} |")
        if "compact" in res:
            a = res["compact"]["vs_small"][h]
            L.append(f"| compact w8+s4c8nhx8 | {h} | {a['new']} | {f(a)} | (vs six-layer published {f(res['compact']['cost_vs_published'][h])}) |")
    (out_dir / f"{name}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[sm130-rule] decide {chain}: verdict {res['verdict']}" +
          (f", compact {'go' if res['compact']['pass'] else 'no go'}" if "compact" in res else "") + "; " +
          ", ".join(f"{h} vs Small {f(res['published']['vs_small'][h])}" for h in c["heads"]))
    return 0


def selftest():
    import io
    import tempfile
    from contextlib import redirect_stdout
    r = report()
    check_pins()
    users, big = r.pools()
    ok = {}
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)

        def relabel(src, old, new, dst, factor=None):
            with open(src, encoding="utf-8") as fi, open(dst, "a", encoding="utf-8") as g:
                for line in fi:
                    x = json.loads(line)
                    if x.get("label") != old:
                        continue
                    x["label"] = new
                    if factor and "benchmark" in x:
                        for h, k in factor.items():
                            if k in x["benchmark"]:
                                x["benchmark"][k]["LogLoss"] *= h
                    g.write(json.dumps(x) + "\n")
        # early: SM106 epoch 27 (arm) vs SM92 epoch 26 (as the "start"): the known +0.26% / +0.19% -> continue
        er = t / "early.jsonl"
        relabel(R106E, "sm106-c1-e27-st-const/w8+s8" + E2S, epoch_label("th", 30) + "/w8+s8" + E2S, er)
        relabel(R106E, "sm92-e26-const/w8+s8" + E2S, CHAINS["th"]["start_label"] + "/w8+s8" + E2S, er)
        with redirect_stdout(io.StringIO()):
            a = early("th", 30, er, t)
        d = json.loads((t / "sm130-early-th-e30.json").read_text())
        ok["early_sm106_e27_continue"] = a == 0 and d["decision"] == "continue" and 0.2 < d["new_minus_uncut_start"]["curve"]["value"] < 0.3
        er2 = t / "early2.jsonl"
        relabel(R106E, "sm106-c1-e27-st-const/w8+s8" + E2S, epoch_label("th", 32) + "/w8+s8" + E2S, er2, {1.0125: "instant"})
        relabel(R106E, "sm92-e26-const/w8+s8" + E2S, CHAINS["th"]["start_label"] + "/w8+s8" + E2S, er2)
        with redirect_stdout(io.StringIO()):
            early("th", 32, er2, t)
        d = json.loads((t / "sm130-early-th-e32.json").read_text())
        ok["early_last_bar_stops_at_1p2"] = d["decision"] == "stop" and d["new_minus_uncut_start"]["instant"]["value"] > 1.2
        ok["early_bar_edges"] = (early_decision({"curve": {"value": 1.5}, "instant": {"value": 0.0}}, ("curve", "instant"), 1.5) == "continue"
                                 and early_decision({"curve": {"value": 1.501}, "instant": {"value": 0.0}}, ("curve", "instant"), 1.5) == "stop"
                                 and early_decision({"instant": {"value": 1.21}}, ("instant",), 1.2) == "stop")
        # final: SM124 relabelled as the six-layer model -> the review's +0.118 / +0.183 vs SM118, go; x1.009 instant -> no go
        fr = t / "rows.jsonl"
        relabel(R124, CHAINS["th"]["tiny"][1] + "/w8+s8" + E2S, CHAINS["th"]["final"] + "/w8+s8" + E2S, fr)
        relabel(R115, CHAINS["io"]["tiny"][1] + "/w8+s8" + E2S, CHAINS["io"]["final"] + "/w8+s8" + E2S, fr)
        relabel(R124, CHAINS["th"]["tiny"][1] + "/w8+s8" + E2S, CHAINS["th"]["final"] + "/" + COMPACT + E2S, fr, {1.009: "instant"})
        with redirect_stdout(io.StringIO()):
            b = decide("th", fr, t)
            c = decide("io", fr, t)
        dt = json.loads((t / "sm130-decision-th.json").read_text())
        di = json.loads((t / "sm130-decision-io.json").read_text())
        ok["final_sm124_as_new_go_review_numbers"] = b == 0 and dt["verdict"] == "go" and \
            (dt["published"]["vs_small"]["curve"]["value"], dt["published"]["vs_small"]["curve"]["ci"]) == (0.118, [0.057, 0.174]) and \
            (dt["published"]["vs_small"]["instant"]["value"], dt["published"]["vs_small"]["instant"]["ci"]) == (0.183, [0.164, 0.204]) and \
            all(dt["published"]["vs_unmodified_tiny"][h]["value"] == 0 for h in ("curve", "instant"))
        ok["final_compact_scaled_no_go"] = "compact" in dt and dt["compact"]["pass"] is False
        ok["final_io_sm115_as_new_go"] = c == 0 and di["verdict"] == "go" and \
            (di["published"]["vs_small"]["instant"]["value"], di["published"]["vs_small"]["instant"]["ci"]) == (0.186, [0.166, 0.208])
        ok["markdown"] = all((t / n).exists() for n in ("sm130-decision-th.md", "sm130-early-th-e30.md"))
    good = all(ok.values())
    print(f"[sm130-rule] selftest {json.dumps(ok)} -> {'OK' if good else 'FAILED'}")
    return 0 if good else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["selftest", "early", "decide"])
    p.add_argument("--chain", choices=sorted(CHAINS), default="th")
    p.add_argument("--epoch", type=int, default=0)
    a = p.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "early":
        return early(a.chain, a.epoch)
    return decide(a.chain)


if __name__ == "__main__":
    raise SystemExit(main())
