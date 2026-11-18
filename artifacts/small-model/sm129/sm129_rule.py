#!/usr/bin/env python
"""Compact state for Tiny (SM129): pre-registered rule (fixed 2026-11-18 before any GPU ticket; CPU, seconds).

Models (no training): the show-time two-head cool-down for Tiny (SM124) (two-head, w64-c1) and the longer cool-down for
Tiny (SM115) (Instant-Only, w64-c1), scored on show-time inputs (`e2s-query`), leak-free (`--split-existing
--deleted-cards neighbour`), 723 users of the 732 set (rule pool: the 722 users the big model's rows have), w8 weights,
with these stored-state formats (scripts/small_model_heldout.py variants):
  published   w8+s8            int8 card and note rows, float32 row scales (the published benchmark format; rows exist)
  fork        w8+s4c8n         int4 card rows, int8 note rows (what the fork stores today; report only)
  compact     w8+s4c8nhx8      the compact int state test (SM47): int4 cards + int8 notes, float16 row scales, int8
                               token-shift vectors (one float16 scale each)                       <- arm 1
  card-only   w8+s4hgx4c8hx8n  the second integer-state test (SM49)'s arm C': int4 cards with one float16 scale per
                               head and int4 card shift vectors, SM47's notes                     <- arm 2
References (committed rows, sha256 pinned, never rescored here): the show-time two-head cool-down for Small (SM118)
(curve 0.300016, instant 0.276599) and the longer Small Instant-Only model (SM113) (instant 0.275933), published w8+s8;
SM124 and SM115 published w8+s8. Report only: SM118 and SM113 at the fork's w8+s4c8n (scored by this run).

Statistic: per head, % = 100 x (Tiny arm / Small - 1) of the review-weighted LogLoss (pct_of_ref of
artifacts/small-model/showtime/showtime_report.py compare(), sha256 pinned), paired user bootstrap 2,000, seed 0, 95% CI.
RULE (per arm, per model):
  pass(model, arm) = every head's 95% CI upper bound <= +1.00% (two-head: curve and instant vs SM118; Instant-Only:
                     instant vs SM113), on all 722 rule users with finite rows.
  verdict(arm)     = "go" if both models pass, "go for <model> only" if one passes, "no go" otherwise.
Reported with it (no verdict): the cost vs the unmodified Tiny (arm - published, arm - fork, % of Tiny's own LogLoss,
CIs), the arm vs Small at the fork's w8+s4c8n, and the serialized bytes per card and per note (sm129-bytes.json).

  python sm129_rule.py selftest   -> the rule's code on known rows (reproduces the outside review's Tiny-vs-Small CIs)
  python sm129_rule.py decide     -> sm129-decision.{json,md}; exit 2 while rows are missing
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
G = HERE / "gpu"
WT = HERE.parents[2]
SMG = WT / "artifacts/small-model"
REPORT = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/showtime/showtime_report.py")
REPORT_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"   # the same pin as SM96 ... SM124
ROWS = G / "sm129-rows.jsonl"
R124, R115, R118 = SMG / "sm124/gpu/sm124-rows.jsonl", SMG / "sm115/gpu/sm115-rows.jsonl", SMG / "sm118/gpu/sm118-rows.jsonl"
R113 = Path("C:/Users/jonat/workspace/wt/sm113/artifacts/small-model/sm113/gpu/sm113-rows.jsonl")
PINS = {R124: "d68823daecc8d87a8197970961c4381f38f143c1f80c5701622cac3da7ae777d", R115: "992fc0b6a6c3c77d232067c9234592f7262d1a48ebb35c7eb07568524da3db0c",
        R118: "430ffe8a809c27079862acf0bfb57b12b279a8472aa5eff19dfb232db378befb",
        R113: "af104a0774b1aa7da7b26bc48ac667a9991d86efd8dda4ac8de7bb51a3fe281a"}
TAG = {"sm124": "sm124-c1-e29st-dec2ep32k-t91s2", "sm115": "sm115-c1-e38st-st-dec2ep32k-t99",
       "sm118": "sm118-e26-dec2ep32k-t91s2", "sm113": "sm113-inst-e35st-st-dec2ep32k-t99"}
PUBLISHED, FORK, COMPACT, CARD_ONLY = "w8+s8", "w8+s4c8n", "w8+s4c8nhx8", "w8+s4hgx4c8hx8n"
ARMS = {"compact": COMPACT, "card-only": CARD_ONLY}
MODELS = {"two-head": {"tiny": "sm124", "small": "sm118", "heads": ("curve", "instant"), "rows": R124, "small_rows": R118},
          "instant-only": {"tiny": "sm115", "small": "sm113", "heads": ("instant",), "rows": R115, "small_rows": R113}}
CEILING = 1.0
NAMES = {"sm124": "the show-time two-head cool-down for Tiny (SM124)", "sm115": "the longer cool-down for Tiny (SM115)",
         "sm118": "the show-time two-head cool-down for Small (SM118)", "sm113": "the longer Small Instant-Only model (SM113)"}
# the outside review (artifacts/small-model/tiny-smaller-codex/analysis.md), published formats, % of Small's LogLoss
REVIEW = {("sm124", "sm118"): {"curve": (0.118, [0.057, 0.174]), "instant": (0.183, [0.164, 0.204])},
          ("sm115", "sm113"): {"instant": (0.186, [0.166, 0.208])}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report():
    got = sha(REPORT)
    if got != REPORT_SHA:
        raise SystemExit(f"[sm129-rule] showtime_report.py changed (sha256 {got[:12]}): refusing")
    spec = importlib.util.spec_from_file_location("showtime_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_pins():
    for path, want in PINS.items():
        if want and sha(path) != want:
            raise SystemExit(f"[sm129-rule] reference rows {path.name} changed: refusing")


def lbl(r, model, fmt):
    return f"{TAG[model]}/{fmt}{r.E2S}"


def load(r, model, fmt, have, rows=None):
    src = {"sm124": R124, "sm115": R115, "sm118": R118, "sm113": R113}[model] if fmt == PUBLISHED else (rows or ROWS)
    return r.load([(src, lbl(r, model, fmt), None)], have)


def pct(x, h):
    return {"value": x[h]["pct_of_ref"], "ci": x[h]["pct_of_ref_ci"], "new": x[h]["new"], "ref": x[h]["ref"],
            "points": x[h]["points"], "points_ci": x[h]["points_ci"], "better": x[h]["users_better"], "worse": x[h]["users_worse"]}


def finite(rows, users, heads):
    import math
    key = {"curve": "ahead", "instant": "instant"}
    return all(math.isfinite(rows[u][key[h]][0]) for u in users for h in heads)


def decide(rows=ROWS, out_dir=HERE):
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    res = {"rule": f"pass = every head's 95% CI upper bound of 100 x (Tiny arm / Small - 1) <= +{CEILING:.2f}% on all "
                   f"{len(users)} rule users; verdict per arm: both models pass -> go", "users": len(users), "models": {},
           "arms": {}}
    missing = []
    for mname, m in MODELS.items():
        small = load(r, m["small"], PUBLISHED, have)
        tiny = load(r, m["tiny"], PUBLISHED, have)
        out = {"published_vs_small": None, "arms": {}}
        if not (have <= set(small) and have <= set(tiny)):
            raise SystemExit(f"[sm129-rule] reference rows incomplete for {mname}")
        x = r.compare(tiny, small, big, users)
        out["published_vs_small"] = {h: pct(x, h) for h in m["heads"]}
        fork_rows = load(r, m["tiny"], FORK, have, rows)
        small_fork = load(r, m["small"], FORK, have, rows)
        for arm, fmt in list(ARMS.items()) + [("fork", FORK)]:
            a = load(r, m["tiny"], fmt, have, rows)
            if not have <= set(a):
                missing.append(f"{mname} {arm}: {len(have & set(a))} of {len(users)}")
                continue
            vs_small = r.compare(a, small, big, users)
            vs_tiny = r.compare(a, tiny, big, users)
            row = {"label": lbl(r, m["tiny"], fmt), "finite": finite(a, users, m["heads"]),
                   "vs_small_published": {h: pct(vs_small, h) for h in m["heads"]},
                   "cost_vs_tiny_published": {h: pct(vs_tiny, h) for h in m["heads"]}}
            if arm != "fork" and have <= set(fork_rows):
                v = r.compare(a, fork_rows, big, users)
                row["cost_vs_tiny_fork_default"] = {h: pct(v, h) for h in m["heads"]}
            if have <= set(small_fork):
                v = r.compare(a, small_fork, big, users)
                row["vs_small_at_fork_w8_s4c8n"] = {h: pct(v, h) for h in m["heads"]}
            if arm != "fork":
                row["pass"] = row["finite"] and all(row["vs_small_published"][h]["ci"][1] <= CEILING for h in m["heads"])
            out["arms"][arm] = row
        if have <= set(small_fork):
            v = r.compare(fork_rows, small_fork, big, users) if have <= set(fork_rows) else None
            out["tiny_fork_vs_small_fork"] = {h: pct(v, h) for h in m["heads"]} if v else None
        res["models"][mname] = out
    if missing:
        print("[sm129-rule] decide pending: " + "; ".join(missing))
        return 2
    for arm in ARMS:
        ok = {mname: res["models"][mname]["arms"][arm]["pass"] for mname in MODELS}
        res["arms"][arm] = {"pass": ok, "verdict": "go" if all(ok.values()) else
                            (f"go for {next(k for k, v in ok.items() if v)} only" if any(ok.values()) else "no go")}
    bpath = HERE / "sm129-bytes.json"
    if bpath.exists():
        b = json.loads(bpath.read_text())["arms"]["Tiny (w64-c1)"]
        res["bytes_tiny"] = {f: {"card": b[f]["card"]["bytes"], "note": b[f]["note"]["bytes"], "card_plus_note": b[f]["card_plus_note"]}
                             for f in b}
    res["verdict"] = {arm: v["verdict"] for arm, v in res["arms"].items()}
    (out_dir / "sm129-decision.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    write_md(res, out_dir)
    print(f"[sm129-rule] verdict compact {res['verdict']['compact']}, card-only {res['verdict']['card-only']} "
          f"({json.dumps({a: v['pass'] for a, v in res['arms'].items()})})")
    return 0


def f(x):
    return f"{x['value']:+.3f}% [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"


def write_md(res, out_dir):
    nice = {"compact": "compact state (SM47 format)", "card-only": "card-only compact state (SM49 arm C')", "fork": "fork default (int4 cards)"}
    L = ["# Compact state for Tiny (SM129): decision", "",
         f"Rule (pre-registered): {res['rule']}. Show-time inputs, leak-free, {res['users']} users of the 732 set, "
         "paired user bootstrap 2,000, seed 0. % = 100 x (Tiny / Small - 1).", ""]
    for arm, v in res["arms"].items():
        L.append(f"- **{nice[arm]}: {v['verdict']}** ({', '.join(f'{k} {chr(10003) if ok else chr(10007)}' for k, ok in v['pass'].items())})")
    L.append("")
    for mname, m in res["models"].items():
        small = NAMES[MODELS[mname]["small"]]
        tiny = NAMES[MODELS[mname]["tiny"]]
        L += [f"## {mname}: {tiny} vs {small}", "",
              "| format | head | Tiny LogLoss | vs Small (published), % [CI] | cost vs Tiny published | cost vs Tiny fork default | vs Small at fork w8+s4c8n | pass |",
              "|---|---|---|---|---|---|---|---|"]
        for h, x in m["published_vs_small"].items():
            L.append(f"| published w8+s8 | {h} | {x['new']} | {f(x)} | 0 | | | |")
        for arm, row in m["arms"].items():
            for h in MODELS[mname]["heads"]:
                vs = row["vs_small_published"][h]
                c1 = f(row["cost_vs_tiny_published"][h])
                c2 = f(row["cost_vs_tiny_fork_default"][h]) if row.get("cost_vs_tiny_fork_default") else ""
                c3 = f(row["vs_small_at_fork_w8_s4c8n"][h]) if row.get("vs_small_at_fork_w8_s4c8n") else ""
                ok = "" if arm == "fork" else ("yes" if row["pass"] else "no")
                L.append(f"| {nice[arm]} | {h} | {vs['new']} | {f(vs)} | {c1} | {c2} | {c3} | {ok} |")
        L.append("")
    if res.get("bytes_tiny"):
        L += ["## Serialized bytes, Tiny (fork layout, sm129-bytes.json; the six-layer Tiny has the same card and note streams)", "",
              "| format | card | note | card + note |", "|---|---|---|---|"]
        for fmt, b in res["bytes_tiny"].items():
            L.append(f"| {fmt} | {b['card']:,} | {b['note']:,} | {b['card_plus_note']:,} |")
    (out_dir / "sm129-decision.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def selftest():
    """Known rows: the outside review's Tiny-vs-Small CIs reproduce through this code (published formats); synthetic
    arms = Tiny's published rows relabelled (cost 0 -> pass) and scaled by 1.009 (instant +0.9% more -> the two-head
    instant CI upper bound crosses 1.00 -> fail); an end-to-end decide on relabelled rows."""
    import io
    import tempfile
    from contextlib import redirect_stdout
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    ok = {}
    for (tiny, small), want in REVIEW.items():
        x = r.compare(load(r, tiny, PUBLISHED, have), load(r, small, PUBLISHED, have), big, users)
        ok[f"{tiny}_vs_{small}_review"] = x["users"] == 722 and all(
            (x[h]["pct_of_ref"], x[h]["pct_of_ref_ci"]) == (v[0], v[1]) for h, v in want.items())
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        rows = t / "rows.jsonl"

        def relabel(src, model, fmts, factor=None):
            with open(src, encoding="utf-8") as fi, open(rows, "a", encoding="utf-8") as g:
                for line in fi:
                    x = json.loads(line)
                    if x.get("label") != lbl(r, model, PUBLISHED):
                        continue
                    for fmt in fmts:
                        y = json.loads(line)
                        y["label"] = lbl(r, model, fmt)
                        if factor and fmt == CARD_ONLY and "benchmark" in y:
                            for h in y["benchmark"]:
                                if h == "instant":
                                    y["benchmark"][h]["LogLoss"] *= factor
                        g.write(json.dumps(y) + "\n")
        relabel(R124, "sm124", (COMPACT, CARD_ONLY, FORK), factor=1.009)
        relabel(R115, "sm115", (COMPACT, CARD_ONLY, FORK), factor=1.009)
        relabel(R118, "sm118", (FORK,))
        relabel(R113, "sm113", (FORK,))
        with redirect_stdout(io.StringIO()):
            code = decide(rows, t)
        d = json.loads((t / "sm129-decision.json").read_text())
        ok["end_to_end_compact_go_zero_cost"] = code == 0 and d["verdict"]["compact"] == "go" and all(
            d["models"][m]["arms"]["compact"]["cost_vs_tiny_published"][h]["value"] == 0 for m in MODELS for h in MODELS[m]["heads"])
        ok["end_to_end_card_only_scaled_fails"] = d["verdict"]["card-only"] == "no go" and \
            d["models"]["two-head"]["arms"]["card-only"]["vs_small_published"]["instant"]["ci"][1] > CEILING
        ok["markdown"] = (t / "sm129-decision.md").exists()
    good = all(ok.values())
    print(f"[sm129-rule] selftest {json.dumps(ok)} -> {'OK' if good else 'FAILED'}")
    return 0 if good else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "decide"
    raise SystemExit({"selftest": selftest, "decide": decide}[cmd]())
