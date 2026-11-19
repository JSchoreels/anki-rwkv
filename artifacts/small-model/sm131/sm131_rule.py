#!/usr/bin/env python
"""Compact state for Small and Big (SM131): pre-registered rule (fixed 2026-11-19 before any GPU ticket; CPU, seconds).

Question: is the compact integer state of the compact int state test (SM47) (int4 cards + int8 notes, float16 row
scales, int8 token-shift vectors; variant w8+s4c8nhx8) free against the fork's default stored state (int4 cards + int8
notes, float32 row scales, float16 shift vectors; w8+s4c8n) for the bigger models, as it was for Tiny (compact state
for Tiny (SM129): +0.000% curve / +0.001% instant)?

Models (no training; show-time inputs `e2s-query`, leak-free `--split-existing --deleted-cards neighbour`, 723 users of
the 732 set, rule pool: the 722 users the big model's rows have, w8 weights):
  small two-head      the show-time two-head cool-down for Small (SM118)      sm118-e26-dec2ep32k-t91s2   (curve, instant)
  small instant-only  the longer Small Instant-Only model (SM113)             sm113-inst-e35st-st-dec2ep32k-t99 (instant)
  big two-head        the show-time big model (SM91s2)                         sm91s2-big-showtime         (curve, instant)
  big instant-only    the show-time instant-only big model (SM99)              sm99-big-inst-e2            (instant)
Formats: published w8+s8 (committed rows, report only); fork default w8+s4c8n (Small: rows of the compact state for Tiny
(SM129), same scorer tree; Big: scored by this run); compact w8+s4c8nhx8 (scored by this run); card-only
w8+s4hgx4c8hx8n (the second integer-state test (SM49)'s arm C', scored by this run, report only).

Statistic: per head, % = 100 x (compact / fork default - 1) of the review-weighted LogLoss (pct_of_ref of
artifacts/small-model/showtime/showtime_report.py compare(), sha256 pinned), paired user bootstrap 2,000, seed 0, 95% CI.
RULE:
  free(model)  = every head's 95% CI upper bound < +0.05%   (two-head: curve and instant; Instant-Only: instant)
  verdict(size) = "free" if both models of the size are free, else "not free" (with the model(s) that are not).
Users: the rule pool users that have finite rows in both formats; at least MIN_USERS = 718 of the 722 (all but at most
the four Big users 481, 2267, 3735, 4852 that ran out of GPU memory in the per-user recalibration for Big (SM128)),
otherwise `decide` exits 2 (pending). The coverage is reported per model.
Reported, no verdict: compact and fork default vs published w8+s8, card-only vs fork default and vs published, the
serialized bytes per card and per note and the saving (sm131-bytes.json).
Rows files are read through a filter that skips and counts unparseable lines (a torn or NUL-filled last line).

  python sm131_rule.py selftest   -> the rule's code on known and synthetic rows
  python sm131_rule.py decide     -> sm131-decision.{json,md}; exit 2 while rows are missing
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
G = HERE / "gpu"
WT = HERE.parents[2]
SMG = WT / "artifacts/small-model"
W = Path("C:/Users/jonat/workspace/wt")
REPORT = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/showtime/showtime_report.py")
REPORT_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"   # the same pin as SM96 ... SM129
ROWS = G / "sm131-rows.jsonl"
R129 = SMG / "sm129/gpu/sm129-rows.jsonl"          # Small fork default rows (w8+s4c8n), same scorer tree
D129 = SMG / "sm129/sm129-decision.json"
PUB = {"sm118": SMG / "sm118/gpu/sm118-rows.jsonl", "sm113": W / "sm113/artifacts/small-model/sm113/gpu/sm113-rows.jsonl",
       "sm91s2": W / "sm91s2/artifacts/small-model/sm91s2/gpu/sm91s2-inputs.jsonl",
       "sm99": W / "sm99/artifacts/small-model/sm99/gpu/sm99-rows.jsonl"}
PINS = {PUB["sm118"]: "430ffe8a809c27079862acf0bfb57b12b279a8472aa5eff19dfb232db378befb",
        PUB["sm113"]: "af104a0774b1aa7da7b26bc48ac667a9991d86efd8dda4ac8de7bb51a3fe281a",
        PUB["sm91s2"]: "e156b6fd907857b22f96d64035c06417b063c88d416a959bc91231ed85075c80",
        PUB["sm99"]: "df4c48e0a0c3e95b99a64629fb262ccb0805194fd0702c1e46d6adf7737dacea",
        R129: "c636b25050345c41c1fc3eb84308451558d6da4dff6ca6621d2305c4714e447a",
        D129: "2b21ae18da38e4df0987e402b7dd9491fe165d73268d0c50c7ac8a604262daba"}
TAG = {"sm118": "sm118-e26-dec2ep32k-t91s2", "sm113": "sm113-inst-e35st-st-dec2ep32k-t99",
       "sm91s2": "sm91s2-big-showtime", "sm99": "sm99-big-inst-e2"}
PUBLISHED, FORK, COMPACT, CARD_ONLY = "w8+s8", "w8+s4c8n", "w8+s4c8nhx8", "w8+s4hgx4c8hx8n"
MODELS = {"sm118": {"size": "Small", "heads": ("curve", "instant"), "fork_rows": "r129"},
          "sm113": {"size": "Small", "heads": ("instant",), "fork_rows": "r129"},
          "sm91s2": {"size": "Big", "heads": ("curve", "instant"), "fork_rows": "rows"},
          "sm99": {"size": "Big", "heads": ("instant",), "fork_rows": "rows"}}
SIZES = {"Small": ("sm118", "sm113"), "Big": ("sm91s2", "sm99")}
BYTES_ARCH = {"Small": "Small (w64)", "Big": "Big (fork)"}
CEILING = 0.05
MIN_USERS = 718
NAMES = {"sm118": "the show-time two-head cool-down for Small (SM118)", "sm113": "the longer Small Instant-Only model (SM113)",
         "sm91s2": "the show-time big model (SM91s2)", "sm99": "the show-time instant-only big model (SM99)"}
KEY = {"curve": "ahead", "instant": "instant"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report():
    got = sha(REPORT)
    if got != REPORT_SHA:
        raise SystemExit(f"[sm131-rule] showtime_report.py changed (sha256 {got[:12]}): refusing")
    spec = importlib.util.spec_from_file_location("showtime_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_pins():
    for path, want in PINS.items():
        if sha(path) != want:
            raise SystemExit(f"[sm131-rule] reference file {path} changed: refusing")


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


def lbl(r, model, fmt):
    return f"{TAG[model]}/{fmt}{r.E2S}"


def pct(x, h):
    return {"value": x[h]["pct_of_ref"], "ci": x[h]["pct_of_ref_ci"], "new": x[h]["new"], "ref": x[h]["ref"],
            "delta": x[h]["delta"], "delta_ci": x[h]["delta_ci"], "better": x[h]["users_better"], "worse": x[h]["users_worse"]}


def finite_users(rows, users, heads):
    return {u for u in users if u in rows and all(math.isfinite(rows[u][KEY[h]][0]) and rows[u][KEY[h]][1] > 0 for h in heads)}


def decide(rows=ROWS, out_dir=HERE, r129=R129):
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    res = {"rule": f"free = every head's 95% CI upper bound of 100 x (compact / fork default - 1) < +{CEILING:.2f}% "
                   f"on the rule users with finite rows in both formats (at least {MIN_USERS} of {len(users)}); "
                   "verdict per size: free if both models are free", "pool_users": len(users), "models": {}, "sizes": {},
           "unparseable_lines": {}}
    pending = []
    with tempfile.TemporaryDirectory() as tmp:
        src = {"rows": clean_copy(rows, tmp, res["unparseable_lines"]) if Path(rows).exists() else None,
               "r129": clean_copy(r129, tmp, res["unparseable_lines"])}
        # amendment 11-19 (design.md): rows marked 'nonfinite' (no benchmark; the int4 card state of a very long card
        # overflowing float32, the same on CPU and GPU) are not finite rows; listed here, the users count as missing
        nonfinite = {}
        for which, path in src.items():
            if path:
                for line in open(path, encoding="utf-8"):
                    row = json.loads(line)
                    if "nonfinite" in row and "benchmark" not in row:
                        nonfinite.setdefault(row.get("label"), set()).add(row.get("user_id"))
        res["nonfinite_rows"] = {k: sorted(v) for k, v in sorted(nonfinite.items())}

        def load(model, fmt, which):
            if fmt == PUBLISHED:
                return r.load([(PUB[model], lbl(r, model, fmt), None)], have)
            path = src[which]
            return r.load([(path, lbl(r, model, fmt), None)], have) if path else {}

        for model, m in MODELS.items():
            heads = m["heads"]
            pub = load(model, PUBLISHED, None)
            fork = load(model, FORK, m["fork_rows"])
            comp = load(model, COMPACT, "rows")
            card = load(model, CARD_ONLY, "rows")
            if not have <= finite_users(pub, users, heads):
                raise SystemExit(f"[sm131-rule] published rows incomplete for {model}")
            both = sorted(finite_users(fork, users, heads) & finite_users(comp, users, heads))
            out = {"name": NAMES[model], "heads": list(heads), "users": len(both),
                   "missing_users": sorted(have - set(both))}
            if len(both) < MIN_USERS:
                pending.append(f"{model}: {len(both)} of {len(users)} users with fork default and compact rows")
                res["models"][model] = out
                continue
            x = r.compare(comp, fork, big, both)
            out["compact_vs_fork_default"] = {h: pct(x, h) for h in heads}
            out["free"] = all(out["compact_vs_fork_default"][h]["ci"][1] < CEILING for h in heads)
            out["compact_vs_published"] = {h: pct(r.compare(comp, pub, big, both), h) for h in heads}
            out["fork_default_vs_published"] = {h: pct(r.compare(fork, pub, big, both), h) for h in heads}
            cboth = sorted(set(both) & finite_users(card, users, heads))
            if len(cboth) >= MIN_USERS:
                y = r.compare(card, fork, big, cboth)
                out["card_only"] = {"users": len(cboth), "vs_fork_default": {h: pct(y, h) for h in heads},
                                    "vs_published": {h: pct(r.compare(card, pub, big, cboth), h) for h in heads},
                                    "free_report_only": all(y[h]["pct_of_ref_ci"][1] < CEILING for h in heads)}
            else:
                out["card_only"] = {"users": len(cboth), "note": "incomplete, not reported"}
            res["models"][model] = out
    if pending:
        print("[sm131-rule] decide pending: " + "; ".join(pending))
        return 2
    for size, ms in SIZES.items():
        ok = {mm: res["models"][mm]["free"] for mm in ms}
        res["sizes"][size] = {"free": ok, "verdict": "free" if all(ok.values()) else
                              "not free (" + ", ".join(NAMES[mm] for mm, v in ok.items() if not v) + ")"}
    bpath = HERE / "sm131-bytes.json"
    if bpath.exists():
        b = json.loads(bpath.read_text())["arms"]
        res["bytes"] = {size: {f: {"card": b[a][f]["card"]["bytes"], "note": b[a][f]["note"]["bytes"],
                                   "card_plus_note": b[a][f]["card_plus_note"]} for f in b[a]}
                        for size, a in BYTES_ARCH.items()}
    res["verdict"] = {size: v["verdict"] for size, v in res["sizes"].items()}
    (out_dir / "sm131-decision.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    write_md(res, out_dir)
    bad = sum(res["unparseable_lines"].values())
    print(f"[sm131-rule] verdict Small {res['verdict']['Small']}, Big {res['verdict']['Big']} "
          f"(users {', '.join(f'{k} {v['users']}' for k, v in res['models'].items())}; unparseable lines {bad})")
    return 0


def f(x):
    return f"{x['value']:+.3f}% [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"


def saving(new, old):
    return f"{100 * (new / old - 1):+.1f}%"


def write_md(res, out_dir):
    L = ["# Compact state for Small and Big (SM131): decision", "",
         f"Rule (pre-registered): {res['rule']}. Show-time inputs, leak-free, the 732 set, paired user bootstrap 2,000, "
         "seed 0. % = 100 x (format / reference - 1) of the review-weighted LogLoss.", ""]
    for size, v in res["sizes"].items():
        L.append(f"- **{size}: compact state {v['verdict']}** (" + ", ".join(
            f"{NAMES[mm]} {'free' if ok else 'not free'}" for mm, ok in v["free"].items()) + ")")
    bad = {k: v for k, v in res["unparseable_lines"].items() if v}
    L += ["", f"Unparseable row lines skipped: {sum(bad.values())}" + (f" ({bad})" if bad else ""), ""]
    L += ["| model | head | users | fork default LogLoss | compact LogLoss | compact vs fork default, % [CI] | free | "
          "compact vs published | fork default vs published | card-only vs fork default (report only) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for mm, m in res["models"].items():
        for h in m["heads"]:
            c = m["compact_vs_fork_default"][h]
            co = m["card_only"].get("vs_fork_default", {}).get(h)
            L.append(f"| {m['name']} | {h} | {m['users']} | {c['ref']} | {c['new']} | {f(c)} | "
                     f"{'yes' if m['free'] else 'no'} | {f(m['compact_vs_published'][h])} | "
                     f"{f(m['fork_default_vs_published'][h])} | {f(co) if co else 'n/a'} |")
    if res.get("nonfinite_rows"):
        L += ["", "Non-finite rows (amendment 11-19; users count as missing): " + "; ".join(
            f"{k} users {v}" for k, v in res["nonfinite_rows"].items())]
    for mm, m in res["models"].items():
        if m["missing_users"]:
            L.append(f"\n{m['name']}: rule on {m['users']} users; missing {m['missing_users']}")
    if res.get("bytes"):
        L += ["", "## Serialized bytes (fork layout, sm131-bytes.json)", "",
              "| size | format | card | note | card + note | card vs fork default | card + note vs fork default |",
              "|---|---|---|---|---|---|---|"]
        for size, fm in res["bytes"].items():
            base = fm[FORK]
            for fmt, b in fm.items():
                L.append(f"| {size} | {fmt} | {b['card']:,} | {b['note']:,} | {b['card_plus_note']:,} | "
                         f"{saving(b['card'], base['card'])} | {saving(b['card_plus_note'], base['card_plus_note'])} |")
    (out_dir / "sm131-decision.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def selftest():
    """(1) the statistic path reproduces the compact state for Tiny (SM129)'s compact-vs-fork-default numbers for the
    show-time two-head cool-down for Tiny (SM124); (2) end-to-end on synthetic rows: relabelled fork-default rows as
    compact -> 0 cost, free, free; (3) the show-time instant-only big model (SM99)'s compact instant x 1.001 -> Big not
    free, Small free; (4) 3 users dropped from one Big model -> rule on 719 users, reported; 10 dropped -> exit 2;
    (5) a NUL-filled line and a torn line in the rows file are skipped and counted."""
    import io
    from contextlib import redirect_stdout
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    ok = {}
    d129 = json.loads(D129.read_text())["models"]["two-head"]["arms"]["compact"]["cost_vs_tiny_fork_default"]
    t = {"sm124": "sm124-c1-e29st-dec2ep32k-t91s2"}
    a = r.load([(R129, f"{t['sm124']}/{COMPACT}{r.E2S}", None)], have)
    b = r.load([(R129, f"{t['sm124']}/{FORK}{r.E2S}", None)], have)
    x = r.compare(a, b, big, users)
    ok["reproduces_sm129_sm124_compact_vs_fork"] = all(
        (x[h]["pct_of_ref"], x[h]["pct_of_ref_ci"]) == (d129[h]["value"], d129[h]["ci"]) for h in ("curve", "instant"))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def build(name, scale_sm99=None, drop=None, junk=False):
            rows = tmp / f"{name}.jsonl"
            with open(rows, "w", encoding="utf-8") as g:
                for model, m in MODELS.items():
                    src, fmt_src = (R129, FORK) if m["size"] == "Small" else (PUB[model], PUBLISHED)
                    outs = (COMPACT, CARD_ONLY) if m["size"] == "Small" else (FORK, COMPACT, CARD_ONLY)
                    with open(src, encoding="utf-8") as fi:
                        for line in fi:
                            y0 = json.loads(line)
                            if y0.get("label") != lbl(r, model, fmt_src):
                                continue
                            for fmt in outs:
                                y = json.loads(line)
                                y["label"] = lbl(r, model, fmt)
                                if drop and model == "sm91s2" and fmt == COMPACT and y["user_id"] in drop:
                                    continue
                                if scale_sm99 and model == "sm99" and fmt == COMPACT and "benchmark" in y:
                                    y["benchmark"]["instant"]["LogLoss"] *= scale_sm99
                                g.write(json.dumps(y) + "\n")
            if junk:
                with open(rows, "ab") as g:
                    g.write(b'{"user_id": 1, "label": "torn\n' + b"\x00" * 300)
            return rows

        def run(rows):
            with redirect_stdout(io.StringIO()) as s:
                code = decide(rows, tmp)
            d = json.loads((tmp / "sm131-decision.json").read_text()) if code == 0 else None
            if d is not None:
                (tmp / "sm131-decision.json").unlink()
            return code, d, s.getvalue()

        code, d, _ = run(build("zero", junk=True))
        ok["end_to_end_zero_cost_free"] = code == 0 and d["verdict"] == {"Small": "free", "Big": "free"} and all(
            d["models"][mm]["compact_vs_fork_default"][h]["value"] == 0 for mm in MODELS for h in MODELS[mm]["heads"])
        ok["unparseable_lines_counted"] = code == 0 and sum(d["unparseable_lines"].values()) == 2
        code, d, _ = run(build("scaled", scale_sm99=1.001))
        ok["sm99_scaled_not_free"] = code == 0 and d["verdict"]["Small"] == "free" and d["verdict"]["Big"].startswith("not free") \
            and not d["models"]["sm99"]["free"] and d["models"]["sm91s2"]["free"]
        code, d, _ = run(build("drop3", drop=set(users[:3])))
        ok["three_users_missing_reported"] = code == 0 and d["models"]["sm91s2"]["users"] == len(users) - 3 \
            and d["models"]["sm91s2"]["missing_users"] == sorted(users[:3])
        code, d, out = run(build("drop10", drop=set(users[:10])))
        ok["ten_users_missing_pending"] = code == 2 and "pending" in out
        ok["markdown"] = (tmp / "sm131-decision.md").exists()
    good = all(ok.values())
    print(f"[sm131-rule] selftest {json.dumps(ok)} -> {'OK' if good else 'FAILED'}")
    return 0 if good else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "decide"
    raise SystemExit({"selftest": selftest, "decide": decide}[cmd]())
