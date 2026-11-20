#!/usr/bin/env python
"""Six-layer Tiny with the card-only state (SM134): pre-registered rule (fixed 2026-11-20 before any GPU ticket; CPU, seconds).

Question: are the six-layer Tiny models of six-layer Tiny (SM130) still within +1.00% of Small per head with the
card-only state format w8+s4hgx4c8hx8n (the second integer-state test (SM49)'s arm C'; 619 B per Tiny card, notes 1,315 B)?
Models (no training; show-time inputs `e2s-query`, leak-free `--split-existing --deleted-cards neighbour`, 723 users of
the 732 set, rule pool = the 722 users of showtime_report.py pools(), w8 weights):
  th  the two-head six-layer Tiny     tag sm130-th-dec2ep32k-t91s2 (curve + instant)  Small = the show-time two-head cool-down for Small (SM118)
  io  the Instant-Only six-layer Tiny tag sm130-io-dec2ep32k-t99 (instant)            Small = the longer Small Instant-Only model (SM113)
Rows: published w8+s8 and compact w8+s4c8nhx8 = six-layer Tiny (SM130)'s committed rows (inputs/sm130-rows.jsonl, a byte
copy, sha256 pinned); card-only = scored by this run (gpu/sm134-rows.jsonl). Small references: committed rows, sha256 pinned.
RULE (SM130's, with the card-only rows): go(chain) iff every trained head's 95% CI upper bound of
  100 x (six-layer card-only / Small published w8+s8 - 1) <= +1.00%, on all 722 rule users with finite rows
  (exit 2 = pending while fewer); verdict = go iff both chains go.
Statistic: pct_of_ref of showtime_report.py compare() (sha256 pinned), paired user bootstrap 2,000, seed 0, 95% CI.
Reported, no verdict: card-only vs six-layer compact, card-only vs six-layer published (also published and compact vs
Small as in SM130), bytes per card / note.

  python sm134_rule.py selftest | decide
"""
import hashlib
import importlib.util
import io
import json
import math
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
G = HERE / "gpu"
WT = HERE.parents[2]
SMG = WT / "artifacts/small-model"
W = Path("C:/Users/jonat/workspace/wt")
REPORT = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/showtime/showtime_report.py")
REPORT_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"
ROWS = G / "sm134-rows.jsonl"
R130 = HERE / "inputs/sm130-rows.jsonl"
R118 = SMG / "sm118/gpu/sm118-rows.jsonl"
R113 = W / "sm113/artifacts/small-model/sm113/gpu/sm113-rows.jsonl"
BYTES = SMG / "sm131/sm131-bytes.json"
PINS = {R130: "f4b38086f872762442d754752542e9d26606fc11fee10bd586c8e2e739f0a5b6",
        R118: "430ffe8a809c27079862acf0bfb57b12b279a8472aa5eff19dfb232db378befb",
        R113: "af104a0774b1aa7da7b26bc48ac667a9991d86efd8dda4ac8de7bb51a3fe281a"}
E2S = "@e2s-query"
PUBLISHED, COMPACT, CARD_ONLY = "w8+s8", "w8+s4c8nhx8", "w8+s4hgx4c8hx8n"
CHAINS = {"th": {"heads": ("curve", "instant"), "tag": "sm130-th-dec2ep32k-t91s2", "small": (R118, "sm118-e26-dec2ep32k-t91s2"),
                 "names": ("the two-head six-layer Tiny (SM130)", "the show-time two-head cool-down for Small (SM118)")},
          "io": {"heads": ("instant",), "tag": "sm130-io-dec2ep32k-t99", "small": (R113, "sm113-inst-e35st-st-dec2ep32k-t99"),
                 "names": ("the Instant-Only six-layer Tiny (SM130)", "the longer Small Instant-Only model (SM113)")}}
CEILING = 1.0
KEY = {"curve": "ahead", "instant": "instant"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report():
    if sha(REPORT) != REPORT_SHA:
        raise SystemExit("[sm134-rule] showtime_report.py changed: refusing")
    spec = importlib.util.spec_from_file_location("showtime_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_pins():
    for path, want in PINS.items():
        if sha(path) != want:
            raise SystemExit(f"[sm134-rule] reference file {path} changed: refusing")


def clean_copy(path, tmp, counts):
    """A copy of `path` with only parseable JSON object lines (a torn or NUL-filled last line is skipped and counted)."""
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


def pct(x, h):
    return {"value": x[h]["pct_of_ref"], "ci": x[h]["pct_of_ref_ci"], "new": x[h]["new"], "ref": x[h]["ref"]}


def finite_users(rows, users, heads):
    return {u for u in users if u in rows and all(math.isfinite(rows[u][KEY[h]][0]) and rows[u][KEY[h]][1] > 0 for h in heads)}


def f(x):
    return f"{x['value']:+.3f}% [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"


def decide(rows=ROWS, out_dir=HERE):
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    res = {"rule": f"go(chain) iff every trained head's 95% CI upper bound of 100 x (card-only six-layer / Small published "
                   f"w8+s8 - 1) <= +{CEILING:.2f}% on all {len(users)} rule users with finite rows; verdict go iff both chains go",
           "pool_users": len(users), "chains": {}, "unparseable_lines": {}}
    pending = []
    with tempfile.TemporaryDirectory() as tmp:
        p130 = clean_copy(R130, tmp, res["unparseable_lines"])
        pnew = clean_copy(rows, tmp, res["unparseable_lines"]) if Path(rows).exists() else None
        nonfinite = {}
        if pnew:
            for line in open(pnew, encoding="utf-8"):
                row = json.loads(line)
                if "nonfinite" in row and "benchmark" not in row:
                    nonfinite.setdefault(row.get("label"), set()).add(row.get("user_id"))
        res["nonfinite_rows"] = {k: sorted(v) for k, v in sorted(nonfinite.items())}
        for chain, c in CHAINS.items():
            heads, tag = c["heads"], c["tag"]
            small = r.load([(c["small"][0], f"{c['small'][1]}/{PUBLISHED}{E2S}", None)], have)
            pub = r.load([(p130, f"{tag}/{PUBLISHED}{E2S}", None)], have)
            comp = r.load([(p130, f"{tag}/{COMPACT}{E2S}", None)], have)
            card = r.load([(pnew, f"{tag}/{CARD_ONLY}{E2S}", None)], have) if pnew else {}
            if not (have <= finite_users(small, users, heads) and have <= finite_users(pub, users, heads)
                    and have <= finite_users(comp, users, heads)):
                raise SystemExit(f"[sm134-rule] reference rows incomplete for {chain}")
            ok_users = finite_users(card, users, heads)
            out = {"model": c["names"][0], "small": c["names"][1], "heads": list(heads), "users": len(ok_users),
                   "missing_users": sorted(have - ok_users)}
            if ok_users != have:
                pending.append(f"{chain}: card-only rows finite for {len(ok_users)} of {len(users)} users")
                res["chains"][chain] = out
                continue
            cs = r.compare(card, small, big, users)
            out["card_only_vs_small"] = {h: pct(cs, h) for h in heads}
            out["pass"] = all(out["card_only_vs_small"][h]["ci"][1] <= CEILING for h in heads)
            out["verdict"] = "go" if out["pass"] else "no go"
            cc, cp = r.compare(card, comp, big, users), r.compare(card, pub, big, users)
            out["card_only_vs_compact"] = {h: pct(cc, h) for h in heads}
            out["card_only_vs_published"] = {h: pct(cp, h) for h in heads}
            out["published_vs_small"] = {h: pct(r.compare(pub, small, big, users), h) for h in heads}
            out["compact_vs_small"] = {h: pct(r.compare(comp, small, big, users), h) for h in heads}
            res["chains"][chain] = out
    if pending:
        print("[sm134-rule] decide pending: " + "; ".join(pending))
        return 2
    res["verdict"] = "go" if all(v["pass"] for v in res["chains"].values()) else "no go (" + ", ".join(
        c for c, v in res["chains"].items() if not v["pass"]) + ")"
    if BYTES.exists():
        b = json.loads(BYTES.read_text())["arms"]["Tiny (w64-c1)"]
        res["bytes_tiny_per_card_note"] = {fm: [b[fm]["card"]["bytes"], b[fm]["note"]["bytes"]] for fm in b}
    (out_dir / "sm134-decision.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    write_md(res, out_dir)
    print(f"[sm134-rule] verdict {res['verdict']}; " + "; ".join(
        f"{k} " + ", ".join(f"{h} vs Small {f(v['card_only_vs_small'][h])}" for h in v["heads"]) for k, v in res["chains"].items()))
    return 0


def write_md(res, out_dir):
    L = ["# Six-layer Tiny with the card-only state (SM134): decision", "", f"Rule (pre-registered): {res['rule']}. "
         "Show-time inputs, leak-free, the 732 set, paired user bootstrap 2,000, seed 0. % = 100 x (new / reference - 1) "
         "of the review-weighted LogLoss.", "", f"**Verdict: {res['verdict']}**", "",
         "| model | head | card-only LogLoss | card-only vs Small published % [CI] | go | card-only vs six-layer compact | "
         "card-only vs six-layer published | compact vs Small (SM130) | published vs Small (SM130) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for ch, v in res["chains"].items():
        for h in v["heads"]:
            a = v["card_only_vs_small"][h]
            L.append(f"| {v['model']} vs {v['small']} | {h} | {a['new']} | {f(a)} | {'yes' if v['pass'] else 'no'} | "
                     f"{f(v['card_only_vs_compact'][h])} | {f(v['card_only_vs_published'][h])} | "
                     f"{f(v['compact_vs_small'][h])} | {f(v['published_vs_small'][h])} |")
    if res.get("bytes_tiny_per_card_note"):
        L += ["", "Tiny state bytes per card / note: " + "; ".join(
            f"{k} {a:,} / {b:,}" for k, (a, b) in res["bytes_tiny_per_card_note"].items())]
    L += ["", f"Unparseable row lines skipped: {sum(res['unparseable_lines'].values())}"]
    if res.get("nonfinite_rows"):
        L.append("Non-finite rows: " + json.dumps(res["nonfinite_rows"]))
    (out_dir / "sm134-decision.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def selftest():
    """(1) SM130's published / compact vs Small numbers are reproduced from the copied rows (th curve +0.467, instant +0.524;
    io instant +0.564; compact +0.483 / +0.545 / +0.603, upper bounds too); (2) bytes: Tiny card-only 619 / 1,315, compact 803;
    (3) card-only rows = compact rows relabelled -> end to end go, card-only vs compact exactly 0, vs Small = compact's;
    (4) card-only instant x 1.02 on the io chain -> no go; (5) one user dropped -> exit 2 pending; (6) junk lines counted."""
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    ok = {}
    exp = {("th", PUBLISHED): (0.467, 0.546, 0.524, 0.567), ("th", COMPACT): (0.483, 0.562, 0.545, 0.589),
           ("io", PUBLISHED): (None, None, 0.564, 0.607), ("io", COMPACT): (None, None, 0.603, 0.647)}
    for (chain, fmt), (c_v, c_hi, i_v, i_hi) in exp.items():
        c = CHAINS[chain]
        small = r.load([(c["small"][0], f"{c['small'][1]}/{PUBLISHED}{E2S}", None)], have)
        new = r.load([(R130, f"{c['tag']}/{fmt}{E2S}", None)], have)
        x = r.compare(new, small, big, users)
        good = abs(x["instant"]["pct_of_ref"] - i_v) < 0.0006 and abs(x["instant"]["pct_of_ref_ci"][1] - i_hi) < 0.0006
        if c_v is not None:
            good &= abs(x["curve"]["pct_of_ref"] - c_v) < 0.0006 and abs(x["curve"]["pct_of_ref_ci"][1] - c_hi) < 0.0006
        ok[f"reproduces_sm130_{chain}_{fmt}"] = bool(good)
    b = json.loads(BYTES.read_text())["arms"]["Tiny (w64-c1)"]
    ok["bytes_tiny"] = (b[CARD_ONLY]["card"]["bytes"], b[CARD_ONLY]["note"]["bytes"], b[COMPACT]["card"]["bytes"]) == (619, 1315, 803)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def build(name, scale_io=None, drop=None, junk=False):
            rows = tmp / f"{name}.jsonl"
            with open(rows, "w", encoding="utf-8") as g:
                for line in open(R130, encoding="utf-8"):
                    y = json.loads(line)
                    chain = "th" if y.get("label", "").startswith(CHAINS["th"]["tag"]) else "io"
                    if y.get("label") != f"{CHAINS[chain]['tag']}/{COMPACT}{E2S}":
                        continue
                    y["label"] = f"{CHAINS[chain]['tag']}/{CARD_ONLY}{E2S}"
                    if drop and chain == "th" and y["user_id"] in drop:
                        continue
                    if scale_io and chain == "io" and "benchmark" in y:
                        y["benchmark"]["instant"]["LogLoss"] *= scale_io
                    g.write(json.dumps(y) + "\n")
            if junk:
                with open(rows, "ab") as g:
                    g.write(b'{"user_id": 1, "label": "torn\n' + b"\x00" * 300)
            return rows

        def run(rows):
            with redirect_stdout(io.StringIO()) as s:
                code = decide(rows, tmp)
            d = json.loads((tmp / "sm134-decision.json").read_text()) if code == 0 else None
            if d is not None:
                (tmp / "sm134-decision.json").unlink()
            return code, d, s.getvalue()

        code, d, _ = run(build("same", junk=True))
        ok["end_to_end_relabelled_go"] = code == 0 and d["verdict"] == "go" and all(
            v["card_only_vs_compact"][h]["value"] == 0 and v["card_only_vs_small"][h]["value"] == v["compact_vs_small"][h]["value"]
            for v in d["chains"].values() for h in v["heads"])
        ok["unparseable_lines_counted"] = code == 0 and sum(d["unparseable_lines"].values()) == 2
        ok["markdown"] = (tmp / "sm134-decision.md").exists()
        code, d, _ = run(build("scaled", scale_io=1.02))
        ok["io_scaled_no_go"] = code == 0 and d["verdict"] == "no go (io)" and d["chains"]["th"]["pass"] and not d["chains"]["io"]["pass"]
        code, d, out = run(build("drop1", drop={users[0]}))
        ok["one_user_missing_pending"] = code == 2 and "pending" in out
    good = all(ok.values())
    print(f"[sm134-rule] selftest {json.dumps(ok)} -> {'OK' if good else 'FAILED'}")
    return 0 if good else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "decide"
    raise SystemExit({"selftest": selftest, "decide": decide}[cmd]())
