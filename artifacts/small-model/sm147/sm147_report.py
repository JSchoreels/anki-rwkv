"""Low-rank structure of the card and note states (SM147): read-out (design.md "Method") -> sm147-result.{json,md}.
Aggregates only (no card ids, no review rows)."""
import json
import random
import sys
from pathlib import Path

S = Path(__file__).resolve().parent
WT = S.parents[2]
sys.path.insert(0, str(WT))
from scripts.state_lowrank import BUCKETS, LOG_BINS, LOG_LO, LOG_STEP, RANKS  # noqa: E402

MODELS = [  # tag, plain name, size, heads read
    ("sm91s2-big-showtime", "Big two-head (SM91s2)", "big", ("ahead", "instant")),
    ("sm99-big-inst-e2", "Big Instant-Only (SM99)", "big", ("instant",)),
    ("sm118-e26-dec2ep32k-t91s2", "Small two-head (SM118)", "small", ("ahead", "instant")),
    ("sm113-inst-e35st-st-dec2ep32k-t99", "Small Instant-Only (SM113)", "small", ("instant",)),
    ("sm130-th-dec2ep32k-t91s2", "six-layer Tiny two-head (SM130)", "tiny", ("ahead", "instant")),
    ("sm130-io-dec2ep32k-t99", "six-layer Tiny Instant-Only (SM130)", "tiny", ("instant",)),
]
LABELS = [("F32", "w8+lrsv"), ("REF", "w8+s8hx8"), ("R1", "w8+lr1"), ("R2", "w8+lr2"), ("R4", "w8+lr4")]
HEADNAME = {"ahead": "curve", "instant": "instant"}
# prior_buckets edges 0,1,3,5,9,16,32,128,300,1000 -> 0, 1-2, 3-4, 5-8, 9-15, 16-31, 32-127, 128+ (last three merged)
PB_NAMES = ["0"] + list(BUCKETS)
PB_MAP = [0, 1, 2, 3, 4, 5, 6, 7, 7, 7]
# int8 entity bytes of the fork compact layout (SM145 design "Bytes"), heads / K / card and note WKV layers
GEOM = {"big": (4, 32, 3, 2, 13891, 9267), "small": (4, 16, 2, 1, 2545, 1315), "tiny": (4, 16, 1, 1, 1315, 1315)}


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def boot(pairs, draws=2000, seed=0):
    """pairs [(label sum, base sum)] per user -> 95% CI of pct(sum a, sum b) (paired user bootstrap)."""
    rng = random.Random(seed)
    n = len(pairs)
    vals = []
    for _ in range(draws):
        pick = [pairs[rng.randrange(n)] for _ in range(n)]
        vals.append(pct(sum(p[0] for p in pick), sum(p[1] for p in pick)))
    vals.sort()
    return vals[int(0.025 * draws)], vals[int(0.975 * draws) - 1]


def quantile_share(hist, q):
    """Kept share at quantile q of the share (from the residual histogram: share P10 = 1 - residual P90); value =
    1 - upper edge of the residual bin, i.e. a lower bound to one 0.1-decade bin."""
    tot = sum(hist.values())
    if not tot:
        return None
    target = (1 - q) * tot
    cum = 0
    for b in range(LOG_BINS + 1):
        cum += hist.get(b, 0)
        if cum >= target:
            upper = 10 ** LOG_LO if b == 0 else 10 ** (LOG_LO + LOG_STEP * b)
            return max(0.0, 1.0 - upper)
    return 0.0


def spectrum(rows):
    """{tag: {stream: {"layers": {layer: {head: stats}}, "buckets": [...], "all": stats}}}; stats = writes, zero,
    mean share per rank, P10 / P50 share per rank."""
    acc = {}
    for x in rows:
        for cell in x.get("lowrank_spectrum", []):
            st = acc.setdefault(x["tag"], {}).setdefault(cell["stream"], {})
            H = len(cell["count"])
            for h in range(H):
                for b in range(len(BUCKETS)):
                    for key in (("lh", cell["layer"], h), ("b", b), ("all",)):
                        a = st.setdefault(key, {"n": 0, "zero": 0, "sum": [0.0] * len(RANKS),
                                                "hist": [dict() for _ in RANKS]})
                        a["n"] += cell["count"][h][b]
                        a["zero"] += cell["zero"][h][b]
                        for r in range(len(RANKS)):
                            a["sum"][r] += cell["share_sum"][h][b][r]
                            for k, v in cell["hist"].get(f"{h},{b},{r}", {}).items():
                                a["hist"][r][int(k)] = a["hist"][r].get(int(k), 0) + v

    def stats(a):
        valid = a["n"] - a["zero"]
        return {"writes": a["n"], "zero": a["zero"],
                "mean": [s / valid if valid else None for s in a["sum"]],
                "p10": [quantile_share(h, 0.10) for h in a["hist"]],
                "p50": [quantile_share(h, 0.50) for h in a["hist"]]}
    out = {}
    for tag, streams in acc.items():
        for stream, cells in streams.items():
            o = out.setdefault(tag, {}).setdefault(stream, {"layers": {}, "buckets": {}, "all": None})
            for key, a in cells.items():
                if key[0] == "lh":
                    o["layers"].setdefault(str(key[1]), {})[str(key[2])] = stats(a)
                elif key[0] == "b":
                    o["buckets"][BUCKETS[key[1]]] = stats(a)
                else:
                    o["all"] = stats(a)
    return out


def logloss(rows):
    by = {}
    for x in rows:
        if "benchmark" in x:
            by[(x["tag"], x["variant"], x["user_id"])] = x
    users = sorted({k[2] for k in by})
    out = {}
    for tag, _, _, heads in MODELS:
        m = out.setdefault(tag, {"labels": {}, "vs": {}, "buckets": {}})
        for name, var in LABELS:
            rs = [by[(tag, var, u)] for u in users]
            m["labels"][name] = {HEADNAME[h]: sum(r["benchmark"][h]["LogLoss"] * r["benchmark"][h]["size"] for r in rs)
                                 / sum(r["benchmark"][h]["size"] for r in rs) for h in ("ahead", "instant")}
            m["labels"][name]["reviews"] = sum(r["benchmark"]["ahead"]["size"] for r in rs)
            # all scope by bucket of the card's earlier answers
            bk = {}
            for h in ("ahead", "instant"):
                n = [0] * len(PB_NAMES)
                ll = [0.0] * len(PB_NAMES)
                per_user = {}
                for r in rs:
                    a = r["prior_buckets"]["all"][h]
                    pu_n, pu_ll = [0] * len(PB_NAMES), [0.0] * len(PB_NAMES)
                    for i, (c, s) in enumerate(zip(a["n"], a["ll"])):
                        j = PB_MAP[i]
                        n[j] += c
                        ll[j] += s
                        pu_n[j] += c
                        pu_ll[j] += s
                    per_user[r["user_id"]] = pu_ll
                bk[HEADNAME[h]] = {"n": n, "ll": ll, "per_user_ll": per_user}
            m["buckets"][name] = bk
        for name, _ in LABELS:
            if name in ("REF", "F32"):
                continue
            for base in ("REF", "F32"):
                for h in ("ahead", "instant"):
                    pairs = []
                    for u in users:
                        a, b = by[(tag, dict(LABELS)[name], u)], by[(tag, dict(LABELS)[base], u)]
                        pairs.append((a["benchmark"][h]["LogLoss"] * a["benchmark"][h]["size"],
                                      b["benchmark"][h]["LogLoss"] * b["benchmark"][h]["size"]))
                    p = pct(sum(q[0] for q in pairs), sum(q[1] for q in pairs))
                    lo, hi = boot(pairs)
                    m["vs"][f"{name}/{base}/{HEADNAME[h]}"] = {"pct": p, "ci": [lo, hi],
                                                               "users_worse": sum(q[0] > q[1] for q in pairs)}
        # bucket % vs REF and F32
        for name, _ in LABELS:
            for base in ("REF", "F32"):
                if name == base:
                    continue
                for h in ("curve", "instant"):
                    a, b = m["buckets"][name][h], m["buckets"][base][h]
                    m["vs"][f"{name}/{base}/{h}/buckets"] = [pct(x, y) if y else None for x, y in zip(a["ll"], b["ll"])]
        for name in m["buckets"]:
            for h in m["buckets"][name]:
                m["buckets"][name][h].pop("per_user_ll")
    return out, users


def fmt(v, d=2, sign=True):
    if v is None:
        return "-"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def share(v):
    return "-" if v is None else f"{100 * v:.2f}%"


def main():
    rows = load(S / "cpu" / "sm147-rows.jsonl")
    spec = spectrum(rows)
    ll, users = logloss(rows)
    res = {"users": users, "spectrum": spec, "logloss": ll}
    (S / "sm147-result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    md = [f"# Low-rank structure of the card and note states (SM147): generated tables ({len(users)} users)", ""]
    md += ["## Energy kept by rank r (stored float32 WKV matrices, every write, all layers and heads)", "",
           "Mean share of the Frobenius energy kept by the best rank-r approximation; P10 = 10% of the written "
           "matrices keep less (lower bound to one 0.1-decade bin of the residual).", "",
           "| model | stream | K | writes | r=1 mean | r=2 mean | r=4 mean | r=8 mean | r=1 P10 | r=2 P10 | r=4 P10 |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, name, size, _ in MODELS:
        for stream in ("card", "note"):
            a = spec[tag][stream]["all"]
            md.append(f"| {name} | {stream} | {GEOM[size][1]} | {a['writes']:,} | "
                      + " | ".join(share(v) for v in a["mean"]) + " | "
                      + " | ".join(share(v) for v in a["p10"][:3]) + " |")
    md += ["", "## Energy kept by rank r, by the entity's answer count (mean share; rank 1 / 2 / 4)", "",
           "| model | stream | " + " | ".join(BUCKETS) + " |", "|---|---|" + "---:|" * len(BUCKETS)]
    for tag, name, _, _ in MODELS:
        for stream in ("card", "note"):
            b = spec[tag][stream]["buckets"]
            cells = []
            for k in BUCKETS:
                s = b.get(k)
                cells.append("-" if not s or not s["writes"] else
                             " / ".join(f"{100 * v:.1f}" for v in s["mean"][:3]))
            md.append(f"| {name} | {stream} | " + " | ".join(cells) + " |")
    md += ["", "Writes per bucket (card stream): " + "; ".join(
        f"{name}: " + ", ".join(f"{spec[tag]['card']['buckets'][k]['writes']:,}" for k in BUCKETS)
        for tag, name, _, _ in MODELS[::2]), ""]
    md += ["## Energy kept by rank r per layer and head (mean share, rank 1 / 2 / 4)", "",
           "| model | stream | layer | head 0 | head 1 | head 2 | head 3 |", "|---|---|---:|---:|---:|---:|---:|"]
    for tag, name, _, _ in MODELS:
        for stream in ("card", "note"):
            for layer, heads in sorted(spec[tag][stream]["layers"].items()):
                md.append(f"| {name} | {stream} | {layer} | " + " | ".join(
                    " / ".join(f"{100 * v:.1f}" for v in heads[h]["mean"][:3]) for h in sorted(heads)) + " |")
    md += ["", "## Benchmark LogLoss of the rank-r truncation (review-weighted over the users)", "",
           "% = 100 x (sum LogLoss label / sum LogLoss base - 1); CI = paired user bootstrap 95% (8 users: rough width "
           "only); worse = users where the label is worse. Heads read: curve and instant for two-head models, instant "
           "for Instant-Only models.", "",
           "| model | head | F32 | REF | R1 | R2 | R4 | R1 vs REF | R2 vs REF | R4 vs REF | R1 vs F32 | R2 vs F32 | R4 vs F32 |",
           "|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|"]
    for tag, name, _, heads in MODELS:
        m = ll[tag]
        for h in heads:
            hn = HEADNAME[h]
            vals = " | ".join(f"{m['labels'][lab][hn]:.5f}" for lab, _ in LABELS)
            vs = " | ".join(f"{fmt(m['vs'][f'{r}/REF/{hn}']['pct'], 3)}% [{fmt(m['vs'][f'{r}/REF/{hn}']['ci'][0], 3)}, "
                            f"{fmt(m['vs'][f'{r}/REF/{hn}']['ci'][1], 3)}] ({m['vs'][f'{r}/REF/{hn}']['users_worse']}/8)"
                            for r in ("R1", "R2", "R4"))
            vf = " | ".join(f"{fmt(m['vs'][f'{r}/F32/{hn}']['pct'], 3)}%" for r in ("R1", "R2", "R4"))
            md.append(f"| {name} | {hn} | {vals} | {vs} | {vf} |")
    md += ["", "Also: F32 vs REF (the int8 format's own cost on these users): " + "; ".join(
        f"{name} {HEADNAME[h]} {fmt(pct(ll[tag]['labels']['F32'][HEADNAME[h]], ll[tag]['labels']['REF'][HEADNAME[h]]), 3)}%"
        for tag, name, _, heads in MODELS for h in heads), ""]
    md += ["## All-scope LogLoss by the card's earlier answers, % vs REF (read heads)", "",
           "| model | head | label | " + " | ".join(PB_NAMES) + " |", "|---|---|---|" + "---:|" * len(PB_NAMES)]
    for tag, name, _, heads in MODELS:
        for h in heads:
            hn = HEADNAME[h]
            for lab in ("F32", "R1", "R2", "R4"):
                v = ll[tag]["vs"][f"{lab}/REF/{hn}/buckets"]
                md.append(f"| {name} | {hn} | {lab} | " + " | ".join(fmt(x, 2) for x in v) + " |")
    md += ["", "Predicted reviews per bucket (all scope, same for every label): " + "; ".join(
        f"{name}: " + ", ".join(f"{c:,}" for c in ll[tag]["buckets"]["REF"]["instant"]["n"])
        for tag, name, _, _ in MODELS[::2]), ""]
    md += ["## Bytes for scale (not a format proposal)", "",
           "Entity bytes if the WKV matrices were stored as rank-r int8 factors with one float16 scale per factor "
           "(2 r K + 4 B per head instead of K x K + 2 K at int8 rows), everything else as the int8 compact layout.", "",
           "| size | card int8 | card r1 / r2 / r4 | note int8 | note r1 / r2 / r4 |", "|---|---:|---|---:|---|"]
    bytes_out = {}
    for size, (H, K, cl, nl, cb, nb) in GEOM.items():
        full = K * K + 2 * K
        c = [cb - cl * H * (full - (2 * r * K + 4)) for r in (1, 2, 4)]
        n = [nb - nl * H * (full - (2 * r * K + 4)) for r in (1, 2, 4)]
        bytes_out[size] = {"card": [cb] + c, "note": [nb] + n}
        md.append(f"| {size} | {cb:,} | " + " / ".join(f"{v:,} ({pct(v, cb):+.0f}%)" for v in c)
                  + f" | {nb:,} | " + " / ".join(f"{v:,} ({pct(v, nb):+.0f}%)" for v in n) + " |")
    res["bytes"] = bytes_out
    (S / "sm147-result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (S / "sm147-result.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
