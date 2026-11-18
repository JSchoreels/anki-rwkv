"""SM47: float16 WKV row scales (h) and int8 token-shift vectors (x8), alone and combined, on top of +s8 and +s4c8n,
for SM36 (the best model) on the 732 pool and its 464 subset. Per head (curve = ahead, instant): review-weighted
benchmark LogLoss; delta vs SM36 w8+s8 and vs the variant's base (w8+s8 / w8+s4c8n of SM45), paired user bootstrap
(2,000 resamples, seed 0, 2.5/97.5 percentiles); % vs the fork (2.76M, w8+s8) on the same users; bits per card / note
and the card+note state of the own collection (13,973 cards, 13,198 notes).
usage: sm47_summary.py [rows.jsonl]   (default heldout-sm47-pools.jsonl next to this file)"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from scripts.state_vq_report import int_entity_bits  # noqa: E402

L = "s-w64kd-gate-sc-lanes-dec12-1ep32k-muon"
S42 = Path("C:/Users/jonat/workspace/wt/sm42/artifacts/small-model")
S45 = Path("C:/Users/jonat/workspace/wt/sm45/artifacts/small-model/sm45")
SMW = Path("C:/Users/jonat/workspace/wt/small-model/artifacts/small-model")
CARDS, NOTES = 13_973, 13_198
# variant -> (wkv bits, scale bits, shift bits; 0 = float32 shift vectors), base
VARIANTS = {"s8h": (8, 16, 0), "s8x8": (8, 32, 8), "s8hx8": (8, 16, 8),
            "s4c8nh": ((4, 8), 16, 0), "s4c8nx8": ((4, 8), 32, 8), "s4c8nhx8": ((4, 8), 16, 8)}
REFS = {"s8": (8, 32, 0), "s4c8n": ((4, 8), 32, 0)}


def load(files, label):
    out = {}
    for f in files:
        if Path(f).exists():
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                if r.get("label") == label and "benchmark" in r:
                    out[r["user_id"]] = r
    return out


def agg(r, us, k):
    n = sum(r[u]["benchmark"][k]["size"] for u in us)
    return sum(r[u]["benchmark"][k]["LogLoss"] * r[u]["benchmark"][k]["size"] for u in us) / n


def boot(a, b, us, k):
    rng, bs = random.Random(0), []
    for _ in range(2000):
        s = [rng.choice(us) for _ in us]
        bs.append(agg(a, s, k) - agg(b, s, k))
    bs.sort()
    return bs[49], bs[1949]


def mb(card, note):
    return (CARDS * card + NOTES * note) / 8 / 1e6


src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "heldout-sm47-pools.jsonl"
rows = {v: load([src], f"{L}/w8+{v}") for v in VARIANTS}
s36 = load([S42 / "heldout-sm36-pools.jsonl"], f"{L}/w8+s8")
bases = {"s8": s36, "s4c8n": load([S45 / "heldout-sm45-pools.jsonl"], f"{L}/w8+s4c8n")}
fork = load([SMW / "heldout.jsonl", SMW / "heldout-pools.jsonl"], "fork/w8+s8")
p464 = set(load([S42 / "heldout-sm36.jsonl"], f"{L}/w8+s8"))

lines = ["# SM47: float16 row scales (h) and int8 shift vectors (x8), SM36 (best model), w8 weights", "",
         "State bits per entity: per WKV row 16 codes + one scale (float32, or float16 with h); per token-shift vector 64 "
         "float32, or with x8 64 int8 codes + one scale (float16 with h, else float32). Card = 2 WKV layers x 4 heads x "
         f"16 rows + 3 shift vectors; note = 1 x 4 x 16 rows + 2 shift vectors. Collection = {CARDS:,} cards + "
         f"{NOTES:,} notes (MB = 10^6 bytes).", "",
         "| Variant | Card bits | Note bits | Collection MB | vs w8+s8 |", "|---|---:|---:|---:|---:|"]
s8mb = mb(*int_entity_bits(8))
for v, spec in {**REFS, **VARIANTS}.items():
    c, n = int_entity_bits(*spec)
    lines.append(f"| w8+{v} | {c:,} | {n:,} | {mb(c, n):.2f} | {100 * (mb(c, n) / s8mb - 1):+.1f}% |")
lines += ["", "Review-weighted benchmark LogLoss; d s8 = variant - SM36 w8+s8, d base = variant - its base (w8+s8 or SM45 "
          "w8+s4c8n), [95% paired user bootstrap]; % vs the big model (fork w8+s8) on the same users; curve = ahead.", ""]
result = {}
for pool, keep in (("732", None), ("464", p464)):
    for v, r in rows.items():
        bname = "s4c8n" if v.startswith("s4c8n") else "s8"
        base = bases[bname]
        us = sorted(u for u in set(r) & set(s36) & set(base) & set(fork) if keep is None or u in keep)
        if not us:
            continue
        parts, res = [], {"users": len(us)}
        for k, name in (("ahead", "curve"), ("instant", "instant")):
            a, s, b, f = agg(r, us, k), agg(s36, us, k), agg(base, us, k), agg(fork, us, k)
            lo, hi = boot(r, s36, us, k)
            text = (f"{name} {a:.5f} ({100 * (a / f - 1):+.2f}% vs big; s8 {s:.5f} {100 * (s / f - 1):+.2f}%) "
                    f"d s8 {a - s:+.5f} [{lo:+.5f}, {hi:+.5f}]")
            res[name] = {"LogLoss": a, "pct_vs_fork": 100 * (a / f - 1), "s8": s, "d_s8": a - s, "d_s8_ci": [lo, hi]}
            if bname != "s8":
                blo, bhi = boot(r, base, us, k)
                text += f"; d base {a - b:+.5f} [{blo:+.5f}, {bhi:+.5f}]"
                res[name].update({"base": b, "d_base": a - b, "d_base_ci": [blo, bhi]})
            parts.append(text)
        c, n = int_entity_bits(*VARIANTS[v])
        res.update({"card_bits": c, "note_bits": n, "collection_mb": mb(c, n)})
        result[f"{pool}/w8+{v}"] = res
        lines.append(f"- {pool} pool, w8+{v} (base w8+{bname}; {c:,} / {n:,} bits, {mb(c, n):.2f} MB), {len(us)} users: "
                     + "; ".join(parts))
if src == HERE / "heldout-sm47-pools.jsonl":
    (HERE / "sm47-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "sm47-summary.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
print("\n".join(lines))
