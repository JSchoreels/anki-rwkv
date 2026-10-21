"""SM49: smaller int states for SM36 (no retraining), 732 pool and its 464 subset (minus the 9 users that never fit the
0.8 GPU memory cap). Per head (curve = ahead, instant): review-weighted benchmark LogLoss; delta vs SM36 w8+s8 and vs the
variant's base (paired user bootstrap, 2,000 resamples, seed 0, 2.5/97.5 percentiles); % vs the fork (2.76M, w8+s8) on
the same users; bits per card / note and the card+note state of the own collection (13,973 cards, 13,198 notes).
Bases: A w8+s4hx8 and C' w8+s4hgx4c8hx8n vs SM47's w8+s4c8nhx8; B w8+s4hx4 vs A; C w8+s4hgx4 vs B.
usage: sm49_summary.py [rows.jsonl]   (default heldout-sm49-pools.jsonl next to this file)"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from scripts.state_vq_report import int_entity_bits  # noqa: E402

L = "s-w64kd-gate-sc-lanes-dec12-1ep32k-muon"
S42 = Path("C:/Users/jonat/workspace/wt/sm42/artifacts/small-model")
S47 = Path("C:/Users/jonat/workspace/wt/sm45/artifacts/small-model/sm47")
SMW = Path("C:/Users/jonat/workspace/wt/small-model/artifacts/small-model")
CARDS, NOTES = 13_973, 13_198
# variant -> (name, int_entity_bits args (wkv bits, scale bits, shift bits, head scale), base)
VARIANTS = {"s4hx8": ("A", (4, 16, 8, False), "s4c8nhx8"),
            "s4hx4": ("B", (4, 16, 4, False), "s4hx8"),
            "s4hgx4": ("C", (4, 16, 4, True), "s4hx4"),
            "s4hgx4c8hx8n": ("C'", ((4, 8), 16, (4, 8), (True, False)), "s4c8nhx8")}
REFS = {"s8": (8, 32, 0), "s4c8nhx8": ((4, 8), 16, 8)}


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


src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "heldout-sm49-pools.jsonl"
rows = {v: load([src], f"{L}/w8+{v}") for v in VARIANTS}
rows["s4c8nhx8"] = load([S47 / "heldout-sm47-pools.jsonl"], f"{L}/w8+s4c8nhx8")
s36 = load([S42 / "heldout-sm36-pools.jsonl"], f"{L}/w8+s8")
fork = load([SMW / "heldout.jsonl", SMW / "heldout-pools.jsonl"], "fork/w8+s8")
p464 = set(load([S42 / "heldout-sm36.jsonl"], f"{L}/w8+s8"))

lines = ["# SM49: smaller int states (int4 notes, int4 shift vectors, per-head scales), SM36 (best model), w8 weights", "",
         "State bits per entity: per WKV row 16 codes + one float16 scale (g: one float16 scale per head of 16 rows "
         "instead); per token-shift vector 64 codes (x8 int8, x4 int4) + one float16 scale. Card = 2 WKV layers x 4 heads "
         f"x 16 rows + 3 shift vectors; note = 1 x 4 x 16 rows + 2 shift vectors. Collection = {CARDS:,} cards + "
         f"{NOTES:,} notes (MB = 10^6 bytes).", "",
         "| Variant | Card bits | Note bits | Collection MB | vs w8+s8 |", "|---|---:|---:|---:|---:|"]
s8mb = mb(*int_entity_bits(8))
for v, spec in [(v, s) for v, s in REFS.items()] + [(v, s[1]) for v, s in VARIANTS.items()]:
    c, n = int_entity_bits(*spec)
    name = f"{VARIANTS[v][0]} " if v in VARIANTS else ("SM47 " if v == "s4c8nhx8" else "SM36 ")
    lines.append(f"| {name}w8+{v} | {c:,} | {n:,} | {mb(c, n):.2f} | {100 * (mb(c, n) / s8mb - 1):+.1f}% |")
lines += ["", "Review-weighted benchmark LogLoss; d s8 = variant - SM36 w8+s8, d base = variant - its base, [95% paired "
          "user bootstrap]; % vs the big model (fork w8+s8) on the same users; curve = ahead.", ""]
result = {}
for pool, keep in (("732", None), ("464", p464)):
    for v, (name, spec, bname) in VARIANTS.items():
        r, base = rows[v], rows[bname]
        us = sorted(u for u in set(r) & set(s36) & set(base) & set(fork) if keep is None or u in keep)
        if not us:
            continue
        parts, res = [], {"users": len(us), "base": f"w8+{bname}"}
        for k, hn in (("ahead", "curve"), ("instant", "instant")):
            a, s, b, f = agg(r, us, k), agg(s36, us, k), agg(base, us, k), agg(fork, us, k)
            lo, hi = boot(r, s36, us, k)
            blo, bhi = boot(r, base, us, k)
            parts.append(f"{hn} {a:.5f} ({100 * (a / f - 1):+.2f}% vs big; s8 {s:.5f} {100 * (s / f - 1):+.2f}%) "
                         f"d s8 {a - s:+.5f} [{lo:+.5f}, {hi:+.5f}]; d base {a - b:+.5f} [{blo:+.5f}, {bhi:+.5f}]")
            res[hn] = {"LogLoss": a, "pct_vs_fork": 100 * (a / f - 1), "s8": s, "d_s8": a - s, "d_s8_ci": [lo, hi],
                       "base": b, "d_base": a - b, "d_base_ci": [blo, bhi]}
        c, n = int_entity_bits(*spec)
        res.update({"card_bits": c, "note_bits": n, "collection_mb": mb(c, n)})
        result[f"{pool}/w8+{v}"] = res
        lines.append(f"- {pool} pool, {name} w8+{v} (base w8+{bname}; {c:,} / {n:,} bits, {mb(c, n):.2f} MB), "
                     f"{len(us)} users: " + "; ".join(parts))
if src == HERE / "heldout-sm49-pools.jsonl":
    (HERE / "sm49-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "sm49-summary.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
print("\n".join(lines))
