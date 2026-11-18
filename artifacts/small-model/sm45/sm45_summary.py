"""SM45: int4 card state + int8 note state (+s4c8n) and int4 everywhere (+s4) for SM36 (the best model) on the 732 pool
and its 464 subset. Per head (curve = ahead, instant): review-weighted benchmark LogLoss; delta vs SM36 w8+s8 (paired user
bootstrap, 2,000 resamples, seed 0, 2.5/97.5 percentiles) and % vs the fork (2.76M, w8+s8) on the same users."""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
L = "s-w64kd-gate-sc-lanes-dec12-1ep32k-muon"
S42 = Path("C:/Users/jonat/workspace/wt/sm42/artifacts/small-model")
SMW = Path("C:/Users/jonat/workspace/wt/small-model/artifacts/small-model")


def load(files, label):
    out = {}
    for f in files:
        if Path(f).exists():
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                if r.get("label") == label and "benchmark" in r:
                    out[r["user_id"]] = r
    return out


rows = {v: load([HERE / "heldout-sm45-pools.jsonl"], f"{L}/w8+{v}") for v in ("s4c8n", "s4")}
base = load([S42 / "heldout-sm36-pools.jsonl"], f"{L}/w8+s8")
fork = load([SMW / "heldout.jsonl", SMW / "heldout-pools.jsonl"], "fork/w8+s8")
p464 = set(load([S42 / "heldout-sm36.jsonl"], f"{L}/w8+s8"))


def agg(r, us, k):
    n = sum(r[u]["benchmark"][k]["size"] for u in us)
    return sum(r[u]["benchmark"][k]["LogLoss"] * r[u]["benchmark"][k]["size"] for u in us) / n


lines = ["# SM45: int4 card state + int8 note state, SM36 (best model), w8 weights", "",
         "Review-weighted benchmark LogLoss; delta = variant - SM36 w8+s8 [95% paired user bootstrap]; % vs the big model "
         "(fork w8+s8) on the same users; curve = ahead.", ""]
for pool, keep in (("732", None), ("464", p464)):
    for v, r in rows.items():
        us = sorted(u for u in set(r) & set(base) & set(fork) if keep is None or u in keep)
        if not us:
            continue
        parts = []
        for k, name in (("ahead", "curve"), ("instant", "instant")):
            a, b, f = agg(r, us, k), agg(base, us, k), agg(fork, us, k)
            rng, bs = random.Random(0), []
            for _ in range(2000):
                s = [rng.choice(us) for _ in us]
                bs.append(agg(r, s, k) - agg(base, s, k))
            bs.sort()
            parts.append(f"{name} {a:.5f} ({100 * (a / f - 1):+.2f}% vs big; s8 {b:.5f} {100 * (b / f - 1):+.2f}%) "
                         f"delta {a - b:+.5f} [{bs[49]:+.5f}, {bs[1949]:+.5f}]")
        lines.append(f"- {pool} pool, w8+{v}, {len(us)} users: " + "; ".join(parts))
(HERE / "sm45-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
