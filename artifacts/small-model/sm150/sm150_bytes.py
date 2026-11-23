#!/usr/bin/env python
"""Rank-1 int8 states (SM150): bytes per card / note state of the scored labels on
the 732 set (CPU, seconds; aggregates only).

Layout: the fork's compact module state as in the int3 young tier (SM145) (sm145_bytes.entity_bytes): 14-byte magic,
1 precision byte, u32 layer count, per layer three u32 lengths, int8 shift vectors (float16 scale + D codes) for the
time mixer and, where kept, the channel mixer, unchanged. The WKV block of one head at rank c: c left and c right
factor vectors of K int8 codes, each with its own float16 scale (2 c K + 4 c bytes; REF: K rows of 2 + K bytes),
plus 1 byte per entity for the stored rank (all heads and layers of an entity share min(n, r)). An entity with n
answers stores rank min(n, r_tier(n)) (exact for n <= r). Shares from the end-of-history answer counts of the 723
users of the 732 set (../sm145/sm145-dist.json: share_le for n <= 1, 2, 3 and the tier entity shares for n < 16).

  python sm150_bytes.py [--out sm150-bytes.json]
"""
import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIST = HERE.parents[3] / "sm145" / "artifacts" / "small-model" / "sm145" / "sm145-dist.json"
ARCH = {"Big": (4, 32, 128, (3, 3), (2, 2)), "Small": (4, 16, 64, (2, 1), (1, 1)), "Tiny": (4, 16, 64, (1, 1), (1, 1))}
# label -> tiers [(first n, rank)], rank None = int8 rows (REF)
LABELS = {"REF": [(1, None)], "Q4": [(1, 4)], "Q2": [(1, 2)], "R12": [(1, 1), (16, 2)], "R1": [(1, 1)]}
# current best tiered int formats (int3 young tier (SM145) / aggressive state tiers for Tiny (SM146) go picks), vs int8
BEST_TIERED = {"Big": -43.48, "Small": -38.57, "Tiny": -42.71}


def overhead(size, kind):
    H, K, D, card, note = ARCH[size]
    layers, cm = card if kind == "card" else note
    return 14 + 1 + 4 + layers * 12 + (layers + cm) * (2 + D), layers, H, K


def entity_bytes(size, kind, rank):
    base, layers, H, K = overhead(size, kind)
    if rank is None:
        return base + layers * H * K * (2 + K)
    return base + layers * H * (2 * rank * K + 4 * rank) + 1


def shares(dist, kind):
    """{n-class: share of answered entities}: classes '1', '2', '3', '4-15', '16-127', '128+'."""
    d = dist["cards" if kind == "card" else "notes"]["share_le"]
    tiers = {t["N"]: t for t in dist["tiers"]}
    lt16 = sum(tiers[16][kind]["entities"][:2])   # share n < 16 (young + mid tiers)
    lt128 = sum(tiers[128][kind]["entities"][:2]) if 128 in tiers else d["127"]
    out = {"1": d["1"], "2": d["2"] - d["1"], "3": d["3"] - d["2"], "4-15": lt16 - d["3"], "16-127": lt128 - lt16,
           "128+": 1 - lt128}
    assert abs(sum(out.values()) - 1) < 1e-9 and min(out.values()) >= 0, out
    return out


REP = {"1": 1, "2": 2, "3": 3, "4-15": 4, "16-127": 16, "128+": 128}


def rank_of(tiers, n):
    kind = None
    for lo, r in tiers:
        if n >= lo:
            kind = r
    return None if kind is None else min(n, kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "sm150-bytes.json"))
    args = ap.parse_args()
    dist = json.loads(DIST.read_text(encoding="utf-8"))
    ncard, nnote = dist["answered_cards"], dist["notes_with_answers"]
    out = {"note": "aggregates only; bytes per answered card / note with answers, 723 fit users of the 732 set",
           "layout": "fork compact module state; head WKV at rank c = 2cK int8 codes + 2c float16 scales; +1 rank byte",
           "shares": {k: shares(dist, k) for k in ("card", "note")}, "sizes": {}}
    for size in ARCH:
        row = {}
        for label, tiers in LABELS.items():
            mean = {}
            for kind in ("card", "note"):
                sh = out["shares"][kind]
                mean[kind] = sum(sh[c] * entity_bytes(size, kind, rank_of(tiers, REP[c])) for c in sh)
            total = mean["card"] * ncard + mean["note"] * nnote
            row[label] = {"card_mean": round(mean["card"], 1), "note_mean": round(mean["note"], 1),
                          "card_r_full": {r: entity_bytes(size, "card", r) for r in (1, 2, 4)},
                          "note_r_full": {r: entity_bytes(size, "note", r) for r in (1, 2, 4)},
                          "total": total}
        ref = row["REF"]["total"]
        for label in row:
            row[label]["vs_int8_pct"] = round(100 * (row[label]["total"] / ref - 1), 2)
            row[label]["total"] = round(row[label]["total"])
        row["best_tiered_vs_int8_pct"] = BEST_TIERED[size]
        out["sizes"][size] = row
    # checks: REF entity bytes = SM145's int8 sizes; scored byte order Q2 < T24 < Q4 < REF for every size
    assert entity_bytes("Big", "card", None) == 13891 and entity_bytes("Big", "note", None) == 9267
    assert entity_bytes("Small", "card", None) == 2545 and entity_bytes("Small", "note", None) == 1315
    for size, row in out["sizes"].items():
        assert row["R1"]["total"] < row["R12"]["total"] < row["Q2"]["total"] < row["Q4"]["total"] < row["REF"]["total"], size
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for size, row in out["sizes"].items():
        print(size, {k: (v["card_mean"], v["note_mean"], v["vs_int8_pct"]) for k, v in row.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
