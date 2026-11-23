#!/usr/bin/env python
"""Smaller shift vectors and factors on T24 states (SM152): bytes per card / note state of the labels on the 732 set,
plus the bit-packing round trip of the new code widths (CPU, seconds; aggregates only).

Layout: the fork's compact module state of the low-rank int8 states (SM148) (sm148_bytes.entity_bytes), with the code
widths as parameters and every code vector bit-packed (little-endian, two's complement, sm145_bytes.pack):
  header   14-byte magic, 1 precision byte, u32 layer count, per layer three u32 lengths, 1 stored-rank byte
  shifts   (layers + channel-mixer layers) shift vectors, each a float16 scale + ceil(D x shift bits / 8) bytes
  WKV      per layer and head at rank c: c left and c right factor vectors, each a float16 scale + ceil(K x factor bits / 8)
           bytes (int8: 2 c K + 4 c bytes, SM148); REF: K rows of a float16 scale + K int8 codes, no rank byte
An entity with n answers stores rank min(n, r_tier(n)) (T24: rank 2 below 16 answers, rank 4 from 16). Shares from the
end-of-history answer counts of the 723 fit users of the 732 set (../sm145/sm145-dist.json), as SM148 / SM150.

  python sm152_bytes.py [--out sm152-bytes.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DIST = HERE.parent / "sm145" / "sm145-dist.json"
sys.path.insert(0, str(HERE.parent / "sm145"))
from sm145_bytes import pack, unpack  # noqa: E402

ARCH = {"Big": (4, 32, 128, (3, 3), (2, 2)), "Small": (4, 16, 64, (2, 1), (1, 1)), "Tiny": (4, 16, 64, (1, 1), (1, 1))}
T24 = [(1, 2), (16, 4)]
# label -> (tiers [(first n, rank)] or None = REF's int8 rows, factor bits, shift bits)
LABELS = {"REF": (None, 8, 8), "T24": (T24, 8, 8), "S4": (T24, 8, 4), "F6": (T24, 6, 8), "F6S4": (T24, 6, 4),
          "Q2": ([(1, 2)], 8, 8)}   # Q2: report only (rank 2 everywhere, SM148 / SM150)
SCORED = ("T24", "S4", "F6", "F6S4")
QMAX = {8: 127, 6: 31, 4: 7}
REP = {"1": 1, "2": 2, "3": 3, "4-15": 4, "16-127": 16, "128+": 128}


def vec(n, bits):
    """Bytes of one code vector: float16 scale + n codes of `bits` bits, bit-packed."""
    return 2 + math.ceil(n * bits / 8)


def parts(size, kind, rank, fbits, sbits):
    """{header, shifts, wkv} bytes of one entity (rank None = REF's int8 rows)."""
    H, K, D, card, note = ARCH[size]
    layers, cm = card if kind == "card" else note
    header = 14 + 1 + 4 + layers * 12 + (0 if rank is None else 1)
    shifts = (layers + cm) * vec(D, sbits)
    wkv = layers * H * K * (2 + K) if rank is None else layers * H * 2 * rank * vec(K, fbits)
    return {"header": header, "shifts": shifts, "wkv": wkv}


def entity_bytes(size, kind, rank, fbits=8, sbits=8):
    return sum(parts(size, kind, rank, fbits, sbits).values())


def shares(dist, kind):
    """{n-class: share of answered entities}: classes '1', '2', '3', '4-15', '16-127', '128+' (as sm148_bytes)."""
    d = dist["cards" if kind == "card" else "notes"]["share_le"]
    tiers = {t["N"]: t for t in dist["tiers"]}
    lt16 = sum(tiers[16][kind]["entities"][:2])
    lt128 = sum(tiers[128][kind]["entities"][:2]) if 128 in tiers else d["127"]
    out = {"1": d["1"], "2": d["2"] - d["1"], "3": d["3"] - d["2"], "4-15": lt16 - d["3"], "16-127": lt128 - lt16,
           "128+": 1 - lt128}
    assert abs(sum(out.values()) - 1) < 1e-9 and min(out.values()) >= 0, out
    return out


def rank_of(tiers, n):
    if tiers is None:
        return None
    kind = None
    for lo, r in tiers:
        if n >= lo:
            kind = r
    return min(n, kind)


def roundtrip():
    """Pack / unpack of random codes at every width used, on the vector lengths of the three sizes; the packed length is
    the byte count of the layout; a real int6 / int4 quantisation (scripts/state_lowrank.quant_vectors, REF's
    int_row_quant at width 4) decodes back to the same floats from its packed codes and float16 scale."""
    rng = np.random.default_rng(0)
    out = {}
    for bits, qmax in QMAX.items():
        for n in (16, 32, 64, 128):
            for _ in range(20):
                codes = rng.integers(-qmax, qmax + 1, n)
                buf = pack(codes, bits)
                assert len(buf) == math.ceil(n * bits / 8), (bits, n, len(buf))
                assert np.array_equal(unpack(buf, n, bits), codes), (bits, n)
            out[f"int{bits} n{n}"] = math.ceil(n * bits / 8)
    root = HERE.parents[2]
    sys.path.insert(0, str(root))
    import torch
    from scripts.small_model_wkv import round_scale_f16
    from scripts.state_lowrank import quant_vectors
    from scripts.state_vq import int_row_quant
    g = torch.Generator().manual_seed(0)
    for K in (16, 32):
        x = torch.randn(64, K, generator=g) * 0.37
        for qmax, bits in ((31, 6), (127, 8)):
            q = quant_vectors(x, -1, qmax)
            s = round_scale_f16(x.abs().amax(-1, keepdim=True) / float(qmax))
            for i in range(x.shape[0]):
                codes = torch.round(q[i] / s[i]).to(torch.int64).numpy()
                back = torch.from_numpy(unpack(pack(codes, bits), K, bits)).float() * s[i]
                assert torch.equal(back, q[i]), (K, qmax, i)
    for D in (64, 128):
        x = torch.randn(64, D, generator=g) * 0.8
        q = int_row_quant(4 | 256)(x)   # x4: REF's shift format at width 4 (float16 scale, codes in [-7, 7])
        s = round_scale_f16(x.abs().amax(-1, keepdim=True) / 7.0)
        for i in range(x.shape[0]):
            codes = torch.round(q[i] / s[i]).to(torch.int64).numpy()
            assert np.abs(codes).max() <= 7
            back = torch.from_numpy(unpack(pack(codes, 4), D, 4)).float() * s[i]
            assert torch.equal(back, q[i]), (D, i)
    out["decode_equals_codec"] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "sm152-bytes.json"))
    args = ap.parse_args()
    dist = json.loads(DIST.read_text(encoding="utf-8"))
    ncard, nnote = dist["answered_cards"], dist["notes_with_answers"]
    out = {"note": "aggregates only; bytes per answered card / note with answers, 723 fit users of the 732 set",
           "layout": "fork compact module state, bit-packed codes; vector = float16 scale + ceil(n bits / 8) bytes; "
                     "+1 rank byte for the low-rank labels",
           "labels": {k: {"tiers": v[0], "factor_bits": v[1], "shift_bits": v[2]} for k, v in LABELS.items()},
           "shares": {k: shares(dist, k) for k in ("card", "note")}, "sizes": {}}
    for size in ARCH:
        row = {}
        for label, (tiers, fb, sb) in LABELS.items():
            mean, split = {}, {}
            for kind in ("card", "note"):
                sh = out["shares"][kind]
                mean[kind] = sum(sh[c] * entity_bytes(size, kind, rank_of(tiers, REP[c]), fb, sb) for c in sh)
                split[kind] = {p: round(sum(sh[c] * parts(size, kind, rank_of(tiers, REP[c]), fb, sb)[p] for c in sh), 1)
                               for p in ("header", "shifts", "wkv")}
            row[label] = {"card_mean": round(mean["card"], 1), "note_mean": round(mean["note"], 1),
                          "card_parts_mean": split["card"], "note_parts_mean": split["note"],
                          "total": mean["card"] * ncard + mean["note"] * nnote}
        ref, t24 = row["REF"]["total"], row["T24"]["total"]
        for label in row:
            row[label]["vs_int8_pct"] = round(100 * (row[label]["total"] / ref - 1), 2)
            row[label]["vs_T24_pct"] = round(100 * (row[label]["total"] / t24 - 1), 2)
            row[label]["total"] = round(row[label]["total"])
        out["sizes"][size] = row
    # checks: REF / T24 / Q2 equal SM148's byte table; the scored byte order F6S4 < (S4, F6) < T24 < REF
    sm148 = json.loads((HERE.parent / "sm148" / "sm148-bytes.json").read_text())["sizes"]
    for size, row in out["sizes"].items():
        for lab in ("REF", "T24", "Q2"):
            assert row[lab]["card_mean"] == sm148[size][lab]["card_mean"] and row[lab]["note_mean"] == sm148[size][lab]["note_mean"], (size, lab)
            assert row[lab]["vs_int8_pct"] == sm148[size][lab]["vs_int8_pct"], (size, lab)
        assert row["F6S4"]["total"] < min(row["S4"]["total"], row["F6"]["total"]), size
        assert max(row["S4"]["total"], row["F6"]["total"]) < row["T24"]["total"] < row["REF"]["total"], size
    out["order"] = {size: sorted(SCORED, key=lambda s: row[s]["total"]) for size, row in out["sizes"].items()}
    out["roundtrip"] = roundtrip()
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for size, row in out["sizes"].items():
        print(size, {k: (v["card_mean"], v["note_mean"], v["vs_int8_pct"], v["vs_T24_pct"]) for k, v in row.items()},
              "order", out["order"][size])
    print("[sm152-bytes] OK")


if __name__ == "__main__":
    main()
