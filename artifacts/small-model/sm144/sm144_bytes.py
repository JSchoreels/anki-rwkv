#!/usr/bin/env python
"""Earlier int8 switch (SM144): bytes per card / note per tier width, the int3 / int2 code packing (kept from SM141), and
the expected bytes of each scheme on the 732 set (CPU, seconds).

Layout = the fork's compact ModuleState (rslib/src/rwkv/precision.rs; the reference encoder of the long-card state drift
on real users (SM136), sm136_bytes.py): 14-byte magic + 1 precision byte + u32 layer count; per layer three u32 lengths,
the time-shift block, the matrix block (H heads x K rows of K values) and the channel-shift block where the layer keeps
its channel mixer. Compact blocks: shift = float16 scale + D int8 codes (every tier keeps int8 shift vectors); matrix
row = float16 scale + K codes at the tier width, packed little-endian (int8: 1 byte per code, int4: 2 per byte as the
fork, int3: 8 codes in 3 bytes, int2: 4 codes per byte), so a row is 2 + ceil(K * bits / 8) bytes.

Checks (exit 1 on a failure):
  - the formula reproduces SM136's encoded sizes (sm136-bytes.json) for int8 cards / notes and int4 cards, every size;
  - pack / unpack round trip of random 3- and 2-bit codes (every code value, K = 16 and 32), bitwise;
  - the scheme totals below use sm144-dist.json (sm144_dist.py) end-of-history counts.

  python sm144_bytes.py [--out sm144-bytes.json]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SM136 = HERE.parent / "sm136" / "sm136-bytes.json"
# per size: H, K, D, card (layers, channel-mixer layers), note (layers, channel-mixer layers)
ARCH = {"Big": (4, 32, 128, (3, 3), (2, 2)), "Small": (4, 16, 64, (2, 1), (1, 1)), "Tiny": (4, 16, 64, (1, 1), (1, 1))}
SM136_KEY = {"Big": "Big (fork)", "Small": "Small (w64)", "Tiny": "Tiny (w64-c1)"}   # the six-layer Tiny (w64-c1-d2p1u1) has the same card / note layers (1 + 1)
QMAX = {8: 127, 4: 7, 3: 3, 2: 1}
# the schemes of design.md: (card spec, note spec); spec = (young bits, M, mid bits, N, old bits), young for n <= M,
# old for n >= N (n = answers of the entity)
SCHEMES = {
    "REF": ((8, 0, 8, None, 8), (8, 0, 8, None, 8)),
    "A": ((4, 0, 4, 128, 8), (8, 0, 8, None, 8)),
    "A32": ((4, 0, 4, 32, 8), (8, 0, 8, None, 8)),
    "A16": ((4, 0, 4, 16, 8), (8, 0, 8, None, 8)),
    "AN32": ((4, 0, 4, 32, 8), (4, 0, 4, 32, 8)),
    "AN16": ((4, 0, 4, 16, 8), (4, 0, 4, 16, 8)),
}


def entity_bytes(size, kind, bits):
    H, K, D, card, note = ARCH[size]
    layers, cm = card if kind == "card" else note
    return 14 + 1 + 4 + layers * 12 + (layers + cm) * (2 + D) + layers * H * K * (2 + math.ceil(K * bits / 8))


def pack(codes, bits):
    """Little-endian bit packing of signed codes in [-qmax, qmax] (two's complement in `bits` bits)."""
    u = (np.asarray(codes, dtype=np.int64) & ((1 << bits) - 1)).astype(np.uint64)
    out = bytearray(math.ceil(len(u) * bits / 8))
    acc, nacc, pos = 0, 0, 0
    for v in u:
        acc |= int(v) << nacc
        nacc += bits
        while nacc >= 8:
            out[pos] = acc & 255
            acc >>= 8
            nacc -= 8
            pos += 1
    if nacc:
        out[pos] = acc & 255
    return bytes(out)


def unpack(buf, n, bits):
    acc = int.from_bytes(buf, "little")
    mask, sign = (1 << bits) - 1, 1 << (bits - 1)
    vals = [(acc >> (i * bits)) & mask for i in range(n)]
    return np.array([v - (1 << bits) if v & sign else v for v in vals], dtype=np.int64)


def tier_of(n, spec):
    young, M, mid, N, old = spec
    return np.where(n <= M, young, np.where((n >= N) if N is not None else False, old, mid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "sm144-bytes.json"))
    args = ap.parse_args()
    ok = True
    table = {s: {k: {b: entity_bytes(s, k, b) for b in (8, 4, 3, 2)} for k in ("card", "note")} for s in ARCH}
    known = json.loads(SM136.read_text())["arms"]
    checks = {}
    for s in ARCH:
        arm = known[SM136_KEY[s]]
        want = {"card8": arm["w8+s8hx8"]["card"]["bytes"], "note8": arm["w8+s8hx8"]["note"]["bytes"],
                "card4": arm["w8+s4c8nhx8"]["card"]["bytes"]}
        got = {"card8": table[s]["card"][8], "note8": table[s]["note"][8], "card4": table[s]["card"][4]}
        checks[s] = {"sm136": want, "formula": got, "equal": want == got}
        ok &= want == got
    rng = np.random.default_rng(0)
    rt = {}
    for bits in (3, 2, 4):
        for K in (16, 32):
            codes = rng.integers(-QMAX[bits], QMAX[bits] + 1, size=K * 64)
            codes[:2 * QMAX[bits] + 1] = np.arange(-QMAX[bits], QMAX[bits] + 1)
            same = all(np.array_equal(unpack(pack(row, bits), K, bits), row) and len(pack(row, bits)) == math.ceil(K * bits / 8)
                       for row in codes.reshape(-1, K))
            rt[f"int{bits}/K{K}"] = bool(same)
            ok &= same
    out = {"layout": "fork compact ModuleState; int8 shift vectors in every tier; matrix row = f16 scale + packed codes",
           "bytes": table, "check_vs_sm136": checks, "pack_round_trip": rt}
    dist_path = HERE / "sm144-dist.json"
    if dist_path.exists():
        dist = json.loads(dist_path.read_text())
        # histograms are not stored; rebuild the tier shares from the stored tier rows (end of history)
        rows = {(t["young_bits"], t["M"], t["N"]): t for t in dist["tiers"]}
        cards, notes = dist["answered_cards"], dist["notes_with_answers"]
        tot = {}
        for name, (cs, ns) in SCHEMES.items():
            tot[name] = {}
            for s in ARCH:
                per = {}
                for kind, spec, count in (("card", cs, cards), ("note", ns, notes)):
                    young, M, mid, N, old = spec
                    if N is None:   # uniform
                        mean = table[s][kind][mid]
                    else:
                        t = rows[(young if M else None, M, N)][kind]["entities"]
                        mean = t[0] * table[s][kind][young] + t[1] * table[s][kind][mid] + t[2] * table[s][kind][old]
                    per[kind] = round(mean, 1)
                ref = cards * table[s]["card"][8] + notes * table[s]["note"][8]
                total = cards * per["card"] + notes * per["note"]
                tot[name][s] = {"mean_card_bytes": per["card"], "mean_note_bytes": per["note"],
                                "total_mb_732": round(total / 1e6, 1), "vs_REF_pct": round(100 * (total / ref - 1), 2)}
        out["schemes"] = {k: {"card": v[0], "note": v[1]} for k, v in SCHEMES.items()}
        out["scheme_bytes_732"] = tot
        out["entities_732"] = {"answered_cards": cards, "notes_with_answers": notes,
                               "never_answered_cards (0 bytes)": dist["never_answered_cards"]}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"[sm144-bytes] {'OK' if ok else 'FAILED'}: sizes vs SM136 {all(c['equal'] for c in checks.values())}, "
          f"pack round trip {all(rt.values())}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
