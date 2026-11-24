#!/usr/bin/env python
"""Energy-triggered rank growth (SM159): state bytes from the rank read-out (state_lowrank.RANK_READOUT aggregates).

Layout: the fork's compact module state of the smaller shift vectors and factors on T24 states (SM152)
(sm152_bytes.parts), bit-packed codes:
  header   14-byte magic, 1 precision byte, u32 layer count, per layer three u32 lengths, plus the stored-rank field:
           1 byte per entity for the plain rank labels (SM148 / SM152: every head shares min(n, r)); for the energy
           labels a 2-bit cap per (layer, head) (caps 2, 3, 4), ceil(layers x heads x 2 / 8) bytes per entity
  shifts   (layers + channel-mixer layers) shift vectors, float16 scale + ceil(D x 8 / 8) bytes each (int8, x8)
  WKV      per layer and head at stored rank c = min(n, cap): 2 c factor vectors, float16 scale + ceil(K x bits / 8)
           bytes each; REF: K rows of a float16 scale + K int8 codes, no rank field
Totals over all answered cards and notes of the users of a read-out (end-of-history ranks), compared with REF's int8
rows on the same entities (bytes vs int8, %) and with T24 (`w8+lr2.16.4q8x8`, or its int6 analog), whose stored
ranks follow from the answer counts: min(n, 2) below 16 answers, 4 from 16 (from entities, entities_n1,
entities_ge16 of the same read-out, so every label is compared on exactly the same entities).
"""
import math

ARCH = {"big": (4, 32, 128, (3, 3), (2, 2)), "small": (4, 16, 64, (2, 1), (1, 1)), "tiny": (4, 16, 64, (1, 1), (1, 1))}


def vec(n, bits):
    return 2 + math.ceil(n * bits / 8)


def stream_bytes(size, kind, st, fbits=8, energy=False, sum_rank=None, rows=False):
    """Total bytes of one stream's answered entities. st: a RANK_READOUT stream entry (entities, layers, H, K,
    sum_rank); sum_rank overrides st's (T24 / Q2 from answer counts); rows = REF's int8 rows."""
    H, K, D, card, note = ARCH[size]
    layers, cm = card if kind == "card" else note
    if (st["H"], st["K"], len(st["layers"])) != (H, K, layers):
        raise SystemExit(f"SM159 bytes: {size} {kind} read-out layout {st['H'], st['K'], st['layers']} "
                         f"!= the layout {(H, K, layers)}")
    E = st["entities"]
    shifts = (layers + cm) * vec(D, 8)
    base = 14 + 1 + 4 + layers * 12
    if rows:
        return E * (base + shifts + layers * H * K * (2 + K))
    meta = math.ceil(layers * H * 2 / 8) if energy else 1
    sr = st["sum_rank"] if sum_rank is None else sum_rank
    return E * (base + meta + shifts) + sr * 2 * vec(K, fbits)


def tier_sum_rank(st, tiers):
    """sum over entities, layers and heads of min(n, tier rank) for tiers 'T24' (2 then 4 from 16) or 'Q2'."""
    LH = len(st["layers"]) * st["H"]
    n1, E, E16 = st["entities_n1"], st["entities"], st["entities_ge16"]
    if tiers == "T24":
        return LH * (n1 + 2 * (E - n1 - E16) + 4 * E16)
    if tiers == "Q2":
        return LH * (n1 + 2 * (E - n1))
    raise ValueError(tiers)


def label_bytes(size, readout, fbits=8, energy=True):
    """{"label": bytes, "REF": bytes, "T24": bytes, "Q2": bytes} summed over card and note (T24 / Q2 at fbits)."""
    out = {"label": 0, "REF": 0, "T24": 0, "Q2": 0}
    for kind in ("card", "note"):
        st = readout[kind]
        out["label"] += stream_bytes(size, kind, st, fbits, energy=energy)
        out["REF"] += stream_bytes(size, kind, st, rows=True)
        out["T24"] += stream_bytes(size, kind, st, fbits, sum_rank=tier_sum_rank(st, "T24"))
        out["Q2"] += stream_bytes(size, kind, st, fbits, sum_rank=tier_sum_rank(st, "Q2"))
    return out


def selftest():
    """SM148's per-entity sizes (sm148-bytes.json card_r_full / REF) from one-entity read-outs."""
    def one(size, kind, n, r):
        H, K, D, card, note = ARCH[size]
        layers = (card if kind == "card" else note)[0]
        return {"entities": 1, "entities_n1": int(n == 1), "entities_ge16": int(n >= 16), "H": H, "K": K,
                "layers": list(range(layers)), "sum_rank": layers * H * min(n, r)}
    assert stream_bytes("big", "card", one("big", "card", 5, 2), rows=True) == 13891
    assert stream_bytes("big", "note", one("big", "note", 5, 2), rows=True) == 9267
    assert stream_bytes("small", "card", one("small", "card", 5, 2), rows=True) == 2545
    assert stream_bytes("small", "note", one("small", "note", 5, 2), rows=True) == 1315
    # SM148 Big card at rank 2 / 4 (sm148-bytes.json card_r_full): 2,388 and ... from 2 c K + 4 c per head + 1
    H, K = 4, 32
    base = 14 + 1 + 4 + 3 * 12 + 1 + (3 + 3) * (2 + 128)
    assert stream_bytes("big", "card", one("big", "card", 5, 2)) == base + 3 * H * (2 * 2 * K + 4 * 2)
    st = one("big", "card", 20, 4)
    assert stream_bytes("big", "card", st, sum_rank=tier_sum_rank(st, "T24")) == base + 3 * H * (2 * 4 * K + 4 * 4)
    # int6 factors: ceil(32 x 6 / 8) = 24 code bytes per vector
    assert stream_bytes("big", "card", one("big", "card", 5, 2), fbits=6) == base + 3 * H * 2 * 2 * (2 + 24)
    # energy rank field: Big card 3 layers x 4 heads x 2 bits = 3 bytes
    assert stream_bytes("big", "card", one("big", "card", 5, 2), energy=True) == \
        stream_bytes("big", "card", one("big", "card", 5, 2)) + 2
    print("sm159_bytes selftest OK")


if __name__ == "__main__":
    selftest()
