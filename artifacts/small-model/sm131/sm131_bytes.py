#!/usr/bin/env python
"""Compact state for Small and Big (SM131): serialized bytes per card and per note, and a round-trip proof (CPU, seconds).

SM131: a copy of the compact state for Tiny (SM129)'s sm129_bytes.py with the dimensions per architecture: Small and
Tiny 4 heads x 16 (64-wide shift vectors); Big (the fork architecture, the show-time big model (SM91s2) and the
show-time instant-only big model (SM99)) 4 heads x 32 (128-wide shift vectors), 3 card layers and 2 note layers, all
with a channel mixer (architecture.json of the Big packages). Known Big totals (the package INTEGRATION.md section 7 and
the fork speed test of 11-14): int4 card 9,271 B, int8 card 15,415 B, int8 note 10,283 B.


A reference encoder / decoder of the fork's stored entity state (rslib/src/rwkv/precision.rs, ModuleState::compact, as
described in wt/sm73s2/artifacts/small-model/sm73s2/DESIGN-ADDENDUM.md "Bytes per card + note corrected"):
  14-byte magic + 1 precision byte + u32 layer count;
  per layer: three u32 lengths (time shift, matrix, channel shift), the time-shift block, the matrix block (4 heads x 16
  rows of 16 values), and the channel-shift block only where the layer keeps its channel mixer.
Blocks per format (the formats the scorer emulates, scripts/small_model_heldout.py parse_variant):
  fork-int8   (w8+s8, the published benchmark state)  shift f16 x 64 (128 B); row = f32 scale + 16 int8 (20 B)
  fork-int4   (w8+s4 rows)                            shift f16 x 64 (128 B); row = f32 scale + 16 int4 packed (12 B)
  compact     (SM47 "h" + "x8")                        shift = f16 scale + 64 int8 (66 B); row = f16 scale + codes
  card-only   (SM49 "hg" + "x4", cards only)           shift = f16 scale + 64 int4 packed (34 B); matrix = 4 f16 head
                                                       scales + 64 rows x 8 B packed int4 (520 B)
Entity formats of the arms (card / note):
  published w8+s8           fork-int8 / fork-int8
  fork default w8+s4c8n     fork-int4 / fork-int8        (what the fork stores today; shifts f16 in the fork, float in
                                                           the scorer: f16 shift rounding is not emulated by the scorer)
  compact w8+s4c8nhx8       compact int4 / compact int8  (SM47)
  card-only w8+s4hgx4c8hx8n card-only int4 / compact int8 (SM49 arm C')

Round trip: for random states (seeded; normal, zero, tiny and huge rows so the f16 clamp and the zero-scale rule are
hit) the scorer's own quantizer (scripts/state_vq.int_row_quant with the variant's bits) gives q(x); the encoder writes
q(x) as codes + scales; the decoder reads them back. Pass = decode(encode(q(x))) == q(x) bitwise for every block, the
byte length equals the known fork totals. Reported, not required: q(decode(...)) == decode(...) (re-quantizing a
read state without an update; the fork never does that, it stores the state after the next update, and at float16
subnormal scales the scorer's quantizer is not idempotent).

  python sm131_bytes.py [--out sm131-bytes.json]
"""
import argparse
import json
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.small_model_wkv import F16_SCALE, HEAD_SCALE  # noqa: E402
from scripts.state_vq import int_row_quant  # noqa: E402

MAGIC = b"ANKI-RWKV-ST01"   # 14 bytes (the fork's own magic is also 14 bytes; only the length matters here)
assert len(MAGIC) == 14
H, K, D = 4, 16, 64   # set per architecture in main() (Big: 4, 32, 128)
DIMS = {"w64": (4, 16, 64), "w64-c1": (4, 16, 64), "w64-c1-d2p1u1": (4, 16, 64), "fork": (4, 32, 128)}
BIG_ARCH = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/hf-family/repo-new/rwkv-big-two-head/architecture.json")
# block specs: (shift kind, matrix kind); precision byte = an index (the fork's byte for its formats, new values else)
BLOCKS = {"fork-int8": ("f16", "row-f32-int8"), "fork-int4": ("f16", "row-f32-int4"),
          "compact-int8": ("q8-f16", "row-f16-int8"), "compact-int4": ("q8-f16", "row-f16-int4"),
          "cardonly-int4": ("q4-f16", "head-f16-int4")}
PRECISION = {k: i for i, k in enumerate(BLOCKS)}
ARMS = {"w8+s8": ("fork-int8", "fork-int8"), "w8+s4c8n": ("fork-int4", "fork-int8"),
        "w8+s4c8nhx8": ("compact-int4", "compact-int8"), "w8+s4hgx4c8hx8n": ("cardonly-int4", "compact-int8")}
# the scorer bits of each block (scripts/small_model_heldout.parse_variant): matrix bits, shift bits (0 = float)
BITS = {"fork-int8": (8, 0), "fork-int4": (4, 0), "compact-int8": (8 | F16_SCALE, 8 | F16_SCALE),
        "compact-int4": (4 | F16_SCALE, 8 | F16_SCALE), "cardonly-int4": (4 | F16_SCALE | HEAD_SCALE, 4 | F16_SCALE)}


def pack4(codes):
    c = (np.asarray(codes, dtype=np.int8).reshape(-1, 2) & 0x0F).astype(np.uint8)
    return (c[:, 0] | (c[:, 1] << 4)).tobytes()


def unpack4(buf, n):
    b = np.frombuffer(buf, dtype=np.uint8)
    lo, hi = (b & 0x0F).astype(np.int8), (b >> 4).astype(np.int8)
    out = np.stack([lo, hi], 1).reshape(-1)[:n]
    return np.where(out > 7, out - 16, out).astype(np.int8)


def f16(x):
    return np.float16(x).tobytes()


def quant_parts(x, bits):
    """(codes int8 array, scale float32 array) exactly as scripts/state_vq.int_row_quant computes them."""
    from scripts.small_model_wkv import round_scale_f16
    qmax = 127.0 if (bits & 255) == 8 else 7.0
    dims = (-2, -1) if bits & HEAD_SCALE else -1
    scale = x.abs().amax(dims, keepdim=True) / qmax
    if bits & F16_SCALE:
        scale = round_scale_f16(scale)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return codes, scale


def enc_shift(kind, v):
    if kind == "f16":
        return v.numpy().astype(np.float16).tobytes(), v.half().float()
    bits = 8 | F16_SCALE if kind == "q8-f16" else 4 | F16_SCALE
    codes, scale = quant_parts(v, bits)
    c = codes.numpy().astype(np.int8)
    body = c.tobytes() if bits & 255 == 8 else pack4(c)
    return f16(scale.item()) + body, codes * scale


def dec_shift(kind, buf):
    if kind == "f16":
        return torch.from_numpy(np.frombuffer(buf, dtype=np.float16).astype(np.float32).copy())
    scale = np.frombuffer(buf[:2], dtype=np.float16).astype(np.float32)[0]
    codes = np.frombuffer(buf[2:], dtype=np.int8) if kind == "q8-f16" else unpack4(buf[2:], D)
    return torch.from_numpy(codes.astype(np.float32)) * torch.tensor(scale)


def enc_matrix(kind, s, bits):
    """s = [H, K, K] (head, row, column); rows quantized as the WKV kernel stores them."""
    codes, scale = quant_parts(s, bits)
    c = codes.numpy().astype(np.int8)
    sc = scale.numpy().astype(np.float32)
    out = b""
    if kind.startswith("head"):
        out += b"".join(f16(sc[h, 0, 0]) for h in range(H))
        out += b"".join(pack4(c[h, r]) for h in range(H) for r in range(K))
    else:
        for h in range(H):
            for r in range(K):
                out += (struct.pack("<f", float(sc[h, r, 0])) if "f32" in kind else f16(sc[h, r, 0]))
                out += c[h, r].tobytes() if kind.endswith("int8") else pack4(c[h, r])
    return out, codes * scale


def dec_matrix(kind, buf):
    out = torch.zeros(H, K, K)
    pos = 0
    if kind.startswith("head"):
        scales = np.frombuffer(buf[:2 * H], dtype=np.float16).astype(np.float32)
        pos = 2 * H
        for h in range(H):
            for r in range(K):
                out[h, r] = torch.from_numpy(unpack4(buf[pos:pos + K // 2], K).astype(np.float32)) * float(scales[h])
                pos += K // 2
        return out
    sbytes = 4 if "f32" in kind else 2
    cbytes = K if kind.endswith("int8") else K // 2
    for h in range(H):
        for r in range(K):
            raw = buf[pos:pos + sbytes]
            scale = struct.unpack("<f", raw)[0] if sbytes == 4 else float(np.frombuffer(raw, dtype=np.float16)[0])
            pos += sbytes
            codes = np.frombuffer(buf[pos:pos + cbytes], dtype=np.int8) if cbytes == K else unpack4(buf[pos:pos + cbytes], K)
            pos += cbytes
            out[h, r] = torch.from_numpy(codes.astype(np.float32)) * torch.tensor(np.float32(scale))
    return out


def encode_entity(fmt, layers):
    """layers = [(time_shift [64], matrix [H, K, K], channel_shift [64] or None)] -> (bytes, dequantized layers)."""
    shift_kind, matrix_kind = BLOCKS[fmt]
    mbits = BITS[fmt][0]
    out = MAGIC + bytes([PRECISION[fmt]]) + struct.pack("<I", len(layers))
    deq = []
    for ts, m, cs in layers:
        tb, tq = enc_shift(shift_kind, ts)
        mb, mq = enc_matrix(matrix_kind, m, mbits)
        cb, cq = enc_shift(shift_kind, cs) if cs is not None else (b"", None)
        out += struct.pack("<III", len(tb), len(mb), len(cb)) + tb + mb + cb
        deq.append((tq, mq, cq))
    return out, deq


def decode_entity(buf):
    assert buf[:14] == MAGIC
    fmt = list(BLOCKS)[buf[14]]
    shift_kind, matrix_kind = BLOCKS[fmt]
    n = struct.unpack("<I", buf[15:19])[0]
    pos, layers = 19, []
    for _ in range(n):
        lt, lm, lc = struct.unpack("<III", buf[pos:pos + 12])
        pos += 12
        ts = dec_shift(shift_kind, buf[pos:pos + lt]); pos += lt
        m = dec_matrix(matrix_kind, buf[pos:pos + lm]); pos += lm
        cs = dec_shift(shift_kind, buf[pos:pos + lc]) if lc else None; pos += lc
        layers.append((ts, m, cs))
    assert pos == len(buf)
    return fmt, layers


def entity_shapes(arch):
    """{card, note}: [has channel mixer per layer] from expertium.variants."""
    if arch == "fork":   # the Big packages' architecture.json (read only)
        a = json.loads(BIG_ARCH.read_text(encoding="utf-8"))
        a = a["architecture"] if isinstance(a.get("architecture"), dict) else a
        st = {x["name"]: x for x in a["streams"]}
        assert a["d_model"] == 128 and all(st[s]["n_heads"] == 4 and st[s]["head_size"] == 32 for s in ("card_id", "note_id"))
        return {e: [k not in st[s]["channel_mixer_stripped_layers"] for k in range(st[s]["n_layers"])]
                for e, s in (("card", "card_id"), ("note", "note_id"))}
    from expertium.variants import variant
    v = variant(arch)
    mods = dict(v.base.modules)
    strip = set(v.strip_cmix)
    return {e: [f"{s}:{k}" not in strip for k in range(mods[s].n_layers)] for e, s in (("card", "card_id"), ("note", "note_id"))}


def random_layers(cmix, gen, kind):
    layers = []
    for has in cmix:
        scale = {"normal": 1.0, "tiny": 1e-6, "huge": 1e7}[kind]     # huge: row scales above the f16 maximum (clamp)
        vscale = min(scale, 1e4)                                       # shift vectors stay inside the f16 range
        ts = torch.randn(D, generator=gen) * vscale
        m = torch.randn(H, K, K, generator=gen) * scale
        if kind == "normal":
            m[0, 3] = 0.0           # an all-zero row: scale 0 -> 1
            m[2] *= 50.0            # one head much larger than the others (per-head scale)
        cs = torch.randn(D, generator=gen) * vscale if has else None
        layers.append((ts, m, cs))
    return layers


def scorer_q(fmt, layers):
    """The scorer's own rounding (int_row_quant) of each block, the reference of the round trip."""
    mbits, sbits = BITS[fmt]
    shift_q = (lambda v: v.half().float()) if not sbits else int_row_quant(sbits)
    rows = int_row_quant(mbits)
    return [(shift_q(ts), rows(m), shift_q(cs) if cs is not None else None) for ts, m, cs in layers]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(HERE / "sm131-bytes.json"))
    args = p.parse_args()
    gen = torch.Generator().manual_seed(0)
    archs = {"Small (w64)": "w64", "Tiny (w64-c1)": "w64-c1", "Big (fork)": "fork"}
    report, ok = {"layout": "fork ModuleState::compact (sm73s2 DESIGN-ADDENDUM) with the compact blocks", "arms": {}}, True
    for name, arch in archs.items():
        globals()["H"], globals()["K"], globals()["D"] = DIMS[arch]
        shapes = entity_shapes(arch)
        for arm, (cfmt, nfmt) in ARMS.items():
            row = {}
            for entity, fmt in (("card", cfmt), ("note", nfmt)):
                checks, restore = [], []
                for kind in ("normal", "tiny", "huge", "normal"):
                    layers = random_layers(shapes[entity], gen, kind)
                    buf, deq = encode_entity(fmt, layers)
                    got_fmt, dec = decode_entity(buf)
                    ref = scorer_q(fmt, layers)
                    same = got_fmt == fmt and all(
                        torch.equal(a, b) for L, R in zip(dec, ref) for a, b in zip(L, R) if a is not None or b is not None)
                    stable = all(torch.equal(a, b) for L, R in zip(dec, scorer_q(fmt, dec)) for a, b in zip(L, R)
                                 if a is not None)
                    checks.append(same)
                    restore.append(stable)
                    nbytes = len(buf)
                row[entity] = {"format": fmt, "layers": len(shapes[entity]), "channel_mixer": shapes[entity],
                               "bytes": nbytes, "round_trip_bitwise": all(checks),
                               "restore_of_a_read_state_exact": all(restore)}
                ok &= all(checks)
            row["card_plus_note"] = row["card"]["bytes"] + row["note"]["bytes"]
            report["arms"].setdefault(name, {})[arm] = row
    # the known fork totals (DESIGN-ADDENDUM): Small int8 4,554 and int4 3,018; Tiny int4 card + int8 note 2,622
    s, t = report["arms"]["Small (w64)"], report["arms"]["Tiny (w64-c1)"]
    known = {"small_int8_4554": s["w8+s8"]["card_plus_note"] == 4554,
             "small_card_int4_1963": s["w8+s4c8n"]["card"]["bytes"] == 1963,
             "tiny_fork_default_1055_1567": (t["w8+s4c8n"]["card"]["bytes"], t["w8+s4c8n"]["note"]["bytes"]) == (1055, 1567),
             "tiny_compact_803_1315": (t["w8+s4c8nhx8"]["card"]["bytes"], t["w8+s4c8nhx8"]["note"]["bytes"]) == (803, 1315),
             "tiny_cardonly_619": t["w8+s4hgx4c8hx8n"]["card"]["bytes"] == 619}
    b = report["arms"]["Big (fork)"]
    known.update({"big_fork_default_9271_10283": (b["w8+s4c8n"]["card"]["bytes"], b["w8+s4c8n"]["note"]["bytes"]) == (9271, 10283),
                  "big_int8_card_15415": b["w8+s8"]["card"]["bytes"] == 15415,
                  "big_layers_3_2_all_channel_mixers": (b["w8+s8"]["card"]["channel_mixer"], b["w8+s8"]["note"]["channel_mixer"])
                  == ([True] * 3, [True] * 2)})
    report["known_totals"] = known
    ok &= all(known.values())
    report["all_ok"] = bool(ok)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for name, arms in report["arms"].items():
        print(name + ": " + "; ".join(f"{a} card {r['card']['bytes']:,} / note {r['note']['bytes']:,} B" for a, r in arms.items()))
    print(f"[sm131-bytes] known totals {known} -> {'ALL OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
