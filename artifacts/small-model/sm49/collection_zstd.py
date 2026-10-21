"""SM49 part 4: lossless compression of the stored int state codes of the own collection (dumped by collection_codes.py
into a private directory), aggregate sizes only -> collection-zstd.json / collection-zstd.md next to this file.
Needs Python >= 3.14 (stdlib compression.zstd, libzstd); no numpy.

Layouts (MB = 10^6 bytes; cards + notes):
  raw        the packed codes (int4 two per byte, int8, float16 scales) = the bit formula
  columnar   one stream per field (card/note x WKV codes, WKV scales, shift codes, shift scales), each zstd'd
  records    one stream of fixed-size entity records (card records, then note records), zstd'd as one file
  per-entity every entity record zstd'd on its own (a per-row database blob), without and with a dictionary trained on
             a 2,000-record sample of the same collection (optimistic: same data)
Order-0 entropy per field: codes per symbol (int4 nibble / int8 byte), scales per float16 value.
usage: collection_zstd.py DUMP_DIR VARIANT [VARIANT ...]"""
import json
import math
import random
import sys
import time
from collections import Counter
from compression import zstd
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEVELS = (1, 3, 9, 19, 22)
FIELDS = ("wkv_codes", "wkv_scales", "shift_codes", "shift_scales")


def entropy_bits(symbols):
    c, n = Counter(symbols), len(symbols)
    return -sum(k * math.log2(k / n) for k in c.values())


def symbols(blob, kind):
    if kind == "f16":
        return [blob[i] | blob[i + 1] << 8 for i in range(0, len(blob), 2)]
    if kind == "int4":
        return [x for b in blob for x in (b & 15, b >> 4)]
    return list(blob)


def z(data, level):
    return len(zstd.compress(data, options={zstd.CompressionParameter.compression_level: level}))


dump, variants = Path(sys.argv[1]), sys.argv[2:]
result = {}
for variant in variants:
    d = dump / variant
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    blobs = {name: (d / f"{name}.bin").read_bytes() for name in meta["fields"]}
    res = {"entities": {e: meta["entities"][e]["count"] for e in meta["entities"]},
           "bits_per_entity": {e: meta["entities"][e]["bits_per_entity"] for e in meta["entities"]}}
    raw = sum(len(b) for b in blobs.values())
    res["raw_mb"] = raw / 1e6
    t0 = time.perf_counter()
    res["columnar_mb"] = {lv: sum(z(b, lv) for b in blobs.values()) / 1e6 for lv in LEVELS}
    records = {}
    for e in meta["entities"]:
        n = meta["entities"][e]["count"]
        parts = [(blobs[f"{e}_{f}"], meta["fields"][f"{e}_{f}"]["bytes_per_entity"]) for f in FIELDS]
        records[e] = [b"".join(b[i * s:(i + 1) * s] for b, s in parts) for i in range(n)]
    stream = b"".join(r for e in records for r in records[e])
    assert len(stream) == raw
    res["records_mb"] = {lv: z(stream, lv) / 1e6 for lv in LEVELS}
    allrec = [r for e in records for r in records[e]]
    res["per_entity_mb"] = {lv: sum(z(r, lv) for r in allrec) / 1e6 for lv in (3, 19)}
    rng = random.Random(0)
    zd = zstd.train_dict(rng.sample(allrec, min(2000, len(allrec))), 64 * 1024)
    res["per_entity_dict64k_mb"] = {lv: (sum(len(zstd.compress(r, level=lv, zstd_dict=zd)) for r in allrec)
                                         + len(zd.dict_content)) / 1e6 for lv in (3, 19)}
    res["seconds"] = round(time.perf_counter() - t0, 1)
    res["entropy_order0"] = {}
    for name, b in blobs.items():
        kind = meta["fields"][name]["symbol"]
        sy = symbols(b, kind)
        h = entropy_bits(sy)
        res["entropy_order0"][name] = {"symbols": len(sy), "raw_bits_per_symbol": {"f16": 16, "int4": 4, "int8": 8}[kind],
                                       "entropy_bits_per_symbol": h / len(sy), "entropy_mb": h / 8 / 1e6,
                                       "raw_mb": len(b) / 1e6}
    res["entropy_order0_total_mb"] = sum(v["entropy_mb"] for v in res["entropy_order0"].values())
    result[variant] = res
    print(variant, json.dumps({k: v for k, v in res.items() if k != "entropy_order0"}), flush=True)

(HERE / "collection-zstd.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
lines = ["# SM49: lossless compression of the stored int state codes, own collection (aggregate only)", "",
         "SM36 states after the whole history of the 2026-10-14 collection copy (read-only); MB = 10^6 bytes, cards + "
         "notes; ratio = raw / compressed. zstd = libzstd 1.5.7 (Python 3.15 compression.zstd).", "",
         "| Variant | Entities (cards / notes) | Raw MB | zstd-3 columnar | zstd-19 columnar | zstd-22 columnar | "
         "zstd-19 records | per-entity zstd-19 | per-entity zstd-3 + 64 KB dict | order-0 entropy |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for v, r in result.items():
    raw = r["raw_mb"]
    cell = lambda mb: f"{mb:.2f} ({raw / mb:.2f}x)"  # noqa: E731
    lines.append(f"| {v} | {r['entities']['card']:,} / {r['entities']['note']:,} | {raw:.2f} | {cell(r['columnar_mb'][3])} | "
                 f"{cell(r['columnar_mb'][19])} | {cell(r['columnar_mb'][22])} | {cell(r['records_mb'][19])} | "
                 f"{cell(r['per_entity_mb'][19])} | {cell(r['per_entity_dict64k_mb'][3])} | "
                 f"{cell(r['entropy_order0_total_mb'])} |")
lines += ["", "Order-0 entropy per field (bits per symbol, raw -> entropy):", ""]
for v, r in result.items():
    lines.append(f"- {v}: " + "; ".join(f"{k} {e['raw_bits_per_symbol']} -> {e['entropy_bits_per_symbol']:.2f} "
                                        f"({e['raw_mb']:.2f} -> {e['entropy_mb']:.2f} MB)"
                                        for k, e in r["entropy_order0"].items()))
(HERE / "collection-zstd.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
