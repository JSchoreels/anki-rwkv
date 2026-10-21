"""SM49 part 4 (CPU only): dump the stored int state codes of every card and note after the whole history of the
user's own collection, for one int state variant of SM36, into a PRIVATE directory outside git (per-entity data of the
user's own collection never leaves it; only collection_zstd.py's aggregate sizes are committed).

The collection is the read-only copy converted on 2026-10-14 (anki-rwkv artifacts/collection-eval/2026-10-14:
collection-dataset + collection-label-db; the live Anki profile is never touched), set up exactly as
wt/sm42/artifacts/small-model/collection/score_collection_small.py does, and scored by scripts/small_model_heldout.py
on the CPU (Hooks path). The stored state is recorded where the scorer stores it: the WKV state after each entity's
last non-skipped step (state_vq.Hooks.wkv) and the token-shift input vectors of that step (state_vq.Hooks.record_shift,
the vectors the next review would read). Codes and scales are recomputed from the float values with the scorer's
rounding and checked to reproduce the scorer's reconstruction exactly (torch.equal) at every step.

usage: collection_codes.py VARIANT OUT_DIR    (e.g. w8+s4hgx4; writes OUT_DIR/<variant>/*.bin + meta.json, and the
       scorer's aggregate row to OUT_DIR/<variant>/collection-row.jsonl)"""
import json
import sys
from pathlib import Path

WT = Path("C:/Users/jonat/workspace/wt/sm49")
COLL = Path(r"C:\Users\jonat\workspace\anki-rwkv\artifacts\collection-eval\2026-10-14")
CK = ("C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon/"
      "expertium/E89-F65e-F77a/master.pth")
TAG = "s-w64kd-gate-sc-lanes-dec12-1ep32k-muon"
sys.path.insert(0, str(WT))
variant, out_root = sys.argv[1], Path(sys.argv[2])
OUT = out_root / variant
assert not str(OUT.resolve()).lower().startswith(str(WT).lower().replace("/", "\\")), "dump must stay outside git"
OUT.mkdir(parents=True, exist_ok=True)

import lmdb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts import compress_common as cc  # noqa: E402
from scripts import small_model_heldout as sm  # noqa: E402
from scripts import state_vq  # noqa: E402
from scripts.small_model_wkv import F16_SCALE, HEAD_SCALE, round_scale_f16  # noqa: E402

# collection setup (as score_collection_small.py)
_run_info = cc.run_info
cc.run_info = lambda: dict(_run_info(), review_counts={"1": 223670}, dataset_path=str(COLL / "collection-dataset"))
cc.data_path = lambda run=None: COLL / "collection-dataset"
sm.heldout_users = lambda counts, pool=464: [1]
_open = lmdb.open
lmdb.open = lambda path, *a, **k: _open(str(COLL / "collection-label-db") if str(path).endswith("label-db") else path,
                                        *a, **k)

_, STATE = sm.parse_variant(variant)
assert isinstance(STATE, sm.IntState), variant


def per_stream(x):
    return dict(zip((0, 2), x)) if isinstance(x, tuple) else {0: x, 2: x}


WKV_BITS, SHIFT_BITS = per_stream(STATE.wkv_bits), per_stream(STATE.shift_bits)
assert all(b & F16_SCALE for b in (*WKV_BITS.values(), *SHIFT_BITS.values())), "f16 scales only"
SPLIT = {0: -1, 2: -1}
WKV, SHIFT = {}, {}   # (stream, split, layer) -> [codes, scales, seen]; (stream, split, layer, role) -> (codes, scales, has)
CHECKS = {"wkv_steps": 0, "shift_calls": 0}


def codes_of(x, bits, dims):
    qmax = 127.0 if (bits & 255) == 8 else 7.0
    scale = x.abs().amax(dims, keepdim=True) / qmax
    if bits & F16_SCALE:
        scale = round_scale_f16(scale)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.clamp(torch.round(x / scale), -qmax, qmax), scale


_install = state_vq.install


def install(model, streams=(0, 2)):
    layout = _install(model, streams)
    from expertium.model import rwkv_model as erm
    for stream in (0, 2):   # split counter: the first block's time mixer starts every split of the stream
        def hook(_module, _args, stream=stream):
            SPLIT[stream] += 1
        model.rwkv_modules[stream].blocks[0].time_mixer.register_forward_pre_hook(hook)
    ref = erm.reference_rwkv7

    def recording(r, k, v, w, a, kd, skip):
        cur = state_vq.Hooks.current
        if cur is not None and cur[2] == "t" and cur[:2] in state_vq.Hooks.wkv:
            B, T, H, K = r.shape
            ACTIVE["key"] = key = (cur[0], SPLIT[cur[0]], cur[1])
            WKV[key] = [torch.zeros(B, H, K, K, dtype=torch.int8), None, torch.zeros(B, dtype=torch.bool)]
        else:
            ACTIVE["key"] = None
        return ref(r, k, v, w, a, kd, skip)
    erm.reference_rwkv7 = recording
    return layout


ACTIVE = {"key": None}
state_vq.install = install
_configure = state_vq.configure


def configure(layout, codecs=None, int_bits=0, shift_bits=0):
    _configure(layout, codecs, int_bits, shift_bits)
    for key, fn in list(state_vq.Hooks.wkv.items()):
        bits = WKV_BITS[key[0]]
        dims = (-2, -1) if bits & HEAD_SCALE else -1

        def wrapped(s, cur, rows, t, fn=fn, bits=bits, dims=dims):
            out = fn(s, cur, rows, t)
            q, scale = codes_of(s, bits, dims)
            assert torch.equal(q * scale, out), (cur, t)
            buf = WKV[ACTIVE["key"]]
            if buf[1] is None:
                buf[1] = torch.zeros((buf[0].shape[0],) + scale.shape[1:], dtype=torch.float32)
            buf[0][rows], buf[1][rows], buf[2][rows] = q.to(torch.int8), scale, True
            CHECKS["wkv_steps"] += 1
            return out
        state_vq.Hooks.wkv[key] = wrapped
    state_vq.Hooks.record_shift = record_shift


def record_shift(vecs, cur):
    stream, layer, role = cur
    keep = ~state_vq.Hooks.skip.bool()
    counts = keep.sum(1)
    has = counts > 0
    final = vecs[(counts.cumsum(0) - 1)[has]].float()
    q, scale = codes_of(final, SHIFT_BITS[stream], -1)
    fn = state_vq.Hooks.shift[cur]
    assert torch.equal(q * scale, fn(final, cur)), cur
    SHIFT[(stream, SPLIT[stream], layer, role)] = (q.to(torch.int8), scale, has)
    CHECKS["shift_calls"] += 1


state_vq.configure = configure


def pack(codes, bits):
    """int8 codes -> bytes: int8 as is; int4 as (code + 8) nibbles, two per byte (first = low nibble)."""
    a = codes.numpy().astype(np.int16).reshape(-1)
    if bits == 8:
        return a.astype(np.int8).tobytes()
    u = (a + 8).astype(np.uint8)
    assert u.size % 2 == 0 and u.min() >= 1 and u.max() <= 15
    return (u[0::2] | (u[1::2] << 4)).tobytes()


def f16(scales):
    h = scales.half()
    assert torch.equal(h.float(), scales), "scales are float16 values"
    return h.numpy().tobytes()


sys.argv = [sys.argv[0], "--device", "cpu", "--model", f"{TAG}=expertium:w64:{CK}", "--only", f"{TAG}/{variant}",
            "--out", str(OUT / "collection-row.jsonl")]
if (OUT / "collection-row.jsonl").exists():
    (OUT / "collection-row.jsonl").unlink()
sm.main()

meta = {"variant": variant, "checks": CHECKS, "entities": {}, "fields": {}}
for stream, entity in ((0, "card"), (2, "note")):
    splits = sorted({k[1] for k in WKV if k[0] == stream})
    layers = sorted({k[2] for k in WKV if k[0] == stream})
    vecs = sorted({k[2:] for k in SHIFT if k[0] == stream})
    fields = {"wkv_codes": [], "wkv_scales": [], "shift_codes": [], "shift_scales": []}
    n = 0
    for split in splits:
        seen = WKV[(stream, split, layers[0])][2]
        for layer in layers:
            assert torch.equal(WKV[(stream, split, layer)][2], seen)
        for lr in vecs:
            assert torch.equal(SHIFT[(stream, split, *lr)][2], seen)
        n += int(seen.sum())
        fields["wkv_codes"].append(torch.stack([WKV[(stream, split, layer)][0][seen] for layer in layers], 1))
        fields["wkv_scales"].append(torch.stack([WKV[(stream, split, layer)][1][seen] for layer in layers], 1))
        fields["shift_codes"].append(torch.stack([SHIFT[(stream, split, *lr)][0] for lr in vecs], 1))
        fields["shift_scales"].append(torch.stack([SHIFT[(stream, split, *lr)][1] for lr in vecs], 1))
    fields = {k: torch.cat(v) for k, v in fields.items()}
    wb, sb = WKV_BITS[stream] & 255, SHIFT_BITS[stream] & 255
    blobs = {"wkv_codes": pack(fields["wkv_codes"], wb), "wkv_scales": f16(fields["wkv_scales"]),
             "shift_codes": pack(fields["shift_codes"], sb), "shift_scales": f16(fields["shift_scales"])}
    meta["entities"][entity] = {"count": n, "layers": len(layers), "shift_vectors": len(vecs), "wkv_bits": wb,
                                "shift_bits": sb, "head_scale": bool(WKV_BITS[stream] & HEAD_SCALE)}
    for name, blob in blobs.items():
        assert len(blob) % n == 0
        (OUT / f"{entity}_{name}.bin").write_bytes(blob)
        meta["fields"][f"{entity}_{name}"] = {"bytes": len(blob), "bytes_per_entity": len(blob) // n,
                                              "symbol": "f16" if "scales" in name else f"int{wb if 'wkv' in name else sb}"}
    meta["entities"][entity]["bits_per_entity"] = 8 * sum(len(b) // n for b in blobs.values())
(OUT / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
print(json.dumps(meta, indent=1))
