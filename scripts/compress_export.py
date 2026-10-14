"""Export the fork's 2.76M RWKV with quantized weights in the fork's weight-file container (ARWKVWEIGHTS2), plus a
reference reader/dequantizer and test vectors for the Rust side. Spec: docs/COMPRESSION_INTEGRATION.md.

ARWKVWEIGHTS2 (little-endian throughout; extends the fork's ARWKVWEIGHTS1 by one encoding byte per tensor):
  magic  b"ARWKVWEIGHTS2" (13 bytes)
  u32    tensor count
  per tensor, in state_dict order:
    u16 name length, UTF-8 name, u8 rank, rank x u32 dims (PyTorch shape, e.g. Linear weight = [out, in])
    u8  encoding: 0 = f32 values, 4 = int4 grouped symmetric, 8 = int8 grouped symmetric
    encoding 0: prod(dims) x f32, row-major (identical to ARWKVWEIGHTS1)
    encoding 4/8 (rank 2 only, dims [rows, cols]):
      u32 group (G, 1 <= G <= cols, along the last axis = Linear input axis; G = cols means per-row/per-channel;
          rows narrower than the scheme's group, e.g. 8 inputs under group 32, are stored with G = cols)
      rows x ceil(cols/G) x f32 scales, row-major (scale[r][j] covers columns j*G .. min(cols,(j+1)*G)-1)
      codes, row-major over all rows*cols elements: int8 = one i8 per element; int4 = two's-complement nibbles,
      element 2k in the low nibble and 2k+1 in the high nibble of byte k (the fork's int4 state packing),
      ceil(rows*cols/2) bytes. Valid codes: -127..127 (int8), -7..7 (int4).
  value = (code as f32) * scale    (one IEEE f32 multiply; this reproduces the evaluated model bit for bit)
  then end of file (no trailing bytes).
Quantization (compress_quant.fake_quant): scale = max|w| over the group / qmax (f32), qmax = 7 or 127, a zero group
gets scale 1.0, code = round_half_to_even(w / scale) clamped to [-qmax, qmax].

usage: compress_export.py --scheme w4-g32     (writes artifacts/compression/export/)
"""
import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts import compress_quant as cq  # noqa: E402

MAGIC_V1 = b"ARWKVWEIGHTS1"
MAGIC_V2 = b"ARWKVWEIGHTS2"
OUT = ROOT / "artifacts" / "compression" / "export"


def quantize_codes(weight, bits, group):
    """(codes int8 [rows, cols], scales f32 [rows, ngroups]) exactly as compress_quant.fake_quant."""
    qmax = 2 ** (bits - 1) - 1
    rows, cols = weight.shape
    g = cols if group is None else group
    groups = math.ceil(cols / g)
    pad = groups * g - cols
    w = torch.nn.functional.pad(weight, (0, pad)) if pad else weight
    w = w.reshape(rows, groups, g)
    scale = w.abs().amax(dim=-1, keepdim=True) / qmax
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    codes = q.reshape(rows, groups * g)[:, :cols].to(torch.int8)
    return codes, scale.reshape(rows, groups).float()


def dequantize(codes, scales, group):
    rows, cols = codes.shape
    expanded = scales.repeat_interleave(group, dim=1)[:, :cols]
    return codes.float() * expanded


def pack_int4(codes):
    flat = codes.flatten().numpy().astype(np.int16)
    if flat.size % 2:
        flat = np.append(flat, 0)
    nibbles = (flat & 0xF).astype(np.uint8)
    return (nibbles[0::2] | (nibbles[1::2] << 4)).tobytes()


def unpack_int4(data, count):
    raw = np.frombuffer(data, dtype=np.uint8)
    nibbles = np.empty(raw.size * 2, dtype=np.int16)
    nibbles[0::2] = raw & 0xF
    nibbles[1::2] = raw >> 4
    nibbles = np.where(nibbles >= 8, nibbles - 16, nibbles)
    return nibbles[:count].astype(np.int8)


def header(name, shape):
    encoded = name.encode("utf-8")
    return struct.pack("<H", len(encoded)) + encoded + struct.pack("<B", len(shape)) + struct.pack(
        f"<{len(shape)}I", *shape)


def write_v2(path, state, names, bits, group):
    out = bytearray(MAGIC_V2 + struct.pack("<I", len(state)))
    qnames = set(names)
    for name, tensor in state.items():
        tensor = tensor.float().contiguous()
        out += header(name, list(tensor.shape))
        if bits is not None and name in qnames:
            rows, cols = tensor.shape
            g = cols if group is None else min(group, cols)  # a group wider than the row = one group per row
            codes, scales = quantize_codes(tensor, bits, group)
            out += struct.pack("<BI", bits, g)
            out += scales.numpy().astype("<f4").tobytes()
            out += pack_int4(codes) if bits == 4 else codes.numpy().astype(np.int8).tobytes()
        else:
            out += struct.pack("<B", 0) + tensor.numpy().astype("<f4").tobytes()
    path.write_bytes(bytes(out))


def write_v1(path, state):
    out = bytearray(MAGIC_V1 + struct.pack("<I", len(state)))
    for name, tensor in state.items():
        tensor = tensor.float().contiguous()
        out += header(name, list(tensor.shape)) + tensor.numpy().astype("<f4").tobytes()
    path.write_bytes(bytes(out))


def read_weights(path):
    """Reference reader for ARWKVWEIGHTS1 and ARWKVWEIGHTS2 -> {name: float32 tensor} (dequantized)."""
    data = Path(path).read_bytes()
    if data.startswith(MAGIC_V2):
        v2, offset = True, len(MAGIC_V2)
    elif data.startswith(MAGIC_V1):
        v2, offset = False, len(MAGIC_V1)
    else:
        raise ValueError("unexpected file magic")

    def take(n):
        nonlocal offset
        chunk = data[offset:offset + n]
        if len(chunk) != n:
            raise ValueError("file ended early")
        offset += n
        return chunk

    (count,) = struct.unpack("<I", take(4))
    tensors = {}
    for _ in range(count):
        (name_len,) = struct.unpack("<H", take(2))
        name = take(name_len).decode("utf-8")
        (rank,) = struct.unpack("<B", take(1))
        shape = list(struct.unpack(f"<{rank}I", take(4 * rank)))
        numel = int(np.prod(shape)) if shape else 1
        encoding = struct.unpack("<B", take(1))[0] if v2 else 0
        if encoding == 0:
            values = np.frombuffer(take(4 * numel), dtype="<f4").copy()
            tensors[name] = torch.from_numpy(values).reshape(shape)
        elif encoding in (4, 8):
            if rank != 2:
                raise ValueError(f"{name}: quantized tensors must be rank 2")
            rows, cols = shape
            (g,) = struct.unpack("<I", take(4))
            if not 1 <= g <= cols:
                raise ValueError(f"{name}: invalid group {g}")
            ngroups = math.ceil(cols / g)
            scales = np.frombuffer(take(4 * rows * ngroups), dtype="<f4").reshape(rows, ngroups)
            if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
                raise ValueError(f"{name}: invalid scale")
            if encoding == 4:
                codes = unpack_int4(take((rows * cols + 1) // 2), rows * cols)
                qmax = 7
            else:
                codes = np.frombuffer(take(rows * cols), dtype=np.int8).copy()
                qmax = 127
            if np.any(np.abs(codes.astype(np.int16)) > qmax):
                raise ValueError(f"{name}: code out of range")
            codes = torch.from_numpy(codes.reshape(rows, cols))
            tensors[name] = dequantize(codes, torch.from_numpy(scales.copy()), g)
        else:
            raise ValueError(f"{name}: unknown encoding {encoding}")
    if offset != len(data):
        raise ValueError("trailing bytes in file")
    return tensors


def sha256_f32(tensor):
    return hashlib.sha256(tensor.float().contiguous().numpy().astype("<f4").tobytes()).hexdigest()


def prediction_vectors(fp32_state, quantized_state, user=4, reviews=256):
    """Python-reference predictions (CPU recurrence, float32) on a public dataset prefix: anki-revlogs-10k user 4,
    its first `reviews` rows as one history (full-history protocol, encoding seed 2025 + user). Values are the
    curve (ahead) and instant recall probabilities of every labelled review, keyed by review_th."""
    from scripts import compress_common as cc
    from scripts.full_run_metrics import evaluate_with_predictions
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare
    run = cc.run_info()
    frame = get_rwkv_data(cc.data_path(run), user).iloc[:reviews]
    batch = prepare(cc.make_samples(frame, user, len(frame)), seed=2025 + user)
    out = {"input": {"dataset": run["dataset_path"].split("/")[-1], "user_id": user, "first_reviews": reviews,
                     "protocol": "one sample of the whole prefix, prepare(seed=2025 + user)"}}
    for label, weights in (("fp32", fp32_state), ("quantized", quantized_state)):
        model = cq.load_model(state=weights)
        with torch.inference_mode():
            _, predictions = evaluate_with_predictions(model, [{"batch": batch}])
        ths = sorted(set(predictions["ahead"]) & set(predictions["instant"]))
        out[label] = {"review_th": ths, "p_curve": [round(float(predictions["ahead"][t]), 7) for t in ths],
                      "p_instant": [round(float(predictions["instant"][t]), 7) for t in ths]}
    a, b = np.array(out["fp32"]["p_curve"]), np.array(out["quantized"]["p_curve"])
    c, d = np.array(out["fp32"]["p_instant"]), np.array(out["quantized"]["p_instant"])
    out["max_abs_diff_quantized_vs_fp32"] = {"curve": float(np.abs(a - b).max()), "instant": float(np.abs(c - d).max())}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="w4-g32")
    parser.add_argument("--dequantized-v1", action="store_true",
                        help="also write the dequantized weights as an ARWKVWEIGHTS1 file (11 MB, for Rust parity tests)")
    parser.add_argument("--predictions", action="store_true", help="add Python-reference predictions (CPU, ~1 min)")
    args = parser.parse_args()
    bits, group = cq.WEIGHT_SCHEMES[args.scheme]
    state = {k: v.float() for k, v in torch.load(cq.FORK_WEIGHTS, map_location="cpu", weights_only=True).items()}
    names = cq.linear_weight_names(state)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"RWKV_trained_on_5000_10000_{args.scheme}.bin"
    write_v2(path, state, names, bits, group)
    reference = cq.quantize_state_dict(state, names, bits, group)
    loaded = read_weights(path)
    assert list(loaded) == list(reference), "tensor order/names differ"
    # torch.equal treats -0.0 == +0.0: fake quantization keeps -0.0 while a packed zero code decodes to +0.0, so
    # the values match but the bytes do not. Hashes and the f32 companion file therefore come from the decoded export.
    mismatched = [n for n in reference if not torch.equal(loaded[n], reference[n].float())]
    assert not mismatched, f"dequantized tensors differ from the evaluated model: {mismatched[:5]}"
    signed_zero_only = [n for n in reference if not torch.equal(loaded[n].view(torch.int32),
                                                                reference[n].float().contiguous().view(torch.int32))]
    if args.dequantized_v1:
        write_v1(OUT / f"RWKV_trained_on_5000_10000_{args.scheme}_dequantized_f32.bin", loaded)
    # Round trip of the original float32 container, as a check of the shared header layout.
    fork_bin = cq.FORK_WEIGHTS.with_suffix(".bin")
    v1 = read_weights(fork_bin)
    assert all(torch.equal(v1[n], state[n]) for n in state), "ARWKVWEIGHTS1 reader disagrees with the .pth"
    vectors = {
        "file": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scheme": args.scheme, "bits": bits, "group": group, "tensors": len(state), "quantized_tensors": len(names),
        "source_fp32_bin": {"file": fork_bin.name, "bytes": fork_bin.stat().st_size,
                            "sha256": hashlib.sha256(fork_bin.read_bytes()).hexdigest()},
        "dequantized_sha256": {n: sha256_f32(t) for n, t in loaded.items()},
        "dequantized_sha256_source": "decoded export (packed zero decodes to +0.0)",
        "signed_zero_only_tensors_vs_fake_quant": len(signed_zero_only),
        "examples": {},
    }
    for name in ("features2card.0.weight", "rwkv_modules.0.blocks.0.time_mixer.W_r.weight", "w_linear.weight"):
        if name not in names:
            continue
        rows, cols = state[name].shape
        g = cols if group is None else min(group, cols)
        codes, scales = quantize_codes(state[name], bits, group)
        deq = loaded[name]
        vectors["examples"][name] = {
            "shape": [rows, cols], "group": g,
            "row0_scales": [float(x) for x in scales[0]],
            "row0_scales_hex": [struct.pack("<f", float(x)).hex() for x in scales[0]],
            "row0_codes_first40": codes[0, :40].tolist(),
            "row0_dequantized_first8_hex": [struct.pack("<f", float(x)).hex() for x in deq[0, :8]],
            "last_row_last_group_codes": codes[-1, (math.ceil(cols / g) - 1) * g:].tolist(),
        }
    if args.predictions:
        vectors["predictions"] = prediction_vectors(state, loaded)
    (OUT / f"test-vectors-{args.scheme}.json").write_text(json.dumps(vectors, indent=1))
    print(json.dumps({k: vectors[k] for k in ("file", "bytes", "sha256", "tensors", "quantized_tensors")}))


if __name__ == "__main__":
    main()
