"""The Big size on the show-time big model (SM107): export a big-model (2,762,884-parameter fork architecture) PyTorch
state dict to the fork's ARWKVWEIGHTS1 f32 weight file (qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.bin).

This is the writer that produced the big model (1 leak-free epoch) (SM72 big) file on 2026-11-10: re-exporting the
fork's RWKV_trained_on_5000_10000.pth reproduces the shipped .bin byte for byte (checked again for SM107).

Format (little-endian): magic b"ARWKVWEIGHTS1", u32 tensor count, then per tensor in state_dict order:
u16 name length, UTF-8 name, u8 rank, rank x u32 dims (PyTorch shape), prod(dims) x f32 row-major.

usage (CPU): python sm107_export.py <state dict .pth> <out .bin>
"""
import hashlib
import struct
import sys
from pathlib import Path

import torch

MAGIC = b"ARWKVWEIGHTS1"


def write_v1(path, state):
    out = bytearray(MAGIC + struct.pack("<I", len(state)))
    for name, t in state.items():
        t = t.float().contiguous()
        n = name.encode()
        out += struct.pack("<H", len(n)) + n + struct.pack("<B", t.dim()) + struct.pack(f"<{t.dim()}I", *t.shape)
        out += t.numpy().astype("<f4").tobytes()
    Path(path).write_bytes(bytes(out))


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    sd = torch.load(src, map_location="cpu", weights_only=True)
    print(type(sd).__name__, len(sd), sum(v.numel() for v in sd.values()), {str(v.dtype) for v in sd.values()})
    write_v1(dst, sd)
    print(dst, Path(dst).stat().st_size, hashlib.sha256(Path(dst).read_bytes()).hexdigest())
