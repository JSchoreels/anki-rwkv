"""Reader for the two weight containers of this package. Needs only numpy.

weights.f32.bin   magic b"ARWKVWEIGHTS1"
weights.int8.bin  magic b"ARWKVWEIGHTS8PC1"
See INTEGRATION.md, section "File formats".
"""
import json
import struct
from pathlib import Path

import numpy as np

F32_MAGIC = b"ARWKVWEIGHTS1"
I8_MAGIC = b"ARWKVWEIGHTS8PC1"


def read_weights(path):
    """Either container -> {name: float32 array}. int8 tensors are dequantized as code * scale (float32)."""
    data = Path(path).read_bytes()
    pos = 0

    def take(n):
        nonlocal pos
        out = data[pos:pos + n]
        if len(out) != n:
            raise ValueError(f"{path}: truncated")
        pos += n
        return out

    if data.startswith(I8_MAGIC):
        take(len(I8_MAGIC))
        int8 = True
    elif data.startswith(F32_MAGIC):
        take(len(F32_MAGIC))
        int8 = False
    else:
        raise ValueError(f"{path}: unknown magic")
    count, = struct.unpack("<I", take(4))
    tensors = {}
    for _ in range(count):
        n, = struct.unpack("<H", take(2))
        name = take(n).decode("utf-8")
        kind = struct.unpack("<B", take(1))[0] if int8 else 0
        rank, = struct.unpack("<B", take(1))
        shape = struct.unpack(f"<{rank}I", take(4 * rank)) if rank else ()
        size = int(np.prod(shape)) if shape else 1
        if kind == 1:
            scale = np.frombuffer(take(4 * shape[0]), dtype="<f4")
            codes = np.frombuffer(take(size), dtype=np.int8).reshape(shape)
            tensors[name] = (codes.astype(np.float32) * scale[:, None]).astype(np.float32)
        else:
            tensors[name] = np.frombuffer(take(4 * size), dtype="<f4").reshape(shape).astype(np.float32)
    if pos != len(data):
        raise ValueError(f"{path}: {len(data) - pos} trailing bytes")
    return tensors


def load_package(package_dir, weights="fp32"):
    """(tensors, architecture dict) from a package folder. weights = 'fp32' or 'int8'."""
    package_dir = Path(package_dir)
    name = {"fp32": "weights.f32.bin", "int8": "weights.int8.bin"}[weights]
    spec = json.loads((package_dir / "architecture.json").read_text())
    return read_weights(package_dir / name), spec
