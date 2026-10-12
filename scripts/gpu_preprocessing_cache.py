"""F02: bounded deterministic samples/layout cache; never stores random encodings.

Entries contain JSON structure and aligned flat CPU tensor bytes. Private file
maps permit writable views without modifying the persisted cache. Misses rebuild;
cache construction, hashing and cold/warm reads remain explicit timing costs.
"""
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from uuid import uuid4

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1
SOURCES = ("rwkv/config.py", "rwkv/architecture.py", "rwkv/data_processing.py",
           "rwkv/prepare_batch.py", "rwkv/model/srs_model.py", "scripts/gpu_batch1.py",
           "scripts/gpu_optimizations.py", "scripts/gpu_preprocessing_cache.py",
           "scripts/gpu_preprocessing_stage.py")
TYPES = {str(dtype): dtype for dtype in (torch.bool, torch.uint8, torch.int8, torch.int16,
                                        torch.int32, torch.int64, torch.float16, torch.bfloat16,
                                        torch.float32, torch.float64)}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity():
    return {name: sha256_file(ROOT / name) for name in SOURCES}


def user_data_identity(dataset, user):
    """Hash actual table contents once per user when preparing the run manifest."""
    dataset = Path(dataset).resolve()
    result = {}
    for table in ("revlogs", "cards", "decks"):
        partition = dataset / table / f"user_id={user}"
        source = partition if partition.exists() else dataset / table
        paths = sorted(source.rglob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"Missing {table} data for user {user}")
        for path in paths:
            before = path.stat()
            digest = sha256_file(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError("Dataset changed during cache fingerprinting")
            result[path.relative_to(dataset).as_posix()] = digest
    return result


def cache_context(dataset_revision, data_files, sources, row, offset, end, chunk=128):
    if (not dataset_revision or not data_files or not sources or not 0 <= offset < end
            or end > row["train_range"][1] - row["train_range"][0] or chunk < 1):
        raise ValueError("Cache identity requires revision, source/data hashes and valid boundaries")
    return {"schema": SCHEMA, "dataset_revision": dataset_revision, "data_files": data_files,
            "sources": sources, "row": row, "offset": offset, "end": end, "chunk_reviews": chunk,
            "policy": {"split": "train", "cross_split_labels": "mask", "equalize_review_ths": [],
                       "sample_dtype": "torch.float32", "layout": "prepare_fast", "padding_factor": 0.9}}


def context_key(context):
    return hashlib.sha256(json.dumps(context, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def deterministic_layout(samples):
    from rwkv.model.srs_model import PreparedBatch
    from scripts.gpu_optimizations import prepare_fast
    # Only the CPU generator is preserved. No CUDA generator is accessed.
    with torch.random.fork_rng(devices=[]):
        deferred = prepare_fast(samples, seed=None, gpu_encodings=True)
    return PreparedBatch(**{field.name: getattr(deferred, field.name) for field in fields(PreparedBatch)})


def _encode(value, tensors):
    if isinstance(value, (torch.Tensor, np.ndarray)):
        array = isinstance(value, np.ndarray)
        tensor = torch.from_numpy(np.ascontiguousarray(value)) if array else value.detach()
        if tensor.device.type != "cpu" or str(tensor.dtype) not in TYPES:
            raise ValueError("Cache supports dense CPU tensors only")
        tensors.append(tensor.contiguous())
        return {"array" if array else "tensor": len(tensors) - 1}
    if is_dataclass(value):
        return {"class": type(value).__name__, "fields": {f.name: _encode(getattr(value, f.name), tensors) for f in fields(value)}}
    if isinstance(value, dict):
        return {"dict": {key: _encode(item, tensors) for key, item in value.items()}}
    if isinstance(value, (list, tuple)):
        return {"list": [_encode(item, tensors) for item in value]}
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    raise TypeError(f"Unsupported cache value {type(value)}")


def _decode(value, tensors):
    from rwkv.data_processing import ModuleData, RWKVSample
    from rwkv.model.srs_model import PreparedBatch
    if not isinstance(value, dict):
        return value
    if "tensor" in value:
        return tensors[value["tensor"]]
    if "array" in value:
        return tensors[value["array"]].numpy()
    if "list" in value:
        return [_decode(item, tensors) for item in value["list"]]
    if "dict" in value:
        return {key: _decode(item, tensors) for key, item in value["dict"].items()}
    classes = {cls.__name__: cls for cls in (ModuleData, RWKVSample, PreparedBatch)}
    return classes[value["class"]](**{key: _decode(item, tensors) for key, item in value["fields"].items()})


class PreprocessingCache:
    def __init__(self, directory, max_bytes=512 * 1024**2):
        self.directory = Path(directory).resolve()
        allowed = [ROOT / "artifacts"]
        controller = os.environ.get("GPU_R01_CONTROLLER_ROOT")
        if controller:
            controller = Path(controller).resolve()
            if ROOT.is_relative_to(controller / "artifacts"):
                allowed.append(controller / "artifacts")
        if not any(self.directory.is_relative_to(path) for path in allowed) or type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("Use a positive byte budget and a cache directory below workspace artifacts")
        self.max_bytes = max_bytes
        self.directory.mkdir(parents=True, exist_ok=True)

    def _owned(self, name):
        if not re.fullmatch(r"[a-f0-9]{64}(?:-[a-f0-9]{32}\.bin|\.json)", name):
            raise ValueError("Invalid cache filename")
        path = (self.directory / name).resolve()
        if path.parent != self.directory:
            raise ValueError("Cache file resolves outside its directory")
        return path

    @contextmanager
    def _lock(self):
        import psutil
        path = self.directory / ".cache.lock"
        deadline = time.monotonic() + 30
        while True:
            try:
                with path.open("x", encoding="utf-8") as stream:
                    json.dump({"pid": os.getpid()}, stream)
                break
            except (FileExistsError, PermissionError):
                try:
                    pid = json.loads(path.read_text(encoding="utf-8"))["pid"]
                    if not psutil.pid_exists(pid):
                        path.unlink(missing_ok=True)
                        continue
                except (ValueError, KeyError, FileNotFoundError, PermissionError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("CPU cache writer lock remained busy")
                time.sleep(0.02)
        try:
            yield
        finally:
            # A contender can briefly hold a Windows read handle on this file.
            for attempt in range(101):
                try:
                    path.unlink(missing_ok=True)
                    break
                except PermissionError:
                    if attempt == 100:
                        raise
                    time.sleep(0.01)

    def disk_bytes(self):
        return sum(path.stat().st_size for path in self.directory.iterdir()
                   if re.fullmatch(r"[a-f0-9]{64}(?:-[a-f0-9]{32}\.bin|\.json)", path.name))

    def _read(self, key, context):
        path = self._owned(key + ".json")
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            payload = metadata["payload"]
            encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
            if metadata["sha256"] != hashlib.sha256(encoded).hexdigest() or payload["context"] != context:
                return None
            if not payload["data_file"].startswith(key + "-"):
                return None
            data = self._owned(payload["data_file"])
            if data.stat().st_size != payload["bytes"] or sha256_file(data) != payload["data_sha256"]:
                return None
            mapping = torch.from_file(str(data), shared=False, size=payload["bytes"], dtype=torch.uint8)
            tensors = []
            for item in payload["tensors"]:
                shape, offset = item["shape"], item["offset"]
                if any(type(n) is not int or n < 0 for n in shape):
                    return None
                dtype = TYPES[item["dtype"]]
                size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
                if offset < 0 or offset + size > mapping.numel() or offset % 8:
                    return None
                tensors.append(mapping[offset:offset + size].view(dtype).view(shape))
            decoded = _decode(payload["tree"], tensors)
            os.utime(path, None)
            return decoded
        except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError):
            return None

    def _evict(self, needed):
        referenced = set()
        for path in self.directory.glob("*.json"):
            if not re.fullmatch(r"[a-f0-9]{64}\.json", path.name):
                continue
            try:
                name = json.loads(path.read_text(encoding="utf-8"))["payload"]["data_file"]
                if name.startswith(path.stem + "-"):
                    referenced.add(self._owned(name).name)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        for path in self.directory.glob("*.bin"):
            if path.name not in referenced:
                try:
                    self._owned(path.name).unlink(missing_ok=True)
                except (OSError, ValueError):
                    pass
        for path in sorted(self.directory.glob("*.json"), key=lambda item: item.stat().st_mtime_ns):
            if self.disk_bytes() + needed <= self.max_bytes:
                return True
            if not re.fullmatch(r"[a-f0-9]{64}\.json", path.name):
                continue
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                name = metadata["payload"]["data_file"]
                if not name.startswith(path.stem + "-"):
                    continue
                self._owned(name).unlink(missing_ok=True)
                self._owned(path.name).unlink(missing_ok=True)
            except (OSError, ValueError, KeyError, TypeError):
                # Windows can retain a mapping while a consumer uses an entry.
                continue
        return self.disk_bytes() + needed <= self.max_bytes

    def get_or_build(self, context, builder):
        started, key = time.perf_counter(), context_key(context)
        with self._lock():
            cached = self._read(key, context)
        if cached is not None:
            return cached, {"hit": True, "persisted": True, "read_seconds": time.perf_counter() - started,
                            "build_seconds": 0.0, "write_seconds": 0.0, "key": key}
        build_started = time.perf_counter()
        result = builder()
        build_seconds = time.perf_counter() - build_started
        write_started = time.perf_counter()
        tensors = []
        tree = _encode(result, tensors)
        descriptions, size = [], 0
        for tensor in tensors:
            size = (size + 7) // 8 * 8
            descriptions.append({"offset": size, "dtype": str(tensor.dtype), "shape": list(tensor.shape)})
            size += tensor.numel() * tensor.element_size()
        data_name = key + "-" + uuid4().hex + ".bin"
        payload = {"context": context, "tree": tree, "tensors": descriptions, "bytes": size,
                   "data_file": data_name, "data_sha256": "0" * 64}
        meta_size = len(json.dumps({"payload": payload, "sha256": "0" * 64}, sort_keys=True).encode("utf-8"))
        persisted = False
        with self._lock():
            cached = self._read(key, context)
            if cached is not None:
                result, persisted = cached, True
            elif self._evict(size + meta_size):
                data = self._owned(data_name)
                temporary = self.directory / (key + "-" + uuid4().hex + ".tmp")
                try:
                    digest, position = hashlib.sha256(), 0
                    with data.open("xb") as stream:
                        for tensor, description in zip(tensors, descriptions):
                            padding = bytes(description["offset"] - position)
                            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
                            stream.write(padding)
                            stream.write(raw)
                            digest.update(padding)
                            digest.update(raw)
                            position = description["offset"] + len(raw)
                    payload["data_sha256"] = digest.hexdigest()
                    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
                    metadata = {"payload": payload, "sha256": hashlib.sha256(encoded).hexdigest()}
                    temporary.write_bytes(json.dumps(metadata, sort_keys=True).encode("utf-8"))
                    temporary.replace(self._owned(key + ".json"))
                    persisted = True
                finally:
                    temporary.unlink(missing_ok=True)
                    if not persisted:
                        data.unlink(missing_ok=True)
        return result, {"hit": False, "persisted": persisted, "read_seconds": build_started - started,
                        "build_seconds": build_seconds, "write_seconds": time.perf_counter() - write_started, "key": key}


def plan_cache(rows, span, directory, dataset, revision, max_bytes=512 * 1024**2):
    """Create small descriptors; workers build misses lazily, so LRU eviction is safe."""
    started = time.perf_counter()
    cache = PreprocessingCache(directory, max_bytes)
    sources, paths = source_identity(), {}
    for row in rows:
        data_files = user_data_identity(dataset, row["user_id"])
        length = row["train_range"][1] - row["train_range"][0]
        for offset in range(0, length, span):
            end = min(length, offset + span)
            paths[row["user_id"], offset] = {"directory": str(cache.directory), "max_bytes": max_bytes,
                                            "dataset": str(Path(dataset).resolve()),
                                            "context": cache_context(revision, data_files, sources, row, offset, end)}
    return paths, {"identity_seconds": time.perf_counter() - started, "planned_batches": len(paths),
                   "directory": str(cache.directory),
                   "max_bytes": max_bytes, "disk_bytes_before": cache.disk_bytes(),
                   "scope": "Content fingerprints; lazy cache build/read costs remain inside batch preparation"}
