"""Opt-in F01–F08 full-history experiments; bounded host and device staging."""

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import fields, replace
import ast
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import time

import torch
from rwkv.data_processing import create_sample, get_rwkv_data, ModuleData, RWKVSample
from rwkv.model.srs_model import PreparedBatch
from rwkv.prepare_batch import prepare
from scripts.gpu_optimizations import GraphTrainingAdapter, _CaptureLoss
from scripts.trial_batches import TrialBatch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / os.environ.get("GPU_BATCH1_DATA_PATH", "data/anki-revlogs-10k")
TRACKS = {
    "control": {},
    "F01": {"prefetch_users": True},
    "F02": {"sample_cache": True},
    "F03": {"worker_local": True, "packed_ipc": True},
    "F04": {"async_transfer": True},
    "F05": {"dynamic_graphs": True},
    "F06": {"reduced_sync": True},
    "F07": {"padding_factor": 0.3},
    "F08": {"fused_shift": True},
    "control-repeat": {},
}


def batch_tensors(batch):
    return [batch.start, batch.labels, batch.label_review_th,
            *(x for name in ("sub_gather", "time_shift_selects", "skips")
              for group in getattr(batch, name) for x in group)]


def map_batch(batch, fn):
    return replace(batch, start=fn(batch.start), labels=fn(batch.labels),
                   label_review_th=fn(batch.label_review_th),
                   **{name: [[fn(x) for x in group] for group in getattr(batch, name)]
                      for name in ("sub_gather", "time_shift_selects", "skips")})


def pack_batch(batch):
    """One shared byte storage, with aligned views for every returned tensor."""
    tensors = batch_tensors(batch)
    offsets, size = [], 0
    for tensor in tensors:
        size = (size + 7) // 8 * 8
        offsets.append(size)
        size += tensor.numel() * tensor.element_size()
    packed = torch.empty(size, dtype=torch.uint8)
    views = []
    for tensor, offset in zip(tensors, offsets):
        target = packed[offset:offset + tensor.numel() * tensor.element_size()]
        target.copy_(tensor.contiguous().view(torch.uint8).flatten())
        views.append(target.view(tensor.dtype).view(tensor.shape))
    iterator = iter(views)
    return map_batch(batch, lambda _: next(iterator))


def split_frame(frame, row, split):
    if len(frame) != row["available_reviews"]:
        raise ValueError(f"User {row['user_id']} history length changed from the recorded dataset")
    begin, end = row[f"{split}_range"]
    section = frame.iloc[begin:end].copy()
    section["has_label"] = (section["has_label"].astype(bool) &
                            section["label_review_th"].between(begin + 1, end)).astype(int)
    return section


def row_frame(row, orig=False):
    """The row's whole history frame: get_rwkv_data of the user, or, for a data-augmentation (SM67) pseudo row
    (row["augment"], scripts/augment.py), the source user's frame restricted to the row's half of the notes. The
    half's orig_review_th column is dropped (the sample builders take a fixed column set) unless orig=True."""
    if "augment" not in row:
        return get_rwkv_data(DATA, row["user_id"])
    from scripts.augment import row_keep
    frame = get_rwkv_data(DATA, row["augment"]["source_user"], keep=row_keep(DATA, row))
    return frame if orig else frame.drop(columns=["orig_review_th"])


def seed_user(row):
    """The user id in the item seed (training_seed + user + offset): a pseudo row keeps its real user's id codes,
    day baseline and dropout draws (SM67)."""
    return row["augment"]["source_user"] if "augment" in row else row["user_id"]


def frame_stream(rows, prefetch=False):
    """At most the current and next complete history are resident."""
    if not prefetch:
        for row in rows:
            yield row, row_frame(row)
        return
    with ThreadPoolExecutor(max_workers=1) as loader:
        pending = loader.submit(row_frame, rows[0])
        for i, row in enumerate(rows):
            frame = pending.result()
            if i + 1 < len(rows):
                pending = loader.submit(row_frame, rows[i + 1])
            yield row, frame


_WORKER_FRAMES = OrderedDict()


def sample_key(row, offset, end, chunk):
    identity = {"row": row, "offset": offset, "end": end, "chunk": chunk}
    for name in ("rwkv/data_processing.py", "rwkv/prepare_batch.py", "scripts/gpu_batch1.py"):
        identity[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    for table in ("revlogs", "cards", "decks"):
        partition = DATA / table / f"user_id={row['user_id']}"
        source = partition if partition.exists() else DATA / table
        identity[table] = [(str(p.relative_to(DATA)), p.stat().st_size, p.stat().st_mtime_ns)
                           for p in sorted(source.rglob("*.parquet"))]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def samples_pack(samples):
    return [{f.name: ({key: {mf.name: getattr(value, mf.name) for mf in fields(ModuleData)}
                       for key, value in sample.modules.items()} if f.name == "modules"
                     else getattr(sample, f.name)) for f in fields(RWKVSample)} for sample in samples]


def samples_unpack(items):
    return [RWKVSample(**{**item, "modules": {k: ModuleData(**v) for k, v in item["modules"].items()}})
            for item in items]


def make_samples(section, user, chunk):
    return [create_sample(user, section.iloc[i:i + chunk], [], torch.float32, torch.device("cpu"))
            for i in range(0, len(section), chunk)]


def prepare_job(row, offset, end, seed, options, section=None, cache_path=None):
    started = time.perf_counter()
    user = row["user_id"]
    if options.get("sample_cache"):
        samples = samples_unpack(torch.load(cache_path, weights_only=False, map_location="cpu"))
    else:
        if section is None:
            if user not in _WORKER_FRAMES:
                _WORKER_FRAMES.clear()
                _WORKER_FRAMES[user] = split_frame(row_frame(row), row, "train")
            section = _WORKER_FRAMES[user].iloc[offset:end]
        from scripts.gpu_r01_samples import samples_for
        samples = samples_for(section, user, options)
    batch = prepare(samples, seed=seed, padding_factor=options.get("padding_factor", 0.9),
                    allowed_excess_in_one_step=options.get("split_excess", 20000))
    if options.get("packed_ipc"):
        batch = pack_batch(batch)
    return TrialBatch(batch, offset, end - offset, len(samples), seed, time.perf_counter() - started)


def build_sample_cache(rows, span, destination):
    """Cache deterministic samples only; encodings are regenerated every update."""
    started = time.perf_counter()
    paths, hits, built = {}, 0, 0
    destination.mkdir(parents=True, exist_ok=True)
    for row, frame in frame_stream(rows):
        section = split_frame(frame, row, "train")
        for offset in range(0, len(section), span):
            end = min(len(section), offset + span)
            key = sample_key(row, offset, end, 128)
            path = destination / f"{key}.pt"
            paths[(row["user_id"], offset)] = str(path)
            if path.exists():
                hits += 1
                continue
            temporary = path.with_suffix(".tmp")
            torch.save(samples_pack(make_samples(section.iloc[offset:end], row["user_id"], 128)), temporary)
            temporary.replace(path)
            built += 1
        print(f"Cache user {row['user_id']}: {built} batches built, {hits} hits", flush=True)
    return paths, {"elapsed_seconds": time.perf_counter() - started, "built_batches": built,
                   "hit_batches": hits, "bytes": sum(Path(p).stat().st_size for p in paths.values()),
                   "scope": "deterministic sample cache creation/load; encodings regenerated during training"}


def training_batches(rows, span, seed, pool, options, cache_paths=None, depth=8):
    """Preserve per-user batch/update order. F01 queues through user boundaries."""
    def jobs():
        source = ((row, None) for row in rows) if (options.get("worker_local") or options.get("sample_cache")) else frame_stream(rows, options.get("prefetch_users", False))
        for row, frame in source:
            section = split_frame(frame, row, "train") if frame is not None else None
            length = row["train_range"][1] - row["train_range"][0]
            for offset in range(0, length, span):
                end = min(length, offset + span)
                args = (row, offset, end, seed + row["user_id"] + offset, options,
                        section.iloc[offset:end] if section is not None else None,
                        cache_paths.get((row["user_id"], offset)) if cache_paths else None)
                yield row, args

    def consume(iterator):
        pending = deque()
        for _ in range(depth):
            entry = next(iterator, None)
            if entry is None:
                break
            row, args = entry
            pending.append((row, pool.submit(prepare_job, *args)))
        try:
            while pending:
                row, future = pending.popleft()
                started = time.perf_counter()
                item = future.result()
                item.wait_seconds = time.perf_counter() - started
                entry = next(iterator, None)
                if entry is not None:
                    next_row, args = entry
                    pending.append((next_row, pool.submit(prepare_job, *args)))
                yield row, item
        finally:
            for _, future in pending:
                future.cancel()

    if options.get("prefetch_users"):
        yield from consume(iter(jobs()))
    else:
        # Baseline deliberately drains workers before loading the next user.
        for row in rows:
            selected = training_batches_single(row, span, seed, options, cache_paths)
            yield from consume(iter(selected))


def training_batches_single(row, span, seed, options, cache_paths):
    frame = None if (options.get("worker_local") or options.get("sample_cache")) else get_rwkv_data(DATA, row["user_id"])
    section = split_frame(frame, row, "train") if frame is not None else None
    length = row["train_range"][1] - row["train_range"][0]
    for offset in range(0, length, span):
        end = min(length, offset + span)
        yield row, (row, offset, end, seed + row["user_id"] + offset, options,
                    section.iloc[offset:end] if section is not None else None,
                    cache_paths.get((row["user_id"], offset)) if cache_paths else None)


def transfer_batches(source, dtype, asynchronous=False):
    device = torch.device("cuda:0")
    if not asynchronous:
        for row, item in source:
            batch = item.batch.to(device)
            batch.start = batch.start.to(dtype)
            yield row, item, batch
        return
    # Pin one upcoming batch in a host thread; no RNG or CUDA work in that thread.
    iterator = iter(source)
    stream = torch.cuda.Stream()
    with ThreadPoolExecutor(max_workers=1) as stage:
        def pin_next():
            entry = next(iterator, None)
            if entry is None:
                return None
            row, item = entry
            pinned = map_batch(item.batch, lambda x: x.pin_memory())
            return row, item, pinned

        def copy(entry):
            if entry is None:
                return None
            row, item, pinned = entry
            with torch.cuda.stream(stream):
                copy_started = torch.cuda.Event(enable_timing=True)
                copy_started.record(stream)
                batch = map_batch(pinned, lambda x: x.to(device, non_blocking=True))
                batch.start = batch.start.to(dtype)
                ready = torch.cuda.Event(enable_timing=True)
                ready.record(stream)
                item.copy_started, item.copy_ended = copy_started, ready
            return row, item, batch, ready, pinned

        current = copy(stage.submit(pin_next).result())
        pending = stage.submit(pin_next)
        while current is not None:
            row, item, batch, ready, pinned = current
            torch.cuda.current_stream().wait_event(ready)
            for tensor in batch_tensors(batch):
                tensor.record_stream(torch.cuda.current_stream())
            # Schedule the second buffer before the consumer launches training.
            following = copy(pending.result())
            pending = stage.submit(pin_next) if following is not None else None
            yield row, item, batch
            current = following


class ReducedSyncLoss:
    """Keep loss/gradient finite checks; avoid internal NaN and count transfers."""
    def __init__(self, model):
        self.model = model
        module = importlib.import_module(type(model).__module__)
        tree = ast.parse(inspect.getsource(module))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SrsRWKV")
        loss = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_get_loss")
        loss.decorator_list = []
        loss = _CaptureLoss().visit(loss)
        namespace = dict(module.__dict__)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[loss], type_ignores=[])), module.__file__, "exec"), namespace)
        self.loss = namespace["_get_loss"]

    def get_loss(self, batch):
        return self.loss(self.model, batch.start, batch.sub_gather, batch.sub_gather_lens,
                         batch.time_shift_selects, batch.skips, batch.num_data,
                         batch.labels, batch.label_review_th)


class DynamicGraphs:
    """Exact hierarchical shape cache, memory-bounded, second-use admission."""
    def __init__(self, base, factory, limit=1, retain_on_miss=False):
        self.base, self.factory, self.limit = base, factory, limit
        self.retain_on_miss = retain_on_miss
        self.seen, self.cache = {}, OrderedDict()
        self.metrics = {"hits": 0, "captures": 0, "eager_updates": 0,
                        "evictions": 0, "capture_seconds": 0.0}

    def select(self, batch):
        key = (batch.num_data, tuple(tuple(g) for g in batch.sub_gather_lens),
               tuple((tuple(x.shape), x.dtype) for x in batch_tensors(batch)))
        if key in self.cache:
            self.metrics["hits"] += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        if self.cache and not self.retain_on_miss:
            # A Default graph's private pool plus an eager full batch cannot fit
            # the fixed 0.72 allocator budget. Release it before an unmatched
            # shape, rather than relying on a live-allocation counter.
            torch.cuda.synchronize()
            self.cache.clear()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            self.metrics["evictions"] += 1
        count = self.seen.get(key, 0) + 1
        self.seen[key] = count
        if len(self.seen) > 4096:
            self.seen.pop(next(iter(self.seen)))
        if count < 2:
            self.metrics["eager_updates"] += 1
            return self.base
        if len(self.cache) >= self.limit:
            torch.cuda.synchronize()
            self.cache.popitem(last=False)
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            self.metrics["evictions"] += 1
        # Every graph owns fresh leaves; no graph invalidates another's parameters.
        adapter = GraphTrainingAdapter(self.factory())
        self.cache[key] = adapter
        self.metrics["captures"] += 1
        return adapter
