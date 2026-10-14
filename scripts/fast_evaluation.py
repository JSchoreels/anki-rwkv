"""Fast held-out evaluation of a checkpoint on a user list (EV01 in docs/EVAL_SPEED.md).

The full run's evaluation (`scripts/gpu_r01_training.run_case` with a `full_run` protocol) scores
each user separately: every 1,024-review section is prepared by the original Python `prepare`
with seed ``2025 + user + validation_start + offset`` and runs as its own eight-chunk FP32 forward,
followed by a per-review Python loop over predictions. The GPU forward is launch-bound at that size.

This path keeps every input value and every per-section reduction of the original:

* the same sections, seeds and per-sample features: F30 sample construction and F19 batched ID
  draws reproduce ``prepare(make_samples(...))`` bitwise (checked by ``scripts/check_fast_evaluation.py``);
* many sections (default 512 chunks, several users) share one FP32 forward. Every chunk is an
  independent sequence set, so batching changes only the hierarchy split padding and GEMM row
  counts. Whether that is bitwise is measured, not assumed (docs/EVAL_SPEED.md);
* per-section loss sums are recomputed on a contiguous copy of the original ``[chunks, T]`` block,
  in section order, so ``curve_log_loss`` etc. follow the original float reductions;
* benchmark metrics use the same label database and the same sklearn/pandas calls on arrays
  (``fast_stats``) with the original dictionary semantics (last labelled write wins).

The original path stays available as ``--mode reference`` (it calls the unchanged
``full_run_metrics`` functions) so the two can be compared on the same machine.

    python scripts/fast_evaluation.py --checkpoint <master.pth> --users-from <run.json>:quick_validation_users \
        --output artifacts/eval-speed/<name> [--mode fast|reference] [--workers 5] [--batch-chunks 512]
"""

from argparse import ArgumentParser
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import bisect
import contextlib
import importlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SECTION_REVIEWS = 1024
CHUNK_REVIEWS = 128
HELDOUT_SEED = 2025
SAMPLE_OPTIONS = {"fast_samples": True, "columnar_samples": True, "batch_segment_features": True, "direct_sample_matrix": True}
DEFAULT_DB_SIZE = 10_000_000_000


# ---------------------------------------------------------------------------------------------
# Rows and sections


def evaluation_rows(users, counts=None, data_path=None):
    """Full-run evaluation rows: the whole history of each user is held out ([0, n])."""
    rows = []
    for user in users:
        if counts is not None and str(user) in counts:
            n = int(counts[str(user)])
        else:
            from rwkv.data_processing import get_rwkv_data
            n = len(get_rwkv_data(Path(data_path), int(user)))
        rows.append({"user_id": int(user), "available_reviews": n, "train_range": [0, n], "validation_range": [0, n]})
    return rows


def section_parts(rows):
    """Every (row, offset, end, seed) section in the original evaluation order."""
    for row in rows:
        begin, end = row["validation_range"]
        for offset in range(0, end - begin, SECTION_REVIEWS):
            yield row, offset, min(offset + SECTION_REVIEWS, end - begin), HELDOUT_SEED + row["user_id"] + begin + offset


def pack_sections(rows, batch_chunks):
    """Group consecutive sections into jobs of at most `batch_chunks` chunks (whole sections only)."""
    job, chunks = [], 0
    for part in section_parts(rows):
        need = -(-(part[2] - part[1]) // CHUNK_REVIEWS)
        if job and chunks + need > batch_chunks:
            yield job
            job, chunks = [], 0
        job.append(part)
        chunks += need
    if job:
        yield job


# ---------------------------------------------------------------------------------------------
# Worker-side preparation (CPU, one thread per worker)

_FRAME = {}


def worker_init():
    import torch
    torch.set_num_threads(1)


def _data_root():
    return Path(os.environ["GPU_BATCH1_DATA_PATH"])


def _section_frame(row):
    from rwkv.data_processing import get_rwkv_data
    from scripts.gpu_batch1 import split_frame
    key = (row["user_id"], tuple(row["validation_range"]))
    if key not in _FRAME:
        _FRAME.clear()
        _FRAME[key] = split_frame(get_rwkv_data(_data_root(), row["user_id"]), row, "validation")
    return _FRAME[key]


def section_features(samples, seed):
    """Per-sample input rows exactly as `prepare(samples, seed=seed)` builds them (F19 draws, CPU encodings).

    Mirrors scripts.gpu_optimizations.prepare_fast(batched_id_encodings=True, gpu_encodings=False).
    """
    import numpy as np
    import torch
    from rwkv.config import DAY_OFFSET_ENCODE_PERIODS, ID_ENCODE_DIMS, RWKV_SUBMODULES
    from scripts.gpu_id_encodings import batched_id_table
    torch.random.default_generator.manual_seed(seed)
    features = []
    with torch.no_grad():
        for data in samples:
            card_features = data.card_features
            gather = [card_features]
            for submodule in RWKV_SUBMODULES:
                if submodule == "user_id":
                    continue
                table, indices = batched_id_table(data.ids[submodule], ID_ENCODE_DIMS[submodule], card_features.dtype)
                gather.append(table[indices])
            for period in DAY_OFFSET_ENCODE_PERIODS:
                baseline = torch.randint(low=0, high=period, size=(1,))
                f = 2 * np.pi / period
                angle = f * ((baseline + data.day_offsets) % period)
                first_angle = f * ((baseline + data.day_offsets_first) % period)
                gather.extend([torch.stack((angle.sin(), angle.cos()), -1).to(card_features.dtype),
                               torch.stack((first_angle.sin(), first_angle.cos()), -1).to(card_features.dtype)])
            features.append(torch.cat(gather, dim=-1))
    return features


def build_batch(samples, features, padding_factor=0.9, allowed_excess_in_one_step=20000):
    """One PreparedBatch over all samples, with prepare()'s split rules, layout and dtypes.

    For the samples of a single section this equals prepare(samples, seed) bitwise; for many
    sections the hierarchy splits are chosen over all samples together.
    """
    import numpy as np
    import torch
    from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
    from rwkv.model.srs_model import PreparedBatch
    from rwkv.prepare_batch import greedy_splits
    lengths = [int(data.card_features.size(0)) for data in samples]
    global_T = max(lengths)
    width = features[0].shape[1]
    start = torch.zeros((len(samples) * global_T, width), dtype=features[0].dtype)
    for index, rows in enumerate(features):
        start[index * global_T:index * global_T + rows.shape[0]] = rows
    current = [index * global_T + np.arange(length, dtype=np.int64) for index, length in enumerate(lengths)]
    skips = [data.skips.numpy() for data in samples]
    splits = greedy_splits(samples, factor=padding_factor, allowed_excess_in_one_step=allowed_excess_in_one_step)
    sub_gather, sub_skips, sub_shifts, sub_lens = [], [], [], []
    for name, _ in DEFAULT_ANKI_RWKV_CONFIG.modules:
        split = [int(r) for r in splits[name]]
        buckets = [[] for _ in split]
        perms = []
        for data_index, data in enumerate(samples):
            module = data.modules[name]
            perm = module.from_perm.numpy()
            perms.append(perm)
            offset = 0
            for split_B, split_len in zip(module.split_B.tolist(), module.split_len.tolist()):
                # prepare(): sample split (B, L) joins the split r with previous r' < L <= r.
                buckets[bisect.bisect_left(split, split_len)].append((data_index, offset, split_B, split_len))
                offset += split_B * split_len
            if offset != lengths[data_index]:
                raise ValueError("Module split sizes do not cover the sample")
        following = [np.zeros(length, dtype=np.int64) for length in lengths]
        gathers, skip_parts, shift_parts = [], [], []
        all_offset = 0
        for r, bucket in zip(split, buckets):
            takes, skip_list, shift_list = [], [], []
            for data_index, offset, split_B, split_len in bucket:
                source = perms[data_index][offset:offset + split_B * split_len]
                take = np.full((split_B, r), -1, dtype=np.int64)
                take[:, :split_len] = current[data_index][source].reshape(split_B, split_len)
                takes.append(take)
                skip_arr = skips[data_index][source].reshape(split_B, split_len)
                if not (~skip_arr[0]).any():
                    raise ValueError("Cannot skip the start of a sequence")
                skip = np.ones((split_B, r), dtype=bool)
                skip[:, :split_len] = skip_arr
                skip_list.append(skip)
                positions = np.broadcast_to(np.arange(split_len), skip_arr.shape)
                previous = np.maximum.accumulate(np.where(skip_arr, 0, positions), axis=1)
                shift = np.zeros((split_B, r), dtype=np.int64)
                shift[:, 1:split_len] = previous[:, :-1]
                shift_list.append(shift)
                destinations = all_offset + np.arange(split_B)[:, None] * r + np.arange(split_len)[None, :]
                following[data_index][source] = destinations.reshape(-1)
                all_offset += split_B * r
            gathers.append(torch.from_numpy(np.concatenate(takes).reshape(-1)))
            skip_parts.append(torch.from_numpy(np.concatenate(skip_list).reshape(-1)))
            shift_parts.append(torch.from_numpy(np.concatenate(shift_list).reshape(-1)))
        current = following
        sub_gather.append(gathers)
        sub_skips.append(skip_parts)
        sub_shifts.append(shift_parts)
        sub_lens.append(split)
    labels = torch.zeros((len(samples), global_T, samples[0].global_labels.shape[1]), dtype=samples[0].global_labels.dtype)
    review_th = torch.full((len(samples), global_T), -1, dtype=samples[0].label_review_ths.dtype)
    for index, data in enumerate(samples):
        labels[index, :lengths[index]] = data.global_labels
        review_th[index, :lengths[index]] = data.label_review_ths
    return PreparedBatch(num_data=len(samples), start=start, sub_gather=sub_gather, sub_gather_lens=sub_lens,
                         skips=sub_skips, time_shift_selects=sub_shifts, labels=labels, label_review_th=review_th)


def prepare_job(parts, pack=True):
    """Prepare one merged batch for consecutive sections; returns the batch and per-section metadata."""
    import numpy as np
    import torch
    from scripts.gpu_r01_samples import samples_for
    started = time.perf_counter()
    samples, features, sections = [], [], []
    for row, offset, end, seed in parts:
        section = _section_frame(row).iloc[offset:end]
        chunk = samples_for(section, row["user_id"], SAMPLE_OPTIONS)
        sections.append({"user_id": row["user_id"], "first": len(samples), "chunks": len(chunk),
                         "T": max(int(s.card_features.size(0)) for s in chunk), "reviews": end - offset})
        features.extend(section_features(chunk, seed))
        samples.extend(chunk)
    batch = build_batch(samples, features)
    labels = batch.labels
    has_label = labels[:, :, 4].int()
    is_query = labels[:, :, 6].int()
    for meta in sections:
        rows = slice(meta["first"], meta["first"] + meta["chunks"])
        meta["ahead_n"] = int(((1 - is_query[rows]) * has_label[rows]).sum())
        meta["imm_n"] = int((is_query[rows] * has_label[rows]).sum())
    # Labelled positions in the original collect_predictions order (row-major per section).
    flat = np.flatnonzero(has_label.reshape(-1).numpy().astype(bool))
    rating = torch.clamp(labels[:, :, 3] - 1, min=0).reshape(-1).float().numpy()[flat]
    query_flat = is_query.reshape(-1).numpy()[flat].astype(bool)
    imm_positions = flat[query_flat]
    for meta in sections:
        lo, hi = meta["first"] * labels.shape[1], (meta["first"] + meta["chunks"]) * labels.shape[1]
        meta["labelled"] = [int(np.searchsorted(flat, lo)), int(np.searchsorted(flat, hi))]
        meta["instant"] = [int(np.searchsorted(imm_positions, lo)), int(np.searchsorted(imm_positions, hi))]
    info = {"sections": sections, "labelled": flat.astype(np.int64), "instant_positions": imm_positions.astype(np.int64),
            "labelled_query": query_flat,
            "labelled_review_th": batch.label_review_th.reshape(-1).numpy()[flat].astype(np.int64),
            "labelled_rating": rating, "global_T": int(labels.shape[1]),
            "reviews": sum(end - offset for _, offset, end, _ in parts),
            "preparation_seconds": time.perf_counter() - started}
    if pack:
        batch = pack_aligned(batch)
    return batch, info


ALIGN = 512  # the CUDA caching allocator's granularity: device views are as aligned as fresh tensors


def pack_aligned(batch):
    """All batch tensors in one CPU byte storage at 512-byte offsets (one H2D copy per batch)."""
    import torch
    from scripts.gpu_batch1 import batch_tensors, map_batch
    tensors = batch_tensors(batch)
    offsets, size = [], 0
    for value in tensors:
        size = -(-size // ALIGN) * ALIGN
        offsets.append(size)
        size += value.numel() * value.element_size()
    storage = torch.empty(size, dtype=torch.uint8)
    views = []
    for value, offset in zip(tensors, offsets):
        target = storage[offset:offset + value.numel() * value.element_size()]
        target.copy_(value.contiguous().view(torch.uint8).flatten())
        views.append(target.view(value.dtype).view(value.shape))
    iterator = iter(views)
    return map_batch(batch, lambda _: next(iterator))


def to_device(batch, device):
    """One pinned, non-blocking copy of a pack_aligned batch; views rebuilt at the same offsets."""
    import torch
    from scripts.gpu_batch1 import batch_tensors, map_batch
    if torch.device(device).type != "cuda":
        return map_batch(batch, lambda x: x.to(device))
    first = batch_tensors(batch)[0]
    storage = first.untyped_storage()
    base = storage.data_ptr()
    whole = torch.empty(0, dtype=torch.uint8).set_(storage)
    if any(t.untyped_storage().data_ptr() != base for t in batch_tensors(batch)):
        return map_batch(batch, lambda x: x.to(device))
    pinned = torch.empty(whole.numel(), dtype=torch.uint8, pin_memory=True)
    pinned.copy_(whole)
    moved = pinned.to(device, non_blocking=True)

    def view(value):
        start = value.data_ptr() - base
        return moved[start:start + value.numel() * value.element_size()].view(value.dtype).view(value.shape)

    return map_batch(batch, view)


# ---------------------------------------------------------------------------------------------
# Benchmark metrics (same label database, same sklearn/pandas calls on arrays)


def benchmark_info(db_path, db_size, user_id):
    """rwkv.get_result.get_benchmark_info, read-only and closing the environment again.

    py-lmdb refuses a second open of the same environment in one process, so nothing is cached.
    """
    import lmdb
    from rwkv.utils import load_tensor
    env = lmdb.open(str(db_path), db_size, readonly=True, lock=False)
    try:
        with env.begin(write=False) as txn:
            if txn.get(f"{user_id}_review_ths".encode()) is not None:
                return (load_tensor(txn, f"{user_id}_review_ths", device="cpu").tolist(),
                        load_tensor(txn, f"{user_id}_rmse_bins", device="cpu").tolist())
    finally:
        env.close()
    return [], []


def fast_stats(review_ths, bins, preds, ratings):
    """rwkv.get_result.get_stats(...)[0] on aligned float64 arrays (unique review_ths)."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error
    gather_y = np.clip(ratings, a_min=0, a_max=1)
    gather_pred = preds
    rmse_raw = root_mean_squared_error(y_true=gather_y, y_pred=gather_pred)
    logloss = log_loss(y_true=gather_y, y_pred=gather_pred, labels=[0, 1])
    try:
        auc = round(roc_auc_score(y_true=gather_y, y_score=gather_pred), 6)
    except Exception:
        auc = None
    if auc is not None and np.isnan(auc):
        auc = None
    # get_stats lists rows bin by bin (first appearance), each in review order; groupby keeps
    # that within-bin order either way, so the per-bin means see the same sequence.
    tmp = pd.DataFrame({"bin": np.asarray(bins, dtype=np.int64), "y": gather_y, "p": gather_pred,
                        "weights": np.ones(len(bins), dtype=np.int64)})
    tmp = tmp.groupby("bin").agg({"y": "mean", "p": "mean", "weights": "sum"}).reset_index()
    rmse_bins = root_mean_squared_error(tmp["y"], tmp["p"], sample_weight=tmp["weights"])
    return {"metrics": {"RMSE": round(rmse_raw, 6), "LogLoss": round(logloss, 6),
                        "RMSE(bins)": round(rmse_bins, 6), "AUC": auc},
            "size": len(review_ths)}


def benchmark_from_arrays(user_id, db_path, db_size, keys, query, ratings, preds, exact_fallback=True):
    """scripts.full_run_metrics.benchmark_metrics for one user's labelled positions in original order."""
    import warnings
    import numpy as np
    review_ths, bins = benchmark_info(db_path, db_size, user_id)
    if not review_ths:
        return {"size": 0, "skipped": "no benchmark test reviews"}
    ths = np.asarray(review_ths, dtype=np.int64)
    if len(np.unique(ths)) != len(ths) and exact_fallback:
        return _benchmark_dicts(user_id, review_ths, bins, keys, query, ratings, preds)
    size = int(max(ths.max(initial=0), keys.max(initial=0))) + 1

    def last_write(mask):
        # Dictionary semantics of collect_predictions: the last labelled position for a key wins.
        k = keys[mask][::-1]
        unique, first = np.unique(k, return_index=True)
        present = np.zeros(size, dtype=bool)
        present[unique] = True
        return unique, len(k) - 1 - first, present

    all_mask = np.ones(len(keys), dtype=bool)
    rating_keys, rating_index, rating_present = last_write(all_mask)
    rating_value = np.zeros(size, dtype=np.float64)
    rating_value[rating_keys] = ratings[rating_index].astype(np.float64)
    result = {"size": len(review_ths)}
    for head, mask in (("ahead", ~query), ("instant", query)):
        head_keys, head_index, head_present = last_write(mask)
        value = np.zeros(size, dtype=np.float64)
        value[head_keys] = preds[mask][head_index].astype(np.float64)
        kept = head_present[ths] & rating_present[ths]
        result[f"{head}_coverage"] = int(kept.sum()) / len(review_ths)
        if not kept.any():
            continue
        kept_ths = ths[kept]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            stats = fast_stats(kept_ths, np.asarray(bins, dtype=np.int64)[kept], value[kept_ths], rating_value[kept_ths])
        result[head] = {**stats["metrics"], "size": stats["size"]}
    return result


def _benchmark_dicts(user_id, review_ths, bins, keys, query, ratings, preds):
    """Fallback: full_run_metrics.benchmark_metrics' dictionaries and get_stats, line for line."""
    import numpy as np
    from rwkv.get_result import get_stats
    predictions = {"ahead": {}, "instant": {}, "rating": {}}
    for key, is_query, rating, pred in zip(keys.tolist(), query.tolist(), ratings.tolist(), preds.tolist()):
        predictions["rating"][key] = float(rating)
        predictions["instant" if is_query else "ahead"][key] = float(pred)
    result = {"size": len(review_ths)}
    for head in ("ahead", "instant"):
        head_preds = predictions[head]
        kept = [i for i, th in enumerate(review_ths) if th in head_preds and th in predictions["rating"]]
        result[f"{head}_coverage"] = len(kept) / len(review_ths)
        if not kept:
            continue
        ths = [review_ths[i] for i in kept]
        bins_dict = {review_ths[i]: bins[i] for i in kept}
        with contextlib.redirect_stdout(io.StringIO()):
            stats, _ = get_stats(user_id, ths, bins_dict, {th: np.float64(head_preds[th]) for th in ths},
                                 {th: np.float64(predictions["rating"][th]) for th in ths})
        result[head] = {**stats["metrics"], "size": stats["size"]}
    return result


def metrics_job(user_id, db_path, db_size, keys, query, ratings, preds):
    return benchmark_from_arrays(user_id, Path(db_path), db_size, keys, query, ratings, preds)


# ---------------------------------------------------------------------------------------------
# Model


def load_model(checkpoint, model="expertium", config=None, device="cuda:0"):
    """FP32 model as the full run evaluates it (`master` after training): JIT, default kernels."""
    import torch
    from rwkv.model import nvrtc
    from scripts.compare import BASELINE_CONFIG, SMALL_CONFIG, BaselineModel, SmallModel
    if torch.device(device).type == "cuda":
        nvrtc.register()
        nvrtc.set_checkpoint_length(32)
    model_class, architecture = {"expertium": (SmallModel, SMALL_CONFIG), "baseline": (BaselineModel, BASELINE_CONFIG)}[model]
    if config:
        module, _, attr = config.partition(":")
        architecture = getattr(importlib.import_module(module), attr)
    master = model_class(architecture).float().to(device)
    state = checkpoint if isinstance(checkpoint, dict) else torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" in state and not any(k.startswith("features2card") for k in state):
        state = state["model"]
    master.load_state_dict(state)
    master.eval()
    return master


def runtime_setup(threads=4):
    """The evaluation-time global settings of run_full_training.run_epoch / run_case."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch._C._set_graph_executor_optimize(False)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.set_num_threads(threads)


def loss_terms(model, batch):
    """forward_batch plus the per-position values of _get_loss (same ops, no reductions)."""
    import torch
    import torch.nn.functional as F
    out_ahead_logits, out_w, out_w_log_p, out_p_logits = model.forward_batch(
        batch.start, batch.sub_gather, batch.sub_gather_lens, batch.time_shift_selects, batch.skips, batch.num_data)
    global_labels = batch.labels.float()
    label_elapsed_seconds, _, label_y, label_rating, has_label, label_is_equalize, is_query = global_labels.unbind(-1)
    has_label = has_label.int()
    is_query = is_query.int()
    label_rating = torch.clamp(label_rating - 1, min=0)
    label_elapsed_seconds = label_elapsed_seconds.unsqueeze(-1)
    curve_probs_raw = model.forgetting_curve(out_w, label_elapsed_seconds)
    curve_logits_raw = torch.log(curve_probs_raw / (1 - curve_probs_raw))
    ahead_logit_residual = model.interp(out_ahead_logits, label_elapsed_seconds)
    curve_logits = curve_logits_raw + ahead_logit_residual
    curve_probs = torch.sigmoid(curve_logits)
    out_p_probs = torch.softmax(out_p_logits, dim=-1)
    out_p_binary = torch.clamp(1.0 - out_p_probs.unbind(dim=-1)[0], min=1e-5, max=1.0 - 1e-5)
    ahead_mask = (1 - is_query) * has_label
    immediate_mask = is_query * has_label
    curve_loss = F.binary_cross_entropy_with_logits(curve_logits, label_y, reduction="none")
    curve_raw_loss = F.binary_cross_entropy_with_logits(curve_logits_raw, label_y, reduction="none")
    B, T = label_rating.shape
    p_loss = F.cross_entropy(out_p_logits.view(-1, 4), label_rating.long().view(-1), reduction="none").view(B, T)
    # full_run_metrics.evaluate_with_predictions: instant binary loss on the clamped p_imm.
    instant = F.binary_cross_entropy(out_p_binary.clamp(1e-5, 1 - 1e-5), batch.labels[:, :, 2], reduction="none")
    return {"nan": torch.isnan(out_ahead_logits).any() | torch.isnan(curve_probs).any(),
            "curve": curve_loss * ahead_mask, "curve_raw": curve_raw_loss * ahead_mask,
            "rating": p_loss * immediate_mask, "instant": instant, "instant_mask": (is_query.bool() & has_label.bool()),
            "p_curve": curve_probs, "p_imm": out_p_binary}


def section_sums(terms, sections, instant_positions):
    """Per-section sums as the per-section evaluation computes them, in fresh tensors of the original shape.

    The curve/raw/rating sums run over a contiguous [chunks, T] copy of the section's block; the
    instant loss over the masked 1-D values (row-major, as `losses[mask]` returns them), gathered
    with precomputed positions so no host synchronization is needed.
    """
    import torch
    sums = []
    instant = terms["instant"].reshape(-1)
    for meta in sections:
        rows, T = slice(meta["first"], meta["first"] + meta["chunks"]), meta["T"]
        for name in ("curve", "curve_raw", "rating"):
            sums.append(terms[name][rows, :T].clone(memory_format=torch.contiguous_format).sum())
        lo, hi = meta["instant"]
        sums.append(instant.index_select(0, instant_positions[lo:hi]).sum())
    return torch.stack(sums)


# ---------------------------------------------------------------------------------------------
# Evaluation drivers


class UserAccumulator:
    def __init__(self, row):
        self.row = row
        self.sections = 0
        self.totals = {"curve_labels": 0, "instant_labels": 0, "curve_log_loss": 0.0,
                       "curve_raw_log_loss": 0.0, "instant_log_loss": 0.0, "instant_rating_cross_entropy": 0.0}
        self.parts = []

    def add_section(self, meta, sums):
        import numpy as np
        ahead_n, imm_n = meta["ahead_n"], meta["imm_n"]
        # _get_loss: float32 sum / float32(1e-8 + count); evaluate_with_predictions: .item() * count.
        ahead_den = np.float32(1e-8) + np.float32(ahead_n)
        imm_den = np.float32(1e-8) + np.float32(imm_n)
        t = self.totals
        t["curve_labels"] += ahead_n
        t["instant_labels"] += imm_n
        t["curve_log_loss"] += float(np.float32(sums[0]) / ahead_den) * ahead_n
        t["curve_raw_log_loss"] += float(np.float32(sums[1]) / ahead_den) * ahead_n
        t["instant_rating_cross_entropy"] += float(np.float32(sums[2]) / imm_den) * imm_n
        t["instant_log_loss"] += float(sums[3])
        self.sections += 1

    def finish(self):
        t = dict(self.totals)
        assert t["curve_labels"] > 0 and t["instant_labels"] > 0
        for key in ["curve_log_loss", "curve_raw_log_loss"]:
            t[key] /= t["curve_labels"]
        for key in ["instant_log_loss", "instant_rating_cross_entropy"]:
            t[key] /= t["instant_labels"]
        return t


def _float32_den_check():
    import numpy as np
    import torch
    for n in (0, 1, 7, 1000, 2**24 + 1):
        a = (torch.tensor(1.0) * 0 + 3.25) / (1e-8 + torch.tensor(n))
        b = np.float32(3.25) / (np.float32(1e-8) + np.float32(n))
        if float(a) != float(b):
            raise AssertionError(f"float32 denominator differs for n={n}")


def evaluate_fast(model, rows, db_path, *, workers=5, batch_chunks=512, depth=None, db_size=DEFAULT_DB_SIZE,
                  progress=None, device="cuda:0", collect_predictions=False, input_hash=False):
    """Fast path: merged batches, exact per-section reductions, array benchmark metrics.

    input_hash=True also returns the original `validation_input_sha256` (sha256 over each section's
    start/labels/label_review_th exactly as the per-section path prepares them), hashed in a helper
    thread; it proves bitwise-identical inputs against a recorded run at the cost of host time.

    Two-stage pipeline: batch k's copy, forward and reductions are queued on the GPU (no host
    synchronization) before batch k-1's results are read back and booked per user.
    """
    import numpy as np
    import torch
    from scripts.trial_batches import preparation_pool
    _float32_den_check()
    model.eval()
    device = torch.device(device)
    cuda = device.type == "cuda"
    jobs = list(pack_sections(rows, batch_chunks))
    sections_left = {row["user_id"]: -(-(row["validation_range"][1] - row["validation_range"][0]) // SECTION_REVIEWS)
                     for row in rows}
    users = {row["user_id"]: UserAccumulator(row) for row in rows}
    order = [row["user_id"] for row in rows]
    results, pending_metrics, predictions = {}, {}, {}
    timing = {"preparation_seconds": 0.0, "wait_seconds": 0.0, "launch_seconds": 0.0, "readback_seconds": 0.0,
              "bookkeeping_seconds": 0.0, "metrics_wait_seconds": 0.0, "batches": 0, "chunks": 0,
              "padded_positions": 0, "sections": 0}
    depth = depth or workers + 2

    def launch(batch, info):
        with torch.no_grad():
            gpu = to_device(batch, device)
            terms = loss_terms(model, gpu)
            # Pageable host-to-device copies synchronize the stream first; stage through pinned memory.
            def h2d(array):
                if not cuda:
                    return torch.from_numpy(array)
                return torch.from_numpy(array).pin_memory().to(device, non_blocking=True)

            sums = section_sums(terms, info["sections"], h2d(info["instant_positions"]))
            labelled = h2d(info["labelled"])
            query = h2d(info["labelled_query"])
            preds = torch.where(query, terms["p_imm"].reshape(-1)[labelled], terms["p_curve"].reshape(-1)[labelled])
            values = torch.cat([sums.double(), preds.double(), terms["nan"].double().reshape(1)])
            host = torch.empty(values.shape, dtype=values.dtype, pin_memory=cuda)
            host.copy_(values, non_blocking=cuda)
            ready = torch.cuda.Event() if cuda else None
            if cuda:
                ready.record()
        timing["batches"] += 1
        timing["chunks"] += int(gpu.num_data)
        timing["padded_positions"] += int(gpu.num_data) * info["global_T"]
        timing["sections"] += len(info["sections"])
        # Keep the device inputs alive until the read-back event (the caching allocator would reuse them).
        return info, host, ready, (gpu, terms, values)

    def finish(info, host, ready, _keep):
        started = time.perf_counter()
        if ready is not None:
            ready.synchronize()
        timing["readback_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        host = host.numpy()
        if host[-1]:
            raise RuntimeError("Non-finite evaluation loss")
        count = 4 * len(info["sections"])
        sums_host = host[:count].astype(np.float32).reshape(-1, 4)
        preds_host = host[count:-1].astype(np.float32)
        for meta, values in zip(info["sections"], sums_host):
            user = meta["user_id"]
            users[user].add_section(meta, values)
            lo, hi = meta["labelled"]
            users[user].parts.append((info["labelled_review_th"][lo:hi], info["labelled_query"][lo:hi],
                                      info["labelled_rating"][lo:hi], preds_host[lo:hi]))
            sections_left[user] -= 1
            if sections_left[user] == 0:
                acc = users[user]
                keys, query_h, ratings, preds_u = (np.concatenate([p[i] for p in acc.parts]) for i in range(4))
                acc.parts = []
                if collect_predictions:
                    predictions[user] = (keys, query_h, ratings, preds_u)
                args = (user, str(db_path), db_size, keys, query_h, ratings, preds_u)
                pending_metrics[user] = submit(metrics_job, *args) if pool else metrics_job(*args)
                results[user] = acc.finish()
                if progress:
                    progress(len(results), len(rows))
        timing["bookkeeping_seconds"] += time.perf_counter() - started

    digest = None
    if input_hash:
        import hashlib
        from concurrent.futures import ThreadPoolExecutor
        digest = hashlib.sha256()
        hasher = ThreadPoolExecutor(max_workers=1)  # one thread: sections are hashed in submission order

        def hash_sections(batch, info):
            width = batch.start.shape[1]
            start = batch.start.view(batch.num_data, info["global_T"], width).numpy()
            labels, review_th = batch.labels.numpy(), batch.label_review_th.numpy()
            for meta in info["sections"]:
                rows, T = slice(meta["first"], meta["first"] + meta["chunks"]), meta["T"]
                digest.update(start[rows, :T].tobytes())
                digest.update(labels[rows, :T].tobytes())
                digest.update(review_th[rows, :T].tobytes())

    with preparation_pool(workers) as pool:
        submit = (lambda fn, *a: pool.submit(fn, *a)) if pool else None
        queue = deque()
        job_iter = iter(jobs)

        def refill():
            while len(queue) < (depth if pool else 1):
                job = next(job_iter, None)
                if job is None:
                    return
                queue.append(submit(prepare_job, job) if pool else job)

        refill()
        inflight = None
        while queue:
            waited = time.perf_counter()
            entry = queue.popleft()
            batch, info = entry.result() if pool else prepare_job(entry)
            timing["wait_seconds"] += time.perf_counter() - waited
            refill()
            timing["preparation_seconds"] += info["preparation_seconds"]
            started = time.perf_counter()
            launched = launch(batch, info)
            timing["launch_seconds"] += time.perf_counter() - started
            if digest is not None:
                hasher.submit(hash_sections, batch, info)
            if inflight is not None:
                finish(*inflight)
            inflight = launched
        if inflight is not None:
            finish(*inflight)
        if digest is not None:
            started = time.perf_counter()
            hasher.shutdown(wait=True)
            timing["input_hash_wait_seconds"] = time.perf_counter() - started
            timing["validation_input_sha256"] = digest.hexdigest()
        waited = time.perf_counter()
        per_user = []
        for user in order:
            metric = pending_metrics[user]
            per_user.append({"user_id": user, **results[user], "benchmark": metric.result() if pool else metric})
        timing["metrics_wait_seconds"] = time.perf_counter() - waited
    return per_user, timing, predictions


def evaluate_reference(model, rows, db_path, *, workers=5, db_size=DEFAULT_DB_SIZE, progress=None, device="cuda:0"):
    """The full run's original per-user evaluation (gpu_r01_training.run_case with a benchmark_db).

    Same calls in the same order; only host timers are added (get_loss already synchronizes on its
    NaN check, so its timer covers the GPU forward). The breakdown is the profile in docs/EVAL_SPEED.md.
    """
    import hashlib
    import torch
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare
    from scripts import full_run_metrics
    from scripts.full_run_metrics import benchmark_metrics, evaluate_with_predictions
    from scripts.gpu_batch1 import DATA, make_samples, split_frame
    from scripts.gpu_r01_training import evaluation_job
    from scripts.trial_batches import preparation_pool
    device = torch.device(device)
    validation_hash = hashlib.sha256()
    timing = {"frame_load_seconds": 0.0, "preparation_wait_seconds": 0.0, "input_hash_seconds": 0.0,
              "host_to_device_seconds": 0.0, "get_loss_seconds": 0.0, "collect_predictions_seconds": 0.0,
              "benchmark_metrics_seconds": 0.0, "sections": 0}

    def timed(name, fn, *args):
        started = time.perf_counter()
        try:
            return fn(*args)
        finally:
            timing[name] += time.perf_counter() - started

    class TimedModel:
        def eval(self):
            return model.eval()

        def get_loss(self, batch):
            timing["sections"] += 1
            return timed("get_loss_seconds", model.get_loss, batch)

    def heldout_batches(row, evaluation_pool):
        user = row["user_id"]
        section = timed("frame_load_seconds", lambda: split_frame(get_rwkv_data(DATA, user), row, "validation"))
        offsets = range(0, len(section), 1024)
        seeds = [2025 + user + row["validation_range"][0] + offset for offset in offsets]
        if evaluation_pool is None:
            batches = (timed("preparation_wait_seconds",
                             lambda o=offset, s=seed: prepare(make_samples(section.iloc[o:o + 1024], user, 128), seed=s))
                       for offset, seed in zip(offsets, seeds))
        else:
            futures = [evaluation_pool.submit(evaluation_job, section.iloc[offset:offset + 1024], user, seed)
                       for offset, seed in zip(offsets, seeds)]
            batches = (timed("preparation_wait_seconds", future.result) for future in futures)
        for batch in batches:
            started = time.perf_counter()
            for tensor in (batch.start, batch.labels, batch.label_review_th):
                validation_hash.update(tensor.numpy().tobytes())
            timing["input_hash_seconds"] += time.perf_counter() - started
            yield batch

    original_collect = full_run_metrics.collect_predictions
    full_run_metrics.collect_predictions = lambda stats, out: timed("collect_predictions_seconds", original_collect, stats, out)
    per_user = []
    try:
        with preparation_pool(workers) as evaluation_pool:
            for row in rows:
                values, predictions = evaluate_with_predictions(
                    TimedModel(), ({"batch": timed("host_to_device_seconds", batch.to, device)}
                                   for batch in heldout_batches(row, evaluation_pool)))
                values["benchmark"] = timed("benchmark_metrics_seconds", benchmark_metrics, row["user_id"],
                                            Path(db_path), db_size, predictions)
                per_user.append({"user_id": row["user_id"], **values})
                if progress:
                    progress(len(per_user), len(rows))
    finally:
        full_run_metrics.collect_predictions = original_collect
    return per_user, {"validation_input_sha256": validation_hash.hexdigest(), **timing}


def evaluate_checkpoint(checkpoint, users, output=None, *, model="expertium", config=None, mode="fast",
                        data_path=None, label_db=None, counts=None, workers=5, batch_chunks=512, threads=4,
                        quiet=False, extra=None, device="cuda:0", input_hash=False):
    """Evaluate `checkpoint` on `users`; returns (and writes to `output`) per-user metrics and aggregates."""
    data_path = Path(data_path or os.environ.get("GPU_BATCH1_DATA_PATH") or ROOT / "data/anki-revlogs-10k")
    os.environ["GPU_BATCH1_DATA_PATH"] = str(data_path)
    runtime_setup(threads)
    import torch
    from scripts.experiment_provenance import process_provenance
    from scripts.full_run_metrics import aggregate as benchmark_aggregate
    from scripts.run_gpu_batch1 import aggregate
    provenance = process_provenance()
    started = time.perf_counter()
    rows = evaluation_rows(users, counts, data_path)
    master = load_model(checkpoint, model, config, device)
    cuda = torch.device(device).type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    reviews = sum(row["validation_range"][1] - row["validation_range"][0] for row in rows)

    def progress(done, total):
        if not quiet:
            elapsed = time.perf_counter() - eval_started
            sys.stdout.write(f"\r[eval-{mode}] users {done}/{total} {elapsed:7.1f}s")
            sys.stdout.flush()

    sync()
    eval_started = time.perf_counter()
    if mode == "fast":
        per_user, timing, _ = evaluate_fast(master, rows, label_db, workers=workers, batch_chunks=batch_chunks,
                                            progress=progress, device=device, input_hash=input_hash)
    elif mode == "reference":
        per_user, timing = evaluate_reference(master, rows, label_db, workers=workers, progress=progress, device=device)
    else:
        raise ValueError(f"Unknown mode {mode}")
    sync()
    seconds = time.perf_counter() - eval_started
    if not quiet:
        sys.stdout.write("\n")
    labels = sum(row["curve_labels"] + row["instant_labels"] for row in per_user)
    report = {"status": "complete", "mode": mode, "checkpoint": str(checkpoint), "model": model, "config": config,
              "users": [row["user_id"] for row in rows], "user_count": len(rows), "data_path": str(data_path),
              "label_db": str(label_db), "workers": workers, "batch_chunks": batch_chunks if mode == "fast" else 8,
              "torch": str(torch.__version__), "device": str(device), "gpu": torch.cuda.get_device_name(0) if cuda else None,
              "evaluation_seconds": seconds, "total_seconds": time.perf_counter() - started,
              "validation_reviews": reviews, "scored_labels": labels,
              "reviews_per_second": reviews / seconds, "labels_per_second": labels / seconds,
              "timing": timing, "provenance": provenance, **(extra or {}),
              "evaluation": aggregate(per_user), "benchmark": benchmark_aggregate(per_user), "per_user": per_user}
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        tmp = output / "result.json.tmp"
        tmp.write_text(json.dumps(report, indent=1))
        tmp.replace(output / "result.json")
    return report


def parse_users(text):
    """`4,5,31`, `@file.json` (list), or `<run.json>:<key>` (e.g. quick_validation_users)."""
    if text.startswith("@"):
        return [int(u) for u in json.loads(Path(text[1:]).read_text())]
    path, sep, key = text.rpartition(":")
    if sep and path.endswith(".json") and Path(path).exists():
        value = json.loads(Path(path).read_text())
        keys = key.split("+")
        return sorted({int(u) for k in keys for u in value[k]}) if len(keys) > 1 else [int(u) for u in value[key]]
    return [int(u) for u in text.split(",") if u.strip()]


def main():
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True, help="state_dict (.pth) of the FP32 master")
    parser.add_argument("--users", required=True, help="4,5,31 | @users.json | <run.json>:quick_validation_users "
                                                        "(join keys with + for a sorted union)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="expertium", choices=["expertium", "baseline"])
    parser.add_argument("--config", help="optional module:ATTR AnkiRWKVConfig override")
    parser.add_argument("--mode", default="fast", choices=["fast", "reference"])
    parser.add_argument("--run-json", help="run.json for dataset_path and review_counts (defaults to the --users file)")
    parser.add_argument("--data-path")
    parser.add_argument("--label-db", help="benchmark label LMDB (defaults to label-db next to run.json)")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--batch-chunks", type=int, default=512)
    parser.add_argument("--limit", type=int, help="first N users only")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-hash", action="store_true", help="fast mode: also compute validation_input_sha256")
    parser.add_argument("--threads", type=int, default=4, help="main-process intra-op threads (run_case uses 4)")
    args = parser.parse_args()
    users = parse_users(args.users)[:args.limit]
    run_json = args.run_json or (args.users.rpartition(":")[0] if args.users.rpartition(":")[0].endswith(".json") else None)
    counts, data_path, label_db = None, args.data_path, args.label_db
    if run_json:
        run = json.loads(Path(run_json).read_text())
        counts = run.get("review_counts")
        base = Path(run_json).resolve().parent
        main_root = base.parents[2] if base.parent.name == "full-run" else ROOT
        data_path = data_path or str(main_root / run["dataset_path"])
        label_db = label_db or str(base / "label-db")
    if not label_db:
        parser.error("--label-db is required without --run-json")
    report = evaluate_checkpoint(args.checkpoint, users, args.output, model=args.model, config=args.config,
                                 mode=args.mode, data_path=data_path, label_db=label_db, counts=counts,
                                 workers=args.workers, batch_chunks=args.batch_chunks, device=args.device,
                                 threads=args.threads, input_hash=args.input_hash)
    print(json.dumps({key: report[key] for key in ("mode", "user_count", "evaluation_seconds", "validation_reviews",
                                                   "reviews_per_second", "evaluation")}, indent=1))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
