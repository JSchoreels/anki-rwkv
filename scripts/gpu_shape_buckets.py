"""F05 hierarchical shape buckets; separate from the active exact-shape suite.

Rows within each recurrent sequence retain their order. New padded positions are
skipped and have no labels. Dropout tensor shapes change: training parity and
broader quality cannot be inferred from dropout-free structural/gradient checks.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from rwkv.model.srs_model import PreparedBatch


def power_two(value):
    return 1 << max(0, (int(value) - 1).bit_length())


@dataclass(frozen=True)
class BucketPolicy:
    lengths: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    maximum_work_ratio: float = 1.5
    # None reproduces the original power-of-two sequence-count prototype.
    count_quantum: int | None = 8

    def __post_init__(self):
        if (not self.lengths or tuple(sorted(set(self.lengths))) != self.lengths
                or any(not isinstance(value, int) or value < 1 for value in self.lengths)):
            raise ValueError("Bucket lengths must be distinct positive increasing integers")
        if not math.isfinite(self.maximum_work_ratio) or self.maximum_work_ratio < 1:
            raise ValueError("Maximum padding/work ratio must be finite and at least one")
        if self.count_quantum is not None and (type(self.count_quantum) is not int or self.count_quantum < 1):
            raise ValueError("Count quantum must be a positive integer or None for powers of two")

    def capacity(self, count):
        if self.count_quantum is None:
            return power_two(max(1, count))
        # Empty bins need one skipped dummy sequence; occupied bins round to
        # the declared quantum. Never pad a whole hierarchy to the next power.
        return 1 if count == 0 else ((count + self.count_quantum - 1) // self.count_quantum) * self.count_quantum


@dataclass
class BucketResult:
    batch: PreparedBatch
    admitted: bool
    reason: str
    metrics: dict
    # Maps each original final output position to the padded output position.
    output_positions: torch.Tensor | None = None


def _work(batch):
    return {"start_positions": batch.start.shape[0],
            "recurrent_positions": sum(value.numel() for group in batch.sub_gather for value in group),
            "label_positions": batch.labels.shape[0] * batch.labels.shape[1]}


def bucket_batch(batch, policy=BucketPolicy()):
    """Construct on CPU before pinning/copying; reject excess work before allocation."""
    original_work = _work(batch)
    fallback = lambda reason, metrics=None: BucketResult(batch, False, reason, {"original": original_work, **(metrics or {})})
    tensors = [batch.start, batch.labels, batch.label_review_th,
               *(tensor for name in ("sub_gather", "time_shift_selects", "skips") for group in getattr(batch, name) for tensor in group)]
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("Bucket on CPU before transfer; do not introduce CUDA synchronization")
    groups = len(batch.sub_gather)
    if not groups or any(len(getattr(batch, field)) != groups for field in ("sub_gather_lens", "time_shift_selects", "skips")):
        raise ValueError("Prepared hierarchy fields have different stream counts")
    n, original_length = batch.labels.shape[:2]
    if n < 1 or original_length < 1 or n != batch.num_data or batch.start.shape[0] != n * original_length or batch.label_review_th.shape != (n, original_length):
        raise ValueError("Input/labels do not have the declared sample layout")
    if (len(batch.sub_gather[-1]) != 1 or batch.sub_gather_lens[-1] != [original_length]
            or batch.sub_gather[-1][0].numel() != n * original_length):
        return fallback("final user stream does not match the sample/label layout")
    if original_length > policy.lengths[-1]:
        return fallback("history exceeds the largest configured length bucket")
    target_n, target_length = policy.capacity(n), policy.lengths[-1]
    layouts, predicted_recurrent = [], 0
    for stream in range(groups):
        lens, gathers = batch.sub_gather_lens[stream], batch.sub_gather[stream]
        if len(lens) != len(gathers) or any(len(getattr(batch, field)[stream]) != len(lens) for field in ("time_shift_selects", "skips")):
            raise ValueError("Prepared hierarchy fields have different split counts")
        counts = {length: 0 for length in policy.lengths}
        source_groups = []
        old_offset = 0
        for index, (length, gather) in enumerate(zip(lens, gathers)):
            if length < 1 or gather.numel() % length:
                raise ValueError("Invalid recurrent sequence layout")
            if length > target_length:
                return fallback("recurrent split exceeds the largest length bucket")
            count = gather.numel() // length
            target = target_length if stream == groups - 1 else next(value for value in policy.lengths if value >= length)
            source_groups.append((index, length, count, target, counts[target], old_offset))
            counts[target] += count
            old_offset += gather.numel()
        active_lengths = (target_length,) if stream == groups - 1 else policy.lengths
        capacities = {length: (target_n if stream == groups - 1 else policy.capacity(counts[length])) for length in active_lengths}
        offsets, new_offset = {}, 0
        for length, count in capacities.items():
            offsets[length] = new_offset
            new_offset += count * length
        layouts.append((source_groups, capacities, offsets, old_offset))
        predicted_recurrent += new_offset
    predicted = {"start_positions": target_n * target_length, "recurrent_positions": predicted_recurrent,
                 "label_positions": target_n * target_length}
    ratios = {key: predicted[key] / original_work[key] for key in original_work}
    if max(ratios.values()) > policy.maximum_work_ratio:
        return fallback("configured padding/work limit exceeded", {"proposed": predicted, "proposed_ratios": ratios})

    start = F.pad(batch.start.reshape(n, original_length, -1), (0, 0, 0, target_length - original_length, 0, target_n - n)).flatten(0, 1)
    labels = F.pad(batch.labels, (0, 0, 0, target_length - original_length, 0, target_n - n))
    review_th = F.pad(batch.label_review_th, (0, target_length - original_length, 0, target_n - n), value=-1)
    old_start = torch.arange(n * original_length)
    previous_mapping = old_start.div(original_length, rounding_mode="floor") * target_length + old_start.remainder(original_length)
    previous_valid = torch.ones_like(old_start, dtype=torch.bool)
    gathers_out, lens_out, skips_out, shifts_out = [], [], [], []
    for stream, (sources, capacities, offsets, old_size) in enumerate(layouts):
        new_gathers = {length: torch.full((count, length), -1, dtype=torch.long) for length, count in capacities.items()}
        new_skips = {length: torch.ones((count, length), dtype=torch.bool) for length, count in capacities.items()}
        new_shifts = {length: torch.zeros((count, length), dtype=torch.long) for length, count in capacities.items()}
        mapping = torch.empty(old_size, dtype=torch.long)
        valid_positions = torch.empty(old_size, dtype=torch.bool)
        for index, length, count, target, first_row, old_offset in sources:
            gather = batch.sub_gather[stream][index].reshape(count, length)
            skip = batch.skips[stream][index].reshape(count, length)
            shift = batch.time_shift_selects[stream][index].reshape(count, length)
            if gather.dtype != torch.long or skip.dtype != torch.bool or shift.dtype != torch.long:
                raise ValueError("Unexpected gather/skip/time-shift dtype")
            if (shift < 0).any() or (shift >= length).any() or (gather >= previous_mapping.numel()).any():
                raise ValueError("Prepared gather or time-shift index is out of bounds")
            real = gather >= 0
            if (real & ~previous_valid[gather.clamp_min(0)]).any() or (real & (gather.gather(1, shift) < 0)).any():
                # Negative gathers are clamped to row zero by the original
                # model. Bucket reordering changes that unused padded row's
                # value, so reject a hierarchy that feeds padding to a real
                # position or time shift rather than assuming it is harmless.
                return fallback("a real recurrent position depends on padded input")
            remapped = previous_mapping[gather.clamp_min(0)].masked_fill(gather < 0, -1)
            new_gathers[target][first_row:first_row + count, :length] = remapped
            new_skips[target][first_row:first_row + count, :length] = skip
            new_shifts[target][first_row:first_row + count, :length] = shift
            positions = torch.arange(count * length)
            mapping[old_offset:old_offset + count * length] = (offsets[target] +
                (first_row + positions.div(length, rounding_mode="floor")) * target + positions.remainder(length))
            valid_positions[old_offset:old_offset + count * length] = real.flatten()
        gathers_out.append([value.flatten() for value in new_gathers.values()])
        skips_out.append([value.flatten() for value in new_skips.values()])
        shifts_out.append([value.flatten() for value in new_shifts.values()])
        lens_out.append(list(capacities))
        previous_mapping = mapping
        previous_valid = valid_positions
    result = PreparedBatch(target_n, start, gathers_out, lens_out, shifts_out, skips_out, labels, review_th)
    return BucketResult(result, True, "hierarchical length/count buckets", {"original": original_work, "bucketed": _work(result),
                        "work_ratios": ratios, "dropout": "shape/RNG consumption changes; training quality must be measured"}, previous_mapping)


class BucketStatistics:
    """Measured counters only; no predicted speed gains."""
    def __init__(self, policy=BucketPolicy()):
        self.policy = policy
        self.admitted = self.fallbacks = 0
        self.reasons = {}
        self.original_recurrent_positions = self.bucketed_recurrent_positions = 0

    def prepare(self, batch):
        result = bucket_batch(batch, self.policy)
        if result.admitted:
            self.admitted += 1
        else:
            self.fallbacks += 1
            self.reasons[result.reason] = self.reasons.get(result.reason, 0) + 1
        self.original_recurrent_positions += _work(batch)["recurrent_positions"]
        self.bucketed_recurrent_positions += _work(result.batch)["recurrent_positions"]
        return result

    def report(self):
        return {"admitted": self.admitted, "eager_fallbacks": self.fallbacks, "fallback_reasons": self.reasons,
                "original_recurrent_positions": self.original_recurrent_positions,
                "actual_recurrent_positions": self.bucketed_recurrent_positions,
                "policy": {"lengths": list(self.policy.lengths), "maximum_work_ratio": self.policy.maximum_work_ratio,
                           "count_quantum": self.policy.count_quantum}}


class BucketedGraphStage:
    """Opt-in integration for the original SrsRWKV graph-capable models.

    Prepare the CPU bucket before transfer, then call backward on the transferred
    bucket. The caller transfers returned model gradients to FP32 master weights
    and performs the ordinary optimizer step. Release the returned active model
    and statistics references after each optimizer step so cache eviction can
    release its graph pool. No running source is monkeypatched.
    """
    def __init__(self, base, factory, policy=BucketPolicy(), retain_on_miss=False):
        from scripts.gpu_batch1 import DynamicGraphs
        self.statistics = BucketStatistics(policy)
        self.graphs = DynamicGraphs(base, factory, limit=1, retain_on_miss=retain_on_miss)
        self.training_mode = None
        self.mode_changes = 0
        self.captured_reserved_mib = []

    def prepare_cpu(self, batch):
        return self.statistics.prepare(batch)

    def _release_graphs(self):
        if self.graphs.cache:
            torch.cuda.synchronize()
            self.graphs.cache.clear()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            self.graphs.metrics["evictions"] += 1

    def backward(self, master, batch, dtype, admitted, seed, training=True):
        from scripts.gpu_optimizations import GraphTrainingAdapter
        import time
        if not batch.start.is_cuda:
            raise ValueError("CUDA Graph stage needs an already transferred batch")
        if self.training_mode is not None and self.training_mode != training:
            # Dropout mode is baked into capture. Never replay an eval graph
            # after switching to training, even when tensor shapes are identical.
            self._release_graphs()
            self.graphs.seen.clear()
            self.mode_changes += 1
        self.training_mode = training
        if admitted:
            active = self.graphs.select(batch)
        else:
            # A policy rejection must remain eager and release any private pool
            # before the original larger history allocates its activations.
            self._release_graphs()
            self.graphs.metrics["eager_updates"] += 1
            active = self.graphs.base
        active.train(training)
        active.zero_grad(set_to_none=True)
        active.copy_downcast_(master, dtype)
        # Model construction and admission can consume RNG before this point.
        torch.manual_seed(seed)
        if isinstance(active, GraphTrainingAdapter):
            new_capture = active.graph is None
            started = time.perf_counter()
            stats = active.graph_update(batch, seed)
            if new_capture:
                torch.cuda.synchronize()
                self.graphs.metrics["capture_seconds"] += time.perf_counter() - started
                self.captured_reserved_mib.append(torch.cuda.memory_reserved(batch.start.device) / 2**20)
        else:
            stats = active.get_loss(batch)
            if stats is None or not torch.isfinite(stats.average_loss):
                raise RuntimeError("Non-finite eager bucket loss")
            stats.average_loss.backward()
        if stats is None or not torch.isfinite(stats.average_loss):
            raise RuntimeError("Non-finite bucket/graph loss")
        return active, stats

    def report(self):
        return {"buckets": self.statistics.report(), "graphs": dict(self.graphs.metrics),
                "training_mode_resets": self.mode_changes,
                "capture_total_reserved_mib": list(self.captured_reserved_mib),
                "memory_scope": "total Torch reserved memory observed after each capture, not isolated private-pool allocation"}
