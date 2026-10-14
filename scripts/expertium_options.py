"""Expertium catch-up switches for the R01 trainer (docs/EXPERTIUM_CATCH_UP.md).

scripts/gpu_r01_training.run_case imports this module only when a track sets one of KEYS (or a full-run protocol
sets `full_run.lr_schedule`); without them the accepted path is untouched. One key per item:

    "architecture": "w80" | "w80-sequential" | "iter36" | {"base": ..., <VariantConfig field>: ...}   item 1, 4
    "lr_schedule": {"decay_start": 0.75, "shape": "one_minus_sin", ...}                               item 2
    "optimizer": "expertium-iter53" | {"kind": "muon", ...}                                            item 3
    "end_to_start_intervals": true                                                                     item 4
    "distillation": {"teacher": "<.pth>", "alpha": 0.9, "decay_alpha": 0.5}                            item 5
"""
from argparse import Namespace
import math
import os
from pathlib import Path

import torch

from expertium.intervals import end_to_start_published  # noqa: F401  (re-exported for the docs/tests)

KEYS = ("architecture", "lr_schedule", "optimizer", "end_to_start_intervals", "distillation")
PEAK_LR, WARMUP_UPDATES = 7e-4, 5
E2S_ENV = "EXPERTIUM_END_TO_START"
# Accepted R01 switches whose kernels assume d_model 32 / head size 32 (or need one that does). A variant of
# another width runs the generic PyTorch or NVRTC path for these; `compatible_options` drops them.
WIDTH32_ONLY = ("row_tiles_exact", "native_wkv", "wkv_tile", "wkv_rows", "exact_index", "skip_padding_sources",
                "native_index", "prefetch_rows", "fused_padded_gather", "batched_split_gather", "native_glue",
                "exact_bonus_products", "exact_gate_glue", "exact_channel_glue", "exact_group_norm",
                "native_group_norm", "recurrence_load_pipeline", "recurrence_prefetch", "wave_aware_backward",
                "compact_previous_state", "fused_shift", "explicit_shift_lerp", "partial_shift_lerp",
                "exact_shift_lerp", "concurrent_splits", "concurrent_split_limit")
# Wide-kernels port (docs/GPU_OPTIMIZATIONS.md W01-W04): exact kernels that also exist for head size 16 and rows
# wider than 32 (d=64/80), each bitwise equal to the generic path there (scripts/check_wide_kernels.py).
# prefetch_rows only affects rows of width <= 32, so it is a no-op on these models.
WIDE_PORTED = ("row_tiles_exact", "exact_index", "skip_padding_sources", "prefetch_rows", "exact_group_norm")
# Switches that rebuild or re-run the student loss outside get_loss.
NOT_WITH_VARIANTS = ("dynamic_graphs", "reduced_sync", "group_adam")


def requested(options, full_run=None):
    return any(key in options for key in KEYS) or bool((full_run or {}).get("lr_schedule"))


def variant_spec(options):
    return options.get("architecture")


def is_width32(spec):
    from expertium.variants import variant
    config = variant(spec).base
    return config.d_model == 32 and all(c.d_model // c.n_heads == 32 for _, c in config.modules)


def is_head16(spec):
    from expertium.variants import variant
    return all(c.d_model // c.n_heads == 16 for _, c in variant(spec).base.modules)


def width32_only(spec):
    """Switches this architecture cannot run: all width-32 kernels, minus the wide-kernels ports at head size 16."""
    if is_head16(spec):
        return tuple(key for key in WIDTH32_ONLY if key not in WIDE_PORTED)
    return WIDTH32_ONLY


def compatible_options(options, wide_kernels=False):
    """Track options with the width-32 kernels removed when the architecture is not width 32.

    wide_kernels=True keeps the switches ported to head size 16 (WIDE_PORTED); the default reproduces the
    generic cases measured before the port."""
    spec = variant_spec(options)
    if spec is None or is_width32(spec):
        return dict(options), []
    unsupported = width32_only(spec) if wide_kernels else WIDTH32_ONLY
    dropped = sorted(key for key in options if key in unsupported)
    return {key: value for key, value in options.items() if key not in unsupported}, dropped


def decay_factor(x, shape):
    """LR multiplier at decay progress x in [0, 1] (upstream RWKV_DECAY_SHAPE; default = his historical shape)."""
    x = min(max(x, 0.0), 1.0)
    if shape == "one_minus_sin":   # upstream default `1 + cos(pi/2 (1 + x))`, identically 1 - sin(pi x / 2)
        return 1.0 + math.cos(0.5 * math.pi * (1.0 + x))
    if shape == "cosine":          # standard half cosine
        return 0.5 * (1.0 + math.cos(math.pi * x))
    if shape == "linear":
        return 1.0 - x
    if shape == "sqrt":
        return 1.0 - math.sqrt(x)
    raise ValueError(f"Unknown decay shape {shape!r}")


def default_learning_rate(updates_done, update_offset):
    """The accepted R01 schedule, verbatim: 5-update linear warmup to 7e-4, then constant."""
    return 7e-4 * min(1.0, (updates_done + 1 + update_offset) / 5)


class LearningRate:
    """Warmup-stable-decay. Progress is measured in this case's training reviews, which are known before training
    (update counts depend on packing). Decay runs over [decay_start, decay_end] of them and ends at final * peak."""

    def __init__(self, schedule, total_reviews):
        unknown = set(schedule) - {"peak", "warmup_updates", "decay_start", "decay_end", "shape", "final"}
        if unknown:
            raise ValueError(f"Unknown lr_schedule keys {sorted(unknown)}")
        self.peak = float(schedule.get("peak", PEAK_LR))
        self.warmup = int(schedule.get("warmup_updates", WARMUP_UPDATES))
        self.start = float(schedule.get("decay_start", 1.0))
        self.end = float(schedule.get("decay_end", 1.0))
        self.shape = schedule.get("shape", "one_minus_sin")
        self.final = float(schedule.get("final", 0.0))
        if not (0.0 <= self.start <= self.end <= 1.0 and 0.0 <= self.final <= 1.0 and self.warmup >= 1):
            raise ValueError(f"Invalid lr_schedule {schedule}")
        decay_factor(0.0, self.shape)
        self.total = int(total_reviews)
        self.schedule = dict(schedule)

    def progress(self, reviews_done):
        if self.end == self.start:
            return 1.0 if reviews_done >= self.start * self.total else 0.0
        return ((reviews_done / self.total) - self.start) / (self.end - self.start)

    def decaying(self, reviews_done):
        return reviews_done >= self.start * self.total and self.start < 1.0

    def __call__(self, updates_done, update_offset, reviews_done):
        lr = self.peak * min(1.0, (updates_done + 1 + update_offset) / self.warmup)
        if self.start < 1.0:
            lr *= self.final + (1.0 - self.final) * decay_factor(self.progress(reviews_done), self.shape)
        return lr


class Teacher:
    """Item 5: a frozen teacher evaluated on the student's batch; get_loss(kd_mix=...) mixes its predictions in.

    Upstream iter 32 replayed stored teacher outputs by step; here the teacher runs online (no_grad, eval mode,
    dropout off) on the identical PreparedBatch. The default teacher architecture is the 2.76M srs-benchmark RWKV.
    """

    def __init__(self, spec, device, dtype):
        self.path = Path(spec["teacher"])
        self.alpha = float(spec.get("alpha", 0.9))
        self.decay_alpha = float(spec.get("decay_alpha", self.alpha))
        if not (0.0 <= self.alpha <= 1.0 and 0.0 <= self.decay_alpha <= 1.0):
            raise ValueError("distillation alphas must lie in [0, 1]")
        architecture = spec.get("teacher_architecture", "baseline")
        teacher_dtype = getattr(torch, spec.get("teacher_dtype", "bfloat16"))
        state = torch.load(self.path, map_location="cpu", weights_only=True)
        if architecture == "baseline":
            from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
            from rwkv.model.srs_model import SrsRWKV
            build = lambda: SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)  # noqa: E731
        else:
            from expertium.variants import build as build_variant
            build = lambda: build_variant(architecture)  # noqa: E731
        master = build().float()
        master.load_state_dict(state, strict=True)
        if teacher_dtype == torch.float32:
            model = master
        else:
            model = build().selective_cast(teacher_dtype)
            model.copy_downcast_(master, teacher_dtype)
        self.model = model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.input_dtype = teacher_dtype
        self.parameters = sum(p.numel() for p in master.parameters())
        self.sha256 = __import__("hashlib").sha256(self.path.read_bytes()).hexdigest()

    @torch.no_grad()
    def targets(self, batch, decaying=False):
        start = batch.start
        batch.start = start.to(self.input_dtype)
        try:
            stats = self.model.get_loss(batch)
        finally:
            batch.start = start
        if stats is None:
            raise RuntimeError("Teacher produced NaN outputs")
        alpha = self.decay_alpha if decaying else self.alpha
        return stats.p_curve.float().clamp(1e-5, 1 - 1e-5), stats.p_imm_all.float(), alpha

    def describe(self):
        return {"teacher": str(self.path), "teacher_sha256": self.sha256, "teacher_parameters": self.parameters,
                "alpha": self.alpha, "decay_alpha": self.decay_alpha}


def make_optimizer(spec, master):
    from scripts.muon_optimizer import expertium_iter53, get_hybrid_optimizer
    if spec == "expertium-iter53":
        return expertium_iter53(master, PEAK_LR)
    if isinstance(spec, dict) and spec.get("kind") == "muon":
        settings = {key: value for key, value in spec.items() if key != "kind"}
        return get_hybrid_optimizer(master, PEAK_LR, **settings)
    raise ValueError(f"Unknown optimizer option {spec!r}")


class CatchUp:
    """Per-case state for the catch-up switches; run_case calls it at five points (see module docstring)."""

    def __init__(self, options, full_run, rows):
        unknown_variant_combo = [key for key in NOT_WITH_VARIANTS if options.get(key)]
        if unknown_variant_combo and (options.get("architecture") is not None or options.get("distillation")):
            raise ValueError(f"Expertium architecture/distillation options do not support {unknown_variant_combo}")
        self.options = options
        self.spec = options.get("architecture")
        if self.spec is None and options.get("distillation"):
            # get_loss(kd_mix=...) lives on the variant class; "iter36" is the default model bitwise.
            self.spec = "iter36"
        if self.spec is not None:
            from expertium.variants import variant
            self.variant = variant(self.spec)
            if not is_width32(self.spec):
                bad = sorted(key for key in options if key in width32_only(self.spec) and options[key])
                if bad:
                    raise ValueError(f"Width-32 kernels {bad} with architecture {self.variant.name}; "
                                     "use scripts.expertium_options.compatible_options")
        schedule = (full_run or {}).get("lr_schedule") or options.get("lr_schedule")
        total = sum(row["train_range"][1] - row["train_range"][0] for row in rows)
        self.schedule = LearningRate(schedule, total) if schedule else None
        self.teacher = None
        if options.get("end_to_start_intervals"):
            # Read by rwkv.data_processing.get_rwkv_data in this process and in every worker spawned after this.
            os.environ[E2S_ENV] = "1"

    def models(self, config, device):
        from scripts.compare import training_models
        from expertium.model.variant_model import VariantSrsRWKV
        dtype = getattr(torch, config["DTYPE"])
        master, model = training_models(VariantSrsRWKV, self.variant, device, dtype, config["SEED"])
        return master, model, dtype

    def fresh_checkpoint(self, seed):
        from expertium.model.variant_model import VariantSrsRWKV
        torch.manual_seed(seed)
        master = VariantSrsRWKV(self.variant).float()
        return {"model": master.state_dict(), "summary": {"steps": 0, "parameters": sum(p.numel() for p in master.parameters())}}

    def optimizer(self, master):
        if "optimizer" not in self.options:
            from rwkv.train_rwkv import get_optimizer
            return get_optimizer(Namespace(PEAK_LR=PEAK_LR), master)
        return make_optimizer(self.options["optimizer"], master)

    def load_teacher(self, device, dtype):
        if self.options.get("distillation"):
            self.teacher = Teacher(self.options["distillation"], device, dtype)

    def learning_rate(self, updates_done, update_offset, reviews_done):
        if self.schedule is None:
            return default_learning_rate(updates_done, update_offset)
        return self.schedule(updates_done, update_offset, reviews_done)

    def loss(self, model, batch, reviews_done):
        if self.teacher is None:
            return model.get_loss(batch)
        decaying = self.schedule is not None and self.schedule.decaying(reviews_done)
        return model.get_loss(batch, kd_mix=self.teacher.targets(batch, decaying))

    def describe(self, master):
        out = {"keys": sorted(key for key in KEYS if key in self.options)}
        if self.spec is not None:
            out["architecture"] = {"name": self.variant.name, "parameters": sum(p.numel() for p in master.parameters())}
        if self.schedule is not None:
            out["lr_schedule"] = {**self.schedule.schedule, "total_reviews": self.schedule.total}
        if "optimizer" in self.options:
            out["optimizer"] = self.options["optimizer"]
        if self.options.get("end_to_start_intervals"):
            out["end_to_start_intervals"] = True
        if self.teacher is not None:
            out["distillation"] = self.teacher.describe()
        return out
