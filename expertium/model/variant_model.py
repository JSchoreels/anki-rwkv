"""Expertium 5k-era model switches as a subclass of the vendored iter36 model (docs/EXPERTIUM_CATCH_UP.md).

`VariantSrsRWKV(VariantConfig)` with every switch off builds the same parameters, in the same order and from
the same random draws, as `SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)` and computes the same loss (checked bitwise by
scripts/check_expertium_catch_up.py). The vendored class itself is unchanged, so default training never runs
this file. Ported from Expertium/rwkv-anki-autoresearch@a9fcf466 rwkv/model/srs_model.py and rwkv_model.py
(pins in expertium/upstream.json); upstream reads environment variables, this reads VariantConfig fields.
"""
from contextlib import contextmanager
import math
from typing import List, Optional, Tuple

import torch

from expertium.model import rwkv_model as _rwkv_model
from expertium.model.rwkv_model import RWKV7, RWKV7ChannelMixer, RWKV7Config, RWKV7Layer, RWKV7TimeMixer
from expertium.model.srs_model import (DTYPE_EXCLUDE, ExactIndexSelect, ExactPaddedIndexSelect, FusedPaddedIndexSelect,
                                       FunctionType, ModuleType, SrsRWKV, SrsRWKVIterStatistics, wddm_flush_point)

# The GRU head's root parameters stay float32 like the linears they replace (upstream DTYPE_EXCLUDE "gru_").
VARIANT_DTYPE_EXCLUDE = DTYPE_EXCLUDE + ["gru_"]


def is_excluded(name):
    return any(query in name for query in VARIANT_DTYPE_EXCLUDE)


def interleave_schedule(depths):
    """Front-loaded round-robin (upstream interleave_schedule, spread=False): sched[stream][round] = layer or -1."""
    rounds = max(depths)
    return [[j if j < depth else -1 for j in range(rounds)] for depth in depths]


def layer_steps(schedule):
    """Round-major (stream, layer) pairs of a schedule; the order the interleaved forward runs them."""
    return [(i, schedule[i][r]) for r in range(len(schedule[0])) for i in range(len(schedule)) if schedule[i][r] >= 0]


class StrippedChannelMixer(ModuleType):
    """Upstream RWKV_STRIP_CMIX keeps a d_model=1 dummy mixer (5 parameters); the layer never calls it."""

    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(1)
        self.lerp_k = torch.nn.Parameter(torch.zeros(1, 1, 1))
        self.W_k = torch.nn.Linear(1, 1, bias=False)
        self.W_v = torch.nn.Linear(1, 1, bias=False)

    @FunctionType
    def forward(self, in_BTC, time_shift_select_BT):
        return in_BTC


@contextmanager
def _stripped_value_lora():
    """Build a time mixer whose value-residual LoRA is upstream's 1x1 dummy (RWKV_STRIP_L0_VLORA).

    A scripted time mixer cannot swap a submodule after construction, so the vendored constructor's global
    LoraSimple is replaced for the duration of that one call.
    """
    original = _rwkv_model.LoraSimple

    def factory(name, d_model, d_lora, layer_id):
        if name == "v":
            return original(name=name, d_model=1, d_lora=1, layer_id=layer_id)
        return original(name=name, d_model=d_model, d_lora=d_lora, layer_id=layer_id)

    _rwkv_model.LoraSimple = factory
    try:
        yield
    finally:
        _rwkv_model.LoraSimple = original


class VariantRWKV7Layer(RWKV7Layer):
    def __init__(self, config: RWKV7Config, layer_id, strip_cmix=False, strip_v_lora=False):
        # RWKV7Layer.__init__ in the same order (same random draws when both switches are off).
        ModuleType.__init__(self)
        if strip_v_lora:
            # Layer 0 sets v0 and never calls its value-residual LoRA (RWKV_STRIP_L0_VLORA).
            assert layer_id == 0
            with _stripped_value_lora():
                self.time_mixer = RWKV7TimeMixer(config, layer_id)
        else:
            self.time_mixer = RWKV7TimeMixer(config, layer_id)
        self.cmix_stripped = bool(strip_cmix)
        self.channel_mixer = StrippedChannelMixer() if strip_cmix else RWKV7ChannelMixer(config, layer_id)
        self.dropout = torch.nn.Dropout(p=config.dropout_layer)

    @FunctionType
    def forward(self, in_BTC, v0_BTC, time_shift_select_BT, skip_BT):
        x_BTC, v0_BTC = self.time_mixer(in_BTC=in_BTC, v0_BTC=v0_BTC, time_shift_select_BT=time_shift_select_BT,
                                        skip_BT=skip_BT)
        if self.cmix_stripped:
            return x_BTC, v0_BTC
        return self.dropout(self.channel_mixer(x_BTC, time_shift_select_BT=time_shift_select_BT)), v0_BTC


class VariantRWKV7(RWKV7):
    def __init__(self, config: RWKV7Config, stream_name, strip_cmix=(), strip_l0_vlora=False):
        ModuleType.__init__(self)
        stripped = {int(entry.partition(":")[2]) for entry in strip_cmix if entry.partition(":")[0] == stream_name}
        self.blocks = torch.nn.ModuleList([
            VariantRWKV7Layer(config, layer_id, strip_cmix=layer_id in stripped,
                              strip_v_lora=strip_l0_vlora and layer_id == 0)
            for layer_id in range(config.layer_offset, config.layer_offset + config.n_layers)])


class VariantSrsRWKV(SrsRWKV):
    def __init__(self, variant):
        ModuleType.__init__(self)
        anki_rwkv_config = variant.base
        self.variant_name = variant.name
        # Same R01 switches as SrsRWKV.__init__; the trainer sets them per track.
        self.exact_index = False
        self.native_rows = False
        self.native_rows_handle = 0
        self.skip_padding_sources = False
        self.wddm_flush_streams = False
        self.batched_split_gather = False
        self.fused_padded_gather = False
        self.concurrent_split_limit = 0
        self.concurrent_splits = False
        self.sync_free_loss = False

        self.gru_n = int(variant.gru_head)
        self.gru_on = self.gru_n > 0
        self.no_ahead_residual = bool(variant.no_ahead_residual or self.gru_on)
        self.zero_features_on = len(variant.zero_features) > 0
        self.zero_features: List[int] = [int(i) for i in variant.zero_features] or [-1]
        self.interleave_on = bool(variant.interleave)
        self.stream_depths: List[int] = [config.n_layers for _, config in anki_rwkv_config.modules]
        # (stream, layer) execution order, round-major; check_expertium_catch_up.py swaps in stream-major order to
        # prove the canonical gather/scatter path reproduces the sequential forward.
        self.layer_steps: List[Tuple[int, int]] = layer_steps(interleave_schedule(self.stream_depths))

        self.card_features_dim = 92
        self.d_model = anki_rwkv_config.d_model
        self.features_fc_dim = anki_rwkv_config.features_fc_mult * self.d_model
        self.ahead_head_dim = anki_rwkv_config.head_fc_mult * self.d_model
        self.p_head_dim = anki_rwkv_config.head_fc_mult * self.d_model
        self.w_head_dim = anki_rwkv_config.head_fc_mult * self.d_model
        self.num_curves = self.gru_n if self.gru_on else anki_rwkv_config.num_curves

        with torch.no_grad():
            self.features2card = torch.nn.Sequential(
                torch.nn.Linear(self.card_features_dim, self.features_fc_dim),
                torch.nn.SiLU(),
                torch.nn.LayerNorm(self.features_fc_dim),
                torch.nn.Linear(self.features_fc_dim, self.d_model),
                torch.nn.SiLU(),
            )
            self.rwkv_modules = torch.nn.ModuleList([
                VariantRWKV7(config, name, variant.strip_cmix, variant.strip_l0_vlora)
                for name, config in anki_rwkv_config.modules])
            self.prehead_norm = torch.nn.LayerNorm(self.d_model)
            self.prehead_dropout = torch.nn.Dropout(p=anki_rwkv_config.dropout)
            if self.gru_on:
                # 1x1 dummies keep the scripted dead branches compilable (upstream counts these 6 parameters).
                self.head_ahead_logits = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU())
            else:
                self.head_ahead_logits = torch.nn.Sequential(
                    torch.nn.Linear(self.d_model, self.ahead_head_dim),
                    torch.nn.ReLU(),
                )
            self.head_w = torch.nn.Sequential(
                torch.nn.Linear(self.d_model, 1 * self.d_model),
                torch.nn.ReLU(),
                torch.nn.LayerNorm(1 * self.d_model),
                torch.nn.Dropout(p=0.1),
                torch.nn.Linear(1 * self.d_model, self.w_head_dim),
            )
            self.head_p = torch.nn.Sequential(
                torch.nn.Linear(self.d_model, self.p_head_dim),
                torch.nn.ReLU(),
            )

            self.max_e = 21
            self.point_spread = 18.5
            self.num_points = anki_rwkv_config.num_points
            if self.gru_on:
                self.ahead_linear = torch.nn.Linear(1, 1)
                self.w_linear = torch.nn.Linear(1, 1)
                # Upstream init: zero weights; biases w uniform, S log-spaced 1 hour .. 1 year, d = 0.5.
                n = self.gru_n
                self.gru_w_weight = torch.nn.Parameter(torch.zeros(n, self.w_head_dim))
                self.gru_w_bias = torch.nn.Parameter(torch.zeros(n))
                self.gru_s_weight = torch.nn.Parameter(torch.zeros(n, self.w_head_dim))
                self.gru_s_bias = torch.nn.Parameter(torch.linspace(math.log(3600.0), math.log(31536000.0), n))
                self.gru_d_weight = torch.nn.Parameter(torch.zeros(n, self.w_head_dim))
                self.gru_d_bias = torch.nn.Parameter(torch.full((n,), math.log(0.5)))
            else:
                self.ahead_linear = torch.nn.Linear(self.ahead_head_dim, self.num_points)
                torch.nn.init.zeros_(self.ahead_linear.weight)
                torch.nn.init.zeros_(self.ahead_linear.bias)

                self.w_linear = torch.nn.Linear(self.w_head_dim, self.num_curves)
                torch.nn.init.zeros_(self.w_linear.weight)
                torch.nn.init.zeros_(self.w_linear.bias)

            self.s_point_spread = 18.5
            self.s_max = 22

            self.p_linear = torch.nn.Linear(self.p_head_dim, 4)
            torch.nn.init.zeros_(self.p_linear.weight)
            self.p_linear.bias.copy_(torch.tensor([-0.3512, -0.0802, 0.4297, -0.2041]))

    # ---- GRU curve head (RWKV_GRU_HEAD) ----
    @torch.jit.ignore
    def _gru_heads(self, x_w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = torch.nn.functional.linear(x_w, self.gru_w_weight, self.gru_w_bias)
        s = torch.nn.functional.linear(x_w, self.gru_s_weight, self.gru_s_bias)
        d = torch.nn.functional.linear(x_w, self.gru_d_weight, self.gru_d_bias)
        return w, s, d

    @FunctionType
    def gru_forgetting_curve(self, w, s_raw, d_raw, label_elapsed_seconds):
        # R(t) = sum_i w_i (1 + t / S_i)^(-d_i), S, d = exp(clamp(., -25, 25)); monotone in t by construction.
        s = torch.exp(torch.clamp(s_raw, min=-25.0, max=25.0))
        d = torch.exp(torch.clamp(d_raw, min=-25.0, max=25.0))
        t = torch.max(torch.tensor(1.0), label_elapsed_seconds)
        r = torch.sum(w * torch.exp(-d * torch.log1p(t / (1e-7 + s))), dim=-1)
        return 1e-5 + (1 - 2 * 1e-5) * r

    @FunctionType
    def head_and_out_variant(self, input):
        x = self.prehead_dropout(self.prehead_norm(input))
        x_w = self.head_w(x).float()
        if self.gru_on:
            out_w_logits, out_s_raw, out_d_raw = self._gru_heads(x_w)
        else:
            out_w_logits = self.w_linear(x_w)
            out_s_raw = torch.zeros(1, dtype=torch.float32, device=x.device)
            out_d_raw = torch.zeros(1, dtype=torch.float32, device=x.device)
        out_w = torch.nn.functional.softmax(out_w_logits, dim=-1)
        out_w_log_p = torch.nn.functional.log_softmax(out_w_logits, dim=-1)
        if self.no_ahead_residual:
            out_ahead_logits = torch.zeros(x.size(0), x.size(1), self.num_points, dtype=torch.float32, device=x.device)
        else:
            out_ahead_logits = self.ahead_linear(self.head_ahead_logits(x).float())
        x_p = self.head_p(x).float()
        return out_ahead_logits, out_w, out_w_log_p, self.p_linear(x_p), out_s_raw, out_d_raw

    # ---- streams ----
    @torch.jit.ignore
    def _interleaved_streams(self, x: torch.Tensor, batch_sub_gather: List[List[torch.Tensor]],
                             batch_sub_gather_lens: List[List[int]], batch_time_shift_selects: List[List[torch.Tensor]],
                             batch_skips: List[List[torch.Tensor]]) -> torch.Tensor:
        """RWKV_INTERLEAVE: round r runs layer r of every stream deep enough, in hierarchy order.

        prepare() chains each stream's gathers on the previous stream's output layout; here every gather is
        re-anchored to the canonical row order of features2card (the order the final output and the labels use),
        so x stays canonical and each layer-step is gather -> one layer -> scatter-back of the non-padding rows.
        v0 (value residual) stays stream-local: set by the stream's layer 0, reused by its later layers.
        """
        cur = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        gathers, positions, targets = [], [], []
        for splits in batch_sub_gather:
            stream_gathers, stream_positions, stream_targets, parts = [], [], [], []
            for split in splits:
                index = split.long()
                canonical = torch.where(index >= 0, cur.index_select(0, index.clamp(min=0)), torch.full_like(index, -1))
                keep = torch.nonzero(canonical >= 0).squeeze(-1)
                stream_gathers.append(canonical)
                stream_positions.append(keep)
                stream_targets.append(canonical.index_select(0, keep))
                parts.append(canonical)
            gathers.append(stream_gathers)
            positions.append(stream_positions)
            targets.append(stream_targets)
            cur = torch.cat(parts)
        # The splits of one stream hold disjoint entities, so their canonical rows are disjoint: one gather and one
        # scatter per (stream, layer) step serve every split (same values as per-split gathers and scatters).
        fused_gather, fused_sizes, fused_positions, fused_targets = [], [], [], []
        for i in range(len(gathers)):
            sizes = [canonical.numel() for canonical in gathers[i]]
            offsets = [0]
            for size in sizes[:-1]:
                offsets.append(offsets[-1] + size)
            fused_gather.append(torch.cat(gathers[i]).clamp(min=0))
            fused_sizes.append(sizes)
            fused_positions.append(torch.cat([keep + offset for keep, offset in zip(positions[i], offsets)]))
            fused_targets.append(torch.cat(targets[i]))
        v0s = [[None] * len(splits) for splits in batch_sub_gather]
        exact = self.exact_index and x.is_cuda
        if exact:
            # Wide-kernels port: sort-free exact gather/scatter, one sort per stream reused by all its layers.
            # Bitwise equal to the deterministic index_select / index_copy below (scripts/check_wide_kernels.py).
            from rwkv.model.exact_index import ExactIndexCopy, OrderedIndexSelect, row_order
            rows = x.size(0)
            gather_orders, position_orders = [], []
            for i in range(len(gathers)):
                if self.skip_padding_sources:
                    # F50: padding sources (-1) leave the backward sum; their gradients are +0 here.
                    source = torch.cat(gathers[i])
                else:
                    source = fused_gather[i]
                gather_orders.append(row_order(source, rows))
                position_orders.append(row_order(fused_positions[i], fused_gather[i].numel()))
        for i, layer in self.layer_steps:
            # Scripted ModuleLists have no Python indexing; children are named by position.
            block = getattr(getattr(self.rwkv_modules, str(i)).blocks, str(layer))
            if exact:
                inputs = OrderedIndexSelect.apply(x, fused_gather[i], *gather_orders[i]).split(fused_sizes[i])
            else:
                inputs = x.index_select(0, fused_gather[i]).split(fused_sizes[i])
            outputs = []
            for s, x_split in enumerate(inputs):
                length = batch_sub_gather_lens[i][s]
                x_in = x_split.view(-1, length, self.d_model)
                v0_in = torch.empty_like(x_in) if layer == 0 else v0s[i][s]
                x_out, v0s[i][s] = block.forward(x_in, v0_in, batch_time_shift_selects[i][s].view(-1, length),
                                                 batch_skips[i][s].view(-1, length))
                outputs.append(x_out.reshape(-1, self.d_model))
            flat = torch.cat(outputs)
            if exact:
                kept = OrderedIndexSelect.apply(flat, fused_positions[i], *position_orders[i])
                x = ExactIndexCopy.apply(x, fused_targets[i], kept)
            else:
                x = x.index_copy(0, fused_targets[i], flat.index_select(0, fused_positions[i]))
        return x

    @FunctionType
    def forward_batch_variant(
        self,
        batch_start: torch.Tensor,
        batch_sub_gather: list[list[torch.Tensor]],
        batch_sub_gather_lens: list[list[int]],
        batch_time_shift_selects: list[list[torch.Tensor]],
        batch_skips: list[list[torch.Tensor]],
        batch_num_data: int,
    ):
        if self.zero_features_on:
            # RWKV_ZERO_FEATURES: the column is constant 0 in training and evaluation (dim 22 = scaled_state).
            batch_start = batch_start.index_fill(
                -1, torch.tensor(self.zero_features, dtype=torch.long, device=batch_start.device), 0.0)
        x = self.features2card(batch_start)

        assert len(batch_sub_gather) == len(self.rwkv_modules)
        if self.interleave_on:
            x = self._interleaved_streams(x, batch_sub_gather, batch_sub_gather_lens, batch_time_shift_selects,
                                          batch_skips)
        else:
            # Same sequential loop and R01 switches as SrsRWKV.forward_batch.
            for i, submodule in enumerate(self.rwkv_modules):
                module_splits = batch_sub_gather[i]
                sub_lens = batch_sub_gather_lens[i]
                time_shift_selects = batch_time_shift_selects[i]
                skips = batch_skips[i]
                if self.batched_split_gather and x.is_cuda:
                    x = self.forward_batched_gather(x, i, module_splits, sub_lens, time_shift_selects, skips)
                elif self.concurrent_splits and x.is_cuda:
                    x = self.forward_split_streams(x, i, module_splits, sub_lens, time_shift_selects, skips)
                else:
                    y = []
                    for split_gather, sub_len, time_shift_select, skip in zip(
                        module_splits, sub_lens, time_shift_selects, skips
                    ):
                        split_index = split_gather if self.fused_padded_gather else torch.clamp(split_gather, min=0)
                        if self.fused_padded_gather and x.is_cuda:
                            module_in = FusedPaddedIndexSelect.apply(x, split_gather).view(-1, sub_len, self.d_model)
                        elif self.native_rows and x.requires_grad and x.is_cuda:
                            module_in = torch.ops.rwkv_native.padded_rows(
                                x, split_gather if self.skip_padding_sources else split_index,
                                self.native_rows_handle).view(-1, sub_len, self.d_model)
                        elif self.exact_index and x.is_cuda:
                            if self.skip_padding_sources:
                                module_in = ExactPaddedIndexSelect.apply(x, split_gather).view(-1, sub_len, self.d_model)
                            else:
                                module_in = ExactIndexSelect.apply(x, split_index).view(-1, sub_len, self.d_model)
                        else:
                            module_in = torch.index_select(x, dim=0, index=split_index).view(-1, sub_len, self.d_model)
                        time_shift_select_BT = time_shift_select.view(-1, sub_len)
                        skip_BT = skip.view(-1, sub_len)
                        module_out = submodule(module_in, time_shift_select_BT=time_shift_select_BT, skip_BT=skip_BT)
                        y.append(module_out.view(-1, self.d_model))
                    x = torch.cat(y)
                if self.wddm_flush_streams and x.requires_grad and x.is_cuda:
                    x = wddm_flush_point(x)

        x = x.view(batch_num_data, -1, self.d_model)
        return self.head_and_out_variant(x)

    @FunctionType
    def _get_loss_variant(
        self,
        batch_start: torch.Tensor,
        batch_sub_gather: list[list[torch.Tensor]],
        batch_sub_gather_lens: list[list[int]],
        batch_time_shift_selects: list[list[torch.Tensor]],
        batch_skips: list[list[torch.Tensor]],
        batch_num_data: int,
        batch_labels: torch.Tensor,
        batch_label_review_th: torch.Tensor,
        kd_mix: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None,
    ):
        out_ahead_logits, out_w, out_w_log_p, out_p_logits, out_s_raw, out_d_raw = self.forward_batch_variant(
            batch_start,
            batch_sub_gather,
            batch_sub_gather_lens,
            batch_time_shift_selects,
            batch_skips,
            batch_num_data,
        )
        nan_probe = out_p_logits if self.no_ahead_residual else out_ahead_logits
        if not self.sync_free_loss and torch.isnan(nan_probe).any():
            return None

        global_labels = batch_labels.float()
        (
            label_elapsed_seconds,
            _,
            label_y,
            label_rating,
            has_label,
            label_is_equalize,
            is_query,
        ) = global_labels.unbind(-1)
        has_label = has_label.int()
        label_is_equalize = label_is_equalize.int()
        is_query = is_query.int()
        hard_y = label_y

        label_rating = torch.clamp(label_rating - 1, min=0)
        # Distillation (upstream iter 32 RWKV_KD_MIX): curve target = alpha * teacher curve + (1 - alpha) * label.
        if kd_mix is not None:
            kd_curve, kd_p, kd_alpha = kd_mix
            label_y = kd_alpha * kd_curve + (1.0 - kd_alpha) * label_y
        label_elapsed_seconds = label_elapsed_seconds.unsqueeze(-1)
        if self.gru_on:
            curve_probs_raw = self.gru_forgetting_curve(out_w, out_s_raw, out_d_raw, label_elapsed_seconds)
        else:
            curve_probs_raw = self.forgetting_curve(out_w, label_elapsed_seconds)
        curve_logits_raw = torch.log(
            curve_probs_raw / (1 - curve_probs_raw)
        )  # inverse sigmoid
        ahead_logit_residual = self.interp(out_ahead_logits, label_elapsed_seconds)
        curve_logits = curve_logits_raw + ahead_logit_residual
        curve_probs = torch.sigmoid(curve_logits)

        out_p_probs = torch.softmax(out_p_logits, dim=-1)
        out_p_again, out_p_1, out_p_2, out_p_3 = out_p_probs.unbind(dim=-1)
        out_p_binary = torch.clamp(1.0 - out_p_again, min=1e-5, max=1.0 - 1e-5)

        if not self.sync_free_loss and torch.isnan(curve_probs).any():
            raise Exception("nan")
        w_loss = torch.nn.functional.kl_div(
            input=out_w_log_p,
            target=torch.ones_like(out_w) / self.num_curves,
            reduction="none",
        ).mean(dim=-1)
        ahead_mask = (1 - is_query) * has_label
        immediate_mask = is_query * has_label
        assert ahead_mask.shape == label_is_equalize.shape
        ahead_equalize_mask = ahead_mask * label_is_equalize

        immediate_equalize_mask = immediate_mask * label_is_equalize
        curve_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            curve_logits, label_y, reduction="none"
        )
        curve_raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            curve_logits_raw, label_y, reduction="none"
        )
        NUM_LABELS = 4
        B, T = label_rating.shape
        if kd_mix is not None:
            # Rating target = alpha * teacher probabilities + (1 - alpha) * one-hot label (soft cross-entropy).
            kd_curve2, kd_p2, kd_alpha2 = kd_mix
            target = kd_alpha2 * kd_p2 + (1.0 - kd_alpha2) * torch.nn.functional.one_hot(
                label_rating.long(), NUM_LABELS).float()
            p_loss = -(target.view(-1, NUM_LABELS)
                       * torch.log_softmax(out_p_logits.float().view(-1, NUM_LABELS), dim=-1)).sum(dim=-1).view(B, T)
        else:
            p_loss = torch.nn.functional.cross_entropy(
                out_p_logits.view(-1, NUM_LABELS),
                label_rating.long().view(-1),
                reduction="none",
            ).view(B, T)
        p_binary_loss = torch.nn.functional.binary_cross_entropy(
            out_p_binary, hard_y, reduction="none"
        )
        ahead_avg = (curve_loss * ahead_mask).sum() / (1e-8 + ahead_mask.sum())
        AHEAD_SCALE = 0.5
        ahead_raw_avg = (curve_raw_loss * ahead_mask).sum() / (1e-8 + ahead_mask.sum())
        AHEAD_RAW_SCALE = 0.5
        immediate_avg = (p_loss * immediate_mask).sum() / (1e-8 + immediate_mask.sum())
        IMMEDIATE_SCALE = 1.0
        w_avg = (w_loss * ahead_mask).sum() / (1e-8 + ahead_mask.sum())
        W_LOSS_SCALE = 1e-5
        ahead_logits_mag_loss = torch.sqrt(
            1e-16 + out_ahead_logits.square().mean(dim=-1)
        )
        ahead_logits_mag_avg = (ahead_logits_mag_loss * ahead_mask).sum() / (
            1e-8 + ahead_mask.sum()
        )
        AHEAD_LOGITS_MAG_LOSS_SCALE = 1e-4
        ahead_logits_diff_loss = torch.sqrt(
            1e-16 + out_ahead_logits.diff().square().mean(dim=-1)
        )
        ahead_logits_diff_avg = (ahead_logits_diff_loss * ahead_mask).sum() / (
            1e-8 + ahead_mask.sum()
        )
        AHEAD_LOGITS_DIFF_LOSS_SCALE = 1e-3
        loss_avg = (
            AHEAD_SCALE * ahead_avg
            + IMMEDIATE_SCALE * immediate_avg
            + AHEAD_RAW_SCALE * ahead_raw_avg
            + W_LOSS_SCALE * w_avg
            + AHEAD_LOGITS_MAG_LOSS_SCALE * ahead_logits_mag_avg
            + AHEAD_LOGITS_DIFF_LOSS_SCALE * ahead_logits_diff_avg
        )
        loss_tensor = (
            AHEAD_SCALE * curve_loss.detach()
            + p_loss.detach()
            + AHEAD_RAW_SCALE * curve_raw_loss.detach()
            + W_LOSS_SCALE * w_loss.detach()
            + AHEAD_LOGITS_MAG_LOSS_SCALE * ahead_logits_mag_loss.detach()
            + AHEAD_LOGITS_DIFF_LOSS_SCALE * ahead_logits_diff_loss.detach()
        )

        ahead_equalize_avg = (curve_loss * ahead_equalize_mask).sum() / (
            1e-8 + ahead_equalize_mask.sum()
        )
        ahead_raw_equalize_avg = (curve_raw_loss * ahead_equalize_mask).sum() / (
            1e-8 + ahead_equalize_mask.sum()
        )
        immediate_binary_equalize_avg = (
            p_binary_loss * immediate_equalize_mask
        ).sum() / (1e-8 + immediate_equalize_mask.sum())

        ahead_n, ahead_equalize_n, imm_n, imm_binary_equalize_n = -1, -1, -1, -1
        if not self.sync_free_loss:
            ahead_n = int(ahead_mask.sum().detach().item())
            ahead_equalize_n = int(ahead_equalize_mask.sum().detach().item())
            imm_n = int(immediate_mask.sum().detach().item())
            imm_binary_equalize_n = int(immediate_equalize_mask.sum().detach().item())
        return SrsRWKVIterStatistics(
            average_loss=loss_avg,
            p_curve=curve_probs.detach(),
            p_imm=out_p_binary.detach(),
            p_imm_all=out_p_probs.detach(),
            loss_tensor=loss_tensor.detach(),
            ahead_avg=ahead_avg.detach(),
            ahead_raw_avg=ahead_raw_avg.detach(),
            ahead_n=ahead_n,
            ahead_equalize_avg=ahead_equalize_avg.detach(),
            ahead_raw_equalize_avg=ahead_raw_equalize_avg.detach(),
            ahead_equalize_n=ahead_equalize_n,
            imm_avg=immediate_avg.detach(),
            imm_n=imm_n,
            imm_binary_equalize_avg=immediate_binary_equalize_avg.detach(),
            imm_binary_equalize_n=imm_binary_equalize_n,
            w_loss_avg=w_avg.detach(),
            ahead_logits_mag_loss_avg=ahead_logits_mag_avg.detach(),
            ahead_logits_diff_loss_avg=ahead_logits_diff_avg.detach(),
            w=out_w.detach(),
            label_review_th=batch_label_review_th.detach(),
            label_elapsed_seconds=label_elapsed_seconds.detach(),
            label_rating=label_rating.detach(),
            is_query=is_query.detach(),
            has_label=has_label.detach(),
        )

    def get_loss(self, batch, kd_mix=None):
        return self._get_loss_variant(
            batch.start,
            batch.sub_gather,
            batch.sub_gather_lens,
            batch.time_shift_selects,
            batch.skips,
            batch.num_data,
            batch.labels,
            batch.label_review_th,
            kd_mix,
        )

    def copy_downcast_(self, master_model, dtype):
        master_params = dict(master_model.named_parameters())
        groups: dict = {}
        for name, param in self.named_parameters():
            target_dtype = torch.float32 if is_excluded(name) else dtype
            assert param.dtype == target_dtype, name
            dst, src = groups.setdefault(target_dtype, ([], []))
            dst.append(param.data)
            src.append(master_params[name].data)
        with torch.no_grad():
            for dst, src in groups.values():
                torch._foreach_copy_(dst, src)

    def selective_cast(self, dtype):
        # SrsRWKV.selective_cast skips the root module, so root parameters (the GRU head) stay float32.
        return SrsRWKV.selective_cast(self, dtype)
