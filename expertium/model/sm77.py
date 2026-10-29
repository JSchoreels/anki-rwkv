"""Tiny architecture additions (SM77, artifacts/small-model/sm77/DESIGN.md on main), opt-in per VariantConfig.

  (A) key_extras: separate removal and insertion key vectors (as in the RWKV-7 paper) plus learnable r / k biases (as
      in Mamba-3) in every time mixer, 4 x d_model values per layer:
        removal key   kd = k_scale * normalize(k * (1 + sm77_k_remove))          (per head)
        insertion key ki = k_scale * normalize(k) * (a + sm77_k_insert * (1 - a)) + sm77_k_bias
        receptance    r  = W_r(.) + sm77_r_bias
      Today: kd = k_scale * normalize(k), ki = kd * a. sm77_k_insert = 1 writes the whole key (no `a` factor), as the
      RWKV-7 paper's k_a = 0; the removal key stays normalized, so the transition keeps its eigenvalue range.
  (B) init_state: the layers of the named streams (card, note) learn the starting state of an entity that has no
      state yet (a never-seen card or note): the WKV state S0 = sm77_init_wkv [H, K, K] instead of zeros, and the
      token-shift vector at its first review = x + sm77_init_shift (time mixer) / + the channel mixer's
      sm77_init_shift, instead of x (the first review reads its own input today). No extra stored state: S0 and the
      offsets are weights.

Every new parameter starts at zero, so a model built with the flags and loaded from a checkpoint without them (the
missing tensors filled with zeros, scripts/sm77_warm.py extend) computes the same outputs. With the flags off nothing
here is built: expertium.model.variant_model keeps today's classes and parameters bit for bit.

Kernel paths (the WKV recurrence itself is unchanged; only its initial state is new):
  * (A) changes r, k and kd before the kernel: no kernel change.
  * (B) needs the recurrence started from S0 with a gradient for S0. Training (CUDA): the SM21 forward-with-state
    kernel (rwkv.model.nvrtc.forward_with_state, row tiles in scripts/gpu_row_tiles.py) plus a new rewrite of the
    backward kernel that also writes dL/dS0 (rwkv.model.wkv_state.wkv_state_grad). Scoring (CUDA): the long-history
    kernel with an initial state (scripts/small_model_wkv.wkv_long(..., state0)); CPU: the per-step references with S0.
    `wkv_from_state` below is the module-level dispatch the held-out scorer re-points (small_model_heldout.py).
"""
from typing import Tuple

import torch
from torch import Tensor

from expertium.model import rwkv_model as _erm
from expertium.model.rwkv_model import (FunctionType, ModuleType, RWKV7ChannelMixer, RWKV7Config, RWKV7TimeMixer)
from rwkv.model.exact_group_norm import ExactGroupNorm


def head_rows(skip_BT: Tensor) -> Tensor:
    """[B, T] bool: the rows at or before each sequence's first non-skipped row, i.e. the rows whose entity has no
    earlier review in the sequence (the rows whose token shift reads their own input and whose WKV starts from S0)."""
    B, T = skip_BT.shape
    t_index = torch.arange(T, device=skip_BT.device).unsqueeze(0).expand(B, T)
    first = torch.where(skip_BT.to(torch.bool), torch.full_like(t_index, T), t_index).amin(dim=1)
    return t_index <= first.unsqueeze(1)


def key_pair(k_BTC: Tensor, k_scale_BTH: Tensor, a_BTHK: Tensor, k_remove: Tensor, k_insert: Tensor, k_bias: Tensor,
             B: int, T: int, H: int, K: int) -> Tuple[Tensor, Tensor]:
    """(A): (removal key kd, insertion key ki) [B, T, H, K]; zero vectors give today's (kd, kd * a) exactly."""
    scale = k_scale_BTH.unsqueeze(-1)
    kd = scale * torch.nn.functional.normalize((k_BTC * (1.0 + k_remove.to(k_BTC.dtype))).view(B, T, H, K), dim=-1, p=2.0)
    kn = scale * torch.nn.functional.normalize(k_BTC.view(B, T, H, K), dim=-1, p=2.0)
    insert = k_insert.to(a_BTHK.dtype).view(H, K)
    ki = kn * (a_BTHK + insert * (1.0 - a_BTHK)) + k_bias.to(kn.dtype).view(H, K)
    return kd, ki


@torch.jit.ignore
def shift_gather(x_BTC: Tensor, select_BT: Tensor, exact: bool) -> Tensor:
    """The stock token-shift gather (exact sort-free kernel on CUDA when the trainer enables it). The gather goes through
    rwkv_model's `torch` so the CPU scorer's shift hooks (scripts/state_vq._TorchProxy) see it."""
    if exact and x_BTC.is_cuda:
        from rwkv.model.exact_index import ExactShiftGather
        return ExactShiftGather.apply(x_BTC, select_BT)
    return _erm.torch.gather(x_BTC, dim=1, index=select_BT.unsqueeze(-1).expand(-1, -1, x_BTC.size(-1)))


@torch.jit.ignore
def stock_wkv(r: Tensor, k: Tensor, v: Tensor, w: Tensor, a: Tensor, kd: Tensor, skip: Tensor) -> Tensor:
    """The stock recurrence from zero state, looked up at call time (the scorer replaces both)."""
    if r.is_cuda:
        return _erm.RWKV7_WKV.apply(r, k, v, w, a, kd, skip)
    return _erm.reference_rwkv7(r, k, v, w, a, kd, skip)


@torch.jit.ignore
def wkv_from_state(r: Tensor, k: Tensor, v: Tensor, w: Tensor, a: Tensor, kd: Tensor, skip: Tensor,
                   state0: Tensor) -> Tensor:
    """(B): the recurrence from `state0` [B, H, K, K] float32, differentiable in state0 (training). The held-out scorer
    replaces this module global with its long-history / state-rounding version (small_model_heldout.build_expertium)."""
    from rwkv.model.wkv_state import wkv_state_grad
    return wkv_state_grad(r, k, v, w, a, kd, skip, state0)[0]


class Sm77TimeMixer(RWKV7TimeMixer):
    __constants__ = ["sm77_keys", "sm77_init"]

    def __init__(self, config: RWKV7Config, layer_id, key_extras=False, init_state=False):
        RWKV7TimeMixer.__init__(self, config, layer_id)   # the stock parameters and random draws, unchanged
        self.sm77_keys = bool(key_extras)
        self.sm77_init = bool(init_state)
        C, H, K = config.d_model, self.H, self.K
        with torch.no_grad():
            if self.sm77_keys:
                self.sm77_r_bias = torch.nn.Parameter(torch.zeros(C))
                self.sm77_k_bias = torch.nn.Parameter(torch.zeros(C))
                self.sm77_k_remove = torch.nn.Parameter(torch.zeros(C))
                self.sm77_k_insert = torch.nn.Parameter(torch.zeros(C))
            if self.sm77_init:
                self.sm77_init_wkv = torch.nn.Parameter(torch.zeros(H, K, K))   # row = value index, as the kernels
                self.sm77_init_shift = torch.nn.Parameter(torch.zeros(C))

    @FunctionType
    def forward(self, in_BTC, v0_BTC, time_shift_select_BT, skip_BT):
        B, T, C = in_BTC.shape
        H, K = self.H, self.K
        if (self.fused_shift or self.native_shift or self.native_wkv or self.native_group_norm or self.native_glue
                or self.state_vq_slot >= 0 or self.state_qmax != float("inf") or self.state_lowrank_rank > 0
                or self.exact_gate_glue != 0 or self.exact_bonus_products):
            raise RuntimeError("SM77 time mixers run the stock, exact-shift and exact-group-norm paths only")

        x_BTC = self.layer_norm(in_BTC)
        x_shift_BTC = shift_gather(x_BTC, time_shift_select_BT, self.exact_shift)
        if self.sm77_init:
            head = head_rows(skip_BT).unsqueeze(-1)
            x_shift_BTC = torch.where(head, x_shift_BTC + self.sm77_init_shift.to(x_shift_BTC.dtype), x_shift_BTC)
        rkvdag_8BTC = torch.lerp(x_BTC.unsqueeze(0), x_shift_BTC.unsqueeze(0), self.rkvdag_lerp)
        r_BTC, k_BTC, v_BTC, d_BTC, a_BTC, g_BTC, k_scale_BTC, v_scale_BTC = rkvdag_8BTC.unbind(dim=0)
        r_BTC = self.W_r(r_BTC)
        if self.sm77_keys:
            r_BTC = r_BTC + self.sm77_r_bias.to(r_BTC.dtype)
        k_BTC = self.W_k(k_BTC)
        k_scale_BTH = torch.nn.functional.sigmoid(self.k_scale_linear(k_scale_BTC))
        v_scale_BTH = torch.nn.functional.sigmoid(self.v_scale_linear(v_scale_BTC))

        if self.layer_id == 0:
            v_BTC = self.W_v(v_BTC)
            v0_BTC = v_BTC
        else:
            v_lerp_BTC = torch.nn.functional.sigmoid(self.v_lora_simple(v_BTC))
            v_BTC = torch.lerp(self.W_v(v_BTC), v0_BTC, v_lerp_BTC)

        a_BTC = torch.nn.functional.sigmoid(self.a_lora_simple(a_BTC))
        g_BTC = self.lora_B_g(torch.nn.functional.sigmoid(self.lora_A_g(g_BTC)))

        _d_BTC = -0.5 - torch.nn.functional.softplus(-self.d_lora_mlp(d_BTC))
        w_BTC = torch.exp(-torch.exp(_d_BTC.float()))

        r_BTHK = r_BTC.view(B, T, H, K)
        v_BTHK = v_scale_BTH.unsqueeze(-1) * torch.nn.functional.normalize(v_BTC.view(B, T, H, K), dim=-1, p=2.0)
        w_BTHK = w_BTC.view(B, T, H, K)
        a_BTHK = a_BTC.view(B, T, H, K)
        if self.sm77_keys:
            k_deformed_BTHK, k_BTHK = key_pair(k_BTC, k_scale_BTH, a_BTHK, self.sm77_k_remove, self.sm77_k_insert,
                                               self.sm77_k_bias, B, T, H, K)
        else:
            k_deformed_BTHK = k_scale_BTH.unsqueeze(-1) * torch.nn.functional.normalize(
                k_BTC.view(B, T, H, K), dim=-1, p=2.0)
            k_BTHK = k_deformed_BTHK * a_BTHK

        if self.sm77_init:
            state0 = self.sm77_init_wkv.float().unsqueeze(0).expand(B, H, K, K).contiguous()
            out_BTHK = wkv_from_state(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT, state0)
        else:
            out_BTHK = stock_wkv(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT)

        if self.exact_group_norm and out_BTHK.is_cuda:
            out_BTC = ExactGroupNorm.apply(out_BTHK.view(B * T, C), self.out_group_norm.weight,
                                           self.out_group_norm.bias, self.out_group_norm.num_groups,
                                           self.out_group_norm.eps).view(B, T, C)
        else:
            out_BTC = self.out_group_norm(out_BTHK.view(B * T, C)).view(B, T, C)
        bonus_products = r_BTHK * self.bonus * k_BTHK
        bonus_BTC = (bonus_products.sum(dim=-1, keepdim=True) * v_BTHK).view(B, T, C)
        gated = g_BTC * (out_BTC + bonus_BTC)
        out_BTC = self.W_o(gated)
        return in_BTC + self.dropout(out_BTC), v0_BTC


class Sm77ChannelMixer(RWKV7ChannelMixer):
    """(B) for the channel mixer: its token shift at an entity's first review = x + sm77_init_shift."""

    def __init__(self, config: RWKV7Config, layer_id):
        RWKV7ChannelMixer.__init__(self, config, layer_id)
        with torch.no_grad():
            self.sm77_init_shift = torch.nn.Parameter(torch.zeros(config.d_model))

    @FunctionType
    def forward(self, in_BTC, time_shift_select_BT, skip_BT):
        if self.native_shift or self.native_glue or self.state_vq_slot >= 0 or self.exact_channel_glue:
            raise RuntimeError("SM77 channel mixers run the stock and exact-shift paths only")
        x_BTC = self.layer_norm(in_BTC)
        x_shift_BTC = shift_gather(x_BTC, time_shift_select_BT, self.exact_shift)
        head = head_rows(skip_BT).unsqueeze(-1)
        x_shift_BTC = torch.where(head, x_shift_BTC + self.sm77_init_shift.to(x_shift_BTC.dtype), x_shift_BTC)
        k_BTK = self.W_k(torch.lerp(x_BTC, x_shift_BTC, self.lerp_k))
        o_BTC = self.W_v(torch.square(torch.nn.functional.relu(k_BTK)))
        return in_BTC + self.dropout(o_BTC)


def new_parameter_names(model):
    """The SM77 parameters of a model (names as in its state_dict)."""
    return [name for name, _ in model.named_parameters() if ".sm77_" in name]
