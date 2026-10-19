"""SM32b: quantization-aware training (QAT) of the card/note state with the SM32 vector-quantization codec.

Model-side glue only; the codec itself is scripts/state_vq.StateCodec and scripts/state_vq.install_training fills
CODECS and sets the mixers' `state_vq_slot`. A mixer whose slot is -1 (the default, every stream when the option is
absent) never reaches this file, so the default path is untouched.

Placement = the CPU scorer's `+vq<recipe>` emulation (scripts/state_vq.py Hooks), now with a straight-through
estimator (STE):
  * WKV: per review of a card/note, the answer's output uses the float update; the STORED state (what the next review
    of that entity starts from) is the codec's reconstruction. Forward value = reconstruction exactly, gradient =
    identity through the codec. `VQStateWKV` is the per-step recurrence with that rule and an explicit backward (the
    plain WKV backward along the quantized trajectory), so autograd keeps one state per step instead of the ~10
    intermediates of a recorded Python loop.
  * Token shift: the vector a review reads from the entity's previous review (time_shift_select != t) is the codec's
    reconstruction; a review that has no previous one (select == t) reads its own input unchanged. Forward value =
    reconstruction exactly, gradient = identity to the gathered vector.
Codebooks are fixed during training (k-means, docs/SMALL_MODEL.md SM32b), so no codebook gradient is formed.
Speed (S09, docs/GPU_OPTIMIZATIONS.md): a codec with `.kernel` (scripts/state_vq_cuda.KernelCodec, set by
install_training for impl "cuda" / "hybrid") runs on CUDA through fused kernels: impl "cuda" = `VQStateWKVKernel`
(one forward launch per split for all steps + the plain WKV backward kernel along the stored trajectory), impl
"hybrid" = this file's torch forward + that backward kernel. Same forward value and STE gradient as VQStateWKV.
Both functions are @torch.jit.ignore: the scripted mixers call them through the Python fallback (the
`quant_aware_rwkv7` pattern), so the default JIT path stays scripted.
"""
from typing import Any, Dict

import torch
from torch import Tensor

from expertium.model.rwkv_ops import single_timestep

# stream index (0 = card, 2 = note) -> codec with .wkv(S [N, H, K, K]) and .shift(x [N, C], role "t" | "c")
CODECS: Dict[int, Any] = {}
# True only in scripts/check_state_vq_qat.py --dry-run: dispatch to the fused kernels on CPU tensors (their CPU
# emulation, scripts/state_vq_cuda_emu.py); training never sets it.
KERNELS_ON_CPU = False


def ste_backward(r, k, v, w, a, kd, skip, before, grad_out):
    """Straight-through WKV gradients along a stored trajectory: before[t] [B, H, K, K] = the state step t starts
    from (any layout that indexes that way, e.g. a [B, T, ...] tensor transposed). The plain WKV backward with the
    quantizer as identity (dq/dnxt = I on non-skipped steps)."""
    grad_out = grad_out.to(r.dtype)
    B, T, H, K = r.shape
    dr, dk, dv, dw, da, dkd = (torch.empty_like(r) for _ in range(6))
    dS = r.new_zeros(B, H, K, K)   # dL / d(stored state after step t)
    zero = r.new_zeros(())
    for t in range(T - 1, -1, -1):
        S = before[t]
        rt, kt, vt, wt, at, kdt = r[:, t], k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t]
        akd = at * kdt
        p = S @ kdt.unsqueeze(-1)                                              # S kd        [B, H, K, 1]
        nxt = S * wt.unsqueeze(-2) - p @ akd.unsqueeze(-2) + vt.unsqueeze(-1) @ kt.unsqueeze(-2)
        sk = skip[:, t, None, None, None]
        g = grad_out[:, t]
        # out_t = nxt r_t; the stored state is q(nxt) on non-skipped rows (STE: dq/dnxt = I), else S.
        dn = g.unsqueeze(-1) * rt.unsqueeze(-2) + torch.where(sk, zero, dS)
        dr[:, t] = (nxt.mT @ g.unsqueeze(-1)).squeeze(-1)
        dv[:, t] = (dn @ kt.unsqueeze(-1)).squeeze(-1)
        dk[:, t] = (dn.mT @ vt.unsqueeze(-1)).squeeze(-1)
        dw[:, t] = (S * dn).sum(-2)
        dakd = -(dn.mT @ p).squeeze(-1)
        dp = -(dn @ akd.unsqueeze(-1))
        dkd[:, t] = (S.mT @ dp).squeeze(-1) + dakd * at
        da[:, t] = dakd * kdt
        dS = dn * wt.unsqueeze(-2) + dp @ kdt.unsqueeze(-2) + torch.where(sk, dS, zero)
    return dr, dk, dv, dw, da, dkd


class VQStateWKV(torch.autograd.Function):
    """Per-step WKV recurrence whose stored state goes through `codec.wkv` after every non-skipped step (STE).

    Inputs r, k, v, w, a, kd [B, T, H, K] (one dtype, float32 in training), skip [B, T] bool. Same per-step
    arithmetic as rwkv_ops.single_timestep / the CPU scorer's _reference_wkv_q. Saves the stored state before every
    step ([T, B, H, K, K]); the backward recomputes each step's float update from it."""

    @staticmethod
    def forward(ctx, r, k, v, w, a, kd, skip, codec):
        B, T, H, K = r.shape
        state = r.new_zeros(B, H, K, K)
        kernel = getattr(codec, "kernel", None) if (r.is_cuda or KERNELS_ON_CPU) else None
        # hybrid: the backward kernel reads the stored states as [B, T, H, K, K]
        before = r.new_empty(B, T, H, K, K) if kernel is not None else r.new_empty(T, B, H, K, K)
        outs = []
        for t in range(T):
            if kernel is not None:
                before[:, t] = state
            else:
                before[t] = state
            out_t, nxt = single_timestep(r[:, t], k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t], state)
            outs.append(out_t)
            stored = codec.wkv(nxt).to(nxt.dtype)
            state = torch.where(skip[:, t, None, None, None], state, stored)
        ctx.kernel = kernel
        ctx.save_for_backward(r, k, v, w, a, kd, skip, before)
        return torch.stack(outs, dim=1)

    @staticmethod
    def backward(ctx, grad_out):
        r, k, v, w, a, kd, skip, before = ctx.saved_tensors
        if ctx.kernel is not None:
            return (*ctx.kernel.backward(r, k, v, w, a, kd, skip, before, grad_out), None, None)
        return (*ste_backward(r, k, v, w, a, kd, skip, before, grad_out), None, None)


class VQStateWKVKernel(torch.autograd.Function):
    """VQStateWKV on CUDA with fused kernels (impl "cuda"): `codec.kernel.forward` runs all steps of the split in one
    launch and returns the stored state before every step; the backward is the plain WKV backward kernel along it."""

    @staticmethod
    def forward(ctx, r, k, v, w, a, kd, skip, codec):
        out, before = codec.kernel.forward(r, k, v, w, a, kd, skip)
        ctx.kernel = codec.kernel
        ctx.save_for_backward(r, k, v, w, a, kd, skip, before)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        r, k, v, w, a, kd, skip, before = ctx.saved_tensors
        return (*ctx.kernel.backward(r, k, v, w, a, kd, skip, before, grad_out), None, None)


@torch.jit.ignore
def vq_wkv(r_BTHK: Tensor, k_BTHK: Tensor, v_BTHK: Tensor, w_BTHK: Tensor, a_BTHK: Tensor, k_deformed_BTHK: Tensor,
           skip_BT: Tensor, slot: int) -> Tensor:
    """QAT WKV for a card/note time mixer: float32 per-step recurrence with the stored state VQ-coded (STE)."""
    out_dtype = k_BTHK.dtype
    args = [t.float().contiguous() for t in (r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK)]
    codec = CODECS[slot]
    on_kernel_device = r_BTHK.is_cuda or KERNELS_ON_CPU
    fn = VQStateWKVKernel if (on_kernel_device and getattr(codec, "impl", "torch") == "cuda") else VQStateWKV
    return fn.apply(*args, skip_BT.contiguous(), codec).to(out_dtype)


@torch.jit.ignore
def vq_shift(x_shift_BTC: Tensor, time_shift_select_BT: Tensor, slot: int, role: int) -> Tensor:
    """QAT token shift: vectors read from the entity's previous review are VQ-coded (STE); role 0 = time mixer,
    1 = channel mixer."""
    B, T, C = x_shift_BTC.shape
    stored = time_shift_select_BT.long() != torch.arange(T, device=x_shift_BTC.device).unsqueeze(0)
    base = x_shift_BTC.detach()
    with torch.no_grad():
        coded = CODECS[slot].shift(base.reshape(B * T, C).float(), "t" if role == 0 else "c")
        coded = torch.where(stored.unsqueeze(-1), coded.reshape(B, T, C).to(base.dtype), base)
    # value = coded exactly (x - x.detach() is an exact zero), gradient = identity to the gathered vector
    return coded + (x_shift_BTC - base)
