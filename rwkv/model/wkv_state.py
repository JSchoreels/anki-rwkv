"""SM21 (docs/SMALL_MODEL.md, "training with carried state"): RWKV-7 WKV started from a carried state.

`wkv_state(r, k, v, w, a, kd, skip, state0)` returns (out [B, T, H, K], final state [B, H, K, K] float32) for the
recurrence started from `state0` [B, H, K, K] (row = value index, column = key index, as reference_rwkv7; zeros
= today's start). The initial state is detached: no gradient flows into it, and the final state is returned
detached (truncated backpropagation through time). The gradients of r, k, v, w, a and kd are the stock ones given
S0, computed by the stock backward kernels from state checkpoints that start at S0.

* CUDA: `RWKV7_WKV_State` = `rwkv.model.nvrtc.forward_with_state` (the production forward kernel rewritten to load
  S0 and store S_T, generic NVRTC layout or, inside `scripts.gpu_row_tiles.row_tiled_backend`, the exact row tiles)
  + the stock `nvrtc.backward`. Nothing here is used unless a caller asks for it: the stock RWKV7_WKV path, its
  kernels and its compiled modules are unchanged.
* CPU: `wkv_with_state`, the per-step `reference_rwkv7` loop started from S0 (the prototype's reference).
"""
import torch

from rwkv.model.rwkv_ops import single_timestep


def wkv_with_state(r, k, v, w, a, kd, skip, state0):
    """`reference_rwkv7` started from `state0` [B, H, K, K] (None = zeros); returns (out [B, T, H, K], final state)."""
    out_dtype = k.dtype
    r, k, v, w, a, kd = (x.float() for x in (r, k, v, w, a, kd))
    B, T, H, K = r.shape
    out = torch.empty(B, T, H, K, dtype=torch.float32, device=r.device)
    state = (torch.zeros(B, H, K, K, dtype=torch.float32, device=r.device) if state0 is None
             else state0.detach().float().clone())
    skip4 = skip.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    for t in range(T):
        out[:, t], nxt = single_timestep(r[:, t], k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t], state)
        state = torch.where(skip4[:, t], state, nxt)
    return out.to(out_dtype), state


class RWKV7_WKV_State(torch.autograd.Function):
    """CUDA WKV from a detached initial state; outputs (out, final state). Backward = the stock NVRTC backward."""

    @staticmethod
    def forward(ctx, r, k, v, w, a, kd, skip, state0):
        from rwkv.model import nvrtc
        out, checkpoints, final = nvrtc.forward_with_state(r, k, v, w, a, kd, skip, state0.detach().float().contiguous())
        ctx.save_for_backward(r, k, v, w, a, kd, skip, checkpoints)
        ctx.mark_non_differentiable(final)
        return out, final

    @staticmethod
    def backward(ctx, grad_out, grad_final):
        from rwkv.model import nvrtc
        r, k, v, w, a, kd, skip, checkpoints = ctx.saved_tensors
        gradients = nvrtc.backward(r, k, v, w, a, kd, skip, checkpoints, grad_out)
        return (*gradients, None, None)


def wkv_state(r, k, v, w, a, kd, skip, state0):
    """(out, final state) from `state0` [B, H, K, K]; CUDA tensors use the NVRTC kernels, CPU the reference."""
    if r.is_cuda:
        return RWKV7_WKV_State.apply(r, k, v, w, a, kd, skip, state0)
    return wkv_with_state(r, k, v, w, a, kd, skip, state0)
