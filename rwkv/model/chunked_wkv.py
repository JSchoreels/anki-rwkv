"""F69 prototype: chunked (parallel-scan) WKV7 recurrence in plain PyTorch. Not wired into training.

Recurrence (identical to `reference_rwkv7` and the native kernels), per (batch, head), state S in R^{K x K}:

    S_t   = S_{t-1} diag(w_t) - (S_{t-1} kd_t) (a_t * kd_t)^T + v_t k_t^T
    out_t = S_t r_t
    carried state: S_t if not skip_t else S_{t-1}   (the output still uses the updated S_t)

Chunk algebra (chunk length C, chunk start state S0, u = kd, z = a * kd, skipped steps use
w' = 1, z' = 0, k' = 0 in the carried recurrence only):

    X_t = S_{t-1} u_t,  (I + A) X = Bm V + S0-term,   A_ts = U~_t . Z~_s,  Bm_ts = U~_t . K~_s  (s < t)
    U~_t = u_t * exp(Lex_t), Z~_s = z'_s * exp(-L_s), K~_s = k'_s * exp(-L_s)
    L = inclusive cumulative log w' inside the chunk, Lex = exclusive (L_{t-1})

so one unit-lower-triangular solve per chunk replaces C sequential steps, and the chunk boundary
state follows S_end = S0 P + U with P, U built from the same solves; only N = ceil(T / C) boundary
matrix products stay sequential. The factorised exp(-L) terms are stable while
C * |log w| stays below about 40 in float32 (the model's w lies in [0.545, 1), so C <= 32 is safe);
`check_decay_range` reports the margin. Everything is differentiable; backward is the autograd derivative
of this forward (no hand-written backward kernel yet, see summary).
"""
import torch
from torch import Tensor


def check_decay_range(w: Tensor, chunk: int) -> float:
    """Largest cumulative |log w| over a chunk (float32 factorisation is safe below ~40)."""
    log_w = torch.log(w.float().clamp_min(1e-30)).neg()
    pad = (-log_w.shape[1]) % chunk
    if pad:
        log_w = torch.nn.functional.pad(log_w, (0, 0, 0, 0, 0, pad))
    b, t, h, k = log_w.shape
    return float(log_w.view(b, t // chunk, chunk, h, k).sum(2).max())


def chunked_wkv(r: Tensor, k: Tensor, v: Tensor, w: Tensor, a: Tensor, kd: Tensor, skip: Tensor,
                chunk: int = 16, compute_dtype=torch.float32) -> Tensor:
    """r,k,v,a,kd: [B,T,H,K] (any float dtype), w: [B,T,H,K] float, skip: [B,T] bool. Returns [B,T,H,K] in r.dtype."""
    out_dtype = r.dtype
    B, T, H, K = r.shape
    pad = (-T) % chunk
    cd = compute_dtype
    r, k, v, w, a, kd = (x.to(cd) for x in (r, k, v, w, a, kd))
    skip = skip.to(torch.bool)
    if pad:
        zero = lambda x: torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad))
        r, k, v, a, kd = map(zero, (r, k, v, a, kd))
        w = torch.nn.functional.pad(w, (0, 0, 0, 0, 0, pad), value=1.0)
        skip = torch.nn.functional.pad(skip, (0, pad), value=True)
    N = (T + pad) // chunk
    # [B,H,N,C,K]
    def split(x):
        return x.view(B, N, chunk, H, K).permute(0, 3, 1, 2, 4)
    r, k, v, w, a, kd = map(split, (r, k, v, w, a, kd))
    keep = (~skip).view(B, N, chunk, 1, 1).permute(0, 3, 1, 2, 4).to(cd)  # [B,1,N,C,1]
    u = kd
    z = a * kd
    # real step-t quantities for the output; masked quantities for the carried recurrence
    w_c = torch.where(keep.bool(), w, torch.ones_like(w))
    z_c = z * keep
    k_c = k * keep
    log_w = torch.log(w_c)
    L = log_w.cumsum(3)
    Lex = L - log_w
    ex = torch.exp(Lex)
    inv = torch.exp(-L)
    gc = torch.exp(L[:, :, :, -1:, :])                        # [B,H,N,1,K] decay over the whole chunk
    q = w * r - u * (z * r).sum(-1, keepdim=True)            # output covector for S_{t-1}
    Ut, Qt = u * ex, q * ex
    Zt, Kt = z_c * inv, k_c * inv
    lower = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=r.device), diagonal=-1)
    A = (Ut @ Zt.transpose(-1, -2)) * lower
    Bm = (Ut @ Kt.transpose(-1, -2)) * lower
    Aq = (Qt @ Zt.transpose(-1, -2)) * lower
    Bq = (Qt @ Kt.transpose(-1, -2)) * lower
    eye = torch.eye(chunk, dtype=cd, device=r.device)
    system = A + eye
    X_loc = torch.linalg.solve_triangular(system, Bm @ v, upper=False, unitriangular=True)   # [B,H,N,C,K(x)]
    Y = torch.linalg.solve_triangular(system, Ut, upper=False, unitriangular=True)           # [B,H,N,C,K(y)]
    O_loc = Bq @ v - Aq @ X_loc + v * (k * r).sum(-1, keepdim=True)
    Qhat = Qt - Aq @ Y
    ident = torch.eye(K, dtype=cd, device=r.device)
    P = (ident - Y.transpose(-1, -2) @ Zt) * gc                                              # columns scaled by gc[y]
    U = (v.transpose(-1, -2) @ Kt - X_loc.transpose(-1, -2) @ Zt) * gc
    # chunk-boundary scan: S_{n+1} = S_n P_n + U_n
    starts = []
    S = torch.zeros(B, H, K, K, dtype=cd, device=r.device)
    for n in range(N):
        starts.append(S)
        S = S @ P[:, :, n] + U[:, :, n]
    S0 = torch.stack(starts, dim=2)                                    # [B,H,N,K,K]
    out = (Qhat @ S0.transpose(-1, -2)) + O_loc                        # out_t[x] = sum_y S0[x,y] Qhat_t[y] + O_loc
    out = out.permute(0, 2, 3, 1, 4).reshape(B, N * chunk, H, K)[:, :T]
    return out.to(out_dtype)


def sequential_wkv(r, k, v, w, a, kd, skip, compute_dtype=torch.float64):
    """Straight-line recurrence in `compute_dtype` (ground truth for tests; autograd-capable)."""
    out_dtype = r.dtype
    r, k, v, w, a, kd = (x.to(compute_dtype) for x in (r, k, v, w, a, kd))
    B, T, H, K = r.shape
    S = torch.zeros(B, H, K, K, dtype=compute_dtype, device=r.device)
    outs = []
    for t in range(T):
        kdt = kd[:, t].unsqueeze(-1)
        Sn = S * w[:, t].unsqueeze(-2) - (S @ kdt) @ (a[:, t] * kd[:, t]).unsqueeze(-2)
        Sn = Sn + v[:, t].unsqueeze(-1) @ k[:, t].unsqueeze(-2)
        outs.append((Sn @ r[:, t].unsqueeze(-1)).squeeze(-1))
        S = torch.where(skip[:, t].view(B, 1, 1, 1), S, Sn)
    return torch.stack(outs, 1).to(out_dtype)
