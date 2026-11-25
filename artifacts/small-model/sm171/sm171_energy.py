#!/usr/bin/env python
"""State-energy penalty (SM171): training hook, CPU proofs. Design: artifacts/small-model/sm171/design.md.

Idea (outside research, pointer only; nothing of its code is used or opened): penalise, during a short cool-down, the
share of each card / note WKV head state's energy that a rank-2 truncation would drop, so that the stored states become
nearly rank 2 and the rank-2 codec (Q2, `w8+lr2q8x8`) costs about what rank 2 then 4 (T24) costs today.

Penalty, per stored write of a card / note WKV head state S (K x K, after the RWKV-7 step S' = S G + v k^T):
    d(S) = 1 - (sigma_1^2 + sigma_2^2) / ||S||_F^2        (dropped energy share under rank-2 truncation, in [0, 1])
    P    = mean over the coded streams (card, note) of the mean of d over every stored write x head of that stream in
           the update;  training gradient = grad(loss) + lambda * grad(P).
The top-2 right singular subspace V (K x 2) of S is tracked by a deflated power step per write ((S^T S)^16 by
repeated squaring, warm-started from the entity's previous write; no SVD, no eigen solver). With V fixed,
sigma_1^2 + sigma_2^2 >= ||S V||_F^2 with equality at the top-2 subspace, and its gradient there is 2 S V V^T (Danskin:
the Ky Fan 2-norm squared is a max of smooth functions), so
    dd/dS = (2 / ||S||^2) ((||S V||^2 / ||S||^2) S - S V V^T)
needs no differentiable SVD (no 1 / (sigma_i^2 - sigma_j^2) terms at ties).

Placement: the carried-state trainer (scripts/state_carry_train.carry_layer) of the run tree, unchanged. `install`
wraps the module function `carry_layer` (looked up at call time by the carry adapter) to mark which card / note block
is running, and replaces `rwkv.model.wkv_state.wkv_state` (imported by carry_layer at each call) by a dispatcher:
  * any other stream, or lambda = 0: the stock function, unchanged (lambda = 0 adds a no-grad diagnostic every
    DIAG_EVERY-th call only, which touches no tensor of the training graph);
  * a card / note block with lambda > 0: the stock function's (out, final) unchanged, then `EnergyTap` (identity on
    out in the forward) whose backward adds lambda * dP/d(k, v, w, a, kd) through the full per-step recurrence (the
    plain WKV backward with zero output gradient and the penalty's state gradient injected at every stored write).
The loss value the trainer logs is the stock loss; only the gradient carries the penalty. The forward of every arm is
the stock trainer's, so arms differ only by their gradients.

  python sm171_energy.py cpu [--out check-cpu-sm171.json]     CPU proofs P1-P5 (a few minutes, 4 threads)
"""
from argparse import ArgumentParser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CODE = Path(os.environ.get("SM171_CODE", "C:/Users/jonat/workspace/wt/sm171-run"))
CODE_COMMIT = "cea0f931"   # origin/sm148 = the SM148 scoring tree = the SM149 run tree; detached, never edited
sys.path.insert(0, str(CODE))

import torch  # noqa: E402

W = Path("C:/Users/jonat/workspace/wt")
DATA = r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"
TINY_P2 = W / "sm130/artifacts/small-model/sm130-th-dec2ep32k-p2"   # six-layer Tiny two-head (SM130) release
TRACK = "E89-F65e-F77a"
STREAMS = (0, 2)          # card, note (rwkv_modules order card, deck, note, preset, user)
RANK = 2
NIT = 1                   # deflated power steps per write (warm start from the entity's previous write)
NIT_INIT = 3              # steps for a carried start state (cold start from OMEGA)
EPS = 1e-20               # ||S||^2 below this: d = 0, no gradient (a zero state)
DIAG_EVERY = 16           # lambda = 0: a no-grad diagnostic of d on every 16th card / note call

CONFIG = {"lam": 0.0, "log": None, "rank": RANK, "nit": NIT}
CURRENT = []              # stream of the card / note block that carry_layer is running (stack)
CODED = {}                # id(time mixer) -> stream, from install()
CALLS = {"card_note": 0, "tap": 0, "diag": 0, "other": 0}
_ORIG = {}
_LOG = {"sum": {}, "n": {}, "calls": 0}   # pending diagnostic sums per stream (GPU scalars), flushed every 64 calls


# ------------------------------------------------------------------------------------------------ math

_OMEGA = {}


def omega(K, device):
    key = (K, str(device))
    if key not in _OMEGA:
        g = torch.Generator().manual_seed(171)
        _OMEGA[key] = torch.linalg.qr(torch.randn(K, RANK, generator=g, dtype=torch.float64))[0].float().to(device)
    return _OMEGA[key]


SQUARINGS = 4             # per write: (S^T S)^(2^4) by repeated squaring (convergence (s3/s2)^32 per write)


def _unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-30)


def _trace(M):
    return M.diagonal(dim1=-2, dim2=-1).sum(-1)


def _power(M):
    """M^(2^SQUARINGS) of a PSD M with trace 1, by repeated squaring, trace-normalised after each squaring."""
    for _ in range(SQUARINGS):
        M = M @ M
        M = M / _trace(M).clamp(min=1e-30)[..., None, None]
    return M


def _top2_once(S, V, fallback):
    """One deflated power step for the top-2 right singular vectors of S [..., K, K] from V [..., K, 2]:
    q1 = unit(P V[:, 0]) with P = (S^T S / tr)^(2^SQUARINGS); then the same on the deflated M2 = (I - q1 q1^T) M
    (I - q1 q1^T) for q2 (so a small sigma_2 / sigma_1 is resolved to float rounding of M, not lost to the power's
    dynamic range). Rank < 2 (M2 ~ 0): q2 = any unit vector orthogonal to q1 (d is then exactly 0)."""
    M = S.mT @ S
    M = M / _trace(M).clamp(min=1e-30)[..., None, None]
    P = _power(M)
    y1 = (P @ V[..., 0:1]).squeeze(-1)
    alt = (P @ fallback[..., 0:1]).squeeze(-1)
    y1 = torch.where(y1.norm(dim=-1, keepdim=True) > 1e-3 * alt.norm(dim=-1, keepdim=True), y1, alt)
    q1 = torch.where(y1.norm(dim=-1, keepdim=True) > 1e-30, _unit(y1), fallback[..., 0])
    proj = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device) - q1.unsqueeze(-1) @ q1.unsqueeze(-2)
    M2 = proj @ M @ proj
    t2 = _trace(M2)
    P2 = _power(M2 / t2.clamp(min=1e-30)[..., None, None])
    f = fallback[..., 1] - (q1 * fallback[..., 1]).sum(-1, keepdim=True) * q1
    f = torch.where(f.norm(dim=-1, keepdim=True) < 0.1, fallback[..., 0] - (q1 * fallback[..., 0]).sum(-1, keepdim=True) * q1, f)
    f = _unit(f - (q1 * f).sum(-1, keepdim=True) * q1)
    y2 = (P2 @ V[..., 1:2]).squeeze(-1)
    alt2 = (P2 @ f.unsqueeze(-1)).squeeze(-1)
    y2 = torch.where(y2.norm(dim=-1, keepdim=True) > 1e-3 * alt2.norm(dim=-1, keepdim=True), y2, alt2)
    y2 = y2 - (q1 * y2).sum(-1, keepdim=True) * q1
    y2 = y2 - (q1 * y2).sum(-1, keepdim=True) * q1
    good = (t2 > 1e-6).unsqueeze(-1) & (y2.norm(dim=-1, keepdim=True) > 1e-20)
    q2 = torch.where(good, _unit(y2), f)
    return torch.stack([q1, q2], -1)


def subspace(S, V, iters, fallback):
    """Top-2 right singular subspace of S [..., K, K]: `iters` deflated power steps (_top2_once) from V."""
    for _ in range(iters):
        V = _top2_once(S, V, fallback)
    return V


def drop_share(S, V):
    """(d [...], E [...], SV [..., K, 2]): dropped energy share 1 - ||S V||^2 / ||S||^2 (0 where ||S||^2 <= EPS)."""
    E = (S * S).sum((-2, -1))
    SV = S @ V
    keep = (SV * SV).sum((-2, -1))
    d = torch.where(E > EPS, (1 - keep / E.clamp(min=EPS)).clamp(min=0), torch.zeros_like(E))
    return d, E, SV


def drop_grad(S, V):
    """dd/dS with V fixed (Danskin): (2 / E) ((keep / E) S - S V V^T); 0 where E <= EPS."""
    E = (S * S).sum((-2, -1), keepdim=True)
    SV = S @ V
    keep = (SV * SV).sum((-2, -1), keepdim=True)
    Ec = E.clamp(min=EPS)
    G = (2 / Ec) * ((keep / Ec) * S - SV @ V.mT)
    return torch.where(E > EPS, G, torch.zeros_like(G))


def step(S, k, v, w, a, kd):
    """The RWKV-7 write S' = S diag(w) - (S kd)(a kd)^T + v k^T (expertium.model.state_vq_hooks.ste_backward's form);
    returns (S', p = S kd [..., K, 1], akd)."""
    akd = a * kd
    p = S @ kd.unsqueeze(-1)
    return S * w.unsqueeze(-2) - p @ akd.unsqueeze(-2) + v.unsqueeze(-1) @ k.unsqueeze(-2), p, akd


def start_V(state):
    K = state.shape[-1]
    fb = omega(K, state.device).expand(*state.shape[:-2], K, RANK)
    V = fb.clone()
    nz = (state.flatten(-2).abs().amax(-1) > 0)
    if bool(nz.any()):
        V = torch.where(nz[..., None, None], subspace(state, V, NIT_INIT, fb), V)
    return V, fb


def energy_forward(k, v, w, a, kd, skip, s0, save=True, nit=None):
    """Per-step recurrence from s0 [B, H, K, K] (None = zeros), float32, no grad. Returns (sum of d over stored writes
    x heads, number of stored writes x heads, before [T, B, H, K, K] or None, Vs [T, B, H, K, 2] or None, final)."""
    nit = CONFIG["nit"] if nit is None else nit
    k, v, w, a, kd = (x.detach().float() for x in (k, v, w, a, kd))
    B, T, H, K = k.shape
    state = k.new_zeros(B, H, K, K) if s0 is None else s0.detach().float().clone()
    V, fb = start_V(state)
    before = k.new_empty(T, B, H, K, K) if save else None
    Vs = k.new_empty(T, B, H, K, RANK) if save else None
    written = ~skip.bool()
    dsum = k.new_zeros(())
    for t in range(T):
        if save:
            before[t] = state
        nxt, _, _ = step(state, k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t])
        Vn = subspace(nxt, V, nit, fb)
        d, _, _ = drop_share(nxt, Vn)
        wt = written[:, t]
        dsum = dsum + (d.sum(-1) * wt).sum()
        m = wt[:, None, None, None]
        state = torch.where(m, nxt, state)
        V = torch.where(m, Vn, V)
        if save:
            Vs[t] = Vn
    count = written.sum() * H
    return dsum, count, before, Vs, state


def energy_backward(k, v, w, a, kd, skip, before, Vs, scale):
    """Gradients of scale * (sum of d over stored writes x heads) w.r.t. (k, v, w, a, kd), V fixed per step: the plain
    WKV backward (expertium.model.state_vq_hooks.ste_backward) with a zero output gradient and dd/dS' injected at
    every stored write. `scale` = a 0-d tensor."""
    k, v, w, a, kd = (x.detach().float() for x in (k, v, w, a, kd))
    B, T, H, K = k.shape
    dk, dv, dw, da, dkd = (torch.zeros_like(k) for _ in range(5))
    dS = k.new_zeros(B, H, K, K)
    zero = k.new_zeros(())
    skip = skip.bool()
    for t in range(T - 1, -1, -1):
        S = before[t]
        kt, vt, wt, at, kdt = k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t]
        nxt, p, akd = step(S, kt, vt, wt, at, kdt)
        sk = skip[:, t, None, None, None]
        dn = torch.where(sk, zero, dS + scale * drop_grad(nxt, Vs[t]))
        dv[:, t] = (dn @ kt.unsqueeze(-1)).squeeze(-1)
        dk[:, t] = (dn.mT @ vt.unsqueeze(-1)).squeeze(-1)
        dw[:, t] = (S * dn).sum(-2)
        dakd = -(dn.mT @ p).squeeze(-1)
        dp = -(dn @ akd.unsqueeze(-1))
        dkd[:, t] = (S.mT @ dp).squeeze(-1) + dakd * at
        da[:, t] = dakd * kdt
        dS = dn * wt.unsqueeze(-2) + dp @ kdt.unsqueeze(-2) + torch.where(sk, dS, zero)
    return dk, dv, dw, da, dkd


# ------------------------------------------------------------------------------------------------ per-update normalisation

class _Update:
    """Accumulator of one update: stored writes x heads per stream (GPU scalars). A new one starts at the first tap
    forward after a backward (all forwards of an update run before its single backward)."""
    current = None
    closed = True

    def __init__(self):
        self.count = {}

    @classmethod
    def get(cls):
        if cls.closed or cls.current is None:
            cls.current, cls.closed = _Update(), False
        return cls.current


class EnergyTap(torch.autograd.Function):
    """Identity on `out` in the forward (the stock WKV output, unchanged); backward = grad_out to `out` plus
    lambda / (n_streams x writes_stream) x d(sum d)/d(k, v, w, a, kd) of this call's stream."""

    @staticmethod
    def forward(ctx, out, r, k, v, w, a, kd, skip, s0, stream):
        dsum, count, before, Vs, _ = energy_forward(k, v, w, a, kd, skip, s0)
        upd = _Update.get()
        upd.count[stream] = upd.count.get(stream, 0) + count
        _log(stream, dsum, count)
        ctx.stream, ctx.update, ctx.dtypes = stream, upd, [x.dtype for x in (k, v, w, a, kd)]
        ctx.save_for_backward(k, v, w, a, kd, skip, before, Vs)
        return out.clone()

    @staticmethod
    def backward(ctx, grad_out):
        _Update.closed = True
        k, v, w, a, kd, skip, before, Vs = ctx.saved_tensors
        upd = ctx.update
        n_streams = len(upd.count)
        scale = CONFIG["lam"] / (n_streams * upd.count[ctx.stream].clamp(min=1).float())
        grads = energy_backward(k, v, w, a, kd, skip, before, Vs, scale)
        grads = [g.to(dt) for g, dt in zip(grads, ctx.dtypes)]
        return (grad_out, None, *grads, None, None, None)


def _log(stream, dsum, count):
    s = _LOG["sum"]
    s[stream] = s.get(stream, 0) + dsum.detach()
    _LOG["n"][stream] = _LOG["n"].get(stream, 0) + count.detach()
    _LOG["calls"] += 1
    if _LOG["calls"] % 64 == 0:
        flush()


def flush():
    if not _LOG["sum"] or not CONFIG["log"]:
        return
    rec = {"t": round(time.time(), 1), "lam": CONFIG["lam"], "calls": dict(CALLS)}
    for stream in sorted(_LOG["sum"]):
        n = float(_LOG["n"][stream])
        rec[f"stream{stream}"] = {"writes_x_heads": int(n), "mean_drop": float(_LOG["sum"][stream]) / max(n, 1.0)}
    _LOG["sum"], _LOG["n"] = {}, {}
    with open(CONFIG["log"], "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------------------------------------ the hook

def _carry_layer(block, *args, **kwargs):
    stream = CODED.get(id(block.time_mixer))
    if stream is None:
        return _ORIG["carry_layer"](block, *args, **kwargs)
    CURRENT.append(stream)
    try:
        return _ORIG["carry_layer"](block, *args, **kwargs)
    finally:
        CURRENT.pop()


def _wkv_state(r, k, v, w, a, kd, skip, state0):
    out, final = _ORIG["wkv_state"](r, k, v, w, a, kd, skip, state0)
    if not CURRENT:
        CALLS["other"] += 1
        return out, final
    stream = CURRENT[-1]
    CALLS["card_note"] += 1
    if CONFIG["lam"] > 0 and torch.is_grad_enabled() and out.requires_grad:
        CALLS["tap"] += 1
        out = EnergyTap.apply(out, r, k, v, w, a, kd, skip, state0, stream)
    elif CALLS["card_note"] % DIAG_EVERY == 1:
        CALLS["diag"] += 1
        with torch.no_grad():
            dsum, count, _, _, _ = energy_forward(k, v, w, a, kd, skip, state0, save=False)
        _log(stream, dsum, count)
    return out, final


def install(model, lam, log=None, nit=NIT):
    """Mark the card / note time mixers of `model` (an expertium-class model) and patch carry_layer / wkv_state."""
    from scripts import state_carry_train
    from rwkv.model import wkv_state as wkv_module
    from scripts.state_vq import training_layout
    if "carry_layer" not in _ORIG:
        _ORIG["carry_layer"] = state_carry_train.carry_layer
        _ORIG["wkv_state"] = wkv_module.wkv_state
    state_carry_train.carry_layer = _carry_layer
    wkv_module.wkv_state = _wkv_state
    CONFIG.update(lam=float(lam), log=log, nit=int(nit))
    CODED.clear()
    layout = training_layout(model)
    for stream, info in layout.items():
        for tm in info["time"]:
            if getattr(tm, "sm77_init", False) or getattr(tm, "sm77_keys", False):
                raise ValueError("SM171: SM77 card / note mixers are not covered (wkv_state_grad path)")
            if getattr(tm, "state_vq_slot", -1) >= 0:
                raise ValueError("SM171: a state-VQ mixer")
            CODED[id(tm)] = stream
    src = Path(__file__).resolve()
    return {"recipe": "sm171-energy", "lambda": float(lam), "rank": RANK, "nit": int(nit), "nit_init": NIT_INIT,
            "penalty": "mean over card / note streams of the mean dropped energy share 1 - (s1^2 + s2^2) / ||S||^2 of"
                       " every stored write x head; gradient only (the logged loss is the stock loss)",
            "streams": {str(s): {"wkv_layers": i["wkv_layers"], "heads": i["heads"]} for s, i in layout.items()},
            "hook": str(src), "hook_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "log": log}


def uninstall():
    from scripts import state_carry_train
    from rwkv.model import wkv_state as wkv_module
    if _ORIG:
        state_carry_train.carry_layer = _ORIG["carry_layer"]
        wkv_module.wkv_state = _ORIG["wkv_state"]
    CODED.clear()
    CONFIG.update(lam=0.0, log=None)


# ------------------------------------------------------------------------------------------------ CPU proofs

def synthetic(B, T, H, K, gen, nonskip=0.6):
    """Inputs with the RWKV-7 value ranges: w in (0, 1), a in (0, 1), unit-norm kd, k = kd * a."""
    r = torch.randn(B, T, H, K, generator=gen)
    v = torch.randn(B, T, H, K, generator=gen)
    kd = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=gen), dim=-1)
    a = torch.sigmoid(torch.randn(B, T, H, K, generator=gen))
    w = torch.exp(-torch.exp(-0.5 - torch.nn.functional.softplus(torch.randn(B, T, H, K, generator=gen))))
    k = kd * a
    skip = torch.rand(B, T, generator=gen) > nonskip
    skip[:, 0] = False
    return r, k, v, w, a, kd, skip


def exact_drop(S):
    s2 = torch.linalg.svdvals(S.double()) ** 2
    E = s2.sum(-1)
    return torch.where(E > EPS, 1 - s2[..., :RANK].sum(-1) / E.clamp(min=EPS), torch.zeros_like(E))


def p1_forward(gen):
    """P1: the penalty's per-step trajectory equals the stock CPU recurrence (rwkv.model.wkv_state.wkv_with_state):
    final states to float rounding; tracked d vs the exact SVD d of every stored write (K = 16, carried and zero
    start); the value is finite and in [0, 1]."""
    from rwkv.model.wkv_state import wkv_with_state
    out = {}
    ok = True
    for carried in (False, True):
        x = synthetic(8, 48, 4, 16, gen)
        s0 = None
        if carried:
            _, s0 = wkv_with_state(*synthetic(8, 12, 4, 16, gen)[:6], torch.zeros(8, 12, dtype=torch.bool), None)
            s0[0] = 0
        dsum, count, before, Vs, final = energy_forward(*x[1:6], x[6], s0)
        _, ref_final = wkv_with_state(*x[:6], x[6], s0)
        # exact d of every stored write, from the saved trajectory
        T = x[1].shape[1]
        ex, tr = [], []
        for t in range(T):
            nxt, _, _ = step(before[t], *(y[:, t] for y in x[1:6]))
            wt = ~x[6][:, t]
            ex.append(exact_drop(nxt)[wt].flatten())
            tr.append(drop_share(nxt, Vs[t])[0][wt].flatten())
        ex, tr = torch.cat(ex), torch.cat(tr).double()
        rel = float(((final - ref_final).abs().max() / ref_final.abs().max()))
        item = {"final_state_max_rel_diff_vs_stock": rel, "writes_x_heads": int(count),
                "mean_d_tracked": float(dsum / count), "mean_d_exact": float(ex.mean()),
                "max_abs_d_tracked_minus_exact": float((tr - ex).abs().max()),
                "p99_abs_d_tracked_minus_exact": float((tr - ex).abs().quantile(0.99)),
                "tracked_ge_exact": bool((tr >= ex - 1e-5).all())}
        out["carried" if carried else "zero_start"] = item
        ok &= (rel < 1e-5 and abs(item["mean_d_tracked"] - item["mean_d_exact"]) < 2e-3 and item["tracked_ge_exact"]
               and 0 <= item["mean_d_tracked"] <= 1 and torch.isfinite(dsum).item())
    return ok, out


def penalty_reference(k, v, w, a, kd, skip, s0, Vs):
    """Autograd reference: the same recurrence as a differentiable loop, d evaluated with the forward's V (detached)."""
    B, T, H, K = k.shape
    state = k.new_zeros(B, H, K, K) if s0 is None else s0.clone()
    total = k.new_zeros(())
    for t in range(T):
        nxt, _, _ = step(state, k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t])
        d, _, _ = drop_share(nxt, Vs[t].detach())
        wt = ~skip[:, t]
        total = total + (d.sum(-1) * wt).sum()
        state = torch.where(wt[:, None, None, None], nxt, state)
    return total


def p2_gradients(gen):
    """P2: energy_backward == autograd through the differentiable reference loop (same V), float64, K = 16, zero and
    carried start; and the Danskin gradient == autograd through the exact SVD d (float64) on generic matrices."""
    out = {}
    ok = True
    for carried in (False, True):
        x = [t.double() if t.is_floating_point() else t for t in synthetic(4, 20, 3, 16, gen)]
        skip = x[6]
        s0 = None
        if carried:
            s0 = torch.randn(4, 3, 16, 16, generator=gen, dtype=torch.float64) * 0.3
            s0[0] = 0
        k, v, w, a, kd = x[1:6]
        # forward in float64 (energy_forward casts to float32; the proof runs the same code in float64)
        state = k.new_zeros(4, 3, 16, 16) if s0 is None else s0.clone()
        V, fb = start_V(state.float())
        V, fb = V.double(), fb.double()
        before, Vs = [], []
        for t in range(20):
            before.append(state)
            nxt, _, _ = step(state, k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t])
            Vn = subspace(nxt, V, NIT, fb)
            m = (~skip[:, t])[:, None, None, None]
            state = torch.where(m, nxt, state)
            V = torch.where(m, Vn, V)
            Vs.append(Vn)
        before, Vs = torch.stack(before), torch.stack(Vs)
        scale = torch.tensor(0.37, dtype=torch.float64)
        ours = _backward64(k, v, w, a, kd, skip, before, Vs, scale)
        leaves = [t.clone().requires_grad_(True) for t in (k, v, w, a, kd)]
        (penalty_reference(*leaves, skip, s0, Vs) * scale).backward()
        ref = [t.grad for t in leaves]
        rel = max(float((o - r).abs().max() / r.abs().max().clamp(min=1e-30)) for o, r in zip(ours, ref))
        nonzero = all(float(o.abs().max()) > 0 for o in ours)
        out["carried" if carried else "zero_start"] = {"max_rel_grad_diff": rel, "nonzero": nonzero}
        ok &= rel < 1e-9 and nonzero
    # generic matrices with a singular-value gap (s3 / s2 <= 0.8): the SVD gradient is defined there
    U = torch.linalg.qr(torch.randn(64, 16, 16, generator=gen, dtype=torch.float64))[0]
    Q = torch.linalg.qr(torch.randn(64, 16, 16, generator=gen, dtype=torch.float64))[0]
    sv = torch.sort(torch.rand(64, 16, generator=gen, dtype=torch.float64), dim=-1, descending=True)[0]
    sv[:, 2:] *= 0.8 * sv[:, 1:2] / sv[:, 2:3]
    S = (U * sv.unsqueeze(-2)) @ Q.mT
    S.requires_grad_(True)
    exact_drop(S).sum().backward()
    with torch.no_grad():
        V = subspace(S, omega(16, "cpu").double().expand(64, 16, 2).clone(), 8, omega(16, "cpu").double().expand(64, 16, 2))
        G = drop_grad(S, V)
    rel = float((G - S.grad).abs().max() / S.grad.abs().max())
    out["danskin_vs_svd_autograd_max_rel"] = rel
    ok &= rel < 1e-6
    return ok, out


def _backward64(k, v, w, a, kd, skip, before, Vs, scale):
    """energy_backward's loop on float64 tensors (the same code with the float32 cast removed)."""
    B, T, H, K = k.shape
    dk, dv, dw, da, dkd = (torch.zeros_like(k) for _ in range(5))
    dS = k.new_zeros(B, H, K, K)
    zero = k.new_zeros(())
    for t in range(T - 1, -1, -1):
        S = before[t]
        kt, vt, wt, at, kdt = k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t]
        nxt, p, akd = step(S, kt, vt, wt, at, kdt)
        sk = skip[:, t, None, None, None]
        dn = torch.where(sk, zero, dS + scale * drop_grad(nxt, Vs[t]))
        dv[:, t] = (dn @ kt.unsqueeze(-1)).squeeze(-1)
        dk[:, t] = (dn.mT @ vt.unsqueeze(-1)).squeeze(-1)
        dw[:, t] = (S * dn).sum(-2)
        dakd = -(dn.mT @ p).squeeze(-1)
        dp = -(dn @ akd.unsqueeze(-1))
        dkd[:, t] = (S.mT @ dp).squeeze(-1) + dakd * at
        da[:, t] = dakd * kdt
        dS = dn * wt.unsqueeze(-2) + dp @ kdt.unsqueeze(-2) + torch.where(sk, dS, zero)
    return dk, dv, dw, da, dkd


def p3_backward_float32_matches_float64(gen):
    """P3: the production float32 energy_backward vs the float64 proof loop on the same inputs (relative 1e-4)."""
    x = synthetic(4, 24, 3, 16, gen)
    k, v, w, a, kd, skip = (*x[1:6], x[6])
    _, _, before, Vs, _ = energy_forward(k, v, w, a, kd, skip, None)
    scale = torch.tensor(0.5)
    g32 = energy_backward(k, v, w, a, kd, skip, before, Vs, scale)
    g64 = _backward64(*(t.double() for t in (k, v, w, a, kd)), skip, before.double(), Vs.double(), scale.double())
    rel = max(float((p.double() - q).abs().max() / q.abs().max()) for p, q in zip(g32, g64))
    return rel < 1e-4, {"max_rel_diff": rel}


def _setup(chunk):
    protocol = json.loads((TINY_P2 / "protocol.json").read_text())
    options = json.loads((TINY_P2 / "decay.json").read_text())["options"]
    options = {k: v for k, v in options.items() if k not in ("packed_ipc",)}
    options["chunk_reviews"] = chunk
    os.environ["GPU_BATCH1_DATA_PATH"] = protocol["dataset_path"]
    return protocol, options


def p4_real_updates(users, chunk, steps_per_user, lam, lr):
    """P4: six-layer Tiny two-head (SM130)'s release weights and Muon state, the release cool-down's options
    (strided-lane carry, show-time inputs, replay KD targets, alpha 0.5), short windows so that users have carried
    windows. Arms: stock (no hook), hook with lambda = 0, hook with lambda > 0, each from the same weights and optimizer
    state. The first update's loss is bitwise equal in all three arms (the forward is unchanged); lambda = 0 = stock
    bitwise on every update (loss and every gradient, diagnostics included); lambda > 0 changes
    the card / note gradients, finite; penalty taps on both streams, carried calls included; optimizer steps stay
    finite; the logged mean dropped share is finite and in [0, 1]."""
    from concurrent.futures import ThreadPoolExecutor
    from expertium.variants import build
    from scripts.expertium_options import CatchUp
    from scripts.gpu_r01_prefetch import training_batches
    from rwkv.train_rwkv import CLIP
    protocol, options = _setup(chunk)
    rows = [next(r for r in protocol["rows"] if r["user_id"] == u) for u in users]
    state_dict = torch.load(TINY_P2 / "expertium" / TRACK / "master.pth", map_location="cpu", weights_only=True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        items = list(training_batches(rows, chunk, options["training_seed"], pool, options, None, 2))
    report = {"users": users, "chunk_reviews": chunk, "lambda": lam,
              "reviews": {r["user_id"]: r["train_range"][1] - r["train_range"][0] for r in rows}, "arms": {}}
    log = HERE / "check-cpu-sm171-energy.jsonl"
    if log.exists():
        log.unlink()
    for arm in ("stock", "lam0", "lam"):
        model = build("w64-c1-d2p1u1").float()
        model.load_state_dict(state_dict, strict=True)
        info = None
        for key in CALLS:
            CALLS[key] = 0
        if arm != "stock":
            info = install(model, 0.0 if arm == "lam0" else lam, log=str(log))
        catch_up = CatchUp(options, {"lr_schedule": {"peak": lr, "warmup_updates": 1, "decay_start": 1.0,
                                                    "decay_end": 1.0}}, rows)
        catch_up.load_teacher(torch.device("cpu"), torch.float32)
        optimizer = catch_up.optimizer(model)
        optimizer.load_state_dict(torch.load(TINY_P2 / "expertium" / TRACK / "optimizer.pth", map_location="cpu",
                                             weights_only=True))   # fresh per arm: load_state_dict shares tensors
        names = [n for n, _ in model.named_parameters()]
        card_note = [n for n in names if n.startswith(("rwkv_modules.0.", "rwkv_modules.2."))]
        steps, per_user, done = [], {}, 0
        for row, item in items:
            u = row["user_id"]
            per_user[u] = per_user.get(u, 0) + 1
            if per_user[u] > steps_per_user:
                continue
            batch = item.batch.to(torch.device("cpu"))
            model.train()
            model.zero_grad(set_to_none=True)
            torch.manual_seed(item.seed)
            catch_up.carry.attach(item, batch, u)
            t0 = time.perf_counter()
            stats = catch_up.loss(model, batch, done, (u, item.offset, item.reviews, item.seed), batch.labels)
            stats.average_loss.backward()
            seconds = time.perf_counter() - t0
            params = dict(model.named_parameters())
            h = hashlib.sha256()
            for n in names:
                if params[n].grad is not None:
                    h.update(params[n].grad.detach().contiguous().numpy().tobytes())
            cn = [params[n].grad for n in card_note if params[n].grad is not None]
            other = [params[n].grad for n in names if n not in card_note and params[n].grad is not None]
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            done += item.reviews
            steps.append({"user": u, "offset": item.offset, "reviews": item.reviews, "loss": float(stats.average_loss.detach()),
                          "grad_norm": float(norm), "grad_sha256": h.hexdigest()[:16],
                          "card_note_grads_finite": all(bool(torch.isfinite(g).all()) for g in cn),
                          "card_note_grad_abs_sum": float(sum(g.abs().sum() for g in cn)),
                          "other_grad_abs_sum": float(sum(g.abs().sum() for g in other)), "seconds": round(seconds, 2)})
        flush()
        report["arms"][arm] = {"steps": steps, "calls": dict(CALLS), "hook": info,
                               "weights_finite_after_steps": all(bool(torch.isfinite(p).all()) for p in model.parameters()),
                               "carry": catch_up.carry.describe() if catch_up.carry else None}
        if arm != "stock":
            uninstall()
    s, z, p = (report["arms"][a]["steps"] for a in ("stock", "lam0", "lam"))
    logged = [json.loads(line) for line in open(log, encoding="utf-8")] if log.exists() else []
    drops = [rec[k]["mean_drop"] for rec in logged for k in rec if k.startswith("stream")]
    report["logged_mean_drop"] = drops
    calls = report["arms"]["lam"]["calls"]
    ok = (len(s) == len(z) == len(p) > 0
          and s[0]["loss"] == z[0]["loss"] == p[0]["loss"] and all(a["loss"] == b["loss"] for a, b in zip(s, z))
          and all(a["grad_sha256"] == b["grad_sha256"] for a, b in zip(s, z))
          and all(a["card_note_grad_abs_sum"] != c["card_note_grad_abs_sum"] for a, c in zip(s, p))
          and all(st["card_note_grads_finite"] for st in p) and calls["tap"] > 0 and calls["other"] > 0
          and all(report["arms"][a]["weights_finite_after_steps"] for a in report["arms"])
          and len(drops) > 0 and all(0 <= d <= 1 for d in drops))
    report["first_update_card_note_grad_change"] = p[0]["card_note_grad_abs_sum"] / s[0]["card_note_grad_abs_sum"] - 1
    report["seconds_per_update"] = {a: sum(st["seconds"] for st in report["arms"][a]["steps"]) / len(s)
                                    for a in report["arms"]}
    return ok, report


def main_cpu(args):
    os.environ["RWKV_NO_JIT"] = "1"
    os.environ["GPU_BATCH1_DATA_PATH"] = DATA
    os.environ["RWKV_DELETED_CARDS"] = "neighbour"
    torch.set_num_threads(4)
    gen = torch.Generator().manual_seed(171)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "scripts", "rwkv", "expertium"], cwd=CODE,
                           capture_output=True, text=True).stdout.strip()
    report = {"code_tree": str(CODE), "code_head": head, "code_dirty": bool(dirty), "torch": torch.__version__,
              "hook_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "checks": {}}
    users = [int(u) for u in args.users.split(",")]
    started = time.perf_counter()
    for name, fn in (("P1_trajectory_and_tracked_drop", lambda: p1_forward(gen)),
                     ("P2_gradients_vs_autograd", lambda: p2_gradients(gen)),
                     ("P3_float32_backward", lambda: p3_backward_float32_matches_float64(gen)),
                     ("P4_tiny_two_head_real_updates",
                      lambda: p4_real_updates(users, args.chunk, args.steps, args.lam, args.lr))):
        t0 = time.perf_counter()
        ok, detail = fn()
        report["checks"][name] = {"pass": bool(ok), "seconds": round(time.perf_counter() - t0, 1), "detail": detail}
        print(f"[sm171] {name}: {'OK' if ok else 'FAIL'} ({time.perf_counter() - t0:.0f} s)", flush=True)
    report["all_ok"] = all(c["pass"] for c in report["checks"].values()) and not report["code_dirty"]
    report["seconds"] = round(time.perf_counter() - started, 1)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[sm171] CPU proofs {'ALL OK' if report['all_ok'] else 'NOT OK'} -> {args.out}")
    return 0 if report["all_ok"] else 1


def main():
    p = ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cpu")
    c.add_argument("--out", default=str(HERE / "check-cpu-sm171.json"))
    c.add_argument("--users", default="4745,2308")
    c.add_argument("--chunk", type=int, default=256)
    c.add_argument("--steps", type=int, default=3)
    c.add_argument("--lam", type=float, default=1.0)
    c.add_argument("--lr", type=float, default=7e-5)
    args = p.parse_args()
    return main_cpu(args)


if __name__ == "__main__":
    raise SystemExit(main())
