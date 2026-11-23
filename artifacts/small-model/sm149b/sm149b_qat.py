#!/usr/bin/env python
"""Rank-2 states with a tail-energy penalty (SM149b): the rank-2 cool-down hook (SM149, straight-through estimator)
plus one training term, and its CPU proofs. Design: artifacts/small-model/sm149b/design.md.

A copy of artifacts/small-model/sm149/sm149_qat.py (the hook: TrainCodec, carried_left, LowRankSTEWKV, the dispatching
shims, install / uninstall), extended by the tail-energy penalty. Nothing of the code tree (SM149_CODE, default
C:/Users/jonat/workspace/wt/sm149-run = origin/sm148, cea0f931) is edited.

Penalty (an idea of outside research, reimplemented from its description only):
  tail = sum_{i > r} sigma_i^2 / (sum_i sigma_i^2 + eps) of the PRE-codec state S' (the float update of a write,
         before the rank-2 truncation), per (row, head), r = 2, eps = 1e-6;
  loss += lambda x mean(tail) over the sampled (row, head) entries of every coded layer and stream of one update;
  sampled: one write step in `stride` (default 8), counted per stream (card / note) across the hook calls; rows whose
  write has n < 3 answers are masked (their S' has rank <= 2, tail exactly 0), as are zero / non-finite states.
Value and gradient without an SVD: T = the best rank-2 approximation of S' (SM148's factored solve with float factors,
codec lr2fx8, no gradient) gives the exact tail numerator ||S' - T||^2 (Eckart-Young), and by Danskin's theorem its
gradient is 2 (S' - T) with T held fixed (T minimises the distance over rank-2 matrices): finite at singular-value
ties (any minimiser gives a valid subgradient) and 0 at rank <= 2. d tail / d S' = (2 (S' - T) - 2 tail S') /
(||S'||^2 + eps). The gradient flows along the whole coded trajectory: the backward adds lambda / N x d tail / d S' to
dL / d S' at the sampled writes inside the straight-through WKV backward (a copy of
expertium.model.state_vq_hooks.ste_backward with that injection; the fused K = 16 kernel cannot take it, so a call with
sampled entries uses the torch loop, a call without keeps the kernel). lambda = 0 is SM149's hook exactly (proof B0).

  python sm149b_qat.py cpu [--out check-cpu-sm149b.json]       CPU proofs B0-B4
"""
from argparse import ArgumentParser
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CODE = Path(os.environ.setdefault("SM149_CODE", "C:/Users/jonat/workspace/wt/sm149-run"))
CODE_COMMIT = "cea0f931"          # origin/sm148 = SM149's run tree (unedited)
sys.path.insert(0, str(CODE))

import torch  # noqa: E402

W = Path("C:/Users/jonat/workspace/wt")
DATA = r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"
SM118_P2 = W / "sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2"
TRACK = "E89-F65e-F77a"
SM149_QAT = HERE.parent / "sm149" / "sm149_qat.py"    # SM149's hook, for the lambda = 0 proof (B0)


# ------------------------------------------------------------------------------------------------ the hook (SM149)

def make_codec(spec="2", quant="q8"):
    return codec_of_recipe(f"lr{spec}{quant}x8")


def codec_of_recipe(recipe):
    """'lr2q8x8' (the SM148 label's state part) -> the configured LowRankCodec, REF's shift / row bits (as SM149)."""
    from scripts.small_model_wkv import F16_SCALE
    from scripts.state_lowrank import LowRankCodec, parse_sm148
    codec = parse_sm148(recipe)
    if codec is not None:
        codec.rows_bits, codec.shift_bits = 8 | F16_SCALE, 8 | F16_SCALE
    if not isinstance(codec, LowRankCodec) or codec.has_rows or codec.float_shift:
        raise ValueError(f"SM149b training recipe {recipe!r}: a pure rank-r lr<spec>q8x8 / lr<spec>fx8 codec")
    return codec


class TrainCodec:
    """CODECS entry of a low-rank stream (as SM149) plus, for the penalty, the float-factor codec of the same spec
    (the exact rank-r truncation T) and the stream's write-step counter (the stride's phase)."""
    lowrank = True
    learned = None
    impl = "torch"

    def __init__(self, codec):
        from scripts.state_vq import int_row_quant
        self.codec = codec
        self._shift = int_row_quant(codec.shift_bits)
        self.float_codec = codec_of_recipe(f"lr{codec.spec}fx8")
        self.counter = 0

    def shift(self, x, role):
        return self._shift(x)

    def sync(self):
        pass


_OMEGA = {}
CHUNK = 16384


def _omega(K, m, device):
    key = (K, m, str(device))
    if key not in _OMEGA:
        g = torch.Generator().manual_seed(149)
        _OMEGA[key] = torch.randn(K, m, generator=g, dtype=torch.float64).to(device)
    return _OMEGA[key]


def carried_left(state, rmax):
    """SM149's left factors of carried stored states (randomised range finder, float64 QR), unchanged."""
    N, H, K, _ = state.shape
    left = state.new_zeros(N, H, K, rmax)
    nz = (state.abs().flatten(2).amax(-1) > 0) & torch.isfinite(state).flatten(2).all(-1)
    if bool(nz.any()):
        S = state[nz]
        om = _omega(K, rmax, state.device)
        parts = []
        for i in range(0, S.shape[0], CHUNK):
            Q, _ = torch.linalg.qr(S[i:i + CHUNK].double() @ om)
            parts.append(Q.to(state.dtype))
        left[nz] = torch.cat(parts)
    return left


# ------------------------------------------------------------------------------------------------ the penalty

TAIL = {"lambda": 0.0, "stride": 8, "eps": 1e-6, "min_answers": 3}
MEASURE = [None]          # proofs only: a _Group that collects the tail of every write (no gradient, stride 1)


class _Group:
    """The penalty entries of one forward / backward pass (one update): sum of tails and count (device scalars, no
    host sync). All forwards of an update run before its backward, so the count is final when a backward reads it."""
    __slots__ = ("sum", "n", "closed", "index")
    counter = 0

    def __init__(self, device):
        self.sum = torch.zeros((), dtype=torch.float64, device=device)
        self.n = torch.zeros((), dtype=torch.float64, device=device)
        self.closed = False
        _Group.counter += 1
        self.index = _Group.counter


_CURRENT = [None]
LOG = {"path": None, "pending": [], "flush_every": 50, "print_every": 100, "unprinted": [], "written": 0}


def _group(device):
    g = _CURRENT[0]
    if g is None or g.closed:
        g = _CURRENT[0] = _Group(device)
    return g


def _close(group):
    if group.closed:
        return
    group.closed = True
    LOG["pending"].append(group)
    if len(LOG["pending"]) >= LOG["flush_every"]:
        flush_log()


def _print(chunk):
    s, n = sum(c[1] for c in chunk), sum(c[2] for c in chunk)
    print(f"\n[sm149b] tail energy, update passes {chunk[0][0]}-{chunk[-1][0]}: mean {s / max(n, 1):.5f} over "
          f"{int(n):,} sampled states, lambda {TAIL['lambda']:g}, stride {TAIL['stride']}", flush=True)


def flush_log(final=False):
    """Write the closed groups' mean tails (one JSON line per update pass, one host sync per flush) to LOG["path"] and
    print one summary line per `print_every` passes (the remainder on the final flush)."""
    pending, LOG["pending"] = LOG["pending"], []
    if pending:
        vals = torch.stack([torch.stack([g.sum, g.n]) for g in pending]).cpu().tolist()
        lines = []
        for g, (s, n) in zip(pending, vals):
            mean = s / n if n else None
            lines.append({"update_pass": g.index, "tail_mean": mean, "entries": int(n),
                          "penalty": TAIL["lambda"] * mean if mean is not None else 0.0,
                          "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            LOG["unprinted"].append((g.index, s, n))
        if LOG["path"]:
            with open(LOG["path"], "a", encoding="utf-8") as f:
                for line in lines:
                    f.write(json.dumps(line) + "\n")
        LOG["written"] += len(lines)
    while len(LOG["unprinted"]) >= LOG["print_every"]:
        _print(LOG["unprinted"][:LOG["print_every"]])
        LOG["unprinted"] = LOG["unprinted"][LOG["print_every"]:]
    if final and LOG["unprinted"]:
        _print(LOG["unprinted"])
        LOG["unprinted"] = []


def tail_terms(nxt, trunc, eps):
    """nxt, trunc [M, H, K, K] float32 (trunc = the best rank-r approximation of nxt, no gradient) -> tail [M, H], the
    gradient d tail / d nxt [M, H, K, K] (Danskin: trunc held fixed), both zeroed where not valid, and the valid mask
    [M, H] (finite, non-zero state)."""
    R = nxt - trunc
    res = R.square().sum((-2, -1))
    tot = nxt.square().sum((-2, -1))
    den = tot + eps
    tail = res / den
    grad = (2.0 * R - (2.0 * tail)[..., None, None] * nxt) / den[..., None, None]
    ok = torch.isfinite(tail) & torch.isfinite(grad).flatten(2).all(-1) & (tot > 0)
    return (torch.where(ok, tail, torch.zeros_like(tail)),
            torch.where(ok[..., None, None], grad, torch.zeros_like(grad)), ok)


def ste_backward_tail(r, k, v, w, a, kd, skip, before, grad_out, inject, coef):
    """expertium.model.state_vq_hooks.ste_backward (straight-through WKV gradients along the stored trajectory,
    before[t] = the state step t starts from), copied, plus the penalty: at step t, coef x inject[t][1] is added to
    dL / d nxt of the rows inject[t][0] (nxt = the pre-codec update of that write)."""
    grad_out = grad_out.to(r.dtype)
    B, T, H, K = r.shape
    dr, dk, dv, dw, da, dkd = (torch.empty_like(r) for _ in range(6))
    dS = r.new_zeros(B, H, K, K)
    zero = r.new_zeros(())
    coef = coef.to(r.dtype)
    for t in range(T - 1, -1, -1):
        S = before[t]
        rt, kt, vt, wt, at, kdt = r[:, t], k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t]
        akd = at * kdt
        p = S @ kdt.unsqueeze(-1)
        nxt = S * wt.unsqueeze(-2) - p @ akd.unsqueeze(-2) + vt.unsqueeze(-1) @ kt.unsqueeze(-2)
        sk = skip[:, t, None, None, None]
        g = grad_out[:, t]
        dn = g.unsqueeze(-1) * rt.unsqueeze(-2) + torch.where(sk, zero, dS)
        if t in inject:
            rows, gp = inject[t]
            dn = dn.index_add(0, rows, gp * coef)
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


class LowRankSTEWKV(torch.autograd.Function):
    """SM149's per-step coded recurrence (forward = the SM148 scorer loop, backward = straight-through), plus the
    tail-energy penalty when `pen` is given ({"group", "stride", "grad"}). The forward VALUE never depends on pen."""

    @staticmethod
    def forward(ctx, r, k, v, w, a, kd, skip, s0, codec, pen):
        from expertium.model.rwkv_ops import single_timestep
        c = codec.codec
        B, T, H, K = r.shape
        kernel = r.is_cuda and K == 16 and not FORCE_TORCH_BACKWARD
        if s0 is None:
            state = r.new_zeros(B, H, K, K)
            left = r.new_zeros(B, H, K, c.rmax)
            base = torch.zeros(B, dtype=torch.long, device=r.device)
        else:
            state = s0.float().clone()
            left = carried_left(state, c.rmax)
            # a carried entity has >= 1 earlier write, so its next write has n >= 2: rank min(n, r) = r for r <= 2
            base = (state.flatten(1).abs().amax(1) > 0).long() * c.rmax
        skip = skip.bool()
        counts = base[:, None] + torch.cumsum((~skip).long(), 1)
        before = r.new_empty(B, T, H, K, K) if kernel else r.new_empty(T, B, H, K, K)
        out = r.new_empty(B, T, H, K)
        inject = {}
        if pen is not None:
            phase, stride, group = codec.counter, pen["stride"], pen["group"]
            codec.counter += T
        for t in range(T):
            if kernel:
                before[:, t] = state
            else:
                before[t] = state
            out[:, t], nxt = single_timestep(r[:, t], k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t], state)
            rows = torch.nonzero(~skip[:, t]).flatten()
            if pen is not None and rows.numel() and (phase + t) % stride == 0:
                pr = rows[counts[rows, t] >= TAIL["min_answers"]]
                if pr.numel():
                    trunc, _ = codec.float_codec.encode(nxt[pr], left[pr], v[pr, t], counts[pr, t], None)
                    tail, grad, ok = tail_terms(nxt[pr], trunc, TAIL["eps"])
                    group.sum += tail.sum().double()
                    group.n += ok.sum().double()
                    if pen["grad"]:
                        inject[t] = (pr, grad)
            if rows.numel():
                s_new, l_new = c.encode(nxt[rows], left[rows], v[rows, t], counts[rows, t], None)
                state[rows] = s_new
                left[rows] = l_new
        ctx.kernel = kernel
        ctx.inject = inject
        ctx.group = pen["group"] if pen is not None and pen["grad"] else None
        ctx.save_for_backward(r, k, v, w, a, kd, skip, before)
        ctx.mark_non_differentiable(state)
        return out, state

    @staticmethod
    def backward(ctx, grad_out, grad_final):
        from expertium.model.state_vq_hooks import ste_backward
        r, k, v, w, a, kd, skip, before = ctx.saved_tensors
        if ctx.inject:
            coef = TAIL["lambda"] / ctx.group.n.clamp(min=1.0)
            tb = before.transpose(0, 1) if ctx.kernel else before   # the kernel layout [B, T, ...], indexed per step
            grads = ste_backward_tail(r, k, v, w, a, kd, skip, tb, grad_out, ctx.inject, coef)
            CALLS["torch_backward"] += 1
        elif ctx.kernel:
            from scripts.state_vq_cuda import wkv_backward
            grads = wkv_backward(r, k, v, w, a, kd, skip, before, grad_out)
        else:
            grads = ste_backward(r, k, v, w, a, kd, skip, before, grad_out)
        if ctx.group is not None:
            _close(ctx.group)
        ctx.inject = None
        return (*grads, None, None, None, None)


FORCE_TORCH_BACKWARD = False
_ORIGINAL = {}
CALLS = {"window": 0, "carried": 0, "penalised": 0, "torch_backward": 0}


def _pen_for(r):
    if MEASURE[0] is not None:
        return {"group": MEASURE[0], "stride": 1, "grad": False}
    if TAIL["lambda"] > 0 and torch.is_grad_enabled() and r.requires_grad:
        CALLS["penalised"] += 1
        return {"group": _group(r.device), "stride": TAIL["stride"], "grad": True}
    return None


class _Shim:
    """SM149's stand-in for state_vq_hooks' autograd classes (low-rank codec -> LowRankSTEWKV, else the original)."""

    def __init__(self, name):
        self.name = name

    def apply(self, *args):
        codec = args[-1]
        if not getattr(codec, "lowrank", False):
            return _ORIGINAL[self.name].apply(*args)
        r, k, v, w, a, kd, skip = args[:7]
        pen = _pen_for(r)
        if self.name == "VQStateWKVLearned":            # (r, k, v, w, a, kd, skip, s0, books, codec): carried path
            s0 = args[7]
            CALLS["window"] += 1
            CALLS["carried"] += s0 is not None
            return LowRankSTEWKV.apply(r, k, v, w, a, kd, skip, s0, codec, pen)
        CALLS["window"] += 1                             # (r, k, v, w, a, kd, skip, codec): one window from zero
        return LowRankSTEWKV.apply(r, k, v, w, a, kd, skip, None, codec, pen)[0]


def install(model, spec="2", quant="q8", recipe=None, tail_lambda=0.0, tail_stride=8):
    """SM149's install plus the penalty settings (lambda, stride). Returns a description (result.json state_vq)."""
    from expertium.model import state_vq_hooks as hooks
    from scripts.state_vq import training_layout
    names = ("VQStateWKV", "VQStateWKVKernel", "VQStateWKVLearned")
    for name in names:
        if name not in _ORIGINAL and isinstance(getattr(hooks, name), type):
            _ORIGINAL[name] = getattr(hooks, name)
    if set(_ORIGINAL) != set(names):
        raise RuntimeError("SM149b install: state_vq_hooks holds another hook's shims (uninstall that one first)")
    TAIL["lambda"], TAIL["stride"] = float(tail_lambda), int(tail_stride)
    if TAIL["lambda"] < 0 or TAIL["stride"] < 1:
        raise ValueError(f"SM149b: lambda >= 0 and stride >= 1, got {TAIL}")
    for name in names:
        setattr(hooks, name, _Shim(name))
    codec = codec_of_recipe(recipe) if recipe else make_codec(spec, quant)
    layout = training_layout(model)
    hooks.CODECS.clear()
    for stream, info in layout.items():
        hooks.CODECS[stream] = TrainCodec(codec)
        for module in info["time"] + info["channel"]:
            if getattr(module, "fused_shift", False):
                raise ValueError("low-rank QAT cannot run with fused_shift")
            module.state_vq_slot = stream
    return {"recipe": recipe or f"lr{spec}{quant}x8", "codec": repr(codec), "fields": codec.fields(),
            "tail_penalty": {"lambda": TAIL["lambda"], "stride": TAIL["stride"], "eps": TAIL["eps"],
                             "min_answers": TAIL["min_answers"], "rank": codec.rmax,
                             "definition": "sum_{i>r} sigma_i^2 / (sum_i sigma_i^2 + eps) of the pre-codec state per "
                                           "(row, head); loss += lambda x mean over the sampled entries of one update"},
            "prototype": str(Path(__file__).resolve()), "prototype_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
            "gradient": "straight-through (identity through the codec) + the tail penalty's exact gradient along the "
                        "coded trajectory",
            "streams": {str(s): {"wkv_layers": i["wkv_layers"], "heads": i["heads"],
                                 "shift_vectors": len(i["shift_vectors"])} for s, i in layout.items()}}


def uninstall(model):
    from expertium.model import state_vq_hooks as hooks
    from scripts.state_vq import training_layout
    for info in training_layout(model).values():
        for module in info["time"] + info["channel"]:
            module.state_vq_slot = -1
    hooks.CODECS.clear()
    for name, cls in _ORIGINAL.items():
        setattr(hooks, name, cls)
    TAIL["lambda"] = 0.0
    _CURRENT[0] = None


# ------------------------------------------------------------------------------------------------ CPU proofs

def synthetic(B, T, H, K, gen, dtype=torch.float32, nonskip=0.5):
    """Inputs with the RWKV-7 value ranges (SM149's): w in (0, 1), a in (0, 1), unit-norm kd, k = kd * a."""
    r = torch.randn(B, T, H, K, generator=gen, dtype=dtype)
    v = torch.randn(B, T, H, K, generator=gen, dtype=dtype)
    kd = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=gen, dtype=dtype), dim=-1)
    a = torch.sigmoid(torch.randn(B, T, H, K, generator=gen, dtype=dtype))
    w = torch.exp(-torch.exp(-0.5 - torch.nn.functional.softplus(torch.randn(B, T, H, K, generator=gen, dtype=dtype))))
    k = kd * a
    skip = torch.rand(B, T, generator=gen) > nonskip
    skip[:, 0] = False
    return r, k, v, w, a, kd, skip


def load_sm149():
    spec = importlib.util.spec_from_file_location("sm149_qat", SM149_QAT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sm149_qat"] = mod
    spec.loader.exec_module(mod)
    return mod


def _digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().float().contiguous().numpy().tobytes())
    return h.hexdigest()


def b0_synthetic(gen, sm149):
    """B0a: lambda = 0 (pen None) == SM149's LowRankSTEWKV bitwise (outputs, final state, every gradient), K 16 and 32,
    zero and carried start; with lambda = 1 the forward value is still SM149's bitwise (the penalty only adds
    gradients), and the gradients differ."""
    out, ok = {}, True
    for K in (16, 32):
        for carried in (False, True):
            x = synthetic(5, 30, 4, K, gen)
            s0 = None
            if carried:
                pre = synthetic(5, 6, 4, K, gen)
                _, s0 = sm149.LowRankSTEWKV.apply(*pre[:6], pre[6], None, sm149.TrainCodec(sm149.make_codec()))
                s0[0] = 0
            g = torch.randn(5, 30, 4, K, generator=gen)
            res = {}
            for name in ("sm149", "lam0", "lam1"):
                leaves = [t.clone().requires_grad_(True) for t in x[:6]]
                if name == "sm149":
                    o, f = sm149.LowRankSTEWKV.apply(*leaves, x[6], s0, sm149.TrainCodec(sm149.make_codec()))
                else:
                    TAIL["lambda"] = 0.0 if name == "lam0" else 1.0
                    pen = None if name == "lam0" else {"group": _Group(torch.device("cpu")), "stride": 8, "grad": True}
                    o, f = LowRankSTEWKV.apply(*leaves, x[6], s0, TrainCodec(make_codec()), pen)
                (o * g).sum().backward()
                res[name] = (o.detach(), f, [t.grad for t in leaves])
            TAIL["lambda"] = 0.0
            a0, b0, c1 = res["sm149"], res["lam0"], res["lam1"]
            row = {"lam0_out_bitwise": bool(torch.equal(a0[0], b0[0])),
                   "lam0_final_bitwise": bool(torch.equal(a0[1], b0[1])),
                   "lam0_grads_bitwise": all(bool(torch.equal(p, q)) for p, q in zip(a0[2], b0[2])),
                   "lam1_out_bitwise": bool(torch.equal(a0[0], c1[0])),
                   "lam1_grads_differ": any(not torch.equal(p, q) for p, q in zip(a0[2], c1[2]))}
            out[f"K{K}_{'carried' if carried else 'zero_start'}"] = row
            ok &= all(row.values())
    return ok, out


def ste_tail_reference(r, k, v, w, a, kd, skip, codec, s0, phase, stride, lam):
    """Independent reference: autograd through a straight-through loop (stored = coded + (nxt - nxt.detach()), SM149's
    reference) whose loss adds lam x mean(tail), the tail from torch.linalg.svdvals of nxt in float64
    (differentiable), at the same sampled steps and mask. Returns (out, final, the penalty term)."""
    from expertium.model.rwkv_ops import single_timestep
    c = codec.codec
    B, T, H, K = r.shape
    state = r.new_zeros(B, H, K, K) if s0 is None else s0.clone()
    left = r.new_zeros(B, H, K, c.rmax) if s0 is None else carried_left(s0, c.rmax)
    base = torch.zeros(B, dtype=torch.long) if s0 is None else (s0.flatten(1).abs().amax(1) > 0).long() * c.rmax
    counts = base[:, None] + torch.cumsum((~skip).long(), 1)
    outs, tails = [], []
    for t in range(T):
        o, nxt = single_timestep(r[:, t], k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t], state)
        outs.append(o)
        rows = torch.nonzero(~skip[:, t]).flatten()
        if rows.numel() and (phase + t) % stride == 0:
            pr = rows[counts[rows, t] >= TAIL["min_answers"]]
            if pr.numel():
                sv = torch.linalg.svdvals(nxt[pr].double())                # [M, H, K] descending
                tot = sv.square().sum(-1)
                tail = sv[..., c.rmax:].square().sum(-1) / (tot + TAIL["eps"])
                tails.append(tail[tot > 0])
        with torch.no_grad():
            coded = nxt.detach().clone()
            s_rows, l_rows = c.encode(nxt.detach()[rows], left[rows], v[:, t].detach()[rows], counts[rows, t], None)
            coded[rows] = s_rows
            left = left.clone()
            left[rows] = l_rows
        stored = coded + (nxt - nxt.detach())
        state = torch.where(skip[:, t, None, None, None], state, stored)
    pen = lam * torch.cat(tails).mean() if tails else r.new_zeros((), dtype=torch.float64)
    return torch.stack(outs, 1), state, pen


def b1_gradients(gen):
    """B1: the penalty's gradient == autograd through an svdvals (float64) straight-through reference, K 16 and 32,
    zero and carried start, stride 8 and 3; the tail value vs svdvals."""
    out, ok = {}, True
    for K in (16, 32):
        for carried in (False, True):
            for stride in (8, 3):
                x = [t.clone() for t in synthetic(5, 40, 4, K, gen)]
                skip = x[6]
                s0 = None
                if carried:
                    pre = synthetic(5, 6, 4, K, gen)
                    _, s0 = LowRankSTEWKV.apply(*pre[:6], pre[6], None, TrainCodec(make_codec()), None)
                    s0[0] = 0
                g = torch.randn(5, 40, 4, K, generator=gen)
                lam = 1.0
                TAIL["lambda"] = lam
                group = _Group(torch.device("cpu"))
                leaves = [t.clone().requires_grad_(True) for t in x[:6]]
                o1, _ = LowRankSTEWKV.apply(*leaves, skip, s0, TrainCodec(make_codec()),
                                            {"group": group, "stride": stride, "grad": True})
                (o1 * g).sum().backward()
                ours = [t.grad.clone() for t in leaves]
                ours_tail = float(group.sum / group.n)
                leaves0 = [t.clone().requires_grad_(True) for t in x[:6]]
                o0, _ = LowRankSTEWKV.apply(*leaves0, skip, s0, TrainCodec(make_codec()), None)
                (o0 * g).sum().backward()
                leaves2 = [t.clone().requires_grad_(True) for t in x[:6]]
                o2, _, pen = ste_tail_reference(*leaves2, skip, TrainCodec(make_codec()), s0, 0, stride, lam)
                ((o2 * g).sum() + pen).backward()
                ref = [t.grad for t in leaves2]
                TAIL["lambda"] = 0.0
                # the penalty's gradient alone (no output gradient), ours vs the svdvals reference: no float32
                # cancellation against the much larger output gradient
                TAIL["lambda"] = lam
                leaves3 = [t.clone().requires_grad_(True) for t in x[:6]]
                o3, _ = LowRankSTEWKV.apply(*leaves3, skip, s0, TrainCodec(make_codec()),
                                            {"group": _Group(torch.device("cpu")), "stride": stride, "grad": True})
                (o3 * 0).sum().backward()
                leaves4 = [t.clone().requires_grad_(True) for t in x[:6]]
                o4, _, pen4 = ste_tail_reference(*leaves4, skip, TrainCodec(make_codec()), s0, 0, stride, lam)
                ((o4 * 0).sum() + pen4).backward()
                TAIL["lambda"] = 0.0
                rel = max(float((p - q).abs().max() / q.abs().max().clamp(min=1e-30)) for p, q in zip(ours, ref))
                pen_rel = max(float((p.grad - q.grad).abs().max() / q.grad.abs().max().clamp(min=1e-30))
                              for p, q in zip(leaves3, leaves4))
                pen_share = max(float((p - z.grad).abs().max() / p.abs().max()) for p, z in zip(ours, leaves0))
                ref_tail = float(pen.detach() / lam)
                row = {"grad_rel_diff_max": rel, "penalty_only_grad_rel_diff_max": pen_rel,
                       "penalty_share_of_grad_max": pen_share, "tail_mean_ours": ours_tail,
                       "tail_mean_svdvals": ref_tail, "entries": int(group.n),
                       "forward_bitwise_vs_reference": bool(torch.equal(o1, o2.detach())),
                       "finite": all(bool(torch.isfinite(t).all()) for t in ours)}
                out[f"K{K}_{'carried' if carried else 'zero'}_stride{stride}"] = row
                ok &= (rel < 1e-4 and pen_rel < 1e-3 and row["finite"] and row["entries"] > 0
                       and abs(ours_tail - ref_tail) <= 1e-3 * ref_tail + 1e-7)
    return ok, out


def b2_ties_young(gen):
    """B2: finite gradients at singular-value ties, on rank <= 2, zero and tiny states, and on young histories.
    (a) tail_terms on crafted S' (16 x 16, random rotations): sigma_2 = sigma_3 tie, triple tie, near tie, rank 2,
    rank 1, zero, tiny norm: value vs svdvals, gradient finite. (b) LowRankSTEWKV with sparse writes at stride 1 (most
    writes young, n < 3 masked): gradients finite; rows that never reach n >= 3 get gradients bitwise equal to the
    no-penalty run; the others differ."""
    out, ok = {"crafted": {}}, True
    K = 16
    q, _ = torch.linalg.qr(torch.randn(K, K, generator=gen, dtype=torch.float64))
    p, _ = torch.linalg.qr(torch.randn(K, K, generator=gen, dtype=torch.float64))
    cases = {"tie_sigma2_sigma3": [3, 1, 1], "triple_tie": [1, 1, 1], "near_tie": [3, 1, 1 - 1e-6], "rank2": [2, 1],
             "rank1": [1], "zero": [], "tiny_norm": [1e-4, 5e-5, 2e-5]}
    fc = make_codec("2", "f")
    for name, s in cases.items():
        d = torch.zeros(K, dtype=torch.float64)
        if s:
            d[:len(s)] = torch.tensor(s, dtype=torch.float64)
        S = (q @ torch.diag(d) @ p.T).float()[None, None].expand(1, 4, K, K).contiguous()
        left = q[:, :2].float()[None, None].expand(1, 4, K, 2).contiguous()   # the stored span of a rank-2 state
        vv = q[:, 2].float()[None, None].expand(1, 4, K).contiguous()         # the new write's direction
        trunc, _ = fc.encode(S, left, vv, torch.full((1,), 5), None)
        tail, grad, valid = tail_terms(S, trunc, TAIL["eps"])
        sv = torch.linalg.svdvals(S.double())
        ref = sv[..., 2:].square().sum(-1) / (sv.square().sum(-1) + TAIL["eps"])
        row = {"tail": float(tail[0, 0]), "tail_svdvals": float(ref[0, 0]), "valid": bool(valid[0, 0]),
               "grad_finite": bool(torch.isfinite(grad).all()), "grad_abs_max": float(grad.abs().max())}
        out["crafted"][name] = row
        ok &= row["grad_finite"] and abs(row["tail"] - row["tail_svdvals"]) < 1e-5
    x = [t.clone() for t in synthetic(12, 24, 4, K, gen, nonskip=0.12)]
    skip = x[6]
    skip[:4, 1:] = True    # rows 0-3: one write only (n = 1 for ever): fully masked
    g = torch.randn(12, 24, 4, K, generator=gen)
    grads = {}
    TAIL["lambda"] = 1.0
    for name in ("lam0", "lam1"):
        leaves = [t.clone().requires_grad_(True) for t in x[:6]]
        group = _Group(torch.device("cpu"))
        pen = None if name == "lam0" else {"group": group, "stride": 1, "grad": True}
        o, _ = LowRankSTEWKV.apply(*leaves, skip, None, TrainCodec(make_codec()), pen)
        (o * g).sum().backward()
        grads[name] = ([t.grad for t in leaves], group)
    TAIL["lambda"] = 0.0
    counts = torch.cumsum((~skip).long(), 1)
    young = [b for b in range(12) if int(counts[b].max()) < 3]
    row = {"rows_never_n3": young, "entries": int(grads["lam1"][1].n),
           "young_share_of_writes": float(((counts < 3) & ~skip).sum() / (~skip).sum()),
           "grads_finite": all(bool(torch.isfinite(t).all()) for t in grads["lam1"][0]),
           "young_rows_bitwise_equal_no_penalty": all(bool(torch.equal(p[young], q[young]))
                                                     for p, q in zip(grads["lam1"][0], grads["lam0"][0])),
           "other_rows_differ": any(not torch.equal(p, q) for p, q in zip(grads["lam1"][0], grads["lam0"][0]))}
    out["young_histories"] = row
    ok &= (row["grads_finite"] and row["young_rows_bitwise_equal_no_penalty"] and row["other_rows_differ"]
           and len(young) >= 4 and row["entries"] > 0)
    return ok, out


def _setup(chunk):
    protocol = json.loads((SM118_P2 / "protocol.prepared.json").read_text())
    options = json.loads((SM118_P2 / "decay.json").read_text())["options"]
    options = {k: v for k, v in options.items() if k not in ("packed_ipc",)}
    options["chunk_reviews"] = chunk
    os.environ["GPU_BATCH1_DATA_PATH"] = protocol["dataset_path"]
    return protocol, options


def trainer_updates(arm, users, chunk, steps_per_user, lr, measure=False, sm149=None):
    """The real model and trainer path on CPU (as SM149's P4): Small two-head (SM118)'s final weights and Muon state,
    its release options (strided-lane carry, show-time inputs, replay KD targets, alpha 0.5), `steps_per_user` updates
    per user at `lr`. arm: 'sm149' (SM149's hook) or ('lam', x) (this hook at lambda x, stride 8). Returns per-update
    loss + gradient digest, the final weights' digest, and with measure=True the mean tail of every write (stride 1,
    no gradient, carry from zero) on the same batches before and after the updates."""
    from concurrent.futures import ThreadPoolExecutor
    from expertium.variants import build
    from scripts.expertium_options import CatchUp
    from scripts.gpu_r01_prefetch import training_batches
    from rwkv.train_rwkv import CLIP
    protocol, options = _setup(chunk)
    rows = [next(r for r in protocol["rows"] if r["user_id"] == u) for u in users]
    with ThreadPoolExecutor(max_workers=1) as pool:
        items = list(training_batches(rows, chunk, options["training_seed"], pool, options, None, 2))
    per_user, picked = {}, []
    for row, item in items:
        per_user[row["user_id"]] = per_user.get(row["user_id"], 0) + 1
        if per_user[row["user_id"]] <= steps_per_user:
            picked.append((row, item))
    state_dict = torch.load(SM118_P2 / "expertium" / TRACK / "master.pth", map_location="cpu", weights_only=True)
    model = build("w64").float()
    model.load_state_dict(state_dict, strict=True)
    if arm == "sm149":
        sm149.install(model)
    else:
        install(model, tail_lambda=arm[1], tail_stride=8)

    def catch():
        cu = CatchUp(options, {"lr_schedule": {"peak": lr, "warmup_updates": 1, "decay_start": 1.0, "decay_end": 1.0}},
                     rows)
        cu.load_teacher(torch.device("cpu"), torch.float32)
        return cu

    def tail_now():
        cu = catch()
        MEASURE[0] = _Group(torch.device("cpu"))
        with torch.no_grad():
            for row, item in picked:
                batch = item.batch.to(torch.device("cpu"))
                model.train()
                torch.manual_seed(item.seed)
                cu.carry.attach(item, batch, row["user_id"])
                cu.loss(model, batch, 0, (row["user_id"], item.offset, item.reviews, item.seed), batch.labels)
        g, MEASURE[0] = MEASURE[0], None
        return float(g.sum / g.n.clamp(min=1)), int(g.n)

    report = {"arm": str(arm), "users": users, "chunk": chunk, "lr": lr}
    if measure:
        report["tail_before"] = tail_now()
    cu = catch()
    optimizer = cu.optimizer(model)
    optimizer.load_state_dict(torch.load(SM118_P2 / "expertium" / TRACK / "optimizer.pth", map_location="cpu",
                                         weights_only=True))
    steps, done = [], 0
    for row, item in picked:
        u = row["user_id"]
        batch = item.batch.to(torch.device("cpu"))
        model.train()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(item.seed)
        cu.carry.attach(item, batch, u)
        t0 = time.perf_counter()
        stats = cu.loss(model, batch, done, (u, item.offset, item.reviews, item.seed), batch.labels)
        stats.average_loss.backward()
        seconds = time.perf_counter() - t0
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        steps.append({"user": u, "offset": item.offset, "reviews": item.reviews, "loss": float(stats.average_loss),
                      "grad_sha256": _digest([stats.average_loss] + grads),
                      "grads_finite": all(bool(torch.isfinite(t).all()) for t in grads), "seconds": round(seconds, 2)})
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        done += item.reviews
    report["steps"] = steps
    report["weights_sha256"] = _digest(list(model.parameters()))
    if measure:
        report["tail_after"] = tail_now()
    if arm == "sm149":
        sm149.uninstall(model)
    else:
        uninstall(model)
    LOG["pending"] = []
    return report


def b0_trainer(users, chunk, steps, lr, sm149):
    """B0b: the real trainer path, lambda = 0 vs SM149's hook: every update's loss and gradient digest, and the final
    weights after the Muon steps, bitwise equal. Plus lambda = 1: SM149's first loss bitwise (the forward value),
    different and finite gradients."""
    a = trainer_updates("sm149", users, chunk, steps, lr, sm149=sm149)
    b = trainer_updates(("lam", 0.0), users, chunk, steps, lr)
    c = trainer_updates(("lam", 1.0), users, chunk, steps, lr)
    same = [x["grad_sha256"] == y["grad_sha256"] and x["loss"] == y["loss"] for x, y in zip(a["steps"], b["steps"])]
    out = {"updates": len(same), "lam0_updates_bitwise": all(same),
           "lam0_weights_bitwise": a["weights_sha256"] == b["weights_sha256"],
           "lam1_first_loss_bitwise": a["steps"][0]["loss"] == c["steps"][0]["loss"],
           "lam1_first_grads_differ": a["steps"][0]["grad_sha256"] != c["steps"][0]["grad_sha256"],
           "lam1_grads_finite": all(s["grads_finite"] for s in c["steps"]),
           "seconds_per_update": {name: sum(s["seconds"] for s in r["steps"]) / len(r["steps"])
                                  for name, r in (("sm149", a), ("lam0", b), ("lam1", c))},
           "losses": {"sm149": [s["loss"] for s in a["steps"]], "lam1": [s["loss"] for s in c["steps"]]}}
    ok = (out["lam0_updates_bitwise"] and out["lam0_weights_bitwise"] and out["lam1_first_loss_bitwise"]
          and out["lam1_first_grads_differ"] and out["lam1_grads_finite"] and len(same) > 0)
    return ok, out


def b3_direction(users, chunk, steps, lrs):
    """B3: the tail moves the right way. From Small two-head (SM118)'s end, the same updates at lambda 0, 1 and 3, then
    the mean tail of every write of the same batches (stride 1, no gradient). Pass: at every LR tried, lambda 1 ends
    below lambda 0 and lambda 3 below lambda 1."""
    out, ok = {}, True
    for lr in lrs:
        res = {}
        for lam in (0.0, 1.0, 3.0):
            rep = trainer_updates(("lam", lam), users, chunk, steps, lr, measure=True)
            res[f"lambda_{lam:g}"] = {"tail_before": rep["tail_before"][0], "tail_after": rep["tail_after"][0],
                                      "entries": rep["tail_after"][1], "updates": len(rep["steps"]),
                                      "loss_mean": sum(s["loss"] for s in rep["steps"]) / len(rep["steps"])}
        t0, t1, t3 = (res[f"lambda_{x}"]["tail_after"] for x in ("0", "1", "3"))
        res["after_lambda1_vs_lambda0_relative"] = (t1 - t0) / t0
        res["after_lambda3_vs_lambda0_relative"] = (t3 - t0) / t0
        out[f"lr_{lr:g}"] = res
        ok &= t1 < t0 and t3 < t1
    return ok, out


def b4_cost(gen):
    """B4 (report): CPU seconds of forward + backward and the memory the penalty keeps for the backward, lambda 0 vs 1
    (stride 8), on split shapes of the SM32b timing sample (rows x steps, 31% writes), K = 16."""
    shapes = [(512, 16), (2048, 4), (128, 64)]
    out = []
    for B, T in shapes:
        x = synthetic(B, T, 4, 16, gen, nonskip=0.31)
        row = {"rows": B, "steps": T, "before_states_mib": round(B * T * 4 * 256 * 4 / 2 ** 20, 2)}
        for lam in (0.0, 1.0):
            TAIL["lambda"] = lam
            leaves = [t.clone().requires_grad_(True) for t in x[:6]]
            pen = {"group": _Group(torch.device("cpu")), "stride": 8, "grad": True} if lam else None
            t0 = time.perf_counter()
            o, _ = LowRankSTEWKV.apply(*leaves, x[6], None, TrainCodec(make_codec()), pen)
            t1 = time.perf_counter()
            extra = sum(gp.numel() * gp.element_size() + rr.numel() * 8 for rr, gp in o.grad_fn.inject.values())
            o.sum().backward()
            t2 = time.perf_counter()
            row[f"lam{lam:g}"] = {"forward_s": round(t1 - t0, 3), "backward_s": round(t2 - t1, 3),
                                  "penalty_saved_mib": round(extra / 2 ** 20, 2)}
        TAIL["lambda"] = 0.0
        row["fwd_bwd_ratio_lam1_over_lam0"] = round(
            (row["lam1"]["forward_s"] + row["lam1"]["backward_s"]) / (row["lam0"]["forward_s"] + row["lam0"]["backward_s"]), 2)
        out.append(row)
    return True, out


def main_cpu(args):
    os.environ["RWKV_NO_JIT"] = "1"
    os.environ["GPU_BATCH1_DATA_PATH"] = DATA
    os.environ["RWKV_DELETED_CARDS"] = "neighbour"
    torch.set_num_threads(4)
    gen = torch.Generator().manual_seed(1492)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "scripts", "rwkv", "expertium"], cwd=CODE,
                           capture_output=True, text=True).stdout.strip()
    sm149 = load_sm149()
    report = {"code_tree": str(CODE), "code_head": head, "code_dirty": bool(dirty), "torch": torch.__version__,
              "sm149_hook": str(SM149_QAT), "sm149_hook_sha256": hashlib.sha256(SM149_QAT.read_bytes()).hexdigest(),
              "this_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "checks": {}}
    users = [int(u) for u in args.users.split(",")]
    lrs = [float(x) for x in args.direction_lrs.split(",")]
    only = set(args.only.split(",")) if args.only else None
    started = time.perf_counter()
    for name, fn in (("B0a_lambda0_equals_sm149_synthetic", lambda: b0_synthetic(gen, sm149)),
                     ("B0b_lambda0_equals_sm149_trainer", lambda: b0_trainer(users, args.chunk, args.steps, args.lr,
                                                                             sm149)),
                     ("B1_penalty_gradient_vs_svdvals_reference", lambda: b1_gradients(gen)),
                     ("B2_ties_young_zero_states_finite", lambda: b2_ties_young(gen)),
                     ("B3_tail_moves_down", lambda: b3_direction(users, args.chunk, args.direction_steps, lrs)),
                     ("B4_cpu_cost", lambda: b4_cost(gen))):
        if only and name.split("_")[0] not in only:
            continue
        t0 = time.perf_counter()
        ok, detail = fn()
        report["checks"][name] = {"pass": bool(ok), "seconds": round(time.perf_counter() - t0, 1), "detail": detail}
        print(f"[sm149b] {name}: {'OK' if ok else 'FAIL'} ({time.perf_counter() - t0:.0f} s)", flush=True)
    report["all_ok"] = (all(c["pass"] for c in report["checks"].values()) and not report["code_dirty"]
                        and head.startswith(CODE_COMMIT) and only is None)
    report["seconds"] = round(time.perf_counter() - started, 1)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[sm149b] CPU proofs {'ALL OK' if report['all_ok'] else 'NOT OK'} -> {args.out}")
    return 0 if report["all_ok"] else 1


def main():
    p = ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cpu")
    c.add_argument("--users", default="4745,2308")
    c.add_argument("--chunk", type=int, default=256)
    c.add_argument("--steps", type=int, default=3)
    c.add_argument("--lr", type=float, default=7e-5)
    c.add_argument("--direction-steps", type=int, default=8)
    c.add_argument("--direction-lrs", default="7e-5,7e-4")
    c.add_argument("--only", default="", help="comma list of B0a,B0b,B1,B2,B3,B4 (debugging; never all_ok)")
    c.add_argument("--out", default=str(HERE / "check-cpu-sm149b.json"))
    args = p.parse_args()
    return main_cpu(args)


if __name__ == "__main__":
    sys.exit(main())
