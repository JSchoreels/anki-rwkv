#!/usr/bin/env python
"""Rank-2 low-rank state cool-down (SM149): prototype training hook (straight-through estimator), CPU proofs and a
short GPU micro-benchmark. Design: artifacts/small-model/sm149/design.md.

The code tree is the low-rank int8 states experiment (SM148)'s worktree (its codec scripts/state_lowrank.py, the
trainer, the state-VQ QAT hooks of SM32b / SM32c), imported read only from SM149_CODE (default
C:/Users/jonat/workspace/wt/sm148, pinned commit in CODE_COMMIT). Nothing in that tree is edited: this file installs
its hook by setting the card / note mixers' `state_vq_slot` (the SM32b switch) and replacing the three autograd class
names of expertium.model.state_vq_hooks with dispatching shims at run time (works with the scripted trainer model). The run tree gets the same code as a
small source change (design.md "Implementation").

Placement = the SM148 scorer's (scripts/state_lowrank.lowrank_loop): per card / note review the answer's output uses
the float update S' = S G + v k^T; the STORED state (what the entity's next review starts from, in this window or
carried to the next one) is the codec's reconstruction: rank min(n, r) truncation of S' through the factored solve,
int8 factor vectors with a float16 scale each. Token-shift vectors read from an earlier review are REF's int8 vectors
(int_row_quant(8 | F16_SCALE)). Gradient: straight-through (the codec is the identity in the backward), i.e. the plain
WKV backward along the coded trajectory (expertium.model.state_vq_hooks.ste_backward; on CUDA with K = 16 the fused
SM32b backward kernel).

  python sm149_qat.py cpu   [--out check-cpu-sm149.json]       CPU proofs P1-P6 (about 70 s, 4 threads)
  python sm149_qat.py bench [--out gpu/bench-sm149.json]       GPU micro-benchmark, <= 10 min (queue ticket)
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
CODE = Path(os.environ.get("SM149_CODE", "C:/Users/jonat/workspace/wt/sm148"))
CODE_COMMIT = "cea0f931"          # origin/sm148 head when the proofs ran (the SM148 scoring tree)
sys.path.insert(0, str(CODE))

import torch  # noqa: E402

W = Path("C:/Users/jonat/workspace/wt")
DATA = r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"
# the release cool-down of Small two-head (SM118): its final weights / Muon state and its own options
SM118_P2 = W / "sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2"
TRACK = "E89-F65e-F77a"
TIMING = W / "decay-resume/artifacts/small-model/state-vq/qat/timing.json"   # SM32b: split shapes of 40 32k updates


# ------------------------------------------------------------------------------------------------ the hook

def make_codec(spec="2", quant="q8"):
    """SM148's codec `w8+lr<spec><quant>x8` with REF's shift / row bits filled in (scripts/small_model_heldout)."""
    return codec_of_recipe(f"lr{spec}{quant}x8")


def codec_of_recipe(recipe):
    """'lr2q8x8' (the SM148 label's state part) -> the configured LowRankCodec; R tiers and float shifts refused.
    REF's (`w8+s8hx8`) bits are set directly (int8 rows / shift vectors with float16 scales), as the SM148 scorer's
    parse_variant fills them; importing scripts/small_model_heldout in the trainer would patch the fork model (P1
    checks that both give the same codec)."""
    from scripts.small_model_wkv import F16_SCALE
    from scripts.state_lowrank import LowRankCodec, parse_sm148
    codec = parse_sm148(recipe)
    if codec is not None:
        codec.rows_bits, codec.shift_bits = 8 | F16_SCALE, 8 | F16_SCALE
    if not isinstance(codec, LowRankCodec) or codec.has_rows or codec.float_shift:
        raise ValueError(f"SM149 training recipe {recipe!r}: a pure rank-r lr<spec>q8x8 / lr<spec>fx8 codec")
    return codec


class TrainCodec:
    """CODECS entry of a low-rank stream: what state_vq_hooks' shift functions call (.shift, .learned) plus the SM148
    codec for the WKV state."""
    lowrank = True
    learned = None
    impl = "torch"

    def __init__(self, codec):
        from scripts.state_vq import int_row_quant
        self.codec = codec
        self._shift = int_row_quant(codec.shift_bits)

    def shift(self, x, role):
        return self._shift(x)

    def sync(self):
        pass


_OMEGA = {}
CHUNK = 16384   # matrices per batched linear-algebra call (cusolver's batch limit, scripts/state_lowrank.EIGH_CHUNK)


def _omega(K, m, device):
    key = (K, m, str(device))
    if key not in _OMEGA:
        g = torch.Generator().manual_seed(149)
        _OMEGA[key] = torch.randn(K, m, generator=g, dtype=torch.float64).to(device)
    return _OMEGA[key]


def carried_left(state, rmax):
    """Left factors of carried stored states [N, H, K, K] (each of rank <= rmax, the codec's output at the end of the
    previous window): a randomised range finder, float64. Y = S Omega with Omega a fixed seeded K x rmax Gaussian, then
    the Householder QR basis Q of Y (K x rmax). rank(S Omega) = rank(S) for a generic Omega, so span(Q) holds col(S)
    (to float64 rounding however small sigma_2 / sigma_1 is; for a rank-1 S the second column is any orthonormal
    direction, which is harmless). Only the span of `left` is used (the factored solve orthonormalises [left, v]), so
    the next write's truncation is the scorer's, which keeps the factors of the same history in one window. No eigen
    problem on the GPU: a float64 K x K batched eigh of a window's carried rows ran out of GPU memory (9.8 GB), and so
    did a 4 x 4 one (8.4 GB of cusolver workspace for 16,384 matrices). Chunks of CHUNK matrices."""
    N, H, K, _ = state.shape
    left = state.new_zeros(N, H, K, rmax)
    nz = (state.abs().flatten(2).amax(-1) > 0) & torch.isfinite(state).flatten(2).all(-1)
    if bool(nz.any()):
        S = state[nz]
        om = _omega(K, rmax, state.device)
        parts = []
        for i in range(0, S.shape[0], CHUNK):
            Q, _ = torch.linalg.qr(S[i:i + CHUNK].double() @ om)   # [M, K, rmax]
            parts.append(Q.to(state.dtype))
        left[nz] = torch.cat(parts)
    return left


class LowRankSTEWKV(torch.autograd.Function):
    """Per-step card / note recurrence whose stored state goes through the SM148 codec after every non-skipped step,
    from a carried state s0 [B, H, K, K] (None = zeros). Forward value = the codec's reconstruction exactly (the
    scorer's lowrank_loop); backward = straight-through. Returns (out [B, T, H, K], last stored state, detached)."""

    @staticmethod
    def forward(ctx, r, k, v, w, a, kd, skip, s0, codec):
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
        for t in range(T):
            if kernel:
                before[:, t] = state
            else:
                before[t] = state
            out[:, t], nxt = single_timestep(r[:, t], k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t], state)
            rows = torch.nonzero(~skip[:, t]).flatten()
            if rows.numel():
                s_new, l_new = c.encode(nxt[rows], left[rows], v[rows, t], counts[rows, t], None)
                state[rows] = s_new
                left[rows] = l_new
        ctx.kernel = kernel
        ctx.save_for_backward(r, k, v, w, a, kd, skip, before)
        ctx.mark_non_differentiable(state)
        return out, state

    @staticmethod
    def backward(ctx, grad_out, grad_final):
        from expertium.model.state_vq_hooks import ste_backward
        r, k, v, w, a, kd, skip, before = ctx.saved_tensors
        if ctx.kernel:
            from scripts.state_vq_cuda import wkv_backward
            grads = wkv_backward(r, k, v, w, a, kd, skip, before, grad_out)
        else:
            grads = ste_backward(r, k, v, w, a, kd, skip, before, grad_out)
        return (*grads, None, None, None)


FORCE_TORCH_BACKWARD = False
_ORIGINAL = {}
CALLS = {"window": 0, "carried": 0}   # hook calls (proof read-out)


class _Shim:
    """Stands in for one of state_vq_hooks' autograd classes. The hook functions (vq_wkv, vq_wkv_state) and the
    scripted mixers that call them (@torch.jit.ignore, so they run as Python) look these class names up in the module
    at call time: a low-rank codec goes to LowRankSTEWKV, any other codec to the original class unchanged."""

    def __init__(self, name):
        self.name = name

    def apply(self, *args):
        codec = args[-1]
        if not getattr(codec, "lowrank", False):
            return _ORIGINAL[self.name].apply(*args)
        r, k, v, w, a, kd, skip = args[:7]
        if self.name == "VQStateWKVLearned":            # (r, k, v, w, a, kd, skip, s0, books, codec): carried path
            s0 = args[7]
            CALLS["window"] += 1
            CALLS["carried"] += s0 is not None
            return LowRankSTEWKV.apply(r, k, v, w, a, kd, skip, s0, codec)
        CALLS["window"] += 1                             # (r, k, v, w, a, kd, skip, codec): one window from zero
        return LowRankSTEWKV.apply(r, k, v, w, a, kd, skip, None, codec)[0]


def install(model, spec="2", quant="q8", recipe=None):
    """Route the card / note mixers of an expertium-class model (w64, Tiny, the fork architecture in training) through
    the low-rank STE hook: set their state_vq_slot (the SM32b switch) and put the low-rank codec in CODECS; the
    autograd classes of state_vq_hooks are replaced by dispatching shims. Returns a description."""
    from expertium.model import state_vq_hooks as hooks
    from scripts.state_vq import training_layout
    names = ("VQStateWKV", "VQStateWKVKernel", "VQStateWKVLearned")
    if not _ORIGINAL:
        for name in names:
            _ORIGINAL[name] = getattr(hooks, name)
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
            "prototype": str(Path(__file__).resolve()), "prototype_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(), "gradient": "straight-through (identity through the codec)",
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


# ------------------------------------------------------------------------------------------------ CPU proofs

def synthetic(B, T, H, K, gen, dtype=torch.float32, nonskip=0.5):
    """Inputs with the RWKV-7 value ranges: w in (0, 1), a in (0, 1), unit-norm kd, k = kd * a."""
    r = torch.randn(B, T, H, K, generator=gen, dtype=dtype)
    v = torch.randn(B, T, H, K, generator=gen, dtype=dtype)
    kd = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=gen, dtype=dtype), dim=-1)
    a = torch.sigmoid(torch.randn(B, T, H, K, generator=gen, dtype=dtype))
    w = torch.exp(-torch.exp(-0.5 - torch.nn.functional.softplus(torch.randn(B, T, H, K, generator=gen, dtype=dtype))))
    k = kd * a
    skip = torch.rand(B, T, generator=gen) > nonskip
    skip[:, 0] = False
    return r, k, v, w, a, kd, skip


def ste_reference(r, k, v, w, a, kd, skip, codec, s0=None):
    """Autograd through a straight-through loop: stored = nxt + (codec(nxt) - nxt).detach() (the forward is the
    codec's value, the gradient the identity). Same codec calls as LowRankSTEWKV."""
    from expertium.model.rwkv_ops import single_timestep
    c = codec.codec
    B, T, H, K = r.shape
    state = r.new_zeros(B, H, K, K) if s0 is None else s0.clone()
    left = r.new_zeros(B, H, K, c.rmax) if s0 is None else carried_left(s0, c.rmax)
    base = torch.zeros(B, dtype=torch.long) if s0 is None else (s0.flatten(1).abs().amax(1) > 0).long() * c.rmax
    counts = base[:, None] + torch.cumsum((~skip).long(), 1)
    outs = []
    for t in range(T):
        o, nxt = single_timestep(r[:, t], k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t], state)
        outs.append(o)
        rows = torch.nonzero(~skip[:, t]).flatten()
        with torch.no_grad():   # the codec on the written rows only, as the scorer and LowRankSTEWKV call it
            coded = nxt.detach().clone()
            s_rows, l_rows = c.encode(nxt.detach()[rows], left[rows], v[:, t].detach()[rows], counts[rows, t], None)
            coded[rows] = s_rows
            left = left.clone()
            left[rows] = l_rows
        stored = coded + (nxt - nxt.detach())   # value = coded exactly (x - x = 0), gradient = identity
        state = torch.where(skip[:, t, None, None, None], state, stored)
    return torch.stack(outs, 1), state


def p1_forward_equals_scorer(gen):
    """P1: LowRankSTEWKV forward == SM148's scorer loop (scripts/state_lowrank.lowrank_loop), bitwise, K 16 and 32."""
    from scripts.state_lowrank import lowrank_loop
    from expertium.model.rwkv_ops import single_timestep
    out = {}
    for K in (16, 32):
        x = synthetic(6, 40, 4, K, gen)
        codec = TrainCodec(make_codec("2", "q8"))
        ours, final = LowRankSTEWKV.apply(*x[:6], x[6], None, codec)
        ref = lowrank_loop(codec.codec, *x[:6], x[6], single_timestep, None)
        float_out = lowrank_loop(make_codec("16", "f"), *x[:6], x[6], single_timestep, None)
        out[f"K{K}"] = {"max_abs_diff_vs_scorer": float((ours - ref).abs().max()),
                        "bitwise": bool(torch.equal(ours, ref)),
                        "codec_active_max_abs_vs_rank16_float": float((ours - float_out).abs().max()),
                        "final_rank_max": int(max(int(torch.linalg.matrix_rank(final[i, j].double(), atol=1e-5))
                                                  for i in range(final.shape[0]) for j in range(final.shape[1])))}
    from scripts.small_model_heldout import parse_variant
    scorer = parse_variant("w8+lr2q8x8")[1]
    ours = codec_of_recipe("lr2q8x8")
    out["codec_equals_scorer_parse"] = {"ours": ours.fields() | {"rows_bits": ours.rows_bits},
                                        "scorer": scorer.fields() | {"rows_bits": scorer.rows_bits}}
    same = ours.fields() == scorer.fields() and ours.rows_bits == scorer.rows_bits
    ok = same and all(v["bitwise"] and v["codec_active_max_abs_vs_rank16_float"] > 1e-3 and v["final_rank_max"] <= 2
                      for k, v in out.items() if k.startswith("K"))
    return ok, out


def p2_gradients(gen):
    """P2: the explicit STE backward == autograd through the straight-through loop (float32, K 16 and 32, with and
    without a carried state), finite, non-zero."""
    out = {}
    ok = True
    for K in (16, 32):
        for carried in (False, True):
            x = [t.clone() for t in synthetic(5, 24, 4, K, gen)]
            skip = x[6]
            codec = TrainCodec(make_codec("2", "q8"))
            s0 = None
            if carried:   # a carried stored state of rank 2 (the codec's output of a short history)
                pre = synthetic(5, 6, 4, K, gen)
                _, s0 = LowRankSTEWKV.apply(*pre[:6], pre[6], None, codec)
                s0[0] = 0   # one row with nothing carried
            leaves = [t.clone().requires_grad_(True) for t in x[:6]]
            o1, f1 = LowRankSTEWKV.apply(*leaves, skip, s0, codec)
            g = torch.randn(o1.shape, generator=gen)
            (o1 * g).sum().backward()
            ours = [t.grad.clone() for t in leaves]
            leaves2 = [t.clone().requires_grad_(True) for t in x[:6]]
            o2, f2 = ste_reference(*leaves2, skip, codec, s0)
            (o2 * g).sum().backward()
            ref = [t.grad for t in leaves2]
            rel = max(float((p - q).abs().max() / q.abs().max().clamp(min=1e-30)) for p, q in zip(ours, ref))
            fwd = float((o1 - o2.detach()).abs().max())
            fin = float((f1 - f2.detach()).abs().max())
            finite = all(bool(torch.isfinite(t).all()) for t in ours)
            nonzero = all(float(t.abs().sum()) > 0 for t in ours)
            name = f"K{K}_{'carried' if carried else 'zero_start'}"
            out[name] = {"grad_rel_diff_max": rel, "forward_max_abs_diff": fwd, "final_state_max_abs_diff": fin,
                         "finite": finite, "nonzero": nonzero}
            ok &= rel < 1e-4 and fwd < 1e-5 and fin < 1e-5 and finite and nonzero
    return ok, out


def p3_carry_split(gen):
    """P3: carried windows == one window. A history cut into 3 windows (each starting from the previous window's last
    stored state, left factors rebuilt from it) gives the outputs of the uncut run (float32 rounding only)."""
    out = {}
    ok = True
    for K in (16, 32):
        x = synthetic(8, 60, 4, K, gen)
        codec = TrainCodec(make_codec("2", "q8"))
        whole, whole_final = LowRankSTEWKV.apply(*x[:6], x[6], None, codec)
        parts, s0 = [], None
        for lo, hi in ((0, 17), (17, 41), (41, 60)):
            o, s0 = LowRankSTEWKV.apply(*[t[:, lo:hi].contiguous() for t in x[:6]], x[6][:, lo:hi].contiguous(),
                                         s0, codec)
            parts.append(o)
        cut = torch.cat(parts, 1)
        rel = float((cut - whole).abs().max() / whole.abs().max())
        out[f"K{K}"] = {"out_rel_diff_max": rel,
                        "final_rel_diff_max": float((s0 - whole_final).abs().max() / whole_final.abs().max())}
        # int8 factor codes may flip on a rounding tie after a rebuilt basis (float32): allow the discrete-codec slack
        ok &= rel < 1e-4
    return ok, out


# proof models: tag -> (run folder of the release model, track, architecture, protocol file)
PROOF_MODELS = {
    "sm118": (SM118_P2, TRACK, "w64", "protocol.prepared.json"),
    "sm91s2": (W / "sm91s2/artifacts/small-model/sm91s2-big-showtime", "big-showtime", "fork", "protocol.json"),
}


def _setup(tag, chunk):
    folder, _, _, proto = PROOF_MODELS[tag]
    protocol = json.loads((folder / proto).read_text())
    options = json.loads((folder / "decay.json").read_text())["options"]
    options = {k: v for k, v in options.items() if k not in ("packed_ipc",)}
    options["chunk_reviews"] = chunk
    os.environ["GPU_BATCH1_DATA_PATH"] = protocol["dataset_path"]
    return protocol, options


def p4_real_updates(tag, users, chunk, steps_per_user, lr):
    """P4: the real model and trainer path on CPU. Small two-head (SM118)'s final weights and Muon state, the release
    cool-down's own options (strided-lane carry, show-time inputs, the replay targets of the show-time big model
    (SM91s2), KD alpha 0.5 of the decay), 2,048-review windows so that every user has carried windows. Per user: the
    same updates with the hook off (control) and on (QAT): loss finite, card / note gradients finite and non-zero,
    QAT loss != control loss, hook calls counted, then optimizer steps at the given LR stay finite."""
    from concurrent.futures import ThreadPoolExecutor
    from expertium.variants import build
    from scripts.expertium_options import CatchUp
    from scripts.gpu_r01_prefetch import training_batches
    from rwkv.train_rwkv import CLIP
    folder, track, arch, _ = PROOF_MODELS[tag]
    protocol, options = _setup(tag, chunk)
    rows = [next(r for r in protocol["rows"] if r["user_id"] == u) for u in users]
    master_path = folder / "expertium" / track / "master.pth"
    state_dict = torch.load(master_path, map_location="cpu", weights_only=True)
    report = {"model": tag, "architecture": arch, "users": users, "chunk_reviews": chunk, "checkpoint": str(master_path),
              "reviews": {r["user_id"]: r["train_range"][1] - r["train_range"][0] for r in rows}, "arms": {}}
    calls = CALLS
    for arm in ("control", "qat"):
        model = build(arch).float()
        model.load_state_dict(state_dict, strict=True)
        info = None
        if arm == "qat":
            CALLS.update(window=0, carried=0)
            info = install(model)
        catch_up = CatchUp(options, {"lr_schedule": {"peak": lr, "warmup_updates": 1, "decay_start": 1.0,
                                                    "decay_end": 1.0}}, rows)
        catch_up.load_teacher(torch.device("cpu"), torch.float32)
        optimizer = catch_up.optimizer(model)
        optimizer.load_state_dict(torch.load(folder / "expertium" / track / "optimizer.pth", map_location="cpu",
                                             weights_only=True))
        steps = []
        names = [n for n, _ in model.named_parameters()]
        card_note = [n for n in names if n.startswith(("rwkv_modules.0.", "rwkv_modules.2."))]
        done = 0
        with ThreadPoolExecutor(max_workers=1) as pool:
            items = list(training_batches(rows, chunk, options["training_seed"], pool, options, None, 2))
        per_user = {}
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
            cn = [params[n].grad for n in card_note if params[n].grad is not None]
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            done += item.reviews
            steps.append({"user": u, "offset": item.offset, "reviews": item.reviews,
                          "loss": float(stats.average_loss), "grad_norm": float(norm),
                          "card_note_grads_finite": all(bool(torch.isfinite(g).all()) for g in cn),
                          "card_note_grad_abs_sum": float(sum(g.abs().sum() for g in cn)),
                          "card_note_grad_tensors": len(cn), "seconds": round(seconds, 2)})
        weights_finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
        report["arms"][arm] = {"steps": steps, "weights_finite_after_steps": weights_finite, "hook": info,
                               "carry": catch_up.carry.describe() if catch_up.carry else None}
        if arm == "qat":
            report["arms"][arm]["hook_calls"] = dict(calls)
            uninstall(model)
    c, q = report["arms"]["control"]["steps"], report["arms"]["qat"]["steps"]
    ok = (len(c) == len(q) > 0
          and all(abs(s["loss"]) < 1e3 and s["card_note_grads_finite"] and s["card_note_grad_abs_sum"] > 0
                  and s["card_note_grad_tensors"] > 0 for s in c + q)
          and report["arms"]["control"]["weights_finite_after_steps"] and report["arms"]["qat"]["weights_finite_after_steps"]
          and q[0]["loss"] != c[0]["loss"] and calls["window"] > 0 and calls["carried"] > 0)
    report["first_update_loss_delta_qat_minus_control"] = q[0]["loss"] - c[0]["loss"]
    report["seconds_per_update"] = {"control": sum(s["seconds"] for s in c) / len(c),
                                    "qat": sum(s["seconds"] for s in q) / len(q)}
    return ok, report


def p5_option_absent(users, chunk):
    """P5: hook installed and removed (uninstall) = stock path: loss and gradients of one update bitwise equal to a
    never-hooked model (the prototype leaves no trace; the run-tree port is proven the same way, design.md)."""
    from concurrent.futures import ThreadPoolExecutor
    from expertium.variants import build
    from scripts.expertium_options import CatchUp
    from scripts.gpu_r01_prefetch import training_batches
    protocol, options = _setup("sm118", chunk)
    rows = [next(r for r in protocol["rows"] if r["user_id"] == u) for u in users[:1]]
    state_dict = torch.load(SM118_P2 / "expertium" / TRACK / "master.pth", map_location="cpu", weights_only=True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        row, item = next(iter(training_batches(rows, chunk, options["training_seed"], pool, options, None, 2)))
    digests = []
    for hooked in (False, True):
        model = build("w64").float()
        model.load_state_dict(state_dict, strict=True)
        if hooked:
            install(model)
            uninstall(model)
        catch_up = CatchUp(options, {}, rows)
        catch_up.load_teacher(torch.device("cpu"), torch.float32)
        batch = item.batch.to(torch.device("cpu"))
        model.train()
        torch.manual_seed(item.seed)
        catch_up.carry.attach(item, batch, row["user_id"])
        stats = catch_up.loss(model, batch, 0, (row["user_id"], item.offset, item.reviews, item.seed), batch.labels)
        stats.average_loss.backward()
        h = hashlib.sha256()
        for t in [stats.average_loss] + [p.grad for p in model.parameters() if p.grad is not None]:
            h.update(t.detach().float().contiguous().numpy().tobytes())
        digests.append(h.hexdigest())
    return digests[0] == digests[1], {"user": row["user_id"], "digests": digests}


def p6_cpu_step_cost(gen):
    """P6 (report only): CPU seconds of LowRankSTEWKV forward + backward vs the plain per-step reference loop with
    autograd, on split shapes of the SM32b 32k timing sample (rows x steps), K = 16."""
    from expertium.model.rwkv_ops import reference_rwkv7
    shapes = [(512, 16), (2048, 4), (128, 64)]
    out = []
    codec = TrainCodec(make_codec("2", "q8"))
    for B, T in shapes:
        x = synthetic(B, T, 4, 16, gen, nonskip=0.31)
        leaves = [t.clone().requires_grad_(True) for t in x[:6]]
        t0 = time.perf_counter()
        o, _ = LowRankSTEWKV.apply(*leaves, x[6], None, codec)
        t1 = time.perf_counter()
        o.sum().backward()
        t2 = time.perf_counter()
        leaves = [t.clone().requires_grad_(True) for t in x[:6]]
        t3 = time.perf_counter()
        o = reference_rwkv7(*leaves, x[6])
        o.sum().backward()
        t4 = time.perf_counter()
        out.append({"rows": B, "steps": T, "qat_forward_s": round(t1 - t0, 3), "qat_backward_s": round(t2 - t1, 3),
                    "plain_reference_fwd_bwd_s": round(t4 - t3, 3)})
    return True, out


def main_cpu(args):
    os.environ["RWKV_NO_JIT"] = "1"
    os.environ["GPU_BATCH1_DATA_PATH"] = DATA     # read at import by scripts/gpu_batch1
    os.environ["RWKV_DELETED_CARDS"] = "neighbour"   # the release cool-down's deleted-card mode (replay targets)
    torch.set_num_threads(4)
    gen = torch.Generator().manual_seed(149)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "scripts", "rwkv", "expertium"], cwd=CODE,
                           capture_output=True, text=True).stdout.strip()
    report = {"code_tree": str(CODE), "code_head": head, "code_dirty": bool(dirty), "torch": torch.__version__,
              "checks": {}}
    users = [int(u) for u in args.users.split(",")]
    started = time.perf_counter()
    for name, fn in (("P1_forward_equals_sm148_scorer", lambda: p1_forward_equals_scorer(gen)),
                     ("P2_ste_gradients", lambda: p2_gradients(gen)),
                     ("P3_carried_windows", lambda: p3_carry_split(gen)),
                     ("P4_small_two_head_sm118_updates",
                      lambda: p4_real_updates("sm118", users, args.chunk, args.steps, args.lr)),
                     ("P4b_big_two_head_sm91s2_updates",
                      lambda: p4_real_updates("sm91s2", [int(u) for u in args.big_users.split(",")], args.chunk,
                                              args.steps, args.lr / 2)),
                     ("P5_unhooked_is_stock", lambda: p5_option_absent(users, args.chunk)),
                     ("P6_cpu_step_cost", lambda: p6_cpu_step_cost(gen))):
        t0 = time.perf_counter()
        ok, detail = fn()
        report["checks"][name] = {"pass": bool(ok), "seconds": round(time.perf_counter() - t0, 1), "detail": detail}
        print(f"[sm149] {name}: {'OK' if ok else 'FAIL'} ({time.perf_counter() - t0:.0f} s)", flush=True)
    report["all_ok"] = all(c["pass"] for c in report["checks"].values()) and not report["code_dirty"]
    report["seconds"] = round(time.perf_counter() - started, 1)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[sm149] CPU proofs {'ALL OK' if report['all_ok'] else 'NOT OK'} -> {args.out}")
    return 0 if report["all_ok"] else 1


# ------------------------------------------------------------------------------------------------ GPU micro-benchmark

def main_bench(args):
    """Seconds per 32k update of the card / note recurrences on CUDA: the plain WKV kernel (forward + backward) vs the
    QAT path, on the split shapes (rows x steps, non-skipped share) of the SM32b timing sample, K = 16 (Small: 2 card
    + 1 note layers; Tiny: 1 + 1) and K = 32 (Big: 3 + 2, at half the reviews per update). Checks first that the CUDA
    forward equals the CPU forward (discrete-codec rule) and the kernel backward equals the torch STE backward.
    Stops itself after --max-seconds. Writes the projection per model (design.md "Estimates")."""
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(0.8)
    gen = torch.Generator().manual_seed(149)
    out = {"checks": {}, "shapes": {}, "code_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE,
                                                                   capture_output=True, text=True).stdout.strip()}
    codec = TrainCodec(make_codec("2", "q8"))
    # correctness: CUDA vs CPU forward, kernel backward vs torch backward (K = 16)
    global FORCE_TORCH_BACKWARD
    for K in (16, 32):
        x = synthetic(64, 30, 4, K, gen)
        cpu_out, _ = LowRankSTEWKV.apply(*x[:6], x[6], None, codec)
        xg = [t.to(dev) for t in x]
        leaves = [t.clone().requires_grad_(True) for t in xg[:6]]
        o, _ = LowRankSTEWKV.apply(*leaves, xg[6], None, codec)
        g = torch.randn(o.shape, generator=gen).to(dev)
        (o * g).sum().backward()
        k_grads = [t.grad.clone() for t in leaves]
        FORCE_TORCH_BACKWARD = True
        leaves = [t.clone().requires_grad_(True) for t in xg[:6]]
        o2, _ = LowRankSTEWKV.apply(*leaves, xg[6], None, codec)
        (o2 * g).sum().backward()
        FORCE_TORCH_BACKWARD = False
        rel = max(float((p - q.grad).abs().max() / q.grad.abs().max()) for p, q in zip(k_grads, leaves))
        out["checks"][f"K{K}"] = {"cuda_vs_cpu_forward_max_abs": float((o.detach().cpu() - cpu_out).abs().max()),
                                  "kernel_vs_torch_backward_rel": rel}
    timing = json.loads(TIMING.read_text())
    updates = timing["per_update"][:args.updates]
    from expertium.model.rwkv_model import RWKV7_WKV
    from rwkv.model import nvrtc
    nvrtc.set_max_timesteps(131072)   # the trainer's value (result.json nvrtc_max_timesteps)

    def run(B, T, nonskip, K, path):
        x = synthetic(B, T, 4, K, gen, nonskip=nonskip)
        xg = [t.to(dev) for t in x]
        leaves = [t.clone().requires_grad_(True) for t in xg[:6]]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if path == "plain":
            o = RWKV7_WKV.apply(*leaves, xg[6])
        else:
            o, _ = LowRankSTEWKV.apply(*leaves, xg[6], None, codec)
        o.float().sum().backward()
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    run(256, 8, 0.5, 16, "qat")   # warm-up (NVRTC / CuPy compiles)
    run(256, 8, 0.5, 32, "qat")
    run(256, 8, 0.5, 16, "plain")
    started = time.perf_counter()
    per = []
    for u in updates:
        if time.perf_counter() - started > args.max_seconds:
            break
        row = {"user": u["user"], "reviews": u["reviews"]}
        for K in (16, 32):
            for path in ("plain", "qat"):
                total = 0.0
                for stream in ("card", "note"):
                    for B, T, n in u[stream]["splits"]:
                        total += run(B, T, n / max(1, B * T), K, path)
                row[f"K{K}_{path}_s_per_layer_pair"] = round(total, 3)
        per.append(row)
    out["updates_measured"] = len(per)
    out["per_update"] = per

    def mean(key):
        return sum(r[key] for r in per) / max(1, len(per))
    m = {key: mean(key) for key in per[0] if key.startswith("K")} if per else {}
    out["mean_seconds_per_32k_update_card_plus_note_one_layer_each"] = m
    # extra seconds per update = (qat - plain) x coded layers (a card + note pair per layer measured above), Big at
    # half the reviews per update (2 x 8,192)
    if m:
        dk16 = m["K16_qat_s_per_layer_pair"] - m["K16_plain_s_per_layer_pair"]
        dk32 = m["K32_qat_s_per_layer_pair"] - m["K32_plain_s_per_layer_pair"]
        out["extra_seconds_per_update"] = {"small (2 card + 1 note)": round(dk16 * 1.5, 3),
                                           "tiny (1 card + 1 note)": round(dk16 * 1.0, 3),
                                           "big (3 card + 2 note, 16k reviews)": round(dk32 * 2.5 * 0.5, 3)}
    out["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_update"}, indent=1))
    return 0


def main():
    p = ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cpu")
    c.add_argument("--users", default="4745,2308")
    c.add_argument("--big-users", default="9037,6013")
    c.add_argument("--chunk", type=int, default=256)
    c.add_argument("--steps", type=int, default=3)
    c.add_argument("--lr", type=float, default=7e-5)
    c.add_argument("--out", default=str(HERE / "check-cpu-sm149.json"))
    b = sub.add_parser("bench")
    b.add_argument("--updates", type=int, default=12)
    b.add_argument("--max-seconds", type=float, default=360.0)
    b.add_argument("--out", default=str(HERE / "gpu" / "bench-sm149.json"))
    args = p.parse_args()
    return main_cpu(args) if args.cmd == "cpu" else main_bench(args)


if __name__ == "__main__":
    sys.exit(main())
