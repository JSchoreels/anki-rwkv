#!/usr/bin/env python
"""CPU check of the SM32b state-VQ QAT training path (`"state_vq"` track option; docs/SMALL_MODEL.md SM32b).

  (a) option absent = training bitwise unchanged: 3 decay updates (bf16 student + fp32 master, AdamW from the
      epoch-10 optimizer state, the decay LR, dropout on, real 1,024-review batches of the decay's own users) give the
      same losses, gradient norms, gradients and final weights with this checkout as with a clean export of HEAD
      (scripted model with the trainer's graph-executor setting, and eager), and the KD-cache folder of the decay
      options is the same with or without "state_vq" (TEACHER_BLIND), i.e. dec10's cache;
  (b) option on, forward = the CPU scorer's `+vq<recipe>` codec: full-history predictions of the scripted training
      model with the QAT hooks vs scripts/small_model_heldout.py's eager CPU path (state_vq.install/configure) on the
      smallest users of users34.json, same checkpoint, w8 weights;
  (c) gradients flow: the explicit VQStateWKV backward equals autograd through a straight-through reference loop
      (float64), the shift hook passes the gradient unchanged, and 3 QAT updates run with finite, non-zero
      card/note gradients and a changed loss vs the option-off run.

  check_state_vq_qat.py all [--recipe vq378x412] [--out artifacts/small-model/state-vq/qat/check.json]
  check_state_vq_qat.py cuda-codec [--out artifacts/small-model/state-vq/qat/cuda-codec.json]
     (GPU, first command of the training ticket) codec + VQStateWKV on CUDA vs CPU, CUDA per-step time; exit 1 stops
  check_state_vq_qat.py timing [--chunks 40] [--out artifacts/small-model/state-vq/qat/timing.json]
     per-step CPU cost of the QAT recurrence, op counts, and loop steps per 32,768-review update over the decay's
     rows, with the expected GPU ticket length.
"""
from argparse import ArgumentParser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SM = Path(r"C:\Users\jonat\workspace\wt\small-model\artifacts\small-model")
EXP = Path(r"C:\Users\jonat\workspace\wt\small-model\artifacts\accuracy\experiments")
E10 = EXP / "scale-w64kd-gate-cached-ctx32768-cont" / "run" / "epoch-10" / "expertium" / "E89-F65e-F77a"
DEC10 = SM / "scale-w64kd-gate-cached-ctx32768-cont-decay"
USERS34 = ROOT / "artifacts" / "small-model" / "state-vq" / "users34.json"
OUT = ROOT / "artifacts" / "small-model" / "state-vq" / "qat"
PY = sys.executable
# the S09 run: Gram-squaring codec (same bits and codebooks as vq378x412), float32 factorization, fused CUDA kernels
RUN_RECIPE, RUN_FACTOR, RUN_IMPL = "vq378x412p", "float32", "cuda"
DEC2048 = SM / "scale-w64kd-gate-cached-ctx2048-cont-decay"
# chunking -> (decay folder, sampled splits file, plain decay seconds for its 3,465 updates (result.json))
CHUNKINGS = {"32k": (DEC10, OUT / "timing.json", 3864.9), "2048": (DEC2048, OUT / "timing-2048.json", 1582.66)}


def sha(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().float().contiguous().numpy().tobytes())
    return h.hexdigest()


def decay_setup(chunk):
    protocol = json.loads((DEC10 / "protocol.json").read_text())
    options = dict(json.loads((DEC10 / "decay.json").read_text())["options"], chunk_reviews=chunk)
    os.environ["GPU_BATCH1_DATA_PATH"] = protocol["dataset_path"]
    return protocol, options


# ------------------------------------------------------------------------------------------------ subprocess parts

def part_train(args):
    """N decay updates on CPU with the tree at --root (sys.path), option on/off; writes JSON to --dump."""
    import torch
    torch.set_num_threads(4)
    torch._C._set_graph_executor_optimize(False)   # expertium_decay.run: jit_optimize absent -> off
    torch.use_deterministic_algorithms(True)
    protocol, options = decay_setup(args.chunk)
    from scripts.expertium_options import CatchUp
    from scripts.gpu_batch1 import prepare_job
    from rwkv.train_rwkv import CLIP, transfer_child_grad_to_master
    rows = [r for r in protocol["rows"] if r["train_range"][1] - r["train_range"][0] >= args.chunk * args.steps][:1]
    catch_up = CatchUp(options, protocol["full_run"], protocol["rows"])
    master, base, dtype = catch_up.models({"DTYPE": "bfloat16", "SEED": options.get("training_seed", 2025)},
                                          torch.device("cpu"))
    master.load_state_dict(torch.load(E10 / "master.pth", map_location="cpu", weights_only=True))
    base.copy_downcast_(master, dtype)
    info = None
    if args.state_vq:
        from scripts.state_vq import install_training
        spec = json.loads(args.state_vq) if args.state_vq.startswith("{") else args.state_vq
        info = install_training(base, spec, torch.device("cpu"))
    optimizer = catch_up.optimizer(master)
    optimizer.load_state_dict(torch.load(E10 / "optimizer.pth", map_location="cpu", weights_only=True))
    offset0 = protocol["full_run"]["update_offset"]
    torch.manual_seed(12345)
    steps, done = [], 0
    names = [n for n, _ in master.named_parameters()]
    card_note = [n for n in names if n.startswith(("rwkv_modules.0.", "rwkv_modules.2."))]
    for i in range(args.steps):
        row = rows[0]
        offset = i * args.chunk
        item = prepare_job(row, offset, offset + args.chunk, options.get("training_seed", 2025) + row["user_id"] + offset,
                           options)
        batch = item.batch
        batch.start = batch.start.to(dtype)
        base.zero_grad(set_to_none=True)
        base.train()
        base.copy_downcast_(master, dtype)
        t0 = time.perf_counter()
        stats = base.get_loss(batch)
        stats.average_loss.backward()
        seconds = time.perf_counter() - t0
        transfer_child_grad_to_master(master, base)
        params = dict(master.named_parameters())
        grads = [params[n].grad if params[n].grad is not None else torch.zeros_like(params[n]) for n in names]
        cn = [params[n].grad for n in card_note if params[n].grad is not None]
        norm = torch.nn.utils.clip_grad_norm_(master.parameters(), CLIP)
        lr = catch_up.learning_rate(i, offset0, done)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        master.zero_grad(set_to_none=True)
        done += item.reviews
        steps.append({"user": row["user_id"], "offset": offset, "loss": float(stats.average_loss).hex(),
                      "loss_value": float(stats.average_loss), "grad_norm": float(norm).hex(), "lr": lr,
                      "grad_sha256": sha(grads), "seconds": round(seconds, 2),
                      "card_note_grads_finite": all(bool(torch.isfinite(g).all()) for g in cn),
                      "card_note_grad_abs_sum": float(sum(g.abs().sum() for g in cn)),
                      "card_note_grad_tensors": len(cn)})
    out = {"root": args.root, "jit": not os.environ.get("RWKV_NO_JIT"), "state_vq": info, "steps": steps,
           "weights_sha256": sha([p for _, p in master.named_parameters()])}
    Path(args.dump).write_text(json.dumps(out, indent=1), encoding="utf-8")


def part_forward(args):
    """Full-history predictions (w8 weights + VQ state) of one user list, scorer path or training path."""
    import torch
    torch.set_num_threads(2)
    from scripts import compress_common as cc
    from scripts import compress_quant as cq
    from scripts import state_vq
    from scripts.full_run_metrics import evaluate_with_predictions
    state = {k: v.float() for k, v in torch.load(E10 / "master.pth", map_location="cpu", weights_only=True).items()}
    bits, group = cq.WEIGHT_SCHEMES["w8-pc"]
    w8 = cq.quantize_state_dict(state, cq.linear_weight_names(state), bits, group)
    if args.path == "scorer":
        from scripts import small_model_heldout as smh
        model = smh.build_expertium("w64", w8)
        layout = state_vq.install(model)
        smh.set_expertium_state_bits(model, 0)
        books = state_vq.Codebooks.load(state_vq.DEFAULT_CODEBOOKS)
        state_vq.configure(layout, {s: state_vq.StateCodec(rec, books, factor_dtype=getattr(torch, args.factor_dtype))
                                    for s, rec in zip(layout, state_vq.recipe(args.recipe))})
    else:
        torch._C._set_graph_executor_optimize(False)
        from expertium.variants import build
        model = build("w64").float()
        model.load_state_dict(w8, strict=True)
        state_vq.install_training(model, {"recipe": args.recipe, "factor_dtype": args.factor_dtype, "impl": "cuda"
                                          if state_vq.recipe(args.recipe)[0].factor == "sq" else "torch"},
                                  torch.device("cpu"))
    run = cc.run_info()
    out = {"path": args.path, "recipe": args.recipe, "factor_dtype": args.factor_dtype,
           "jit": isinstance(model, torch.jit.ScriptModule), "users": {}}
    for user in args.users:
        batch = cc.full_history_batch(cc.data_path(run), user)
        t0 = time.perf_counter()
        values, predictions = evaluate_with_predictions(model, [{"batch": batch}])
        out["users"][str(user)] = {"values": values, "seconds": round(time.perf_counter() - t0, 1),
                                   "predictions": {h: {str(k): v for k, v in p.items()} for h, p in predictions.items()}}
    Path(args.dump).write_text(json.dumps(out), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ in-process checks

def reference_loop(r, k, v, w, a, kd, skip, codec):
    """Straight-through reference: autograd records the loop; the codec value enters as a detached offset."""
    import torch
    from expertium.model.rwkv_ops import single_timestep
    B, T, H, K = r.shape
    state = r.new_zeros(B, H, K, K)
    outs = []
    for t in range(T):
        out_t, nxt = single_timestep(r[:, t], k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t], state)
        outs.append(out_t)
        stored = codec.wkv(nxt.detach()).to(nxt.dtype) + (nxt - nxt.detach())
        state = torch.where(skip[:, t, None, None, None], state, stored)
    return torch.stack(outs, dim=1)


def grad_check(recipe_name):
    import torch
    from expertium.model.state_vq_hooks import VQStateWKV, vq_shift, CODECS
    from scripts import state_vq
    books = state_vq.Codebooks.load(state_vq.DEFAULT_CODEBOOKS)
    results = {}
    gen = torch.Generator().manual_seed(7)
    for name, rec in zip(("card", "note"), state_vq.recipe(recipe_name)):
        codec = state_vq.StateCodec(rec, books, 1 << 20, 8192)
        B, T, H, K = 6, 14, 4, 16
        rnd = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)  # noqa: E731
        r = rnd(B, T, H, K)
        k = torch.nn.functional.normalize(rnd(B, T, H, K), dim=-1) * 0.8
        v = torch.nn.functional.normalize(rnd(B, T, H, K), dim=-1) * 0.7
        w = torch.exp(-torch.exp(rnd(B, T, H, K) * 0.5 - 1.5))
        a = torch.sigmoid(rnd(B, T, H, K))
        kd = torch.nn.functional.normalize(rnd(B, T, H, K), dim=-1)
        skip = torch.rand(B, T, generator=gen) < 0.3
        skip[:, 0] = False
        G = rnd(B, T, H, K)
        grads = []
        for fn in (lambda *x: VQStateWKV.apply(*x, skip, codec), lambda *x: reference_loop(*x, skip, codec)):
            leaves = [t.clone().requires_grad_(True) for t in (r, k, v, w, a, kd)]
            out = fn(*leaves)
            (out * G).sum().backward()
            grads.append((out.detach(), [t.grad for t in leaves]))
        (o1, g1), (o2, g2) = grads
        rel = max(float((x - y).abs().max() / y.abs().max().clamp_min(1e-30)) for x, y in zip(g1, g2))
        results[f"wkv_{name}"] = {"forward_bitwise": bool(torch.equal(o1, o2)), "max_rel_grad_diff": rel,
                                  "ok": bool(torch.equal(o1, o2)) and rel < 1e-9}
    # shift hook: value = codec on stored positions, own input elsewhere; gradient = identity
    CODECS.clear()
    CODECS[0] = state_vq.StateCodec(state_vq.recipe(recipe_name)[0], books, 1 << 20, 8192)
    x = torch.randn(3, 9, 64, generator=gen).requires_grad_(True)
    select = torch.tensor([[0, 0, 1, 2, 2, 4, 5, 5, 7]] * 3)
    y = vq_shift(x, select, 0, 0)
    G = torch.randn(3, 9, 64, generator=gen)
    y.backward(G)
    stored = select != torch.arange(9)[None]
    expect = CODECS[0].shift(x.detach().reshape(-1, 64), "t").reshape(3, 9, 64)
    results["shift"] = {"grad_identity": bool(torch.equal(x.grad, G)),
                        "own_input_unchanged": bool(torch.equal(y.detach()[~stored], x.detach()[~stored])),
                        "stored_coded": bool(torch.equal(y.detach()[stored], expect[stored])),
                        "stored_changed": bool((y.detach()[stored] != x.detach()[stored]).any())}
    results["shift"]["ok"] = all(results["shift"].values())
    CODECS.clear()
    return results


def kd_cache_check():
    from scripts.expertium_options import TEACHER_BLIND, WIDTH32_ONLY
    options = json.loads((DEC10 / "decay.json").read_text())["options"]
    digest = lambda o: hashlib.sha256(json.dumps({k: v for k, v in o.items() if k not in TEACHER_BLIND and  # noqa: E731
                                                  k not in WIDTH32_ONLY}, sort_keys=True).encode()).hexdigest()[:12]
    plain, qat = digest(options), digest(dict(options, state_vq="vq378x412"))
    used = json.loads((DEC10 / "expertium" / "E89-F65e-F77a" / "result.json").read_text())
    used = used["expertium_catch_up"]["distillation"]["cache"]
    text = (ROOT / "scripts" / "gpu_r01_prefetch.py").read_text(encoding="utf-8")
    return {"digest_plain": plain, "digest_state_vq": qat, "dec10_cache": used,
            "prefetch_allowlist_has_state_vq": '"state_vq"' in text,
            "ok": plain == qat and Path(used).name == f"batches-{plain}" and Path(used).is_dir() and '"state_vq"' in text}


def codec_sq_check(recipe_name, users=None):
    """(b) the factorization change on REAL stored states (the SM32 corpus of training-user states): the recipe's
    Gram-squaring codec (float32 and float64) vs the deployed eigh codec (float64, the scorer's), WKV reconstructions
    per entity; a pick difference shows as a large reconstruction difference."""
    import numpy as np
    import torch
    from dataclasses import replace
    from scripts import state_vq
    books = state_vq.Codebooks.load(state_vq.DEFAULT_CODEBOOKS)
    files = sorted((state_vq.DEFAULT_CODEBOOKS.parent / "corpus").glob("*.npz"))[:users]
    states = {0: [], 2: []}
    for f in files:
        z = np.load(f)
        for key in z.files:
            if key.startswith("wkv/"):
                states[int(key.split("/")[1])].append(torch.from_numpy(z[key]))
    out = {"corpus_users": len(files)}
    for (stream, X), rec in zip(states.items(), state_vq.recipe(recipe_name)):
        X = torch.cat(X)
        A = X.reshape(-1, state_vq.K, state_vq.K).double()
        an = A / A.abs().amax((1, 2)).clamp_min(1e-30)[:, None, None]
        ev = torch.linalg.eigvalsh(an @ an.transpose(1, 2))
        ratio = (ev[:, -2] / ev[:, -1].clamp_min(1e-300)).float()
        deployed = state_vq.StateCodec(replace(rec, factor="eigh"), books).wkv(X)
        norm = X.flatten(1).norm(dim=1).clamp_min(1e-30)
        entry = {"entities": int(X.shape[0]), "heads": int(A.shape[0]),
                 "lambda2_over_lambda1": {**{f"q{q}": round(float(torch.quantile(ratio, q / 100)), 4)
                                             for q in (50, 90, 99, 99.9)}, "max": round(float(ratio.max()), 4)}}
        for name in ("float64", "float32"):
            got = state_vq.StateCodec(replace(rec, factor="sq"), books, factor_dtype=getattr(torch, name)).wkv(X)
            rel = (got - deployed).flatten(1).norm(dim=1) / norm
            entry[f"sq_{name}"] = {"entities_rel_diff_above_1e-5": int((rel > 1e-5).sum()),
                                   "max_rel_diff": float(rel.max()), "mean_rel_diff": float(rel.mean())}
        entry["ok"] = all(entry[f"sq_{n}"]["entities_rel_diff_above_1e-5"] <= 0.001 * X.shape[0]
                          for n in ("float64", "float32"))
        out[{0: "card", 2: "note"}[stream]] = entry
    out["ok"] = out["card"]["ok"] and out["note"]["ok"]
    return out


def kernel_compile_check():
    """(b) NVRTC compile of the fused forward (float32 / float64 factorization) and of the backward instance for the
    RTX 5080 (sm_120), without a CUDA device. Spills would mean the 256-thread block is register-starved."""
    import torch
    from scripts import state_vq_cuda
    out = {name: state_vq_cuda.compile_only(factor_dtype=fd, verbose_arch="sm_120")
           for name, fd in (("float32", torch.float32), ("float64", torch.float64))}
    out["ok"] = all("0 bytes spill stores" in " ".join(v["forward"]["ptxas"]) for v in out.values())
    return out


def _states_torch(inputs, skip, codec):
    """Stored state before every step of the torch QAT path (VQStateWKV.forward), [B, T, H, K, K]."""
    import torch
    from expertium.model.rwkv_ops import single_timestep
    r, k, v, w, a, kd = inputs
    B, T, H, K = r.shape
    state, out = r.new_zeros(B, H, K, K), []
    for t in range(T):
        out.append(state)
        _, nxt = single_timestep(r[:, t], k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t], state)
        state = torch.where(skip[:, t, None, None, None], state, codec.wkv(nxt).to(nxt.dtype))
    return torch.stack(out, 1)


def compare_kernel_case(codec, inputs, skip, G, kernel_forward, kernel_backward, deployed=None):
    """Kernel forward / backward (CUDA, or the CPU emulation) vs the torch QAT path of `codec` on the device of
    `inputs`. Stored states: one-step consistency (the torch codec applied to the kernel's own previous stored state,
    so a near-tie flip counts once instead of diverging the rest of the row) and whole-trajectory agreement; the
    backward is compared with the explicit STE backward along the kernel's trajectory. `deployed` (optional): the
    eigh float64 codec, one-step consistency as well (the deployed codec vs what training stores)."""
    import torch
    from expertium.model.state_vq_hooks import VQStateWKV, ste_backward
    from expertium.model.rwkv_ops import single_timestep
    r, k, v, w, a, kd = inputs
    B, T, H, K = r.shape
    out_k, before_k = kernel_forward(*inputs, skip)
    out_k, before_k = out_k.to(r.device), before_k.to(r.device)
    with torch.no_grad():
        out_t = VQStateWKV.apply(*inputs, skip, codec)
        traj = (before_k - _states_torch(inputs, skip, codec)).flatten(3).abs().amax(-1)   # [B, T, H]
        steps = {"torch": [0, 0.0]}
        if deployed is not None:
            steps["deployed"] = [0, 0.0]
        total = 0
        for t in range(T - 1):
            _, nxt = single_timestep(r[:, t], k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t], before_k[:, t])
            m = (~skip[:, t])[:, None].expand(B, H)
            total += int(m.sum())
            for name, c in (("torch", codec), ("deployed", deployed)):
                if c is None:
                    continue
                d = (c.wkv(nxt).to(nxt.dtype) - before_k[:, t + 1]).flatten(2).abs().amax(-1)[m]
                steps[name][0] += int((d > 1e-5).sum())
                steps[name][1] = max(steps[name][1], float(d.max()) if d.numel() else 0.0)
    grads_k = kernel_backward(*inputs, skip, before_k, G)
    grads_t = ste_backward(*inputs, skip, before_k.transpose(0, 1), G)
    rel = max(float((x.to(y.device) - y).abs().max() / y.abs().max().clamp_min(1e-30)) for x, y in zip(grads_k, grads_t))
    entry = {"rows": B, "T": T, "heads": H, "coded_head_steps": total,
             "one_step_mismatch_above_1e-5": {n: s[0] for n, s in steps.items()},
             "one_step_max_abs_diff": {n: s[1] for n, s in steps.items()},
             "trajectory_heads_diverged_above_1e-5": int((traj.amax(1) > 1e-5).sum()), "heads_total": B * H,
             "out_max_abs_diff": float((out_k - out_t).abs().max()), "backward_max_rel_diff": rel}
    entry["ok"] = all(s[0] <= max(1, 0.001 * total) for s in steps.values()) and rel < 1e-4
    return entry


def emulated_kernel_check(recipe_name, factor_dtype):
    """(b, c) the UNCHANGED CUDA sources run on the CPU (scripts/state_vq_cuda_emu.py, MSVC thread emulation) on
    synthetic sequences: fused forward vs the torch QAT path, checkpoint-1 backward kernel vs the explicit STE
    backward."""
    import torch
    from scripts import state_vq, state_vq_cuda, state_vq_cuda_emu as emu
    if not emu.available():
        return {"ok": None, "skipped": "no MSVC Build Tools for the emulation"}
    books = state_vq.Codebooks.load(state_vq.DEFAULT_CODEBOOKS)
    fd = getattr(torch, factor_dtype)
    gen = torch.Generator().manual_seed(11)
    out = {}
    for name, rec in zip(("card", "note"), state_vq.recipe(recipe_name)):
        codec = state_vq.StateCodec(rec, books, 1 << 20, 8192, fd)
        kc = state_vq_cuda.KernelCodec(codec, torch.device("cpu"), fd)
        B, T, H, K = 3, 16, 2, 16
        rnd = lambda *s: torch.randn(*s, generator=gen)  # noqa: E731
        inputs = (rnd(B, T, H, K), torch.nn.functional.normalize(rnd(B, T, H, K), dim=-1) * 0.8,
                  torch.nn.functional.normalize(rnd(B, T, H, K), dim=-1) * 0.7,
                  torch.exp(-torch.exp(rnd(B, T, H, K) * 0.5 - 1.5)), torch.sigmoid(rnd(B, T, H, K)),
                  torch.nn.functional.normalize(rnd(B, T, H, K), dim=-1))
        skip = torch.rand(B, T, generator=gen) < 0.25
        skip[:, 0] = False
        G = rnd(B, T, H, K)
        out[name] = compare_kernel_case(
            codec, inputs, skip, G, lambda *x: emu.forward(kc, *x, factor_double=fd == torch.float64), emu.backward)
    out["ok"] = all(v["ok"] for v in out.values())
    return out


def part_capture(args):
    """(b) real inputs: the QAT WKV calls of the training model on one user's full history (CPU, torch path), then
    the emulated CUDA kernels on the busiest rows of the two largest calls per stream vs the torch path, with the
    deployed eigh codec's one-step consistency."""
    import torch
    from dataclasses import replace
    torch.set_num_threads(2)
    torch._C._set_graph_executor_optimize(False)
    from scripts import compress_common as cc
    from scripts import compress_quant as cq
    from scripts import state_vq, state_vq_cuda, state_vq_cuda_emu as emu
    from scripts.full_run_metrics import evaluate_with_predictions
    from expertium.model import state_vq_hooks as hooks
    from expertium.variants import build
    state = {k: v.float() for k, v in torch.load(E10 / "master.pth", map_location="cpu", weights_only=True).items()}
    bits, group = cq.WEIGHT_SCHEMES["w8-pc"]
    model = build("w64").float()
    model.load_state_dict(cq.quantize_state_dict(state, cq.linear_weight_names(state), bits, group), strict=True)
    fd = getattr(torch, args.factor_dtype)
    state_vq.install_training(model, {"recipe": args.recipe, "factor_dtype": args.factor_dtype}, torch.device("cpu"))
    calls, original = [], hooks.VQStateWKV

    class Recorder:
        @staticmethod
        def apply(r, k, v, w, a, kd, skip, codec):
            slot = next(s for s, c in hooks.CODECS.items() if c is codec)
            calls.append((slot, [t.detach().clone() for t in (r, k, v, w, a, kd)], skip.clone()))
            return original.apply(r, k, v, w, a, kd, skip, codec)
    hooks.VQStateWKV = Recorder
    try:
        evaluate_with_predictions(model, [{"batch": cc.full_history_batch(cc.data_path(cc.run_info()), args.users[0])}])
    finally:
        hooks.VQStateWKV = original
    books = state_vq.Codebooks.load(state_vq.DEFAULT_CODEBOOKS)
    out = {"user": args.users[0], "calls": len(calls), "recipe": args.recipe, "factor_dtype": args.factor_dtype,
           "cases": []}
    for slot in (0, 2):
        codec = hooks.CODECS[slot]
        deployed = state_vq.StateCodec(replace(codec.rec, factor="eigh"), books)
        kc = state_vq_cuda.KernelCodec(codec, torch.device("cpu"), fd)
        mine = [c for c in calls if c[0] == slot]
        busiest = max(mine, key=lambda c: int((~c[2]).sum()))
        longest = max((c for c in mine if c is not busiest), key=lambda c: (c[2].shape[1], int((~c[2]).sum())))
        mine = [busiest, longest]
        for _, inputs, skip in mine:
            rows = (~skip).sum(1).argsort(descending=True)[:args.rows]
            T = min(skip.shape[1], args.steps)
            sub = tuple(t[rows, :T].contiguous() for t in inputs)
            sk = skip[rows, :T].contiguous()
            G = torch.randn(sub[0].shape, generator=torch.Generator().manual_seed(len(out["cases"])))
            entry = compare_kernel_case(codec, sub, sk, G,
                                        lambda *x: emu.forward(kc, *x, factor_double=fd == torch.float64),
                                        emu.backward, deployed)
            entry.update(stream={0: "card", 2: "note"}[slot], split_shape=list(skip.shape))
            out["cases"].append(entry)
            print(json.dumps(entry), flush=True)
    out["ok"] = len(out["cases"]) == 4 and all(c["ok"] for c in out["cases"])
    Path(args.dump).write_text(json.dumps(out, indent=1), encoding="utf-8")


def run(cmd, env_extra=None, log=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", **(env_extra or {}))
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as fh:
        code = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if code:
        raise SystemExit(f"{' '.join(map(str, cmd))} failed ({code}); see {log}")
    return time.perf_counter() - t0


def compare_forward(a, b):
    import numpy as np
    out = {}
    for user, ua in a["users"].items():
        ub = b["users"][user]
        diffs = []
        for head in ("ahead", "instant", "rating"):
            pa, pb = ua["predictions"][head], ub["predictions"][head]
            if set(pa) != set(pb):
                return {"ok": False, "error": f"user {user} {head}: different review sets"}
            keys = sorted(pa)
            diffs.append(np.abs(np.array([pa[k] for k in keys], dtype=float) - np.array([pb[k] for k in keys], dtype=float)))
        allv = np.concatenate([d.ravel() for d in diffs])
        vals = {k: abs(ua["values"][k] - ub["values"][k]) for k in ua["values"] if isinstance(ua["values"][k], float)}
        out[user] = {"predictions": int(allv.size), "max_abs_diff": float(allv.max()), "mean_abs_diff": float(allv.mean()),
                     "share_above_1e-5": float((allv > 1e-5).mean()), "max_abs_metric_diff": max(vals.values()),
                     "seconds": [ua["seconds"], ub["seconds"]]}
    out["ok"] = all(u["max_abs_metric_diff"] < 1e-5 and u["mean_abs_diff"] < 1e-5 for u in out.values())
    return out


def codec_change(deployed, new, training):
    """Per-user metric differences: scorer with the "p" recipe and the training path vs the deployed scorer."""
    out = {}
    for user, ud in deployed["users"].items():
        vals = {k: v for k, v in ud["values"].items() if isinstance(v, float)}
        out[user] = {name: {k: run_["users"][user]["values"][k] - v for k, v in vals.items()
                            if k.endswith("log_loss") or k.endswith("cross_entropy")}
                     for name, run_ in (("p_scorer_minus_deployed", new), ("training_minus_deployed", training))}
    worst = max(abs(d) for u in out.values() for part in u.values() for d in part.values())
    return {"users": out, "max_abs_metric_diff": worst, "ok": worst < 1e-4}


def main_all(args):
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    logs = OUT / "check-logs"
    logs.mkdir(exist_ok=True)
    report = {"recipe": args.recipe, "factor_dtype": args.factor_dtype, "codebooks_sha256": None,
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    from scripts import state_vq
    report["codebooks_sha256"] = state_vq.file_sha256(state_vq.DEFAULT_CODEBOOKS)
    report["head"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    report["dirty"] = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", "scripts",
                                                    "expertium", "rwkv"], text=True).strip())
    # (c1) explicit backward vs autograd, shift STE; KD key
    report["c_grad_check"] = grad_check(args.recipe)
    report["a_kd_cache"] = kd_cache_check()
    print(json.dumps({"c_grad_check": report["c_grad_check"], "a_kd_cache": report["a_kd_cache"]}), flush=True)
    # (b) S09: the Gram-squaring codec vs the deployed eigh codec on real states; the CUDA sources (compile for
    # sm_120, then executed on the CPU through the MSVC emulation) vs the torch QAT path, (c) incl. their backward
    deployed_recipe = args.recipe[:-1] if args.recipe.endswith("p") and args.recipe[:-1] in state_vq.RECIPES else None
    t0 = time.perf_counter()
    report["b_codec_sq_vs_eigh"] = codec_sq_check(args.recipe) if deployed_recipe else {"ok": True, "skipped": "eigh recipe"}
    report["b_kernel_compile"] = kernel_compile_check()
    report["bc_kernel_emulated_synthetic"] = emulated_kernel_check(args.recipe, args.factor_dtype)
    print(json.dumps({k: report[k] for k in ("b_codec_sq_vs_eigh", "b_kernel_compile", "bc_kernel_emulated_synthetic")}),
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    # (a) HEAD export vs this tree, option absent; (c2) option on
    me = str(Path(__file__).resolve())
    with tempfile.TemporaryDirectory(prefix="sm32b-head-") as head:
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", "HEAD", "scripts", "expertium", "rwkv", "config.py",
                                  "utils.py", "features"], capture_output=True, check=True).stdout
        import io
        import tarfile
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(head, filter="data")
        dumps = {}
        for mode, env in (("jit", {}), ("eager", {"RWKV_NO_JIT": "1"})):
            for tree, root in (("head", head), ("tree", str(ROOT))):
                dump = OUT / f"check-train-{mode}-{tree}.json"
                sec = run([PY, "-X", "utf8", me, "part-train", "--root", root, "--dump", str(dump), "--steps",
                           str(args.steps), "--chunk", str(args.chunk)], env, logs / f"train-{mode}-{tree}.log")
                dumps[(mode, tree)] = json.loads(dump.read_text())
                print(f"[check] train {mode} {tree}: {sec:.0f}s", flush=True)
        dump = OUT / "check-train-jit-qat.json"
        spec = json.dumps({"recipe": args.recipe, "factor_dtype": args.factor_dtype, "impl": args.impl})
        sec = run([PY, "-X", "utf8", me, "part-train", "--root", str(ROOT), "--dump", str(dump), "--steps", str(args.steps),
                   "--chunk", str(args.chunk), "--state-vq", spec], {}, logs / "train-jit-qat.log")
        dumps[("jit", "qat")] = json.loads(dump.read_text())
        print(f"[check] train jit qat: {sec:.0f}s", flush=True)
    a = {}
    for mode in ("jit", "eager"):
        h, t = dumps[(mode, "head")], dumps[(mode, "tree")]
        same = [x == y for x, y in zip(([s["loss"], s["grad_norm"], s["grad_sha256"]] for s in h["steps"]),
                                        ([s["loss"], s["grad_norm"], s["grad_sha256"]] for s in t["steps"]))]
        a[mode] = {"steps": len(same), "steps_bitwise_equal": sum(same), "weights_equal": h["weights_sha256"] == t["weights_sha256"],
                   "losses": [s["loss_value"] for s in t["steps"]], "seconds_per_step": [s["seconds"] for s in t["steps"]]}
        a[mode]["ok"] = all(same) and len(same) == args.steps and a[mode]["weights_equal"]
    report["a_option_off_bitwise"] = a
    off, on = dumps[("jit", "tree")], dumps[("jit", "qat")]
    c2 = {"losses_off": [s["loss_value"] for s in off["steps"]], "losses_qat": [s["loss_value"] for s in on["steps"]],
          "seconds_per_step_off": [s["seconds"] for s in off["steps"]],
          "seconds_per_step_qat": [s["seconds"] for s in on["steps"]],
          "card_note_grads_finite": all(s["card_note_grads_finite"] for s in on["steps"]),
          "card_note_grad_abs_sum": [s["card_note_grad_abs_sum"] for s in on["steps"]],
          "grad_differs_from_off": all(x["grad_sha256"] != y["grad_sha256"] for x, y in zip(off["steps"], on["steps"])),
          "weights_differ_from_off": off["weights_sha256"] != on["weights_sha256"], "install": on["state_vq"]}
    c2["ok"] = (c2["card_note_grads_finite"] and all(x > 0 for x in c2["card_note_grad_abs_sum"])
                and c2["grad_differs_from_off"] and all(map(lambda v: v == v and abs(v) < 1e3, c2["losses_qat"])))
    report["c_qat_updates"] = c2
    # (b) forward: scorer CPU path vs scripted training path
    counts = json.loads((Path(r"C:\Users\jonat\workspace\anki-rwkv") / "artifacts" / "full-run" / "run-8k-20261013"
                         / "run.json").read_text())["review_counts"]
    users = sorted(json.loads(USERS34.read_text())["users"], key=lambda u: counts[str(u)])[:args.users]
    fwd = {}
    runs = [("training", args.recipe, args.factor_dtype, {}), ("scorer", args.recipe, "float64", {"RWKV_NO_JIT": "1"})]
    if deployed_recipe:
        runs.append(("scorer", deployed_recipe, "float64", {"RWKV_NO_JIT": "1"}))
    for path, rec, fdt, env in runs:
        tag = f"{path}-{rec}-{fdt}"
        dump = OUT / f"check-forward-{tag}.json"
        sec = run([PY, "-X", "utf8", me, "part-forward", "--path", path, "--recipe", rec, "--factor-dtype", fdt,
                   "--dump", str(dump), "--users", *map(str, users)], env, logs / f"forward-{tag}.log")
        fwd[tag] = json.loads(dump.read_text())
        print(f"[check] forward {tag}: {sec:.0f}s", flush=True)
    # training path (the run's recipe / factor dtype, scripted) vs the scorer with the same recipe
    b = compare_forward(fwd[f"scorer-{args.recipe}-float64"], fwd[f"training-{args.recipe}-{args.factor_dtype}"])
    b["users"] = {str(u): counts[str(u)] for u in users}
    b["training_path_scripted"] = fwd[f"training-{args.recipe}-{args.factor_dtype}"]["jit"]
    report["b_forward_vs_scorer"] = b
    if deployed_recipe:
        # the codec change on the scorer's metrics (PTQ of epoch 10, w8): deployed eigh recipe vs the "p" recipe,
        # and the training path vs the deployed scorer
        report["b_codec_change_logloss"] = codec_change(fwd[f"scorer-{deployed_recipe}-float64"],
                                                        fwd[f"scorer-{args.recipe}-float64"],
                                                        fwd[f"training-{args.recipe}-{args.factor_dtype}"])
    # (b) the emulated CUDA kernels on captured real inputs (one user, busiest rows)
    dump = OUT / "check-kernel-real-inputs.json"
    from scripts import state_vq_cuda_emu
    if state_vq_cuda_emu.available():
        sec = run([PY, "-X", "utf8", me, "part-capture", "--recipe", args.recipe, "--factor-dtype", args.factor_dtype,
                   "--users", str(users[0]), "--rows", "3", "--steps", "40", "--dump", str(dump)], {"RWKV_NO_JIT": "1"},
                  logs / "kernel-real-inputs.log")
        report["bc_kernel_emulated_real_inputs"] = json.loads(dump.read_text())
        print(f"[check] emulated kernels on real inputs: {sec:.0f}s", flush=True)
    else:
        report["bc_kernel_emulated_real_inputs"] = {"ok": None, "skipped": "no MSVC Build Tools for the emulation"}
    emulated = [report["bc_kernel_emulated_synthetic"]["ok"], report["bc_kernel_emulated_real_inputs"]["ok"]]
    oks = {"a": all(v["ok"] for v in a.values()) and report["a_kd_cache"]["ok"],
           "b": (b["ok"] and report["b_codec_sq_vs_eigh"]["ok"] and report["b_kernel_compile"]["ok"]
                 and report.get("b_codec_change_logloss", {"ok": True})["ok"] and all(e is not False for e in emulated)),
           "c": all(v["ok"] for v in report["c_grad_check"].values()) and c2["ok"] and all(e is not False for e in emulated)}
    report["kernel_emulation_ran"] = all(e is not None for e in emulated)
    report["ok"] = oks
    (ROOT / args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print("ALL OK" if all(oks.values()) else f"FAILED {oks}")
    return 0 if all(oks.values()) else 1


# ------------------------------------------------------------------------------------------------ timing

class OpCounter:
    def __init__(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        class Mode(TorchDispatchMode):
            count = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                Mode.count += 1
                return func(*args, **(kwargs or {}))
        self.Mode = Mode


def sample_updates(decay_dir, count, seed=1):
    """Loop shapes of `count` sampled updates of a decay (CPU batch preparation, no model): per update and stream
    the splits as [rows, length, non-skipped row steps] and the loop steps (sum of split lengths)."""
    import random
    decay_dir = Path(decay_dir)
    protocol = json.loads((decay_dir / "protocol.json").read_text())
    options = json.loads((decay_dir / "decay.json").read_text())["options"]
    os.environ["GPU_BATCH1_DATA_PATH"] = protocol["dataset_path"]
    from scripts.gpu_batch1 import prepare_job
    span = options.get("batch_chunks", 1) * options["chunk_reviews"]
    chunks = [(row, off) for row in protocol["rows"]
              for off in range(0, row["train_range"][1] - row["train_range"][0], span)]
    random.Random(seed).shuffle(chunks)
    per = []
    for row, off in chunks[:count]:
        end = min(off + span, row["train_range"][1] - row["train_range"][0])
        batch = prepare_job(row, off, end, options.get("training_seed", 2025) + row["user_id"] + off, options).batch
        entry = {"user": row["user_id"], "reviews": end - off}
        for s, name in ((0, "card"), (2, "note")):
            lens = [int(x) for x in batch.sub_gather_lens[s]]
            entry[name] = {"steps": sum(lens),
                           "splits": [[int(g.numel() // n), n, int((~sk.bool()).sum())]
                                      for g, n, sk in zip(batch.sub_gather[s], lens, batch.skips[s])]}
        per.append(entry)
    return {"decay": str(decay_dir), "chunk_reviews": options["chunk_reviews"],
            "batch_chunks": options.get("batch_chunks", 1), "updates_total": len(chunks), "per_update": per}


def main_timing(args):
    """CPU: loop shapes of the decay's updates (sampled), per-step CPU cost and aten op counts of the torch QAT step
    for the deployed eigh recipe and the run's Gram-squaring recipe, and the CPU-based GPU estimate of the TORCH path
    (the fused kernels are timed on the GPU by `bench`)."""
    import numpy as np
    import torch
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    from expertium.model.rwkv_ops import reference_rwkv7
    from expertium.model.state_vq_hooks import VQStateWKV
    from scripts import state_vq
    books = state_vq.Codebooks.load(state_vq.DEFAULT_CODEBOOKS)
    report = {"threads": 4, **sample_updates(args.decay_dir, args.chunks)}
    per = report["per_update"]
    for name in ("card", "note"):
        steps = np.array([e[name]["steps"] for e in per], dtype=float)
        rows = np.array([sum(r * n for r, n, _ in e[name]["splits"]) for e in per], dtype=float)
        coded = np.array([sum(c for _, _, c in e[name]["splits"]) for e in per], dtype=float)
        report[f"{name}_steps_per_update"] = {"mean": steps.mean(), "p90": float(np.percentile(steps, 90)),
                                              "max": steps.max()}
        report[f"{name}_row_steps_per_update"] = {"mean": rows.mean(), "coded_mean": coded.mean()}
    print(json.dumps({k: v for k, v in report.items() if k != "per_update"}, default=float), flush=True)
    # per-step CPU cost and op counts (forward + backward) of the torch QAT step
    counter = OpCounter()
    deployed = args.recipe[:-1] if args.recipe.endswith("p") else args.recipe
    variants = {"deployed_eigh_float64": (deployed, torch.float64), "run_sq": (args.recipe, getattr(torch, args.factor_dtype))}
    report["per_step_cpu"] = {}
    for variant, (rname, fd) in variants.items():
        codecs = {n: state_vq.StateCodec(rec, books, 1 << 20, 8192, fd) for n, rec in zip(("card", "note"), state_vq.recipe(rname))}
        cost = []
        for name, B, T in (("card", 800, 50), ("note", 200, 120), ("card", 1, 64), ("note", 1, 64)):
            gen = torch.Generator().manual_seed(B * 1000 + T)
            H, K = 4, 16
            leaves = [torch.randn(B, T, H, K, generator=gen) for _ in range(6)]
            leaves[3], leaves[4] = torch.sigmoid(leaves[3]), torch.sigmoid(leaves[4])
            for i in (1, 2, 5):
                leaves[i] = torch.nn.functional.normalize(leaves[i], dim=-1)
            leaves = [t.requires_grad_(True) for t in leaves]
            skip = torch.zeros(B, T, dtype=torch.bool)
            t0 = time.perf_counter()
            out = VQStateWKV.apply(*leaves, skip, codecs[name])
            t1 = time.perf_counter()
            out.sum().backward()
            t2 = time.perf_counter()
            with torch.no_grad():
                t3 = time.perf_counter()
                reference_rwkv7(*leaves, skip)
                t4 = time.perf_counter()
            with counter.Mode():
                counter.Mode.count = 0
                o = VQStateWKV.apply(*[t[:, :4] for t in leaves], skip[:, :4], codecs[name])
                fwd_ops = counter.Mode.count / 4
                counter.Mode.count = 0
                o.sum().backward()
                bwd_ops = counter.Mode.count / 4
            cost.append({"stream": name, "rows": B, "T": T, "fwd_ms_per_step": (t1 - t0) / T * 1e3,
                         "bwd_ms_per_step": (t2 - t1) / T * 1e3, "plain_fwd_ms_per_step": (t4 - t3) / T * 1e3,
                         "fwd_ops_per_step": fwd_ops, "bwd_ops_per_step": bwd_ops})
            print(variant, json.dumps(cost[-1]), flush=True)
        report["per_step_cpu"][variant] = cost
    # CPU-based GPU estimate of the torch path: per step = ops x (CPU per-op cost at 1 row + 6 us launch); the eigh
    # variant also syncs the host every step (not modelled)
    chunk = next(k for k, v in CHUNKINGS.items() if Path(v[0]) == Path(args.decay_dir))
    base_s = CHUNKINGS[chunk][2] / report["updates_total"]
    steps = {"card": report["card_steps_per_update"]["mean"], "note": report["note_steps_per_update"]["mean"]}
    report["gpu_estimate_torch_path"] = {"assumed_launch_us_per_op": 6.0, "plain_seconds_per_update": base_s}
    for variant, cost in report["per_step_cpu"].items():
        tiny = {e["stream"]: e for e in cost if e["rows"] == 1}
        added = 0.0
        for name, layers in (("card", 2), ("note", 1)):
            e = tiny[name]
            ops = e["fwd_ops_per_step"] + e["bwd_ops_per_step"]
            us = (e["fwd_ms_per_step"] + e["bwd_ms_per_step"]) * 1e3 / ops
            added += layers * steps[name] * ops * (us + 6.0) * 1e-6
        report["gpu_estimate_torch_path"][variant] = {"added_seconds_per_update": added,
                                                      "unpadded_minutes": report["updates_total"] * (base_s + added) / 60}
    out = ROOT / (args.out or ("artifacts/small-model/state-vq/qat/" + CHUNKINGS[chunk][1].name))
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_update", "per_step_cpu")}, indent=1, default=float))
    return 0


# ------------------------------------------------------------------------------------------------ GPU (bench, ticket gate)

BENCH_PATHS = ("old", "old+kbwd", "torch-p", "torch-p+kbwd", "cuda")


def bench_codecs(dev, recipe_name, factor_dtype, path):
    """Codecs {stream: codec} of one QAT path. old = the committed SM32b step (deployed eigh recipe, float64, torch
    loop); +kbwd = the same torch forward with the fused backward kernel; torch-p = the Gram-squaring recipe in the torch
    loop; cuda = the fused forward and backward kernels (the S09 run)."""
    import torch
    from scripts import state_vq
    deployed = recipe_name[:-1] if recipe_name.endswith("p") else recipe_name
    rname, fdt, impl = {"old": (deployed, "float64", "torch"), "old+kbwd": (deployed, "float64", "hybrid"),
                        "torch-p": (recipe_name, factor_dtype, "torch"),
                        "torch-p+kbwd": (recipe_name, factor_dtype, "hybrid"),
                        "cuda": (recipe_name, factor_dtype, "cuda")}[path]
    books = state_vq.Codebooks.load(state_vq.DEFAULT_CODEBOOKS)
    return {stream: state_vq.attach_kernel(state_vq.StateCodec(rec, books, 1 << 20, 8192, getattr(torch, fdt)).to(dev),
                                           impl, dev)
            for stream, rec in zip(state_vq.STATE_STREAMS, state_vq.recipe(rname))}


def _sequence_inputs(rows, length, coded, dev, gen, dtype):
    """Random WKV inputs of one split (r, k, v, w, a, kd leaves with grad; w float32), a skip mask with `coded`
    non-skipped row steps on average, and an output gradient."""
    import torch
    shape = (rows, length, 4, 16)
    r = torch.randn(shape, device=dev, generator=gen)
    k = torch.nn.functional.normalize(torch.randn(shape, device=dev, generator=gen), dim=-1) * 0.8
    v = torch.nn.functional.normalize(torch.randn(shape, device=dev, generator=gen), dim=-1) * 0.7
    w = torch.exp(-torch.exp(torch.randn(shape, device=dev, generator=gen) * 0.5 - 1.5))
    a = torch.sigmoid(torch.randn(shape, device=dev, generator=gen))
    kd = torch.nn.functional.normalize(torch.randn(shape, device=dev, generator=gen), dim=-1)
    skip = torch.rand(rows, length, device=dev, generator=gen) >= coded / max(rows * length, 1)
    leaves = ([t.to(dtype).requires_grad_(True) for t in (r, k, v)] + [w.requires_grad_(True)]
              + [t.to(dtype).requires_grad_(True) for t in (a, kd)])
    return leaves, skip, torch.randn(shape, device=dev, generator=gen).to(dtype)


def bench_update(update, codecs, dev, gen, plain=False):
    """Seconds of the card (2 layers) and note (1 layer) WKV forward + backward of one sampled update, all splits, in
    the trainer's form (bf16 student tensors; the QAT paths go through state_vq_hooks.vq_wkv, the plain one through
    the nvrtc RWKV7_WKV kernels the decay uses). Inputs are generated before the clock starts."""
    import torch
    from expertium.model import state_vq_hooks as hooks
    from expertium.model.rwkv_ops import RWKV7_WKV
    hooks.CODECS.clear()
    hooks.CODECS.update(codecs or {})
    cases = [(slot, layers, _sequence_inputs(rows, length, coded, dev, gen, torch.bfloat16))
             for slot, name, layers in ((0, "card", 2), (2, "note", 1)) for rows, length, coded in update[name]["splits"]]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for slot, layers, (leaves, skip, G) in cases:
        for _ in range(layers):
            if plain:
                out = RWKV7_WKV.apply(*leaves, skip)
            else:
                out = hooks.vq_wkv(*leaves, skip, slot)
            out.backward(G)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    hooks.CODECS.clear()
    return seconds


def bench_shift(update, codecs, dev, gen):
    """Seconds of the token-shift codec of one update (card: 2 time + 1 channel vector per row step, note: 1 + 1)."""
    import torch
    jobs = [(codecs[slot], role, rows * length) for slot, name, roles in ((0, "card", "ttc"), (2, "note", "tc"))
            for rows, length, _ in update[name]["splits"] for role in roles]
    xs = [torch.nn.functional.normalize(torch.randn(n, 64, device=dev, generator=gen), dim=-1) * 5.6 for _, _, n in jobs]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for (codec, role, _), x in zip(jobs, xs):
            codec.shift(x, role)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def gpu_correctness(dev, recipe_name, factor_dtype):
    """Fused kernels on CUDA vs the torch QAT path (same Gram-squaring codec) on CUDA and on the CPU, and vs the
    deployed eigh float64 codec (one-step consistency of the stored states); the backward kernel vs the explicit STE
    backward; the shift codec CUDA vs CPU."""
    import torch
    from dataclasses import replace
    from scripts import state_vq
    books = state_vq.Codebooks.load(state_vq.DEFAULT_CODEBOOKS)
    gen = torch.Generator().manual_seed(3)
    fd = getattr(torch, factor_dtype)
    out = {}
    for name, rec in zip(("card", "note"), state_vq.recipe(recipe_name)):
        kc = state_vq.attach_kernel(state_vq.StateCodec(rec, books, 1 << 20, 8192, fd).to(dev), "cuda", dev).kernel
        torch_gpu = state_vq.StateCodec(rec, books, 1 << 20, 8192, fd).to(dev)
        torch_cpu = state_vq.StateCodec(rec, books, 1 << 20, 8192, fd)
        deployed_cpu = state_vq.StateCodec(replace(rec, factor="eigh"), books)
        B, T, H, K = (4, 8, 2, 16) if DRY_RUN else (64, 24, 4, 16)
        rnd = lambda *s: torch.randn(*s, generator=gen)  # noqa: E731
        inputs = (rnd(B, T, H, K), torch.nn.functional.normalize(rnd(B, T, H, K), dim=-1) * 0.8,
                  torch.nn.functional.normalize(rnd(B, T, H, K), dim=-1) * 0.7,
                  torch.exp(-torch.exp(rnd(B, T, H, K) * 0.5 - 1.5)), torch.sigmoid(rnd(B, T, H, K)),
                  torch.nn.functional.normalize(rnd(B, T, H, K), dim=-1))
        skip = torch.rand(B, T, generator=gen) < 0.2
        skip[:, 0] = False
        G = rnd(B, T, H, K)
        on = lambda *x: [t.to(dev) for t in x]  # noqa: E731
        entry = {"vs_torch_cuda": compare_kernel_case(torch_gpu, tuple(on(*inputs)), skip.to(dev), G.to(dev),
                                                      kc.forward, kc.backward),
                 "vs_torch_cpu_and_deployed": compare_kernel_case(
                     torch_cpu, inputs, skip, G, lambda *x: kc.forward(*on(*x)), lambda *x: kc.backward(*on(*x)),
                     deployed_cpu)}
        x = torch.nn.functional.normalize(torch.randn(20000, 64, generator=gen), dim=-1) * 5.6
        entry["shift_rows_above_1e-4"] = {role: int(((torch_cpu.shift(x, role) - torch_gpu.shift(x.to(dev), role).cpu())
                                                     .abs().amax(1) > 1e-4).sum()) for role in ("t", "c")}
        entry["ok"] = (entry["vs_torch_cuda"]["ok"] and entry["vs_torch_cpu_and_deployed"]["ok"]
                       and all(v <= 20 for v in entry["shift_rows_above_1e-4"].values()))
        out[name] = entry
        print(json.dumps({name: entry}), flush=True)
    out["ok"] = all(v["ok"] for v in out.values())
    return out


def bench_chunking(chunking, paths, dev, recipe_name, factor_dtype, updates, slow_updates, cap_seconds=40.0):
    """Per-update seconds (mean over sampled updates) of the plain kernels and each QAT path, the shift codec, and
    the projected ticket minutes for the decay at `chunking`."""
    import numpy as np
    import torch
    decay_dir, splits_file, plain_total = CHUNKINGS[chunking]
    timing = json.loads(splits_file.read_text())
    if DRY_RUN:   # tiny splits for the CPU code-path test
        timing["per_update"] = [{"card": {"splits": [[2, 5, 6], [1, 3, 3]]}, "note": {"splits": [[2, 6, 8]]}}] * 3
    sample = timing["per_update"]
    n_updates = timing["updates_total"]
    base = plain_total / n_updates
    gen = torch.Generator(device=dev).manual_seed(5)
    res = {"updates_total": n_updates, "plain_decay_seconds_per_update": base,
           "steps_per_update": {n: timing[f"{n}_steps_per_update"]["mean"] for n in ("card", "note")}, "paths": {}}
    bench_update(sample[0], None, dev, gen, plain=True)                      # warm-up (kernels, allocator)
    res["plain_seconds"] = float(np.mean([bench_update(u, None, dev, gen, plain=True) for u in sample[:updates]]))
    shift_codecs = bench_codecs(dev, recipe_name, factor_dtype, "torch-p")
    bench_shift(sample[0], shift_codecs, dev, gen)
    res["shift_seconds"] = float(np.mean([bench_shift(u, shift_codecs, dev, gen) for u in sample[:updates]]))
    for path in paths:
        try:
            codecs = bench_codecs(dev, recipe_name, factor_dtype, path)
            torch.cuda.reset_peak_memory_stats(dev)
            first = bench_update(sample[0], codecs, dev, gen)                # warm-up (compile, cuSOLVER, first use)
            count = updates if path == "cuda" else slow_updates
            times = [] if first > cap_seconds else [bench_update(u, codecs, dev, gen) for u in sample[1:1 + count]]
            sec = float(np.mean(times)) if times else first
            added = sec - res["plain_seconds"] + res["shift_seconds"]
            entry = {"seconds_per_update": sec, "timed_updates": len(times) or 1, "warmup_seconds": first,
                     "added_seconds_per_update": added,
                     "peak_allocated_mib": torch.cuda.max_memory_allocated(dev) / 2**20,
                     "projected_ticket_minutes": n_updates * (base + added) / 60 + 5}
        except Exception as exc:  # noqa: BLE001  (the other paths still print)
            entry = {"error": f"{type(exc).__name__}: {exc}"}
        res["paths"][path] = entry
        print(f"[bench] {chunking} {path}: {json.dumps(entry)}", flush=True)
    return res


DRY_RUN = False


def _dry_run_setup():
    """--dry-run: the bench / cuda-codec code paths on the CPU with tiny shapes; the fused kernels run through the
    MSVC emulation (scripts/state_vq_cuda_emu.py), the plain WKV through the torch reference. No timing meaning."""
    global DRY_RUN
    import torch
    from expertium.model import rwkv_ops, state_vq_hooks
    from scripts import state_vq_cuda, state_vq_cuda_emu as emu
    DRY_RUN = True
    state_vq_hooks.KERNELS_ON_CPU = True
    for name in ("synchronize", "reset_peak_memory_stats"):
        setattr(torch.cuda, name, lambda *a, **k: None)
    torch.cuda.max_memory_allocated = lambda *a, **k: 0
    torch.cuda.get_device_name = lambda *a, **k: "cpu (dry run)"

    class Plain:
        @staticmethod
        def apply(r, k, v, w, a, kd, skip):
            return rwkv_ops.reference_rwkv7(r.float(), k.float(), v.float(), w, a.float(), kd.float(), skip).to(r.dtype)
    rwkv_ops.RWKV7_WKV = Plain
    state_vq_cuda.KernelCodec.forward = lambda self, *x: emu.forward(self, *x, factor_double=self.factor_dtype == torch.float64)
    state_vq_cuda.wkv_backward = emu.backward
    return torch.device("cpu")


def _gpu_setup():
    if DRY_RUN:
        return _dry_run_setup()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS, as the decay sets it
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("needs the GPU")
    torch.use_deterministic_algorithms(True)
    from rwkv.model import nvrtc
    nvrtc.register()
    nvrtc.set_checkpoint_length(32)
    nvrtc.set_max_timesteps(4 * 32768)   # the decay's limit at 32,768-review chunks
    return torch.device("cuda:0")


def main_bench(args):
    """GPU micro-benchmark (a short queue ticket, no training): correctness of the fused kernels on CUDA, then per
    update seconds of the plain kernels, the old torch step and the new paths at 32k x 1 and 2,048 x 16 (sampled
    updates of the two decays, timing.json / timing-2048.json), and the projected decay ticket lengths."""
    t_start = time.perf_counter()
    dev = _gpu_setup()
    import torch
    report = {"device": torch.cuda.get_device_name(0), "recipe": args.recipe, "factor_dtype": args.factor_dtype,
              "float32_matmul_precision": torch.get_float32_matmul_precision(),
              "head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()}
    report["correctness"] = gpu_correctness(dev, args.recipe, args.factor_dtype)
    report["chunkings"] = {c: bench_chunking(c, BENCH_PATHS, dev, args.recipe, args.factor_dtype, args.updates, 2)
                           for c in ("32k", "2048")}
    report["wall_seconds"] = time.perf_counter() - t_start
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n[bench] {report['device']}, recipe {args.recipe} ({args.factor_dtype}), correctness "
          f"{'OK' if report['correctness']['ok'] else 'FAILED'}; wall {report['wall_seconds']:.0f} s")
    print(f"{'chunking':9} {'path':13} {'ms/update':>10} {'added ms':>9} {'ticket min':>10}")
    for chunking, res in report["chunkings"].items():
        print(f"{chunking:9} {'plain':13} {res['plain_seconds'] * 1e3:10.0f} {'':9} "
              f"{res['updates_total'] * res['plain_decay_seconds_per_update'] / 60:10.0f}   "
              f"(plain decay {res['plain_decay_seconds_per_update']:.3f} s/update; shift codec "
              f"{res['shift_seconds'] * 1e3:.0f} ms/update, included in 'added')")
        for path, e in res["paths"].items():
            if "error" in e:
                print(f"{chunking:9} {path:13} ERROR {e['error'][:100]}")
            else:
                print(f"{chunking:9} {path:13} {e['seconds_per_update'] * 1e3:10.0f} "
                      f"{e['added_seconds_per_update'] * 1e3:9.0f} {e['projected_ticket_minutes']:10.0f}")
    return 0 if report["correctness"]["ok"] else 1


def main_cuda(args):
    """First command of the QAT decay ticket (about a minute): the fused kernels' correctness on CUDA (gpu_correctness)
    and the run's ticket length re-estimated from sampled updates of its chunking with its impl; exit 1 (the ticket
    stops before the smoke) on a mismatch or above --max-minutes."""
    dev = _gpu_setup()
    import torch
    report = {"device": torch.cuda.get_device_name(0), "recipe": args.recipe, "factor_dtype": args.factor_dtype,
              "impl": args.impl, "chunking": args.chunking}
    report["correctness"] = gpu_correctness(dev, args.recipe, args.factor_dtype) if args.impl == "cuda" else {"ok": True}
    sq = args.recipe.endswith("p")
    path = {"cuda": "cuda", "hybrid": "torch-p+kbwd" if sq else "old+kbwd", "torch": "torch-p" if sq else "old"}[args.impl]
    res = bench_chunking(args.chunking, (path,), dev, args.recipe, args.factor_dtype, min(args.updates, 4), 1)
    report["estimate"] = res
    minutes = res["paths"][path].get("projected_ticket_minutes", float("inf"))
    report["ok"] = report["correctness"]["ok"] and minutes <= args.max_minutes
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[cuda-codec] {path} at {args.chunking}: expected decay {minutes:.0f} min (cap {args.max_minutes:g})")
    if not report["correctness"]["ok"]:
        print("CUDA KERNEL MISMATCH", flush=True)
    elif minutes > args.max_minutes:
        print(f"CUDA CODEC OK, but {minutes:.0f} min > --max-minutes {args.max_minutes:g}: stopping before training", flush=True)
    else:
        print("CUDA CODEC OK", flush=True)
    return 0 if report["ok"] else 1


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all", "timing", "cuda-codec", "bench", "part-train", "part-forward",
                                            "part-capture"))
    parser.add_argument("--recipe", default=RUN_RECIPE)
    parser.add_argument("--out", default="")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--dump", default="")
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--state-vq", default="")
    parser.add_argument("--path", choices=("scorer", "training"))
    parser.add_argument("--users", type=int, nargs="+", default=[6])
    parser.add_argument("--chunks", type=int, default=40)
    parser.add_argument("--device", default="cuda:0", help="cuda-codec only (cpu = code-path test of the command)")
    parser.add_argument("--factor-dtype", default=RUN_FACTOR, choices=("float64", "float32"),
                        help="the run's factorization type (part-forward: of the codec of that path)")
    parser.add_argument("--impl", default=RUN_IMPL, choices=("torch", "hybrid", "cuda"), help="cuda-codec: the run's")
    parser.add_argument("--chunking", default="32k", choices=tuple(CHUNKINGS), help="cuda-codec: the run's")
    parser.add_argument("--rows", type=int, default=3, help="part-capture: rows per captured call")
    parser.add_argument("--steps", type=int, default=3, help="part-train: updates; part-capture: steps per row")
    parser.add_argument("--updates", type=int, default=8, help="bench / cuda-codec: sampled updates timed per path")
    parser.add_argument("--decay-dir", default=str(DEC10), help="timing: the decay whose updates are sampled")
    parser.add_argument("--dry-run", action="store_true", help="bench / cuda-codec: CPU code-path test (emulated kernels)")
    parser.add_argument("--max-minutes", type=float, default=720, help="cuda-codec: refuse (exit 1) above this estimate")
    args = parser.parse_args()
    if args.command in ("part-train", "part-forward", "part-capture"):
        sys.path.insert(0, args.root)
        if args.command == "part-train":
            return part_train(args)
        if args.command == "part-capture":
            return part_capture(args)
        return part_forward(args)
    sys.path.insert(0, str(ROOT))
    global DRY_RUN
    DRY_RUN = args.dry_run
    if args.command == "timing":
        return main_timing(args)
    if args.command == "cuda-codec":
        args.out = args.out or "artifacts/small-model/state-vq/qat/cuda-codec.json"
        return main_cuda(args)
    if args.command == "bench":
        args.out = args.out or "artifacts/small-model/state-vq/qat/bench.json"
        return main_bench(args)
    args.users = args.users[0] if len(args.users) == 1 else 6
    args.out = args.out or "artifacts/small-model/state-vq/qat/check.json"
    return main_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
