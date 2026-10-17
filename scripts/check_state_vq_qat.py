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
        info = install_training(base, args.state_vq, torch.device("cpu"))
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
        state_vq.configure(layout, {s: state_vq.StateCodec(rec, books)
                                    for s, rec in zip(layout, state_vq.recipe(args.recipe))})
    else:
        torch._C._set_graph_executor_optimize(False)
        from expertium.variants import build
        model = build("w64").float()
        model.load_state_dict(w8, strict=True)
        state_vq.install_training(model, args.recipe, torch.device("cpu"))
    run = cc.run_info()
    out = {"path": args.path, "jit": isinstance(model, torch.jit.ScriptModule), "users": {}}
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


def main_all(args):
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    logs = OUT / "check-logs"
    logs.mkdir(exist_ok=True)
    report = {"recipe": args.recipe, "codebooks_sha256": None, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    from scripts import state_vq
    report["codebooks_sha256"] = state_vq.file_sha256(state_vq.DEFAULT_CODEBOOKS)
    report["head"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    report["dirty"] = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", "scripts",
                                                    "expertium", "rwkv"], text=True).strip())
    # (c1) explicit backward vs autograd, shift STE; KD key
    report["c_grad_check"] = grad_check(args.recipe)
    report["a_kd_cache"] = kd_cache_check()
    print(json.dumps({"c_grad_check": report["c_grad_check"], "a_kd_cache": report["a_kd_cache"]}), flush=True)
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
        sec = run([PY, "-X", "utf8", me, "part-train", "--root", str(ROOT), "--dump", str(dump), "--steps", str(args.steps),
                   "--chunk", str(args.chunk), "--state-vq", args.recipe], {}, logs / "train-jit-qat.log")
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
    for path, env in (("scorer", {"RWKV_NO_JIT": "1"}), ("training", {})):
        dump = OUT / f"check-forward-{path}.json"
        sec = run([PY, "-X", "utf8", me, "part-forward", "--path", path, "--recipe", args.recipe, "--dump", str(dump),
                   "--users", *map(str, users)], env, logs / f"forward-{path}.log")
        fwd[path] = json.loads(dump.read_text())
        print(f"[check] forward {path}: {sec:.0f}s", flush=True)
    b = compare_forward(fwd["scorer"], fwd["training"])
    b["users"] = {str(u): counts[str(u)] for u in users}
    b["training_path_scripted"] = fwd["training"]["jit"]
    report["b_forward_vs_scorer"] = b
    oks = {"a": all(v["ok"] for v in a.values()) and report["a_kd_cache"]["ok"], "b": b["ok"],
           "c": all(v["ok"] for v in report["c_grad_check"].values()) and c2["ok"]}
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


def main_timing(args):
    import random
    import numpy as np
    import torch
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    from expertium.model.rwkv_ops import reference_rwkv7
    from expertium.model.state_vq_hooks import VQStateWKV
    from scripts import state_vq
    books = state_vq.Codebooks.load(state_vq.DEFAULT_CODEBOOKS)
    codecs = {name: state_vq.StateCodec(rec, books, 1 << 20, 8192) for name, rec in zip(("card", "note"),
                                                                                       state_vq.recipe(args.recipe))}
    report = {"recipe": args.recipe, "threads": 4}
    # 1. loop steps per 32,768-review update over the decay's own rows (sampled chunks)
    protocol, options = decay_setup(32768)
    from scripts.gpu_batch1 import prepare_job
    chunks = [(row, off) for row in protocol["rows"]
              for off in range(0, row["train_range"][1] - row["train_range"][0], 32768)]
    random.Random(1).shuffle(chunks)
    per = []
    for row, off in chunks[:args.chunks]:
        end = min(off + 32768, row["train_range"][1] - row["train_range"][0])
        batch = prepare_job(row, off, end, 2025 + row["user_id"] + off, options).batch
        entry = {"user": row["user_id"], "reviews": end - off}
        for s, name in ((0, "card"), (2, "note")):
            lens = [int(x) for x in batch.sub_gather_lens[s]]
            entry[name] = {"steps": sum(lens), "splits": [[int(g.numel() // l), l] for g, l in zip(batch.sub_gather[s], lens)]}
        per.append(entry)
    card_steps = np.array([e["card"]["steps"] for e in per], dtype=float)
    note_steps = np.array([e["note"]["steps"] for e in per], dtype=float)
    report["updates_total"] = len(chunks)
    report["sampled_updates"] = len(per)
    report["card_steps_per_update"] = {"mean": card_steps.mean(), "p90": float(np.percentile(card_steps, 90)), "max": card_steps.max()}
    report["note_steps_per_update"] = {"mean": note_steps.mean(), "p90": float(np.percentile(note_steps, 90)), "max": note_steps.max()}
    report["per_update"] = per
    # 2. per-step CPU cost and op counts, forward + backward, at split shapes seen above
    counter = OpCounter()
    shapes = [("card", 4000, 17), ("card", 800, 50), ("note", 2000, 15), ("note", 200, 283), ("note", 50, 800),
              ("card", 1, 64), ("note", 1, 64)]
    steps_cost = []
    for name, B, T in shapes:
        gen = torch.Generator().manual_seed(B * 1000 + T)
        H, K = 4, 16
        r = torch.randn(B, T, H, K, generator=gen)
        k = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=gen), dim=-1) * 0.8
        v = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=gen), dim=-1) * 0.7
        w = torch.exp(-torch.exp(torch.randn(B, T, H, K, generator=gen) * 0.5 - 1.5))
        a = torch.sigmoid(torch.randn(B, T, H, K, generator=gen))
        kd = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=gen), dim=-1)
        skip = torch.zeros(B, T, dtype=torch.bool)
        leaves = [t.requires_grad_(True) for t in (r, k, v, w, a, kd)]
        t0 = time.perf_counter()
        out = VQStateWKV.apply(*leaves, skip, codecs[name])
        t1 = time.perf_counter()
        out.sum().backward()
        t2 = time.perf_counter()
        with torch.no_grad():
            t3 = time.perf_counter()
            reference_rwkv7(r, k, v, w, a, kd, skip)
            t4 = time.perf_counter()
        with counter.Mode():
            counter.Mode.count = 0
            o = VQStateWKV.apply(*[t[:, :4] for t in leaves], skip[:, :4], codecs[name])
            fwd_ops = counter.Mode.count / 4
            counter.Mode.count = 0
            o.sum().backward()
            bwd_ops = counter.Mode.count / 4
            counter.Mode.count = 0
            with torch.no_grad():
                reference_rwkv7(*[t[:, :4] for t in leaves], skip[:, :4])
            ref_ops = counter.Mode.count / 4
        steps_cost.append({"stream": name, "rows": B, "T": T, "fwd_ms_per_step": (t1 - t0) / T * 1e3,
                           "bwd_ms_per_step": (t2 - t1) / T * 1e3, "plain_fwd_ms_per_step": (t4 - t3) / T * 1e3,
                           "fwd_ops_per_step": fwd_ops, "bwd_ops_per_step": bwd_ops, "plain_ops_per_step": ref_ops})
        print(json.dumps(steps_cost[-1]), flush=True)
    report["per_step_cpu"] = steps_cost
    # 3. shift codec cost per update (vectorized over all rows of a split; CPU)
    x = torch.randn(32768 * 3, 64)
    t0 = time.perf_counter()
    codecs["card"].shift(x, "t")
    report["shift_codec_cpu_seconds_per_98k_vectors"] = time.perf_counter() - t0
    # 4. GPU estimate: per step = host ops x (CPU per-op cost at 1 row + launch); eigh syncs every step
    tiny = {e["stream"]: e for e in steps_cost if e["rows"] == 1}
    est = {}
    for name in ("card", "note"):
        e = tiny[name]
        host_us = (e["fwd_ms_per_step"] + e["bwd_ms_per_step"]) * 1e3 / (e["fwd_ops_per_step"] + e["bwd_ops_per_step"])
        est[name] = {"cpu_us_per_op_tiny": host_us, "ops_per_step": e["fwd_ops_per_step"] + e["bwd_ops_per_step"]}
    launch_us = 6.0
    layers = {"card": 2, "note": 1}
    steps = {"card": report["card_steps_per_update"]["mean"], "note": report["note_steps_per_update"]["mean"]}
    added = sum(layers[n] * steps[n] * est[n]["ops_per_step"] * (est[n]["cpu_us_per_op_tiny"] + launch_us) * 1e-6
                for n in ("card", "note"))
    base_s = 3864.9 / 3465
    report["gpu_estimate"] = {"assumed_launch_us_per_op": launch_us, "per_stream": est,
                              "added_host_seconds_per_update": added, "dec10_seconds_per_update": base_s,
                              "updates": 3465, "unpadded_minutes": 3465 * (base_s + added) / 60}
    (ROOT / args.out).write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_update",)}, indent=1, default=float))
    return 0


def main_cuda(args):
    """GPU part of the ticket (seconds, before the smoke): the codec and VQStateWKV on CUDA vs CPU, and the CUDA
    per-step time of the QAT recurrence at the decay's split shapes. Exit 1 stops the ticket before training."""
    # deterministic cuBLAS (as the decay's protocol sets it) before the first CUDA call
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from expertium.model.state_vq_hooks import VQStateWKV
    from scripts import state_vq
    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda-codec needs the GPU")
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    torch.use_deterministic_algorithms(True)
    books = state_vq.Codebooks.load(state_vq.DEFAULT_CODEBOOKS)
    gen = torch.Generator().manual_seed(3)
    report, ok = {"recipe": args.recipe, "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}, True
    for name, rec in zip(("card", "note"), state_vq.recipe(args.recipe)):
        cpu = state_vq.StateCodec(rec, books, 1 << 20, 8192)   # the scorer's float64 factorization
        gpu = state_vq.StateCodec(rec, books, 1 << 20, 8192, getattr(torch, args.factor_dtype)).to(dev)
        # states with a decaying spectrum (sum of outer products), like trained card/note WKV states
        n, H, K = 4096, 4, 16
        u = torch.nn.functional.normalize(torch.randn(n, H, 6, K, generator=gen), dim=-1)
        v = torch.nn.functional.normalize(torch.randn(n, H, 6, K, generator=gen), dim=-1)
        c = 0.3 * torch.exp(-torch.arange(6.0) * 1.2) * torch.exp(torch.randn(n, H, 6, generator=gen) * 0.3)
        S = torch.einsum("nhr,nhri,nhrj->nhij", c, u, v)
        rw_cpu, rw_gpu = cpu.wkv(S), gpu.wkv(S.to(dev)).cpu()
        d = (rw_cpu - rw_gpu).flatten(1).abs().amax(1)
        x = torch.nn.functional.normalize(torch.randn(20000, 64, generator=gen), dim=-1) * 5.6
        sh = {role: (cpu.shift(x, role) - gpu.shift(x.to(dev), role).cpu()).abs().amax(1) for role in ("t", "c")}
        B, T = 64, 20
        r = torch.randn(B, T, H, K, generator=gen)
        kk = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=gen), dim=-1) * 0.8
        vv = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=gen), dim=-1) * 0.7
        w = torch.exp(-torch.exp(torch.randn(B, T, H, K, generator=gen) * 0.5 - 1.5))
        a = torch.sigmoid(torch.randn(B, T, H, K, generator=gen))
        kd = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=gen), dim=-1)
        skip = torch.rand(B, T, generator=gen) < 0.2
        skip[:, 0] = False
        outs = []
        for device, codec in (("cpu", cpu), (dev, gpu)):
            leaves = [t.to(device).requires_grad_(True) for t in (r, kk, vv, w, a, kd)]
            o = VQStateWKV.apply(*leaves, skip.to(device), codec)
            o.square().sum().backward()
            outs.append((o.detach().cpu(), [t.grad.cpu() for t in leaves]))
        out_diff = float((outs[0][0] - outs[1][0]).abs().max())
        grad_rel = max(float((x1 - x2).abs().max() / x1.abs().max().clamp_min(1e-30)) for x1, x2 in zip(outs[0][1], outs[1][1]))
        entry = {"wkv_rows": int(d.numel()), "wkv_rows_above_1e-4": int((d > 1e-4).sum()), "wkv_max_abs_diff": float(d.max()),
                 "shift_rows_above_1e-4": {k: int((s > 1e-4).sum()) for k, s in sh.items()},
                 "recurrence_out_max_abs_diff": out_diff, "recurrence_grad_max_rel_diff": grad_rel}
        entry["ok"] = (entry["wkv_rows_above_1e-4"] <= 0.001 * d.numel()
                       and all(v <= 0.001 * x.shape[0] for v in entry["shift_rows_above_1e-4"].values()))
        ok = ok and entry["ok"]
        # per-step time of the QAT recurrence (forward + backward) vs split rows, both factorization dtypes
        entry["per_step_ms"] = {}
        for dtype_name in ("float64", "float32"):
            codec = state_vq.StateCodec(rec, books, 1 << 20, 8192, getattr(torch, dtype_name)).to(dev)
            row = {}
            for rows in (1, 50, 200, 800, 4000, 8000):
                steps = 12
                leaves = [torch.randn(rows, steps, H, K, device=dev) for _ in range(6)]
                with torch.no_grad():
                    for i in (3, 4):
                        leaves[i] = torch.sigmoid(leaves[i])
                    for i in (1, 2, 5):
                        leaves[i] = torch.nn.functional.normalize(leaves[i], dim=-1)
                leaves = [t.requires_grad_(True) for t in leaves]
                sk = torch.zeros(rows, steps, dtype=torch.bool, device=dev)
                for rep in range(2):   # first pass warms up kernels / cuSOLVER handles
                    sync()
                    t0 = time.perf_counter()
                    o = VQStateWKV.apply(*leaves, sk, codec)
                    o.sum().backward()
                    sync()
                    row[rows] = (time.perf_counter() - t0) / steps * 1e3
            entry["per_step_ms"][dtype_name] = row
        report[name] = entry
        print(json.dumps({name: entry}), flush=True)
    # expected ticket length: sampled updates of the decay (timing.json, CPU) x measured per-step time
    import numpy as np
    timing = json.loads((OUT / "timing.json").read_text())
    base_s = timing["gpu_estimate"]["dec10_seconds_per_update"]
    est = {}
    for dtype_name in ("float64", "float32"):
        per_update = []
        for upd in timing["per_update"]:
            sec = 0.0
            for name, layers in (("card", 2), ("note", 1)):
                xs = sorted(report[name]["per_step_ms"][dtype_name])
                ys = [report[name]["per_step_ms"][dtype_name][x] for x in xs]
                for rows, length in upd[name]["splits"]:
                    sec += layers * length * float(np.interp(np.log(max(rows, 1)), np.log(xs), ys)) / 1e3
            per_update.append(sec)
        added = float(np.mean(per_update))
        est[dtype_name] = {"added_seconds_per_update": added,
                           "minutes": timing["updates_total"] * (base_s + added) / 60 + 5}
    report["estimate"] = est
    report["factor_dtype"] = args.factor_dtype
    minutes = est[args.factor_dtype]["minutes"]
    report["ok"] = ok and minutes <= args.max_minutes
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[cuda-codec] expected decay: {est['float64']['minutes']:.0f} min with float64 factorization, "
          f"{est['float32']['minutes']:.0f} min with float32 (this run: {args.factor_dtype}, cap {args.max_minutes:g} min)")
    if not ok:
        print("CUDA CODEC MISMATCH", flush=True)
    elif minutes > args.max_minutes:
        print(f"CUDA CODEC OK, but {minutes:.0f} min > --max-minutes {args.max_minutes:g}: stopping before training", flush=True)
    else:
        print("CUDA CODEC OK", flush=True)
    return 0 if report["ok"] else 1


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all", "timing", "cuda-codec", "part-train", "part-forward"))
    parser.add_argument("--recipe", default="vq378x412")
    parser.add_argument("--out", default="")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--dump", default="")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--state-vq", default="")
    parser.add_argument("--path", choices=("scorer", "training"))
    parser.add_argument("--users", type=int, nargs="+", default=[3])
    parser.add_argument("--chunks", type=int, default=40)
    parser.add_argument("--device", default="cuda:0", help="cuda-codec only (cpu = code-path test of the command)")
    parser.add_argument("--factor-dtype", default="float64", choices=("float64", "float32"), help="cuda-codec: the run's")
    parser.add_argument("--max-minutes", type=float, default=720, help="cuda-codec: refuse (exit 1) above this estimate")
    args = parser.parse_args()
    if args.command in ("part-train", "part-forward"):
        sys.path.insert(0, args.root)
        if args.command == "part-train":
            return part_train(args)
        return part_forward(args)
    sys.path.insert(0, str(ROOT))
    if args.command == "timing":
        args.out = args.out or "artifacts/small-model/state-vq/qat/timing.json"
        return main_timing(args)
    if args.command == "cuda-codec":
        args.out = args.out or "artifacts/small-model/state-vq/qat/cuda-codec.json"
        return main_cuda(args)
    args.users = args.users[0] if len(args.users) == 1 else 3
    args.out = args.out or "artifacts/small-model/state-vq/qat/check.json"
    return main_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
