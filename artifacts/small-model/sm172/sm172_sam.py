#!/usr/bin/env python
"""Sharpness-aware cool-down (SM172): the SAM step for the run tree's trainer, and its CPU proofs.
Design: artifacts/small-model/sm172/design.md.

SAM (sharpness-aware minimization, plain form): per update, with g = the update's gradient at the weights w,
    e = rho * g / ||g||          (global 2-norm over every parameter that has a gradient)
    g_sam = gradient of the same update's loss at w + e     (same windows, same carried input states, same dropout seed)
    w <- optimizer step from w with g_sam                   (the weights are restored to w bitwise before the step)
The trainer's update is unchanged otherwise: the logged loss and gradient norm are those of the first pass (at w), the
carried card / note / deck / preset / user states written back after the update are those of the first pass (at w),
and g_sam goes through the trainer's own clip (CLIP) before the optimizer (Muon + AdamW, `expertium-iter53`) steps.
e is computed from the clipped first gradient: clipping only rescales g, so e is the same.

Placement (no edit of the run tree, as SM171): `install(child, rho, log)` (called from sm172_run.py through the
trainer's `state_vq` track option) patches two methods of `scripts.expertium_options.CatchUp`:
  * `loss`: the outer call of an update snapshots the carry table (a clone of every slot; the first pass then commits
    in place) and the attached entity rows, and records the call (model, batch, key, labels);
  * `optimizer`: the optimizer it returns gets an instance `step` that runs the second pass and then the stock step.
In the second pass the table is the snapshot (so every window reads the state as of the update start, as in the first
pass); its commits go to the snapshot, which is then dropped, and the first pass's table is put back. The SM171 hook
(installed at lambda 0, i.e. the control's code path) is switched off during the second pass, so its no-grad dropped-
share diagnostic samples the first-pass calls exactly as in the control.
Master / child: the trainer keeps fp32 master weights and a bfloat16 child (selectively cast). e is added to the
master, the child is downcast from it (copy_ rounds once, as MasterChildSync.downcast), the child's gradient is moved
to the master in fp32, then the master is restored and the child downcast again. With bfloat16 child weights a small
e is partly rounding: `<log>` records, every 16th update, the part of the downcast perturbation along e and the rest.

  python sm172_sam.py cpu [--out check-cpu-sm172.json]     CPU proofs S1-S2 (a few minutes, 4 threads, < 3 GB)
"""
from argparse import ArgumentParser
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SM171 = HERE.parent / "sm171"
CODE = Path(os.environ.get("SM172_CODE", "C:/Users/jonat/workspace/wt/sm172-run"))
CODE_COMMIT = "cea0f931"   # origin/sm148: the tree SM171 trained and scored in; detached, never edited
os.environ["SM171_CODE"] = str(CODE)
for p in (str(CODE), str(SM171)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

import sm171_energy  # noqa: E402

DIAG_EVERY = 16                      # survival diagnostic every 16th SAM step
SURVIVAL_RHOS = (0.01, 0.02, 0.05, 0.1, 0.2)
FLUSH_EVERY = 64                     # one host sync per 64 SAM steps for the log
CONFIG = {"rho": None, "log": None, "check": False}
STATE = {"master": None, "pairs": None, "dtype_of": None, "pending": None, "depth": 0}
STATS = {"steps": 0, "diag": 0, "second_pass_calls": 0}
CHECKS = []                          # CONFIG["check"]: per step (params at entry, params before the inner step) hashes
_ORIG = {}
_ACC = {}


def _tensor_hash(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update((t.detach().float() + 0.0).contiguous().cpu().numpy().tobytes())   # -0.0 hashed as +0.0
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ log

def _acc(name, value):
    _ACC[name] = _ACC.get(name, 0) + value.detach().float()


def flush():
    if not _ACC or not CONFIG["log"]:
        return
    n = float(_ACC.pop("n"))
    nd = float(_ACC.pop("nd", 0))
    rec = {"t": round(time.time(), 1), "rho": CONFIG["rho"], "stats": dict(STATS), "steps_in_record": int(n)}
    for key in ("grad_norm_clipped", "loss1", "loss2", "dloss", "first_order", "grad_norm2", "nonfinite_scale"):
        if key in _ACC:
            rec[key] = float(_ACC.pop(key)) / max(n, 1.0)
    if nd:
        rec["survival_updates"] = int(nd)
        rec["survival"] = {}
        for r in sorted({*SURVIVAL_RHOS, CONFIG["rho"]}):
            proj, noise = _ACC.pop(f"proj{r}", None), _ACC.pop(f"noise{r}", None)
            if proj is not None:
                rec["survival"][str(r)] = {"along_e": float(proj) / nd, "rest_over_rho": float(noise) / nd}
    _ACC.clear()
    with open(CONFIG["log"], "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------------------------------------ the hook

def _clone_slots(slots):
    return {key: {name: (t.clone() if isinstance(t, torch.Tensor) else t) for name, t in slot.items()}
            for key, slot in slots.items()}


def _loss(self, model, batch, reviews_done, key=None, labels=None):
    if CONFIG["rho"] is None or STATE["depth"] > 0:
        return _ORIG["loss"](self, model, batch, reviews_done, key, labels)
    carry = self.carry
    if carry is not None and carry.table.gpu_mib is not None:
        raise RuntimeError("SM172: the budgeted carry table (table_gpu_mib) is not covered")
    pending = {"catch_up": self, "model": model, "batch": batch, "reviews_done": reviews_done, "key": key,
               "labels": labels, "pre": _clone_slots(carry.table.slots) if carry is not None else None,
               "entities": list(carry.entities) if carry is not None and carry.entities is not None else None}
    STATE["depth"] += 1
    try:
        stats = _ORIG["loss"](self, model, batch, reviews_done, key, labels)
    finally:
        STATE["depth"] -= 1
    if stats is not None:
        pending["loss1"] = stats.average_loss.detach()
    STATE["pending"] = pending
    return stats


def _optimizer(self, master):
    optimizer = _ORIG["optimizer"](self, master)
    if CONFIG["rho"] is not None:
        STATE["master"] = master
        inner = optimizer.step
        optimizer.step = lambda closure=None: _sam_step(inner, closure)
    return optimizer


@contextmanager
def _energy_hook_off():
    """The SM171 hook (lambda 0 here) is arithmetic-neutral; off during the second pass, so that its diagnostic samples
    the same card / note calls as in the control."""
    from scripts import state_carry_train
    from rwkv.model import wkv_state as wkv_module
    on = bool(sm171_energy._ORIG) and state_carry_train.carry_layer is sm171_energy._carry_layer
    if on:
        if sm171_energy.CONFIG["lam"] != 0:
            raise RuntimeError("SM172: SAM runs with the SM171 hook at lambda 0 only")
        state_carry_train.carry_layer = sm171_energy._ORIG["carry_layer"]
        wkv_module.wkv_state = sm171_energy._ORIG["wkv_state"]
    try:
        yield
    finally:
        if on:
            state_carry_train.carry_layer = sm171_energy._carry_layer
            wkv_module.wkv_state = sm171_energy._wkv_state


def _setup_pairs(master, child):
    if child is master:
        STATE["pairs"] = None
        STATE["dtype_of"] = {id(p): p.dtype for p in master.parameters()}
        return
    masters = dict(master.named_parameters())
    names = [n for n, _ in child.named_parameters()]
    if sorted(names) != sorted(masters):
        raise RuntimeError("SM172: master and child parameter names differ")
    STATE["pairs"] = [(masters[n], c) for n, c in child.named_parameters()]
    STATE["dtype_of"] = {id(m): c.dtype for m, c in STATE["pairs"]}


@torch.no_grad()
def _downcast():
    groups = {}
    for m, c in STATE["pairs"] or ():
        sources, targets = groups.setdefault(c.dtype, ([], []))
        sources.append(m.data)
        targets.append(c.data)
    for sources, targets in groups.values():
        torch._foreach_copy_(targets, sources)


@torch.no_grad()
def _survival(params, saved, e, rho):
    """Per candidate radius r: the downcast perturbation d = cast(w + e r / rho) - cast(w) split into its part along
    e_r (coefficient, 1 = all of it) and the rest (norm / r)."""
    for r in sorted({*SURVIVAL_RHOS, rho}):
        dot = torch.zeros((), device=saved[0].device)
        dd = torch.zeros((), device=saved[0].device)
        for p, w, ep in zip(params, saved, e):
            dt = STATE["dtype_of"][id(p)]
            er = ep * (r / rho)
            d = (w + er).to(dt).float() - w.to(dt).float()
            dot += (d * er).sum()
            dd += (d * d).sum()
        proj = dot / (r * r)
        rest = torch.sqrt(torch.clamp(dd - dot * dot / (r * r), min=0)) / r
        _acc(f"proj{r}", proj)
        _acc(f"noise{r}", rest)


def _sam_step(inner, closure=None):
    if closure is not None:
        raise RuntimeError("SM172: the trainer steps without a closure")
    pending, STATE["pending"] = STATE["pending"], None
    if pending is None:
        raise RuntimeError("SM172: optimizer step without a recorded CatchUp.loss call (graphs / reduced_sync?)")
    from rwkv.train_rwkv import CLIP
    master, child, rho = STATE["master"], pending["model"], CONFIG["rho"]
    if STATE["dtype_of"] is None:
        _setup_pairs(master, child)
    if pending["key"] is None:
        raise RuntimeError("SM172: a batch without a key (seed) cannot be replayed")
    params = [p for p in master.parameters() if p.grad is not None]
    grads = [p.grad for p in params]
    check = CONFIG["check"]
    entry = _tensor_hash(master.parameters()) if check else None
    gnorm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
    scale = rho / (gnorm + 1e-12)
    finite = torch.isfinite(scale)
    scale = torch.where(finite, scale, torch.zeros_like(scale))
    with torch.no_grad():
        saved = [p.detach().clone() for p in params]
        e = torch._foreach_mul(grads, scale)
        torch._foreach_add_(params, e)
    diag = rho > 0 and STATS["steps"] % DIAG_EVERY == 0
    if diag:
        _survival(params, saved, e, rho)
        STATS["diag"] += 1
    _downcast()
    for c in child.parameters():
        c.grad = None
    for p in master.parameters():
        p.grad = None
    carry = pending["catch_up"].carry
    post = None
    if carry is not None:
        post = carry.table.slots
        carry.table.slots = pending["pre"]
        carry.entities = pending["entities"]
    torch.manual_seed(pending["key"][3])
    STATE["depth"] += 1
    try:
        with _energy_hook_off():
            stats = _ORIG["loss"](pending["catch_up"], child, pending["batch"], pending["reviews_done"],
                                  pending["key"], pending["labels"])
            if stats is None:
                raise RuntimeError("SM172: the second pass produced no loss")
            stats.average_loss.backward()
    finally:
        STATE["depth"] -= 1
        if carry is not None:
            carry.table.slots = post
    STATS["second_pass_calls"] += 1
    with torch.no_grad():
        if STATE["pairs"] is not None:
            for m, c in STATE["pairs"]:
                if c.grad is not None:
                    m.grad = c.grad.detach().to(m.dtype, copy=True)
                    m.grad.add_(0.0)   # as MasterChildSync.transfer_grads: -0.0 becomes +0.0
                    c.grad = None
    norm2 = torch.nn.utils.clip_grad_norm_(master.parameters(), CLIP, error_if_nonfinite=False)
    with torch.no_grad():
        torch._foreach_copy_(params, saved)
    if check:
        CHECKS.append({"entry": entry, "before_inner_step": _tensor_hash(master.parameters()),
                       "sam_grad": _tensor_hash(p.grad for p in master.parameters() if p.grad is not None),
                       "loss2": float(stats.average_loss.detach())})
    out = inner()
    _downcast()
    STATS["steps"] += 1
    _acc("n", torch.ones(()))
    if diag:
        _acc("nd", torch.ones(()))
    _acc("grad_norm_clipped", gnorm)
    _acc("grad_norm2", norm2)
    _acc("nonfinite_scale", (~finite).float())
    loss2 = stats.average_loss.detach()
    _acc("loss2", loss2)
    if "loss1" in pending:
        _acc("loss1", pending["loss1"])
        _acc("dloss", loss2.float() - pending["loss1"].float())
    _acc("first_order", rho * gnorm)
    if STATS["steps"] % FLUSH_EVERY == 0:
        flush()
    return out


def install(child, rho, log=None, check=False):
    """SAM of radius rho on the next CatchUp optimizer (child: the trainer's training model, for the record)."""
    from scripts.expertium_options import CatchUp
    rho = float(rho)
    if not rho >= 0:
        raise ValueError(f"SM172: rho {rho}")
    if "loss" not in _ORIG:
        _ORIG["loss"], _ORIG["optimizer"] = CatchUp.loss, CatchUp.optimizer
    CatchUp.loss, CatchUp.optimizer = _loss, _optimizer
    CONFIG.update(rho=rho, log=log, check=bool(check))
    STATE.update(master=None, pairs=None, dtype_of=None, pending=None, depth=0)
    for key in STATS:
        STATS[key] = 0
    CHECKS.clear()
    _ACC.clear()
    src = Path(__file__).resolve()
    return {"recipe": "sm172-sam", "rho": rho, "form": "plain SAM: e = rho g / ||g|| (global 2-norm, clipped first "
            "gradient), second pass at w + e on the same windows, carried input states and dropout seed; the weights "
            "are restored before the stock clip + optimizer step with the second gradient; carried states written "
            "back and the logged loss / norm are the first pass's", "survival_rhos": list(SURVIVAL_RHOS),
            "diag_every": DIAG_EVERY, "child_parameters": sum(p.numel() for p in child.parameters()),
            "hook": str(src), "hook_sha256_lf": hashlib.sha256(src.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "log": log}


def uninstall():
    from scripts.expertium_options import CatchUp
    if _ORIG:
        CatchUp.loss, CatchUp.optimizer = _ORIG["loss"], _ORIG["optimizer"]
    CONFIG.update(rho=None, log=None, check=False)


# ------------------------------------------------------------------------------------------------ CPU proofs

def _table_hash(carry):
    slots = carry.table.slots
    return _tensor_hash(t for key in sorted(slots) for name, t in sorted(slots[key].items())
                        if isinstance(t, torch.Tensor))


def _arm(arm, items, rows, options, lr, rho, state_dict):
    """One arm of the trainer's update on CPU with a separate master and child (MasterChildSync, as on the GPU), the
    SM171 hook at lambda 0 (the control's path) and SAM of radius rho (None = off, the control)."""
    from expertium.variants import build
    from scripts.expertium_options import CatchUp
    from scripts.gpu_r01_update import MasterChildSync
    from rwkv.train_rwkv import CLIP
    master = build("w64-c1-d2p1u1").float()
    master.load_state_dict(state_dict, strict=True)
    child = build("w64-c1-d2p1u1").float()
    child.load_state_dict(state_dict, strict=True)
    for key in sm171_energy.CALLS:
        sm171_energy.CALLS[key] = 0
    sm171_energy.install(child, 0.0, log=None)
    if rho is not None:
        install(child, rho, check=True)
    catch_up = CatchUp(options, {"lr_schedule": {"peak": lr, "warmup_updates": 1, "decay_start": 1.0,
                                                "decay_end": 1.0}}, rows)
    catch_up.load_teacher(torch.device("cpu"), torch.float32)
    optimizer = catch_up.optimizer(master)
    optimizer.load_state_dict(torch.load(sm171_energy.TINY_P2 / "expertium" / sm171_energy.TRACK / "optimizer.pth",
                                         map_location="cpu", weights_only=True))
    sync = MasterChildSync(master, child)
    steps, done = [], 0
    for u, item in items:
        batch = item.batch.to(torch.device("cpu"))
        optimizer.zero_grad(set_to_none=True)
        child.zero_grad(set_to_none=True)
        child.train()
        sync.downcast()
        torch.manual_seed(item.seed)
        catch_up.carry.attach(item, batch, u)
        stats = catch_up.loss(child, batch, done, (u, item.offset, item.reviews, item.seed), batch.labels)
        stats.average_loss.backward()
        sync.transfer_grads()
        g1 = _tensor_hash(p.grad for p in master.parameters() if p.grad is not None)
        norm = torch.nn.utils.clip_grad_norm_(master.parameters(), CLIP)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        done += item.reviews
        steps.append({"user": u, "offset": item.offset, "loss": float(stats.average_loss.detach()),
                      "grad_norm": float(norm), "first_grad": g1,
                      "step_grad": _tensor_hash(p.grad for p in master.parameters() if p.grad is not None),
                      "weights": _tensor_hash(master.parameters()), "carry": _table_hash(catch_up.carry),
                      "child_equals_master": all(torch.equal(m, c) for m, c in zip(master.parameters(),
                                                                                    child.parameters())),
                      "finite": all(bool(torch.isfinite(p).all()) for p in master.parameters())})
        if rho is not None:
            steps[-1].update(CHECKS[-1])
    out = {"steps": steps, "energy_calls": dict(sm171_energy.CALLS), "sam_stats": dict(STATS)}
    if rho is not None:
        uninstall()
    sm171_energy.uninstall()
    return out


def _reference_first(items, rows, options, rho, state_dict):
    """Independent second-pass gradient of the first update: a fresh model at w + rho g / ||g|| (g: the stock first
    gradient, clipped), a fresh carry table, the same seed; no hook of this file involved."""
    from expertium.variants import build
    from scripts.expertium_options import CatchUp
    from rwkv.train_rwkv import CLIP
    u, item = items[0]
    batch = item.batch.to(torch.device("cpu"))
    key = (u, item.offset, item.reviews, item.seed)
    model = build("w64-c1-d2p1u1").float()
    model.load_state_dict(state_dict, strict=True)
    model.train()
    catch_up = CatchUp(options, {"lr_schedule": {"peak": 7e-5, "warmup_updates": 1, "decay_start": 1.0,
                                                "decay_end": 1.0}}, rows)
    catch_up.load_teacher(torch.device("cpu"), torch.float32)
    torch.manual_seed(item.seed)
    catch_up.carry.attach(item, batch, u)
    stats = catch_up.loss(model, batch, 0, key, batch.labels)
    stats.average_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    params = [p for p in model.parameters() if p.grad is not None]
    grads = [p.grad.detach().clone() for p in params]
    gnorm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
    scale = rho / (gnorm + 1e-12)
    with torch.no_grad():
        torch._foreach_add_(params, torch._foreach_mul(grads, scale))
    model.zero_grad(set_to_none=True)
    fresh = CatchUp(options, {"lr_schedule": {"peak": 7e-5, "warmup_updates": 1, "decay_start": 1.0,
                                             "decay_end": 1.0}}, rows)
    fresh.load_teacher(torch.device("cpu"), torch.float32)
    torch.manual_seed(item.seed)
    fresh.carry.attach(item, batch, u)
    stats2 = fresh.loss(model, batch, 0, key, batch.labels)
    stats2.average_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    return {"sam_grad": _tensor_hash(p.grad for p in model.parameters() if p.grad is not None),
            "loss1": float(stats.average_loss.detach()), "loss2": float(stats2.average_loss.detach()),
            "first_order": float(rho * gnorm)}


def s1_real_updates(users, chunk, steps_per_user, rho, lr):
    """S1: six-layer Tiny two-head (SM130)'s release weights and Muon state, the release cool-down's options
    (strided-lane carry, show-time inputs, replay KD targets), short windows so that users have carried windows,
    master and child separate as on the GPU. Arms: control (SM171 hook at lambda 0, no SAM), SAM rho = 0, SAM rho.
    Rules: rho 0 = control bitwise on every update (loss, first and stepped gradients, weights, carry table);
    rho > 0: first update's loss and carry table = control's, weights restored bitwise before every inner step, the
    second-pass gradient of the first update = an independent reference bitwise, later weights differ, all finite,
    the first-pass hook calls (SM171 diagnostic) = control's."""
    from concurrent.futures import ThreadPoolExecutor
    from scripts.gpu_r01_prefetch import training_batches
    protocol, options = sm171_energy._setup(chunk)
    rows = [next(r for r in protocol["rows"] if r["user_id"] == u) for u in users]
    state_dict = torch.load(sm171_energy.TINY_P2 / "expertium" / sm171_energy.TRACK / "master.pth",
                            map_location="cpu", weights_only=True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        items = list(training_batches(rows, chunk, options["training_seed"], pool, options, None, 2))
    per_user, kept = {}, []
    for row, item in items:
        per_user[row["user_id"]] = per_user.get(row["user_id"], 0) + 1
        if per_user[row["user_id"]] <= steps_per_user:
            kept.append((row["user_id"], item))
    report = {"users": users, "chunk_reviews": chunk, "rho": rho, "updates": len(kept), "arms": {}}
    t = {}
    for arm, r in (("control", None), ("rho0", 0.0), ("rho", rho)):
        t0 = time.perf_counter()
        report["arms"][arm] = _arm(arm, kept, rows, options, lr, r, state_dict)
        t[arm] = round(time.perf_counter() - t0, 1)
    report["arm_seconds"] = t
    report["reference_first_update"] = _reference_first(kept, rows, options, rho, state_dict)
    c, z, p = (report["arms"][a]["steps"] for a in ("control", "rho0", "rho"))
    fields = ("loss", "first_grad", "step_grad", "weights", "carry")
    ok_rho0 = all(a[f] == b[f] for a, b in zip(c, z) for f in fields)
    ok_first = p[0]["loss"] == c[0]["loss"] and p[0]["carry"] == c[0]["carry"] and p[0]["first_grad"] == c[0]["first_grad"]
    ok_restore = all(s["entry"] == s["before_inner_step"] for s in p + z)
    ok_ref = p[0]["sam_grad"] == report["reference_first_update"]["sam_grad"]
    ok_diff = all(a["weights"] != b["weights"] for a, b in zip(c, p))
    ok_finite = (all(s["finite"] for s in c + z + p)
                 and all(s["child_equals_master"] for s in z + p))   # SAM re-downcasts; the stock path at the next update
    ok_calls = report["arms"]["rho"]["energy_calls"] == report["arms"]["control"]["energy_calls"]
    ok_second = report["arms"]["rho"]["sam_stats"]["second_pass_calls"] == len(kept)
    report["rules"] = {"rho0_equals_control": ok_rho0, "first_update_equals_control": ok_first,
                       "weights_restored_before_step": ok_restore, "sam_gradient_equals_reference": ok_ref,
                       "weights_differ": ok_diff, "finite_and_child_synced": ok_finite,
                       "first_pass_hook_calls_equal_control": ok_calls, "one_second_pass_per_update": ok_second}
    report["dloss_over_first_order"] = [round((s["loss2"] - s["loss"]) / max(1e-12, rho * s["grad_norm"]), 3)
                                        for s in p]
    return all(report["rules"].values()), report


def s2_bf16_survival(users, chunk, steps_per_user):
    """S2 (report, no rule): real first gradients of the release model on windows of up to 2,048 reviews (CPU, float32); for each
    candidate radius r, the bfloat16 downcast of w + r g / ||g|| as the GPU child sees it: the part of the perturbation
    along e (1 = all of it) and the rest (norm / r), with the trainer's selective cast (fp32 parameters stay exact)."""
    from concurrent.futures import ThreadPoolExecutor
    from expertium.variants import build
    from scripts.expertium_options import CatchUp
    from scripts.gpu_r01_prefetch import training_batches
    from rwkv.train_rwkv import CLIP
    protocol, options = sm171_energy._setup(chunk)
    rows = [next(r for r in protocol["rows"] if r["user_id"] == u) for u in users]
    state_dict = torch.load(sm171_energy.TINY_P2 / "expertium" / sm171_energy.TRACK / "master.pth",
                            map_location="cpu", weights_only=True)
    model = build("w64-c1-d2p1u1").float()
    model.load_state_dict(state_dict, strict=True)
    cast = build("w64-c1-d2p1u1").selective_cast(torch.bfloat16)
    dtype_of = {n: p.dtype for n, p in cast.named_parameters()}
    with ThreadPoolExecutor(max_workers=1) as pool:
        items = list(training_batches(rows, chunk, options["training_seed"], pool, options, None, 2))
    catch_up = CatchUp(options, {"lr_schedule": {"peak": 7e-5, "warmup_updates": 1, "decay_start": 1.0,
                                                "decay_end": 1.0}}, rows)
    catch_up.load_teacher(torch.device("cpu"), torch.float32)
    out, per_user = [], {}
    names = dict(model.named_parameters())
    for row, item in items:
        u = row["user_id"]
        per_user[u] = per_user.get(u, 0) + 1
        if per_user[u] > steps_per_user:
            continue
        batch = item.batch.to(torch.device("cpu"))
        model.zero_grad(set_to_none=True)
        model.train()
        torch.manual_seed(item.seed)
        catch_up.carry.attach(item, batch, u)
        stats = catch_up.loss(model, batch, 0, (u, item.offset, item.reviews, item.seed), batch.labels)
        stats.average_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        used = [(n, p) for n, p in names.items() if p.grad is not None]
        g = torch.cat([p.grad.reshape(-1) for _, p in used])
        rec = {"user": u, "offset": item.offset, "reviews": item.reviews, "grad_norm_clipped": float(g.norm()),
               "bf16_share_of_grad_energy": float(sum(p.grad.pow(2).sum() for n, p in used
                                                      if dtype_of[n] == torch.bfloat16) / g.pow(2).sum()),
               "survival": {}}
        for r in SURVIVAL_RHOS:
            dot = dd = 0.0
            for n, p in used:
                e = p.grad * (r / g.norm())
                w = p.detach()
                d = (w + e).to(dtype_of[n]).float() - w.to(dtype_of[n]).float()
                dot += float((d * e).sum())
                dd += float((d * d).sum())
            rec["survival"][str(r)] = {"along_e": round(dot / r ** 2, 3),
                                       "rest_over_rho": round(max(dd - dot * dot / r ** 2, 0) ** 0.5 / r, 3)}
        out.append(rec)
    return True, {"chunk_reviews": chunk, "users": users, "bf16_parameters": sum(
        p.numel() for n, p in cast.named_parameters() if p.dtype == torch.bfloat16),
        "parameters": sum(p.numel() for p in cast.parameters()), "updates": out}


def main_cpu(args):
    os.environ["RWKV_NO_JIT"] = "1"
    os.environ["GPU_BATCH1_DATA_PATH"] = sm171_energy.DATA
    os.environ["RWKV_DELETED_CARDS"] = "neighbour"
    torch.set_num_threads(4)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "scripts", "rwkv", "expertium"], cwd=CODE,
                           capture_output=True, text=True).stdout.strip()
    src = Path(__file__).resolve()
    report = {"code_tree": str(CODE), "code_head": head, "code_dirty": bool(dirty), "torch": torch.__version__,
              "hook_sha256_lf": hashlib.sha256(src.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
              "sm171_hook_sha256_lf": hashlib.sha256(Path(sm171_energy.__file__).read_bytes().replace(
                  b"\r\n", b"\n")).hexdigest(), "checks": {}}
    users = [int(u) for u in args.users.split(",")]
    started = time.perf_counter()
    for name, fn in (("S1_tiny_two_head_real_updates", lambda: s1_real_updates(users, args.chunk, args.steps,
                                                                                args.rho, args.lr)),
                     ("S2_bf16_survival_report", lambda: s2_bf16_survival(users, args.survival_chunk, 2))):
        t0 = time.perf_counter()
        ok, detail = fn()
        report["checks"][name] = {"pass": bool(ok), "seconds": round(time.perf_counter() - t0, 1), "detail": detail}
        print(f"[sm172] {name}: {'OK' if ok else 'FAIL'} ({time.perf_counter() - t0:.0f} s)", flush=True)
    report["all_ok"] = (all(c["pass"] for c in report["checks"].values()) and not report["code_dirty"]
                        and head.startswith(CODE_COMMIT)
                        and report["sm171_hook_sha256_lf"].startswith("388d83e0"))
    report["seconds"] = round(time.perf_counter() - started, 1)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[sm172] CPU proofs {'ALL OK' if report['all_ok'] else 'NOT OK'} -> {args.out}")
    return 0 if report["all_ok"] else 1


def main():
    p = ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cpu")
    c.add_argument("--out", default=str(HERE / "check-cpu-sm172.json"))
    c.add_argument("--users", default="4745,2308")
    c.add_argument("--chunk", type=int, default=256)
    c.add_argument("--steps", type=int, default=3)
    c.add_argument("--rho", type=float, default=0.05)
    c.add_argument("--lr", type=float, default=7e-5)
    c.add_argument("--survival-chunk", type=int, default=2048)
    args = p.parse_args()
    return main_cpu(args)


if __name__ == "__main__":
    raise SystemExit(main())
