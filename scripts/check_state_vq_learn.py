"""CPU check of SM32c: state-VQ QAT with learned catalogs and carried state (scripts/state_vq.py LearnedBooks,
expertium/model/state_vq_hooks.py VQStateWKVLearned / vq_shift_mask / vq_wkv_state, scripts/state_vq_cuda.py record +
dstored kernels, scripts/state_carry_train.py carry_layer). No GPU: the CUDA sources run through the MSVC emulation.

  K1  fused forward kernel (emulated): record=False is bitwise the SM32b launch (record=True gives the same out and
      stored states); from a carried s0 the kernel's stored states agree one step at a time with the torch codec and
      its picks / q(n)^2 equal the torch encode of the kernel's own previous state; `final` = the last stored state
  K2  dstored backward copy (emulated): the six input gradients are bitwise the original kernel's; dstored equals the
      explicit straight-through backward's dL/d(stored) (rel < 1e-4)
  G1  catalog gradient, torch path: VQStateWKVLearned (inputs and catalog) vs autograd through a loop whose stored
      state is q(n)^2 (sum of picked rows)_u (..)_v^T with the picks frozen (rel < 1e-5), from zero and from s0
  G2  catalog gradient, kernel path (emulated forward + dstored backward + codebook_grad) vs the same reference
  S1  learned shift: value bitwise the frozen codec's; gradient to x = identity on every row; catalog gradient = the
      scattered (G * scale) of the picked rows
  E1  export: LearnedBooks.export -> Codebooks.load -> StateCodec gives the learned values; a changed catalog changes
      the file
  T1  trainer path on real users (gate users of the plain 32k run, epoch 10 checkpoint, chunk 2,048 x 2 windows, strided
      lanes, float32 eager CPU): state_vq {"recipe": "vq378x412p", "learn": true} with state_carry: every update's loss
      is finite; the WKV and both shift catalogs get finite non-zero gradients; card/note slots of the table are float32
      and hold rank-1 states; an AdamW group step moves the catalogs; with frozen catalogs (learn absent) the first
      update's loss equals the learned run's (same catalog values) within 1e-6
Prints ALL OK and writes artifacts/small-model/check-state-vq-learn.json.

  .venv/Scripts/python.exe -X utf8 scripts/check_state_vq_learn.py [--users 1340] [--skip-trainer]
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = os.environ.get("GPU_BATCH1_DATA_PATH", r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
RUN = Path("C:/Users/jonat/workspace/wt/small-model/artifacts/accuracy/experiments/scale-w64kd-gate-cached-ctx32768-cont")
CKPT = RUN / "run" / "epoch-10" / "expertium" / "E89-F65e-F77a" / "master.pth"
RECIPE = "vq378x412p"


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def synthetic(gen, B, T, H, K=16, dtype=None):
    import torch
    rnd = lambda *s: torch.randn(*s, generator=gen, dtype=dtype or torch.float32)  # noqa: E731
    inputs = (rnd(B, T, H, K), torch.nn.functional.normalize(rnd(B, T, H, K), dim=-1) * 0.8,
              torch.nn.functional.normalize(rnd(B, T, H, K), dim=-1) * 0.7,
              torch.exp(-torch.exp(rnd(B, T, H, K) * 0.5 - 1.5)), torch.sigmoid(rnd(B, T, H, K)),
              torch.nn.functional.normalize(rnd(B, T, H, K), dim=-1))
    skip = torch.rand(B, T, generator=gen) < 0.25
    skip[:, 0] = False
    return inputs, skip


def reference(inputs, skip, s0, codec, books):
    """Autograd reference: per step, picks / q(n)^2 from the codec on the detached update (frozen), stored state =
    c (sum_s books[s][pick])_u (..)_v^T + (nxt - nxt.detach()) (straight-through to the update)."""
    import torch
    from expertium.model.rwkv_ops import single_timestep
    r, k, v, w, a, kd = inputs
    B, T, H, K = r.shape
    state = r.new_zeros(B, H, K, K) if s0 is None else s0.clone()
    outs = []
    for t in range(T):
        out_t, nxt = single_timestep(r[:, t], k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t], state)
        outs.append(out_t)
        _, picks, c = codec.encode_wkv(nxt.detach())
        acc = sum(books[s][picks[..., s]] for s in range(picks.shape[-1]))        # [B, H, 32]
        stored = c[..., None, None] * acc[..., :K, None] * acc[..., None, K:] + (nxt - nxt.detach())
        state = torch.where(skip[:, t, None, None, None], state, stored)
    return torch.stack(outs, dim=1)


def coded_s0(gen, codec, B, H, K=16):
    """A carried state as the trainer stores it: the codec's reconstruction of a random state."""
    import torch
    S = torch.randn(B, H, K, K, generator=gen) * 0.3
    return codec.encode_wkv(S)[0].contiguous()


def unit_checks(report, fails):
    import torch
    from expertium.model import state_vq_hooks as hooks
    from expertium.model.rwkv_ops import single_timestep
    from expertium.model.state_vq_hooks import VQStateWKVLearned, ste_backward
    from scripts import state_vq, state_vq_cuda, state_vq_cuda_emu as emu
    books = state_vq.Codebooks.load(state_vq.DEFAULT_CODEBOOKS)
    recs = state_vq.recipe(RECIPE)
    gen = torch.Generator().manual_seed(32)
    learned = state_vq.LearnedBooks(books, recs, "cpu")
    with torch.no_grad():   # move the catalogs off the k-means file, so "learned values are used" is visible
        learned.wkv.add_(0.01 * torch.randn(learned.wkv.shape, generator=gen))
        for p in learned.shift.values():
            p.add_(0.01 * torch.randn(p.shape, generator=gen))
    have_emu = emu.available()
    for name, rec in zip(("card", "note"), recs):
        codec = state_vq.StateCodec(rec, books, 1 << 20, 8192, torch.float32)
        codec.learned = learned
        kc = state_vq_cuda.KernelCodec(codec, torch.device("cpu"), torch.float32)
        codec.kernel, codec.impl = kc, "torch"
        codec.sync()
        if not torch.equal(kc.books, learned.wkv[:rec.stages[0]].detach().transpose(1, 2).contiguous()):
            fails.append(f"{name}: KernelCodec.refresh did not take the learned catalog")
        B, T, H = 3, 14, 2
        inputs, skip = synthetic(gen, B, T, H)
        s0 = coded_s0(gen, codec, B, H)
        G = torch.randn(B, T, H, 16, generator=gen)
        entry = {}
        # ---- K1 / K2 (emulated kernels)
        if have_emu:
            out0, before0 = emu.forward(kc, *inputs, skip)
            out1, before1, final1, picks1, cs1 = emu.forward(kc, *inputs, skip, record=True)
            k1_same = bool(torch.equal(out0, out1) and torch.equal(before0, before1))
            out2, before2, final2, picks2, cs2 = emu.forward(kc, *inputs, skip, s0=s0, record=True)
            r, k, v, w, a, kd = inputs
            mism, pick_mism, coded = 0, 0, 0
            for t in range(T):
                _, nxt = single_timestep(r[:, t], k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t], before2[:, t])
                rec_t, pk, cs = codec.encode_wkv(nxt)
                nxt_state = before2[:, t + 1] if t + 1 < T else final2
                for b in range(B):
                    if skip[b, t]:
                        if not torch.equal(nxt_state[b], before2[b, t]) or (picks2[b, t] != -1).any():
                            mism += 1
                        continue
                    coded += H
                    mism += int(((rec_t[b] - nxt_state[b]).flatten(1).abs().amax(-1) > 1e-5).sum())
                    pick_mism += int((pk[b].to(torch.int32) != picks2[b, t]).any(-1).sum())
                    pick_mism += int(((cs[b] - cs2[b, t]).abs() > 1e-6 * cs[b].abs().clamp_min(1e-30)).sum())
            k1 = {"record_off_equals_on_bitwise": k1_same, "s0_one_step_mismatch": mism, "pick_or_scale_mismatch": pick_mism,
                  "coded_head_steps": coded, "s0_used": bool(torch.equal(before2[:, 0], s0))}
            k1["ok"] = k1_same and mism <= max(1, coded // 1000) and pick_mism <= max(1, coded // 1000) and k1["s0_used"]
            grads_old = emu.backward(*inputs, skip, before2, G)
            grads_new, ds = emu.backward_dstored(*inputs, skip, before2, G)
            ds_ref = torch.zeros(T, B, H, 16, 16)
            ste_backward(*inputs, skip, before2.transpose(0, 1), G, dstored=ds_ref)
            k2 = {"grads_bitwise_original": all(torch.equal(x, y) for x, y in zip(grads_old, grads_new)),
                  "dstored_rel_diff": rel(ds, ds_ref.transpose(0, 1)),
                  "dstored_zero_on_skipped": bool((ds[skip] == 0).all())}
            k2["ok"] = k2["grads_bitwise_original"] and k2["dstored_rel_diff"] < 1e-4 and k2["dstored_zero_on_skipped"]
            entry["K1_forward_kernel"], entry["K2_dstored_backward"] = k1, k2
        # ---- G1 / G2: catalog gradients
        for start in ("zero", "s0"):
            init = None if start == "zero" else s0
            for path in ("torch",) + (("kernel",) if have_emu else ()):
                leaves = [t.clone().requires_grad_(True) for t in inputs]
                if path == "kernel":
                    hooks.KERNELS_ON_CPU = True
                    codec.impl = "cuda"
                    codec.kernel = _EmuKernel(kc)
                try:
                    out, final = VQStateWKVLearned.apply(*leaves, skip, init, learned.wkv, codec)
                finally:
                    hooks.KERNELS_ON_CPU = False
                    codec.impl, codec.kernel = "torch", kc
                learned.wkv.grad = None
                (out * G).sum().backward()
                got = [t.grad for t in leaves] + [learned.wkv.grad.clone()]
                ref_leaves = [t.clone().requires_grad_(True) for t in inputs]
                ref_books = learned.wkv.detach().clone().requires_grad_(True)
                ref_out = reference(ref_leaves, skip, init, codec, ref_books)
                (ref_out * G).sum().backward()
                want = [t.grad for t in ref_leaves] + [ref_books.grad]
                diffs = [rel(x, y) for x, y in zip(got, want)]
                tol = 1e-5 if path == "torch" else 1e-3
                g = {"out_rel_diff": rel(out.detach(), ref_out.detach()), "input_grad_rel_diff": max(diffs[:6]),
                     "catalog_grad_rel_diff": diffs[6], "catalog_grad_rows": int((want[6].abs().sum(-1) > 0).sum()),
                     "unused_stages_zero": bool((got[6][rec.stages[0]:] == 0).all())}
                g["ok"] = (g["out_rel_diff"] < tol and max(diffs) < tol and g["catalog_grad_rows"] > 0
                           and g["unused_stages_zero"])
                entry[f"G_{path}_from_{start}"] = g
        # ---- S1: learned shift
        hooks.CODECS.clear()
        hooks.CODECS[0] = codec
        x = torch.randn(3, 9, 64, generator=gen).requires_grad_(True)
        stored = torch.rand(3, 9, generator=gen) < 0.6
        for p in learned.shift.values():
            p.grad = None
        y = hooks.vq_shift_mask(x, stored, 0, 0)
        Gs = torch.randn(3, 9, 64, generator=gen)
        (y * Gs).sum().backward()
        frozen = codec.shift(x.detach().reshape(-1, 64), "t").reshape(3, 9, 64)
        want = torch.zeros_like(learned.shift["t"])
        with torch.no_grad():
            flat, gflat = x.detach().reshape(-1, 64)[stored.flatten()], Gs.reshape(-1, 64)[stored.flatten()]
            n = flat.norm(dim=-1)
            unit, scale = flat / n[:, None], codec.shift_nq["t"](n)
            sub = 64 // rec.shift_m
            for p in range(rec.shift_m):
                _, pk = state_vq.rvq_picks(unit[:, p * sub:(p + 1) * sub].contiguous(), codec.shift_books["t"][p])
                for st in range(rec.shift_stages):
                    want[p, st].index_add_(0, pk[:, st], gflat[:, p * sub:(p + 1) * sub] * scale[:, None])
        s1 = {"value_bitwise_frozen": bool(torch.equal(y.detach()[stored], frozen[stored])
                                           and torch.equal(y.detach()[~stored], x.detach()[~stored])),
              "x_grad_identity": bool(torch.equal(x.grad, Gs)),
              "catalog_grad_rel_diff": rel(learned.shift["t"].grad, want),
              "other_role_untouched": learned.shift["c"].grad is None}
        s1["ok"] = s1["value_bitwise_frozen"] and s1["x_grad_identity"] and s1["catalog_grad_rel_diff"] < 1e-5 \
            and s1["other_role_untouched"]
        entry["S1_shift"] = s1
        hooks.CODECS.clear()
        report["checks"][name] = entry
        for key, value in entry.items():
            if not value["ok"]:
                fails.append(f"{name}:{key}")
    # ---- R1: rank-1 carry storage (CarryTable for VQ-coded streams)
    from scripts.state_carry_train import rank1_pack, rank1_unpack
    codec = state_vq.StateCodec(recs[0], books, 1 << 20, 8192, torch.float32)
    S = codec.encode_wkv(torch.randn(64, 4, 16, 16, generator=gen) * 0.3)[0]
    S[:5] = 0
    back = rank1_unpack(rank1_pack(S))
    r1 = {"max_rel_err": rel(back, S), "zeros_stay_zero": bool((back[:5] == 0).all()),
          "bytes_per_head": 2 * 16 * 4}
    r1["ok"] = r1["max_rel_err"] < 1e-6 and r1["zeros_stay_zero"]
    report["checks"]["R1_rank1_carry_storage"] = r1
    if not r1["ok"]:
        fails.append("R1")
    # ---- E1: export round trip
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "books.npz"
        sha1 = learned.export(path)
        loaded = state_vq.Codebooks.load(path)
        codecs = [state_vq.StateCodec(rec, loaded, 1 << 20, 8192) for rec in recs]
        same = all(torch.equal(c.wkv_books[0][s], learned.wkv[s].detach()) for c, rec in zip(codecs, recs)
                   for s in range(rec.stages[0]))
        same = same and all(torch.equal(c.shift_books[role][p][st], learned.shift[role][p, st].detach())
                            for c, rec in zip(codecs, recs) for role in ("t", "c")
                            for p in range(rec.shift_m) for st in range(rec.shift_stages))
        norms = loaded.wkv_norm == books.wkv_norm and loaded.shift_norm == books.shift_norm
        with torch.no_grad():
            learned.wkv[0, 0, 0] += 0.5
        sha2 = learned.export(path)
    e1 = {"values_round_trip": same, "norm_ranges_kept": norms, "changed_catalog_changes_file": sha1 != sha2}
    e1["ok"] = all(e1.values())
    report["checks"]["E1_export"] = e1
    if not e1["ok"]:
        fails.append("E1")
    report["emulated_kernels"] = have_emu
    if not have_emu:
        fails.append("no MSVC emulation: K1/K2/G2 not run")


class _EmuKernel:
    """KernelCodec interface (forward with s0/record, backward with dstored) on the CPU emulation."""

    def __init__(self, kc):
        self.kc = kc

    def refresh(self, stages):
        self.kc.refresh(stages)

    def forward(self, r, k, v, w, a, kd, skip, s0=None, record=False):
        from scripts import state_vq_cuda_emu as emu
        return emu.forward(self.kc, r, k, v, w, a, kd, skip, s0=s0, record=record)

    def backward(self, r, k, v, w, a, kd, skip, before, grad, dstored=False):
        from scripts import state_vq_cuda_emu as emu
        if dstored:
            return emu.backward_dstored(r, k, v, w, a, kd, skip, before, grad)
        return emu.backward(r, k, v, w, a, kd, skip, before, grad)


def trainer_check(report, fails, users):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from expertium.variants import build
    from scripts import state_vq
    from scripts.expertium_options import CatchUp
    from scripts.gpu_r01_prefetch import training_batches
    protocol = json.loads((RUN / "run" / "epoch-10" / "protocol.json").read_text())
    run_options = dict(json.loads((RUN / "experiment.json").read_text())["options"])
    plain = {k: v for k, v in run_options.items() if k not in ("packed_ipc", "distillation")}
    plain.update(chunk_reviews=2048, batch_chunks=2)
    rows = [next(r for r in protocol["rows"] if r["user_id"] == u) for u in users]
    span = plain["batch_chunks"] * plain["chunk_reviews"]
    out = {"users": users, "checkpoint": str(CKPT), "chunk": "2,048 x 2", "runs": {}}
    first_losses = {}
    for learn in (True, False):
        model = build("w64").float()
        model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True), strict=True)
        spec = {"recipe": RECIPE, **({"learn": True} if learn else {})}
        options = dict(plain, state_carry={"lanes": "strided"}, state_vq=spec)
        info = state_vq.install_training(model, spec, "cpu")
        lb = state_vq.LEARNED
        opt = torch.optim.AdamW(list(lb.parameters()), lr=1e-3, weight_decay=0.0) if lb is not None else None
        catch_up = CatchUp(options, {}, rows)
        run = {"codebook_handling": info["codebook_handling"], "updates": []}
        with ThreadPoolExecutor(max_workers=1) as pool:
            batches = [(row, item, item.batch.to(torch.device("cpu")))
                       for row, item in training_batches(rows, span, options["training_seed"], pool, options, None, 2)]
        limit = 3 if learn else 1
        for n, (row, item, batch) in enumerate(batches[:limit]):
            model.train()
            model.zero_grad(set_to_none=True)
            if opt is not None:
                opt.zero_grad(set_to_none=True)
            torch.manual_seed(item.seed)
            catch_up.carry.attach(item, batch, row["user_id"])
            t0 = time.perf_counter()
            stats = catch_up.loss(model, batch, 0, (row["user_id"], item.offset, item.reviews, item.seed))
            stats.average_loss.backward()
            u = {"user": row["user_id"], "offset": item.offset, "loss": float(stats.average_loss),
                 "seconds": round(time.perf_counter() - t0, 1)}
            if n == 0:
                first_losses[learn] = float(stats.average_loss)
            if lb is not None:
                grads = {"wkv": lb.wkv.grad, "shift_t": lb.shift["t"].grad, "shift_c": lb.shift["c"].grad}
                u["catalog_grad_norm"] = {k: (None if g is None else float(g.norm())) for k, g in grads.items()}
                u["catalog_grad_rows"] = {k: (0 if g is None else int((g.abs().sum(-1) > 0).sum()))
                                          for k, g in grads.items()}
                before = [p.detach().clone() for p in lb.parameters()]
                opt.step()
                u["catalog_moved"] = any(not torch.equal(b, p.detach()) for b, p in zip(before, lb.parameters()))
            table = catch_up.carry.table
            u["fp32_streams"] = sorted(table.fp32_streams)
            ranks = []
            for (stream, layer), slot in table.slots.items():
                if stream in table.fp32_streams:
                    u.setdefault("slot_dtypes", {})[f"{stream}/{layer}"] = str(slot["S"].dtype)
                    u.setdefault("slot_shapes", {})[f"{stream}/{layer}"] = list(slot["S"].shape[1:])
                    u.setdefault("slot_rank1", {})[f"{stream}/{layer}"] = bool(slot.get("rank1"))
                    ranks.append(float((slot["S"][slot["has"]].abs().sum(-1) > 0).float().mean()) if slot["has"].any() else 1.0)
            u["filled_factor_share"] = min(ranks) if ranks else None
            run["updates"].append(u)
        out["runs"]["learned" if learn else "frozen"] = run
        state_vq.uninstall_training(model)
    lr_ = out["runs"]["learned"]["updates"]
    ok = all(u["loss"] == u["loss"] and abs(u["loss"]) < 1e3 for u in lr_)
    ok = ok and all(all(v and v > 0 and v == v for v in u["catalog_grad_norm"].values()) and u["catalog_moved"] for u in lr_)
    ok = ok and all(u["fp32_streams"] == [0, 2] and all(d == "torch.float32" for d in u.get("slot_dtypes", {}).values())
                    for u in lr_)
    ok = ok and all(all(u["slot_rank1"].values()) and all(sh[-2:] == [2, 16] for sh in u["slot_shapes"].values())
                    for u in lr_)
    out["first_update_loss_learned_minus_frozen"] = first_losses[True] - first_losses[False]
    ok = ok and abs(out["first_update_loss_learned_minus_frozen"]) < 1e-6
    out["ok"] = ok
    report["checks"]["T1_trainer"] = out
    if not ok:
        fails.append("T1")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--users", default="1340")
    parser.add_argument("--skip-trainer", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/small-model/check-state-vq-learn.json"))
    args = parser.parse_args()
    os.environ["RWKV_NO_JIT"] = "1"
    os.environ["GPU_BATCH1_DATA_PATH"] = DATA
    sys.path.insert(0, str(ROOT))
    import torch
    torch.set_num_threads(4)
    started = time.perf_counter()
    report = {"recipe": RECIPE, "checks": {}}
    fails = []
    unit_checks(report, fails)
    if not args.skip_trainer:
        trainer_check(report, fails, [int(u) for u in args.users.split(",")])
    report["fails"] = fails
    report["passed"] = not fails and not args.skip_trainer
    report["seconds"] = round(time.perf_counter() - started, 1)
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps(report, indent=1, default=str)[:6000])
    print("ALL OK" if not fails else f"FAILED: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
