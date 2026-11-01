#!/usr/bin/env python
"""CPU proof of the opt-in grade-order training option (grade-order sanity (SM33)): track option
"grade_order": {"weight": W, "density": D, "horizons_days": [...], "seed": S}.

On one real user's first reviews (128-review samples, eager CPU model with the leak-free small model's (SM72s) epoch-21
weights, the cool-down's start):
  1. option absent: samples bit for bit equal to scripts/gpu_r01_samples.py before SM33; loss, loss tensor and every
     gradient bit for bit equal to expertium/model/variant_model.py before SM33; weight 0.0 = absent, bitwise;
     fix_leading_shift changes nothing on the batch without probes;
  2. probe rows (density D): every non-probe row's input and output unchanged (|delta| <= 1e-5: the CPU reference
     recurrence is per step, but the batch shapes differ), the probe with the review's real rating = the real row,
     the loss at weight 0 unchanged (<= 1e-6 relative); probes leading a card's sequence are covered;
  3. weight W: loss = loss(weight 0) + W x penalty, the penalty = an independent float64 NumPy recompute of
     mean relu(logit R_g(t) - logit R_{g+1}(t)) over adjacent probe pairs and horizons; gradients differ;
  4. synthetic curves: penalty exactly 0 with zero gradients when every pair is ordered; > 0 when one is not, with
     the gradient lowering the lower grade's curve and raising the higher grade's;
  5. TorchScript (no RWKV_NO_JIT, subprocess): the scripted model compiles and gives the same loss at weight W.

  .venv/Scripts/python.exe scripts/check_grade_order.py [--output artifacts/small-model/sm33/check-cpu-grade-order.json]
"""
from argparse import ArgumentParser
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("GPU_BATCH1_DATA_PATH", r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"))
os.environ["GPU_BATCH1_DATA_PATH"] = str(DATA)
START = Path(r"C:\Users\jonat\workspace\wt\sm72s\artifacts\accuracy\experiments\scale-w64kd-final-sc-lanes-leakfree\run"
             r"\epoch-21\expertium\E89-F65e-F77a\master.pth")
HORIZONS = [1, 3, 7, 15, 30, 60, 120, 240, 480, 960, 1920, 3650]


def previous_module(path, needle, tmp, name):
    """`path` as of the parent of the first commit whose diff adds `needle` (the code before SM33)."""
    first = subprocess.run(["git", "log", "--reverse", "--format=%H", "-S", needle, "HEAD", "--", path], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split()
    rev = f"{first[0]}^" if first else "HEAD"   # uncommitted: HEAD is the code before
    base = subprocess.run(["git", "rev-parse", rev], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    source = subprocess.run(["git", "show", f"{base}:{path}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert needle not in source
    file = Path(tmp) / f"{name}.py"
    file.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, base


def sections(frame, n, chunk=128):
    return [frame.iloc[i:i + chunk] for i in range(0, n, chunk)]


def jit_child(args):
    """Subprocess: the scripted model's loss at weight W on the saved probe batch."""
    import torch
    from expertium import variants
    from expertium.model.variant_model import VariantSrsRWKV
    torch.set_num_threads(2)
    batch = torch.load(args.jit_batch, weights_only=False)
    model = VariantSrsRWKV(variants.variant("w64"))
    model.load_state_dict(torch.load(START, map_location="cpu", weights_only=True))
    model.eval()
    scripted = isinstance(model, torch.jit.ScriptModule)
    model.grade_order_weight = args.weight
    model.grade_order_horizons = [float(d) * 86400.0 for d in HORIZONS]
    with torch.no_grad():
        stats = model.get_loss(batch)
    print(json.dumps({"scripted": scripted, "loss": stats.average_loss.item()}))
    return 0


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/small-model/sm33/check-cpu-grade-order.json")
    parser.add_argument("--user", type=int, default=7)
    parser.add_argument("--reviews", type=int, default=768)
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--weight", type=float, default=0.1)
    parser.add_argument("--jit-batch", default="")
    args = parser.parse_args()
    if args.jit_batch:
        return jit_child(args)
    os.environ["RWKV_NO_JIT"] = "1"
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare
    from expertium import variants
    from scripts import gpu_r01_samples as gs
    started = time.perf_counter()
    results = {}

    def record(name, value, ok):
        results[name] = {"value": value, "ok": bool(ok)}
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {value}", flush=True)

    spec = {"weight": args.weight, "density": args.density, "horizons_days": HORIZONS, "seed": 33}
    frame = get_rwkv_data(DATA, args.user)
    parts = sections(frame, args.reviews)
    seed = 2025 + args.user

    def samples(module, probes):
        kw = {} if probes is False else {"grade_probes": probes}
        return [module.create_sample_fast(args.user, part, [], torch.float32, torch.device("cpu"), **kw) for part in parts]

    def same_samples(a, b):
        for x, y in zip(a, b):
            for f in ("card_features", "global_labels", "review_ths", "label_review_ths", "day_offsets", "day_offsets_first", "skips"):
                if not torch.equal(getattr(x, f), getattr(y, f)):
                    return False
            if any(not torch.equal(x.ids[k], y.ids[k]) for k in x.ids):
                return False
            for k in x.modules:
                mx, my = x.modules[k], y.modules[k]
                if not (np.array_equal(mx.split_len, my.split_len) and np.array_equal(mx.split_B, my.split_B)
                        and torch.equal(mx.from_perm, my.from_perm) and torch.equal(mx.to_perm, my.to_perm)):
                    return False
        return len(a) == len(b)

    def model_of(cls, weight=None):
        m = cls(variants.variant("w64"))
        m.load_state_dict(torch.load(START, map_location="cpu", weights_only=True))
        m.train()
        if weight is not None:
            m.grade_order_weight = weight
            m.grade_order_horizons = [float(d) * 86400.0 for d in HORIZONS]
        return m

    def loss_grads(model, batch):
        torch.manual_seed(11)
        model.zero_grad(set_to_none=True)
        stats = model.get_loss(batch)
        stats.average_loss.backward()
        return stats, [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]

    def same(a, b):
        return (torch.equal(a[0].average_loss, b[0].average_loss) and torch.equal(a[0].loss_tensor, b[0].loss_tensor)
                and len(a[1]) == len(b[1]) and all(torch.equal(x, y) for x, y in zip(a[1], b[1])))

    from expertium.model import variant_model as vm
    with tempfile.TemporaryDirectory() as tmp:
        old_samples, base_s = previous_module("scripts/gpu_r01_samples.py", "grade_probes", tmp, "samples_before_sm33")
        old_model, base_m = previous_module("expertium/model/variant_model.py", "grade_order_weight", tmp, "variant_before_sm33")
        # 1. option absent
        plain = samples(gs, None)
        record(f"samples, option absent == gpu_r01_samples before SM33 ({base_s[:8]}), bitwise",
               len(plain), same_samples(samples(old_samples, False), plain))
        record("samples, density 0 == option absent, bitwise", len(plain),
               same_samples(plain, samples(gs, {**spec, "density": 0.0})))
        batch_plain = prepare(plain, seed=seed)
        record("fix_leading_shift on the batch without probes changes nothing", gs.fix_leading_shift(batch_plain),
               gs.fix_leading_shift(batch_plain) == 0)
        torch.manual_seed(2025)
        old = loss_grads(model_of(old_model.VariantSrsRWKV), batch_plain)
        absent = loss_grads(model_of(vm.VariantSrsRWKV), batch_plain)
        zero = loss_grads(model_of(vm.VariantSrsRWKV, 0.0), batch_plain)
        record(f"option absent == variant_model before SM33 ({base_m[:8]}): loss + loss tensor + gradients bitwise",
               old[0].average_loss.item(), same(old, absent))
        record("weight 0.0 == option absent bitwise", zero[0].average_loss.item(), same(absent, zero))
        record("weight 0.1 on the batch without probes == option absent bitwise (no probe pairs: penalty 0)",
               None, same(absent, loss_grads(model_of(vm.VariantSrsRWKV, args.weight), batch_plain)))
        # 2. probe rows
        probed = samples(gs, spec)
        batch_probe = prepare(probed, seed=seed)
        changed = gs.fix_leading_shift(batch_probe)
        labels = torch.cat([s.global_labels for s in probed])     # one row per sample row, in sample order
        marker = labels[:, 1].numpy()
        grade = np.where(marker < 0, -marker, 0).astype(int)
        n_probe = int((grade == 1).sum())
        firsts = int(sum(((s.global_labels[:, 1] == -1) & (torch.arange(len(s.skips)) == 0)).sum() for s in probed))
        record("probe rows inserted (reviews probed, shift indices fixed, samples starting with a probe)",
               {"probed_reviews": n_probe, "rows_plain": int(sum(len(s.skips) for s in plain)),
                "rows_probed": int(sum(len(s.skips) for s in probed)), "shift_fixed": changed, "leading": firsts},
               n_probe > 0 and changed > 0)
        model = model_of(vm.VariantSrsRWKV)
        model.eval()
        with torch.no_grad():
            a = model.forward_batch_variant(batch_plain.start, batch_plain.sub_gather, batch_plain.sub_gather_lens,
                                            batch_plain.time_shift_selects, batch_plain.skips, batch_plain.num_data)
            b = model.forward_batch_variant(batch_probe.start, batch_probe.sub_gather, batch_probe.sub_gather_lens,
                                            batch_probe.time_shift_selects, batch_probe.skips, batch_probe.num_data)
        names = ("ahead", "w", "w_log_p", "p_logits", "s_raw", "d_raw")
        worst_in, worst_out, worst_real = 0.0, {}, {}
        for i, (sp, sq) in enumerate(zip(plain, probed)):
            g = sq.global_labels[:, 1]
            keep = torch.nonzero(g >= 0).squeeze(1)
            n = len(sp.skips)
            assert len(keep) == n
            worst_in = max(worst_in, (sp.card_features - sq.card_features[keep]).abs().max().item())
            for name, x, y in zip(names, a, b):
                if x.dim() < 3:
                    continue
                worst_out[name] = max(worst_out.get(name, 0.0), (x[i, :n] - y[i, keep]).abs().max().item())
            # the probe with the real rating = the real row (the real row follows its 4 probes)
            ratings = sq.card_features[:, 9:13].argmax(dim=1) + 1
            real_after = {}
            pos = torch.nonzero(g < 0).squeeze(1).tolist()
            for p in pos:
                gg = int(-g[p])
                r = p + (5 - gg)               # probes g .. 4, then the real row
                if int(ratings[r]) == gg and g[r] >= 0:
                    for name, y in zip(names, b):
                        if y.dim() >= 3:
                            worst_real[name] = max(worst_real.get(name, 0.0), (y[i, p] - y[i, r]).abs().max().item())
        record("non-probe rows: inputs bitwise equal", worst_in, worst_in == 0.0)
        record("non-probe rows: outputs unchanged (max |delta|)", worst_out, max(worst_out.values()) <= 1e-5)
        record("probe with the real rating == real row (max |delta|)", worst_real, max(worst_real.values()) <= 1e-5)
        # eval mode (no dropout): train-mode dropout masks are drawn per row, so extra rows change the real rows' masks
        def eval_model():
            m = model_of(vm.VariantSrsRWKV, 0.0)
            m.eval()
            return m
        e_plain, e_probe = loss_grads(eval_model(), batch_plain), loss_grads(eval_model(), batch_probe)
        rel = abs(e_probe[0].average_loss.item() - e_plain[0].average_loss.item()) / abs(e_plain[0].average_loss.item())
        gmax = max((x - y).abs().max().item() for x, y in zip(e_plain[1], e_probe[1]))
        gscale = max(x.abs().max().item() for x in e_plain[1])
        record("weight 0 with probe rows (eval mode, no dropout): loss unchanged (relative), gradients (max |delta| / max |g|)",
               {"loss_rel": rel, "grad_max_abs": gmax, "grad_max": gscale}, rel <= 1e-6 and gmax <= 1e-5 * max(1.0, gscale))
        z_probe = loss_grads(model_of(vm.VariantSrsRWKV, 0.0), batch_probe)
        # 3. weight W
        w = loss_grads(model_of(vm.VariantSrsRWKV, args.weight), batch_probe)
        penalty = vm.GRADE_ORDER_LAST[0].item()
        pairs = int(vm.GRADE_ORDER_LAST[1])
        gap = abs(w[0].average_loss.item() - (z_probe[0].average_loss.item() + args.weight * penalty))
        # independent float64 recompute from the eval-mode outputs of the same weights (train-mode dropout is off in
        # these weights' config? use the train pass's own outputs instead: recompute from a no-grad train forward)
        m = model_of(vm.VariantSrsRWKV)
        torch.manual_seed(11)
        with torch.no_grad():
            out = m.forward_batch_variant(batch_probe.start, batch_probe.sub_gather, batch_probe.sub_gather_lens,
                                          batch_probe.time_shift_selects, batch_probe.skips, batch_probe.num_data)
        ow, osr, odr = (x.double().numpy() for x in (out[1], out[4], out[5]))
        gl = batch_probe.labels[..., 1].numpy()
        gr = np.where(gl < 0, -gl, 0)
        nx = np.pad(gr[:, 1:], ((0, 0), (0, 1)))
        bi, ti = np.nonzero((gr >= 1) & (gr <= 3) & (nx == gr + 1))
        t = np.array(HORIZONS, dtype=np.float64) * 86400.0

        def logit_r(bb, tt):
            s = np.exp(np.clip(osr[bb, tt], -25, 25))[:, None, :]
            d = np.exp(np.clip(odr[bb, tt], -25, 25))[:, None, :]
            r = np.sum(ow[bb, tt][:, None, :] * np.exp(-d * np.log1p(np.maximum(1.0, t)[None, :, None] / (1e-7 + s))), -1)
            r = 1e-5 + (1 - 2e-5) * r
            return np.log(r / (1 - r))
        ref = float(np.mean(np.maximum(0.0, logit_r(bi, ti) - logit_r(bi, ti + 1))))
        record("weight W: loss = loss(0) + W x penalty", {"penalty": penalty, "pairs": pairs, "abs_gap": gap},
               gap <= 1e-6 * max(1.0, abs(w[0].average_loss.item())))
        record("penalty == independent float64 NumPy recompute", {"torch": penalty, "numpy": ref, "pairs_numpy": len(bi)},
               len(bi) == pairs and abs(penalty - ref) <= 1e-5 * max(1.0, ref) and penalty > 0)
        differs = sum(not torch.equal(x, y) for x, y in zip(z_probe[1], w[1]))
        record("weight W: gradients differ", f"{differs}/{len(w[1])} tensors", differs > 0)
        # 4. synthetic curves
        mm = model_of(vm.VariantSrsRWKV)
        T = 6
        lab = torch.zeros(1, T, 7)
        lab[0, :4, 1] = torch.tensor([-1.0, -2.0, -3.0, -4.0])
        lab[0, 4:, 1] = 3.14159
        wts = torch.full((1, T, 3), 1.0 / 3)
        base_s = torch.tensor([10.0, 12.0, 14.0])
        s_ok = torch.stack([base_s + k for k in (0.0, 1.0, 2.0, 3.0, 0.5, 0.5)]).unsqueeze(0).requires_grad_(True)
        d_raw = torch.zeros(1, T, 3, requires_grad=True)
        p_ok = vm.grade_order_penalty(mm, wts, s_ok, d_raw, lab, [float(h) * 86400 for h in HORIZONS])
        p_ok.backward()
        record("synthetic ordered curves: penalty exactly 0, zero gradient",
               {"penalty": p_ok.item(), "grad_abs_sum": s_ok.grad.abs().sum().item()},
               p_ok.item() == 0.0 and s_ok.grad.abs().sum().item() == 0.0)
        s_bad = torch.stack([base_s + k for k in (0.0, 3.0, 1.0, 4.0, 0.5, 0.5)]).unsqueeze(0).requires_grad_(True)  # Hard > Good
        p_bad = vm.grade_order_penalty(mm, wts, s_bad, d_raw, lab, [float(h) * 86400 for h in HORIZONS])
        p_bad.backward()
        gh, gg = s_bad.grad[0, 1].sum().item(), s_bad.grad[0, 2].sum().item()
        record("synthetic Hard > Good: penalty > 0; gradient lowers Hard's stability, raises Good's, others 0",
               {"penalty": p_bad.item(), "dP/dS_hard": gh, "dP/dS_good": gg,
                "others": s_bad.grad[0, [0, 3, 4, 5]].abs().sum().item()},
               p_bad.item() > 0 and gh > 0 and gg < 0 and s_bad.grad[0, [0, 3, 4, 5]].abs().sum().item() == 0.0)
        # 5. TorchScript
        path = Path(tmp) / "batch.pt"
        torch.save(batch_probe, path)
        env = {k: v for k, v in os.environ.items() if k != "RWKV_NO_JIT"}
        proc = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--jit-batch", str(path),
                               "--weight", str(args.weight)], capture_output=True, text=True, env=env, cwd=ROOT)
        try:
            child = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            child = {"error": (proc.stderr or proc.stdout)[-800:]}
        m_eval = model_of(vm.VariantSrsRWKV, args.weight)
        m_eval.eval()
        with torch.no_grad():
            eager = m_eval.get_loss(batch_probe).average_loss.item()
        record("TorchScript model compiles; loss at weight W == eager (eval mode)", {**child, "eager": eager},
               child.get("scripted") is True and abs(child.get("loss", 1e9) - eager) <= 1e-5 * max(1.0, abs(eager)))
    ok = all(r["ok"] for r in results.values())
    report = {"ok": ok, "spec": spec, "user": args.user, "reviews": args.reviews, "start_weights": str(START),
              "seconds": round(time.perf_counter() - started, 1), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"[sm33-penalty-check] {'PASS' if ok else 'FAIL'} ({len(results)} checks) -> {args.output}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
