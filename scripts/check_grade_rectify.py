#!/usr/bin/env python
"""CPU proof of the grade order by construction (SM95): track option "grade_rectify": {"density": D, "seed": S} and the
scorer flag scripts/small_model_heldout.py --grade-rectify.

  1. option absent: training samples bit for bit equal to scripts/gpu_r01_samples.py before SM95; the scorer's
     leak-free full-history batch bit for bit equal to scripts/small_model_heldout.py before SM95; loss, loss tensor
     and every gradient bit for bit equal to expertium/model/variant_model.py before SM95; grade_rectify on a batch
     without probes = absent, bitwise; fix_leading_shift(probes=3) is a no-op without probes;
  2. probe rows (training layout at density D, scorer layout at density 1): every non-probe row's input bitwise and
     output within 1e-5 (eval mode; the CPU reference recurrence runs per step but the batch shapes differ), three
     probes per probed review whose ratings plus the real row's are {1, 2, 3, 4}, the slot map puts each grade in its
     slot, probes leading a card's sequence are covered; the instant head is unchanged;
  3. the rectified curve: each probed real row's scored probability (stats.p_curve) == an independent float64
     recompute (forward outputs -> the four curves at the row's label time -> literal stack PAVA -> the row's grade),
     unprobed rows and already ordered reviews bit for bit the raw curve; train-mode loss finite, gradients finite and
     different from the unrectified loss, gradient reaching the probe rows;
  4. order exact: zero breaks of R_Again(t) <= R_Hard(t) <= R_Good(t) <= R_Easy(t) on every probed review at 241
     horizons (1 s .. 100 years) and at the label time, float32 (torch) and float64 (NumPy), for the trained weights
     and three random initialisations; 2,000,000 adversarial random 4-vectors (ties, permutations, extremes) in float32
     and float64; the operator == the stack PAVA (float64, <= 1e-12) and the identity on ordered inputs, bitwise;
     intervals of the rectified curves ordered at 12 target retentions 0.50 .. 0.99 and the fork's display identical
     with and without its PAVA (the fork's PAVA is a no-op); the raw curves of the same reviews do break (the check is
     not vacuous);
  5. TorchScript (no RWKV_NO_JIT, subprocess): the scripted model compiles with grade_rectify and gives the eager loss;
  6. preview (not a gate): curve LogLoss of the scorer layout, raw vs rectified, on the check user.

  .venv/Scripts/python.exe scripts/check_grade_rectify.py [--output artifacts/small-model/sm95/check-cpu-grade-rectify.json]
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


def previous_module(path, needle, tmp, name):
    """`path` as of the parent of the first commit whose diff adds `needle` (the code before SM95)."""
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


def jit_child(args):
    import torch
    from expertium import variants
    from expertium.model.variant_model import VariantSrsRWKV
    torch.set_num_threads(2)
    batch = torch.load(args.jit_batch, weights_only=False)
    model = VariantSrsRWKV(variants.variant("w64"))
    model.load_state_dict(torch.load(START, map_location="cpu", weights_only=True))
    model.eval()
    model.grade_rectify = True
    with torch.no_grad():
        stats = model.get_loss(batch)
    print(json.dumps({"scripted": isinstance(model, torch.jit.ScriptModule), "loss": stats.average_loss.item()}))
    return 0


def scorer_child(args):
    """Subprocess: build the scorer's leak-free full-history batch of one user with the given small_model_heldout.py
    (imported first: it installs its state-quant hooks before rwkv is imported) and save it."""
    import torch
    spec = importlib.util.spec_from_file_location("heldout_under_test", args.scorer_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from scripts import compress_common as cc
    run = cc.run_info()
    _, batch, error, _ = module.LeakBatches([args.user], cc.data_path(run), run["review_counts"], "neighbour", 0)[0]
    torch.save({"batch": batch, "error": error}, args.scorer_out)
    return 0


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/small-model/sm95/check-cpu-grade-rectify.json")
    parser.add_argument("--user", type=int, default=7)
    parser.add_argument("--reviews", type=int, default=768)
    parser.add_argument("--scorer-reviews", type=int, default=2000)
    parser.add_argument("--density", type=float, default=0.3)
    parser.add_argument("--jit-batch", default="")
    parser.add_argument("--scorer-module", default="")
    parser.add_argument("--scorer-out", default="")
    parser.add_argument("--scorer-user", type=int, default=4)
    args = parser.parse_args()
    if args.jit_batch:
        return jit_child(args)
    if args.scorer_module:
        args.user = args.scorer_user
        return scorer_child(args)
    os.environ["RWKV_NO_JIT"] = "1"
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare
    from expertium import variants
    from expertium.model import grade_rectify as gr
    from expertium.model import variant_model as vm
    from scripts import gpu_r01_samples as gs
    from scripts import sm95_rectify as sr
    from scripts import sm71_leak
    from scripts import compress_common as cc
    started = time.perf_counter()
    results = {}

    def record(name, value, ok):
        results[name] = {"value": value, "ok": bool(ok)}
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {value}", flush=True)

    spec = {"density": args.density, "seed": 95}
    frame = get_rwkv_data(DATA, args.user)
    parts = [frame.iloc[i:i + 128] for i in range(0, args.reviews, 128)]
    seed = 2025 + args.user

    def samples(module, probes):
        kw = {} if probes is False else {"rectify_probes": probes}
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

    def same_batch(a, b):
        fields = ("start", "labels", "label_review_th")
        if any(not torch.equal(getattr(a, f), getattr(b, f)) for f in fields):
            return False
        for name in ("sub_gather", "time_shift_selects", "skips"):
            for xa, xb in zip(getattr(a, name), getattr(b, name)):
                if len(xa) != len(xb) or any(not torch.equal(u, v) for u, v in zip(xa, xb)):
                    return False
        return a.sub_gather_lens == b.sub_gather_lens and a.num_data == b.num_data

    def model_of(cls, rectify=None, state=None):
        m = cls(variants.variant("w64"))
        m.load_state_dict(state if state is not None else torch.load(START, map_location="cpu", weights_only=True))
        m.train()
        if rectify is not None:
            m.grade_rectify = rectify
        return m

    def loss_grads(model, batch):
        torch.manual_seed(11)
        model.zero_grad(set_to_none=True)
        stats = model.get_loss(batch)
        stats.average_loss.backward()
        return stats, [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]

    def same(a, b):
        return (torch.equal(a[0].average_loss, b[0].average_loss) and torch.equal(a[0].loss_tensor, b[0].loss_tensor)
                and torch.equal(a[0].p_curve, b[0].p_curve)
                and len(a[1]) == len(b[1]) and all(torch.equal(x, y) for x, y in zip(a[1], b[1])))

    def forward(model, batch):
        with torch.no_grad():
            return model.forward_batch_variant(batch.start, batch.sub_gather, batch.sub_gather_lens,
                                               batch.time_shift_selects, batch.skips, batch.num_data)

    def slot_rows(labels):
        """[P] real rows and [P, 4] slot rows (flat) of a probed batch, from the labels alone (numpy recompute of the
        layout rule, independent of grade_rectify.rectify_slots)."""
        marker = labels[..., 1].reshape(-1).numpy()
        out_r, out_s = [], []
        for r in range(3, len(marker)):
            if marker[r] >= 0 and (marker[r - 3:r] < 0).all():
                grades = list((-marker[r - 3:r]).astype(int))
                g = ({1, 2, 3, 4} - set(grades)).pop()
                rows = dict(zip(grades, range(r - 3, r)))
                rows[g] = r
                out_r.append(r)
                out_s.append([rows[k] for k in (1, 2, 3, 4)])
        return np.array(out_r, dtype=np.int64), np.array(out_s, dtype=np.int64).reshape(-1, 4)

    def curves_np(out, rows, t_seconds):
        """float64 R of flat rows at t (seconds, broadcast [len(rows), H])."""
        w, s_raw, d_raw = (x.reshape(-1, x.size(-1)).double().numpy()[rows] for x in (out[1], out[4], out[5]))
        s = np.exp(np.clip(s_raw, -25, 25))[:, None, :]
        d = np.exp(np.clip(d_raw, -25, 25))[:, None, :]
        t = np.maximum(1.0, t_seconds)[..., None]
        r = np.sum(w[:, None, :] * np.exp(-d * np.log1p(t / (1e-7 + s))), -1)
        return 1e-5 + (1 - 2e-5) * r

    with tempfile.TemporaryDirectory() as tmp:
        old_samples, base_s = previous_module("scripts/gpu_r01_samples.py", "rectify_probes", tmp, "samples_before_sm95")
        old_model, base_m = previous_module("expertium/model/variant_model.py", "grade_rectify", tmp, "variant_before_sm95")
        first = subprocess.run(["git", "log", "--reverse", "--format=%H", "-S", "grade_rectify", "HEAD", "--",
                                "scripts/small_model_heldout.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        base_h = subprocess.run(["git", "rev-parse", f"{first[0]}^" if first else "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        old_scorer_file = Path(tmp) / "heldout_before_sm95.py"
        old_scorer_file.write_text(subprocess.run(["git", "show", f"{base_h}:scripts/small_model_heldout.py"], cwd=ROOT,
                                                  capture_output=True, text=True, check=True).stdout, encoding="utf-8")
        assert "grade_rectify" not in old_scorer_file.read_text(encoding="utf-8")
        # ---------------------------------------------------------------- 1. option absent
        plain = samples(gs, None)
        record(f"training samples, option absent == gpu_r01_samples before SM95 ({base_s[:8]}), bitwise",
               len(plain), same_samples(samples(old_samples, False), plain))
        record("training samples, density 0 == option absent, bitwise", len(plain),
               same_samples(plain, samples(gs, {**spec, "density": 0.0})))
        batch_plain = prepare(plain, seed=seed)
        record("fix_leading_shift(probes=3) on the batch without probes changes nothing",
               gs.fix_leading_shift(batch_plain, probes=3), gs.fix_leading_shift(batch_plain, probes=3) == 0)
        run = cc.run_info()
        counts = run["review_counts"]
        got = {}
        for tag, module_file in (("new", ROOT / "scripts/small_model_heldout.py"), ("old", old_scorer_file)):
            target = Path(tmp) / f"scorer-{tag}.pt"
            proc = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--scorer-module", str(module_file),
                                   "--scorer-out", str(target), "--scorer-user", "4"], capture_output=True, text=True, cwd=ROOT)
            got[tag] = torch.load(target, weights_only=False) if target.exists() else {"batch": None, "error": proc.stderr[-800:]}
        b_new, e_new, b_old, e_old = got["new"]["batch"], got["new"]["error"], got["old"]["batch"], got["old"]["error"]
        record(f"scorer batch (leak-free full history, user 4), flag absent == small_model_heldout before SM95 ({base_h[:8]}), bitwise",
               {"errors": [e_new, e_old], "rows": int(b_new.labels.shape[1]) if b_new is not None else None},
               e_new is None and e_old is None and same_batch(b_new, b_old))
        torch.manual_seed(2025)
        old = loss_grads(model_of(old_model.VariantSrsRWKV), batch_plain)
        absent = loss_grads(model_of(vm.VariantSrsRWKV), batch_plain)
        record(f"option absent == variant_model before SM95 ({base_m[:8]}): loss + loss tensor + p_curve + gradients bitwise",
               old[0].average_loss.item(), same(old, absent))
        record("grade_rectify False == absent, bitwise", None, same(absent, loss_grads(model_of(vm.VariantSrsRWKV, False), batch_plain)))
        record("grade_rectify True on a batch without probes == absent, bitwise", None,
               same(absent, loss_grads(model_of(vm.VariantSrsRWKV, True), batch_plain)))
        # ---------------------------------------------------------------- 2. probe rows, training layout
        probed = samples(gs, spec)
        batch_probe = prepare(probed, seed=seed)
        changed = gs.fix_leading_shift(batch_probe, probes=3)
        real_rows, slots = slot_rows(batch_probe.labels)
        feats = batch_probe.start.reshape(-1, batch_probe.start.size(-1)).numpy()
        slot_grade = feats[slots.reshape(-1), 9:13].argmax(axis=1).reshape(-1, 4) + 1
        leading = int(sum(int((s.global_labels[0, 1] < 0).item()) for s in probed))
        record("training layout: probes inserted (probed reviews, rows, shift indices fixed, samples starting with a probe)",
               {"probed_reviews": len(real_rows), "rows_plain": int(sum(len(s.skips) for s in plain)),
                "rows_probed": int(sum(len(s.skips) for s in probed)), "shift_fixed": changed, "leading": leading},
               len(real_rows) > 0 and changed > 0 and int(sum(len(s.skips) for s in probed)) - int(sum(len(s.skips) for s in plain)) == 3 * len(real_rows))
        record("training layout: slot g holds the row whose rating one-hot is g (all probed reviews)",
               {"reviews": len(slots)}, bool((slot_grade == np.arange(1, 5)[None, :]).all()))
        model = model_of(vm.VariantSrsRWKV)
        model.eval()
        a, b = forward(model, batch_plain), forward(model, batch_probe)
        names = ("ahead", "w", "w_log_p", "p_logits", "s_raw", "d_raw")
        worst_in, worst_out = 0.0, {}
        for i, (sp, sq) in enumerate(zip(plain, probed)):
            keep = torch.nonzero(sq.global_labels[:, 1] >= 0).squeeze(1)
            n = len(sp.skips)
            assert len(keep) == n
            worst_in = max(worst_in, (sp.card_features - sq.card_features[keep]).abs().max().item())
            for name, x, y in zip(names, a, b):
                if x.dim() >= 3:
                    worst_out[name] = max(worst_out.get(name, 0.0), (x[i, :n] - y[i, keep]).abs().max().item())
        record("training layout: non-probe rows' inputs bitwise equal", worst_in, worst_in == 0.0)
        record("training layout: non-probe rows' outputs unchanged (max |delta|)", worst_out, max(worst_out.values()) <= 1e-5)
        # ---------------------------------------------------------------- 3. the rectified curve (training layout)
        m_off, m_on = model_of(vm.VariantSrsRWKV, False), model_of(vm.VariantSrsRWKV, True)
        m_off.eval()
        m_on.eval()
        with torch.no_grad():
            s_off, s_on = m_off.get_loss(batch_probe), m_on.get_loss(batch_probe)
        t_lab = batch_probe.labels[..., 0].reshape(-1).double().numpy()
        r4 = curves_np(b, slots.reshape(-1), np.repeat(t_lab[real_rows], 4)[:, None])[:, 0].reshape(-1, 4)
        grade = slot_grade[np.arange(len(slots)), :]  # == 1..4
        real_grade = np.array([int(np.flatnonzero(slots[k] == real_rows[k])[0]) for k in range(len(slots))])
        rect64 = np.array([gr.pava4_scalar(v) for v in r4])[np.arange(len(slots)), real_grade]
        p_on = s_on.p_curve.reshape(-1).double().numpy()
        p_off = s_off.p_curve.reshape(-1).double().numpy()
        # p_curve = sigmoid(logit(R) + ahead residual); the residual of the GRU model is a constant (zero ahead logits)
        resid = float(np.median(np.log(p_off[real_rows] / (1 - p_off[real_rows]))
                                - np.log(r4[np.arange(len(slots)), real_grade] / (1 - r4[np.arange(len(slots)), real_grade]))))
        expect = 1 / (1 + np.exp(-(np.log(rect64 / (1 - rect64)) + resid)))
        broken = ~np.all(np.diff(r4, axis=1) >= 0, axis=1)
        other = np.setdiff1d(np.arange(len(p_on)), real_rows)
        ordered_rows = real_rows[~broken]
        record("rectified p_curve of probed rows == float64 recompute (stack PAVA of the four curves at the label time)",
               {"probed": len(real_rows), "broken_raw": int(broken.sum()), "max_abs": float(np.abs(p_on[real_rows] - expect).max()),
                "residual_logit": resid},
               broken.sum() > 0 and float(np.abs(p_on[real_rows] - expect).max()) <= 2e-6)
        record("unprobed rows and ordered probed reviews: p_curve bitwise the raw curve; instant head bitwise",
               {"ordered_probed": len(ordered_rows)},
               np.array_equal(p_on[other], p_off[other]) and np.array_equal(p_on[ordered_rows], p_off[ordered_rows])
               and torch.equal(s_on.p_imm, s_off.p_imm))
        tr_off = loss_grads(model_of(vm.VariantSrsRWKV, False), batch_probe)
        tr_on = loss_grads(model_of(vm.VariantSrsRWKV, True), batch_probe)
        finite = all(torch.isfinite(g).all() for g in tr_on[1]) and torch.isfinite(tr_on[0].average_loss)
        differs = sum(not torch.equal(x, y) for x, y in zip(tr_off[1], tr_on[1]))
        # gradient reaching the probe rows: d loss / d curve of a probe slot is non-zero when it is pooled with the real row
        mg = model_of(vm.VariantSrsRWKV, True)
        mg.eval()
        out = mg.forward_batch_variant(batch_probe.start, batch_probe.sub_gather, batch_probe.sub_gather_lens,
                                       batch_probe.time_shift_selects, batch_probe.skips, batch_probe.num_data)
        s_leaf = out[4].detach().clone().requires_grad_(True)
        cp = mg.gru_forgetting_curve(out[1].detach(), s_leaf, out[5].detach(), batch_probe.labels[..., 0].unsqueeze(-1))
        rc = gr.rectify_curve(cp, batch_probe.labels.float())
        rc.reshape(-1)[torch.as_tensor(real_rows)].log().sum().backward()
        probe_flat = np.setdiff1d(slots.reshape(-1), real_rows)
        gprobe = s_leaf.grad.reshape(-1, s_leaf.size(-1)).abs().sum(1).numpy()
        record("train mode: rectified loss finite, gradients finite and different; gradient reaches pooled probe rows only",
               {"loss": tr_on[0].average_loss.item(), "tensors_differ": f"{differs}/{len(tr_on[1])}",
                "probe_rows_with_grad": int((gprobe[probe_flat] > 0).sum()), "non_slot_rows_with_grad": int((gprobe[other[~np.isin(other, slots.reshape(-1))]] > 0).sum())},
               finite and differs > 0 and (gprobe[probe_flat] > 0).sum() > 0
               and (gprobe[other[~np.isin(other, slots.reshape(-1))]] > 0).sum() == 0)
        # ---------------------------------------------------------------- 2b/3b. scorer layout (density 1)
        sframe, _ = sm71_leak.neutralise(get_rwkv_data(DATA, args.user), "neighbour", DATA, args.user)
        sframe = sframe.iloc[:args.scorer_reviews]
        sr.uninstall_scorer_probes()
        sb_plain = prepare(cc.make_samples(sframe, args.user, len(sframe)), seed=seed)
        sr.install_scorer_probes()
        try:
            sb_probe = prepare(cc.make_samples(sframe, args.user, len(sframe)), seed=seed)
        finally:
            sr.uninstall_scorer_probes()
        s_changed = gs.fix_leading_shift(sb_probe, probes=3)
        s_real, s_slots = slot_rows(sb_probe.labels)
        lab = sb_plain.labels.reshape(-1, sb_plain.labels.size(-1))
        n_labelled_real = int(((lab[:, 6] == 0) & (lab[:, 4] == 1)).sum())
        sfeats = sb_probe.start.reshape(-1, sb_probe.start.size(-1)).numpy()
        s_grade = sfeats[s_slots.reshape(-1), 9:13].argmax(axis=1).reshape(-1, 4) + 1
        record("scorer layout: three probes before every labelled committed row; slot g holds grade g",
               {"probed": len(s_real), "labelled_committed_rows": n_labelled_real, "shift_fixed": s_changed,
                "rows_plain": int(sb_plain.labels.shape[1]), "rows_probed": int(sb_probe.labels.shape[1])},
               len(s_real) == n_labelled_real and bool((s_grade == np.arange(1, 5)[None, :]).all()) and s_changed > 0)
        sa, sb = forward(model, sb_plain), forward(model, sb_probe)
        keep = np.flatnonzero(sb_probe.labels[0, :, 1].numpy() >= 0)
        s_worst = {name: float((x[0] - y[0, torch.as_tensor(keep)]).abs().max()) for name, x, y in zip(names, sa, sb) if x.dim() >= 3}
        s_in = float((sb_plain.start.reshape(-1, sb_plain.start.size(-1)) - sb_probe.start.reshape(-1, sb_probe.start.size(-1))[torch.as_tensor(keep)]).abs().max())
        record("scorer layout: non-probe rows' inputs bitwise, outputs within 1e-5", {"inputs": s_in, **s_worst},
               s_in == 0.0 and max(s_worst.values()) <= 1e-5)
        # ---------------------------------------------------------------- 4. order exact at every horizon
        H = np.concatenate([[1.0], np.logspace(0, np.log10(100 * 365.25 * 86400), 240)])
        order = {}

        def order_check(tag, outs, real, sl, tlab):
            ht = np.concatenate([np.repeat(H[None, :], len(real), 0), tlab[real][:, None]], axis=1)     # [P, 242]
            flat = sl.reshape(-1)
            # float32 torch path (the model's own curve and rectifier)
            w, s_raw, d_raw = (x.reshape(-1, x.size(-1))[torch.as_tensor(flat)] for x in (outs[1], outs[4], outs[5]))
            t32 = torch.as_tensor(np.repeat(ht, 4, axis=0), dtype=torch.float32).unsqueeze(-1)
            r32 = model.gru_forgetting_curve(w.float().unsqueeze(1), s_raw.float().unsqueeze(1), d_raw.float().unsqueeze(1), t32)
            r32 = r32.view(len(real), 4, -1).transpose(1, 2)                                           # [P, H, 4]
            q32 = gr.pava4_torch(r32)
            r64 = curves_np(outs, flat, np.repeat(ht, 4, axis=0)).reshape(len(real), 4, -1).transpose(0, 2, 1)
            q64 = gr.pava4_numpy(r64)
            raw_break = float((~np.all(np.diff(r64, axis=-1) >= 0, axis=-1)).any(axis=1).mean())
            inc32 = float((q32[:, 1:-1, :] - q32[:, :-2, :]).max())   # rectified curves non-increasing in t (sorted horizons only)
            order[tag] = {"reviews": len(real), "horizons": int(ht.shape[1]), "raw_reviews_broken": raw_break,
                          "rectified_breaks_f32": int((q32[..., 1:] < q32[..., :-1]).sum()),
                          "rectified_breaks_f64": int((q64[..., 1:] < q64[..., :-1]).sum()),
                          "f32_vs_f64_max_abs": float(np.abs(q32.double().numpy() - q64).max()),
                          "max_increase_in_t_f32": inc32}
            return order[tag]["rectified_breaks_f32"] == 0 and order[tag]["rectified_breaks_f64"] == 0

        t_s = sb_probe.labels[..., 0].reshape(-1).double().numpy()
        ok_order = order_check("trained (SM72s epoch 21), scorer layout", sb, s_real, s_slots, t_s)
        for k in range(3):
            torch.manual_seed(100 + k)
            fresh = vm.VariantSrsRWKV(variants.variant("w64")).float()
            # a fresh model's GRU weights start at zero (all grades identical); perturb them so the curves differ
            with torch.no_grad():
                for name in ("gru_w_weight", "gru_s_weight", "gru_d_weight"):
                    getattr(fresh, name).normal_(0.0, 0.3)
            fresh.eval()
            ok_order &= order_check(f"random init {k} (GRU head weights ~ N(0, 0.3))", forward(fresh, sb_probe), s_real, s_slots, t_s)
        record("order exact: zero rectified breaks at 242 horizons per probed review (float32 torch, float64 NumPy)", order,
               ok_order and order["trained (SM72s epoch 21), scorer layout"]["raw_reviews_broken"] > 0)
        rng = np.random.default_rng(95)
        adv = rng.random((1_000_000, 4))
        adv[::7] = np.round(adv[::7] * 4) / 4                          # ties
        adv[1::7] = np.sort(adv[1::7], axis=1)[:, ::-1]                # fully reversed
        adv[2::7] = np.sort(adv[2::7], axis=1)                         # ordered
        adv[3::7] = 1e-5 + (1 - 2e-5) * (adv[3::7] > 0.5)              # extremes
        adv[4::7] = 0.5 + 1e-7 * rng.standard_normal((len(adv[4::7]), 4))   # near-ties
        q64 = gr.pava4_numpy(adv)
        q32 = gr.pava4_torch(torch.as_tensor(adv, dtype=torch.float32))
        sub = adv[:20000]
        scalar = np.array([gr.pava4_scalar(v) for v in sub])
        ordered = np.all(np.diff(adv, axis=1) >= 0, axis=1)
        a32 = torch.as_tensor(adv, dtype=torch.float32)
        record("adversarial 4-vectors (1,000,000 float64 + float32): zero breaks; == stack PAVA; identity on ordered rows, bitwise",
               {"breaks_f64": int((q64[:, 1:] < q64[:, :-1]).sum()), "breaks_f32": int((q32[:, 1:] < q32[:, :-1]).sum()),
                "max_abs_vs_stack_pava": float(np.abs(q64[:20000] - scalar).max()), "ordered_rows": int(ordered.sum())},
               (q64[:, 1:] >= q64[:, :-1]).all() and bool((q32[:, 1:] >= q32[:, :-1]).all())
               and float(np.abs(q64[:20000] - scalar).max()) <= 1e-12
               and np.array_equal(q64[ordered], adv[ordered]) and torch.equal(q32[torch.as_tensor(ordered)], a32[torch.as_tensor(ordered)]))
        # intervals at any target retention + the fork's PAVA is a no-op on rectified curves
        w, s_raw, d_raw = (x.reshape(-1, x.size(-1)).double().numpy()[s_slots.reshape(-1)] for x in (sb[1], sb[4], sb[5]))
        n3 = w.shape[-1]
        params = np.concatenate([w, s_raw, d_raw], axis=-1).reshape(len(s_real), 4, 3 * n3)
        targets = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95, 0.97, 0.99)
        iv_breaks = {t: int((np.diff(sr.intervals(params, t, True), axis=1) < 0).sum()) for t in targets}
        iv_raw = {t: float((np.diff(sr.intervals(params, t, False), axis=1) < 0).any(axis=1).mean()) for t in (0.8, 0.9, 0.95)}
        fk = sr.fork_display(params, 0.90, True, True)
        fk0 = sr.fork_display(params, 0.90, True, False)
        fraw = sr.fork_display(params, 0.90, False, True)
        record("rectified intervals ordered at 12 target retentions; fork display with == without its PAVA (bitwise); "
               "the fork's PAVA on the raw curves == the rectified curves' display",
               {"interval_breaks": iv_breaks, "raw_reviews_broken": iv_raw,
                "fork_pava_changes_rectified": int((fk != fk0).sum()),
                "fork_raw_pava_vs_rectified_max_rel": float(np.max(np.abs(fraw - fk) / np.maximum(fk, 1e-9)))},
               all(v == 0 for v in iv_breaks.values()) and np.array_equal(fk, fk0) and max(iv_raw.values()) > 0)
        # ---------------------------------------------------------------- 5. TorchScript
        path = Path(tmp) / "batch.pt"
        torch.save(batch_probe, path)
        env = {k: v for k, v in os.environ.items() if k != "RWKV_NO_JIT"}
        proc = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--jit-batch", str(path)],
                              capture_output=True, text=True, env=env, cwd=ROOT)
        try:
            child = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            child = {"error": (proc.stderr or proc.stdout)[-800:]}
        eager = s_on.average_loss.item()
        record("TorchScript model compiles with grade_rectify; loss == eager (eval mode)", {**child, "eager": eager},
               child.get("scripted") is True and abs(child.get("loss", 1e9) - eager) <= 1e-5 * max(1.0, abs(eager)))
        # ---------------------------------------------------------------- 6. preview: curve LogLoss raw vs rectified
        with torch.no_grad():
            st_raw, st_rect = model_of(vm.VariantSrsRWKV, False), model_of(vm.VariantSrsRWKV, True)
            st_raw.eval()
            st_rect.eval()
            q_raw, q_rect = st_raw.get_loss(sb_probe), st_rect.get_loss(sb_probe)
        lab = sb_probe.labels.reshape(-1, sb_probe.labels.size(-1))
        mask = ((lab[:, 6] == 0) & (lab[:, 4] == 1)).numpy()
        y = lab[:, 2].numpy()[mask]

        def ll(p):
            p = np.clip(p.reshape(-1).double().numpy()[mask], 1e-7, 1 - 1e-7)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        record("preview (not a gate): curve LogLoss of the scorer layout on the check user, raw vs rectified (fp32 CPU)",
               {"reviews": int(mask.sum()), "raw": ll(q_raw.p_curve), "rectified": ll(q_rect.p_curve),
                "rectified_changed_share": float(np.mean(q_raw.p_curve.reshape(-1).numpy()[mask] != q_rect.p_curve.reshape(-1).numpy()[mask])),
                "label_y_is_hard": bool(np.isin(y, (0.0, 1.0)).all())}, True)
    ok = all(r["ok"] for r in results.values())
    report = {"ok": ok, "spec": spec, "user": args.user, "reviews": args.reviews, "scorer_reviews": args.scorer_reviews,
              "start_weights": str(START), "seconds": round(time.perf_counter() - started, 1), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"[sm95-check] {'PASS' if ok else 'FAIL'} ({len(results)} checks) -> {args.output}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
