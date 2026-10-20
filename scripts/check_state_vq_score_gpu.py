"""SM44 check: the held-out scorer's GPU `+vq<recipe>` path (scripts/state_vq.py "GPU scoring": the eager model's
state-VQ slots, fused kernel for rank-1 streams with the Gram-squaring factorization, torch step loop for rank-2
streams, shift codec on the device) against the CPU scorer's `+vq<recipe>` path (scripts/state_vq.py Hooks: reference
WKV + shift gather, deployed eigh codec, float64), full histories of small evaluation users, w8 weights.

  cpu   CPU-first, no GPU. Per case (model, codebooks, recipe) and user: the CPU scorer (reference) vs the GPU path's
        code on CPU tensors, (A) "torch": rank-1 streams through the torch codec with factor "sq" (the kernel's
        arithmetic in torch), and (B) "emulated" (--emulate-users): the UNCHANGED .cu kernel through the MSVC thread
        emulation (scripts/state_vq_cuda_emu.py) inside the same scorer path. Writes the reference predictions to
        --ref-dir for the gpu mode.
  gpu   (queue ticket, ~5 min) the GPU path on CUDA vs the stored CPU references; the float-state GPU vs CPU difference
        of the same users (base numerics, reported only); then timing (reviews/s of w8+s8 and of every recipe on
        --timing-users mid-size users). Exit 1 on a mismatch.
Pass: benchmark LogLoss (curve = ahead head, instant) within --tol (1e-5) of the CPU scorer for every user and case;
Discrete codec tolerance (gpu mode, +vq cases only): per user within --discrete-tol (3e-4) on both heads, and the review-weighted
mean of the signed d within --discrete-mean-tol (1e-4); rule, per-user d and the mean are recorded in the JSON.
max |dp| over all labelled predictions is reported (near-tie code flips move a few predictions, not the metric).

  .venv/Scripts/python.exe -X utf8 scripts/check_state_vq_score_gpu.py cpu [--emulate-users 1]
  .venv/Scripts/python.exe -X utf8 scripts/check_state_vq_score_gpu.py gpu
"""
from argparse import ArgumentParser
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("RWKV_NO_JIT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts import small_model_heldout as smh  # noqa: E402  (installs the scorer's patches)
from scripts import compress_common as cc  # noqa: E402
from scripts import compress_quant as cq  # noqa: E402
from scripts import state_vq  # noqa: E402

SM = ROOT / "artifacts" / "small-model"
SM42 = ("C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay1ep32k-lanes/"
        "expertium/E89-F65e-F77a/master.pth")
SM32C_DIR = SM / "scale-w64kd-gate-cached-ctx32768-cont-decay1ep32k-lanes-vqlearn" / "expertium" / "E89-F65e-F77a"
KMEANS = SM / "state-vq" / "codebooks.npz"
# (case name, checkpoint, codebooks, recipe, how many of the check users)
CASES = [("sm42/w8+vq378x412", SM42, KMEANS, "vq378x412", 5),
         ("sm42/w8+vq770x828", SM42, KMEANS, "vq770x828", 3),
         ("sm32c/w8+vq378x412", str(SM32C_DIR / "master.pth"), SM32C_DIR / "state-vq-books.npz", "vq378x412", 2)]
OUT = SM / "sm44"


def pool_users(n, smallest=True):
    """The n smallest (or near-median) users of the 732 pool by review count."""
    counts = cc.run_info()["review_counts"]
    pool = json.loads((SM / "eval-users-732.json").read_text())["users"]
    order = sorted(pool, key=lambda u: (counts[str(u)], u))
    if smallest:
        return order[:n], counts
    mid = len(order) // 2
    return order[mid:mid + n], counts


def load_model(path, device):
    base = {k: v.float() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
    model = smh.build_expertium("w64", base)
    names = cq.linear_weight_names(base)
    bits, group = cq.WEIGHT_SCHEMES["w8-pc"]
    model.load_state_dict(cq.quantize_state_dict(base, names, bits, group))
    smh.set_expertium_state_bits(model, 0)   # float card/note state on the long WKV (the +vq slots replace it)
    return model.to(device)


def evaluate(model, batch, labels, user):
    from scripts.compress_metrics import metrics
    from scripts.full_run_metrics import evaluate_with_predictions
    with torch.inference_mode():
        values, preds = evaluate_with_predictions(model, [{"batch": batch}])
    return values, preds, metrics(user, labels, preds)


def exact_logloss(labels, preds):
    """Unrounded benchmark LogLoss per head over the labelled test reviews (y = the rating target)."""
    ths = [th for th in labels[0] if th in preds["ahead"] and th in preds["instant"] and th in preds["rating"]]
    y = np.clip(np.array([preds["rating"][th] for th in ths], dtype=np.float64), 0, 1)
    out = {}
    for head in ("ahead", "instant"):
        p = np.clip(np.array([preds[head][th] for th in ths], dtype=np.float64), 1e-15, 1 - 1e-15)
        out[head] = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()) if len(ths) else None
    return out


def compare(ref, cand, tol, rule="strict"):
    """ref / cand: dicts with preds, bench, exact. Returns the comparison entry."""
    d = {}
    for head in ("ahead", "instant"):
        a, b = ref["preds"][head], cand["preds"][head]
        common = sorted(set(a) & set(b))
        dp = np.abs(np.array([a[k] for k in common]) - np.array([b[k] for k in common])) if common else np.zeros(0)
        d[head] = {"predictions": len(common), "missing": len(set(a) ^ set(b)),
                   "max_abs_dp": float(dp.max()) if dp.size else 0.0, "mean_abs_dp": float(dp.mean()) if dp.size else 0.0,
                   "n_dp_over_1e-4": int((dp > 1e-4).sum()),
                   "benchmark_logloss_ref": ref["bench"].get(head, {}).get("LogLoss"),
                   "benchmark_logloss": cand["bench"].get(head, {}).get("LogLoss"),
                   "d_benchmark_logloss": (None if head not in ref["bench"] else
                                           cand["bench"][head]["LogLoss"] - ref["bench"][head]["LogLoss"]),
                   "d_exact_logloss": cand["exact"][head] - ref["exact"][head]}
    for key in ("curve_log_loss", "instant_log_loss"):
        d[f"d_{key}"] = cand["values"][key] - ref["values"][key]
    ok = all(abs(d[h]["d_exact_logloss"]) <= tol and d[h]["missing"] == 0 and
             (d[h]["d_benchmark_logloss"] is None or abs(d[h]["d_benchmark_logloss"]) <= tol + 1e-6)
             for h in ("ahead", "instant"))
    d["ok"] = bool(ok)
    d["rule"] = rule
    d["tol"] = tol
    return d


def run_cpu_reference(model, layout_cpu, codecs_cpu, batch, labels, user):
    state_vq.set_scoring(layout_cpu["slots"], None)
    state_vq.configure(layout_cpu["hooks"], codecs_cpu)
    values, preds, bench = evaluate(model, batch, labels, user)
    return {"values": values, "preds": preds, "bench": bench, "exact": exact_logloss(labels, preds)}


CALLS = {"wkv": 0, "shift": 0, "kernel": 0}


def count_calls():
    """Wrap the scoring entry points (after install_scoring) so the report shows the slot path really ran."""
    from expertium.model import rwkv_model as erm
    wkv, shift, kernel = state_vq.score_wkv, state_vq.score_shift, state_vq._score_kernel

    def counted(fn, key):
        def inner(*a, **k):
            CALLS[key] += 1
            return fn(*a, **k)
        return inner
    erm.vq_wkv, erm.vq_shift = counted(wkv, "wkv"), counted(shift, "shift")
    state_vq._score_kernel = counted(kernel, "kernel")


def run_float(model, layout, batch, labels, user):
    """Float card/note state (no coding on either path): the codec must move the metric away from this."""
    if layout.get("hooks"):
        state_vq.configure(layout["hooks"], None, 0)
    state_vq.set_scoring(layout["slots"], None)
    values, preds, bench = evaluate(model, batch, labels, user)
    return {"values": values, "preds": preds, "bench": bench, "exact": exact_logloss(labels, preds)}


def run_gpu_path(model, layout, codecs, batch, labels, user):
    if layout.get("hooks"):
        state_vq.configure(layout["hooks"], None, 0)   # the CPU Hooks off (float), the slots do the coding
    state_vq.set_scoring(layout["slots"], codecs)
    t0 = time.perf_counter()
    values, preds, bench = evaluate(model, batch, labels, user)
    if batch.labels.is_cuda:
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    state_vq.set_scoring(layout["slots"], None)
    return {"values": values, "preds": preds, "bench": bench, "exact": exact_logloss(labels, preds), "seconds": seconds}


def save_ref(folder, case, user, ref):
    folder.mkdir(parents=True, exist_ok=True)
    keys = sorted(ref["preds"]["ahead"])
    np.savez_compressed(folder / f"{case.replace('/', '_')}_{user}.npz", review_th=np.array(keys, dtype=np.int64),
                        p_ahead=np.array([ref["preds"]["ahead"][k] for k in keys]),
                        p_instant=np.array([ref["preds"]["instant"].get(k, np.nan) for k in keys]),
                        rating=np.array([ref["preds"]["rating"].get(k, np.nan) for k in keys]),
                        meta=np.array(json.dumps({"values": ref["values"], "bench": ref["bench"],
                                                  "exact": ref["exact"]})))


def load_ref(folder, case, user):
    z = np.load(folder / f"{case.replace('/', '_')}_{user}.npz")
    meta = json.loads(str(z["meta"]))
    preds = {"ahead": dict(zip(z["review_th"].tolist(), z["p_ahead"].tolist())),
             "instant": {k: v for k, v in zip(z["review_th"].tolist(), z["p_instant"].tolist()) if not math.isnan(v)},
             "rating": {k: v for k, v in zip(z["review_th"].tolist(), z["rating"].tolist()) if not math.isnan(v)}}
    return {"preds": preds, **meta}


def main():
    parser = ArgumentParser()
    parser.add_argument("mode", choices=("cpu", "gpu"))
    parser.add_argument("--users", type=int, default=5, help="check users: the N smallest of the 732 pool")
    parser.add_argument("--emulate-users", type=int, default=0, help="cpu: also run the emulated .cu kernel on the "
                        "first N users of the sm42 vq378x412 case")
    parser.add_argument("--timing-users", type=int, default=2, help="gpu: near-median users of the 732 pool to time")
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--discrete-tol", type=float, default=3e-4,
                        help="gpu, +vq cases: per-user |d benchmark LogLoss| bound (discrete codec tolerance)")
    parser.add_argument("--discrete-mean-tol", type=float, default=1e-4,
                        help="gpu, +vq cases: bound on |review-weighted mean of the signed d| over the check users")
    parser.add_argument("--factor-dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--ref-dir", default=str(OUT / "check-ref"))
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    cpu = args.mode == "cpu"
    device = torch.device("cpu" if cpu else "cuda")
    fd = getattr(torch, args.factor_dtype)
    if cpu:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    else:
        from scripts import compress_wkv_cuda
        compress_wkv_cuda.register_cuda()
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("HELDOUT_GPU_MEMORY_FRACTION", "0.8")))
    import lmdb
    from scripts.compress_metrics import read_labels
    from scripts.experiment_provenance import process_provenance
    users, counts = pool_users(args.users)
    env = lmdb.open(str(cc.RUN / "label-db"), readonly=True, lock=False, map_size=10_000_000_000)
    run = cc.run_info()
    data = cc.data_path(run)
    ref_dir = Path(args.ref_dir)
    report = {"mode": args.mode, "tol_benchmark_logloss": args.tol, "discrete_tol": args.discrete_tol,
              "discrete_mean_tol": args.discrete_mean_tol, "users": {u: counts[str(u)] for u in users},
              "factor_dtype_kernel": args.factor_dtype, "cases": {}, "provenance": process_provenance()}
    if not cpu:
        report["device"] = torch.cuda.get_device_name(0)
    fails = []
    batches, label_cache = {}, {}
    for u in users:
        batches[u] = cc.full_history_batch(data, u)
        with env.begin() as txn:
            label_cache[u] = read_labels(txn, u)
    models = {}
    for case, ckpt, books_path, name, n_users in CASES:
        if ckpt not in models:
            model = load_model(ckpt, device)
            layout = {"slots": state_vq.install_scoring(model)}
            count_calls()
            if cpu:
                layout["hooks"] = state_vq.install(model)
            models[ckpt] = (model, layout)
        model, layout = models[ckpt]
        books = state_vq.Codebooks.load(books_path)
        codecs = state_vq.score_codecs(name, books, device, fd)
        entry = {"checkpoint": str(ckpt), "codebooks": str(books_path), "codebooks_sha256": state_vq.file_sha256(books_path),
                 "recipe": name, "path": state_vq.score_path(codecs), "users": {}}
        print(f"[sm44-check] {case}: {entry['path']}", flush=True)
        if cpu:
            codecs_cpu = {s: state_vq.StateCodec(rec, books) for s, rec in zip(layout["hooks"], state_vq.recipe(name))}
        for i, u in enumerate(users[:n_users]):
            batch = batches[u].to(device)
            labels = label_cache[u]
            res = {}
            if i == 0:   # the codec is active: float state differs from the reference
                flt = run_float(model, layout, batch, labels, u)
                ref0 = run_cpu_reference(model, layout, codecs_cpu, batch, labels, u) if cpu else load_ref(ref_dir, case, u)
                c = compare(ref0, flt, args.tol)
                res["float_vs_ref"] = {h: {"d_exact_logloss": c[h]["d_exact_logloss"], "max_abs_dp": c[h]["max_abs_dp"]}
                                       for h in ("ahead", "instant")}
                if max(c[h]["max_abs_dp"] for h in ("ahead", "instant")) < 1e-4:
                    fails.append(f"{case} user {u}: float state equals the coded reference (codec not active)")
            calls0 = dict(CALLS)
            if cpu:
                t0 = time.perf_counter()
                ref = run_cpu_reference(model, layout, codecs_cpu, batch, labels, u)
                res["cpu_scorer_seconds"] = round(time.perf_counter() - t0, 2)
                save_ref(ref_dir, case, u, ref)
                state_vq.SCORE_EMULATE = None
                cand = run_gpu_path(model, layout, codecs, batch, labels, u)
                res["torch"] = compare(ref, cand, args.tol)
                res["torch"]["seconds"] = round(cand["seconds"], 2)
                if case == CASES[0][0] and i < args.emulate_users:
                    from scripts import state_vq_cuda_emu as emu
                    state_vq.SCORE_EMULATE = lambda kc, *x: emu.forward(kc, *x, factor_double=kc.factor_dtype == torch.float64)[0]
                    cand = run_gpu_path(model, layout, codecs, batch, labels, u)
                    state_vq.SCORE_EMULATE = None
                    res["emulated"] = compare(ref, cand, args.tol)
                    res["emulated"]["seconds"] = round(cand["seconds"], 2)
            else:
                ref = load_ref(ref_dir, case, u)
                cand = run_gpu_path(model, layout, codecs, batch, labels, u)
                discrete = name.startswith("vq")   # discrete codeword choices: near-tie flips, see AGENT.md
                res["gpu"] = (compare(ref, cand, args.discrete_tol, "discrete") if discrete else
                              compare(ref, cand, args.tol))
                res["gpu"]["seconds"] = round(cand["seconds"], 3)
                res["gpu"]["reviews_per_second"] = round(counts[str(u)] / cand["seconds"], 1)
            res["slot_calls"] = {k: CALLS[k] - calls0[k] for k in CALLS}
            if res["slot_calls"]["wkv"] == 0 or res["slot_calls"]["shift"] == 0:
                fails.append(f"{case} user {u}: the state-VQ slots were not called")
            for kind in ("torch", "emulated", "gpu"):
                if kind in res and not res[kind]["ok"]:
                    fails.append(f"{case} user {u} {kind}")
            entry["users"][str(u)] = res
            summary = {k: (v["ahead"]["d_exact_logloss"], v["instant"]["d_exact_logloss"],
                           max(v["ahead"]["max_abs_dp"], v["instant"]["max_abs_dp"]), v["ok"])
                       for k, v in res.items() if isinstance(v, dict) and "ok" in v}
            print(f"[sm44-check] {case} user {u} ({counts[str(u)]} reviews): (dLL curve, dLL instant, max|dp|, ok) "
                  f"{summary}", flush=True)
        if not cpu and name.startswith("vq"):   # discrete rule 2: no sign bias over the check users
            rows = [(counts[str(u)], entry["users"][str(u)]["gpu"]) for u in users[:n_users]]
            wmean = {}
            for h in ("ahead", "instant"):
                ds = [(n, g[h]["d_benchmark_logloss"] if g[h]["d_benchmark_logloss"] is not None
                       else g[h]["d_exact_logloss"]) for n, g in rows]
                wmean[h] = sum(n * d for n, d in ds) / sum(n for n, _ in ds)
            entry["discrete_rule"] = {"rule": "discrete codec tolerance", "per_user_tol": args.discrete_tol,
                                      "weighted_mean_tol": args.discrete_mean_tol,
                                      "per_user_d_benchmark_logloss": {str(u): {h: (g[h]["d_benchmark_logloss"]
                                          if g[h]["d_benchmark_logloss"] is not None else g[h]["d_exact_logloss"])
                                          for h in ("ahead", "instant")} for u, (_, g) in zip(users[:n_users], rows)},
                                      "review_weighted_mean_d": wmean,
                                      "mean_ok": all(abs(v) <= args.discrete_mean_tol for v in wmean.values())}
            print(f"[sm44-check] {case} discrete rule: weighted mean d {wmean} (tol {args.discrete_mean_tol})", flush=True)
            if not entry["discrete_rule"]["mean_ok"]:
                fails.append(f"{case} weighted mean d {wmean} above {args.discrete_mean_tol}")
        report["cases"][case] = entry
    # base numerics (reported, not gated): float card/note state, no codec on either side, SM42 w8
    model, layout = models[SM42]
    base = {}
    for u in users:
        batch = batches[u].to(device)
        flt = run_float(model, layout, batch, label_cache[u], u)
        if cpu:
            save_ref(ref_dir, "sm42/w8-float", u, flt)
        else:
            c = compare(load_ref(ref_dir, "sm42/w8-float", u), flt, args.tol)
            base[str(u)] = {h: {"d_exact_logloss": c[h]["d_exact_logloss"], "max_abs_dp": c[h]["max_abs_dp"]}
                            for h in ("ahead", "instant")}
            print(f"[sm44-check] base (float state) GPU vs CPU user {u}: {base[str(u)]}", flush=True)
    if base:
        report["base_float_state_gpu_vs_cpu"] = base
    if not cpu:   # timing on mid-size users: w8+s8 (the plain GPU scorer) vs each recipe
        t_users, _ = pool_users(args.timing_users, smallest=False)
        model, layout = models[SM42]
        timing = {}
        for u in t_users:
            batch = cc.full_history_batch(data, u).to(device)
            with env.begin() as txn:
                labels = read_labels(txn, u)
            row = {"reviews": counts[str(u)]}
            state_vq.set_scoring(layout["slots"], None)
            smh.set_expertium_state_bits(model, 8)
            evaluate(model, batch, labels, u)   # warm-up
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            evaluate(model, batch, labels, u)
            torch.cuda.synchronize()
            row["w8+s8_seconds"] = round(time.perf_counter() - t0, 3)
            smh.set_expertium_state_bits(model, 0)
            for name in sorted({c[3] for c in CASES}):
                codecs = state_vq.score_codecs(name, state_vq.Codebooks.load(KMEANS), device, fd)
                cand = run_gpu_path(model, layout, codecs, batch, labels, u)
                row[f"w8+{name}_seconds"] = round(cand["seconds"], 3)
            timing[str(u)] = row
            print(f"[sm44-check] timing user {u}: {row}", flush=True)
        total = sum(r["reviews"] for r in timing.values())
        report["timing"] = {"users": timing, "reviews_per_second": {
            key.replace("_seconds", ""): round(total / sum(r[key] for r in timing.values()), 1)
            for key in next(iter(timing.values())) if key.endswith("_seconds")}}
        print(f"[sm44-check] reviews/s: {report['timing']['reviews_per_second']}", flush=True)
    env.close()
    report["fails"] = fails
    report["passed"] = not fails
    out = Path(args.output) if args.output else OUT / f"check-sm44-{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"[sm44-check] {'ALL OK' if not fails else 'FAIL: ' + '; '.join(fails)} -> {out}", flush=True)
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
