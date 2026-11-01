"""SM04: CPU inference timing of the recurrent per-review step, fork 2.76M vs w64 vs w32 (docs/SMALL_MODEL.md, Q1).

Times `review()` (one review of every stream: features2card, five RWKV streams, heads) of
  fork  - the fork's `srs_model_rnn.SrsRWKVRnn` (imported read-only from the fork checkout, no bytecode written there),
  w64   - expertium.variants.build("w64", rnn=True)   (real checkpoint if present),
  w32   - expertium.variants.build("iter36", rnn=True) (d=32; random init, speed does not depend on the values),
in f32 and "int8" weights, at batch 1 (the runtime case: one review) and batch 256 (cards in parallel, amortises
Python dispatch). States are warmed with real recurrent steps, then every timed step reuses the same states.

"int8" is the shipped w8-pc weight layout (compress_quant fake-quant, per output channel) executed with torch's
dynamic int8 Linear kernels (quantized engine recorded in the JSON; activations are quantized on the fly, so the arithmetic is not the fork's
dequantize-at-load f32). Treat it as an indication of an int8 kernel, not of the fork's runtime: the fork loads f32.
GRU-head / functional linears stay f32. MACs are analytic: nn.Linear in*out per row (hooks) + 3 * numel per
WKV matrix state (decay, k*v outer product, r read) + 3 * N * head width for the power-curve head.

    python scripts/small_model_cpu_timing.py --threads 1 --out cpu-timing.json [--smoke]

SM18 (`--impl eager,numpy`, default `eager` = the output above unchanged): `numpy` adds the low-overhead step of
scripts/small_model_fast_step.py (fused matmuls, folded LayerNorm affines, no torch dispatch; f32 only) as
precisions["numpy_f32"], plus an exactness check against the eager model over `--check-steps` real reviews of one
user (max |delta| of the predicted p_curve / p_imm and of the rating logits).
"""
import argparse
import importlib
import json
import platform
import statistics
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

FORK_DIR = Path(r"C:\Users\jonat\workspace\anki\qt\aqt\rwkv_inference")
FORK_WEIGHTS = FORK_DIR / "RWKV_trained_on_5000_10000.pth"
W64_CHECKPOINT = ROOT / ("artifacts/accuracy/experiments/scale-w64kd-gate-cached-ctx2048/run/epoch-02/expertium/"
                         "E89-F65e-F77a/master.pth")


def cpu_model():
    try:
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        if out:
            return out
    except Exception:
        pass
    return platform.processor()


def load_state(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    return {k: v.float() for k, v in state.items()}


def build_fork():
    sys.dont_write_bytecode = True  # never write __pycache__ into the fork checkout
    package = types.ModuleType("fork_rwkv_inference")
    package.__path__ = [str(FORK_DIR)]
    sys.modules["fork_rwkv_inference"] = package
    arch = importlib.import_module("fork_rwkv_inference.architecture")
    rnn = importlib.import_module("fork_rwkv_inference.srs_model_rnn")
    model = rnn.SrsRWKVRnn(arch.DEFAULT_ANKI_RWKV_CONFIG).float().eval()
    real = FORK_WEIGHTS.exists()
    if real:
        model.load_state_dict(load_state(FORK_WEIGHTS), strict=True)
    return model, real


def build_variant(arch, checkpoint):
    from expertium.variants import build
    model = build(arch, rnn=True).float().eval()
    real = checkpoint is not None and Path(checkpoint).exists()
    if real:
        model.load_state_dict(load_state(checkpoint), strict=True)
    else:  # random init: rkvdag_lerp is torch.empty (garbage/NaN) and some matrices are zero; noise like the export parity
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.endswith("rkvdag_lerp"):  # torch.empty in the model: finite garbage up to 1e38 -> inf
                    p.fill_(0.5)
                p.copy_(torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0))
                if p.dim() >= 2 and not name.endswith("rkvdag_lerp"):
                    p.add_(torch.randn(p.shape, generator=g) * 0.02)
    return model, real


def make_int8(model):
    """w8-pc fake-quant of every 2-D .weight (the shipped layout), then torch dynamic int8 Linear kernels."""
    import copy
    from scripts import compress_quant as cq
    q = copy.deepcopy(model)
    state = q.state_dict()
    q.load_state_dict(cq.quantize_state_dict(state, cq.linear_weight_names(state), 8, None))
    qc = torch.ao.quantization.per_channel_dynamic_qconfig
    return torch.ao.quantization.quantize_dynamic(q, {torch.nn.Linear: qc}, dtype=torch.qint8).eval()


def tensors_in(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from tensors_in(v)
    elif isinstance(obj, (tuple, list)):
        for v in obj:
            yield from tensors_in(v)


def warm(model, batch, steps=3):
    g = torch.Generator().manual_seed(0)
    feats = torch.randn(batch, 92, generator=g) * 0.5
    states = [None] * 5
    with torch.inference_mode():
        for _ in range(steps):
            out = model.review(feats, *states)
            states = list(out[3:])
    return feats, states


def count_macs(model):
    """MACs of one review (batch 1): Linear in*out via hooks, WKV 3*numel per matrix state, GRU head."""
    total = {"linear": 0}

    def hook(m, inp, out):
        total["linear"] += m.in_features * m.out_features
    feats, states = warm(model, 1, steps=2)
    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, torch.nn.Linear)]
    with torch.inference_mode():
        out = model.review(feats, *states)
    for h in handles:
        h.remove()
    wkv = sum(3 * t.numel() for t in tensors_in(out[3:]) if t.dim() >= 4)
    gru = 3 * model.num_curves * model.w_head_dim if getattr(model, "gru_on", False) else 0
    return {"linear": total["linear"], "wkv": wkv, "gru_head": gru, "total": total["linear"] + wkv + gru}


def time_steps(model, batch, steps, repeats, warmup):
    feats, states = warm(model, batch)
    samples, means = [], []
    with torch.inference_mode():
        for _ in range(warmup):
            model.review(feats, *states)
        for _ in range(repeats):
            rep = []
            for _ in range(steps):
                t0 = time.perf_counter()
                model.review(feats, *states)
                rep.append(time.perf_counter() - t0)
            samples += rep
            means.append(statistics.fmean(rep))
    q = statistics.quantiles(samples, n=100, method="inclusive")
    step = {"median_ms": statistics.median(samples) * 1e3, "p90_ms": q[89] * 1e3,
            "mean_ms": statistics.fmean(samples) * 1e3, "min_ms": min(samples) * 1e3,
            "repeat_means_ms": [m * 1e3 for m in means], "n": len(samples)}
    return {"batch": batch, "step": step,
            "per_review_median_ms": step["median_ms"] / batch, "per_review_p90_ms": step["p90_ms"] / batch,
            "reviews_per_s_median": batch / (step["median_ms"] / 1e3)}


DATA = Path(r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
STREAMS = ["card_id", "deck_id", "note_id", "preset_id", "user_id"]


def real_history(n, user=7):
    """First n reviews of a real user: (features (n,92) f32, ids {stream: (n,)}, skips (n,), label seconds (n,))."""
    import os
    os.environ.setdefault("GPU_BATCH1_DATA_PATH", str(DATA))
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare
    from scripts.gpu_batch1 import make_samples
    frame = get_rwkv_data(DATA, user)
    sample = make_samples(frame.iloc[:n], user, n)[0]
    batch = prepare([sample], seed=2025)
    return batch.start.float(), sample.ids, sample.skips, batch.labels[0, :, 0], sample


def curve_p(model, out, t):
    """Recall probability and p_imm from a head-output dict (tensors); t (1,1) elapsed seconds."""
    if hasattr(model, "curve_probability"):
        p_curve = model.curve_probability(out, t)
    else:  # fork SrsRWKVRnn.run()
        raw = model.forgetting_curve(out["w"], t)
        p_curve = torch.sigmoid(torch.log(raw / (1 - raw)) + model.interp(out["ahead"], t))
    p = torch.softmax(out["p_logits"], dim=-1)
    return p_curve.reshape(()), torch.clamp(1.0 - p[0, 0], min=1e-5, max=1.0 - 1e-5), out["p_logits"].reshape(-1)


@torch.inference_mode()
def history_outputs(model, fast, n):
    """Per-review (p_curve, p_imm, rating logits) over n real reviews, eager (`fast` None) or the numpy step."""
    import numpy as np
    feats, ids, skips, times, _ = real_history(n)
    n = feats.size(0)
    ents = {s: {} for s in STREAMS}
    rows = []
    for i in range(n):
        keys = {s: int(ids[s][i]) for s in STREAMS}
        states = [ents[s].get(keys[s]) for s in STREAMS]
        if fast is not None:
            out, nxt = fast.step_dict(np.ascontiguousarray(feats[i:i + 1].numpy()), states)
            out = {k: torch.from_numpy(np.array(v)) for k, v in out.items()}
        elif hasattr(model, "step"):
            by_name = dict(zip(STREAMS, states))
            out, nxt_by = model.step(feats[i:i + 1], {k: v for k, v in by_name.items()})
            nxt = [nxt_by[s] for s in STREAMS]
        else:
            card, deck, note, preset, user = states
            res = model.review(feats[i:i + 1], card, note, deck, preset, user)
            out = {"ahead": res[0], "w": res[1], "p_logits": res[2]}
            nxt = [res[3], res[5], res[4], res[6], res[7]]
        if not bool(skips[i]):
            for s, st in zip(STREAMS, nxt):
                ents[s][keys[s]] = st
        rows.append(torch.cat([a.reshape(-1) for a in curve_p(model, out, times[i].view(1, 1))]))
    return torch.stack(rows)


def exactness(model, steps):
    from scripts.small_model_fast_step import NumpyStep
    fast = NumpyStep(model)
    ref = history_outputs(model, None, steps)
    new = history_outputs(model, fast, steps)
    delta = (ref - new).abs()
    return {"steps": int(ref.size(0)), "max_abs_p_curve": delta[:, 0].max().item(), "max_abs_p_imm": delta[:, 1].max().item(),
            "max_abs_p_probabilities": delta[:, :2].max().item(), "max_abs_rating_logits": delta[:, 2:].max().item(),
            "user": 7, "state_int8_rounding": "not covered by this script (no s8 path in the eager timing step)"}


def time_numpy(fast, batch, steps, repeats, warmup):
    import numpy as np
    g = np.random.default_rng(0)
    feats = (g.standard_normal((batch, 92)) * 0.5).astype(np.float32)
    states = [None] * 5
    for _ in range(3):
        states = fast.review(feats, *states)[3:]
    states = list(states)
    # review() takes (card, note, deck, preset, user) and returns the same order
    samples, means = [], []
    for _ in range(warmup):
        fast.review(feats, *states)
    for _ in range(repeats):
        rep = []
        for _ in range(steps):
            t0 = time.perf_counter()
            fast.review(feats, *states)
            rep.append(time.perf_counter() - t0)
        samples += rep
        means.append(statistics.fmean(rep))
    q = statistics.quantiles(samples, n=100, method="inclusive")
    step = {"median_ms": statistics.median(samples) * 1e3, "p90_ms": q[89] * 1e3, "mean_ms": statistics.fmean(samples) * 1e3,
            "min_ms": min(samples) * 1e3, "repeat_means_ms": [m * 1e3 for m in means], "n": len(samples)}
    return {"batch": batch, "step": step, "per_review_median_ms": step["median_ms"] / batch,
            "per_review_p90_ms": step["p90_ms"] / batch, "reviews_per_s_median": batch / (step["median_ms"] / 1e3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true", help="100 batch-1 steps and 20 batch-256 steps, 1 repeat")
    parser.add_argument("--steps", type=int, default=1000, help="batch-1 steps per repeat")
    parser.add_argument("--steps256", type=int, default=60, help="batch-256 steps per repeat")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--models", default="fork,w64,w32")
    parser.add_argument("--impl", default="eager", help="comma list of eager,numpy (SM18); eager alone = the SM04 output")
    parser.add_argument("--check-steps", type=int, default=1000, help="real reviews for the numpy exactness check")
    args = parser.parse_args()
    if args.smoke:
        args.steps, args.steps256, args.repeats, args.warmup = 100, 20, 1, 10
    torch.set_num_threads(args.threads)
    torch.empty_like = torch.zeros_like  # step() seeds v0 with empty_like: uninitialised memory can hold NaN
    builders = {"fork": build_fork, "w64": lambda: build_variant("w64", W64_CHECKPOINT),
                "w32": lambda: build_variant("iter36", None)}
    import os
    result = {"machine": {"cpu": cpu_model(), "platform": platform.platform(), "torch": torch.__version__,
                          "threads": args.threads, "logical_cpus": os.cpu_count(),
                          "quantized_engine": torch.backends.quantized.engine},
              "settings": {"smoke": args.smoke, "steps_batch1": args.steps, "steps_batch256": args.steps256,
                           "repeats": args.repeats, "warmup": args.warmup},
              "note": "int8 = w8-pc weights run with torch dynamic int8 Linear (engine in machine.quantized_engine); the fork loads f32. "
                      "Models with real_weights false use random init (speed does not depend on values).",
              "models": {}}
    for name in args.models.split(","):
        started = time.perf_counter()
        model, real = builders[name]()
        entry = {"parameters": sum(p.numel() for p in model.parameters()), "real_weights": real,
                 "macs_per_review": count_macs(model), "precisions": {}}
        impls = args.impl.split(",")
        if "numpy" in impls:
            from scripts.small_model_fast_step import NumpyStep
            fast = NumpyStep(model)
            entry["precisions"]["numpy_f32"] = {
                "batch1": time_numpy(fast, 1, args.steps, args.repeats, args.warmup),
                "batch256": time_numpy(fast, 256, args.steps256, args.repeats, max(2, args.warmup // 10))}
            b1, b256 = (entry["precisions"]["numpy_f32"][k] for k in ("batch1", "batch256"))
            print(f"[{name} numpy_f32] b1 {b1['per_review_median_ms']:.3f} ms (p90 {b1['per_review_p90_ms']:.3f}); "
                  f"b256 {b256['per_review_median_ms']:.4f} ms/review", flush=True)
            if args.check_steps:
                entry["numpy_exactness"] = exactness(model, args.check_steps)
                print(f"[{name} numpy exactness] {entry['numpy_exactness']}", flush=True)
        for precision in (("f32", "int8") if "eager" in impls else ()):
            m = model if precision == "f32" else make_int8(model)
            entry["precisions"][precision] = {
                "batch1": time_steps(m, 1, args.steps, args.repeats, args.warmup),
                "batch256": time_steps(m, 256, args.steps256, args.repeats, max(2, args.warmup // 10))}
            b1, b256 = (entry["precisions"][precision][k] for k in ("batch1", "batch256"))
            print(f"[{name} {precision}] b1 {b1['per_review_median_ms']:.3f} ms (p90 {b1['per_review_p90_ms']:.3f}); "
                  f"b256 {b256['per_review_median_ms']:.4f} ms/review", flush=True)
        entry["seconds"] = time.perf_counter() - started
        result["models"][name] = entry
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"[done] {args.out}", flush=True)


if __name__ == "__main__":
    main()
