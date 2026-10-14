#!/usr/bin/env python
"""CPU checks for the Expertium catch-up options (docs/EXPERTIUM_CATCH_UP.md).

Proves the defaults are unchanged and each option does what it claims, on one real user's batch:

  1. architecture: VariantSrsRWKV("iter36") == SrsRWKV bitwise (initial weights, loss, every gradient);
     w80 has 558,209 parameters (upstream 558,212 minus PAVA's 3); strict parallel -> recurrent load; the
     canonical gather/scatter path run stream-major reproduces the sequential forward exactly; recurrent
     predictions match the parallel model row by row (iter36, w80, w80-sequential, and the vendored RNN).
  2. lr_schedule: the default and a never-decaying schedule equal the accepted R01 expression bitwise.
  3. optimizer: the F82 default grouping is unchanged; expertium-iter53 puts every 2-D weight-decay matrix and
     the LoRA matrices on Muon, the rest on AdamW; one step is finite and restores the written learning rate.
  4. end_to_start_intervals: unset leaves get_rwkv_data unchanged; set subtracts the duration (sentinel kept).
  5. distillation: alpha 0 reproduces the plain loss; the fork teacher loads strictly (if present).

  .venv/Scripts/python.exe scripts/check_expertium_catch_up.py [--output artifacts/expertium-catch-up/check-cpu.json]
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("GPU_BATCH1_DATA_PATH", r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"))
os.environ["GPU_BATCH1_DATA_PATH"] = str(DATA)
TEACHER = Path(r"C:\Users\jonat\workspace\anki\qt\aqt\rwkv_inference\RWKV_trained_on_5000_10000.pth")


def perturb(model, seed, scale=0.05):
    import torch
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(scale * torch.randn(parameter.shape, generator=generator))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/expertium-catch-up/check-cpu.json"))
    parser.add_argument("--user", type=int, default=7)
    args = parser.parse_args()
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:  # noqa: BLE001 - priority is a courtesy only
        pass
    import torch
    torch.set_num_threads(2)
    from argparse import Namespace
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare
    from rwkv.train_rwkv import get_optimizer
    from scripts.gpu_batch1 import make_samples
    from expertium import variants
    from expertium.architecture import DEFAULT_ANKI_RWKV_CONFIG
    from expertium.model.srs_model import SrsRWKV
    from expertium.model.srs_model_rnn import SrsRWKVRnn
    from scripts import expertium_options as xo
    started = time.perf_counter()
    results = {}

    def record(name, value, ok):
        results[name] = {"value": value, "ok": bool(ok)}
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {value}", flush=True)

    frame = get_rwkv_data(DATA, args.user)
    sample = make_samples(frame.iloc[:300], args.user, 128)[0]
    batch = prepare([sample], seed=2025)

    # 1. architecture
    torch.manual_seed(1)
    base = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)
    torch.manual_seed(1)
    same = variants.build("iter36")
    sb, sv = base.state_dict(), same.state_dict()
    record("iter36 variant initial weights bitwise", list(sb) == list(sv) and all(torch.equal(sb[k], sv[k]) for k in sb),
           list(sb) == list(sv) and all(torch.equal(sb[k], sv[k]) for k in sb))
    base.train()
    same.train()
    torch.manual_seed(5)
    loss_base = base.get_loss(batch)
    loss_base.average_loss.backward()
    torch.manual_seed(5)
    loss_same = same.get_loss(batch)
    loss_same.average_loss.backward()
    grads = dict(base.named_parameters())
    equal_grads = all((p.grad is None and grads[n].grad is None) or torch.equal(p.grad, grads[n].grad)
                      for n, p in same.named_parameters())
    record("iter36 variant training loss bitwise", loss_base.average_loss.item(),
           torch.equal(loss_base.average_loss, loss_same.average_loss))
    record("iter36 variant gradients bitwise", equal_grads, equal_grads)
    for name, expected in (("iter36", variants.ITER36_PARAMETERS), ("w80", variants.W80_PARAMETERS),
                           ("w80-sequential", variants.W80_PARAMETERS)):
        model = variants.build(name)
        count = sum(p.numel() for p in model.parameters())
        record(f"{name} parameters", count, count == expected)
        recurrent = variants.build(name, rnn=True)
        recurrent.load_state_dict(model.state_dict(), strict=True)
        record(f"{name} strict recurrent load", True, True)
    wide = variants.build("w80")
    perturb(wide, 3)
    sequential = variants.build("w80-sequential")
    sequential.load_state_dict(wide.state_dict(), strict=True)
    wide.eval()
    sequential.eval()
    with torch.no_grad():
        reference = sequential.get_loss(batch)
        steps = wide.layer_steps
        wide.layer_steps = [(i, layer) for i in range(5) for layer in range(wide.stream_depths[i])]
        stream_major = wide.get_loss(batch)
        wide.layer_steps = steps
        interleaved = wide.get_loss(batch)
    gap = max((reference.p_curve - stream_major.p_curve).abs().max().item(),
              (reference.p_imm - stream_major.p_imm).abs().max().item())
    record("w80 canonical gather/scatter, stream-major == sequential (max abs)", gap, gap == 0.0)
    moved = (reference.p_curve - interleaved.p_curve).abs().max().item()
    record("w80 interleave changes predictions (max abs)", moved, moved > 1e-4)
    wide.train()
    torch.manual_seed(0)
    stats = wide.get_loss(batch)
    stats.average_loss.backward()
    record("w80 training loss finite and GRU head gets gradients", stats.average_loss.item(),
           torch.isfinite(stats.average_loss).item() and wide.gru_s_weight.grad.abs().sum().item() > 0)
    labelled = batch.labels[0, :, 4] > 0
    features, label_times = batch.start.float(), batch.labels[0, :, 0]
    for name in ("iter36", "w80", "w80-sequential"):
        model = variants.build(name)
        perturb(model, 11, 0.01)
        model.eval()
        with torch.no_grad():
            parallel = model.get_loss(batch)
        recurrent = variants.build(name, rnn=True)
        recurrent.load_state_dict(model.state_dict(), strict=True)
        recurrent.eval()
        curve, instant = recurrent.predict_rows(features, sample.ids, sample.skips, label_times)
        gap = max((curve - parallel.p_curve[0])[labelled].abs().max().item(),
                  (instant - parallel.p_imm[0])[labelled].abs().max().item())
        record(f"{name} recurrent vs parallel predictions (max abs, weights +N(0,0.01))", gap, gap < 1e-4)
    model = variants.build("iter36")
    perturb(model, 11, 0.01)
    model.eval()
    with torch.no_grad():
        parallel = model.get_loss(batch)
    vendored = SrsRWKVRnn(DEFAULT_ANKI_RWKV_CONFIG)
    vendored.load_state_dict(model.state_dict(), strict=True)
    vendored.eval()
    states, curve, instant = {}, [], []
    with torch.inference_mode():
        for row in range(features.size(0)):
            keys = [int(sample.ids[s][row]) for s in ("card_id", "note_id", "deck_id", "preset_id")]
            current = [states.get(("card_id", keys[0])), states.get(("note_id", keys[1])),
                       states.get(("deck_id", keys[2])), states.get(("preset_id", keys[3])), states.get("user")]
            ahead, w, rating, *nxt = vendored.review(features[row:row + 1], *current)
            if not bool(sample.skips[row]):
                for stream, key, state in zip(("card_id", "note_id", "deck_id", "preset_id"), keys, nxt[:4]):
                    states[(stream, key)] = state
                states["user"] = nxt[4]
            t = label_times[row].view(1, 1)
            raw = vendored.forgetting_curve(w, t)
            curve.append(torch.sigmoid(torch.log(raw / (1 - raw)) + vendored.interp(ahead, t)).view(()))
            instant.append(torch.clamp(1 - torch.softmax(rating, -1)[0, 0], 1e-5, 1 - 1e-5))
    gap = max((torch.stack(curve) - parallel.p_curve[0])[labelled].abs().max().item(),
              (torch.stack(instant) - parallel.p_imm[0])[labelled].abs().max().item())
    record("vendored SrsRWKVRnn (iter36) vs parallel predictions (max abs)", gap, gap < 1e-4)

    from rwkv.model import nvrtc
    source = (ROOT / "rwkv/model/csrc/cuda/rwkv7_cuda.cu").read_text()
    kernels = source[source.index("namespace rwkv {"):source.index("template <int CHUNK_LEN=32, typename F>\nstd::tuple")]
    rewritten = nvrtc.kernel_source(torch.float32, 16)
    record("NVRTC K=32 source untouched; K=16 rewrite applies", True,
           nvrtc.head_source(kernels, 32) is kernels and "const int K = 16;" in rewritten and "K = 32" not in rewritten)

    # 2. lr_schedule
    never = xo.LearningRate({"decay_start": 1.0}, 1000)
    same_lr = all(xo.default_learning_rate(u, o) == 7e-4 * min(1.0, (u + 1 + o) / 5) == never(u, o, r)
                  for u in range(0, 40) for o in (0, 13789) for r in (0, 500, 1000))
    record("default and never-decaying lr equal the accepted expression", same_lr, same_lr)
    decay = xo.LearningRate({"decay_start": 0.75, "shape": "one_minus_sin"}, 1000)
    values = [decay(100, 0, r) for r in (0, 750, 875, 1000)]
    record("decay 0.75..1.0 one_minus_sin at reviews 0/750/875/1000", values,
           values[0] == values[1] == 7e-4 and abs(values[2] - 7e-4 * (1 - 0.7071067811865476)) < 1e-12
           and abs(values[3]) < 1e-12)

    # 3. optimizer
    from scripts.muon_optimizer import expertium_iter53, get_hybrid_optimizer
    reference_groups = get_optimizer(Namespace(PEAK_LR=7e-4), base).param_groups
    f82 = get_hybrid_optimizer(base)
    record("F82 projections grouping unchanged",
           [len(g["params"]) for g in f82.muon.param_groups], [len(g["params"]) for g in f82.muon.param_groups]
           == [len(reference_groups[0]["params"]), len(reference_groups[1]["params"])])
    wide = variants.build("w80")
    hybrid = expertium_iter53(wide)
    on_muon = sum(p.numel() for g in hybrid.muon.param_groups for p in g["params"])
    lora = sum(p.numel() for p in hybrid.muon.param_groups[-1]["params"])
    total = sum(p.numel() for p in wide.parameters())
    record("w80 expertium-iter53: Muon / LoRA-group / AdamW parameters", [on_muon, lora, total - on_muon],
           lora > 0 and hybrid.muon.param_groups[-1]["weight_decay"] == 0.0 and on_muon < total)
    wide.train()
    torch.manual_seed(0)
    wide.get_loss(batch).average_loss.backward()
    for group in hybrid.param_groups:
        group["lr"] = 7e-4
    before = [p.detach().clone() for p in wide.parameters()]
    hybrid.step()
    finite = all(torch.isfinite(p).all() for p in wide.parameters())
    changed = sum(not torch.equal(a, p) for a, p in zip(before, wide.parameters()))
    record("w80 expertium-iter53 step finite, lr restored, parameters changed", changed,
           finite and changed > 0 and all(g["lr"] == 7e-4 for g in hybrid.param_groups))

    # 4. end_to_start_intervals
    os.environ.pop(xo.E2S_ENV, None)
    plain = get_rwkv_data(DATA, args.user)
    record("get_rwkv_data unchanged without the switch", True, plain.equals(frame))
    os.environ[xo.E2S_ENV] = "1"
    try:
        corrected = get_rwkv_data(DATA, args.user)
    finally:
        os.environ.pop(xo.E2S_ENV, None)
    import numpy as np
    raw = __import__("pandas").read_parquet(DATA / "revlogs" / f"user_id={args.user}")
    expected = np.where(raw["elapsed_seconds"] == -1, -1,
                        np.floor(np.maximum(raw["elapsed_seconds"] - raw["duration"] / 1000.0, 0.0)))
    ok = np.array_equal(corrected["elapsed_seconds"].to_numpy(), expected.astype(np.int64)) and \
        np.array_equal(corrected["elapsed_days"].to_numpy(), plain["elapsed_days"].to_numpy())
    record("end_to_start: elapsed_seconds - duration, -1 kept, elapsed_days unchanged",
           float((corrected["elapsed_seconds"] != plain["elapsed_seconds"]).mean()), ok)

    # 5. distillation
    model = variants.build("iter36")
    model.eval()
    with torch.no_grad():
        plain_stats = model.get_loss(batch)
        teacher = (torch.rand(plain_stats.p_curve.shape), torch.softmax(torch.randn(*plain_stats.p_curve.shape, 4), -1))
        zero = model.get_loss(batch, kd_mix=(teacher[0], teacher[1], 0.0))
        full = model.get_loss(batch, kd_mix=(teacher[0], teacher[1], 1.0))
    gap = (plain_stats.average_loss - zero.average_loss).abs().item()
    record("distillation alpha 0 == plain loss (abs diff)", gap, gap < 1e-6)
    record("distillation alpha 1 changes the loss", full.average_loss.item(), abs(full.average_loss.item() - plain_stats.average_loss.item()) > 1e-4)
    if TEACHER.exists():
        load = xo.Teacher({"teacher": str(TEACHER), "teacher_dtype": "float32"}, torch.device("cpu"), torch.float32)
        targets = load.targets(batch)
        record("fork 2.76M teacher loads strictly and predicts in (0, 1)", load.parameters,
               load.parameters == 2_762_884 and bool(((targets[0] > 0) & (targets[0] < 1)).all()))
    results["seconds"] = time.perf_counter() - started
    failed = [name for name, value in results.items() if isinstance(value, dict) and not value["ok"]]
    results["failed"] = failed
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"{len(failed)} failed; {results['seconds']:.1f} s; {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
