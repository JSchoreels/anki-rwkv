"""Kernel-level cost of deterministic CUDA algorithms in the R01 update; not a training run.

Profiles warm updates on fixed real windows with deterministic algorithms on and
off and lists the CUDA kernels whose summed device time differs most.
"""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def profile(model, deterministic, output):
    from scripts.benchmark_r01_components import ACCEPTED, windows
    data_path, selected = windows(model)
    os.environ["GPU_BATCH1_DATA_PATH"] = str(ROOT / data_path)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from torch.profiler import ProfilerActivity, profile as torch_profile
    from argparse import Namespace
    torch._C._set_graph_executor_optimize(False)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    from rwkv.model import nvrtc
    from rwkv.train_rwkv import CLIP, get_optimizer, transfer_child_grad_to_master
    from rwkv.data_processing import get_rwkv_data
    from scripts.benchmark_gpu_trial import benchmark_models, fresh_checkpoint
    from scripts.gpu_batch1 import split_frame
    from scripts.gpu_b01 import prepare_job, transfer_batches
    torch.cuda.set_per_process_memory_fraction(0.72, 0)
    nvrtc.register()
    nvrtc.set_checkpoint_length(32)
    device = torch.device("cuda:0")
    span = (128 if model == "baseline" else 256) * 128
    prepared = []
    for row, length in selected[:2]:
        frame = split_frame(get_rwkv_data(Path(os.environ["GPU_BATCH1_DATA_PATH"]), row["user_id"]), row, "train")
        for offset in range(0, min(length, 2 * span), span):
            end = min(length, offset + span)
            prepared.append((row, prepare_job(row, offset, end, 2025 + row["user_id"] + offset, ACCEPTED[model], frame.iloc[offset:end])))
    master, base, dtype = benchmark_models({"MODEL": model, "DTYPE": "bfloat16", "SEED": 2025, "WKV_BACKEND": "nvrtc"}, device)
    master.load_state_dict(fresh_checkpoint(model, 2025)["model"])
    base.copy_downcast_(master, dtype)
    batches = [(item, batch) for _, item, batch in transfer_batches(iter(prepared), dtype, False)]
    optimizer = get_optimizer(Namespace(PEAK_LR=7e-4), master)

    def update(item, batch):
        torch.manual_seed(item.seed)
        optimizer.zero_grad(set_to_none=True)
        base.zero_grad(set_to_none=True)
        base.copy_downcast_(master, dtype)
        stats = base.get_loss(batch)
        stats.average_loss.backward()
        transfer_child_grad_to_master(master, base)
        torch.nn.utils.clip_grad_norm_(master.parameters(), CLIP)
        optimizer.step()
        return stats.average_loss.item()

    for item, batch in batches * 2:
        update(item, batch)
    torch.cuda.synchronize()
    with torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
        for item, batch in batches:
            update(item, batch)
        torch.cuda.synchronize()
    kernels = {}
    for event in prof.key_averages():
        if event.device_type.name == "CUDA" or getattr(event, "device_time_total", 0):
            kernels[event.key] = {"device_us": getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0)),
                                  "count": event.count}
    output.write_text(json.dumps({"model": model, "deterministic": deterministic, "updates": len(batches),
                                  "kernels": kernels}, indent=2) + "\n", encoding="utf-8")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="expertium")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", choices=("on", "off"))
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if args.child:
        profile(args.model, args.child == "on", output / f"{args.model}-deterministic-{args.child}.json")
        return
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Output must remain under workspace artifacts")
    output.mkdir(parents=True, exist_ok=True)
    for mode in ("on", "off"):
        subprocess.run([sys.executable, __file__, "--model", args.model, "--output", str(output), "--child", mode], check=True)
    runs = {mode: json.loads((output / f"{args.model}-deterministic-{mode}.json").read_text())["kernels"] for mode in ("on", "off")}
    names = set(runs["on"]) | set(runs["off"])
    rows = sorted(((runs["on"].get(n, {}).get("device_us", 0) - runs["off"].get(n, {}).get("device_us", 0), n) for n in names), reverse=True)
    total = {mode: sum(k["device_us"] for k in runs[mode].values()) for mode in runs}
    summary = {"model": args.model, "device_us_total": total,
               "largest_increases": [{"kernel": n[:160], "extra_device_us": d,
                                      "on": runs["on"].get(n), "off": runs["off"].get(n)} for d, n in rows[:15]]}
    (output / f"{args.model}-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1)[:4000])


if __name__ == "__main__":
    main()
