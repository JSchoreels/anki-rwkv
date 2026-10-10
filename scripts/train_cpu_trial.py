"""Train the unmodified baseline on bounded real-user history chunks on CPU."""

from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.data_processing import create_sample, get_rwkv_data
from rwkv.model.srs_model import SrsRWKV
from rwkv.prepare_batch import prepare
from rwkv.train_rwkv import CLIP, get_optimizer


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train-cpu-trial.toml")
    args = parser.parse_args()
    with open(args.config, "rb") as source:
        config = tomllib.load(source)
    assert 1 <= config["USER_START"] <= config["USER_END"]
    assert config["REVIEWS_PER_USER"] >= 0 and config["CHUNK_REVIEWS"] > 0
    assert config["EPOCHS"] > 0 and config["WARMUP_STEPS"] > 0
    assert config["CPU_THREADS"] > 0
    torch.set_num_threads(config["CPU_THREADS"])
    torch.manual_seed(config["SEED"])
    output = Path(config["OUTPUT_DIR"])
    checkpoint = Path(config["CHECKPOINT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    model = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG).float()
    optimizer = get_optimizer(Namespace(PEAK_LR=config["PEAK_LR"]), model)
    initial_head = model.w_linear.weight.detach().clone()
    summary = {"config": config, "torch": str(torch.__version__), "device": "cpu",
               "parameters": sum(p.numel() for p in model.parameters()),
               "users": {}, "steps": 0, "review_presentations": 0, "skipped_unlabelled_chunks": 0}
    start_time = time.monotonic()
    probe = None
    print(f"CPU training: {summary['parameters']:,} parameters; users "
          f"{config['USER_START']}-{config['USER_END']}", flush=True)
    with (output / "metrics.jsonl").open("w") as metrics:
        for epoch in range(config["EPOCHS"]):
            for user in range(config["USER_START"], config["USER_END"] + 1):
                frame = get_rwkv_data(Path(config["DATA_PATH"]), user)
                cap = config["REVIEWS_PER_USER"]
                used = min(len(frame), cap) if cap else len(frame)
                if not used:
                    raise ValueError(f"User {user} has no reviews")
                summary["users"].setdefault(str(user), {"available_reviews": len(frame),
                                              "selected_reviews_per_epoch": used, "optimizer_steps": 0})
                print(f"User {user}: using {used:,}/{len(frame):,} reviews", flush=True)
                for offset in range(0, used, config["CHUNK_REVIEWS"]):
                    section = frame.iloc[offset:min(used, offset + config["CHUNK_REVIEWS"])]
                    sample = create_sample(user, section, [], torch.float32, torch.device("cpu"))
                    batch = prepare([sample], seed=config["SEED"] + user + offset)
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    stats = model.get_loss(batch)
                    if stats is None or not torch.isfinite(stats.average_loss):
                        raise RuntimeError(f"Non-finite loss for user {user}, offset {offset}")
                    if stats.ahead_n + stats.imm_n == 0:
                        summary["skipped_unlabelled_chunks"] += 1
                        print(f"Skipping unlabelled chunk: user {user}, offset {offset}", flush=True)
                        del stats, batch, sample
                        continue
                    if probe is None:
                        probe = batch
                        model.eval()
                        with torch.no_grad():
                            summary["fixed_probe_loss_before"] = model.get_loss(probe).average_loss.item()
                        model.train()
                    stats.average_loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP, error_if_nonfinite=True)
                    lr = config["PEAK_LR"] * min(1.0, (summary["steps"] + 1) / config["WARMUP_STEPS"])
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.step()
                    summary["steps"] += 1
                    summary["users"][str(user)]["optimizer_steps"] += 1
                    summary["review_presentations"] += len(section)
                    row = {"step": summary["steps"], "epoch": epoch + 1, "user_id": user,
                           "review_start": offset + 1, "review_end": offset + len(section),
                           "loss": stats.average_loss.item(), "curve_loss": stats.ahead_avg.item(),
                           "instant_loss": stats.imm_avg.item(), "curve_labels": stats.ahead_n,
                           "instant_labels": stats.imm_n, "gradient_norm": norm.item(), "lr": lr,
                           "elapsed_seconds": time.monotonic() - start_time}
                    metrics.write(json.dumps(row) + "\n")
                    metrics.flush()
                    print(f"Step {row['step']}: loss={row['loss']:.5f}, "
                          f"curve={row['curve_loss']:.5f}, instant={row['instant_loss']:.5f}, "
                          f"elapsed={row['elapsed_seconds']:.1f}s", flush=True)
                    del stats, batch, sample
                del frame
    if any(item["optimizer_steps"] == 0 for item in summary["users"].values()):
        raise RuntimeError("A selected user contributed no supervised training chunks; increase REVIEWS_PER_USER")
    model.eval()
    with torch.no_grad():
        summary["fixed_probe_loss_after"] = model.get_loss(probe).average_loss.item()
    summary["head_max_weight_change"] = (model.w_linear.weight.detach() - initial_head).abs().max().item()
    if not summary["head_max_weight_change"] > 0:
        raise RuntimeError("Optimizer did not change the curve-head weights")
    model_path = checkpoint / "rwkv_cpu_trial.pth"
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), checkpoint / "rwkv_cpu_trial_optim.pth")
    summary["checkpoint"] = str(model_path)
    summary["elapsed_seconds"] = time.monotonic() - start_time
    summary["status"] = "complete"
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
