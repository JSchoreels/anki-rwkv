#!/usr/bin/env python
"""Tiny cool-down with session inputs (SM183): a few real training updates of a model (--release = the folder whose protocol rows
and options are used, --arch; copy of sm181_updates.py, --optimizer none = a fresh optimizer as in a pass-1 cool-down) on the CPU, inside one code
tree (--tree), for the CPU proofs of sm179_check.py. The real trainer path of the screen (release options: strided-lane
carry, show-time inputs, replay KD targets, Muon state), as the state-energy penalty screen's (SM171) proof P4.

  SM179_INPUTS=<mode or unset> python sm179_updates.py --tree <code tree> --master <pth> --optimizer <pth>
      --users 4745,2308 --chunk 256 --steps 3 --out <json>

Per update: loss (float hex), sha256 of the batch's model input (start) and of every gradient, the input width, and
the absolute sum of the gradient on the extra input columns (24 .. 24 + n - 1 of features2card.0.weight).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--tree", required=True)
p.add_argument("--master", required=True)
p.add_argument("--optimizer", required=True)
p.add_argument("--users", default="4745,2308")
p.add_argument("--chunk", type=int, default=256)
p.add_argument("--steps", type=int, default=3)
p.add_argument("--lr", type=float, default=7e-5)
p.add_argument("--out", required=True)
p.add_argument("--release", required=True)
p.add_argument("--arch", required=True)
args = p.parse_args()

TREE = Path(args.tree).resolve()
sys.path.insert(0, str(TREE))
os.chdir(TREE)
# read by scripts/gpu_batch1.py at import time
os.environ["GPU_BATCH1_DATA_PATH"] = r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"
import torch  # noqa: E402

torch.set_num_threads(4)
RELEASE = None   # set from --release


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()[:16]


def main():
    from concurrent.futures import ThreadPoolExecutor
    from expertium.variants import build
    from scripts.expertium_options import CatchUp
    from scripts.gpu_r01_prefetch import training_batches
    from rwkv.train_rwkv import CLIP
    try:
        from rwkv import extra_inputs
    except ImportError:   # the stock tree (cea0f931) has no SM179 module
        extra_inputs = type("none", (), {"DIMS": 0, "MODE": ""})
    RELEASE = Path(args.release)
    protocol = json.loads((RELEASE / "protocol.json").read_text())
    options = json.loads((RELEASE / "decay.json").read_text())["options"]
    options = {k: v for k, v in options.items() if k not in ("packed_ipc",)}
    options["chunk_reviews"] = args.chunk
    os.environ["GPU_BATCH1_DATA_PATH"] = protocol["dataset_path"]
    users = [int(u) for u in args.users.split(",")]
    rows = [next(r for r in protocol["rows"] if r["user_id"] == u) for u in users]
    with ThreadPoolExecutor(max_workers=1) as pool:
        items = list(training_batches(rows, args.chunk, options["training_seed"], pool, options, None, 2))
    model = build(args.arch).float()
    model.load_state_dict(torch.load(args.master, map_location="cpu", weights_only=True), strict=True)
    catch_up = CatchUp(options, {"lr_schedule": {"peak": args.lr, "warmup_updates": 1, "decay_start": 1.0,
                                                "decay_end": 1.0}}, rows)
    catch_up.load_teacher(torch.device("cpu"), torch.float32)
    optimizer = catch_up.optimizer(model)
    if args.optimizer != "none":
        optimizer.load_state_dict(torch.load(args.optimizer, map_location="cpu", weights_only=True))
    names = [n for n, _ in model.named_parameters()]
    n_extra = extra_inputs.DIMS
    steps, per_user, done = [], {}, 0
    for row, item in items:
        u = row["user_id"]
        per_user[u] = per_user.get(u, 0) + 1
        if per_user[u] > args.steps:
            continue
        batch = item.batch.to(torch.device("cpu"))
        model.train()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(item.seed)
        catch_up.carry.attach(item, batch, u)
        t0 = time.perf_counter()
        stats = catch_up.loss(model, batch, done, (u, item.offset, item.reviews, item.seed), batch.labels)
        stats.average_loss.backward()
        params = dict(model.named_parameters())
        h = hashlib.sha256()
        for n in names:
            if params[n].grad is not None:
                h.update(params[n].grad.detach().contiguous().numpy().tobytes())
        w = params["features2card.0.weight"]
        extra_grad = float(w.grad[:, 24:24 + n_extra].abs().sum()) if n_extra else 0.0
        start = batch.start
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        optimizer.step()
        done += item.reviews
        steps.append({"user": u, "offset": item.offset, "reviews": item.reviews,
                      "loss": float(stats.average_loss.detach()), "loss_hex": float(stats.average_loss.detach()).hex(),
                      "start_width": int(start.shape[-1]), "start_sha256": sha(start),
                      "extra_columns_abs_max": float(start[..., 24:24 + n_extra].abs().max()) if n_extra else 0.0,
                      "grad_sha256": h.hexdigest()[:16], "grad_norm": float(norm), "extra_grad_abs_sum": extra_grad,
                      "weights_finite": all(bool(torch.isfinite(q).all()) for q in model.parameters()),
                      "seconds": round(time.perf_counter() - t0, 2)})
    out = {"tree": str(TREE), "inputs": extra_inputs.MODE or "(unset)", "dims": n_extra, "master": args.master,
           "users": users, "chunk": args.chunk, "steps": steps}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"inputs": out["inputs"], "steps": len(steps), "first_loss": steps[0]["loss"] if steps else None}))


if __name__ == "__main__":
    main()
