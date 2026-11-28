#!/usr/bin/env python
"""AUC ceiling (SM182): a few real training updates of six-layer Tiny two-head (SM130) on the CPU, inside one code tree
(--tree), for the CPU proofs of sm182_check.py. The real trainer path of the screen (release options: strided-lane
carry, show-time inputs, replay KD targets, Muon state), as the session inputs screen's (SM179) sm179_updates.py.

  python sm182_updates.py --tree <code tree> [--pair-rank '{"weight": 1.0, "logloss": 0.0}'] --users 4745,2308
      --chunk 256 --steps 3 [--no-step] --out <json>

Per update: loss (float hex), the two pair terms (curve, instant) when the option is on, sha256 of the batch's model
input and of every gradient, the gradient norm before clipping, and whether the weights stay finite after the step.
--no-step: no optimizer step (every update is measured at the start weights; for the scale of the two loss terms).
--no-kd: without the distillation targets (the pair term uses hard labels only; the teacher on the CPU is large).
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--tree", required=True)
p.add_argument("--pair-rank", default="")
p.add_argument("--users", default="4745,2308")
p.add_argument("--chunk", type=int, default=256)
p.add_argument("--steps", type=int, default=3)
p.add_argument("--lr", type=float, default=7e-5)
p.add_argument("--no-step", action="store_true")
p.add_argument("--no-kd", action="store_true")
p.add_argument("--out", required=True)
args = p.parse_args()

TREE = Path(args.tree).resolve()
sys.path.insert(0, str(TREE))
os.chdir(TREE)
os.environ["GPU_BATCH1_DATA_PATH"] = r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"
import torch  # noqa: E402

torch.set_num_threads(4)
RELEASE = Path("C:/Users/jonat/workspace/wt/sm130/artifacts/small-model/sm130-th-dec2ep32k-p2")
CASE = RELEASE / "expertium" / "E89-F65e-F77a"


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()[:16]


def main():
    from concurrent.futures import ThreadPoolExecutor
    from expertium.variants import build
    from scripts.expertium_options import CatchUp
    from scripts.gpu_r01_prefetch import training_batches
    from rwkv.train_rwkv import CLIP
    protocol = json.loads((RELEASE / "protocol.json").read_text())
    options = json.loads((RELEASE / "decay.json").read_text())["options"]
    options = {k: v for k, v in options.items() if k not in ("packed_ipc",)}
    options["chunk_reviews"] = args.chunk
    if args.no_kd:   # the pair term never reads the teacher; without it the CPU stays far under 6 GB
        options.pop("distillation")
    spec = json.loads(args.pair_rank) if args.pair_rank else None
    if spec:
        options["pair_rank"] = spec
    os.environ["GPU_BATCH1_DATA_PATH"] = protocol["dataset_path"]
    users = [int(u) for u in args.users.split(",")]
    rows = [next(r for r in protocol["rows"] if r["user_id"] == u) for u in users]
    model = build("w64-c1-d2p1u1").float()
    model.load_state_dict(torch.load(CASE / "master.pth", map_location="cpu", weights_only=True), strict=True)
    applied = None
    if spec:
        from expertium.model import variant_model as vm
        applied = vm.apply_pair_rank(model, spec)
    catch_up = CatchUp(options, {"lr_schedule": {"peak": args.lr, "warmup_updates": 1, "decay_start": 1.0,
                                                "decay_end": 1.0}}, rows)
    catch_up.load_teacher(torch.device("cpu"), torch.float32)
    optimizer = catch_up.optimizer(model)
    optimizer.load_state_dict(torch.load(CASE / "optimizer.pth", map_location="cpu", weights_only=True))
    names = [n for n, _ in model.named_parameters()]
    steps, per_user, done = [], {}, 0
    with ThreadPoolExecutor(max_workers=1) as pool:
        for row, item in training_batches(rows, args.chunk, options["training_seed"], pool, options, None, 2):
            u = row["user_id"]
            per_user[u] = per_user.get(u, 0) + 1
            if per_user[u] > args.steps:
                if all(per_user.get(x, 0) > args.steps for x in users):
                    break
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
            grads_finite = True
            for n in names:
                if params[n].grad is not None:
                    h.update(params[n].grad.detach().contiguous().numpy().tobytes())
                    grads_finite = grads_finite and bool(torch.isfinite(params[n].grad).all())
            start = batch.start
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            if not args.no_step:
                for group in optimizer.param_groups:
                    group["lr"] = args.lr
                optimizer.step()
            done += item.reviews
            step = {"user": u, "offset": item.offset, "reviews": item.reviews,
                    "loss": float(stats.average_loss.detach()), "loss_hex": float(stats.average_loss.detach()).hex(),
                    "start_sha256": sha(start), "grad_sha256": h.hexdigest()[:16], "grad_norm": float(norm),
                    "grads_finite": grads_finite,
                    "weights_finite": all(bool(torch.isfinite(q).all()) for q in model.parameters()),
                    "seconds": round(time.perf_counter() - t0, 2)}
            if spec:
                c, i = vm.PAIR_RANK_LAST
                step.update(pair_curve=float(c), pair_instant=float(i),
                            pair_curve_hex=float(c).hex(), pair_instant_hex=float(i).hex(),
                            curve_rows=int(((1 - stats.is_query) * stats.has_label).sum()),
                            instant_rows=int((stats.is_query * stats.has_label).sum()))
                vm.PAIR_RANK_LAST[0] = vm.PAIR_RANK_LAST[1] = None
            steps.append(step)
    out = {"tree": str(TREE), "pair_rank": applied, "no_kd": args.no_kd, "users": users, "chunk": args.chunk, "no_step": args.no_step,
           "steps": steps, "all_finite": all(math.isfinite(s["loss"]) and s["grads_finite"] and s["weights_finite"]
                                             for s in steps)}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"pair_rank": applied, "steps": len(steps), "first_loss": steps[0]["loss"] if steps else None}))


if __name__ == "__main__":
    main()
