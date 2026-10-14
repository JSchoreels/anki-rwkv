"""Quantization-aware fine-tuning (QAT) of the fork's 2.76M RWKV with fake-quantized nn.Linear weights (GPU).

Start: the fork's RWKV_trained_on_5000_10000.pth. Data: training users of the 8k/2k run (main checkout run.json)
outside dataset users 5000-10000, so the run's validation users stay held out and the fork's own training users are
not revisited. Samples are 128-review windows (the R01 training regime; the NVRTC training kernel accepts at most 383
rows), cut from one random contiguous span of <= 8,192 reviews per user and pool; a batch packs windows from several
users up to --batch-reviews reviews. Upstream trained the fork model on segments of up to 32,768 reviews, so this
fine-tune is shorter-context than the original training (the fp32 fine-tune control measures that effect).

Training follows rwkv/train_rwkv.py: fp32 master weights, bfloat16 child for forward/backward, the upstream AdamW
parameter groups (eps 1e-18, betas 0.9/0.999, weight decay 0.01), gradient clip 0.5, loss = stats.average_loss.
QAT: before every step the child receives fake_quant(master) for every nn.Linear weight (symmetric abs-max scales
recomputed from the current master weights); child gradients are added to the master unchanged (straight-through
estimator). --bits 0 runs the same fine-tune without quantization (control).
LR: linear warmup over --warmup steps, then cosine decay to 0 at --time-budget seconds of training.

usage: compress_qat.py --bits 4 --group 32 --time-budget 2400 --out artifacts/compression/qat/w4g32-run1
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts import compress_common as cc  # noqa: E402
from scripts import compress_quant as cq  # noqa: E402

HEAD_TARGETS = ["head", "p_linear", "s_linear", "d_linear", "w_linear", "ahead_linear", "head_ahead_logit",
                "head_w", "head_s", "head_d", "head_p"]


def get_optimizer(model, lr):
    """Copy of rwkv/train_rwkv.get_optimizer (that module imports wandb at import time)."""
    groups = {"decay": [], "channel_mixer": [], "head": [], "encode": [], "other": []}
    for name, param in model.named_parameters():
        if "weight" in name and "lora" not in name and "scale" not in name and len(param.squeeze().shape) >= 2:
            if any(t in name for t in HEAD_TARGETS):
                groups["head"].append(param)
            elif "features2card" in name:
                groups["encode"].append(param)
            elif "channel_mixer" in name:
                groups["channel_mixer"].append(param)
            else:
                groups["decay"].append(param)
        else:
            groups["other"].append(param)
    decay = {"decay": 0.01, "channel_mixer": 0.01, "head": 0.01, "encode": 1e-2, "other": 0.0}
    return torch.optim.AdamW([{"params": groups[k], "weight_decay": decay[k], "lr": lr} for k in groups],
                             eps=1e-18, betas=(0.90, 0.999))


class SegmentBatches(torch.utils.data.IterableDataset):
    """Endless prepared training batches. Each worker loads a few random users at a time, cuts their histories into
    contiguous segments (random phase), shuffles them and packs batches of <= batch_reviews reviews."""

    def __init__(self, users, counts, data, segment, batch_reviews, users_per_pool, seed, span):
        self.span = span
        self.users, self.counts, self.data = users, counts, data
        self.segment, self.batch_reviews, self.users_per_pool, self.seed = segment, batch_reviews, users_per_pool, seed

    def __iter__(self):
        from rwkv.data_processing import create_sample, get_rwkv_data
        from rwkv.prepare_batch import prepare
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = random.Random(self.seed * 1000 + wid)
        weights = [min(self.counts[str(u)], self.span) for u in self.users]  # ~reviews actually used
        while True:
            pool = []
            reviews_in_pool = 0
            while reviews_in_pool < 3 * self.batch_reviews or len({p[0] for p in pool}) < self.users_per_pool:
                user = rng.choices(self.users, weights=weights)[0]
                frame = get_rwkv_data(self.data, user)
                n = len(frame)
                if n < 16:
                    continue
                # One random contiguous span of <= `span` reviews per user and pool, cut into `segment`-review
                # samples (keeps pools mixed); users weighted by the reviews used.
                start = rng.randrange(n - self.span + 1) if n > self.span else 0
                stop = min(n, start + self.span)
                segments = [(a, min(stop, a + self.segment)) for a in range(start, stop, self.segment)]
                segments = [(a, b) for a, b in segments if b - a >= 16]
                for s, e in segments:
                    sample = create_sample(user, frame.iloc[s:e], [], torch.float32, torch.device("cpu"))
                    pool.append((user, e - s, sample))
                    reviews_in_pool += e - s
                del frame
            rng.shuffle(pool)
            while pool:
                batch, total, users = [], 0, []
                while pool and (total + pool[-1][1] <= self.batch_reviews or not batch):
                    user, length, sample = pool.pop()
                    batch.append(sample)
                    users.append(user)
                    total += length
                yield {"batch": prepare(batch, seed=rng.randrange(2**31)), "reviews": total, "users": users}


def first(items):
    return items[0]


def copy_fake_quant(master, child, qnames, bits, group):
    master_params = dict(master.named_parameters())
    with torch.no_grad():
        for name, param in child.named_parameters():
            source = master_params[name]
            if bits and name in qnames:
                source = cq.fake_quant(source, bits, group)
            param.copy_(source.to(param.dtype))


def transfer_grads(master, child):
    master_params = dict(master.named_parameters())
    for name, param in child.named_parameters():
        if param.grad is None:
            continue
        target = master_params[name]
        if target.grad is None:
            target.grad = param.grad.float().clone()
        else:
            target.grad.add_(param.grad.float())
        param.grad = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=4, help="4 or 8; 0 = no quantization (fine-tune control)")
    parser.add_argument("--group", type=int, default=32, help="0 = per output channel")
    parser.add_argument("--init", default=str(cq.FORK_WEIGHTS))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--time-budget", type=float, default=2400, help="training seconds (cosine end)")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--segment", type=int, default=128,
                        help="reviews per sample; the NVRTC training kernel accepts <= 383 rows (reviews + queries)")
    parser.add_argument("--span", type=int, default=8192, help="contiguous reviews taken from one user per pool")
    parser.add_argument("--batch-reviews", type=int, default=32768)
    parser.add_argument("--users-per-pool", type=int, default=3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--save-every", type=float, default=900, help="checkpoint every N training seconds")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="cpu: code-path test only")
    args = parser.parse_args()

    cpu = args.device == "cpu"
    if cpu:
        torch.set_num_threads(2)
    else:
        from rwkv.model.nvrtc import register
        register()
    from scripts.experiment_provenance import process_provenance

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    prov = process_provenance()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run = cc.run_info()
    users = sorted(cc.outside_fork_range(run["train_users"]))
    counts = run["review_counts"]
    group = args.group or None
    init = torch.load(args.init, map_location="cpu", weights_only=True)
    master = cq.load_model(state={k: v.float() for k, v in init.items()}, reference_wkv=cpu).to(device)
    child = cq.load_model(state={k: v.float() for k, v in init.items()}, reference_wkv=cpu)
    child = child.selective_cast(torch.bfloat16).to(device)
    qnames = set(cq.linear_weight_names(master))
    optimizer = get_optimizer(master, args.lr)
    config = {**vars(args), "train_users": len(users), "train_user_reviews": sum(counts[str(u)] for u in users),
              "quantized_tensors": len(qnames), "provenance": prov, "dtype": "bfloat16 child / float32 master",
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    loader = torch.utils.data.DataLoader(
        SegmentBatches(users, counts, cc.data_path(run), args.segment, args.batch_reviews, args.users_per_pool,
                       args.seed, args.span),
        batch_size=None, num_workers=args.workers, prefetch_factor=2 if args.workers else None)
    log = (out / "steps.jsonl").open("a", encoding="utf-8")
    train_seconds = 0.0
    wait_seconds = 0.0
    reviews = 0
    labels = 0
    step = 0
    ooms = 0
    last_save = 0.0
    window = []
    started = time.perf_counter()
    iterator = iter(loader)
    print(f"[qat] bits={args.bits} group={group} users={len(users)} lr={args.lr} budget={args.time_budget}s", flush=True)
    while True:
        t_wait = time.perf_counter()
        item = next(iterator)
        wait = time.perf_counter() - t_wait
        if step > 0:
            wait_seconds += wait
        t0 = time.perf_counter()
        progress = min(1.0, train_seconds / args.time_budget)
        warm = min(1.0, (step + 1) / args.warmup)
        lr = args.lr * warm * 0.5 * (1 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g["lr"] = lr
        batch = item["batch"].to(device)
        batch.start = batch.start.to(torch.bfloat16)  # as scripts/benchmark_gpu_trial.update for the bf16 child
        copy_fake_quant(master, child, qnames, args.bits, group)
        child.train()
        try:
            stats = child.get_loss(batch)
            ok = stats is not None and bool(torch.isfinite(stats.average_loss))
            if ok:
                stats.average_loss.backward()
                transfer_grads(master, child)
                norm = torch.nn.utils.clip_grad_norm_(master.parameters(), 0.5)
                optimizer.step()
        except torch.OutOfMemoryError:
            stats, ok = None, False
            ooms += 1
            print(f"[qat] OOM at step {step + 1} ({item['reviews']} reviews); batch skipped ({ooms} so far)", flush=True)
            del batch
            torch.cuda.empty_cache()
            if ooms >= 5:
                raise
        optimizer.zero_grad(set_to_none=True)
        child.zero_grad(set_to_none=True)
        if not cpu:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if step > 0:  # the first step includes kernel compilation
            train_seconds += dt + wait
        step += 1
        reviews += item["reviews"]
        row = {"step": step, "lr": lr, "reviews": item["reviews"], "users": item["users"], "ok": ok,
               "seconds": round(dt, 3), "wait": round(wait, 3)}
        if ok:
            row.update(loss=stats.average_loss.item(), ahead=stats.ahead_avg.item(), imm=stats.imm_avg.item(),
                       ahead_n=int(stats.ahead_n), grad_norm=float(norm))
            labels += int(stats.ahead_n)
            window.append(row)
        log.write(json.dumps(row) + "\n")
        if step % 20 == 0:
            log.flush()
            recent = window[-20:]
            print(f"[qat] step {step} lr {lr:.2e} loss {sum(r['loss'] for r in recent) / max(1, len(recent)):.4f} "
                  f"ahead {sum(r['ahead'] for r in recent) / max(1, len(recent)):.4f} | train {train_seconds:.0f}s "
                  f"wait {wait_seconds:.0f}s | {reviews / max(train_seconds, 1e-9):,.0f} reviews/s | "
                  f"peak {(0 if cpu else torch.cuda.max_memory_reserved()) / 2**20:,.0f} MiB", flush=True)
        done = train_seconds >= args.time_budget or (args.max_steps and step >= args.max_steps)
        if done or train_seconds - last_save >= args.save_every:
            last_save = train_seconds
            name = "final.pth" if done else f"step-{step:05d}.pth"
            torch.save(master.state_dict(), out / name)
        if done:
            break
    log.close()
    summary = {"steps": step, "train_seconds": round(train_seconds, 1), "loader_wait_seconds": round(wait_seconds, 1),
               "wall_seconds": round(time.perf_counter() - started, 1), "reviews": reviews, "curve_labels": labels,
               "reviews_per_second": round(reviews / max(train_seconds, 1e-9), 1),
               "peak_reserved_mib": round((0 if cpu else torch.cuda.max_memory_reserved()) / 2**20), "oom_skipped_batches": ooms,
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[qat] done {summary}", flush=True)
    os._exit(0)  # skip worker teardown of the endless loader


if __name__ == "__main__":
    main()
