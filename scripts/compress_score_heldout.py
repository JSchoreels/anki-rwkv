"""Score weight-quantized variants of 2.76M RWKV checkpoints on held-out benchmark users (GPU, full-history protocol).

Users: validation users of the 8k/2k run (artifacts/full-run/run-8k-20261013/run.json in the main checkout) outside
dataset users 5000-10000, which the fork's checkpoint was trained on; a seeded sample of --users of them.
Protocol: each user's whole history as one sequence (upstream benchmark test-DB layout and the fork's replay), fp32
weights after fake quantization, NVRTC WKV kernel. Metrics: upstream get_stats on the run's label DB (FSRS-6
benchmark test reviews), curve (ahead) and instant heads. Resumable: one JSONL row per (user, label).

usage: compress_score_heldout.py --weights fork=<path> [--weights qat=<path>] --variants fp32,w8-pc,w4-g32 --users 300
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts import compress_common as cc  # noqa: E402
from scripts import compress_quant as cq  # noqa: E402


class UserBatches(torch.utils.data.Dataset):
    def __init__(self, users, data, counts):
        self.users, self.data, self.counts = users, data, counts

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        started = time.perf_counter()
        try:
            batch = cc.full_history_batch(self.data, user)
        except Exception as error:  # recorded, not fatal
            return user, None, repr(error), time.perf_counter() - started
        return user, batch, None, time.perf_counter() - started


def first(items):
    return items[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", action="append", required=True, help="tag=path (repeatable)")
    parser.add_argument("--variants", required=True, help="comma list of compress_quant.WEIGHT_SCHEMES keys")
    parser.add_argument("--only", action="append", default=[], help="tag:variant pairs to score instead of the full grid")
    parser.add_argument("--users", type=int, default=300)
    parser.add_argument("--user-seed", type=int, default=0)
    parser.add_argument("--max-reviews", type=int, default=0, help="skip users with more reviews (0 = no cap)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="artifacts/compression/heldout-scores.jsonl")
    parser.add_argument("--time-limit", type=float, default=0, help="stop starting new users after N seconds")
    args = parser.parse_args()

    from rwkv.model.nvrtc import register
    register()
    from scripts.compress_metrics import metrics, read_labels
    from scripts.experiment_provenance import process_provenance
    from scripts.full_run_metrics import evaluate_with_predictions
    import lmdb

    device = torch.device("cuda")
    run = cc.run_info()
    data = cc.data_path(run)
    counts = run["review_counts"]
    users = cc.heldout_users(args.users, args.user_seed, run)
    if args.max_reviews:
        users = [u for u in users if counts[str(u)] <= args.max_reviews]
    sources = dict(item.split("=", 1) for item in args.weights)
    grid = [(tag, v) for tag in sources for v in args.variants.split(",")]
    if args.only:
        grid = [tuple(item.split(":", 1)) for item in args.only]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for row in map(json.loads, out.read_text().splitlines()):
            done.setdefault(row["user_id"], set()).add(row["label"])
    todo = [u for u in users if not {f"{t}/{v}" for t, v in grid} <= done.get(u, set())]
    print(f"[heldout] {len(users)} users, {len(todo)} to score, grid {grid}", flush=True)
    states = {}
    for tag, path in sources.items():
        base = torch.load(path, map_location="cpu", weights_only=True)
        base = {k: v.float() for k, v in base.items()}
        names = cq.linear_weight_names(base)
        for t, variant in grid:
            if t == tag:
                bits, group = cq.WEIGHT_SCHEMES[variant]
                states[(tag, variant)] = (cq.quantize_state_dict(base, names, bits, group),
                                          cq.weight_bytes(base, names, bits, group)["total"], path)
    model = None
    if any(not tag.startswith("expertium") for tag, _ in grid):
        model = cq.load_model(state=next(v[0] for k, v in states.items() if not k[0].startswith("expertium")),
                              reference_wkv=False).to(device)
    expertium = None
    if any(tag.startswith("expertium") for tag, _ in grid):
        # Expertium iter36 (the run's small model) as an equal-protocol reference, fp32 only.
        from expertium.architecture import DEFAULT_ANKI_RWKV_CONFIG as ECONFIG
        from expertium.model.srs_model import SrsRWKV as ESrsRWKV
        expertium = ESrsRWKV(ECONFIG).float()
        expertium.load_state_dict(next(v[0] for k, v in states.items() if k[0].startswith("expertium")))
        expertium = expertium.to(device).eval()
    env = lmdb.open(str(cc.RUN / "label-db"), readonly=True, lock=False)
    prov = process_provenance()
    loader = torch.utils.data.DataLoader(UserBatches(todo, data, counts), batch_size=1, num_workers=args.workers,
                                         collate_fn=first, prefetch_factor=1 if args.workers else None)
    started = time.perf_counter()
    reviews_scored = 0
    for index, (user, batch, error, prep_seconds) in enumerate(loader):
        if args.time_limit and time.perf_counter() - started > args.time_limit:
            print(f"[heldout] time limit reached after {index} users", flush=True)
            break
        with env.begin() as txn:
            labels = read_labels(txn, user)
        rows = []
        if batch is None or labels is None:
            rows.append({"user_id": user, "label": "error", "error": error or "no labels"})
        else:
            gpu_batch = batch.to(device)
            for tag, variant in grid:
                label = f"{tag}/{variant}"
                if label in done.get(user, set()):
                    continue
                state, nbytes, path = states[(tag, variant)]
                current = expertium if tag.startswith("expertium") else model
                if current is model:
                    model.load_state_dict(state)
                t0 = time.perf_counter()
                try:
                    with torch.inference_mode():
                        values, predictions = evaluate_with_predictions(current, [{"batch": gpu_batch}])
                except torch.OutOfMemoryError as oom:
                    torch.cuda.empty_cache()
                    rows.append({"user_id": user, "label": label, "error": f"OOM: {oom}"[:300]})
                    continue
                torch.cuda.synchronize()
                seconds = time.perf_counter() - t0
                rows.append({"user_id": user, "label": label, "weights": tag, "variant": variant,
                             "checkpoint": path, "weight_bytes": nbytes, "reviews": counts[str(user)],
                             "prepare_seconds": round(prep_seconds, 2), "model_seconds": round(seconds, 3),
                             **values, "benchmark": metrics(user, labels, predictions)})
                reviews_scored += counts[str(user)]
            del gpu_batch
        with out.open("a", encoding="utf-8") as f:
            for row in rows:
                row["provenance_commit"] = prov["git_commit"]
                row["provenance_dirty"] = prov["git_dirty"]
                f.write(json.dumps(row) + "\n")
        elapsed = time.perf_counter() - started
        print(f"[heldout] {index + 1}/{len(todo)} user {user} ({counts[str(user)]} reviews) prep {prep_seconds:.1f}s "
              f"| {elapsed:.0f}s elapsed, {reviews_scored / max(elapsed, 1e-9):,.0f} review-evals/s", flush=True)
    env.close()


if __name__ == "__main__":
    main()
