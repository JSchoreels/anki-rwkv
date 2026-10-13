"""F05 CUDA Graphs on the R01 path (opt-in, off by default).

Current contents: the CPU-only shape audit (`audit`), which prepares the exact
167-update sequence of an R01 protocol on the CPU and measures how often exact
shapes repeat and what bucketed padding would cost. The graph runtime is added
below the audit once the audit has chosen between exact-shape replay and buckets.
"""
from argparse import ArgumentParser
from collections import Counter
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def shape_keys(batch):
    """Return (hierarchy key, full key) for one CPU PreparedBatch.

    The hierarchy key is what `DynamicGraphs` used (sample layout, split lengths
    and index tensor shapes). The full key also contains the packed feature
    tensors of `gpu_packed_features`, whose table sizes depend on the data.
    """
    from scripts.gpu_batch1 import batch_tensors as base
    hierarchy = (int(batch.num_data), tuple(tuple(int(n) for n in g) for g in batch.sub_gather_lens),
                 tuple((tuple(x.shape), str(x.dtype)) for x in base(batch)))
    packed = getattr(batch, "packed", None)
    extra = tuple((name, tuple(v.shape)) for name, v in sorted(packed.items())) if packed else ()
    return hierarchy, (hierarchy, extra)


def second_use_hits(keys, limit=None):
    """Replays when a key is captured on its second sighting (the DynamicGraphs rule).

    `limit` bounds live graphs (LRU); None means unbounded. Returns (hits, captures)."""
    from collections import OrderedDict
    seen, cache, hits, captures = Counter(), OrderedDict(), 0, 0
    for key in keys:
        if key in cache:
            hits += 1
            cache.move_to_end(key)
            continue
        seen[key] += 1
        if seen[key] >= 2:
            captures += 1
            cache[key] = True
            if limit is not None and len(cache) > limit:
                cache.popitem(last=False)
    return hits, captures


def audit(args):
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from scripts.r01_workflow import configure_runtime
    configure_runtime(ROOT, 1)
    cohort_protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    handoff = json.loads(Path(args.handoff).read_text(encoding="utf-8"))
    options = dict(handoff["accepted_options"])
    os.environ["GPU_BATCH1_DATA_PATH"] = str(ROOT / cohort_protocol["dataset_path"])
    import torch
    torch.set_num_threads(1)
    from rwkv.data_processing import get_rwkv_data
    from scripts.gpu_batch1 import split_frame
    from scripts.gpu_preprocessing_stage import prepare_job
    from scripts.gpu_shape_buckets import BucketPolicy, bucket_batch
    options.pop("packed_ipc", None)  # keep tensors unpacked so shapes are directly readable
    span = options.get("batch_chunks", 512) * 128
    policies = {"default_q8": BucketPolicy(),
                "q32": BucketPolicy(count_quantum=32),
                "q128_ratio2": BucketPolicy(count_quantum=128, maximum_work_ratio=2.0),
                "pow2_ratio2": BucketPolicy(count_quantum=None, maximum_work_ratio=2.0)}
    rows = cohort_protocol["rows"]
    records, started = [], time.perf_counter()
    if args.limit_users:
        rows = rows[:args.limit_users]
    for row in rows:
        frame = split_frame(get_rwkv_data(Path(os.environ["GPU_BATCH1_DATA_PATH"]), row["user_id"]), row, "train")
        length = row["train_range"][1] - row["train_range"][0]
        for offset in range(0, length, span):
            end = min(length, offset + span)
            seed = cohort_protocol["seeds"]["training"] if isinstance(cohort_protocol.get("seeds"), dict) and "training" in cohort_protocol["seeds"] else 2025
            _, item = row, prepare_job(row, offset, end, seed + row["user_id"] + offset, options, frame.iloc[offset:end])
            batch = item.batch
            hierarchy, full = shape_keys(batch)
            rec = {"user_id": row["user_id"], "offset": offset, "reviews": int(batch.num_data),
                   "labels_shape": list(batch.labels.shape),
                   "recurrent_positions": sum(int(t.numel()) for g in batch.sub_gather for t in g),
                   "hierarchy_key": repr(hierarchy), "full_key": repr(full),
                   "split_lens": [[int(n) for n in g] for g in batch.sub_gather_lens],
                   "split_counts": [[int(t.numel() // n) for t, n in zip(g, lens)] for g, lens in zip(batch.sub_gather, batch.sub_gather_lens)],
                   "buckets": {}}
            for name, policy in policies.items():
                result = bucket_batch(batch, policy)
                metrics = result.metrics
                bucket = {"admitted": result.admitted, "reason": result.reason}
                if result.admitted:
                    from scripts.gpu_batch1 import batch_tensors as base
                    bucket["key"] = repr((int(result.batch.num_data), tuple(tuple(int(n) for n in g) for g in result.batch.sub_gather_lens),
                                          tuple((tuple(x.shape), str(x.dtype)) for x in base(result.batch))))
                    bucket["recurrent_positions"] = sum(int(t.numel()) for g in result.batch.sub_gather for t in g)
                    bucket["split_lens"] = [[int(n) for n in g] for g in result.batch.sub_gather_lens]
                    bucket["split_counts"] = [[int(t.numel() // n) for t, n in zip(g, lens)]
                                              for g, lens in zip(result.batch.sub_gather, result.batch.sub_gather_lens)]
                bucket["work_ratios"] = metrics.get("work_ratios")
                rec["buckets"][name] = bucket
            records.append(rec)
        print(f"[audit] user {row['user_id']}; {len(records)} updates; {time.perf_counter() - started:.0f}s", flush=True)
    summary = {"updates": len(records), "options": options, "span": span,
               "elapsed_seconds": time.perf_counter() - started}
    for label in ("hierarchy_key", "full_key"):
        keys = [r[label] for r in records]
        summary[label] = {"distinct": len(set(keys)), "repeated_updates": sum(1 for k in keys if Counter(keys)[k] > 1),
                          "second_use_hits_unbounded": second_use_hits(keys)[0],
                          "second_use_hits_limit1": second_use_hits(keys, 1)[0]}
    summary["policies"] = {}
    for name in policies:
        admitted = [r for r in records if r["buckets"][name]["admitted"]]
        keys = [r["buckets"][name]["key"] for r in admitted]
        orig = sum(r["recurrent_positions"] for r in admitted)
        padded = sum(r["buckets"][name]["recurrent_positions"] for r in admitted)
        seq = [r["buckets"][name]["key"] if r["buckets"][name]["admitted"] else ("eager", i) for i, r in enumerate(records)]
        summary["policies"][name] = {
            "admitted": len(admitted), "eager": len(records) - len(admitted),
            "reasons": dict(Counter(r["buckets"][name]["reason"] for r in records if not r["buckets"][name]["admitted"])),
            "distinct_keys": len(set(keys)),
            "second_use_hits_unbounded": second_use_hits(seq)[0],
            "second_use_hits_limit4": second_use_hits(seq, 4)[0],
            "recurrent_padding_overhead_percent": (100 * (padded / orig - 1)) if orig else None}
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "result.json").write_text(json.dumps({"summary": summary, "records": records}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary["hierarchy_key"]), json.dumps(summary["full_key"]))
    print(json.dumps(summary["policies"], indent=1)[:3000])


def tail(args):
    """Bounded microbenchmark: host and device time of the shape-independent update tail (the only part a
    graph could replay across all 167 updates): child downcast, grad transfer, clipping and AdamW."""
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from scripts.r01_workflow import configure_runtime
    configure_runtime(ROOT, 1)
    import statistics
    import torch
    from argparse import Namespace
    from scripts.benchmark_gpu_trial import benchmark_models, fresh_checkpoint
    from scripts.experiment_provenance import process_provenance
    from scripts.gpu_r01_update import MasterChildSync
    from rwkv.train_rwkv import get_optimizer
    device = torch.device("cuda:0")
    master, base, dtype = benchmark_models({"MODEL": "expertium", "DTYPE": "bfloat16", "SEED": 2025, "WKV_BACKEND": "nvrtc"}, device)
    master.load_state_dict(fresh_checkpoint("expertium", 2025)["model"])
    base.copy_downcast_(master, dtype)
    optimizer = get_optimizer(Namespace(PEAK_LR=7e-4), master)
    grouped = MasterChildSync(master, base)
    parameters = sum(p.numel() for p in master.parameters())
    host, gpu = [], []
    for step in range(args.warmup + args.repeats):
        for p in base.parameters():
            p.grad = torch.full_like(p, 1e-3)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        t0 = time.perf_counter()
        grouped.downcast()
        grouped.transfer_grads()
        torch.nn.utils.clip_grad_norm_(master.parameters(), 1.0, error_if_nonfinite=False)
        for group in optimizer.param_groups:
            group["lr"] = 7e-4 * min(1.0, (step + 1) / 5)
        optimizer.step()
        t1 = time.perf_counter()
        e1.record()
        torch.cuda.synchronize()
        if step >= args.warmup:
            host.append((t1 - t0) * 1000)
            gpu.append(e0.elapsed_time(e1))
    result = {"provenance": process_provenance(), "parameters": parameters, "parameter_tensors": len(list(master.parameters())),
              "warmup": args.warmup, "repeats": args.repeats,
              "tail_host_ms": {"median": statistics.median(host), "min": min(host), "max": max(host)},
              "tail_gpu_ms": {"median": statistics.median(gpu), "min": min(gpu), "max": max(gpu)},
              "scope": "accepted-bundle grouped sync + clip + AdamW on the real model, synthetic constant gradients; "
                       "upper bound of what graphing the shape-independent tail could save per update"}
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result["tail_host_ms"]), json.dumps(result["tail_gpu_ms"]))


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tail", help="GPU microbenchmark of the shape-independent update tail (needs the gpu claim)")
    t.add_argument("--output", required=True)
    t.add_argument("--warmup", type=int, default=10)
    t.add_argument("--repeats", type=int, default=50)
    a = sub.add_parser("audit", help="CPU-only exact-shape repeat and bucket padding audit")
    a.add_argument("--protocol", required=True)
    a.add_argument("--handoff", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--limit-users", type=int, default=0)
    args = parser.parse_args()
    if args.command == "audit":
        audit(args)
    else:
        tail(args)


if __name__ == "__main__":
    main()
