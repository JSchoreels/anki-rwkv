"""SM21 GPU check of the trainer's carried-state path (scripts/state_carry_train.py) with the run's kernels.

Second part of the SM21 parity ticket (after scripts/check_state_carry_kernels.py); never run outside the queue.
Setup as scripts/gpu_r01_training.py: the ctx2048 run's options (row_tiles_exact 4, exact shift / index / group norm,
skip_padding_sources, deterministic algorithms), the w64 e6 checkpoint, bf16 student copy of the fp32 master.

  P  empty table = stock: the first update of each user (16 x 2,048), train mode, bf16, the trainer's dropout seed:
     loss and every gradient, bitwise flag and max |diff| (relative to max |stock|);
  X  exact carry on the GPU: fp32 model, eval mode, one 2,048-review window per update, fp32 table: every row's rating
     probabilities vs the stock full-history forward on the GPU (max |dp|), and the deploy table (int8/fp16);
  T  cost: forward + backward per 16 x 2,048 update of a long user, stock vs carry (table filled by the previous
     update), median of repeats, and peak memory (informational; the experiment measures reviews/s itself).
Pass: P loss relative |diff| <= 1e-3 and gradients relative <= 5e-2 (bf16), X fp32 max |dp| <= 1e-3.

  .venv/Scripts/python.exe -X utf8 scripts/check_state_carry_gpu.py --output artifacts/small-model/check-state-carry-gpu.json
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = os.environ.get("GPU_BATCH1_DATA_PATH", r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
os.environ["GPU_BATCH1_DATA_PATH"] = DATA
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT / "artifacts/cache/cupy-cache"))
RUN = Path("C:/Users/jonat/workspace/wt/small-model/artifacts/accuracy/experiments/scale-w64kd-gate-cached-ctx2048")
CKPT = RUN / "run" / "epoch-06" / "expertium" / "E89-F65e-F77a" / "master.pth"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--users", default="1340,8254")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    torch._C._set_graph_executor_optimize(False)
    torch.use_deterministic_algorithms(True, warn_only=False)
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import ExitStack
    from rwkv.model import exact_index, nvrtc
    from scripts.expertium_options import CatchUp
    from scripts.gpu_r01_prefetch import training_batches
    from scripts.gpu_row_tiles import row_tiled_backend
    started = time.perf_counter()
    protocol = json.loads((RUN / "run" / "epoch-01" / "protocol.json").read_text())
    options = {k: v for k, v in json.loads((RUN / "experiment.json").read_text())["options"].items()
               if k not in ("packed_ipc", "distillation")}
    users = [int(u) for u in args.users.split(",")]
    rows = [next(r for r in protocol["rows"] if r["user_id"] == u) for u in users]
    length = lambda r: r["train_range"][1] - r["train_range"][0]  # noqa: E731
    # a user with 2-4 updates: T times its second update (table filled by the first); all its batches are prepared
    long_row = min((r for r in protocol["rows"] if length(r) > 2 * 32768), key=length)
    device = torch.device("cuda:0")
    report = {"device": torch.cuda.get_device_name(0), "users": users, "long_user": long_row["user_id"],
              "options": options, "checks": {}}
    fails = []
    nvrtc.register()
    nvrtc.set_checkpoint_length(32)
    nvrtc.set_stream_launch(options.get("stream_launches", False))
    exact_index.set_shared_orders(options.get("shared_shift_orders", False))
    exact_index.set_prefetch_rows(options.get("prefetch_rows", False))
    stack = ExitStack()
    stack.enter_context(row_tiled_backend(rows=options["row_tiles_exact"], tile=32, exact=True))
    longest = max(r["train_range"][1] - r["train_range"][0] for r in rows)
    nvrtc.set_max_timesteps(max(4 * options["chunk_reviews"], 4 * longest))

    def configure(model):
        for module in model.modules():
            for key in ("exact_shift", "exact_index", "exact_group_norm"):
                if hasattr(module, key):
                    setattr(module, key, options.get(key, False))
        model.skip_padding_sources = bool(options.get("skip_padding_sources"))
        model.sync_free_loss = bool(options.get("sync_free_loss"))
        return model

    def items(opts, span, user_rows):
        with ThreadPoolExecutor(max_workers=1) as pool:
            prepared = list(training_batches(user_rows, span, opts["training_seed"], pool, opts, None, 2))
        return [(row, item, item.batch.to(device)) for row, item in prepared]

    carry_options = dict(options, state_carry=True)
    catch_up = CatchUp(carry_options, {}, rows)
    master, base, dtype = catch_up.models({"DTYPE": "bfloat16", "SEED": 2025}, device)
    master.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True), strict=True)
    base.copy_downcast_(master, dtype)
    configure(base)

    def update(model, batch, item, row, carry):
        torch.manual_seed(item.seed)
        model.zero_grad(set_to_none=True)
        if carry is not None:
            carry.carry.attach(item, batch, row["user_id"])
            stats = carry.loss(model, batch, 0)
        else:
            stats = model.get_loss(batch)
        stats.average_loss.backward()
        return stats

    # P: empty table = stock, bf16 train mode
    p_rows = []
    for row, item, batch in items(carry_options, options["batch_chunks"] * options["chunk_reviews"], rows):
        if item.offset != 0:
            continue
        batch.start = batch.start.to(dtype)
        base.train()
        stock = update(base, batch, item, row, None)
        stock_loss = stock.average_loss.detach().clone()
        stock_grads = [None if p.grad is None else p.grad.detach().clone() for p in base.parameters()]
        carried = update(base, batch, item, row, CatchUp(carry_options, {}, [row]))
        grads = [None if p.grad is None else p.grad.detach() for p in base.parameters()]
        rel = max(float((a.float() - b.float()).abs().max()) / max(float(a.float().abs().max()), 1e-12)
                  for a, b in zip(stock_grads, grads) if a is not None and b is not None)
        bitwise = torch.equal(stock_loss, carried.average_loss.detach()) and all(
            (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
            for a, b in zip(stock_grads, grads))
        loss_rel = float((stock_loss - carried.average_loss.detach()).abs() / stock_loss.abs())
        p_rows.append({"user": row["user_id"], "bitwise": bitwise, "loss_rel_diff": loss_rel, "grad_rel_diff": rel})
    p = bool(p_rows) and all(r["loss_rel_diff"] <= 1e-3 and r["grad_rel_diff"] <= 5e-2 for r in p_rows)
    report["checks"]["P_empty_table_vs_stock"] = {"pass": p, "rows": p_rows}
    fails += [] if p else ["P"]

    # X: exact carry with fp32 kernels vs the stock full-history forward
    fp32 = configure(master)
    fp32.eval()
    one = dict(carry_options, batch_chunks=1)
    x_rows = []
    from rwkv.data_processing import get_rwkv_data
    from scripts.gpu_batch1 import split_frame
    from scripts.gpu_r01_samples import samples_for
    from scripts.state_carry_train import user_segment_features
    for row in rows:
        user = row["user_id"]
        frame = user_segment_features(split_frame(get_rwkv_data(Path(DATA), user), row, "train"))
        reviews = len(frame)

        def keys(opts, item):
            (sample,) = samples_for(frame.iloc[item.offset:item.offset + item.reviews], user, opts)
            return (sample.review_ths.long() * 2 + sample.skips.long()).tolist()
        full_options = dict(one, chunk_reviews=reviews)
        ((_, full_item, full),) = items(full_options, reviews, [row])
        with torch.no_grad():
            out = fp32.forward_batch_variant(full.start, full.sub_gather, full.sub_gather_lens, full.time_shift_selects,
                                             full.skips, full.num_data)
        full_p = torch.softmax(out[3].float(), dim=-1)[0]
        where = {k: i for i, k in enumerate(keys(full_options, full_item))}
        windows = items(one, options["chunk_reviews"], [row])
        result = {"user": user, "windows": len(windows)}
        for precision in ("fp32", "deploy"):
            run = CatchUp(dict(one, state_carry={"precision": precision}), {}, [row])
            worst = 0.0
            for _, item, batch in windows:
                index = torch.tensor([where[k] for k in keys(one, item)], device=device)
                run.carry.attach(item, batch, user)
                with torch.no_grad():
                    stats = run.loss(fp32, batch, 0)
                worst = max(worst, float((stats.p_imm_all[0, :index.numel()].float() - full_p.index_select(0, index)).abs().max()))
            result[f"{precision}_max_abs_dp"] = worst
        x_rows.append(result)
    x = all(r["fp32_max_abs_dp"] <= 1e-3 and r["windows"] > 1 for r in x_rows)
    report["checks"]["X_exact_carry_vs_full_history"] = {"pass": x, "rows": x_rows}
    fails += [] if x else ["X"]

    # T: cost per update
    long_items = items(carry_options, options["batch_chunks"] * options["chunk_reviews"], [long_row])[:2]
    timing = {}
    base.train()
    for label in ("stock", "carry"):
        carry = CatchUp(carry_options, {}, [long_row]) if label == "carry" else None
        times = []
        torch.cuda.reset_peak_memory_stats()
        for repeat in range(args.repeats + 1):
            for row, item, batch in long_items:
                batch.start = batch.start.to(dtype)
                if carry is not None and item.offset == 0:
                    carry.carry.user = None  # restart the user so the table refills from offset 0
                torch.cuda.synchronize()
                begin = time.perf_counter()
                update(base, batch, item, row, carry)
                torch.cuda.synchronize()
                if repeat and item.offset > 0:
                    times.append(time.perf_counter() - begin)
        timing[label] = {"median_seconds": statistics.median(times), "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                         "reviews": long_items[1][1].reviews}
        if carry is not None:
            timing[label]["table_bytes"] = carry.carry.table.nbytes()
    timing["carry_over_stock"] = timing["carry"]["median_seconds"] / timing["stock"]["median_seconds"]
    report["checks"]["T_update_cost"] = timing
    stack.close()
    report["fails"] = fails
    report["passed"] = not fails
    report["seconds"] = round(time.perf_counter() - started, 1)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"P": p_rows, "X": x_rows, "T": timing}))
    print("ALL OK" if not fails else f"FAILED: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
