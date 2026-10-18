"""SM21 v2 (strided lanes) GPU check of the trainer's carried-state path; the v2 counterpart of
scripts/check_state_carry_gpu.py (same setup: the ctx2048 run's options, row_tiles_exact 4, exact shift / index /
group norm, skip_padding_sources, deterministic algorithms, the w64 e6 checkpoint). Never run outside the queue.

  P  v2 empty table = stock: the first update (16 lanes, all lane starts) of each user, bf16 train mode, the trainer's
     dropout seed: loss and every gradient (bitwise flag, relative max |diff|);
  X  v2 exact carry on the GPU, fp32 model, eval mode, fp32 table, vs the stock full-history forward on the GPU:
     zero lane starts: rows of the first lane (the exact history prefix) max |dp|; seeded lane starts (each lane's
     table seeded with the full-history state at its start, from v1 one-window-per-update fp32 carry): every row;
     users 8254 at 2 lanes and a 3-update user at the production 16 lanes; the deploy table is reported;
  T  cost: forward + backward of the second 16 x 2,048 update of a 3-update user, stock vs v1 vs v2 (tables filled by
     the first update), median of repeats, peak memory;
  M  memory on the largest training user (v2 has one slot per (lane, entity)): v1 and v2 deploy tables after the first
     update (slots are allocated at full size at the first commit), table bytes and peak memory over two updates.
Pass: P loss relative |diff| <= 1e-3 and gradients relative <= 5e-2 (bf16), X fp32 max |dp| <= 1e-3 (both variants).

  .venv/Scripts/python.exe -X utf8 scripts/check_state_carry_lanes_gpu.py --output artifacts/small-model/check-state-carry-lanes-gpu.json
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
    parser.add_argument("--cpu-smoke", action="store_true",
                        help="CPU, no NVRTC/row tiles, 2 lanes, M on the 3-update user: catches Python errors before a ticket")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    torch._C._set_graph_executor_optimize(False)
    if not args.cpu_smoke:
        torch.use_deterministic_algorithms(True, warn_only=False)
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import ExitStack
    from rwkv.data_processing import get_rwkv_data
    from rwkv.model import exact_index, nvrtc
    from scripts.check_state_carry_lanes import clone_slots, seed_lane_starts
    from scripts.expertium_options import CatchUp
    from scripts.gpu_batch1 import split_frame
    from scripts.gpu_r01_prefetch import training_batches
    from scripts.gpu_r01_samples import samples_for
    from scripts.gpu_row_tiles import row_tiled_backend
    from scripts.state_carry_train import strided_layout, user_segment_features
    started = time.perf_counter()
    protocol = json.loads((RUN / "run" / "epoch-01" / "protocol.json").read_text())
    options = {k: v for k, v in json.loads((RUN / "experiment.json").read_text())["options"].items()
               if k not in ("packed_ipc", "distillation")}
    if args.cpu_smoke:  # 2 lanes and small users: every code path at a few GB of RAM (the CPU WKV keeps per-step states)
        options["batch_chunks"] = 2
    W, L = options["chunk_reviews"], options["batch_chunks"]
    users = [int(u) for u in args.users.split(",")]
    rows = [next(r for r in protocol["rows"] if r["user_id"] == u) for u in users]
    length = lambda r: r["train_range"][1] - r["train_range"][0]  # noqa: E731
    long_row = min((r for r in protocol["rows"] if length(r) > 2 * L * W), key=length)  # 3 updates
    largest_row = long_row if args.cpu_smoke else max(protocol["rows"], key=length)
    device = torch.device("cpu" if args.cpu_smoke else "cuda:0")
    cuda = device.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    peak_mib = (lambda: torch.cuda.max_memory_allocated() / 2**20) if cuda else (lambda: None)
    reset_peak = torch.cuda.reset_peak_memory_stats if cuda else (lambda: None)
    report = {"device": torch.cuda.get_device_name(0) if cuda else "cpu (smoke)", "users": users, "long_user": long_row["user_id"],
              "largest_user": largest_row["user_id"], "options": options, "checks": {}}
    fails = []
    stack = ExitStack()
    if cuda:
        nvrtc.register()
        nvrtc.set_checkpoint_length(32)
        nvrtc.set_stream_launch(options.get("stream_launches", False))
        exact_index.set_shared_orders(options.get("shared_shift_orders", False))
        exact_index.set_prefetch_rows(options.get("prefetch_rows", False))
        stack.enter_context(row_tiled_backend(rows=options["row_tiles_exact"], tile=32, exact=True))
        longest = max(length(r) for r in rows + [long_row])
        nvrtc.set_max_timesteps(max(4 * W, 4 * longest))

    def configure(model):
        for module in model.modules():
            for key in ("exact_shift", "exact_index", "exact_group_norm"):
                if hasattr(module, key):
                    setattr(module, key, options.get(key, False))
        model.skip_padding_sources = bool(options.get("skip_padding_sources"))
        model.sync_free_loss = bool(options.get("sync_free_loss"))
        return model

    def items(opts, user_rows, limit=None):
        out = []
        with ThreadPoolExecutor(max_workers=1) as pool:
            for row, item in training_batches(user_rows, opts["batch_chunks"] * W, opts["training_seed"], pool, opts, None, 2):
                out.append((row, item, item.batch.to(device)))
                if limit is not None and len(out) == limit:
                    break
        return out

    def opts(lanes, spec):
        return dict(options, batch_chunks=lanes, state_carry=spec)

    v1 = opts(L, True)
    v2 = opts(L, {"lanes": "strided"})
    catch_up = CatchUp(v2, {}, rows)
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

    def phase(name):
        print(f"[sm21v2-gpu {time.perf_counter() - started:7.1f}s] {name}", flush=True)

    phase("P")
    # P: v2 first update (empty table) = stock, bf16 train mode
    p_rows = []
    for row, item, batch in [x for r in rows + [long_row] for x in items(v2, [r], limit=1)]:
        batch.start = batch.start.to(dtype)
        base.train()
        stock = update(base, batch, item, row, None)
        stock_loss = stock.average_loss.detach().clone()
        stock_grads = [None if p.grad is None else p.grad.detach().clone() for p in base.parameters()]
        carried = update(base, batch, item, row, CatchUp(v2, {}, [row]))
        grads = [None if p.grad is None else p.grad.detach() for p in base.parameters()]
        rel = max(float((a.float() - b.float()).abs().max()) / max(float(a.float().abs().max()), 1e-12)
                  for a, b in zip(stock_grads, grads) if a is not None and b is not None)
        bitwise = torch.equal(stock_loss, carried.average_loss.detach()) and all(
            (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
            for a, b in zip(stock_grads, grads))
        loss_rel = float((stock_loss - carried.average_loss.detach()).abs() / stock_loss.abs())
        p_rows.append({"user": row["user_id"], "lanes": item.carry_lanes, "bitwise": bitwise,
                       "loss_rel_diff": loss_rel, "grad_rel_diff": rel})
    p = bool(p_rows) and all(r["loss_rel_diff"] <= 1e-3 and r["grad_rel_diff"] <= 5e-2 for r in p_rows)
    report["checks"]["P_v2_empty_table_vs_stock"] = {"pass": p, "rows": p_rows}
    fails += [] if p else ["P"]
    base.zero_grad(set_to_none=True)

    phase("X")
    # X: v2 exact carry with fp32 kernels vs the stock full-history forward
    fp32 = configure(master)
    fp32.eval()
    x_rows = []
    for row, lanes in ((rows[-1], 2), (long_row, L)):
        user = row["user_id"]
        frame = user_segment_features(split_frame(get_rwkv_data(Path(DATA), user), row, "train"))
        reviews = len(frame)
        full_options = dict(opts(1, {"precision": "fp32"}), chunk_reviews=reviews)
        with ThreadPoolExecutor(max_workers=1) as pool:
            ((_, _, full),) = [(r, i, i.batch.to(device)) for r, i in
                               training_batches([row], reviews, options["training_seed"], pool, full_options, None, 2)]
        with torch.no_grad():
            out = fp32.forward_batch_variant(full.start, full.sub_gather, full.sub_gather_lens, full.time_shift_selects,
                                             full.skips, full.num_data)
        full_p = torch.softmax(out[3].float(), dim=-1)[0]
        (sample,) = samples_for(frame, user, full_options)
        where = {k: i for i, k in enumerate((sample.review_ths.long() * 2 + sample.skips.long()).tolist())}

        def index_of(window, chunk_options):
            (s,) = samples_for(frame.iloc[window * W:(window + 1) * W], user, chunk_options)
            return torch.tensor([where[k] for k in (s.review_ths.long() * 2 + s.skips.long()).tolist()], device=device)
        layout = strided_layout(-(-reviews // W), lanes)
        starts = {window: lane for lane, window in layout[0] if window > 0}
        result = {"user": user, "lanes": lanes, "updates": len(layout)}
        phase(f"X user {user} ({reviews} reviews, {lanes} lanes, {len(layout)} updates)")
        for precision in ("fp32", "deploy"):
            one = opts(1, {"precision": precision})
            v1_run = CatchUp(one, {}, [row])
            snaps = {}
            for window, (_, item, batch) in enumerate(items(one, [row])):
                if window in starts:
                    snaps[starts[window]] = clone_slots(v1_run.carry.table)
                v1_run.carry.attach(item, batch, user)
                with torch.no_grad():
                    v1_run.loss(fp32, batch, 0)
            laned = opts(lanes, {"precision": precision, "lanes": "strided"})
            laned_items = items(laned, [row])
            for mode in ("zero", "seeded"):
                run = CatchUp(laned, {}, [row])
                first_lane, rest = 0.0, 0.0
                for u, ((_, item, batch), update_pairs) in enumerate(zip(laned_items, layout)):
                    run.carry.attach(item, batch, user)
                    if u == 0 and mode == "seeded":
                        seed_lane_starts(run.carry.table, snaps)
                    with torch.no_grad():
                        stats = run.loss(fp32, batch, 0)
                    for b, (lane, window) in enumerate(update_pairs):
                        index = index_of(window, laned)
                        dp = float((stats.p_imm_all[b, :index.numel()].float() - full_p.index_select(0, index)).abs().max())
                        if lane == layout[0][0][0]:
                            first_lane = max(first_lane, dp)
                        else:
                            rest = max(rest, dp)
                result[f"{mode}_{precision}"] = {"max_abs_dp_first_lane": first_lane, "max_abs_dp_other_lanes": rest}
        x_rows.append(result)
    x = all(r["zero_fp32"]["max_abs_dp_first_lane"] <= 1e-3 and r["seeded_fp32"]["max_abs_dp_first_lane"] <= 1e-3
            and r["seeded_fp32"]["max_abs_dp_other_lanes"] <= 1e-3 and r["updates"] > 1 for r in x_rows)
    report["checks"]["X_v2_exact_carry_vs_full_history"] = {"pass": x, "rows": x_rows}
    fails += [] if x else ["X"]

    phase("T")
    # T: cost per update (second update of the 3-update user)
    timing = {}
    base.train()
    for label, spec in (("stock", None), ("v1", v1), ("v2", v2)):
        long_items = items(spec or v1, [long_row], limit=2)
        carry = CatchUp(spec, {}, [long_row]) if spec is not None else None
        times = []
        reset_peak()
        for repeat in range(args.repeats + 1):
            for row, item, batch in long_items:
                batch.start = batch.start.to(dtype)
                if carry is not None and item.offset == 0:
                    carry.carry.user = None  # restart the user so the table refills from offset 0
                sync()
                begin = time.perf_counter()
                update(base, batch, item, row, carry)
                sync()
                if repeat and item.offset > 0:
                    times.append(time.perf_counter() - begin)
        timing[label] = {"median_seconds": statistics.median(times),
                         "peak_memory_mib": peak_mib(), "reviews": long_items[1][1].reviews}
        if carry is not None:
            timing[label]["table_bytes"] = carry.carry.table.nbytes()
        del long_items
    timing["v1_over_stock"] = timing["v1"]["median_seconds"] / timing["stock"]["median_seconds"]
    timing["v2_over_stock"] = timing["v2"]["median_seconds"] / timing["stock"]["median_seconds"]
    timing["v2_over_v1"] = timing["v2"]["median_seconds"] / timing["v1"]["median_seconds"]
    report["checks"]["T_update_cost"] = timing

    phase("M")
    # M: table size and peak memory on the largest training user (first two updates; full-size slots after the first)
    memory = {}
    for label, spec in (("v1", v1), ("v2", v2)):
        if cuda:
            torch.cuda.empty_cache()
        reset_peak()
        run = CatchUp(spec, {}, [largest_row])
        begin = time.perf_counter()
        try:
            for row, item, batch in items(spec, [largest_row], limit=2):
                batch.start = batch.start.to(dtype)
                update(base, batch, item, row, run)
                del batch
            sync()
            memory[label] = {"table_bytes": run.carry.table.nbytes(), "slots": run.carry.table.size}
        except torch.cuda.OutOfMemoryError as error:  # informational: report it instead of losing the other checks
            memory[label] = {"out_of_memory": str(error)[:300]}
        memory[label].update(peak_memory_mib=peak_mib(),
                             seconds_incl_preparation=round(time.perf_counter() - begin, 1))
        del run
        base.zero_grad(set_to_none=True)
    report["checks"]["M_largest_user_memory"] = memory
    stack.close()
    report["fails"] = fails
    report["passed"] = not fails
    report["seconds"] = round(time.perf_counter() - started, 1)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"P": p_rows, "X": x_rows, "T": timing, "M": memory}))
    print("ALL OK" if not fails else f"FAILED: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
