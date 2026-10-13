"""F92a check: worker-loaded history frames give the same packed batches and RNG as main-process slices.

Runs every training job of the frozen protocol twice through spawned worker pools: once in the
accepted path (the main process loads each user's frame and pickles the history slice to a worker)
and once with `worker_frames` (the job carries no slice; the worker loads and splits the frame
itself). Compares packed batch bytes, offsets, seeds, review counts and post-preparation RNG in
update order. CPU only.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_r01_worker_counts import digest  # noqa: E402


def checked_prepare(arguments):
    import torch
    from scripts.gpu_preprocessing_stage import prepare_job
    item = prepare_job(*arguments)
    return {"user": arguments[0]["user_id"], "offset": item.offset, "seed": item.seed, "reviews": item.reviews,
            "batch_sha256": digest(item.batch), "rng_sha256": digest(torch.random.get_rng_state()),
            "worker_pid": os.getpid()}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--track", required=True, help="Accepted control track whose options are used")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=0, help="Limit to the first N training users (0 = all)")
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists() or not output.resolve().is_relative_to(ROOT / "artifacts"):
        parser.error("Use a fresh artifacts output")
    protocol = json.loads((ROOT / args.protocol).read_text())
    os.environ["GPU_BATCH1_DATA_PATH"] = str(ROOT / protocol["dataset_path"])
    import torch
    torch.set_num_threads(1)
    from scripts.gpu_r01_prefetch import training_batches
    from scripts.trial_batches import worker_init
    from scripts.experiment_provenance import process_provenance
    plan = json.loads((ROOT / args.plan).read_text())
    options = plan["case_options_by_model"]["expertium"][args.track]
    rows = protocol["rows"][:args.users] if args.users else protocol["rows"]
    span = options["batch_chunks"] * 128
    seed = options.get("training_seed", 2025)
    workers = options["preparation_workers"]
    result = {"provenance": process_provenance(), "control_track": args.track, "workers": workers, "users": len(rows),
              "scope": "every training job of the protocol in update order through the R01 ordered-prefetch generator; "
                       "spawned pools; packed batch bytes, offsets, seeds, review counts and post-preparation RNG; CPU only",
              "variants": {}}

    class Item:
        def __init__(self, value):
            self.value = value

    class Recorder:
        """Pool adapter: run checked_prepare on the same arguments the generator submits."""
        def __init__(self, pool):
            self.pool = pool

        def submit(self, function, *arguments):
            future = self.pool.submit(checked_prepare, arguments)
            class Wrapped:
                def result(_):
                    return Item(future.result())
                def cancel(_):
                    return future.cancel()
            return Wrapped()

    for name, extra in (("main_slices", {}), ("worker_frames", {"worker_frames": True})):
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=worker_init) as pool:
            items = [item.value for _, item in training_batches(rows, span, seed, Recorder(pool), {**options, **extra}, None, workers * 2)]
        result["variants"][name] = {"seconds": time.perf_counter() - started, "jobs": len(items),
                                    "distinct_worker_user_loads": len({(i["worker_pid"], i["user"]) for i in items}),
                                    "items": [{k: v for k, v in i.items() if k != "worker_pid"} for i in items]}
        print(f"[variant] {name}: {len(items)} jobs in {result['variants'][name]['seconds']:.1f} s", flush=True)
    control, candidate = (result["variants"][k]["items"] for k in ("main_slices", "worker_frames"))
    mismatches = [i for i, (a, b) in enumerate(zip(control, candidate)) if a != b]
    result["status"] = "passed" if len(control) == len(candidate) and control and not mismatches else "failed"
    result["summary"] = {"jobs": len(control), "mismatched_jobs": mismatches[:20], "all_equal": result["status"] == "passed",
                         "worker_frame_loads": result["variants"]["worker_frames"]["distinct_worker_user_loads"],
                         "users": len(rows)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "summary": result["summary"]}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
