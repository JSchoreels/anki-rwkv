#!/usr/bin/env python
"""Short R01 training runs comparing Expertium catch-up options for speed (docs/EXPERTIUM_CATCH_UP.md).

Every case trains one pass over the same users, in the same order and seed, with the accepted R01 trainer and no
evaluation; only the track options differ. Reviews/s is training review presentations over training seconds
(result.json `reviews_per_second`), the R01 timing scope; process CPU seconds are recorded beside it.

  prepare --out artifacts/expertium-catch-up/speed-<date> [--users 120 --seed 20261014]
  case    --out DIR --case NAME           (one GPU process per case; run through scripts/r01_queue.py)
  all     --out DIR [--cases a,b,...] [--budget-minutes 55]   (GPU check, then each case in a fresh process)
  report  --out DIR

Cases (each a set of overrides on the accepted track E89-F65e-F77a of plan iteration-0090):
  w32            accepted options, default model (control)
  w32-generic    accepted options minus the width-32-only kernels (what every non-32 variant must drop)
  w32-variant    as w32-generic, through the variant class ("architecture": "iter36")
  w80-sequential "architecture": "w80-sequential"
  w80            "architecture": "w80" (interleaved, the champion)
  w64            "architecture": "w64" (s64h, his shipped d=64 trunk)
  w80-muon       w80 + "optimizer": "expertium-iter53"
  w32-kd         accepted options + distillation from the 2.76M fork teacher
"""
from argparse import ArgumentParser, Namespace
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MAIN = Path(r"C:\Users\jonat\workspace\anki-rwkv")
DATASET = MAIN / "data/anki-revlogs-10k-original-20261011"
REVISION = "75299740cff05894ef42d7ad990666691efdd2da"
TRACK = "E89-F65e-F77a"
PLAN = ROOT / "artifacts/retraining/gpu-retrain-r01/plans/iteration-0090-claude.json"
TEACHER = Path(r"C:\Users\jonat\workspace\anki\qt\aqt\rwkv_inference\RWKV_trained_on_5000_10000.pth")


def cases():
    from scripts.expertium_options import compatible_options
    accepted = json.loads(PLAN.read_text())["case_options_by_model"]["expertium"][TRACK]
    generic, dropped = compatible_options({**accepted, "architecture": "w80"})
    generic.pop("architecture")
    return {"w32": accepted, "w32-generic": generic, "w32-variant": {**generic, "architecture": "iter36"},
            "w80-sequential": {**generic, "architecture": "w80-sequential"}, "w80": {**generic, "architecture": "w80"},
            "w64": {**generic, "architecture": "w64"},
            "w80-muon": {**generic, "architecture": "w80", "optimizer": "expertium-iter53"},
            "w32-kd": {**accepted, "distillation": {"teacher": str(TEACHER), "alpha": 0.9}}}, dropped


def prepare(out, users, seed):
    import pyarrow.parquet as pq
    from scripts.run_full_training import R01_COHORT, stratified_split, atomic_json
    out.mkdir(parents=True, exist_ok=False)
    counts = {u: pq.ParquetFile(DATASET / "revlogs" / f"user_id={u}" / "data.parquet").metadata.num_rows
              for u in range(1, 10001)}
    cohort = R01_COHORT if R01_COHORT.exists() else MAIN / R01_COHORT.relative_to(ROOT)  # read-only
    reserved = set(json.loads(cohort.read_text())["independent_final_test_users"])
    others = [user for user in counts if user not in reserved]
    chosen = stratified_split(others, counts, {"train": users}, seed)["train"]
    random.Random(seed).shuffle(chosen)
    definitions, dropped = cases()
    config = {"users": chosen, "review_counts": {str(u): counts[u] for u in chosen},
              "training_reviews": sum(counts[u] for u in chosen), "seed": seed, "cases": definitions,
              "dropped_width32_options": dropped, "dataset_path": str(DATASET), "dataset_revision": REVISION,
              "split": "stratified by history-length decile, outside the preserved final-test cohort"}
    atomic_json(out / "speed.json", config)
    print(json.dumps({"users": users, "training_reviews": config["training_reviews"], "dropped": dropped}))
    return 0


def run_case(out, name):
    import psutil
    from scripts.run_full_training import atomic_json, rows_for
    config = json.loads((out / "speed.json").read_text())
    options = config["cases"][name]
    folder = out / name
    protocol = {"rows": rows_for(config["users"], config["review_counts"]), "evaluation_rows": [],
                "dataset_path": config["dataset_path"], "dataset_revision": config["dataset_revision"],
                "training": {"epochs": 1, "chunk_reviews": 128, "gpu_memory_fraction": 0.95},
                "execution_mode": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, "full_run": {"evaluate": False}}
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder / "protocol.json", protocol)
    os.environ["GPU_BATCH1_DATA_PATH"] = config["dataset_path"]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUPY_CACHE_DIR"] = str(ROOT / "artifacts/cache/cupy-cache")
    import torch
    from scripts import gpu_batch1, gpu_r01_training
    from scripts.run_gpu_r01 import R01SourceWatch
    torch._C._set_graph_executor_optimize(bool(options.get("jit_optimize", False)))
    torch.use_deterministic_algorithms(not options.get("nondeterministic", False), warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory = not options.get("skip_uninitialized_fill", False)
    gpu_batch1.TRACKS = {TRACK: options}
    gpu_r01_training.SourceWatch = R01SourceWatch
    if options.get("gpu_packed_features"):
        from scripts.gpu_r01_staging import training_batches
        from scripts.gpu_b01 import transfer_batches
        gpu_batch1.training_batches = training_batches
        gpu_batch1.transfer_batches = transfer_batches
    elif options.get("prefetch_users"):
        from scripts.gpu_r01_prefetch import training_batches
        gpu_batch1.training_batches = training_batches
    process = psutil.Process()
    cpu_before, system_before = process.cpu_times(), psutil.cpu_times()
    started = time.perf_counter()
    code = gpu_r01_training.run_case(Namespace(model="expertium", track=TRACK, smoke=False, expected_source=None,
                                               output=folder.relative_to(ROOT).as_posix()))
    cpu_after, system_after = process.cpu_times(), psutil.cpu_times()
    busy = lambda t: t.user + t.system + getattr(t, "interrupt", 0.0) + getattr(t, "dpc", 0.0)  # noqa: E731
    atomic_json(folder / "process.json", {
        "wall_seconds": time.perf_counter() - started, "exit": code,
        "main_process_cpu_seconds": (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system),
        "system_busy_cpu_seconds": busy(system_after) - busy(system_before),
        "note": "system_busy_cpu_seconds covers the whole machine (workers, other processes) over the case wall time"})
    return code


def report(out):
    config = json.loads((out / "speed.json").read_text())
    rows = []
    for name in config["cases"]:
        path = out / name / "expertium" / TRACK / "result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        process = json.loads((out / name / "process.json").read_text()) if (out / name / "process.json").exists() else {}
        rows.append({"case": name, "status": result.get("status"), "reviews_per_second": result.get("reviews_per_second"),
                     "training_seconds": result.get("training_seconds"), "updates": result.get("updates"),
                     "reviews": result.get("review_presentations"),
                     "parameters": (result.get("expertium_catch_up") or {}).get("architecture", {}).get("parameters"),
                     "main_process_cpu_seconds": process.get("main_process_cpu_seconds"),
                     "system_busy_cpu_seconds": process.get("system_busy_cpu_seconds"),
                     "peak_allocated_mib": result.get("peak_allocated_mib"), "error": result.get("error")})
    base = next((r["reviews_per_second"] for r in rows if r["case"] == "w32" and r["reviews_per_second"]), None)
    lines = ["| Case | Status | Parameters | Reviews/s | vs w32 | Training s | Updates | Main CPU s | Machine CPU s | Peak MiB |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rate = r["reviews_per_second"]
        lines.append(f"| {r['case']} | {r['status']} | {r['parameters'] or '—'} | {rate:,.0f} | "
                     f"{(rate / base - 1) * 100:+.1f}% | {r['training_seconds']:.1f} | {r['updates']} | "
                     f"{r['main_process_cpu_seconds'] or 0:.0f} | {r['system_busy_cpu_seconds'] or 0:.0f} | "
                     f"{r['peak_allocated_mib']:.0f} |" if rate and base else
                     f"| {r['case']} | {r['status']} | — | — | — | — | — | — | — | — |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


ORDER = ("w32", "w64", "w80", "w32-generic", "w80-sequential", "w80-muon", "w32-variant", "w32-kd")


def run_all(out, names, budget_minutes):
    """One queue job: the GPU parity check, then cases in order until the time budget would be exceeded."""
    import subprocess
    started = time.perf_counter()
    log = out / "all.log"

    def note(text):
        with log.open("a", encoding="utf-8") as handle:
            print(text, file=handle)
        print(text, flush=True)

    check = subprocess.call([sys.executable, "-X", "utf8", str(ROOT / "scripts/check_expertium_gpu.py"),
                             "--output", str(out / "check-gpu.json")], cwd=ROOT)
    note(f"check_expertium_gpu exit {check}")
    accepted = json.loads((out / "speed.json").read_text())["cases"]["w32"]
    if not (out / "exact-iter36-variant").exists():
        # Default-unchanged proof on CUDA: the accepted track vs the same track through the variant class.
        exact = subprocess.call([sys.executable, "-X", "utf8", str(ROOT / "scripts/check_r01_exact_options.py"),
                                 "--a", json.dumps(accepted), "--b", json.dumps({**accepted, "architecture": "iter36"}),
                                 "--output", str(out / "exact-iter36-variant")], cwd=ROOT)
        note(f"check_r01_exact_options accepted vs architecture=iter36 exit {exact}")
    for name in names:
        elapsed = (time.perf_counter() - started) / 60
        if (out / name).exists():
            continue
        if elapsed > budget_minutes:
            note(f"budget reached at {elapsed:.1f} min; not started: {name}")
            continue
        code = subprocess.call([sys.executable, "-X", "utf8", str(Path(__file__)), "case", "--out", str(out),
                                "--case", name], cwd=ROOT)
        note(f"{name} exit {code} after {(time.perf_counter() - started) / 60:.1f} min")
    return report(out)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "case", "report", "all"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--users", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--cases", default=",".join(ORDER))
    parser.add_argument("--budget-minutes", type=float, default=50)
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    if args.command == "prepare":
        return prepare(out, args.users, args.seed)
    if args.command == "case":
        return run_case(out, args.case)
    if args.command == "all":
        return run_all(out, args.cases.split(","), args.budget_minutes)
    return report(out)


if __name__ == "__main__":
    raise SystemExit(main())
