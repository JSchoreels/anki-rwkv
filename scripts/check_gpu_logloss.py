"""Paired temporal log-loss screening for the fixed-batch GPU experiments.

This is a bounded regression screen on user 17, not a full accuracy study.
Every variant uses the same initialization, seeded batches, and later holdout.
Evaluation always uses the original inputs and updated FP32 master weights.
"""

from argparse import ArgumentParser
from contextlib import nullcontext, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scripts.experiment_provenance import process_provenance, provenance_rows
OUT = ROOT / "artifacts/benchmarks/perf-optimizations/logloss"
METRICS = ("curve_log_loss", "instant_log_loss")
_SOURCE = None
_VALIDATION_CACHE = {}


def cases(model):
    from scripts.benchmark_gpu_optimizations import VARIANTS
    variants = dict(VARIANTS)
    variants["combined"] = ({"CUDA_GRAPH": True, "FAST_COPIES": True}
                            if model == "baseline" else
                            {"CUDA_GRAPH": True, "PREPARATION": "cpu"})
    if model == "expertium":
        variants["combined-gpu-packed"] = {"CUDA_GRAPH": True, "PREPARATION": "gpu-packed"}
    variants["reference-repeat"] = {}
    return variants


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_case(args, shared_pool=None):
    provenance = process_provenance()
    import pandas as pd
    import torch
    from rwkv import data_processing as processing
    from rwkv.model import nvrtc
    from scripts.benchmark_gpu_trial import fresh_checkpoint, run_variant
    from scripts.compare import evaluate
    from scripts.gpu_power import limited_gpu
    from scripts.trial_batches import preparation_pool, prepare_section

    output = OUT / args.model / args.variant / str(args.seed)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.72, 0)
    chunks, workers = (128, 4) if args.model == "baseline" else (256, 6)
    config = {"MODEL": args.model, "DTYPE": "bfloat16", "WKV_BACKEND": "nvrtc",
              "CHUNK_REVIEWS": 128, "BATCH_CHUNKS": chunks, "PREFETCH_WORKERS": workers,
              "PREFETCH_BATCHES": workers * 2, "REVIEWS_PER_USER": 0, "SEED": args.seed,
              **cases(args.model)[args.variant]}
    report = {"model": args.model, "variant": args.variant, "seed": args.seed,
              "provenance": provenance,
              "config": config, "status": "running", "steps": args.steps,
              "warmup_updates": 6, "training_dropout": "enabled"}
    atomic_json(output / "result.json", report)
    try:
        global _SOURCE
        if _SOURCE is None:
            _SOURCE = processing.get_rwkv_data(ROOT / "data/anki-revlogs-10k", 17)
        source = _SOURCE
        train_begin, train_end, validation_end = 4096, 12288, 20480
        if len(source) < validation_end:
            raise ValueError("User 17 does not have enough reviews for the fixed holdout")

        def section(begin, end):
            selected = source.iloc[begin:end].copy()
            selected["has_label"] = (selected["has_label"].astype(bool) &
                                      selected["label_review_th"].between(begin + 1, end)).astype(int)
            labelled = selected.loc[selected["has_label"].astype(bool), "label_review_th"]
            assert len(labelled) and labelled.between(begin + 1, end).all()
            return selected

        train = section(train_begin, train_end)
        heldout = section(train_end, validation_end)
        report["data"] = {"user": 17, "available_reviews": len(source),
                          "unique_training_reviews": len(train),
                          "train_review_range": [train_begin + 1, train_end],
                          "heldout_review_range": [train_end + 1, validation_end],
                          "heldout_raw_reviews": len(heldout),
                          "next_review_targets_confined_to_each_split": True,
                          "training_window_repeated": True}
        if args.seed not in _VALIDATION_CACHE:
            validation = []
            for offset in range(0, len(heldout), 1024):
                batch = prepare_section(heldout.iloc[offset:offset + 1024], 17,
                                        train_end + offset, 128, args.seed + 17 + train_end + offset).batch
                validation.append(batch)
            _VALIDATION_CACHE[args.seed] = validation
        validation = _VALIDATION_CACHE[args.seed]
        fingerprint = hashlib.sha256()
        for batch in validation:
            for tensor in (batch.start, batch.labels, batch.label_review_th):
                fingerprint.update(tensor.numpy().tobytes())
        report["validation_input_sha256"] = fingerprint.hexdigest()

        def validate(master):
            # All trained variants are evaluated through identical arithmetic,
            # including original features, standard tiles and FP32 weights.
            nvrtc.set_checkpoint_length(32)
            nvrtc.set_raw_pointer_launch(False)
            def moved():
                for batch in validation:
                    yield {"batch": batch.to(torch.device("cuda:0"))}
            metrics = evaluate(master, moved())
            assert all(math.isfinite(metrics[key]) for key in METRICS)
            weights = hashlib.sha256()
            for name, parameter in master.named_parameters():
                weights.update(name.encode())
                weights.update(parameter.detach().cpu().numpy().tobytes())
            return {"evaluation": "original preparation, dropout off, FP32 master, tile32",
                    "metrics": metrics, "trained_parameter_sha256": weights.hexdigest()}

        needed = (6 + args.steps + 1) * chunks * 128
        frame = pd.concat([train] * ((needed + len(train) - 1) // len(train)), ignore_index=True)
        checkpoint = fresh_checkpoint(args.model, args.seed)
        initial = hashlib.sha256()
        for name, tensor in checkpoint["model"].items():
            initial.update(name.encode())
            initial.update(tensor.numpy().tobytes())
        report["initial_parameter_sha256"] = initial.hexdigest()
        with limited_gpu(0, 0, output):
            context = preparation_pool(workers) if shared_pool is None else nullcontext(shared_pool)
            with context as pool:
                if args.variant == "reference-repeat":
                    # Discard this model and optimizer. Prime both train/eval JIT
                    # paths, then start again from the identical fresh checkpoint.
                    run_variant("discarded-jit-primer", frame, config, checkpoint, output,
                                6, 1, pool, after_training=validate)
                    report["discarded_jit_primer_updates"] = 7
                result = run_variant(args.variant, frame, config, checkpoint, output,
                                     6, args.steps, pool, after_training=validate)
        report.update(status="complete", result=result)
    except Exception as exc:
        report.update(status="failed", error=str(exc), error_type=type(exc).__name__)
        traceback.print_exc()
    atomic_json(output / "result.json", report)
    return 0 if report["status"] == "complete" else 1


def summarize(report):
    completed = [item for item in report["results"] if item["status"] == "complete"]
    controls = {(item["model"], item["seed"]): item for item in completed
                if item["variant"] == "reference"}
    # The repeat control establishes normal JIT execution after cold profiling.
    controls.update({(item["model"], item["seed"]): item for item in completed
                     if item["variant"] == "reference-repeat"})
    cold = {(item["model"], item["seed"]): item for item in completed
            if item["variant"] == "reference"}
    report["repeat_control_deltas"] = [
        {"model": model, "seed": seed,
         "delta": {key: control["result"]["quality"]["metrics"][key] -
                        cold[(model, seed)]["result"]["quality"]["metrics"][key]
                   for key in METRICS}}
        for (model, seed), control in controls.items()
        if control["variant"] == "reference-repeat" and (model, seed) in cold]
    comparisons = []
    for model in ("baseline", "expertium"):
        for variant in cases(model):
            if variant in ("reference", "reference-repeat"):
                continue
            paired = []
            for item in completed:
                if (item["model"], item["variant"]) != (model, variant):
                    continue
                control = controls.get((model, item["seed"]))
                if control is None:
                    continue
                assert item["validation_input_sha256"] == control["validation_input_sha256"]
                assert item["initial_parameter_sha256"] == control["initial_parameter_sha256"]
                metrics = item["result"]["quality"]["metrics"]
                reference = control["result"]["quality"]["metrics"]
                assert metrics["curve_labels"] == reference["curve_labels"]
                assert metrics["instant_labels"] == reference["instant_labels"]
                paired.append({"seed": item["seed"], "control": control["variant"],
                               "delta": {key: metrics[key] - reference[key] for key in METRICS}})
            if not paired:
                continue
            comparisons.append({"model": model, "variant": variant, "paired_seeds": paired,
                                "mean_delta": {key: statistics.mean(p["delta"][key] for p in paired)
                                               for key in METRICS},
                                "max_delta": {key: max(p["delta"][key] for p in paired)
                                              for key in METRICS},
                                "worse_by_over_0.001_on_any_seed": any(
                                    p["delta"][key] > 0.001 for p in paired for key in METRICS)})
    report["comparisons"] = comparisons


def markdown(report):
    rows = ["# Paired GPU optimization log-loss screen", "",
            "User 17: train on reviews 4097–12288, then evaluate on held-out reviews 12289–20480. "
            "Next-review targets stay inside their split. Training repeats the same 8192-review window "
            f"for 6 warmup + {report['steps']} updates, using enabled dropout, BF16 computation and FP32 master weights.", "",
            "Each variant starts from the same fresh weights and AdamW state as its architecture's paired control. "
            "Validation uses original features, dropout disabled, FP32 master weights, standard WKV tile32. "
            "Batch sizes stay at 128 chunks for default and 256 for Expertium.", "",
            "This is a short regression screen on one user's later reviews. It does not establish "
            "non-inferiority for complete 100-user training. Positive deltas mean worse log loss. "
            "The 0.001 flag is a screening threshold, not a confidence interval or a quality guarantee.", "",
            "When available, deltas use the repeated control after a discarded JIT primer. "
            "The cold control is also retained below to expose profiling/warmup effects.", "",
            f"Seeds: {', '.join(map(str, report['seeds']))}. Status: **{report['status']}**.", "",
            "| Model | Variant | Paired seeds | Mean curve Δ | Mean instant Δ | Worst curve Δ | Worst instant Δ | Any Δ > 0.001 |",
            "|---|---|---:|---:|---:|---:|---:|---|" ]
    for item in report.get("comparisons", []):
        mean, worst = item["mean_delta"], item["max_delta"]
        rows.append(f"| {item['model']} | {item['variant']} | {len(item['paired_seeds'])} | "
                    f"{mean['curve_log_loss']:+.6f} | {mean['instant_log_loss']:+.6f} | "
                    f"{worst['curve_log_loss']:+.6f} | {worst['instant_log_loss']:+.6f} | "
                    f"{'yes' if item['worse_by_over_0.001_on_any_seed'] else 'no'} |")
    rows.extend(["", "Control validation scores:", "",
                 "| Model | Control | Seed | Curve log loss | Instant log loss | Curve labels | Instant labels |",
                 "|---|---|---:|---:|---:|---:|---:|"])
    for item in report["results"]:
        if item["variant"] in ("reference", "reference-repeat") and item["status"] == "complete":
            m = item["result"]["quality"]["metrics"]
            rows.append(f"| {item['model']} | {item['variant']} | {item['seed']} | {m['curve_log_loss']:.6f} | "
                        f"{m['instant_log_loss']:.6f} | {m['curve_labels']} | {m['instant_labels']} |")
    if report.get("repeat_control_deltas"):
        rows.extend(["", "Repeat-minus-cold control deltas (same initialization/data; expose JIT and run variation):", "",
                     "| Model | Seed | Curve Δ | Instant Δ |", "|---|---:|---:|---:|"])
        for item in report["repeat_control_deltas"]:
            rows.append(f"| {item['model']} | {item['seed']} | {item['delta']['curve_log_loss']:+.6f} | "
                        f"{item['delta']['instant_log_loss']:+.6f} |")
    if report.get("preparation"):
        windows = report["preparation"]
        total = sum(item["bf16_feature_values"] for item in windows)
        unpacked = sum(item["gpu_bf16_different_values"] for item in windows)
        packed = sum(item["gpu_packed_bf16_different_values"] for item in windows)
        rows.extend(["", "BF16 feature parity:", "",
                     f"Across {len(windows)} real-history windows ({total:,} feature values), "
                     f"unpacked GPU preparation changes {unpacked:,} BF16 values; packed GPU preparation changes {packed:,}."])
    if report.get("calendar_precision"):
        calendar = report["calendar_precision"]
        rows.extend(["", f"Enumerating every integer remainder for all seven calendar periods "
                     f"({calendar['values']:,} sine/cosine values) finds "
                     f"{calendar['packed_bf16_different']} packed BF16 differences and "
                     f"{calendar['unpacked_bf16_different']} unpacked BF16 differences against CPU. "
                     "The packed path uses the original CPU scalar frequency rounding before GPU multiplication."])
    rows.extend(["", "Experiment source revisions (per-run metadata):", "",
                 "| Suite | Runs | Cases | Experiment commit | Working tree | Source snapshot |",
                 "|---|---:|---|---|---|---|"])
    rows.extend(provenance_rows(report, "Log loss", "../../../"))
    return "\n".join(rows) + "\n"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="store_true")
    parser.add_argument("--model", choices=("baseline", "expertium"))
    parser.add_argument("--variant")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026, 2027])
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--in-process", action="store_true",
                        help="Reuse CPU preparation workers and JIT/kernel caches across variants")
    args = parser.parse_args()
    provenance = process_provenance()
    if args.steps <= 0:
        parser.error("Steps must be positive")
    if args.case:
        if not args.model or args.variant not in cases(args.model):
            parser.error("Case needs a model and known variant")
        return run_case(args)
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "provenance": provenance,
              "steps": args.steps, "seeds": args.seeds, "results": [],
              "scope": "paired temporal holdout regression screen, user17, repeated training window",
              "screening_delta_threshold": 0.001}
    path = OUT / "comparison.json"
    if args.resume and path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["steps"] != args.steps or report["seeds"] != args.seeds:
            parser.error("Resume protocol must match saved steps and seeds")
        report["status"] = "running"
    report.setdefault("source_sessions", []).append({"resume": args.resume,
                                                      "provenance": provenance})
    successful = {(i["model"], i["variant"], i["seed"]) for i in report["results"]
                  if i["status"] == "complete"}
    from contextlib import ExitStack
    with ExitStack() as stack:
        pools = {}
        if args.in_process:
            from scripts.trial_batches import preparation_pool
            pools = {model: stack.enter_context(preparation_pool(workers))
                     for model, workers in (("baseline", 4), ("expertium", 6))}
        planned = [(seed, model, variant) for seed in args.seeds
                   for model in ("baseline", "expertium") for variant in cases(model)]
        for seed, model, variant in planned:
            if (model, variant, seed) in successful:
                continue
            folder = OUT / model / variant / str(seed)
            folder.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__)), "--case", "--model", model,
                       "--variant", variant, "--seed", str(seed), "--steps", str(args.steps)]
            print(f"Starting {model}/{variant}, seed {seed}", flush=True)
            started = time.perf_counter()
            with (folder / "console.log").open("w", encoding="utf-8") as log:
                if args.in_process:
                    from argparse import Namespace
                    with redirect_stdout(log), redirect_stderr(log):
                        code = run_case(Namespace(model=model, variant=variant, seed=seed,
                                                  steps=args.steps), shared_pool=pools[model])
                else:
                    try:
                        process = subprocess.run(command, cwd=ROOT, stdout=log,
                                                 stderr=subprocess.STDOUT, timeout=420)
                        code = process.returncode
                    except subprocess.TimeoutExpired:
                        code = -1
            result_path = folder / "result.json"
            item = (json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists()
                    else {"model": model, "variant": variant, "seed": seed, "status": "failed",
                          "error": "Missing result or timed out"})
            if item["status"] != "complete" or code:
                item["status"] = "failed"
            item.update(process_exit_code=code, elapsed_seconds=time.perf_counter() - started)
            report["results"] = [i for i in report["results"]
                                 if (i["model"], i["variant"], i["seed"]) != (model, variant, seed)]
            report["results"].append(item)
            summarize(report)
            atomic_json(path, report)
            (OUT / "summary.md").write_text(markdown(report), encoding="utf-8")
            if item["status"] == "complete":
                metrics = item["result"]["quality"]["metrics"]
                print(f"Finished {model}/{variant}, seed {seed}: "
                      f"curve={metrics['curve_log_loss']:.6f}, instant={metrics['instant_log_loss']:.6f}", flush=True)
            else:
                print(f"FAILED {model}/{variant}, seed {seed}: {item.get('error', '')[:180]}", flush=True)
    report.update(status="complete" if all(i["status"] == "complete" for i in report["results"]) else "failed",
                  finished_utc=datetime.now(timezone.utc).isoformat())
    expected = {(model, variant, seed) for model in ("baseline", "expertium")
                for variant in cases(model) for seed in args.seeds}
    actual = {(item["model"], item["variant"], item["seed"]) for item in report["results"]}
    report["expected_cases"] = len(expected)
    report["missing_cases"] = sorted(expected - actual)
    if report["missing_cases"]:
        report["status"] = "failed"
    report["control_strategy"] = "Prefer repeated control after a discarded train/eval JIT primer; retain cold controls"
    if report["status"] == "complete":
        import torch
        from scripts.check_gpu_optimizations import check_preparation, check_calendar_precision
        torch.set_num_threads(4)
        report["preparation"] = check_preparation()
        report["calendar_precision"] = check_calendar_precision()
        atomic_json(OUT / "bf16-preparation.json", report["preparation"])
        atomic_json(OUT / "calendar-precision.json", report["calendar_precision"])
    summarize(report)
    atomic_json(path, report)
    (OUT / "summary.md").write_text(markdown(report), encoding="utf-8")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
