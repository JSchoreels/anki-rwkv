"""R01 sensitivity study: does F22 (non-deterministic kernels) change held-out loss
beyond seed-to-seed variation? Diagnostic only; not an optimization iteration or
the 500-user validation gate.

`prepare` freezes a study protocol over a fixed cohort stage whose tracks vary the
training seed (initialization, dropout, batch encodings) and the execution mode;
held-out inputs keep seed 2025 so every run scores identical evaluation batches.
Runs launch through scripts/run_isolated_gpu_r01.py. `report` aggregates them with
the recorded seed-2025 runs.
"""

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "artifacts/retraining/gpu-retrain-r01"
METRICS = ("curve_log_loss_label_weighted", "curve_log_loss_user_average",
           "instant_log_loss_label_weighted", "instant_log_loss_user_average")
EXPERTIUM = {"prefetch_users": True, "preparation_workers": 8, "gpu_packed_features": True, "packed_ipc": True,
             "async_transfer": True, "batched_id_encodings": True, "grouped_sync": True, "main_threads": 1,
             "bulk_transfer": True, "parallel_evaluation": True}
MODES = {"expertium": {"E06-E11-F26-F27": {**EXPERTIUM, "exact_shift": True, "exact_index": True},
                       "E06-E11-F22": {**EXPERTIUM, "nondeterministic": True}},
         "baseline": {"F01-F21": {"prefetch_users": True, "parallel_evaluation": True},
                      "F01-F22-F21": {"prefetch_users": True, "nondeterministic": True, "parallel_evaluation": True}}}


def deterministic(track):
    return "F22" not in track.split("-")


def prepare(args):
    from scripts.experiment_provenance import process_provenance
    from scripts.run_gpu_r01 import R01SourceWatch
    output = BASE / f"iteration-{args.iteration}"
    if (output / "protocol.json").exists():
        raise FileExistsError("Preserve the frozen study; use a new iteration number")
    template = json.loads((BASE / f"iteration-{args.template}/protocol.json").read_text(encoding="utf-8"))
    cohort_path = ROOT / template["cohort_file"]
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if hashlib.sha256(cohort_path.read_bytes()).hexdigest() != template["cohort_sha256"]:
        raise ValueError("Cohort changed since the template protocol")
    stage = cohort["stages"][args.stage]
    tracks, options, explanations = {}, {}, {}
    for model in args.models:
        tracks[model], options[model], explanations[model] = [], {}, {}
        for mode, base in MODES[model].items():
            for seed in args.seeds:
                track = f"S{seed % 100:02d}-{mode}"
                tracks[model].append(track)
                options[model][track] = {**base, "training_seed": seed}
                explanations[model][track] = (f"Training seed {seed} ({'deterministic exact' if deterministic(mode) else 'non-deterministic F22'} kernels); "
                                              f"held-out inputs keep seed 2025. Options as track {mode}.")
            for seed in args.repeat_seeds:
                track = f"R{seed % 100:02d}-{mode}"
                if not deterministic(mode):
                    tracks[model].append(track)
                    options[model][track] = {**base, "training_seed": seed}
                    explanations[model][track] = (f"Repeat with training seed {seed} and non-deterministic F22 kernels: "
                                                  "measures run-to-run variation from kernel non-determinism alone.")
    provenance = process_provenance()
    output.mkdir(parents=True)
    loop = (ROOT / "docs/GPU_RETRAIN_LOOP.md").read_bytes()
    (output / "loop-definition.md").write_bytes(loop)
    protocol = {**{key: template[key] for key in ("cohort_file", "cohort_sha256", "dataset_path", "dataset_revision", "sampling_seed",
                                                    "training", "execution_mode", "evaluation", "holdout", "timing")},
                **stage, "iteration": args.iteration, "protocol_status": "frozen before experiments",
                "study": f"F22 seed sensitivity, {args.stage}-user cohort stage",
                "authorization": "2026-10-12 human request to Claude: Yeah let's try F17 but also re-explore iteration 11 with a bit more users and/or seed just to see if the logloss >1% was just an accident or not",
                "status": "diagnostic sensitivity study; not an optimization iteration, not broader validation (500 evaluation users not reached)",
                "models": args.models, "seeds": [2025], "training_seeds": args.seeds, "repeat_seeds": args.repeat_seeds,
                "evaluation_users": len(stage["heldout_users"]), "training_user_count": len(stage["training_users"]),
                "study_tracks": tracks, "case_options_by_model": options, "case_explanations": explanations,
                "planned_updates": {model: sum((row["available_reviews"] + chunks * 128 - 1) // (chunks * 128) for row in stage["rows"])
                                    for model, chunks in (("baseline", 128), ("expertium", 256))},
                "analysis": "Per metric: deterministic seed spread; paired non-deterministic minus deterministic difference at equal seeds; "
                            "repeat difference at equal seed. F22's effect is indistinguishable from seed variation if the mean paired difference "
                            "is small relative to the deterministic seed spread.",
                "loop_sha256": hashlib.sha256(loop).hexdigest(), "frozen_utc": datetime.now(timezone.utc).isoformat(),
                "preparation_provenance": provenance, "implementation_sha256": R01SourceWatch(provenance).implementation_sha256}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"study": str(output), "tracks": tracks}, indent=1))


def relative(value, reference):
    return 100 * (value / reference - 1)


def report(args):
    output = BASE / f"iteration-{args.iteration}"
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    extra = json.loads(args.include) if args.include else {}
    runs = []
    for model in protocol["models"]:
        for track in protocol["study_tracks"][model]:
            path = output / model / track / "result.json"
            if path.exists():
                runs.append((model, track, path))
        for track, path in extra.get(model, {}).items():
            runs.append((model, track, ROOT / path))
    table, results = [], {}
    for model, track, path in runs:
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("status") != "complete":
            continue
        seed = case["options"].get("training_seed", 2025)
        mode = "non-deterministic" if case.get("execution_mode", {}).get("deterministic_algorithms") is False else "deterministic"
        kind = "repeat" if track.startswith("R") else "seed"
        row = {"model": model, "track": track, "seed": seed, "mode": mode, "kind": kind,
               "result": path.relative_to(ROOT).as_posix(), "reviews_per_second": case["reviews_per_second"],
               "validation_input_sha256": case["validation_input_sha256"], **{m: case["evaluation"][m] for m in METRICS}}
        table.append(row)
    summary = {}
    for model in protocol["models"]:
        rows = [r for r in table if r["model"] == model]
        if len({r["validation_input_sha256"] for r in rows}) > 1:
            raise ValueError("Held-out inputs differ across runs; the study is not comparable")
        det = {r["seed"]: r for r in rows if r["mode"] == "deterministic" and r["kind"] == "seed"}
        nd = {r["seed"]: r for r in rows if r["mode"] == "non-deterministic" and r["kind"] == "seed"}
        rep = [r for r in rows if r["kind"] == "repeat"]
        paired = sorted(set(det) & set(nd))
        model_summary = {"deterministic_seeds": sorted(det), "nondeterministic_seeds": sorted(nd), "paired_seeds": paired, "metrics": {}}
        for metric in METRICS:
            det_values = [det[s][metric] for s in sorted(det)]
            mean = statistics.mean(det_values)
            diffs = [relative(nd[s][metric], det[s][metric]) for s in paired]
            seed_spread = [relative(v, mean) for v in det_values]
            item = {"deterministic_mean": mean,
                    "deterministic_seed_sd_percent": statistics.stdev(seed_spread) if len(seed_spread) > 1 else None,
                    "deterministic_seed_range_percent": [min(seed_spread), max(seed_spread)],
                    "paired_nondeterministic_minus_deterministic_percent": diffs,
                    "mean_paired_difference_percent": statistics.mean(diffs) if diffs else None,
                    "paired_difference_sd_percent": statistics.stdev(diffs) if len(diffs) > 1 else None}
            if len(diffs) > 1 and item["paired_difference_sd_percent"]:
                item["paired_t"] = item["mean_paired_difference_percent"] / (item["paired_difference_sd_percent"] / math.sqrt(len(diffs)))
            repeats = []
            for r in rep:
                original = nd.get(r["seed"])
                if original:
                    repeats.append(relative(r[metric], original[metric]))
            item["nondeterministic_repeat_difference_percent"] = repeats
            model_summary["metrics"][metric] = item
        summary[model] = model_summary
    result = {"study": protocol["study"], "status": protocol["status"], "generated_utc": datetime.now(timezone.utc).isoformat(),
              "runs": table, "summary": summary}
    (output / "study-result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [f"# {protocol['study']}", "", f"{protocol['status'].capitalize()}. Authorization: {protocol['authorization']}.", "",
             f"Cohort stage: {len(protocol['training_users'])} training / {len(protocol['heldout_users'])} unseen held-out users, "
             f"{protocol['training_reviews']:,} training / {protocol['validation_reviews']:,} held-out reviews. Training seeds vary initialization, dropout and batch encodings; "
             "held-out inputs keep seed 2025 (identical validation-input hashes are enforced). Deterministic runs use the exact accepted kernels; non-deterministic runs add F22.", "",
             "## Runs", "", "| Model | Track | Seed | Kernels | Reviews/s | Curve W | Curve U | Instant W | Instant U |", "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in sorted(table, key=lambda r: (r["model"], r["mode"], r["kind"], r["seed"])):
        lines.append(f"| {r['model']} | [{r['track']}]({Path(r['result']).relative_to(output.relative_to(ROOT)).as_posix() if r['result'].startswith(output.relative_to(ROOT).as_posix()) else '../../../../' + r['result']}) | {r['seed']} | {r['mode']}{' repeat' if r['kind'] == 'repeat' else ''} | "
                     f"{r['reviews_per_second']:,.0f} | " + " | ".join(f"{r[m]:.6f}" for m in METRICS) + " |")
    lines += ["", "## Effect of F22 relative to seed variation", "",
              "Percentages are relative to the deterministic run (paired) or the deterministic seed mean (spread). Positive means higher, worse loss.", "",
              "| Model | Metric | Deterministic seed SD | Seed range | Paired F22 − deterministic, each seed | Mean paired | Paired t | F22 repeat at equal seed |", "|---|---|---:|---|---|---:|---:|---|"]
    for model, item in summary.items():
        for metric, m in item["metrics"].items():
            sd = m["deterministic_seed_sd_percent"]
            lines.append(f"| {model} | {metric} | {'—' if sd is None else f'{sd:.2f}%'} | {m['deterministic_seed_range_percent'][0]:+.2f}% to {m['deterministic_seed_range_percent'][1]:+.2f}% | "
                         + ", ".join(f"{d:+.2f}%" for d in m["paired_nondeterministic_minus_deterministic_percent"]) + " | "
                         + ("—" if m["mean_paired_difference_percent"] is None else f"{m['mean_paired_difference_percent']:+.2f}%") + " | "
                         + (f"{m['paired_t']:.2f}" if "paired_t" in m else "—") + " | "
                         + (", ".join(f"{d:+.2f}%" for d in m["nondeterministic_repeat_difference_percent"]) or "—") + " |")
    lines += ["", "Interpretation and limits are recorded in the round summary; this study does not satisfy the 500-evaluation-user validation gate.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=1)[:3000])


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--iteration", required=True)
    p.add_argument("--template", required=True, help="Iteration whose frozen protocol supplies cohort, training and execution settings")
    p.add_argument("--stage", choices=("100", "250", "500"), required=True)
    p.add_argument("--models", nargs="+", choices=("baseline", "expertium"), required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--repeat-seeds", nargs="*", type=int, default=[])
    r = sub.add_parser("report")
    r.add_argument("--iteration", required=True)
    r.add_argument("--include", help='JSON {model: {track: result path}} for recorded runs from earlier iterations')
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else report(args)


if __name__ == "__main__":
    main()
