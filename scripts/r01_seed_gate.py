"""P-B multi-seed non-inferiority screen for non-exact R01 candidates (opt-in, pending approval).

Proposal P-B in docs/R01_NEXT_ROUND_PLAN.md would replace the single-seed 0.1% loss
ceiling for non-exact candidates. Until the human approves P-B and sets the margin,
this tool only reports; it never changes a frozen protocol, the active gate, a
promotion decision or a round budget, and it launches nothing.

Inputs are N>=3 candidate and N>=3 reference complete full-history result.json files
on the same cohort (identical held-out inputs and exposure) that differ in training
seed. Each input is a path or a `store:<key>` token resolved through
scripts/r01_result_store.py, so reference seed runs are trained once per accepted
reference and reused across candidates.

Statistics (relative fractions, 0.01 = 1%): for each of the four held-out losses the
point estimate is the ratio of seed-averaged aggregates minus one. The decision bound
is a two-level bootstrap that resamples held-out users (paired across sides) and
training seeds within each side (shared seed draws when both sides used the same
seeds), with the conventions of scripts/gpu_r01_statistics.py: seed 20261012, 10,000
resamples, alpha 0.05, familywise upper quantile 1 - alpha/4 over the four metrics.
A user-only bootstrap (seeds fixed) and a seed-level paired t bound are reported as
sensitivities; with three seeds the seed-level bound is wide and the two-level
bootstrap understates seed variance by about (N-1)/N.
"""

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METRICS = tuple(task + "_log_loss_" + kind for task in ("curve", "instant")
                for kind in ("label_weighted", "user_average"))
DEFAULTS = {"resamples": 10000, "bootstrap_seed": 20261012, "alpha": 0.05, "family_tests": 4, "minimum_seeds": 3}
COHORT_FIELDS = ("training_users", "evaluation_users", "validation_input_sha256", "updates", "review_presentations")
SEED_CONFIG = ("SEED",)
WORKER_CONFIG = ("PREFETCH_WORKERS", "PREFETCH_BATCHES")


def resolve(token):
    """Return (path, store_key or None); a store token fails closed without the store."""
    if not token.startswith("store:"):
        path = Path(token)
        return (path if path.is_absolute() else ROOT / path).resolve(), None
    key = token[len("store:"):]
    try:
        from scripts import r01_result_store
    except ImportError as error:
        raise SystemExit(f"{token}: scripts/r01_result_store.py is unavailable ({error}); pass explicit result.json paths") from error
    if hasattr(r01_result_store, "resolve"):
        return Path(r01_result_store.resolve(key)).resolve(), key
    # Current store layout: the token names an entry file (result-store/<kind>/<xx>/<key>/<sha16>.json);
    # the store's own valid() must accept it (result hash unchanged, complete status, runtime manifest).
    entry_path = (ROOT / key).resolve()
    if not (hasattr(r01_result_store, "valid") and hasattr(r01_result_store, "read_json")) or not entry_path.is_file():
        raise SystemExit(f"{token}: not a result-store entry file; pass explicit result.json paths")
    entry = r01_result_store.read_json(entry_path)
    reasons = r01_result_store.valid(ROOT, entry)
    if reasons:
        raise SystemExit(f"{token}: store refused reuse: {'; '.join(reasons)}")
    return (ROOT / entry["value"]["result"]).resolve(), key


def load(token):
    path, key = resolve(token)
    if path.is_dir():
        path = path / "result.json"
    case = json.loads(path.read_text(encoding="utf-8"))
    if case.get("status") != "complete" or case.get("smoke_only"):
        raise ValueError(f"{path}: not a complete full-history run")
    for field in ("per_user", "evaluation", "options", "config", *COHORT_FIELDS):
        if field not in case:
            raise ValueError(f"{path}: missing {field}")
    try:
        label = path.relative_to(ROOT).as_posix()
    except ValueError:
        label = path.as_posix()
    return {"path": label, "store_key": key, "case": case,
            "seed": case["options"].get("training_seed", 2025)}


def without(mapping, keys):
    return {k: v for k, v in mapping.items() if k not in keys}


def check_side(name, runs, minimum, allow_duplicates):
    if len(runs) < minimum:
        raise ValueError(f"{name}: {len(runs)} runs; at least {minimum} training seeds are required")
    seeds = [run["seed"] for run in runs]
    if len(set(seeds)) != len(seeds) and not allow_duplicates:
        raise ValueError(f"{name}: duplicate training seeds {seeds}; same-seed replicates measure execution noise, not seed spread")
    first = runs[0]["case"]
    for run in runs[1:]:
        case = run["case"]
        if without(case["options"], ("training_seed",)) != without(first["options"], ("training_seed",)):
            raise ValueError(f"{name}: options differ beyond training_seed ({run['path']})")
        if without(case["config"], SEED_CONFIG + WORKER_CONFIG) != without(first["config"], SEED_CONFIG + WORKER_CONFIG):
            raise ValueError(f"{name}: config differs beyond the seed ({run['path']})")
        if case.get("implementation_sha256") != first.get("implementation_sha256"):
            raise ValueError(f"{name}: implementation differs across seeds ({run['path']})")
    return seeds


def matrices(candidates, references):
    """Per-task (mass, candidate[k,u], reference[j,u]); checks cohort and label pairing."""
    everything = candidates + references
    base = everything[0]["case"]
    for run in everything[1:]:
        for field in COHORT_FIELDS:
            if run["case"][field] != base[field]:
                raise ValueError(f"Cohort/exposure mismatch in {field}: {run['path']} vs {everything[0]['path']}")
    users = sorted(row["user_id"] for row in base["per_user"])
    if len(set(users)) != len(users) or any(type(u) is not int or u <= 0 for u in users):
        raise ValueError("Distinct positive integer user IDs required")
    rows = []
    for run in everything:
        indexed = {row["user_id"]: row for row in run["case"]["per_user"]}
        if sorted(indexed) != users or len(indexed) != len(run["case"]["per_user"]):
            raise ValueError(f"{run['path']}: held-out user set differs")
        rows.append(indexed)
    arrays = {}
    for task in ("curve", "instant"):
        labels = np.asarray([rows[0][u][task + "_labels"] for u in users], dtype=float)
        losses = np.empty((len(everything), len(users)))
        for i, (run, indexed) in enumerate(zip(everything, rows)):
            for j, user in enumerate(users):
                row = indexed[user]
                if type(row[task + "_labels"]) is not int or row[task + "_labels"] <= 0 or row[task + "_labels"] != labels[j]:
                    raise ValueError(f"{run['path']}: paired positive {task} label counts required (user {user})")
                value = row[task + "_log_loss"]
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{run['path']}: finite positive {task} loss required (user {user})")
                losses[i, j] = value
            # Self-check: per-user rows must reproduce the recorded aggregates.
            for kind, mass in (("label_weighted", labels), ("user_average", np.ones(len(users)))):
                recorded = run["case"]["evaluation"][task + "_log_loss_" + kind]
                recomputed = float(np.average(losses[i], weights=mass))
                if not math.isclose(recomputed, recorded, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(f"{run['path']}: per-user {task} rows do not reproduce {kind} aggregate")
        arrays[task] = (labels, losses[:len(candidates)], losses[len(candidates):])
    return users, arrays


def aggregate(losses, mass):
    """Per-seed aggregate losses for one metric."""
    return losses @ mass / mass.sum()


def bootstrap(arrays, n_c, n_r, paired, *, resamples, seed, resample_seeds):
    rng = np.random.default_rng(seed)
    users = next(iter(arrays.values()))[1].shape[1]
    samples = {key: [] for key in METRICS}
    for start in range(0, resamples, 256):
        size = min(256, resamples - start)
        index = rng.integers(0, users, (size, users))
        if resample_seeds:
            draw_c = rng.integers(0, n_c, (size, n_c))
            draw_r = draw_c if paired else rng.integers(0, n_r, (size, n_r))
            weight_c = np.stack([np.bincount(row, minlength=n_c) for row in draw_c]) / n_c
            weight_r = np.stack([np.bincount(row, minlength=n_r) for row in draw_r]) / n_r
        else:
            weight_c = np.full((size, n_c), 1 / n_c)
            weight_r = np.full((size, n_r), 1 / n_r)
        for task, (labels, cand, ref) in arrays.items():
            mix_c = np.take_along_axis(weight_c @ cand, index, axis=1)
            mix_r = np.take_along_axis(weight_r @ ref, index, axis=1)
            for kind in ("label_weighted", "user_average"):
                mass = labels[index] if kind == "label_weighted" else np.ones(index.shape)
                samples[task + "_log_loss_" + kind].append((mix_c * mass).sum(axis=1) / (mix_r * mass).sum(axis=1) - 1)
    return {key: np.concatenate(value) for key, value in samples.items()}


def spread(values):
    mean = statistics.mean(values)
    relative = [value / mean - 1 for value in values]
    return {"mean": mean, "seed_sd_relative": statistics.stdev(relative) if len(relative) > 1 else None,
            "seed_range_relative": [min(relative), max(relative)]}


def evaluate(candidates, references, *, margin, resamples, bootstrap_seed, alpha, family_tests,
             minimum_seeds=3, allow_duplicate_seeds=False):
    if not (isinstance(margin, float) and math.isfinite(margin) and 0 < margin < 0.2):
        raise ValueError("Margin must be a relative fraction in (0, 0.2), set explicitly by the human")
    if type(resamples) is not int or resamples < 1000 or not 0 < alpha < 1 or type(family_tests) is not int or family_tests < 4:
        raise ValueError("Invalid bootstrap settings")
    seeds_c = check_side("candidate", candidates, minimum_seeds, allow_duplicate_seeds)
    seeds_r = check_side("reference", references, minimum_seeds, allow_duplicate_seeds)
    degenerate = len(set(seeds_c)) != len(seeds_c) or len(set(seeds_r)) != len(seeds_r)
    paired = not degenerate and sorted(seeds_c) == sorted(seeds_r)
    if paired:
        candidates = sorted(candidates, key=lambda run: run["seed"])
        references = sorted(references, key=lambda run: run["seed"])
    users, arrays = matrices(candidates, references)
    two_level = bootstrap(arrays, len(candidates), len(references), paired,
                          resamples=resamples, seed=bootstrap_seed, resample_seeds=True)
    user_only = bootstrap(arrays, len(candidates), len(references), paired,
                          resamples=resamples, seed=bootstrap_seed, resample_seeds=False)
    from scipy.stats import t as student
    level = 1 - alpha / family_tests
    metrics = {}
    for task, (labels, cand, ref) in arrays.items():
        for kind in ("label_weighted", "user_average"):
            key = task + "_log_loss_" + kind
            mass = labels if kind == "label_weighted" else np.ones(len(users))
            per_c, per_r = aggregate(cand, mass), aggregate(ref, mass)
            point = float(per_c.mean() / per_r.mean() - 1)
            upper = float(np.quantile(two_level[key], level))
            item = {"candidate_seed_mean": float(per_c.mean()), "reference_seed_mean": float(per_r.mean()),
                    "relative_delta": point,
                    "relative_ci95_two_level": [float(v) for v in np.quantile(two_level[key], [0.025, 0.975])],
                    "relative_familywise_upper": upper,
                    "relative_familywise_upper_user_only": float(np.quantile(user_only[key], level)),
                    "reference_seed_spread": spread([float(v) for v in per_r]),
                    "candidate_seed_spread": spread([float(v) for v in per_c]),
                    "upper_below_margin": upper < margin}
            if paired:
                deltas = [float(c / r - 1) for c, r in zip(per_c, per_r)]
                sd = statistics.stdev(deltas)
                item["paired_seed_deltas"] = deltas
                item["seed_level_t_familywise_upper"] = statistics.mean(deltas) + float(student.ppf(level, len(deltas) - 1)) * sd / math.sqrt(len(deltas))
            metrics[key] = item
    reference_sd = [m["reference_seed_spread"]["seed_sd_relative"] for m in metrics.values()]
    passed = all(m["upper_below_margin"] for m in metrics.values())
    if degenerate:
        decision = "degenerate input: same-seed replicates measure execution noise, not seed spread; no gate decision"
    else:
        decision = ("non-inferior on all four losses at the margin (P-B pending approval; informational only)" if passed
                    else "non-inferiority not shown (P-B pending approval; informational only)")

    def cost(runs):
        return {"runs": len(runs), "from_store": sum(run["store_key"] is not None for run in runs),
                "training_seconds": sum(run["case"]["training_seconds"] for run in runs),
                "process_wall_seconds": sum(run["case"].get("process_wall_seconds") or 0 for run in runs)}

    run_cost = {"candidate": cost(candidates), "reference": cost(references)}
    run_cost["full_runs_total"] = len(candidates) + len(references)
    run_cost["full_runs_new_for_this_candidate"] = len(candidates) + len(references) - run_cost["reference"]["from_store"]
    config_differences = sorted(k for k in set(candidates[0]["case"]["config"]) | set(references[0]["case"]["config"])
                                if k not in SEED_CONFIG + WORKER_CONFIG
                                and candidates[0]["case"]["config"].get(k) != references[0]["case"]["config"].get(k))
    option_differences = sorted(k for k in set(candidates[0]["case"]["options"]) | set(references[0]["case"]["options"])
                                if k != "training_seed"
                                and candidates[0]["case"]["options"].get(k) != references[0]["case"]["options"].get(k))
    return {"scope": "P-B multi-seed non-inferiority screen; pending human approval; not promotion, not broader validation",
            "status": "informational (P-B inactive)", "decision": decision, "all_four_upper_below_margin": passed and not degenerate,
            "margin_relative": margin, "margin_over_max_reference_seed_sd": (margin / max(reference_sd)) if all(reference_sd) else None,
            "settings": {"resamples": resamples, "bootstrap_seed": bootstrap_seed, "alpha": alpha, "family_tests": family_tests,
                         "upper_quantile": level, "decision_bound": "two-level bootstrap (users paired; seeds within side)",
                         "seed_pairing": "shared seed draws (identical seed sets)" if paired else "independent seed draws"},
            "paired_users": len(users), "candidate_seeds": [r["seed"] for r in candidates], "reference_seeds": [r["seed"] for r in references],
            "degenerate_same_seed": degenerate, "option_differences": option_differences, "config_differences": config_differences,
            "metrics": metrics, "run_cost": run_cost,
            "inputs": {side: [{"path": r["path"], "store_key": r["store_key"], "seed": r["seed"],
                               "implementation_sha256": r["case"].get("implementation_sha256"),
                               "git_commit": (r["case"].get("provenance") or {}).get("git_commit"),
                               "trained_parameter_sha256": r["case"].get("trained_parameter_sha256")}
                              for r in runs] for side, runs in (("candidate", candidates), ("reference", references))}}


def brief(report):
    lines = [f"{report['decision']} | margin {100 * report['margin_relative']:.3f}% | users {report['paired_users']} | "
             f"seeds C{report['candidate_seeds']} R{report['reference_seeds']} | runs {report['run_cost']['full_runs_total']} "
             f"(new for this candidate {report['run_cost']['full_runs_new_for_this_candidate']})"]
    for key, m in report["metrics"].items():
        sd = m["reference_seed_spread"]["seed_sd_relative"]
        seed_t = f", seed-t {100 * m['seed_level_t_familywise_upper']:+.3f}%" if "seed_level_t_familywise_upper" in m else ""
        sd_text = f"{100 * sd:.3f}%" if sd is not None else "n/a"
        lines.append(f"{key}: delta {100 * m['relative_delta']:+.3f}% upper {100 * m['relative_familywise_upper']:+.3f}% "
                     f"(user-only {100 * m['relative_familywise_upper_user_only']:+.3f}%{seed_t}) ref seed SD {sd_text}")
    return "\n".join(lines)


def main():
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--candidate", action="append", required=True, help="candidate result.json, case dir or store:<key>; repeat per seed")
    parser.add_argument("--reference", action="append", required=True, help="reference result.json, case dir or store:<key>; repeat per seed")
    parser.add_argument("--margin", type=float, required=True, help="relative non-inferiority margin (0.005 = 0.5%%); set by the human, no default")
    parser.add_argument("--resamples", type=int, default=DEFAULTS["resamples"])
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULTS["bootstrap_seed"])
    parser.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    parser.add_argument("--allow-duplicate-seeds", action="store_true", help="degenerate same-seed check only; never a gate decision")
    parser.add_argument("--output", type=Path, help="write the full JSON report here (with tool provenance)")
    args = parser.parse_args()
    report = evaluate([load(t) for t in args.candidate], [load(t) for t in args.reference], margin=args.margin,
                      resamples=args.resamples, bootstrap_seed=args.bootstrap_seed, alpha=args.alpha,
                      family_tests=DEFAULTS["family_tests"], minimum_seeds=DEFAULTS["minimum_seeds"],
                      allow_duplicate_seeds=args.allow_duplicate_seeds)
    if args.output:
        from scripts.experiment_provenance import process_provenance
        report = {"created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv, "provenance": process_provenance(), **report}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(brief(report))


if __name__ == "__main__":
    main()
