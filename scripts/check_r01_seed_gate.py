"""CPU checks for scripts/r01_seed_gate.py (P-B, pending approval); no GPU, no runs launched.

Synthetic cases check the arithmetic, the fail-closed input rules and the store hook.
Recorded cases cross-check against the iteration-0014 F22 seed study (three paired
training seeds per side, compared with its independently written study-result.json)
and the iteration-0047 same-seed replicates (degenerate: bitwise-equal losses).
"""

from argparse import ArgumentParser
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts import r01_seed_gate as gate

R = ROOT / "artifacts/retraining/gpu-retrain-r01"
FAST = dict(resamples=2000, bootstrap_seed=20261012, alpha=0.05, family_tests=4)


def synthetic_case(seed, scale, *, users=12, seed_noise=None, hash_="v" * 64, options=None):
    rng = np.random.default_rng(1000 + seed)
    noise = seed_noise if seed_noise is not None else 1 + 0.005 * rng.standard_normal()
    rows = []
    base = np.random.default_rng(7)
    labels = base.integers(50, 5000, users)
    curve = 0.3 + 0.2 * base.random(users)
    instant = 0.5 + 0.2 * base.random(users)
    for i in range(users):
        user_noise = 1 + 0.01 * rng.standard_normal(2)
        rows.append({"user_id": i + 1, "curve_labels": int(labels[i]), "instant_labels": int(labels[i]),
                     "curve_log_loss": float(curve[i] * noise * user_noise[0] * scale),
                     "instant_log_loss": float(instant[i] * noise * user_noise[1] * scale)})
    evaluation = {}
    for task in ("curve", "instant"):
        values = np.asarray([r[task + "_log_loss"] for r in rows])
        evaluation[task + "_log_loss_label_weighted"] = float(np.average(values, weights=labels))
        evaluation[task + "_log_loss_user_average"] = float(values.mean())
    return {"status": "complete", "smoke_only": False, "per_user": rows, "evaluation": evaluation,
            "options": {"track_option": True, **(options or {}), "training_seed": seed},
            "config": {"SEED": seed, "LR": 0.001}, "implementation_sha256": "i" * 64,
            "training_users": 80, "evaluation_users": users, "validation_input_sha256": hash_,
            "updates": 100, "review_presentations": 1000, "training_seconds": 50.0, "process_wall_seconds": 90.0}


def main():
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Preserve evidence; use a new artifact directory")
    output.mkdir(parents=True)
    result = {"provenance": process_provenance(), "checks": [], "status": "running"}

    def check(name, value):
        if not value:
            raise AssertionError(name)
        result["checks"].append(name)

    def rejects(name, function, error=ValueError):
        try:
            function()
        except error:
            result["checks"].append(name)
            return
        raise AssertionError(name)

    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)

        def runs(cases, side):
            loaded = []
            for i, case in enumerate(cases):
                path = folder / f"{side}-{i}-{id(case)}.json"
                path.write_text(json.dumps(case), encoding="utf-8")
                loaded.append(gate.load(str(path)))
            return loaded

        seeds = (11, 12, 13)
        reference = [synthetic_case(s, 1.0) for s in seeds]
        # Constant paired ratio: every resample ratio equals the shift, so all bounds are exact.
        shifted = deepcopy(reference)
        for case in shifted:
            for row in case["per_user"]:
                row["curve_log_loss"] *= 1.002
                row["instant_log_loss"] *= 1.002
            for key in case["evaluation"]:
                case["evaluation"][key] *= 1.002
        report = gate.evaluate(runs(shifted, "c"), runs(reference, "r"), margin=0.01, **FAST)
        check("constant shift: paired seed pairing", report["settings"]["seed_pairing"].startswith("shared"))
        check("constant shift: point, two-level and user-only bounds equal +0.2%",
              all(math.isclose(m[k], 0.002, rel_tol=1e-9) for m in report["metrics"].values()
                  for k in ("relative_delta", "relative_familywise_upper", "relative_familywise_upper_user_only")))
        check("constant shift: passes a 1% margin", report["all_four_upper_below_margin"])
        check("constant shift: fails a 0.1% margin",
              not gate.evaluate(runs(shifted, "c"), runs(reference, "r"), margin=0.001, **FAST)["all_four_upper_below_margin"])
        check("run cost counts six full runs, none from store",
              report["run_cost"]["full_runs_total"] == 6 and report["run_cost"]["full_runs_new_for_this_candidate"] == 6)

        # Seed-to-seed noise widens the two-level bound beyond the user-only bound.
        noisy = [synthetic_case(s, 1.0, seed_noise=1 + d) for s, d in zip((21, 22, 23), (-0.01, 0.0, 0.012))]
        noisy_ref = [synthetic_case(s, 1.0, seed_noise=1 + d) for s, d in zip((21, 22, 23), (0.004, -0.006, 0.002))]
        report = gate.evaluate(runs(noisy, "c"), runs(noisy_ref, "r"), margin=0.05, **FAST)
        check("seed noise: two-level upper exceeds user-only upper",
              all(m["relative_familywise_upper"] > m["relative_familywise_upper_user_only"] for m in report["metrics"].values()))
        check("seed noise: reference seed SD reported", all(m["reference_seed_spread"]["seed_sd_relative"] > 0 for m in report["metrics"].values()))
        check("seed noise: seed-level t bound reported for paired seeds", all("seed_level_t_familywise_upper" in m for m in report["metrics"].values()))
        again = gate.evaluate(runs(noisy, "c"), runs(noisy_ref, "r"), margin=0.05, **FAST)
        check("bootstrap is deterministic for a fixed seed", again["metrics"] == report["metrics"])

        worse = [synthetic_case(s, 1.03) for s in seeds]
        check("3% regression fails a 1% margin",
              not gate.evaluate(runs(worse, "c"), runs(reference, "r"), margin=0.01, **FAST)["all_four_upper_below_margin"])
        unpaired = [synthetic_case(s, 1.0) for s in (31, 32, 33)]
        check("different seed sets use independent seed draws",
              gate.evaluate(runs(unpaired, "c"), runs(reference, "r"), margin=0.01, **FAST)["settings"]["seed_pairing"].startswith("independent"))

        rejects("two seeds rejected", lambda: gate.evaluate(runs(reference[:2], "c"), runs(reference, "r"), margin=0.01, **FAST))
        rejects("duplicate seeds rejected", lambda: gate.evaluate(runs([reference[0]] * 3, "c"), runs(reference, "r"), margin=0.01, **FAST))
        rejects("missing/invalid margin rejected", lambda: gate.evaluate(runs(reference, "c"), runs(reference, "r"), margin=0.0, **FAST))
        rejects("held-out input hash mismatch rejected",
                lambda: gate.evaluate(runs([synthetic_case(s, 1.0, hash_="w" * 64) for s in seeds], "c"), runs(reference, "r"), margin=0.01, **FAST))
        mixed = [synthetic_case(11, 1.0), synthetic_case(12, 1.0, options={"other": 1}), synthetic_case(13, 1.0)]
        rejects("options differing within a side rejected", lambda: gate.evaluate(runs(mixed, "c"), runs(reference, "r"), margin=0.01, **FAST))
        relabelled = deepcopy(reference)
        relabelled[1]["per_user"][0]["curve_labels"] += 1
        rejects("paired label counts enforced", lambda: gate.evaluate(runs(reference, "c"), runs(relabelled, "r"), margin=0.01, **FAST))
        tampered = deepcopy(reference)
        tampered[0]["evaluation"]["curve_log_loss_user_average"] *= 1.01
        rejects("per-user rows must reproduce recorded aggregates", lambda: gate.evaluate(runs(tampered, "c"), runs(reference, "r"), margin=0.01, **FAST))
        smoke = deepcopy(reference[0])
        smoke["smoke_only"] = True
        rejects("smoke runs rejected", lambda: runs([smoke], "c"))

        # Store hook: fails closed without resolve(); counts reused reference runs when present.
        saved = sys.modules.get("scripts.r01_result_store")
        try:
            sys.modules["scripts.r01_result_store"] = types.ModuleType("scripts.r01_result_store")
            rejects("store token without resolve() fails closed", lambda: gate.load("store:any"), SystemExit)
            paths = {f"ref-{s}": folder / f"store-{s}.json" for s in seeds}
            for case, (key, path) in zip(reference, paths.items()):
                path.write_text(json.dumps(case), encoding="utf-8")
            fake = types.ModuleType("scripts.r01_result_store")
            fake.resolve = lambda key: paths[key]
            sys.modules["scripts.r01_result_store"] = fake
            stored = [gate.load(f"store:{key}") for key in paths]
            report = gate.evaluate(runs(shifted, "c"), stored, margin=0.01, **FAST)
            check("store references counted as reused",
                  report["run_cost"]["reference"]["from_store"] == 3 and report["run_cost"]["full_runs_new_for_this_candidate"] == 3)
            # Entry-file adapter for the store layout without resolve(): valid() decides reuse.
            entry_file = folder / "entry.json"
            entry_file.write_text(json.dumps({"value": {"result": str(paths["ref-11"])}}), encoding="utf-8")
            adapter = types.ModuleType("scripts.r01_result_store")
            adapter.read_json = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
            adapter.valid = lambda root, entry: []
            sys.modules["scripts.r01_result_store"] = adapter
            check("store entry file resolves to its recorded result", gate.load(f"store:{entry_file}")["seed"] == 11)
            adapter.valid = lambda root, entry: ["result file changed since it was indexed"]
            rejects("store entry refused by valid() fails closed", lambda: gate.load(f"store:{entry_file}"), SystemExit)
        finally:
            if saved is None:
                sys.modules.pop("scripts.r01_result_store", None)
            else:
                sys.modules["scripts.r01_result_store"] = saved

    # Recorded F22 seed study (iteration 0014): three paired seeds per side.
    study = R / "iteration-0014/expertium"
    candidates = [gate.load(str(study / f"S{s}-E06-E11-F22")) for s in (26, 27, 28)]
    references = [gate.load(str(study / f"S{s}-E06-E11-F26-F27")) for s in (26, 27, 28)]
    recorded = gate.evaluate(candidates, references, margin=0.01, resamples=10000, bootstrap_seed=20261012, alpha=0.05, family_tests=4)
    (output / "recorded-0014-f22-margin-1pct.json").write_text(json.dumps(recorded, indent=2) + "\n", encoding="utf-8")
    independent = json.loads((R / "iteration-0014/study-result.json").read_text(encoding="utf-8"))["summary"]["expertium"]
    position = {seed: i for i, seed in enumerate(independent["paired_seeds"])}
    check("0014: paired per-seed deltas match the recorded study",
          all(math.isclose(100 * d, independent["metrics"][key]["paired_nondeterministic_minus_deterministic_percent"][position[seed]], rel_tol=1e-9, abs_tol=1e-12)
              for key, m in recorded["metrics"].items() for d, seed in zip(m["paired_seed_deltas"], (2026, 2027, 2028))))
    check("0014: reference seed means match recorded aggregates",
          all(math.isclose(m["reference_seed_mean"], sum(r["case"]["evaluation"][key] for r in references) / 3, rel_tol=1e-12)
              for key, m in recorded["metrics"].items()))
    check("0014: 20 paired held-out users", recorded["paired_users"] == 20)

    # Same-seed replicates (iteration 0047): degenerate check only.
    replicas = [gate.load(str(R / f"iteration-0047/replicates/r{i}/expertium/E33-F48-F50")) for i in range(1, 7)]
    rejects("0047: same-seed replicates rejected without the degenerate flag",
            lambda: gate.evaluate(replicas[:3], replicas[3:], margin=0.01, **FAST))
    degenerate = gate.evaluate(replicas[:3], replicas[3:], margin=0.01, allow_duplicate_seeds=True, **FAST)
    check("0047: degenerate input yields no gate decision", degenerate["degenerate_same_seed"] and not degenerate["all_four_upper_below_margin"])
    check("0047: exact replicas give zero delta and zero bounds",
          all(m["relative_delta"] == 0 and m["relative_familywise_upper"] == 0 for m in degenerate["metrics"].values()))

    result.update(status="passed", recorded_0014_brief=gate.brief(recorded).splitlines())
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"passed {len(result['checks'])} checks; evidence {output.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
