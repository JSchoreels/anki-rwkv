"""Synthetic R01 ratio/bound checks; no GPU or predictive-quality measurements."""

from argparse import ArgumentParser
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts.gpu_r01_statistics import METRICS, paired_relative_losses


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Preserve evidence; use a new artifact directory")
    output.mkdir(parents=True)
    result = {"provenance": process_provenance(), "synthetic_only": True, "checks": [], "status": "running"}
    settings = dict(resamples=1024, bootstrap_seed=20251012, family_tests=4, alpha=0.05, relative_limit=0.01)
    def check(name, value):
        if not value:
            raise AssertionError(name)
        result["checks"].append(name)
    def rows(values=(0.5, 0.5), labels=(1, 1)):
        return [{"user_id": index + 1, "curve_log_loss": value, "instant_log_loss": value,
                 "curve_labels": count, "instant_labels": count} for index, (value, count) in enumerate(zip(values, labels))]
    def compare(left, right):
        return paired_relative_losses(left, right, **settings)
    def reject(name, left, right, **overrides):
        try:
            paired_relative_losses(left, right, **{**settings, **overrides})
        except (ValueError, KeyError):
            check(name, True)
        else:
            raise AssertionError(name)
    try:
        base = rows()
        same = compare(base, list(reversed(base)))
        check("pairs by user identity rather than row order", same["all_four_strict_upper_bounds_below_limit"] and all(value["relative_delta"] == 0 for value in same["metrics"].values()))
        check("constant 0.9 percent worsening satisfies strict arithmetic bound", compare(rows((0.5045, 0.5045)), base)["all_four_strict_upper_bounds_below_limit"])
        check("one percent boundary does not satisfy strict less-than rule", not compare(rows((0.505, 0.505)), base)["all_four_strict_upper_bounds_below_limit"])
        mixed = compare(rows((0.11, 0.89), (1, 9)), rows((0.1, 0.9), (1, 9)))
        check("relative user average uses aggregate ratio, not average user percentages", mixed["metrics"]["curve_log_loss_user_average"]["relative_delta"] == 0)
        check("label-weighted ratio uses distinct aggregate denominator", math.isclose(mixed["metrics"]["curve_log_loss_label_weighted"]["relative_delta"], 8.12 / 8.2 - 1, abs_tol=1e-14))
        check("point estimate passing does not imply uncertainty bound passes", not mixed["all_four_strict_upper_bounds_below_limit"])
        changed = rows()
        changed[0]["instant_log_loss"] = 0.52
        individual = compare(changed, base)
        check("Instant regression cannot be hidden by Curve agreement", not individual["all_four_strict_upper_bounds_below_limit"] and individual["metrics"][METRICS[0]]["strict_upper_below_limit"])
        check("fixed bootstrap seed reproduces full output", compare(changed, base) == individual)
        reject("duplicate users rejected", [base[0], base[0]], base)
        reject("different user sets rejected", base[:1], base)
        bad = deepcopy(base); bad[0]["curve_labels"] += 1
        reject("paired label mismatch rejected", bad, base)
        bad = deepcopy(base); bad[0]["curve_labels"] = True
        reject("boolean reference label counts rejected", base, bad)
        bad = deepcopy(base); bad[0]["instant_log_loss"] = float("nan")
        reject("nonfinite loss rejected", bad, base)
        bad = deepcopy(base); del bad[0]["instant_log_loss"]
        reject("missing task cannot establish four-score bound", bad, base)
        reject("zero reference is unavailable rather than fabricated percent", base, rows((0, 0)))
        reject("invalid familywise settings rejected", base, base, family_tests=0)
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(result["checks"]), "synthetic_only": True}))


if __name__ == "__main__":
    main()
