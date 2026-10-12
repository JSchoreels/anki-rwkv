"""Compare run-level and per-update gain estimators over the 0047 calibration replicates (analysis only)."""
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.r01_paired_timing import compare, load_run, trimmed_gain

R = ROOT / "artifacts/retraining/gpu-retrain-r01"
runs = [load_run(R / f"iteration-0047/replicates/r{i}/expertium/E33-F48-F50") for i in range(1, 7)]
pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
estimators = {"total wall": [], "median update wall": [], "trimmed 5 MAD wall": [], "trimmed 3 MAD wall": [],
              "total gpu span": [], "median update gpu span": []}
for i, j in pairs:
    c = compare([runs[i]], [runs[j]], resamples=200)
    estimators["total wall"].append(c["wall"]["gain_percent"])
    estimators["median update wall"].append(c["wall"]["median_update_gain_percent"])
    estimators["total gpu span"].append(c["gpu"]["gain_percent"])
    estimators["median update gpu span"].append(c["gpu"]["median_update_gain_percent"])
    estimators["trimmed 5 MAD wall"].append(trimmed_gain(runs[i]["wall"], runs[j]["wall"], 5)["gain_percent"])
    estimators["trimmed 3 MAD wall"].append(trimmed_gain(runs[i]["wall"], runs[j]["wall"], 3)["gain_percent"])
summary = {name: {"aa_pairs": len(values), "sd_percent": statistics.pstdev(values),
                  "mean_abs_percent": statistics.fmean(abs(v) for v in values)} for name, values in estimators.items()}
# Heterogeneous effect: speed up only the 65,536-review updates of r2 by 3% and compare against r1.
big = {k for k, key in enumerate(runs[0]["keys"]) if key[3] == 65536}
base = runs[1]["wall"]
modified = [w * (0.97 if k in big else 1.0) for k, w in enumerate(base)]
true_effect = 100 * (sum(base) / sum(modified) - 1)
recovered = {f"trimmed {t} MAD": trimmed_gain(runs[0]["wall"], modified, t)["gain_percent"] - trimmed_gain(runs[0]["wall"], base, t)["gain_percent"]
             for t in (3, 5)}
r2_r4 = compare([runs[1]], [runs[3]], resamples=200)["wall"]["segment_gain_percent"]
out = {"estimators": summary, "estimates": estimators,
       "heterogeneous_effect_check": {"true_total_effect_percent": true_effect, "recovered_percent": recovered,
                                      "note": "3% faster on the 87 large updates only; recovered = modified minus unmodified estimate on the same pair"},
       "r4_vs_r2_segment_gain_percent": r2_r4}
(Path(__file__).with_name("estimators.json")).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
for name, value in summary.items():
    print(f"{name:24s} SD {value['sd_percent']:.2f}%  mean|gain| {value['mean_abs_percent']:.2f}%")
print(f"heterogeneous check: true {true_effect:.2f}%, recovered {recovered}")
