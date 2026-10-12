"""Compare instrumented repeat 0046 with accepted 0042 and same-code-path 0045 (analysis only, no training)."""
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "artifacts/diagnostics/r01-round-0041-0045-review"))
from scripts.profile_gpu_resources import intervals, summarize
from scripts.r01_workflow import write_json
import segments  # noqa: E402  (equal-work console segments from the 0041-0045 review)

R = ROOT / "artifacts/retraining/gpu-retrain-r01"
CASES = {"0042": ("E33-F48-F50", R / "iteration-0042-expertium-console.log"),
         "0045": ("E33-F48-F50-F54", R / "iteration-0045-expertium-console.log"),
         "0046": ("E33-F48-F50", R / "iteration-0046-instrumented-repeat-console.log")}
PARITY = ("step", "user_id", "offset", "reviews", "seed", "loss", "gradient_norm", "curve_labels", "instant_labels", "lr")


def case(iteration):
    track, console = CASES[iteration]
    path = R / f"iteration-{iteration}/expertium/{track}"
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    updates = [json.loads(line) for line in (path / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [json.loads(line) for line in (path / "resources.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    weights = intervals(rows)
    total, tree, used = [], [], []
    for row, weight in zip(rows, weights):
        cpus = [p.get("cpu_percent_one_core_100") for p in row["processes"]]
        if row["phase"].startswith("training") and row["cpu_per_core_percent"] and cpus and None not in cpus:
            cores = len(row["cpu_per_core_percent"])
            total.append(sum(row["cpu_per_core_percent"]) / cores)
            tree.append(sum(cpus) / cores)
            used.append(weight)
    power = [json.loads(line) for line in (path / "gpu_power.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    busy = [r for r in power if r["utilization.gpu"] >= 20]
    state = {"busy_power_median_w": statistics.median(r["power.draw"] for r in busy),
             "busy_watts_per_utilization_point": statistics.median(r["power.draw"] / r["utilization.gpu"] for r in busy),
             "memory_used_max_mib": max(r["memory.used"] for r in power)}
    if "clocks.sm" in busy[0]:
        state.update(busy_sm_clock_median_mhz=statistics.median(r["clocks.sm"] for r in busy),
                     busy_sm_clock_min_mhz=min(r["clocks.sm"] for r in busy),
                     busy_pstates=sorted({r["pstate"] for r in busy}))
    background = json.loads((R / f"iteration-{iteration}/background-load.json").read_text(encoding="utf-8"))
    return {"result": result, "updates": updates, "segments": segments.segments(segments.curve(console)),
            "summary": {key: result.get(key) for key in ("reviews_per_second", "training_seconds", "validation_seconds",
                                                         "discarded_primer_seconds", "case_elapsed_seconds", "peak_reserved_mib",
                                                         "allocator_stats", "device_used_at_start_mib")}
            | {"preparation_seconds_sum": sum(u.get("preparation_seconds") or 0 for u in updates),
               "queue_wait_seconds_sum": sum(u.get("queue_wait_seconds") or 0 for u in updates),
               "prelaunch_background_cpu_percent": background.get("total_cpu_percent"),
               "training_total_host_cpu_percent": summarize(total, used)["mean"],
               "training_experiment_tree_cpu_percent": summarize(tree, used)["mean"],
               "training_background_cpu_residual_proxy_percent": summarize([a - b for a, b in zip(total, tree)], used)["mean"],
               "gpu_state": state}}


cases = {name: case(name) for name in CASES}
a, b = cases["0042"], cases["0046"]
out = {"parity_0046_vs_0042": {
           "updates_bitwise_equal": len(a["updates"]) == len(b["updates"]) and all(
               all(x[k] == y[k] for k in PARITY) for x, y in zip(a["updates"], b["updates"])),
           **{key: a["result"][key] == b["result"][key] for key in ("initial_parameter_sha256", "trained_parameter_sha256", "validation_input_sha256")},
           "scores_equal": a["result"]["evaluation"] == b["result"]["evaluation"]},
       "throughput_change_0046_vs_0042_percent": 100 * (b["result"]["reviews_per_second"] / a["result"]["reviews_per_second"] - 1),
       "segment_time_ratio_0046_vs_0042": [y / x for x, y in zip(a["segments"], b["segments"])],
       "segment_time_ratio_0045_vs_0042": [y / x for x, y in zip(a["segments"], cases["0045"]["segments"])],
       "cases": {name: value["summary"] for name, value in cases.items()},
       "limits": "0042 lacks per-update timing; segments come from 1-second console progress. 0045 adds F54 (exact, -0.34%). One run each; no causal attribution."}
write_json(Path(__file__).with_name("comparison.json"), out)
print(json.dumps({k: out[k] for k in ("parity_0046_vs_0042", "throughput_change_0046_vs_0042_percent")}))
for name, value in out["cases"].items():
    s = value
    print(name, f"rps {s['reviews_per_second']:.0f} prelaunch bg {s['prelaunch_background_cpu_percent']}% residual bg {s['training_background_cpu_residual_proxy_percent']:.1f}% "
          f"tree {s['training_experiment_tree_cpu_percent']:.1f}% W/util {s['gpu_state']['busy_watts_per_utilization_point']:.2f}")
