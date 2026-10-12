"""Paired per-update timing between exact R01 runs (R01-M01; analysis only, no training).

Exact runs process identical updates in identical order, so update k of one run pairs with update k
of another. Each side may list several runs; the per-update minimum across them is the side's
best-of time, which removes transient spikes (the 2026-10-12 MAX suggestion). Pairing removes
transients and update-mix effects, not a whole-run slow machine state; the state flags below
(clocks, P-state, other-process VRAM, allocator retries) help recognise that case.

Per-update wall time is the interval between consecutive update starts (`host_started_s`); the
last update runs to `training_seconds`. `gpu_ms` is the device span between an update's start
and end events.
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
KEY = ("step", "user_id", "offset", "reviews", "seed")


def load_run(case):
    case = Path(case)
    result = json.loads((case / "result.json").read_text(encoding="utf-8"))
    updates = [json.loads(line) for line in (case / "updates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not updates or "host_started_s" not in updates[0]:
        raise ValueError(f"{case}: no per-update timing (runs before R01-M01 instrumentation cannot be paired)")
    starts = [u["host_started_s"] for u in updates]
    wall = [b - a for a, b in zip(starts, starts[1:])] + [result["training_seconds"] - starts[-1]]
    return {"case": case, "result": result, "keys": [tuple(u.get(k) for k in KEY) for u in updates],
            "wall": wall, "gpu": [u["gpu_ms"] / 1000 for u in updates], "lead_in": starts[0],
            "losses": [(u["loss"], u["gradient_norm"]) for u in updates], "state": machine_state(case, result)}


def machine_state(case, result):
    """Slow-state signals recorded alongside the run; descriptive, not causal."""
    state = {"device_used_at_start_mib": result.get("device_used_at_start_mib"),
             "peak_reserved_mib": result.get("peak_reserved_mib"),
             "allocator_stats": result.get("allocator_stats")}
    power = case / "gpu_power.jsonl"
    if power.exists():
        rows = [json.loads(line) for line in power.read_text(encoding="utf-8").splitlines() if line.strip()]
        busy = [r for r in rows if r.get("utilization.gpu", 0) >= 20]
        if busy:
            state["busy_samples"] = len(busy)
            state["busy_power_median_w"] = statistics.median(r["power.draw"] for r in busy)
            state["busy_watts_per_utilization_point"] = statistics.median(r["power.draw"] / r["utilization.gpu"] for r in busy)
            if "clocks.sm" in busy[0]:
                state["busy_sm_clock_median_mhz"] = statistics.median(r["clocks.sm"] for r in busy)
                state["busy_mem_clock_median_mhz"] = statistics.median(r["clocks.mem"] for r in busy)
                state["busy_pstates"] = sorted({r["pstate"] for r in busy})
                state["busy_clock_event_reasons"] = sorted({r["clocks_event_reasons.active"] for r in busy})
        state["memory_used_max_mib"] = max(r["memory.used"] for r in rows)
    return state


def best_of(runs, field):
    """Per-update minimum across runs of one side."""
    return [min(values) for values in zip(*(run[field] for run in runs))]


def compare(a_runs, b_runs, resamples=10000, seed=20261012, segments=8):
    keys = a_runs[0]["keys"]
    for run in a_runs + b_runs:
        if run["keys"] != keys:
            raise ValueError(f"{run['case']}: update sequence differs; pairing requires identical updates")
    exact = all(run["losses"] == a_runs[0]["losses"] for run in a_runs + b_runs)
    out = {"updates": len(keys), "exact_update_values": exact,
           "a": [run["case"].as_posix() for run in a_runs], "b": [run["case"].as_posix() for run in b_runs]}
    for field in ("wall", "gpu"):
        a, b = best_of(a_runs, field), best_of(b_runs, field)
        ratios = [y / x for x, y in zip(a, b) if x > 0 and y > 0]
        rng = random.Random(seed)
        n = len(a)
        boot = []
        for _ in range(resamples):
            idx = [rng.randrange(n) for _ in range(n)]
            boot.append(sum(b[i] for i in idx) / sum(a[i] for i in idx))
        boot.sort()
        reviews = [k[3] for k in keys]
        total, edges, acc = sum(reviews), [], 0
        for i, r in enumerate(reviews):
            acc += r
            if acc >= total * (len(edges) + 1) / segments and len(edges) < segments - 1:
                edges.append(i + 1)
        cuts = [0, *edges, n]
        seg = [sum(b[i:j]) / sum(a[i:j]) for i, j in zip(cuts, cuts[1:]) if j > i]
        out[field] = {"a_seconds": sum(a), "b_seconds": sum(b),
                      # Gain is B over A in reviews/s terms: positive means B is faster.
                      "gain_percent": 100 * (sum(a) / sum(b) - 1),
                      "median_update_gain_percent": 100 * (1 / statistics.median(ratios) - 1),
                      "gain_ci95_percent": [100 * (1 / boot[int(0.975 * resamples)] - 1), 100 * (1 / boot[int(0.025 * resamples)] - 1)],
                      "segment_gain_percent": [100 * (1 / r - 1) for r in seg]}
    out["lead_in_seconds"] = {"a": [r["lead_in"] for r in a_runs], "b": [r["lead_in"] for r in b_runs]}
    out["machine_state"] = {run["case"].as_posix(): run["state"] for run in a_runs + b_runs}
    out["note"] = ("CI resamples updates within these runs; it does not cover a whole-run machine-state shift, "
                   "which needs repeated or interleaved runs. Best-of sides use the per-update minimum.")
    return out


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--a", nargs="+", required=True, help="Reference case directories (best-of when several)")
    parser.add_argument("--b", nargs="+", required=True, help="Candidate case directories (best-of when several)")
    parser.add_argument("--output", type=Path, help="Optional JSON output under artifacts/")
    args = parser.parse_args()
    a_runs, b_runs = [load_run(ROOT / p) for p in args.a], [load_run(ROOT / p) for p in args.b]
    for run in a_runs + b_runs:
        if run["case"].resolve().is_relative_to(ROOT):
            run["case"] = run["case"].resolve().relative_to(ROOT)
    out = compare(a_runs, b_runs)
    if args.output:
        target = (ROOT / args.output).resolve()
        if not target.is_relative_to(ROOT / "artifacts"):
            parser.error("Write analysis outputs under artifacts/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for field in ("wall", "gpu"):
        f = out[field]
        print(f"{field}: B vs A gain {f['gain_percent']:+.2f}% (CI {f['gain_ci95_percent'][0]:+.2f}..{f['gain_ci95_percent'][1]:+.2f}), "
              f"median update {f['median_update_gain_percent']:+.2f}%, segments " + " ".join(f"{g:+.1f}" for g in f["segment_gain_percent"]))
    print(f"updates {out['updates']}, exact update values {out['exact_update_values']}")


if __name__ == "__main__":
    main()
