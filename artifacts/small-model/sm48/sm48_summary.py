"""SM48 timing summary: per run and per arm, scoring-loop review-evals/s (the scorer's own counter, from the first
prepared user to the last row), process wall seconds (incl. imports, CUDA self-test, model build), GPU utilisation
(nvidia-smi, 0.5 s samples over the whole process) and the row compare vs run 1-old.
Writes timing/summary.json and timing/summary.md."""
import csv
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = HERE / "timing"
EVALS_PER_RUN = 2 * json.loads((HERE / "timing-users.json").read_text())["reviews"]   # 2 variants per user


def scorer_rate(log):
    lines = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if "review-evals/s" in line]
    if not lines:
        return None, None
    m = re.search(r"\| (\d+)s, ([\d,]+) review-evals/s", lines[-1])
    return int(m.group(1)), float(m.group(2).replace(",", ""))


def smi(path):
    util, mem = [], []
    try:
        for row in csv.reader(path.open()):
            if len(row) >= 3:
                try:
                    util.append(float(row[1]))
                    mem.append(float(row[2]))
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    return (round(statistics.mean(util), 1) if util else None, max(mem) if mem else None, len(util))


def main():
    runs = [json.loads(line) for line in (T / "runs.jsonl").read_text().splitlines()]
    out = []
    for run in runs:
        name = run["run"]
        _, rate = scorer_rate(T / f"run-{name}.log")
        util, mem, samples = smi(T / f"smi-{name}.csv")
        rows = [json.loads(line) for line in (T / f"run-{name}.jsonl").read_text().splitlines()]
        ok = [r for r in rows if "benchmark" in r]
        cmp_path = T / f"compare-{name}.json"
        cmp = json.loads(cmp_path.read_text()) if cmp_path.exists() else None
        out.append({**run, "rows": len(ok), "error_rows": len(rows) - len(ok),
                    "scorer_review_evals_per_s": rate,
                    "scoring_seconds": round(EVALS_PER_RUN / rate, 1) if rate else None,
                    "model_seconds_sum": round(sum(r["model_seconds"] for r in ok), 1),
                    "gpu_util_mean_pct": util, "gpu_mem_max_mib": mem, "smi_samples": samples,
                    "compare_vs_1_old": None if cmp is None else {k: cmp[k] for k in (
                        "pairs", "bitwise_equal_pairs", "passed", "max_abs_delta")}})
    arms = {}
    for r in out:
        arms.setdefault(r["arm"], []).append(r)
    base = statistics.mean(r["scorer_review_evals_per_s"] for r in arms["old"])
    summary = {"evals_per_run": EVALS_PER_RUN, "runs": out, "arms": {}}
    for arm, rs in arms.items():
        rate = statistics.mean(r["scorer_review_evals_per_s"] for r in rs)
        summary["arms"][arm] = {"flags": rs[0]["flags"], "runs": len(rs),
                                "review_evals_per_s": round(rate), "review_evals_per_s_runs": [
                                    r["scorer_review_evals_per_s"] for r in rs],
                                "speedup_vs_old": round(rate / base, 3),
                                "process_seconds": [r["process_seconds"] for r in rs],
                                "gpu_util_mean_pct": [r["gpu_util_mean_pct"] for r in rs],
                                "gpu_mem_max_mib": [r["gpu_mem_max_mib"] for r in rs]}
    (T / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = ["# SM48 timing: held-out scoring speed A/B", "",
             f"40 users of the 732 pool (artifacts/small-model/sm48/timing-users.json), SM36 w8+s8 + w8+s4c8n = "
             f"{EVALS_PER_RUN:,} review-evals per run; rate = the scorer's own counter (loop only); process = wall incl. "
             "startup; GPU util = nvidia-smi mean over the process.", "",
             "| Arm | Flags | Runs | review-evals/s (runs) | vs old | process s | GPU util % | GPU mem max MiB |",
             "|---|---|---:|---|---:|---|---|---|"]
    for arm, a in summary["arms"].items():
        lines.append(f"| {arm} | `{a['flags'] or 'default'}` | {a['runs']} | {a['review_evals_per_s']:,} "
                     f"({', '.join(f'{x:,.0f}' for x in a['review_evals_per_s_runs'])}) | {a['speedup_vs_old']:.2f}x | "
                     f"{', '.join(str(x) for x in a['process_seconds'])} | "
                     f"{', '.join(str(x) for x in a['gpu_util_mean_pct'])} | "
                     f"{', '.join(str(x) for x in a['gpu_mem_max_mib'])} |")
    lines += ["", "Row compare vs run 1-old (AGENT.md evaluation tolerance; bitwise = every compared value equal):", ""]
    for r in out:
        c = r["compare_vs_1_old"]
        if c:
            worst = {k: v for k, v in c["max_abs_delta"].items() if v}
            lines.append(f"- run {r['run']}: {c['bitwise_equal_pairs']}/{c['pairs']} pairs bitwise, passed {c['passed']}"
                         f"{', max deltas ' + json.dumps(worst) if worst else ''}; error rows {r['error_rows']}")
    (T / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
