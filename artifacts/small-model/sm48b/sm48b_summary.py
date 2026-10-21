"""SM48b timing summary: per run review-evals/s (scorer counter), process seconds, GPU util / memory (nvidia-smi),
torch peak allocated/reserved (scorer's last line) and the row compare vs run 1 (lanes 1). Writes timing/summary.{json,md}."""
import csv
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = HERE / "timing"
EVALS = 4 * json.loads((HERE / "timing-users.json").read_text())["reviews"]   # 4 variants per user


def log_facts(log):
    text = log.read_text(encoding="utf-8", errors="replace")
    rates = re.findall(r"\| (\d+)s, ([\d,]+) review-evals/s", text)
    peak = re.search(r"peak torch allocated ([\d.]+) GiB, reserved ([\d.]+) GiB", text)
    return (float(rates[-1][1].replace(",", "")) if rates else None,
            float(peak.group(1)) if peak else None, float(peak.group(2)) if peak else None)


def smi(path):
    util, mem = [], []
    try:
        for row in csv.reader(path.open()):
            if len(row) >= 3:
                try:
                    util.append(float(row[1])); mem.append(float(row[2]))
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    return (round(statistics.mean(util), 1) if util else None, max(mem) if mem else None)


def main():
    out = []
    for run in map(json.loads, (T / "runs.jsonl").read_text().splitlines()):
        name = run["run"]
        rate, alloc, resv = log_facts(T / f"run-{name}.log")
        util, mem = smi(T / f"smi-{name}.csv")
        rows = [json.loads(line) for line in (T / f"run-{name}.jsonl").read_text().splitlines()]
        ok = [r for r in rows if "benchmark" in r]
        cp = T / f"compare-{name}.json"
        cmp = json.loads(cp.read_text()) if cp.exists() else None
        out.append({**run, "rows": len(ok), "error_rows": len(rows) - len(ok), "review_evals_per_s": rate,
                    "torch_peak_allocated_gib": alloc, "torch_peak_reserved_gib": resv,
                    "gpu_util_mean_pct": util, "gpu_mem_max_mib": mem,
                    "compare_vs_run1": None if cmp is None else {k: cmp[k] for k in (
                        "pairs", "bitwise_equal_pairs", "passed", "max_abs_delta", "only_a", "only_b")}})
    base = statistics.mean(r["review_evals_per_s"] for r in out if r["arm"] == "l1")
    for r in out:
        r["vs_lanes1"] = round(r["review_evals_per_s"] / base, 3)
    (T / "summary.json").write_text(json.dumps({"evals_per_run": EVALS, "runs": out}, indent=1))
    lines = ["# SM48b timing: lanes with int state variants", "",
             f"40 users of the 732 pool, SM36 w8+s8, w8+s4c8n, w8+s8hx8, w8+s4c8nhx8 = {EVALS:,} review-evals per run; "
             "rate = scorer counter; mem = nvidia-smi max / torch peak allocated, reserved.", "",
             "| Run | Flags | review-evals/s | vs lanes 1 | process s | GPU util % | nvidia-smi max MiB | torch alloc / reserved GiB | rows | err |",
             "|---|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    for r in out:
        lines.append(f"| {r['run']} | `{r['flags'] or 'default'}`{' (lock)' if r.get('lock') else ''} | {r['review_evals_per_s']:,.0f} | "
                     f"{r['vs_lanes1']:.2f}x | {r['process_seconds']} | {r['gpu_util_mean_pct']} | {r['gpu_mem_max_mib']:.0f} | "
                     f"{r['torch_peak_allocated_gib']} / {r['torch_peak_reserved_gib']} | {r['rows']} | {r['error_rows']} |")
    lines += ["", "Row compare vs run 1 (AGENT.md evaluation tolerance):", ""]
    for r in out:
        c = r["compare_vs_run1"]
        if c:
            worst = {k: v for k, v in c["max_abs_delta"].items() if v}
            lines.append(f"- {r['run']}: {c['bitwise_equal_pairs']}/{c['pairs']} pairs bitwise, passed {c['passed']}, "
                         f"only_a {c['only_a']} only_b {c['only_b']}{', max deltas ' + json.dumps(worst) if worst else ''}")
    (T / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
