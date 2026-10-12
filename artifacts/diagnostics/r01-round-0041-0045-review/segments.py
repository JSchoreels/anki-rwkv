"""Align exact R01 runs by work done and compare time per work segment (analysis only, no training).

Console progress lines print cumulative reviews (0.01M precision) and the average rate since training
start; elapsed = reviews / average rate. Exact runs process identical updates in identical order, so
time to reach the same cumulative review count is comparable across runs.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "artifacts/retraining/gpu-retrain-r01"
LINE = re.compile(r"(\d+)/80 users\s+([\d.]+)M \|\s+([\d,]+) rev/s avg")
LOGS = {i: R / f"iteration-{i}-expertium-console.log" for i in
        ["0033", "0034", "0035", "0036", "0037", "0038", "0039", "0041", "0042", "0043", "0044", "0045"]}
LOGS["0040"] = R / "iteration-0040/console.log"
TOTAL = 7_654_975
EDGES = [i / 8 for i in range(9)]  # eight equal-work segments


def curve(path):
    points = [(0.0, 0.0)]
    for match in LINE.finditer(path.read_text(encoding="utf-8", errors="replace")):
        reviews = float(match.group(2)) * 1e6
        rate = float(match.group(3).replace(",", ""))
        if reviews > points[-1][0] and rate > 0:
            points.append((reviews, reviews / rate))
    return points


def time_at(points, reviews):
    for (r0, t0), (r1, t1) in zip(points, points[1:]):
        if r0 <= reviews <= r1:
            return t0 + (t1 - t0) * (reviews - r0) / (r1 - r0)
    return None


def segments(points):
    marks = [time_at(points, edge * TOTAL) for edge in EDGES[:-1]] + [None]
    # The last printed point may stop before the total; extrapolate the final segment at its last rate.
    last_r, last_t = points[-1]
    marks[-1] = last_t + (TOTAL - last_r) * (last_t - points[-2][1]) / (last_r - points[-2][0])
    return [b - a for a, b in zip(marks, marks[1:])]


if __name__ == "__main__":
    out = {}
    for name, path in LOGS.items():
        if path.exists():
            out[name] = [round(s, 3) for s in segments(curve(path))]
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("segments.json")
    target.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for name, values in out.items():
        print(name, f"sum {sum(values):6.2f}", " ".join(f"{v:5.2f}" for v in values))
