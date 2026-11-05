"""The small model in the review simulator (SM41), part B post-hoc check: the gap vs FSRS-7 at matched memorised cards.

The plan's metric matches on realised retention. A per-card target that defers slow cards raises the pass rate while
memorising fewer cards, so this check also compares minutes at the same memorised count (day 60, the instant head's
count), over the common memorised range of the policy's and FSRS-7's points, paired per user, 95% CI over users.
Usage: sm41_frontier.py ROWS_GLOB FAMILY [FAMILY ...] (family = the lane suffix, e.g. s2, n0.5).
"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import sm41_fullrun as FR   # noqa: E402
from scripts import sm41_policy as P     # noqa: E402

HERE = Path(__file__).resolve().parents[1] / "artifacts" / "small-model" / "sm41"


def gap(a, b, y, x="memorised"):
    a = [p for p in a if np.isfinite(p[x]) and np.isfinite(p[y])]
    b = [p for p in b if np.isfinite(p[x]) and np.isfinite(p[y])]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    lo = max(min(p[x] for p in a), min(p[x] for p in b))
    hi = min(max(p[x] for p in a), max(p[x] for p in b))
    if hi <= lo:
        return np.nan
    f = lambda pts, v: np.interp(v, *zip(*sorted((p[x], p[y]) for p in pts)))   # noqa: E731
    return float(np.mean([f(a, v) / f(b, v) - 1 for v in np.linspace(lo, hi, 11)]))


def main():
    rows = FR.load_rows(sys.argv[1])
    full = {(r["user"], r["split_day"]): r for r in FR.load_rows(str(HERE / "fullrun" / "rows" / "*.jsonl"))}
    sample = {u["user"] for u in json.loads(P.SAMPLE.read_text())["users"]}
    pairs = [(r["user"], FR.split_policies(r), FR.split_policies(full[(r["user"], r["split_day"])])) for r in rows]
    for fam in sys.argv[2:]:
        for y, what in (("ret", "matched retention (plan)"), ("min_30_59", "days 30-59 minutes"),
                        ("min_all", "60-day minutes")):
            for held in (False, True):
                per = {}
                for u, sp, fr in pairs:
                    if held and u in sample:
                        continue
                    a = [sp[k] for k in sp if k.endswith("-" + fam)]
                    b = [fr[f"fsrs7-{v:.2f}"] for v in FR.LEVELS]
                    per.setdefault(u, []).append(P.overlap_gap(a, b) if y == "ret" else gap(a, b, y))
                g = [np.nanmean(v) for v in per.values() if np.isfinite(v).any()]
                if not g:
                    continue
                m, h, n = FR.ci(g)
                print(f"{fam} | {what} | {'held-out' if held else 'all'} | {100 * m:+.1f} ± {100 * h:.1f} | n {n} | "
                      f"median {100 * np.median(g):+.1f} | cheaper {100 * np.mean(np.array(g) < 0):.0f}%")


if __name__ == "__main__":
    main()
