#!/usr/bin/env python
"""The Big / Small / Tiny model sizes (SM104): accuracy cost of the fork's compact state formats for the tiny model
(the smaller instant-only model, SM101) next to the show-time instant-only model, stage 2 (SM98).

Rows: `run-precision.sh` (scripts/small_model_heldout.py on CPU, f32 weights, leak-free `--deleted-cards neighbour`,
show-time inputs `--inputs e2s-query`, the 80 random users of the 732 set drawn for the int4 drift check (SM78)).
Formats: fp32 (f32 state), fp32+s8 (int8 card / note WKV rows = the fork's `int8`), fp32+s4c8n (int4 card + int8 note
WKV rows = the fork's default `cards_int4_notes_int8`). The research scorer keeps the token-shift vectors at f32; the
fork rounds them to f16 (a smaller rounding than the WKV rows' int8 / int4).

Metric: benchmark instant LogLoss per user (the scorer's `benchmark.instant`, the same review set as the decision rules),
review-weighted over users; delta = format minus f32 state of the same model, in % of that model's f32-state LogLoss;
paired user bootstrap 2,000 resamples, seed 0, 95% CI (the show-time report's convention). "Tiny minus Small" is the
difference of the two models' costs on the same bootstrap resamples (the % of each model's own f32 LogLoss).

usage (from the worktree root): $PY -X utf8 artifacts/small-model/sm101/sm104/sm104_precision.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = {"sm101": "the tiny model (the smaller instant-only model, SM101)",
          "sm98": "the Small instant model (the show-time instant-only model, stage 2, SM98)"}
FORMATS = {"fp32": "f32 state", "fp32+s8": "int8 cards + int8 notes", "fp32+s4c8n": "int4 cards + int8 notes (default)"}
HEADS = ("instant", "ahead")


def load():
    rows = {}
    for path in sorted(HERE.glob("scores*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if "error" in r or "benchmark" not in r:
                continue
            b = r["benchmark"]
            rows[(r["tag"], r["variant"], r["user_id"])] = {h: (b[h]["LogLoss"], b[h]["size"]) for h in HEADS}
    return rows


def main():
    rows = load()
    users = sorted({u for (_, _, u) in rows})
    users = [u for u in users if all((m, f, u) in rows for m in MODELS for f in FORMATS)]
    idx = np.random.default_rng(0).integers(0, len(users), size=(2000, len(users)))
    out = {"users": len(users), "user_ids": users, "unit": "% of the same model's f32-state LogLoss",
           "bootstrap": "paired user bootstrap, 2,000 resamples, seed 0, 95% CI", "heads": {}}
    for head in HEADS:
        w = np.array([rows[("sm98", "fp32", u)][head][1] for u in users], float)
        for m in MODELS:
            for f in FORMATS:
                assert all(rows[(m, f, u)][head][1] == w[i] for i, u in enumerate(users)), (m, f, head)
        ll = {(m, f): np.array([rows[(m, f, u)][head][0] for u in users]) for m in MODELS for f in FORMATS}
        res = {"reviews": int(w.sum()), "logloss": {}, "cost": {}, "tiny_minus_small": {}}
        boots = {}
        for m in MODELS:
            base = ll[(m, "fp32")]
            res["logloss"][m] = {f: round(float((ll[(m, f)] * w).sum() / w.sum()), 6) for f in FORMATS}
            res["cost"][m] = {}
            for f in ("fp32+s8", "fp32+s4c8n"):
                d = ll[(m, f)] - base
                boot = (d[idx] * w[idx]).sum(1) / (base[idx] * w[idx]).sum(1) * 100
                boots[(m, f)] = boot
                res["cost"][m][f] = {"pct": round(float((d * w).sum() / (base * w).sum() * 100), 4),
                                     "pct_ci": [round(float(np.percentile(boot, q)), 4) for q in (2.5, 97.5)],
                                     "delta": round(float((d * w).sum() / w.sum()), 7),
                                     "users_worse": int((d > 0).sum()), "users_better": int((d < 0).sum())}
        for f in ("fp32+s8", "fp32+s4c8n"):
            a = ll[("sm101", f)] - ll[("sm101", "fp32")]
            b = ll[("sm98", f)] - ll[("sm98", "fp32")]
            pa = (a * w).sum() / (ll[("sm101", "fp32")] * w).sum() * 100
            pb = (b * w).sum() / (ll[("sm98", "fp32")] * w).sum() * 100
            boot = boots[("sm101", f)] - boots[("sm98", f)]
            res["tiny_minus_small"][f] = {"pct": round(float(pa - pb), 4),
                                          "pct_ci": [round(float(np.percentile(boot, q)), 4) for q in (2.5, 97.5)]}
        out["heads"][head] = res
    (HERE / "sm104-precision.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    r = out["heads"]["instant"]
    md = [f"# Tiny vs Small: accuracy cost of the compact state formats (SM104), instant head", "",
          f"{len(users)} users of the 732 set (the int4 drift check (SM78)'s 80 random users), {r['reviews']:,} benchmark "
          "reviews, CPU, f32 weights, leak-free, show-time inputs. Cost = format minus f32 state of the same model, in % "
          "of that model's f32-state instant LogLoss; paired user bootstrap 2,000, seed 0, 95% CI.", "",
          "| model | f32 state LogLoss | int8 cost, % [95% CI] | int4 cards + int8 notes cost, % [95% CI] |",
          "|---|---|---|---|"]
    for m, name in MODELS.items():
        c = r["cost"][m]
        md.append(f"| {name} | {r['logloss'][m]['fp32']:.6f} | {c['fp32+s8']['pct']:+.4f} [{c['fp32+s8']['pct_ci'][0]:+.4f}, "
                  f"{c['fp32+s8']['pct_ci'][1]:+.4f}] | {c['fp32+s4c8n']['pct']:+.4f} [{c['fp32+s4c8n']['pct_ci'][0]:+.4f}, "
                  f"{c['fp32+s4c8n']['pct_ci'][1]:+.4f}] |")
    t = r["tiny_minus_small"]
    md.append(f"| Tiny's cost minus Small's | | {t['fp32+s8']['pct']:+.4f} [{t['fp32+s8']['pct_ci'][0]:+.4f}, "
              f"{t['fp32+s8']['pct_ci'][1]:+.4f}] | {t['fp32+s4c8n']['pct']:+.4f} [{t['fp32+s4c8n']['pct_ci'][0]:+.4f}, "
              f"{t['fp32+s4c8n']['pct_ci'][1]:+.4f}] |")
    md += ["", "Raw numbers, the curve head (untrained for both models, not used) and per-format LogLoss: "
           "`sm104-precision.json`.", ""]
    (HERE / "sm104-precision.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
