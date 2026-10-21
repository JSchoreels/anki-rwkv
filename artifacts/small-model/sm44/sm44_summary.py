"""SM44: GPU scoring of the compressed card/note state (~400 and ~800 bits), paired summary, CPU only.

Rows (full-history held-out protocol, w8 weights):
  sm44/heldout-sm44.jsonl          the SM44 GPU rows: SM42 w8+vq378x412, w8+vq770x828, w8+s8 (re-score) and the SM32c
                                   model (learned codebooks) w8+vq378x412, w8+s8
  wt/sm42 heldout-sm42*.jsonl      SM42 w8+s8 rows scored before (used where SM44 has no SM42 w8+s8 row; the users with
                                   both give a reproducibility check)
  wt/small-model heldout*.jsonl    fork w8+s8 (the 2.76M model, int8 state)
Per pool (732, and its 464 subset), users = those with a row for every label shown and the fork. Per head (curve =
ahead, instant): review-weighted benchmark LogLoss, % vs the same model's w8+s8, vs SM42 w8+s8 and vs the fork's
w8+s8, each with a paired 95% user bootstrap (2,000 resamples, seed 0, 2.5/97.5 percentiles).
Writes sm44-summary.json and sm44-summary.md next to this file.

usage: python -X utf8 artifacts/small-model/sm44/sm44_summary.py
"""
import json
import os
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
SM = OUT.parent
SMW = Path("C:/Users/jonat/workspace/wt/small-model/artifacts/small-model")
S42W = Path("C:/Users/jonat/workspace/wt/sm42/artifacts/small-model")
ROWS = OUT / "heldout-sm44.jsonl"
L42 = "s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes"
L32C = "s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes-vqlearn"
FORK = "fork/w8+s8"
HEADS = {"curve": "ahead", "instant": "instant"}
# label, model (its w8+s8 is the "same model" reference), description
LABELS = [(f"{L42}/w8+s8", L42, "SM42 (best model), int8 state"),
          (f"{L42}/w8+vq378x412", L42, "SM42, codebooks fitted after training (k-means), ~400 bits"),
          (f"{L42}/w8+vq770x828", L42, "SM42, k-means codebooks, ~800 bits"),
          (f"{L32C}/w8+s8", L32C, "SM32c (compression trained in, learned codebooks), int8 state"),
          (f"{L32C}/w8+vq378x412", L32C, "SM32c, its learned codebooks, ~400 bits")]


def load(paths, label):
    per = {}
    for path in paths:
        if not Path(path).exists():
            continue
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if label not in line:
                continue
            row = json.loads(line)
            bench = row.get("benchmark", {})
            if row.get("label") == label and "ahead" in bench and "instant" in bench:
                per[row["user_id"]] = row
    return per


def boot_pct(new, ref, sizes, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sizes), size=(n, len(sizes)))
    w = sizes[idx]
    pct = ((new[idx] * w).sum(1) / w.sum(1) / ((ref[idx] * w).sum(1) / w.sum(1)) - 1) * 100
    point = ((new * sizes).sum() / (ref * sizes).sum() - 1) * 100
    return float(point), [float(np.percentile(pct, 2.5)), float(np.percentile(pct, 97.5))]


def ll(row, head):
    return row["benchmark"][HEADS[head]]["LogLoss"]


def size(row, head):
    return row["benchmark"][HEADS[head]]["size"]


def main():
    rows = {label: load([ROWS], label) for label, _, _ in LABELS}
    speed = {}
    for label, r in rows.items():   # GPU model time of the SM44 rows
        secs = sum(x["model_seconds"] for x in r.values())
        if secs:
            revs = sum(x["reviews"] for x in r.values())
            speed[label] = {"users": len(r), "reviews": revs, "reviews_per_second_model": round(revs / secs, 1)}
    old42 = load([S42W / "heldout-sm42.jsonl", S42W / "heldout-sm42-pools.jsonl"], f"{L42}/w8+s8")
    new42 = rows[f"{L42}/w8+s8"]
    both = sorted(set(old42) & set(new42))
    repro = {"users": len(both)}
    if both:
        repro.update({h: float(max(abs(ll(new42[u], h) - ll(old42[u], h)) for u in both)) for h in HEADS})
    rows[f"{L42}/w8+s8"] = {**old42, **new42}
    fork = load([SMW / "heldout.jsonl", SMW / "heldout-pools.jsonl"], FORK)
    import sys
    sys.path.insert(0, str(SM.parents[1]))
    from scripts.eval_pools import pool_users
    pools = {p: set(pool_users(p)) for p in (732, 464)}   # 464 is a subset of 732
    check = OUT / "check-sm44-gpu.json"
    out = {"rows_file": str(ROWS), "sm42_w8s8_reproducibility_max_abs_dLogLoss": repro, "rows_per_label":
           {label: len(r) for label, r in rows.items()}, "fork_rows": len(fork), "pools": {}}
    if check.exists():
        out["gpu_check"] = {k: json.loads(check.read_text()).get(k) for k in ("passed", "timing", "device")}
    out["scoring_speed"] = speed
    lines = ["# SM44: compressed card/note state on the GPU scorer (~400 and ~800 bits)", "",
             "Full-history held-out protocol, w8 weights. The compressed rows (`+vq`) were scored on the GPU (SM44: fused "
             "kernel with the Gram-squaring factorization in float64 for rank-1 streams, a per-step torch loop for the "
             "rank-2 note of vq770x828); the CPU-first check and the GPU check ticket compare that path with the CPU "
             "scorer (`check-sm44-cpu.json`, `check-sm44-gpu.json`). Paired, review-weighted benchmark LogLoss; "
             "[ ] = 95% user bootstrap (2,000 resamples, seed 0, 2.5/97.5 percentiles). Fork = the 2.76M model, int8 state.",
             ""]
    for pool in (732, 464):
        users_pool = pools[pool]
        if not users_pool:
            continue
        present = [lab for lab, _, _ in LABELS if users_pool & set(rows[lab])]
        users = sorted(users_pool & set(fork) & set.intersection(*(set(rows[lab]) for lab in present))) if present else []
        entry = {"labels": present, "users": len(users), "pool_users": len(users_pool),
                 "missing": {lab: len(users_pool - set(rows[lab])) for lab, _, _ in LABELS}}
        if not users:
            out["pools"][str(pool)] = entry
            continue
        for head in HEADS:
            sz = np.array([size(fork[u], head) for u in users], dtype=float)
            fk = np.array([ll(fork[u], head) for u in users])
            entry[head] = {"eval_reviews": int(sz.sum()), "fork": float((fk * sz).sum() / sz.sum())}
            for lab, model, _ in LABELS:
                if lab not in present:
                    continue
                v = np.array([ll(rows[lab][u], head) for u in users])
                e = {"logloss": float((v * sz).sum() / sz.sum())}
                e["vs_fork_pct"], e["vs_fork_ci95"] = boot_pct(v, fk, sz)
                same = f"{model}/w8+s8"
                if same != lab and same in present:
                    ref = np.array([ll(rows[same][u], head) for u in users])
                    e["vs_same_model_s8_pct"], e["vs_same_model_s8_ci95"] = boot_pct(v, ref, sz)
                    e["vs_same_model_s8_abs"] = e["logloss"] - float((ref * sz).sum() / sz.sum())
                if model != L42 and f"{L42}/w8+s8" in present:
                    ref = np.array([ll(rows[f"{L42}/w8+s8"][u], head) for u in users])
                    e["vs_sm42_s8_pct"], e["vs_sm42_s8_ci95"] = boot_pct(v, ref, sz)
                entry[head][lab] = e
        out["pools"][str(pool)] = entry
        lines += [f"## Pool {pool}: {len(users)} users ({entry[next(iter(HEADS))]['eval_reviews']:,} curve test reviews)", ""]
        bits = {}
        for lab in present:
            sv = next((r.get("state_vq") for r in rows[lab].values() if r.get("state_vq")), None)
            bits[lab] = f"{sv['card']['bits']} / {sv['note']['bits']}" if sv else "int8"
        for head in HEADS:
            e0 = entry[head]
            lines += [f"**{head}** (fork w8+s8 {e0['fork']:.5f})", "",
                      "| Label | Card / note bits | LogLoss | vs same model w8+s8 | vs SM42 w8+s8 | vs fork w8+s8 |",
                      "|---|---|---:|---:|---:|---:|"]
            for lab, model, desc in LABELS:
                if lab not in present:
                    continue
                e = e0[lab]
                fmt = lambda k: (f"{e[k + '_pct']:+.2f}% [{e[k + '_ci95'][0]:+.2f}, {e[k + '_ci95'][1]:+.2f}]"  # noqa: E731
                                 if k + "_pct" in e else "ref" if (k == "vs_same_model_s8" and lab.endswith("w8+s8"))
                                 or (k == "vs_sm42_s8" and model == L42 and lab.endswith("w8+s8")) else "")
                s42 = fmt("vs_sm42_s8") if model != L42 else (
                    fmt("vs_same_model_s8") if not lab.endswith("w8+s8") else "ref")
                lines.append(f"| `{lab}` ({desc}) | {bits[lab]} | {e['logloss']:.5f} | {fmt('vs_same_model_s8')} | {s42} | "
                             f"{fmt('vs_fork')} |")
            lines.append("")
    lines += ["## Checks and speed", "",
              f"- SM42 w8+s8 re-scored by SM44 vs its earlier rows (wt/sm42): {repro.get('users', 0)} users, max |dLogLoss| "
              f"curve {repro.get('curve', float('nan')):.2e}, instant {repro.get('instant', float('nan')):.2e}." if both else
              "- SM42 w8+s8: earlier rows (wt/sm42) used, no SM44 re-score.",
              f"- Rows per label: {out['rows_per_label']}; fork rows {len(fork)}."]
    if speed:
        lines.append("- GPU scorer model time (review-evals per second of model time, all SM44 rows of the label): " +
                     "; ".join(f"`{k}` {v['reviews_per_second_model']:,.0f}/s ({v['users']} users)" for k, v in speed.items()))
    if "gpu_check" in out and out["gpu_check"].get("timing"):
        lines.append(f"- GPU check timing (near-median users): {out['gpu_check']['timing']['reviews_per_second']} reviews/s.")
    lines.append("")
    tmp = OUT / "sm44-summary.json.tmp"
    tmp.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, OUT / "sm44-summary.json")
    (OUT / "sm44-summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({p: {k: v for k, v in e.items() if k in ("users", "missing")} for p, e in out["pools"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
