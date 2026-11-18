"""SM46: large-budget codebook state compression (vq5902x3028: 5,902 bits per card, 3,028 per note; int8 card state is
26,624 bits) on SM36 (the best small model), scored after training on the GPU. CPU pairing summary.

Rows: sm46/heldout-sm46.jsonl (SM36 w8+vq5902x3028), SM36 w8+s8 (wt/sm42 heldout-sm36-pools.jsonl + heldout-sm36.jsonl),
fork w8+s8 (the 2.76M model; wt/small-model heldout.jsonl + heldout-pools.jsonl). For the 732 pool and its 464 subset,
users = those with all three rows. Per head (curve = ahead, instant): review-weighted benchmark LogLoss; delta vs SM36
w8+s8 (absolute and %) and % vs the fork on the same users, each with a paired user bootstrap (2,000 resamples, seed 0,
2.5/97.5 percentiles). Writes sm46-summary.md / .json next to this file.

usage: python -X utf8 artifacts/small-model/sm46/sm46_summary.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
S42 = Path("C:/Users/jonat/workspace/wt/sm42/artifacts/small-model")
SMW = Path("C:/Users/jonat/workspace/wt/small-model/artifacts/small-model")
L = "s-w64kd-gate-sc-lanes-dec12-1ep32k-muon"
RECIPE = "vq5902x3028"
VQ, S8, FORK = f"{L}/w8+{RECIPE}", f"{L}/w8+s8", "fork/w8+s8"
HEADS = {"curve": "ahead", "instant": "instant"}


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


def wmean(v, w):
    return float((v * w).sum() / w.sum())


def boot(new, ref, w, n=2000, seed=0):
    """Paired user bootstrap of the review-weighted means: (delta, [lo, hi], pct, [lo, hi])."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(w), size=(n, len(w)))
    ww = w[idx]
    a = (new[idx] * ww).sum(1) / ww.sum(1)
    b = (ref[idx] * ww).sum(1) / ww.sum(1)
    d, p = a - b, (a / b - 1) * 100
    pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    return (wmean(new, w) - wmean(ref, w), pct(d), (wmean(new, w) / wmean(ref, w) - 1) * 100, pct(p))


def main():
    from scripts.eval_pools import pool_users
    vq = load([OUT / "heldout-sm46.jsonl"], VQ)
    s8 = load([S42 / "heldout-sm36.jsonl", S42 / "heldout-sm36-pools.jsonl"], S8)
    fork = load([SMW / "heldout.jsonl", SMW / "heldout-pools.jsonl"], FORK)
    secs = sum(r["model_seconds"] for r in vq.values())
    out = {"rows": {"vq": len(vq), "s8": len(s8), "fork": len(fork)}, "recipe": RECIPE, "pools": {}}
    if secs:
        out["gpu_model_reviews_per_second"] = round(sum(r["reviews"] for r in vq.values()) / secs, 1)
        out["gpu_model_hours"] = round(secs / 3600, 2)
    sv = next((r.get("state_vq") for r in vq.values() if r.get("state_vq")), None)
    if sv:
        out["bits"] = {"card": sv["card"]["bits"], "note": sv["note"]["bits"]}
        out["codebooks_sha256"] = next(iter(vq.values())).get("vq_codebooks_sha256")
        out["state_vq_path"] = next(iter(vq.values())).get("state_vq_path")
    check = OUT / "check-sm46-gpu.json"
    if check.exists():
        c = json.loads(check.read_text())
        out["gpu_check"] = {"passed": c.get("passed"), "fails": c.get("fails"), "timing": c.get("timing")}
    lines = [f"# SM46: codebook state compression at ~6,000 / ~3,000 bits (`{RECIPE}`), SM36, w8 weights", "",
             f"Card {out.get('bits', {}).get('card', 5902):,} bits, note {out.get('bits', {}).get('note', 3028):,} bits "
             "per entity (int8 state: 26,624 / 14,336). Codebooks fitted after training on SM36's own states (k-means, "
             "no retraining). Full-history held-out protocol; review-weighted benchmark LogLoss; delta = vq - SM36 "
             "w8+s8; [ ] = 95% paired user bootstrap (2,000 resamples, seed 0, 2.5/97.5 percentiles); % vs the big model "
             "(fork w8+s8) on the same users; curve = ahead head.", ""]
    for pool in (732, 464):
        users = sorted(set(pool_users(pool)) & set(vq) & set(s8) & set(fork))
        entry = {"users": len(users), "pool_users": len(pool_users(pool)),
                 "missing_vq": len(set(pool_users(pool)) - set(vq))}
        if not users:
            out["pools"][str(pool)] = entry
            lines += [f"- Pool {pool}: no paired users yet.", ""]
            continue
        lines += [f"## Pool {pool}: {len(users)} of {entry['pool_users']} users", "",
                  "| Head | Test reviews | vq LogLoss | SM36 w8+s8 | delta vs s8 | % vs s8 | vq % vs fork | s8 % vs fork | fork |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for head, key in HEADS.items():
            w = np.array([fork[u]["benchmark"][key]["size"] for u in users], dtype=float)
            g = lambda rows: np.array([rows[u]["benchmark"][key]["LogLoss"] for u in users])  # noqa: E731
            a, b, f = g(vq), g(s8), g(fork)
            d, dci, p, pci = boot(a, b, w)
            _, _, pf, pfci = boot(a, f, w)
            _, _, sf, sfci = boot(b, f, w)
            entry[head] = {"eval_reviews": int(w.sum()), "vq": wmean(a, w), "s8": wmean(b, w), "fork": wmean(f, w),
                           "delta_vs_s8": d, "delta_vs_s8_ci95": dci, "pct_vs_s8": p, "pct_vs_s8_ci95": pci,
                           "vq_pct_vs_fork": pf, "vq_pct_vs_fork_ci95": pfci, "s8_pct_vs_fork": sf,
                           "s8_pct_vs_fork_ci95": sfci}
            lines.append(f"| {head} | {int(w.sum()):,} | {wmean(a, w):.5f} | {wmean(b, w):.5f} | {d:+.5f} "
                         f"[{dci[0]:+.5f}, {dci[1]:+.5f}] | {p:+.3f}% [{pci[0]:+.3f}, {pci[1]:+.3f}] | "
                         f"{pf:+.2f}% [{pfci[0]:+.2f}, {pfci[1]:+.2f}] | {sf:+.2f}% [{sfci[0]:+.2f}, {sfci[1]:+.2f}] | "
                         f"{wmean(f, w):.5f} |")
        lines.append("")
        out["pools"][str(pool)] = entry
    if secs:
        lines.append(f"- GPU scorer: {len(vq)} users, {out['gpu_model_hours']} h of model time, "
                     f"{out['gpu_model_reviews_per_second']:,.0f} reviews/s; path {out.get('state_vq_path')}.")
    if "gpu_check" in out:
        lines.append(f"- GPU check: passed {out['gpu_check']['passed']}, fails {out['gpu_check']['fails']}.")
    lines.append(f"- Codebooks sha256 {out.get('codebooks_sha256')}.")
    lines.append("")
    tmp = OUT / "sm46-summary.json.tmp"
    tmp.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, OUT / "sm46-summary.json")
    (OUT / "sm46-summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
