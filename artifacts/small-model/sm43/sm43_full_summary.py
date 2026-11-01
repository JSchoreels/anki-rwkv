"""SM43 full: paired comparison of arm B (125 users x 110 epochs) against arm A (1,250 users x 11 epochs), CPU only.

Rows: the 732-pool w8+s8 rows of both arms in sm43/heldout-sm43full-pools.jsonl; the fork's w8+s8 rows from
artifacts/small-model/heldout.jsonl + heldout-pools.jsonl (context only). Users = those with rows for A, B and the fork.
Per head (curve = ahead, instant): review-weighted LogLoss, delta % of B vs A with a 95% user bootstrap (2,000 resamples,
seed 0, 2.5/97.5 percentiles), both arms vs the fork, and history-length tertiles (same math as the SM43 mini decision).
Writes sm43-full-decision.json and sm43-full-summary.md.

usage: python -X utf8 artifacts/small-model/sm43/sm43_full_summary.py
"""
import json
import os
from pathlib import Path

import numpy as np

WT = Path("C:/Users/jonat/workspace/wt/small-model")
SM = WT / "artifacts/small-model"
OUT = SM / "sm43"
EXP = WT / "artifacts/accuracy/experiments"
A, B = "sm43full-a-1250u-11ep", "sm43full-b-125u-110ep"
LA, LB, LF = f"s-w64kd-{A}/w8+s8", f"s-w64kd-{B}/w8+s8", "fork/w8+s8"
HEADS = {"curve": "ahead", "instant": "instant"}
MARGIN = 0.25
MIN_USERS = 700


def load_rows(paths, label):
    per = {}
    for path in paths:
        if not Path(path).exists():
            continue
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            bench = row.get("benchmark", {})
            if row.get("label") == label and "ahead" in bench and "instant" in bench:
                per[row["user_id"]] = {h: (bench[k]["LogLoss"], bench[k]["size"], bench[k]["RMSE(bins)"]) for h, k in HEADS.items()}
                per[row["user_id"]]["reviews"] = row.get("reviews", 0)
    return per


def boot_pct(new, ref, sizes, n=2000, seed=0):
    """Review-weighted (new - ref) / ref in percent: point estimate and 95% user-bootstrap interval."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sizes), size=(n, len(sizes)))
    w = sizes[idx]
    pct = ((new[idx] * w).sum(1) / w.sum(1) / ((ref[idx] * w).sum(1) / w.sum(1)) - 1) * 100
    point = ((new * sizes).sum() / (ref * sizes).sum() - 1) * 100
    return float(point), [float(np.percentile(pct, 2.5)), float(np.percentile(pct, 97.5))]


def arm_info(name):
    ex = json.loads((EXP / name / "experiment.json").read_text())
    phases = [json.loads(l) for l in (EXP / name / "phases.jsonl").read_text().splitlines()] if (EXP / name / "phases.jsonl").exists() else []
    train = [p for p in phases if p.get("phase") == "train" and p.get("status") == "complete"]
    secs = sum(p["training_seconds"] for p in train)
    revs = sum(p["review_presentations"] for p in train)
    return {**ex["sm43"], "epochs_done": len(train), "training_hours": round(secs / 3600, 2),
            "wall_hours": round(sum(p["wall_seconds"] for p in train) / 3600, 2),
            "reviews_per_second": round(revs / secs) if secs else None}


def main():
    jsonl = OUT / "heldout-sm43full-pools.jsonl"
    a, b = load_rows([jsonl], LA), load_rows([jsonl], LB)
    fork = load_rows([SM / "heldout.jsonl", SM / "heldout-pools.jsonl"], LF)
    users = sorted(set(a) & set(b) & set(fork))
    out = {"child": LB, "parent": LA, "users": len(users), "margin_pct": MARGIN, "arms": {"a": arm_info(A), "b": arm_info(B)},
           "rule": "PASS upper95 <= +0.25% on both heads; EXTEND point <= +0.25% on both, an upper above; FAIL point > +0.25% on either"}
    if len(users) < MIN_USERS:
        out.update(verdict="INCOMPLETE", reason=f"{len(users)} common users < {MIN_USERS} (A {len(a)}, B {len(b)}, fork {len(fork)})")
    else:
        ok, fail = True, False
        for head in HEADS:
            sz = np.array([b[u][head][1] for u in users], dtype=float)
            new, ref, fk = (np.array([d[u][head][0] for u in users]) for d in (b, a, fork))
            point, ci = boot_pct(new, ref, sz)
            entry = {"child": float((new * sz).sum() / sz.sum()), "parent": float((ref * sz).sum() / sz.sum()),
                     "fork": float((fk * sz).sum() / sz.sum()), "delta_pct": point, "ci95_pct": ci, "eval_reviews": int(sz.sum())}
            entry["child_vs_fork_pct"], entry["child_vs_fork_ci95_pct"] = boot_pct(new, fk, sz)
            entry["parent_vs_fork_pct"], entry["parent_vs_fork_ci95_pct"] = boot_pct(ref, fk, sz)
            thirds = np.array_split(np.argsort([b[u]["reviews"] for u in users]), 3)
            entry["tertiles_delta_pct"] = {k: boot_pct(new[t], ref[t], sz[t], n=500)[0] for k, t in zip(("light", "mid", "heavy"), thirds)}
            ok &= ci[1] <= MARGIN
            fail |= point > MARGIN
            out[head] = entry
        out["verdict"] = "FAIL" if fail else ("PASS" if ok else "EXTEND")
    tmp = OUT / "sm43-full-decision.json.tmp"
    tmp.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, OUT / "sm43-full-decision.json")

    ra, rb = out["arms"]["a"], out["arms"]["b"]
    lines = ["# SM43 full: data diversity vs epochs (equal reviews seen, with LR decay)", "",
             f"- Arm A ({A}): {ra['users']:,} gate users x {ra['constant_epochs']} constant-LR + {ra['decay_epochs']} decay epoch(s), "
             f"{ra['reviews_per_epoch']:,} reviews/epoch, {ra['reviews_total']:,} total; {ra['epochs_done']} epochs done, "
             f"{ra['training_hours']} h training, {ra['reviews_per_second'] or 0:,} reviews/s.",
             f"- Arm B ({B}): {rb['users']:,} of those users x {rb['constant_epochs']} constant-LR + {rb['decay_epochs']} decay epochs, "
             f"{rb['reviews_per_epoch']:,} reviews/epoch, {rb['reviews_total']:,} total; {rb['epochs_done']} epochs done, "
             f"{rb['training_hours']} h training, {rb['reviews_per_second'] or 0:,} reviews/s.",
             "- Recipe: scale-w64kd-gate-cached-ctx2048 options (2,048-review windows, cached KD), seed 2025, order seed 20261013, from scratch. "
             "Decay: linear 7e-4 -> 0 over the decay epochs (scripts/lr_tail_decay.py), KD alpha 0.5 while decaying.",
             f"- Scoring: 732 pool, w8+s8, {out['users']} users with rows for A, B and the fork. Paired, review-weighted; "
             "95% user bootstrap (2,000 resamples, seed 0).", ""]
    if out["verdict"] == "INCOMPLETE":
        lines += [f"**INCOMPLETE**: {out['reason']}", ""]
    else:
        lines += ["| head | A LogLoss | B LogLoss | B vs A | 95% CI | A vs fork | B vs fork | B vs A light / mid / heavy |",
                  "|---|---|---|---|---|---|---|---|"]
        for head in HEADS:
            e = out[head]
            t = e["tertiles_delta_pct"]
            lines.append(f"| {head} | {e['parent']:.4f} | {e['child']:.4f} | {e['delta_pct']:+.2f}% | [{e['ci95_pct'][0]:+.2f}%, {e['ci95_pct'][1]:+.2f}%] | "
                         f"{e['parent_vs_fork_pct']:+.2f}% | {e['child_vs_fork_pct']:+.2f}% | {t['light']:+.2f}% / {t['mid']:+.2f}% / {t['heavy']:+.2f}% |")
        lines += ["", f"Verdict (B as the candidate, A as the reference; margin +{MARGIN}%): **{out['verdict']}**. {out['rule']}.",
                  "Mini (10-19, 1,000 users x 1 epoch vs 100 x 10, no decay): B was +0.36% curve / +0.46% instant worse (FAIL).", ""]
    (OUT / "sm43-full-summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: out.get(k) for k in ("users", "verdict")} | {h: {k: out[h][k] for k in ("delta_pct", "ci95_pct")} for h in HEADS if h in out}))
    return 0 if out["verdict"] != "INCOMPLETE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
