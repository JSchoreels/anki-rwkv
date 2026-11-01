"""Pick the training chunk size for `runbook-next.sh scale` from the ctx2048 probe (docs/SMALL_MODEL.md 5.7).

Rule (k = 1, paired on the users both labels have, review-weighted as small_model_report.py):
  use the candidate (2,048-review chunks) only if
    - it has a row for every user the base has (464 full-history validation users),
    - LogLoss(cand) - LogLoss(base) <= -MARGIN on both heads, with the 95% user-bootstrap upper bound < 0,
    - its decay trained at most MAX_SLOWDOWN times as long as the base decay;
  otherwise keep the base (1,024-review chunks). A missing probe (failed, OOM, no rows) also keeps the base.
MARGIN 0.001 is about 2x the k = 3 full-history band of screen B (0.00053 / 0.00056) and a fifth of the
128 -> 1,024 effect (-0.0047 / -0.0054), so a single seed cannot pass it on noise.

Epoch estimates for the queue: 61 min per gate-tier epoch at 128-review chunks (SMALL_MODEL.md 5.6) scaled by the
chosen decay's training time over the cold 128-chunk decay cal-ctl-s2; epoch 1 gets +10 min (cold teacher cache).

Prints "CHUNK BATCH EPOCH_MIN EPOCH1_MIN" and writes the decision JSON.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from small_model_report import load, rw

CKPT_RESULT = "expertium/E89-F65e-F77a/result.json"
MARGIN = 0.001
MAX_SLOWDOWN = 1.5
OPTIONS = {"cand": (2048, 16), "base": (1024, 32)}


def train_seconds(sm, name):
    path = Path(sm) / name / CKPT_RESULT
    return json.loads(path.read_text())["training_seconds"] if path.exists() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--sm", required=True, help="artifacts/small-model folder (decay runs)")
    parser.add_argument("--cand", default="cal-ctx2048-s1")
    parser.add_argument("--base", default="cal-ctx1024-s1")
    parser.add_argument("--speed-ref", default="cal-ctl-s2", help="cold 128-chunk decay for the epoch estimate")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    per, _ = load(args.jsonl)
    cand, base = f"{args.cand}/w8+s8", f"{args.base}/w8+s8"
    decision = {"rule": __doc__.split("\n\n")[1], "candidate": cand, "base": base, "margin": MARGIN,
                "max_slowdown": MAX_SLOWDOWN, "heads": {}}
    reasons = []
    if base not in per:
        raise SystemExit(f"base {base} has no rows")
    users = sorted(per[base])
    missing = [u for u in users if u not in per.get(cand, {})]
    decision["users"] = len(users)
    decision["candidate_missing_users"] = len(missing)
    if missing:
        reasons.append(f"candidate incomplete ({len(missing)} of {len(users)} users missing)")
    else:
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(users), size=(2000, len(users)))
        for head in ("ahead", "instant"):
            c = np.array([per[cand][u][head]["LogLoss"] for u in users])
            b = np.array([per[base][u][head]["LogLoss"] for u in users])
            s = np.array([per[base][u][head]["size"] for u in users], dtype=float)
            delta = rw(c, s) - rw(b, s)
            boot = ((c[idx] - b[idx]) * s[idx]).sum(1) / s[idx].sum(1)
            upper = float(np.percentile(boot, 97.5))
            decision["heads"][head] = {"cand": rw(c, s), "base": rw(b, s), "delta": delta, "ci95_upper": upper}
            if not (delta <= -MARGIN and upper < 0):
                reasons.append(f"{head}: delta {delta:+.5f} (upper {upper:+.5f}) not <= -{MARGIN}")
    t_cand, t_base, t_ref = (train_seconds(args.sm, n) for n in (args.cand, args.base, args.speed_ref))
    decision["training_seconds"] = {args.cand: t_cand, args.base: t_base, args.speed_ref: t_ref}
    if t_cand is None or t_base is None:
        reasons.append("training time missing")
    elif t_cand > MAX_SLOWDOWN * t_base:
        reasons.append(f"candidate {t_cand / t_base:.2f}x slower than base (> {MAX_SLOWDOWN})")
    pick = "base" if reasons else "cand"
    chunk, batch = OPTIONS[pick]
    t_pick = t_cand if pick == "cand" else t_base
    scale = (t_pick / t_ref) if (t_pick and t_ref) else 1.3
    epoch_min = math.ceil(61 * scale)
    decision.update({"pick": pick, "chunk_reviews": chunk, "batch_chunks": batch, "reasons": reasons,
                     "epoch_min": epoch_min, "epoch1_min": epoch_min + 10})
    Path(args.out).write_text(json.dumps(decision, indent=1) + "\n")
    print(chunk, batch, epoch_min, epoch_min + 10)


if __name__ == "__main__":
    main()
