"""SM46: per-user sequential step counts of the card and note streams on the 732 pool (CPU, data preparation only).

The GPU scorer codes multi-component (rank >= 2) WKV streams in a per-step torch loop (scripts/state_vq.py score_wkv):
one loop iteration per time step of every split, the codec on the non-skipped rows of that step. Its cost is roughly
  card: 2 layers x sum over splits of T (steps)  +  note: 1 layer x sum of T,
each step a fixed number of small kernel launches, plus work proportional to the coded rows. This script writes, per
user of the pool, the batch's split shapes reduced to: reviews, steps (sum of T), active steps (steps with at least one
coded row) and coded rows (non-skipped row-steps) per stream, for the GPU time estimate in the SM46 runbook.

usage: python -X utf8 artifacts/small-model/sm46/step_profile.py [--pool 732] [--out step-profile-732.json]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import torch  # noqa: E402

torch.set_num_threads(2)

from scripts import compress_common as cc  # noqa: E402
from scripts.eval_pools import pool_users  # noqa: E402


def profile(batch):
    out = {}
    for name, i in (("card", 0), ("note", 2)):
        steps = active = rows = splits = maxb = 0
        for T, skip in zip(batch.sub_gather_lens[i], batch.skips[i]):
            sk = skip.view(-1, T).bool()
            coded = ~sk
            T = int(T)
            steps += T
            active += int(coded.any(0).sum())
            rows += int(coded.sum())
            splits += 1
            maxb = max(maxb, int(sk.shape[0]))
        out[name] = {"splits": splits, "steps": steps, "active_steps": active, "coded_rows": rows, "max_rows": maxb}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=732)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "step-profile-732.json"))
    args = ap.parse_args()
    run = cc.run_info()
    counts = run["review_counts"]
    data = cc.data_path(run)
    out = Path(args.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    users = pool_users(args.pool)
    t0 = time.perf_counter()
    for n, u in enumerate(users):
        if str(u) in res:
            continue
        batch = cc.full_history_batch(data, u)
        res[str(u)] = {"reviews": int(counts[str(u)]), **profile(batch)}
        if n % 25 == 0:
            out.write_text(json.dumps(res))
            print(f"[steps] {n + 1}/{len(users)} user {u} {res[str(u)]} {time.perf_counter() - t0:.0f}s", flush=True)
    out.write_text(json.dumps(res))
    tot = {k: sum(r[s][k] for r in res.values() for s in ("card", "note")) for k in ("steps", "active_steps", "coded_rows")}
    print(f"[steps] {len(res)} users, {sum(r['reviews'] for r in res.values()):,} reviews; totals {tot}", flush=True)


if __name__ == "__main__":
    main()
