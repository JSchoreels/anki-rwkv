#!/usr/bin/env python
"""Long-card state drift on real users (SM136): the user pools, fixed before any scoring (CPU, about 1 min; reads only the
revlogs' card_id column, no model output).

  main pool (rule): the 723 fit users of the 732 set (wt/sm72s eval-users-732-fit.json) that have at least one card with
             >= 128 prior answers in the revlog -> users-sm136.json
  extension (report only): validation users of the 8k split with id >= 5000 (scripts/eval_pools.py pool 1500 minus every
             user of the 732 pool, never-fit ones included) with at least one card with >= 300 prior answers ->
             users-sm136-ext.json. Unseen by the Small and Tiny models; training users of the big models.
  pool-sm136.json: aggregate counts only (users, predicted reviews per bucket of the card's prior answers; a predicted
             review = a revlog row with >= 1 prior answer of its card, the scorer's labelled reviews).

  python sm136_pool.py [--check]   (--check: recompute and compare with the committed files, exit 1 on a difference)
"""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
U732FIT = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json")
EDGES = [0, 32, 128, 300, 1000]
NAMES = ["0-31", "32-127", "128-299", "300-999", "1000+"]
MAIN_MIN, EXT_MIN = 128, 300


def buckets(prior):
    p = prior[prior >= 1]
    return np.bincount(np.searchsorted(np.asarray(EDGES[1:]), p, side="right"), minlength=len(EDGES))


def main():
    from scripts import compress_common as cc
    from scripts.eval_pools import pool_users
    from scripts.small_model_heldout import card_prior_answers
    data = cc.data_path(cc.run_info())
    fit = json.loads(U732FIT.read_text())
    u723 = fit["users"]
    u732 = set(u723) | set(fit["never_fit"])
    ext_candidates = sorted(u for u in pool_users(1500) if u >= 5000 and u not in u732)
    out = {"edges": EDGES, "buckets": NAMES, "note": "aggregate counts; predicted review = revlog row with >= 1 prior answer of its card"}
    files = {}
    for name, users, thr in (("main", u723, MAIN_MIN), ("extension", ext_candidates, EXT_MIN)):
        chosen, tot, nusers, per_user_reviews = [], np.zeros(len(EDGES), np.int64), np.zeros(len(EDGES), np.int64), 0
        for u in users:
            prior = card_prior_answers(data, u)
            if prior.max() >= thr:
                b = buckets(prior)
                chosen.append(u)
                tot += b
                nusers += b > 0
                per_user_reviews += len(prior)
        out[name] = {"rule": f"{'723 fit users of the 732 set' if name == 'main' else 'validation users id >= 5000 outside the 732 set'} "
                             f"with a card of >= {thr} prior answers", "candidates": len(users), "users": len(chosen),
                     "revlog_rows": int(per_user_reviews), "predicted_reviews_per_bucket": tot.tolist(),
                     "users_per_bucket": nusers.tolist()}
        files["users-sm136.json" if name == "main" else "users-sm136-ext.json"] = {
            "note": f"Long-card state drift on real users (SM136), {name} pool: {out[name]['rule']} (sm136_pool.py). "
                    "Public benchmark user ids only.", "users": chosen}
    files["pool-sm136.json"] = out
    if sys.argv[1:] == ["--check"]:
        same = all(json.loads((HERE / f).read_text()) == v for f, v in files.items())
        print(f"[sm136-pool] check {'OK' if same else 'DIFFERS'}")
        return 0 if same else 1
    for f, v in files.items():
        (HERE / f).write_text(json.dumps(v, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
