#!/usr/bin/env python
"""Again pressed as Hard (SM132): how often Again occurs in the 723 fitted users of the 732 set (CPU, about a minute).

Reads only the `rating` and `state` columns of each user's revlog. Writes again-share.json: the share of Again (rating 1)
among all reviews (review-weighted over the pool and the median user), among review-state reviews (lapses) and the
share of reviews in relearning streaks, so p reads as "p x share of all reviews changed". Per-user values stay local.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from scripts import compress_common as cc   # noqa: E402

USERS = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json")


def main():
    data = cc.data_path()
    users = json.loads(USERS.read_text())["users"]
    tot = {"reviews": 0, "again": 0, "review_state": 0, "again_review_state": 0, "relearning_rows": 0,
           "learning_or_new_again": 0}
    per_user = []
    for u in users:
        df = pd.read_parquet(data / "revlogs" / f"user_id={u}", columns=["rating", "state"])
        r, s = df["rating"].to_numpy(), df["state"].to_numpy()
        n, a = len(r), int((r == 1).sum())
        tot["reviews"] += n
        tot["again"] += a
        tot["review_state"] += int((s == 2).sum())
        tot["again_review_state"] += int(((r == 1) & (s == 2)).sum())
        tot["relearning_rows"] += int((s == 3).sum())
        tot["learning_or_new_again"] += int(((r == 1) & (s <= 1)).sum())
        per_user.append(a / max(n, 1))
    pu = np.array(per_user)
    out = {"users": len(users), **tot,
           "again_share_of_all_reviews": round(tot["again"] / tot["reviews"], 5),
           "again_share_user_median": round(float(np.median(pu)), 5),
           "again_share_user_p10_p90": [round(float(np.percentile(pu, q)), 5) for q in (10, 90)],
           "lapse_share_of_review_state_reviews": round(tot["again_review_state"] / tot["review_state"], 5),
           "relearning_share_of_all_reviews": round(tot["relearning_rows"] / tot["reviews"], 5),
           "changed_share_of_all_reviews_at_p": {f"{p:g}": round(p * tot["again"] / tot["reviews"], 5)
                                                 for p in (0.05, 0.1, 0.2, 0.3, 0.5)}}
    (HERE / "again-share.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
