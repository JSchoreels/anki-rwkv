"""SM24: student-vs-fork gap by review type / same-day on all dumped users (artifacts/small-model/sm24-dump).
Review type = revlog `state` of the predicted review (pre-review, SM-2 scheduler state); same-day = the card was already
reviewed earlier that day. Aggregate output only."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SM = Path(__file__).resolve().parents[1]
DUMP = SM / "sm24-dump"
ORIG = Path(r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
NAMES = {0: "new", 1: "learning", 2: "review", 3: "relearning", 4: "filtered/cram"}
users = sorted(int(p.name.split("_")[0]) for p in DUMP.glob("*_st_w8+s8.npz") if (DUMP / p.name.replace("_st_", "_fork_")).exists())
acc = {}
for u in users:
    d = {m: np.load(DUMP / f"{u}_{m}_w8+s8.npz") for m in ("fork", "st")}
    assert (d["fork"]["review_th"] == d["st"]["review_th"]).all()
    rv = pd.read_parquet(ORIG / "revlogs" / f"user_id={u}", columns=["card_id", "day_offset", "state"])
    same = rv.groupby(["card_id", "day_offset"]).cumcount().to_numpy() > 0
    i = d["st"]["review_th"] - 1; y = d["st"]["y"]
    keys = {"type": rv["state"].to_numpy()[i], "same-day": same[i].astype(int)}
    for h in ("ahead", "instant"):
        L = {m: -(y * np.log(np.clip(d[m][f"p_{h}"], 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - d[m][f"p_{h}"], 1e-7, 1))) for m in d}
        for k, v in keys.items():
            for val in np.unique(v):
                s = v == val; a = acc.setdefault((h, k, int(val)), np.zeros(4))
                a += [s.sum(), L["fork"][s].sum(), L["st"][s].sum(), (L["st"][s] > L["fork"][s]).sum()]
out = {"users": len(users), "rows": []}
print(f"{len(users)} users")
for h in ("ahead", "instant"):
    for k in ("type", "same-day"):
        rows = {v: a for (hh, kk, v), a in acc.items() if hh == h and kk == k}
        n = sum(a[0] for a in rows.values()); gap = sum(a[2] - a[1] for a in rows.values())
        tot = 100 * (sum(a[2] for a in rows.values()) / sum(a[1] for a in rows.values()) - 1)
        print(f"[{h}] by {k}: overall student vs fork {tot:+.2f}%")
        for v, a in sorted(rows.items()):
            lab = NAMES[v] if k == "type" else ("yes" if v else "no")
            r = {"head": h, "by": k, "value": lab, "share": a[0] / n, "student_vs_fork_pct": 100 * (a[2] / a[1] - 1),
                 "share_of_gap": (a[2] - a[1]) / gap, "reviews": int(a[0])}
            out["rows"].append(r)
            print(f"   {lab:14s} share {100 * r['share']:5.1f}%  student vs fork {r['student_vs_fork_pct']:+.2f}%  share of gap {100 * r['share_of_gap']:5.0f}%")
(SM / "feature-screen" / "sm24-bytype.json").write_text(json.dumps(out, indent=1))
