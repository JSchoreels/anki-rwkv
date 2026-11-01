import sys, numpy as np
from pathlib import Path
M = ("fork", "st", "forknost")
users = sorted(int(p.stem) for p in Path("fx").glob("*.npz") if all(Path(f"dump/{p.stem}_{m}_w8+s8.npz").exists() for m in M))
acc = {}
for u in users:
    z = np.load(f"fx/{u}.npz"); d = {m: np.load(f"dump/{u}_{m}_w8+s8.npz") for m in M}
    i = d["st"]["review_th"] - 1; y = d["st"]["y"]
    keys = {"review type of the predicted review": z["review_type"][i], "same-day review of the card": (z["card_reviews_today"][i] > 0).astype(float)}
    for h in ("ahead", "instant"):
        L = {m: -(y * np.log(np.clip(d[m][f"p_{h}"], 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - d[m][f"p_{h}"], 1e-6, 1))) for m in M}
        for kn, kv in keys.items():
            for v in np.unique(kv):
                s = kv == v; a = acc.setdefault((h, kn, v), np.zeros(4))
                a += [s.sum(), L["fork"][s].sum(), L["st"][s].sum(), L["forknost"][s].sum()]
names = {0: "new", 1: "learning", 2: "review", 3: "relearning"}
for h in ("ahead", "instant"):
    tot = sum(a[0] for (hh, kn, v), a in acc.items() if hh == h and kn.startswith("review"))
    gap_tot = sum(a[2] - a[1] for (hh, kn, v), a in acc.items() if hh == h and kn.startswith("review"))
    print(f"[{h}]  share of reviews | student vs fork | share of the student's total gap | fork without review type vs fork")
    for (hh, kn, v), a in sorted(acc.items(), key=lambda x: (x[0][1], x[0][2])):
        if hh != h: continue
        lab = names.get(int(v), str(v)) if kn.startswith("review") else ("yes" if v else "no")
        print(f"  {kn:38s} {lab:10s} {100*a[0]/tot:5.1f}% | {100*(a[2]/a[1]-1):+.2f}% | {100*(a[2]-a[1])/gap_tot:5.0f}% | {100*(a[3]/a[1]-1):+.2f}%")
