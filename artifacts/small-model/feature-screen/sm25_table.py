"""SM25 table: fork under relabelled history review types (sm25.py) vs the normal fork and the student, by slice."""
import numpy as np
from pathlib import Path
MODELS = ("fork", "st", "forknost", "forknosteps", "forkdaysteps")
users = sorted(int(p.stem) for p in Path("fx").glob("*.npz") if all(Path(f"dump/{p.stem}_{m}_w8+s8.npz").exists() for m in MODELS))
acc = {}
for u in users:
    d = {m: np.load(f"dump/{u}_{m}_w8+s8.npz") for m in MODELS}
    assert all((d[m]["review_th"] == d["fork"]["review_th"]).all() for m in MODELS)
    st = np.load(f"fx/{u}.npz")["review_type"][d["fork"]["review_th"] - 1]; y = d["fork"]["y"]
    for h in ("ahead", "instant"):
        for m in MODELS:
            p = np.clip(d[m][f"p_{h}"], 1e-7, 1 - 1e-7); L = -(y * np.log(p) + (1 - y) * np.log(1 - p))
            for s, mk in (("all", st >= 0), ("learning", st == 1), ("review", st == 2), ("relearning", st == 3)):
                acc[(h, m, s)] = acc.get((h, m, s), 0) + L[mk].sum()
print(f"{len(users)} users; log loss vs the normal fork (w8+s8), by review type of the predicted review")
for h in ("ahead", "instant"):
    print(f"[{h}]")
    for m in MODELS[1:]:
        print(f"  {m:13s} " + "  ".join(f"{s} {100 * (acc[(h, m, s)] / acc[(h, 'fork', s)] - 1):+.2f}%" for s in ("all", "learning", "review", "relearning")))
