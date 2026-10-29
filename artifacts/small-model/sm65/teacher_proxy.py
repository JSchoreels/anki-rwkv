"""CPU proxy: teacher LogLoss on final-tier training users from the existing replay caches (shipped big model with leak
inputs, big model (1 leak-free epoch) with neighbour inputs) and their probability average. Binary labels y (recall);
curve = curve head p, instant = 1 - p_again. Users split by whether the big model trained on them (5000-10000)."""
import json, sys, random, os
import numpy as np
from pathlib import Path
sys.path.insert(0, r"C:/Users/jonat/workspace/wt/sm72s")
from rwkv.data_processing import get_rwkv_data
DATA = Path(r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
C = Path(r"C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay")
run = json.load(open(r"C:/Users/jonat/workspace/wt/sm55/artifacts/small-model/sm55-final8k-run.json"))
rc = {int(k): v for k, v in run["review_counts"].items()}
users = [u for u in run["train_users"] if rc[u] <= 60000]
rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
seen = rng.sample([u for u in users if u >= 5000], n); unseen = rng.sample([u for u in users if u < 5000], n)
def ll(p, y):
    p = np.clip(p, 1e-5, 1 - 1e-5); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
def dec(t):
    t = t.astype(np.float32); return 1 / (1 + np.exp(-t[:, 0])), 1 - t[:, 1]
out = {}
for name, group in (("seen", seen), ("unseen", unseen)):
    acc = {k: [0.0, 0] for k in ["ship_c", "ship_i", "lf_c", "lf_i", "ens_c", "ens_i"]}
    per = []
    for u in group:
        f = get_rwkv_data(DATA, u).sort_values("review_th")
        y = f["y"].to_numpy().astype(np.float64)
        a = np.load(C / "fork-fp32-final8k" / "users" / f"{u}.npz")["targets"]
        b = np.load(C / "big-leakfree-1ep-final8k-neighbour" / "users" / f"{u}.npz")["targets"]
        (ac, ai), (bc, bi) = dec(a), dec(b)
        mc = ~np.isnan(ac) & ~np.isnan(bc); mi = ~np.isnan(ai) & ~np.isnan(bi)
        for k, p, m in (("ship_c", ac, mc), ("lf_c", bc, mc), ("ens_c", (ac + bc) / 2, mc),
                        ("ship_i", ai, mi), ("lf_i", bi, mi), ("ens_i", (ai + bi) / 2, mi)):
            acc[k][0] += ll(p[m], y[m]).sum(); acc[k][1] += int(m.sum())
    out[name] = {k: round(v[0] / v[1], 5) for k, v in acc.items()}
    out[name]["users"] = len(group); out[name]["rows_c"] = acc["lf_c"][1]
print(json.dumps(out, indent=1))
