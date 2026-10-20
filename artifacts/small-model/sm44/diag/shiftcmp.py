"""All diverging shift-codec picks before the first diverging WKV reconstruction: gaps. usage: WT CASES CASE A B"""
import sys, json
WT, CJ, CASE, FA, FB = sys.argv[1:6]
sys.path.insert(0, WT)
import numpy as np, torch
from scripts import state_vq
case = [c for c in json.load(open(CJ)) if c[0] == CASE][0]
books = state_vq.Codebooks.load(case[2])
codecs = {s: state_vq.StateCodec(r, books) for s, r in zip((0, 2), state_vq.recipe(case[3]))}
def load(f):
    z = np.load(f); calls = {}
    for key in z.files:
        i, kind, *tag, what = key.split("/")
        calls.setdefault(int(i), {"tag": "/".join(tag)})[what] = z[key]
    return [calls[i] for i in sorted(calls)]
A, B = load(FA), load(FB)
out = []
for i, (a, b) in enumerate(zip(A, B)):
    if not a["tag"].startswith("shift"):
        continue
    rows = np.nonzero((a["picks"] != b["picks"]).any(-1))[0]
    if not len(rows):
        continue
    st = None
    xt = torch.from_numpy(a["x"])
    for c in codecs.values():
        for pos in c.shift_books.values():
            for cand in pos:
                if cand[0].shape[1] == a["x"].shape[1] and len(cand) == a["picks"].shape[1] and np.array_equal(state_vq.rvq_picks(xt, cand)[1].numpy(), a["picks"]):
                    st = cand
    for r in rows[:3]:
        s0 = int(np.nonzero(a["picks"][r] != b["picks"][r])[0][0])
        acc = sum(st[s][a["picks"][r][s]].double().numpy() for s in range(s0)) if s0 else 0
        cb = st[s0].double().numpy()
        def d(x): return (((x.astype(np.float64) - acc)[None] - cb) ** 2).sum(-1)
        da, db = d(a["x"][r]), d(b["x"][r])
        pa, pb = a["picks"][r][s0], b["picks"][r][s0]
        out.append({"rvq_call": i, "tag": a["tag"], "diff_rows": int(len(rows)), "rows_in_call": int(len(a["picks"])), "stage": s0,
                    "max_abs_dx": float(np.abs(a["x"][r] - b["x"][r]).max()),
                    "gap_on_cpu_x": float(da[pb] - da[pa]), "gap_on_gpu_x": float(db[pb] - db[pa]),
                    "rel_gap_cpu": float((da[pb] - da[pa]) / da[pa])})
    if len(out) >= 6:
        break
print(json.dumps(out, indent=1))
