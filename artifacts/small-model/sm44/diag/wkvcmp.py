"""First WKV-codec call whose reconstruction differs (CPU vs GPU logs of StateCodec.wkv in/out); explain the matrix.
usage: python wkvcmp.py WT CASES_JSON CASE CPU.wkv.npz GPU.wkv.npz [THRESH]"""
import sys
import json
WT, CJ, CASE, FA, FB = sys.argv[1:6]
TH = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-4
sys.path.insert(0, WT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
sys.argv = sys.argv[:1]
from scripts import state_vq  # noqa: E402

case = [c for c in json.load(open(CJ)) if c[0] == CASE][0]
books = state_vq.Codebooks.load(case[2])
recs = state_vq.recipe(case[3])
za, zb = np.load(FA), np.load(FB)
n = len(za.files) // 2
_rv = np.load(FA.replace(".wkv.npz", ""))
_tags = sorted({(int(k.split("/")[0]), k.split("/")[2]) for k in _rv.files if k.split("/")[2].startswith("wkv")})
WKV_TAGS = [t for _, t in _tags][::recs[0].rank]
assert n == len(zb.files) // 2
K = 16
stats = {"calls": n, "matrices": 0, "diff_matrices": 0}
first = None
for i in range(n):
    ia, oa, ib, ob = za[f"{i}/in"], za[f"{i}/out"], zb[f"{i}/in"], zb[f"{i}/out"]
    scale = np.abs(oa).reshape(oa.shape[0], oa.shape[1], -1).max(-1) + 1e-30
    d = np.abs(oa - ob).reshape(oa.shape[0], oa.shape[1], -1).max(-1) / scale
    din = np.abs(ia - ib).reshape(ia.shape[0], ia.shape[1], -1).max(-1) / (np.abs(ia).reshape(ia.shape[0], ia.shape[1], -1).max(-1) + 1e-30)
    stats["matrices"] += d.size
    bad = np.argwhere(d > TH)
    stats["diff_matrices"] += len(bad)
    if len(bad) and first is None:
        H = ia.shape[1]
        stream = int(WKV_TAGS[i].replace("wkv", ""))
        rec = recs[0 if stream == 0 else 1]
        codec = state_vq.StateCodec(rec, books)
        rows = []
        for r, h in bad[:50]:
            out = {"row": int(r), "head": int(h), "rel_out_diff": float(d[r, h]), "rel_in_diff": float(din[r, h])}
            for name, A in (("cpu", ia[r, h]), ("gpu", ib[r, h])):
                At = torch.from_numpy(A)[None].double()
                u, v, sig, val = state_vq.factorize(At, rec.rank)
                s = np.linalg.svd(A.astype(np.float64), compute_uv=False)
                comps = []
                for j in range(rec.rank):
                    q = torch.cat([u[:, j], v[:, j]], -1).float()
                    _, pk = state_vq.rvq_picks(q, codec.wkv_books[j])
                    comps.append({"valid": bool(val[0, j]), "picks": pk[0].tolist(), "q": q[0].numpy()})
                out[name] = {"sv_rel": (s[:rec.rank + 1] / s[0]).tolist(), "comps": comps}
            # which component differs, and how (direction change of q, or a pick flip with the same q)
            out["comp_diff"] = []
            for j in range(rec.rank):
                ca, cb = out["cpu"]["comps"][j], out["gpu"]["comps"][j]
                if ca["picks"] != cb["picks"] and (ca["valid"] or cb["valid"]):
                    stg = next(s for s in range(len(ca["picks"])) if ca["picks"][s] != cb["picks"][s])
                    # float64 squared-distance gap of the two picks at that stage, on the CPU q
                    acc = np.zeros(32)
                    for s in range(stg):
                        acc += codec.wkv_books[j][s][ca["picks"][s]].double().numpy()
                    cb_ = codec.wkv_books[j][stg].double().numpy()
                    res = ca["q"].astype(np.float64) - acc
                    dd = ((res[None] - cb_) ** 2).sum(-1)
                    out["comp_diff"].append({"comp": j, "stage": stg, "valid": [ca["valid"], cb["valid"]],
                                             "max_dq": float(np.abs(ca["q"] - cb["q"]).max()),
                                             "gap_sq_dist_cpu": float(dd[cb["picks"][stg]] - dd[ca["picks"][stg]]),
                                             "rel_gap": float((dd[cb["picks"][stg]] - dd[ca["picks"][stg]]) / dd[ca["picks"][stg]]),
                                             "sv_ratio_j_jp1": float(out["cpu"]["sv_rel"][j + 1] / out["cpu"]["sv_rel"][j]) if j + 1 < len(out["cpu"]["sv_rel"]) else None})
            for name in ("cpu", "gpu"):
                for c in out[name]["comps"]:
                    c.pop("q")
            rows.append(out)
        first = {"call": i, "shape": list(ia.shape), "n_bad": int(len(bad)), "rows": rows}
print(json.dumps({"stats": stats, "first": first}, indent=1, default=str))
