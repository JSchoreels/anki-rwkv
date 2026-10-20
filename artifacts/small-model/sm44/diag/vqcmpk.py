"""Fused-kernel pick log (vqdiag fused) vs CPU torch-loop log: first divergence + gap census of all diverging picks
before propagation. usage: python vqcmpk.py WT CASE CPU.npz FUSED.npz"""
import sys
import json
WT, CASE, FA, FB = sys.argv[1:5]
sys.path.insert(0, WT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
sys.argv = sys.argv[:1]
from scripts import check_state_vq_score_gpu as chk  # noqa: E402
from scripts import state_vq  # noqa: E402

case = [c for c in chk.CASES if c[0] == CASE][0]
books = state_vq.Codebooks.load(case[2])
codecs = state_vq.score_codecs(case[3], books, "cpu", torch.float64, kernel=False)


def load(f):
    z = np.load(f)
    calls = {}
    for key in z.files:
        i, kind, *tag, what = key.split("/")
        calls.setdefault(int(i), {"tag": "/".join(tag), "kind": kind})[what] = z[key]
    return [calls[i] for i in sorted(calls)]


def stage_gaps(x, picks, st):
    acc = np.zeros(x.shape, dtype=np.float64)
    out = []
    for s, cb in enumerate(st):
        cb = cb.double().numpy()
        d = ((x[None].astype(np.float64) - acc[None] - cb) ** 2).sum(-1)
        o = np.argsort(d)
        out.append((int(o[0]), int(o[1]), float(d[o[1]] - d[o[0]]), float(d[o[0]])))
        acc = acc + cb[picks[s]]
    return out


A, B = load(FA), load(FB)
ai = 0
first = None
diff_entries = 0
for b in B:
    if b["kind"] == "rvq":
        a = A[ai]
        ai += 1
        assert a["tag"] == b["tag"] and a["picks"].shape == b["picks"].shape, (a["tag"], b["tag"])
        rows = np.nonzero((a["picks"] != b["picks"]).any(-1))[0]
        if len(rows) and first is None:
            first = {"where": f"cpu call {ai - 1} {a['tag']}", "rows": len(rows), "per_row": []}
            xt = torch.from_numpy(a["x"])
            st = None
            for c in codecs.values():
                for pos in c.shift_books.values():
                    for cand in pos:
                        if cand[0].shape[1] == a["x"].shape[1] and len(cand) == a["picks"].shape[1] and                                 np.array_equal(state_vq.rvq_picks(xt, cand)[1].numpy(), a["picks"]):
                            st = cand
            for rr in rows:
                s0 = int(np.nonzero(b["picks"][rr] != a["picks"][rr])[0][0])
                gg = stage_gaps(a["x"][rr], a["picks"][rr], st)[s0]
                first["per_row"].append({"stage": s0, "cpu_best": gg[0], "second": gg[1], "gpu_pick": int(b["picks"][rr, s0]),
                                         "gap_sq_dist": gg[2], "rel_gap": gg[2] / max(gg[3], 1e-30),
                                         "max_abs_dx": float(np.abs(a["x"][rr] - b["x"][rr]).max())})
        continue
    kp = b["picks"]                               # [B, T, H, S]
    slot = int(b["tag"].replace("wkv", ""))
    st = codecs[slot].wkv_books[0]
    Bn, T, H, S = kp.shape
    for t in range(T):
        rows = np.nonzero(kp[:, t, 0, 0] >= 0)[0]
        if not len(rows):
            continue
        a = A[ai]
        ai += 1
        assert a["tag"] == b["tag"], (a["tag"], b["tag"])
        g = kp[rows, t].reshape(-1, S)
        assert g.shape == a["picks"].shape, (g.shape, a["picks"].shape)
        bad = np.nonzero((g != a["picks"]).any(-1))[0]
        if len(bad) and first is None:
            r = int(bad[0])
            first = {"where": f"cpu call {ai - 1} {a['tag']} (kernel step t={t})", "rows_in_step": int(g.shape[0]),
                     "diff_rows": int(len(bad)), "entity_row": int(rows[r // H]), "head": int(r % H),
                     "picks_cpu": a["picks"][r].tolist(), "picks_kernel": g[r].tolist(),
                     "per_row": []}
            for rr in bad:
                s0 = int(np.nonzero(g[rr] != a["picks"][rr])[0][0])
                gg = stage_gaps(a["x"][rr], a["picks"][rr], st)[s0]
                first["per_row"].append({"stage": s0, "cpu_best": gg[0], "second": gg[1], "kernel_pick": int(g[rr, s0]),
                                         "gap_sq_dist": gg[2], "rel_gap": gg[2] / max(gg[3], 1e-30)})
assert ai == len(A), (ai, len(A))
print(json.dumps(first, indent=1))
