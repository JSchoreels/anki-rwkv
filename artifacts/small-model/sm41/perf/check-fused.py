import json, sys, glob, numpy as np
S = sys.argv[1]
def load(n):
    r = json.loads(open(f"{S}/{n}/row.jsonl").readline())
    z = np.load(glob.glob(f"{S}/{n}/*.npz")[0], allow_pickle=True)
    return r, z
def strip(l): return l.replace("-g0.5_0.2", "").replace("-gm0.5_m0.2", "")
TIMING = {"wall_seconds", "lane_reviews_per_second", "warmup_seconds", "finished"}
r0, z0 = load("ref_imp"); r1, z1 = load("dev_imp")
diff = [k for k in r0 if k not in TIMING and r0[k] != r1.get(k)]
print("dev == ref (global drift), row fields differing:", diff, "| npz keys", list(z0.keys()))
for k in z0.keys():
    a, b = z0[k], z1[k]
    print("  npz", k, a.shape, "identical" if (a.shape == b.shape and (np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else np.array_equal(a, b))) else "DIFF")
# fused vs separate: per lane, review by review
rf, zf = load("fused")
LOGC = ["day","card","kind","ed","q_es","rating","ms","p_true","ivl","aim_out","aim_prev","r_fsrs7","r_curve","t90_before","t90_after"]
for ref, tag in (("ref_none", None), ("ref_imp", "-g0.5_0.2"), ("ref_dec", "-gm0.5_m0.2")):
    rr, zr = load(ref)
    for i, ln in enumerate(rr["lanes"]):
        if ln == "tf":
            name = "tf"
        else:
            n, s = ln.split("#"); 
            name = (n.replace("-free", tag + "-free") if n.endswith("-free") else n + tag) + "#" + s if tag else ln
        j = rf["lanes"].index(name)
        mr, mf = zr["lane"] == i, zf["lane"] == j
        exact = []; worst = 0.0
        for c in LOGC:
            a, b = zr[c][mr], zf[c][mf]
            if a.shape != b.shape: exact.append(f"{c}:len {a.shape} vs {b.shape}"); continue
            if c in ("day","card","kind","ed","rating","ms","ivl"):
                if not np.array_equal(a, b): exact.append(c)
            else:
                ok = np.isfinite(a) & np.isfinite(b)
                if (np.isfinite(a) != np.isfinite(b)).any(): exact.append(c + ":nan")
                if ok.any(): worst = max(worst, float(np.max(np.abs(a[ok] - b[ok]))))
        accd = max((float(np.max(np.abs(np.array(rr["acc"][k][i]) - np.array(rf["acc"][k][j])) / np.maximum(1, np.abs(np.array(rr["acc"][k][i]))))) for k in rr["acc"]))
        mi = abs(rr["memorised_instant"][i] - rf["memorised_instant"][j])
        print(f"{ref:9s} {name:32s} reviews {mr.sum():5d} integer cols {'identical' if not exact else exact}  max float diff {worst:.1e}  acc rel {accd:.1e}  memorised diff {mi:.2e}")
