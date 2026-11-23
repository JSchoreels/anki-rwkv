"""SM149 stage 1 read-out: reuses the helpers of sm148_report.py (not edited). CPU only. Prints and writes sm149-stage1.json."""
import importlib.util, json, sys
from pathlib import Path
S148 = Path("C:/Users/jonat/workspace/wt/sm148/artifacts/small-model/sm148")
spec = importlib.util.spec_from_file_location("sm148_report", S148 / "sm148_report.py")
m = importlib.util.module_from_spec(spec); sys.modules["sm148_report"] = m; spec.loader.exec_module(m)
HERE = Path(__file__).resolve().parent
st = m.showtime()
E = m.E2S; L = m.LABELS
o_rows, _, _ = m.read_rows(S148 / "gpu/sm148-rows-732.jsonl")
n_rows, _, _ = m.read_rows(HERE / "gpu/sm149-rows-732.jsonl")
tag = "sm118-e26-dec2ep32k-t91s2"
users732 = set(json.load(open(m.U732))["users"])
def get(rows, pre, s): return rows[f"{pre}/{L[s]}{E}"]
orig = {s: get(o_rows, tag, s) for s in ("REF", "Q4", "T24", "Q2")}
qat = {s: get(n_rows, "sm149-sm118-qat", s) for s in ("REF", "Q2", "T24")}
ctl = {s: get(n_rows, "sm149-sm118-ctl", s) for s in ("REF", "Q2")}
common = set(orig["REF"])
for d in (qat, ctl):
    for r in d.values(): common &= set(r)
for r in orig.values(): common &= set(r)
common = [u for u in common if u in users732]
common = sorted(common)
print("common users", len(common))
H = ("curve", "instant")
def cmp(a, b): return m.dataset_cmp(a, b, H, common, st)
out = {"users": len(common), "r2_vs_orig_int8": {}, "r2_vs_own_int8": {}, "vs_orig_same_label": {}, "extra": {}}
out["r2_vs_orig_int8"]["QAT_REF"] = cmp(qat["REF"], orig["REF"])
out["r2_vs_orig_int8"]["CTL_REF"] = cmp(ctl["REF"], orig["REF"])
for s in ("Q2", "T24"):
    out["r2_vs_orig_int8"][f"QAT_{s}"] = cmp(qat[s], orig["REF"])
    out["r2_vs_own_int8"][f"QAT_{s}"] = cmp(qat[s], qat["REF"])
    out["vs_orig_same_label"][f"QAT_{s}"] = cmp(qat[s], orig[s])
out["r2_vs_orig_int8"]["CTL_Q2"] = cmp(ctl["Q2"], orig["REF"])
out["r2_vs_own_int8"]["CTL_Q2"] = cmp(ctl["Q2"], ctl["REF"])
out["vs_orig_same_label"]["CTL_Q2"] = cmp(ctl["Q2"], orig["Q2"])
out["extra"]["orig_Q2_vs_orig_REF"] = cmp(orig["Q2"], orig["REF"])
out["extra"]["orig_T24_vs_orig_REF"] = cmp(orig["T24"], orig["REF"])
out["extra"]["orig_Q4_vs_orig_REF"] = cmp(orig["Q4"], orig["REF"])
out["extra"]["QAT_Q2_vs_CTL_Q2"] = cmp(qat["Q2"], ctl["Q2"])
out["extra"]["QAT_REF_vs_CTL_REF"] = cmp(qat["REF"], ctl["REF"])
cq = out["r2_vs_own_int8"]["QAT_Q2"]["instant"]["value"]; cc = out["r2_vs_own_int8"]["CTL_Q2"]["instant"]["value"]
out["gate"] = {"C_qat": cq, "C_ctl": cc, "ratio": cq / cc, "verdict": "no effect" if cq > 0.5 * cc else "partial"}
r2 = {}
for k, v in out["r2_vs_orig_int8"].items():
    r2[k] = all(v[h]["ci"][1] <= m.CEIL_ALL for h in H)
out["R2_pass_vs_orig_int8"] = r2
out["R2_pass_vs_own_int8"] = {k: all(v[h]["ci"][1] <= m.CEIL_ALL for h in H) for k, v in out["r2_vs_own_int8"].items()}
# buckets
bk = {}
for name, new, ref in (("QAT_Q2_vs_orig_REF", qat["Q2"], orig["REF"]), ("CTL_Q2_vs_orig_REF", ctl["Q2"], orig["REF"]),
                       ("QAT_Q2_vs_QAT_REF", qat["Q2"], qat["REF"]), ("CTL_Q2_vs_CTL_REF", ctl["Q2"], ctl["REF"]),
                       ("orig_Q2_vs_orig_REF", orig["Q2"], orig["REF"])):
    nn = {u: new[u] for u in common}; rr = {u: ref[u] for u in common}
    bk[name] = {f"{lo}-{hi-1 if hi else ''}": {h: m.bucket_cmp(nn, rr, m.KEY[h], lo, hi) for h in H} for lo, hi in m.BUCKETS}
out["buckets"] = bk
json.dump(out, open(HERE / "sm149-stage1.json", "w"), indent=1)
def f(x): return m.fmt(x)
for sec in ("r2_vs_orig_int8", "r2_vs_own_int8", "vs_orig_same_label", "extra"):
    print("##", sec)
    for k, v in out[sec].items(): print(k, "| curve", f(v["curve"]), "| instant", f(v["instant"]))
print(out["gate"]); print(r2); print(out["R2_pass_vs_own_int8"])
for name, b in bk.items():
    print("##", name)
    for k, v in b.items(): print(k, v["instant"] and (f(v["instant"]), v["instant"]["reviews"]), "|", v["curve"] and f(v["curve"]))
