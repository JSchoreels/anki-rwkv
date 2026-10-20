"""SM47 diagnostic: wraps small_model_wkv.round_scale_f16 during a CPU scorer run and prints how many float16 row /
shift-vector scales (absmax / qmax) are float16 subnormals (< 2^-14) or underflow to 0 (then 1: the row is stored as
zeros). usage: python f16_scale_stats.py scripts/small_model_heldout.py --device cpu ... --only TAG/w8+s8hx8"""
import sys, runpy, torch
sys.path.insert(0, "C:/Users/jonat/workspace/wt/sm45")
from scripts import small_model_wkv as m, state_vq
stats = {"n": 0, "sub": 0, "zero": 0, "min": 1e9, "max": 0}
orig = m.round_scale_f16
def wrap(scale):
    s = scale.flatten()
    stats["n"] += s.numel(); stats["sub"] += int(((s > 0) & (s < 2**-14)).sum()); stats["zero"] += int((s == 0).sum())
    pos = s[s > 0]
    if pos.numel(): stats["min"] = min(stats["min"], float(pos.min())); stats["max"] = max(stats["max"], float(s.max()))
    r = orig(scale); stats["underflow_to_zero"] = stats.get("underflow_to_zero", 0) + int(((r == 0) & (s.view_as(r) > 0)).sum()); return r
m.round_scale_f16 = wrap
_inst = state_vq.install
def inst(model, *a, **k):
    lay = _inst(model, *a, **k); print("layout", {s: (i["wkv_layers"], i["heads"], i["shift_vectors"]) for s, i in lay.items()}); return lay
state_vq.install = inst
sys.argv = sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    print("f16 scale stats", stats)
