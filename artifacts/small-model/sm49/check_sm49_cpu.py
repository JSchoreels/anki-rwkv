"""SM49 CPU check on the 3 users of `--pool 732 --count 3` (1425, 1893, 2388) -> check-sm49-cpu.json:
 (i) w8+s8, w8+s4c8n, w8+s8hx8, w8+s4c8nhx8 of check-cpu.jsonl (this code) vs SM47's committed CPU rows, and
     w8+s4c8nhx8 of check-emu.jsonl vs SM47's emulated rows (exact, every logged metric);
 (ii) each new variant differs from SM47's w8+s4c8nhx8 and carries the expected entity bits;
 (iii) the CUDA dispatch run on CPU tensors (--emulate-gpu-path: state-VQ slots, IntScoreCodec, small_model_wkv.reference
      for the WKV) vs the CPU scorer (Hooks path, state_vq.int_row_quant) for the 4 new variants (exact)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SM47 = Path("C:/Users/jonat/workspace/wt/sm45/artifacts/small-model/sm47")
OLD = ("w8+s8", "w8+s4c8n", "w8+s8hx8", "w8+s4c8nhx8")
# new variant -> expected (card, note) bits
NEW = {"w8+s4hx8": (11_824, 6_176), "w8+s4hx4": (11_056, 5_664), "w8+s4hgx4": (9_136, 4_704),
       "w8+s4hgx4c8hx8n": (9_136, 10_272)}
METRICS = ("curve_log_loss", "curve_raw_log_loss", "instant_log_loss", "instant_rating_cross_entropy")


def rows(path):
    return {(r["user_id"], r["variant"]): r for r in map(json.loads, open(path, encoding="utf-8")) if "benchmark" in r}


def vals(r):
    return [r[m] for m in METRICS] + [r["benchmark"][h][k] for h in ("ahead", "instant")
                                      for k in ("LogLoss", "RMSE", "RMSE(bins)", "AUC", "size")]


cpu, emu = rows(HERE / "check-cpu.jsonl"), rows(HERE / "check-emu.jsonl")
old_cpu, old_emu = rows(SM47 / "check-cpu.jsonl"), rows(SM47 / "check-emu.jsonl")
users = sorted({u for u, _ in cpu})
out = {"users": users, "metrics": list(METRICS) + ["benchmark ahead/instant LogLoss, RMSE, RMSE(bins), AUC, size"]}
out["i_old_variants_reproduce_sm47"] = {v: all(vals(cpu[(u, v)]) == vals(old_cpu[(u, v)]) for u in users) for v in OLD}
out["i_old_variants_reproduce_sm47"]["emulated w8+s4c8nhx8"] = all(
    vals(emu[(u, "w8+s4c8nhx8")]) == vals(old_emu[(u, "w8+s4c8nhx8")]) for u in users)
out["ii_new_vs_sm47_smallest"] = {}
for v, bits in NEW.items():
    base = "w8+s4c8nhx8"
    b = [cpu[(u, v)]["benchmark"][h]["LogLoss"] - cpu[(u, base)]["benchmark"][h]["LogLoss"]
         for u in users for h in ("ahead", "instant")]
    eb = cpu[(users[0], v)]["state_entity_bits"]
    out["ii_new_vs_sm47_smallest"][v] = {
        "differs": any(vals(cpu[(u, v)]) != vals(cpu[(u, base)]) for u in users),
        "benchmark_dlogloss_per_user_head": b, "entity_bits": eb, "bits_as_expected": (eb["card"], eb["note"]) == bits,
        "state_fields": {k: cpu[(users[0], v)].get(k) for k in ("state_bits", "state_scale_bits", "state_shift_bits",
                                                                 "state_scale_per")}}
out["iii_gpu_path_on_cpu_vs_cpu"] = {}
for v in NEW:
    k = [(u, v) for u in users]
    out["iii_gpu_path_on_cpu_vs_cpu"][v] = {
        "exact": all(vals(cpu[x]) == vals(emu[x]) for x in k),
        "max_abs_dlogloss_benchmark": max(abs(cpu[x]["benchmark"][h]["LogLoss"] - emu[x]["benchmark"][h]["LogLoss"])
                                          for x in k for h in ("ahead", "instant")),
        "max_abs_d_curve_log_loss": max(abs(cpu[x]["curve_log_loss"] - emu[x]["curve_log_loss"]) for x in k),
        "max_abs_d_instant_log_loss": max(abs(cpu[x]["instant_log_loss"] - emu[x]["instant_log_loss"]) for x in k),
        "path": emu[k[0]].get("state_vq_path")}
worst = max(r["max_abs_dlogloss_benchmark"] for r in out["iii_gpu_path_on_cpu_vs_cpu"].values())
out["passed"] = (all(out["i_old_variants_reproduce_sm47"].values())
                 and all(r["differs"] and r["bits_as_expected"] for r in out["ii_new_vs_sm47_smallest"].values())
                 and worst <= 1e-5)
(HERE / "check-sm49-cpu.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, indent=1))
