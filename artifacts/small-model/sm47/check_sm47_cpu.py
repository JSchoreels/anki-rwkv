"""SM47 CPU check on the 3 smallest 732-pool users -> check-sm47-cpu.json:
 (i) w8+s8 / w8+s4c8n of check-cpu.jsonl (this code) vs the committed SM45 CPU rows (exact, every logged metric);
 (ii) each new variant differs from its base (w8+s8 / w8+s4c8n);
 (iii) the CUDA dispatch run on CPU tensors (--emulate-gpu-path: state-VQ slots, IntScoreCodec, small_model_wkv.reference
      for the WKV) vs the CPU scorer (Hooks path, state_vq.int_row_quant) for the 6 new variants."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SM45 = HERE.parent / "sm45" / "check-cpu.jsonl"
NEW = ("s8h", "s8x8", "s8hx8", "s4c8nh", "s4c8nx8", "s4c8nhx8")
BASE = {v: "w8+s8" if v.startswith("s8") else "w8+s4c8n" for v in NEW}
METRICS = ("curve_log_loss", "curve_raw_log_loss", "instant_log_loss", "instant_rating_cross_entropy")


def rows(path):
    return {(r["user_id"], r["variant"]): r for r in map(json.loads, open(path, encoding="utf-8")) if "benchmark" in r}


def vals(r):
    return [r[m] for m in METRICS] + [r["benchmark"][h][k] for h in ("ahead", "instant")
                                      for k in ("LogLoss", "RMSE", "RMSE(bins)", "AUC", "size")]


cpu, emu, old = rows(HERE / "check-cpu.jsonl"), rows(HERE / "check-emu.jsonl"), rows(SM45)
users = sorted({u for u, _ in cpu})
out = {"users": users, "metrics": list(METRICS) + ["benchmark ahead/instant LogLoss, RMSE, RMSE(bins), AUC, size"]}
out["i_old_variants_reproduce_sm45"] = {v: all(vals(cpu[(u, v)]) == vals(old[(u, v)]) for u in users)
                                        for v in ("w8+s8", "w8+s4c8n")}
out["ii_new_vs_base"] = {}
for v in NEW:
    d = [cpu[(u, "w8+" + v)]["curve_log_loss"] - cpu[(u, BASE[v])]["curve_log_loss"] for u in users]
    b = [cpu[(u, "w8+" + v)]["benchmark"][h]["LogLoss"] - cpu[(u, BASE[v])]["benchmark"][h]["LogLoss"]
         for u in users for h in ("ahead", "instant")]
    out["ii_new_vs_base"]["w8+" + v] = {"base": BASE[v], "differs": any(x != 0 for x in d),
                                        "curve_log_loss_delta_per_user": d, "max_abs_benchmark_dlogloss": max(map(abs, b)),
                                        "entity_bits": cpu[(users[0], "w8+" + v)]["state_entity_bits"]}
out["iii_gpu_path_on_cpu_vs_cpu"] = {}
for v in NEW:
    k = [(u, "w8+" + v) for u in users]
    out["iii_gpu_path_on_cpu_vs_cpu"]["w8+" + v] = {
        "max_abs_dlogloss_benchmark": max(abs(cpu[x]["benchmark"][h]["LogLoss"] - emu[x]["benchmark"][h]["LogLoss"])
                                          for x in k for h in ("ahead", "instant")),
        "max_abs_d_curve_log_loss": max(abs(cpu[x]["curve_log_loss"] - emu[x]["curve_log_loss"]) for x in k),
        "max_abs_d_instant_log_loss": max(abs(cpu[x]["instant_log_loss"] - emu[x]["instant_log_loss"]) for x in k),
        "path": emu[k[0]].get("state_vq_path")}
worst = max(r["max_abs_dlogloss_benchmark"] for r in out["iii_gpu_path_on_cpu_vs_cpu"].values())
out["passed"] = (all(out["i_old_variants_reproduce_sm45"].values())
                 and all(r["differs"] for r in out["ii_new_vs_base"].values()) and worst <= 1e-5)
(HERE / "check-sm47-cpu.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, indent=1))
