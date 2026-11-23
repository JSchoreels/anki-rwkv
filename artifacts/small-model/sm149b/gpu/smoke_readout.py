"""SM149b GPU smoke read-out: throughput of the new arm's first updates (scratch copy sm149b-timing-tail1) paired with
the same updates of SM149's QAT arm (its updates.jsonl: same users, windows and order), peak GPU memory, the tail log.
Writes gpu/smoke-sm149b.json."""
import json
import re
from pathlib import Path

R = Path("C:/Users/jonat/workspace/wt/sm149-run/artifacts/small-model")
HERE = Path(__file__).resolve().parent
TRACK = "E89-F65e-F77a"


def updates(folder):
    p = R / folder / "expertium" / TRACK / "updates.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


new = updates("sm149b-timing-tail1")
old = updates("sm149-sm118-qat")
n = 0
for a, b in zip(new, old):
    if (a["user_id"], a["offset"], a["reviews"]) != (b["user_id"], b["offset"], b["reviews"]):
        break
    n += 1
skip = 3   # the first updates include start-up compiles on both sides
pair_new, pair_old = new[skip:n], old[skip:n]
reviews = sum(u["reviews"] for u in pair_new)
gpu_new = sum(u["gpu_ms"] for u in pair_new) / 1000
gpu_old = sum(u["gpu_ms"] for u in pair_old) / 1000
out = {"what": "SM149b GPU smoke: new arm (lambda 1.0, stride 8) vs SM149's QAT arm, same first updates",
       "updates_new": len(new), "common_prefix_updates": n, "paired_updates_used": len(pair_new),
       "paired_reviews": reviews,
       "gpu_seconds": {"new": round(gpu_new, 1), "sm149_qat": round(gpu_old, 1)},
       "reviews_per_second_gpu": {"new": round(reviews / gpu_new), "sm149_qat": round(reviews / gpu_old)},
       "new_over_sm149_qat_gpu_time": round(gpu_new / gpu_old, 3),
       "loss_mean": {"new": sum(u["loss"] for u in pair_new) / len(pair_new),
                     "sm149_qat": sum(u["loss"] for u in pair_old) / len(pair_old)},
       "sm149_qat_full_run_reviews_per_second": 9993}
log = (HERE / "timing-tail1.out").read_text(errors="replace")
m = re.search(r"process peak GPU memory: allocated ([\d,]+) MiB, reserved ([\d,]+) MiB", log)
if m:
    out["peak_allocated_mib"], out["peak_reserved_mib"] = (int(x.replace(",", "")) for x in m.groups())
smi = [ln.split(",") for ln in (HERE / "timing-nvsmi.csv").read_text().splitlines() if "MiB" in ln]
if smi:
    out["nvidia_smi_used_mib_max"] = max(int(x[1].strip().split()[0]) for x in smi)
    out["nvidia_smi_util_mean"] = round(sum(int(x[2].strip().split()[0]) for x in smi) / len(smi), 1)
m = re.search(r"low-rank hook calls: (\{.*?\})", log)
out["hook_calls"] = json.loads(m.group(1)) if m else None
tail = [json.loads(x) for x in (R / "sm149b-timing-tail1" / "tail.jsonl").read_text().splitlines() if x.strip()] \
    if (R / "sm149b-timing-tail1" / "tail.jsonl").exists() else []
if tail:
    vals = [t["tail_mean"] for t in tail if t["tail_mean"] is not None]
    out["tail"] = {"passes": len(tail), "first_10_mean": sum(vals[:10]) / len(vals[:10]),
                   "last_10_mean": sum(vals[-10:]) / len(vals[-10:]),
                   "entries_mean": sum(t["entries"] for t in tail) / len(tail),
                   "penalty_over_loss_first_10": sum(vals[:10]) / len(vals[:10]) / out["loss_mean"]["new"]}
(HERE / "smoke-sm149b.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
