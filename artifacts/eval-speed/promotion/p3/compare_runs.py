"""EV07: compare the two smoke runs (fast_evaluation false/true): training identity, result.json schema, metrics, page data."""
import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = HERE.parents[3]
sys.path.insert(0, str(W))
from scripts.run_full_training import page_state

ref_run, fast_run = HERE / "run-reference", HERE / "run-fast"
case = "epoch-01/expertium/E89-F65e-F77a/result.json"
a, b = (json.loads((run / case).read_text()) for run in (ref_run, fast_run))


def keys(value, prefix=""):
    out = set()
    if isinstance(value, dict):
        for k, v in value.items():
            if k in ("per_user",):
                out.add(prefix + k)
                continue
            out.add(prefix + k)
            out |= keys(v, prefix + k + ".")
    return out


training = {k: (a.get(k), b.get(k), a.get(k) == b.get(k)) for k in
            ("status", "trained_parameter_sha256", "initial_parameter_sha256", "updates", "review_presentations",
             "training_curve_labels", "training_instant_labels", "update_offset_end")}
ka, kb = keys(a), keys(b)
per_user_keys = (sorted(set(json.dumps(sorted(keys(u))) for u in a["per_user"])), sorted(set(json.dumps(sorted(keys(u))) for u in b["per_user"])))
out = HERE / "compare-eval.json"
subprocess.run([sys.executable, str(W / "scripts/check_fast_evaluation.py"), "compare", "--reference", str(ref_run / case),
                "--candidate", str(fast_run / case), "--output", str(out)], check=False, stdout=subprocess.DEVNULL)
evaluation = json.loads(out.read_text())
pa, pb = page_state(ref_run), page_state(fast_run)
pka, pkb = keys(pa), keys(pb)


def page_numbers(state):
    return {f"epoch{row['epoch']}.{k}": v for row in state["epochs"] for k, v in (row["evaluation"] or {}).items()
            if isinstance(v, float)}


na, nb = page_numbers(pa), page_numbers(pb)
report = {"training_identical": all(v[2] for v in training.values()), "training": training,
          "result_keys_only_reference": sorted(ka - kb), "result_keys_only_fast": sorted(kb - ka),
          "per_user_keys_equal": per_user_keys[0] == per_user_keys[1],
          "evaluation_mode": (a.get("evaluation_mode"), b.get("evaluation_mode")),
          "validation_seconds": (a.get("validation_seconds"), b.get("validation_seconds")),
          "validation_input_sha256": (a.get("validation_input_sha256"), b.get("validation_input_sha256")),
          "tolerance": evaluation["tolerance"],
          "page_keys_only_reference": sorted(pka - pkb), "page_keys_only_fast": sorted(pkb - pka),
          "page_max_abs_delta": max((abs(na[k] - nb[k]) for k in na if k in nb), default=0.0),
          "page_estimates": (pa["estimates"], pb["estimates"]), "page_status": (pa["status"], pb["status"])}
(HERE / "compare-runs.json").write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k not in ("training",)}, default=str)[:2500])
