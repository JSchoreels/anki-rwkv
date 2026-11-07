#!/usr/bin/env python
"""The instant-only small model (SM57) stage 2: the GPU check (first ticket, ~5 min; DESIGN.md "Stage 2 plan").

  python -X utf8 artifacts/small-model/sm57/sm57s2_gpu_check.py --out artifacts/small-model/sm57/gpu/check-gpu-sm57-stage2.json

The trainer's smoke mode (2 users x 512 reviews, sm57_gpu_check.smoke) of the epoch-1 folders (fresh init):
  G1 the joint epoch-1 folder trained by the wt/sm92 tree (before the option) and by this tree: complete, trained weights
     (master.pth sha256 and trained_parameter_sha256) bit for bit equal;
  G2 the instant-only epoch-1 folder trained by this tree: complete, curve_loss_weight 0.0 recorded, weights differ;
  G3 same init: initial_parameter_sha256 of all three smokes equal, and equal to the leak-free small model (SM72s)
     epoch 1's (the release's own fresh init);
  memory: peak allocated / reserved MiB and nvidia-smi max per smoke.
"""
from argparse import ArgumentParser
import importlib.util
import json
from pathlib import Path
import shutil
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
_spec = importlib.util.spec_from_file_location("sm57_gpu_check", HERE / "sm57_gpu_check.py")
GC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GC)
SM = ROOT / "artifacts/small-model"
TRACK = "E89-F65e-F77a"
RUN72S = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run")


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "gpu" / "check-gpu-sm57-stage2.json"))
    args = ap.parse_args()
    runs, scratches = {}, []
    for key, tree, folder, name in (("joint_base_tree", GC.BASE_TREE, SM / "sm57s2-joint-e01", "sm57s2-gpucheck-joint"),
                                    ("joint_this_tree", ROOT, SM / "sm57s2-joint-e01", "sm57s2-gpucheck-joint"),
                                    ("inst_this_tree", ROOT, SM / "sm57s2-inst-e01", "sm57s2-gpucheck-inst")):
        scratch, runs[key] = GC.smoke(tree, folder, name)
        res = scratch / "expertium" / TRACK / "result.json"
        runs[key]["initial_parameter_sha256"] = json.loads(res.read_text()).get("initial_parameter_sha256") if res.exists() else None
        scratches.append(scratch)
        print(f"[sm57s2-gpu] {key}: " + json.dumps({k: v for k, v in runs[key].items() if k != "stderr_tail"}), flush=True)
    a, b, c = runs["joint_base_tree"], runs["joint_this_tree"], runs["inst_this_tree"]
    e1 = json.loads((RUN72S / "epoch-01" / "expertium" / TRACK / "result.json").read_text())["initial_parameter_sha256"]
    checks = {
        "G1_joint_smoke_base_tree_equals_this_tree_bitwise": all(x["exit"] == 0 and x["status"] == "complete" for x in (a, b))
        and a["master_sha256"] is not None and a["master_sha256"] == b["master_sha256"]
        and a["trained_parameter_sha256"] == b["trained_parameter_sha256"] and b["curve_loss_weight"] is None,
        "G2_instant_only_smoke_trains_with_weight0": c["exit"] == 0 and c["status"] == "complete" and c["curve_loss_weight"] == 0.0
        and c["master_sha256"] not in (None, b["master_sha256"]),
        "G3_same_fresh_init_all_smokes_equals_sm72s_epoch1": a["initial_parameter_sha256"] == b["initial_parameter_sha256"]
        == c["initial_parameter_sha256"] == e1}
    ok = all(checks.values())
    if ok:
        for s in scratches:
            shutil.rmtree(s, ignore_errors=True)
    out = {"ok": ok, "checks": checks, "runs": runs, "sm72s_epoch1_initial_parameter_sha256": e1,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"[sm57s2-gpu] {'ALL OK' if ok else 'FAIL'}: " + ", ".join(f"{k} {'OK' if v else 'FAIL'}" for k, v in checks.items())
          + f"; peak allocated MiB joint {b['peak_allocated_mib']}, instant-only {c['peak_allocated_mib']}; nvidia-smi max "
            f"{b['nvidia_smi_max_used_mib']} / {c['nvidia_smi_max_used_mib']} (baseline {b['nvidia_smi_baseline_mib']})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
