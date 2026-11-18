#!/usr/bin/env python
"""The instant-only small model (SM57) stage 3 (curve-only): the GPU check (first ticket, ~2 min; DESIGN.md "Stage 3 plan").

  python -X utf8 artifacts/small-model/sm57/sm57s3_gpu_check.py --out artifacts/small-model/sm57/gpu/check-gpu-sm57-stage3.json

The trainer's smoke mode (2 users x 512 reviews of epoch 1, sm57_gpu_check.smoke, real replay KD) by this tree:
  G1 stage 2's joint epoch-1 folder: complete, the trained weights (master.pth sha256 and trained_parameter_sha256) bit
     for bit those of stage 2's GPU check joint smoke (gpu/check-gpu-sm57-stage2.json): the GPU and this tree still
     reproduce the joint arm exactly;
  G2 the curve-only epoch-1 folder: complete, result.json options instant_loss_weight 0.0, weights differ from the
     joint smoke's and from stage 2's instant-only smoke's;
  G3 same init: initial_parameter_sha256 of both smokes == stage 2's == the leak-free small model (SM72s) epoch 1's;
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
    ap.add_argument("--out", default=str(HERE / "gpu" / "check-gpu-sm57-stage3.json"))
    args = ap.parse_args()
    stage2 = json.loads((HERE / "gpu" / "check-gpu-sm57-stage2.json").read_text())["runs"]
    runs, scratches = {}, []
    for key, folder, name in (("joint_this_tree", SM / "sm57s2-joint-e01", "sm57s3-gpucheck-joint"),
                              ("curve_this_tree", SM / "sm57s3-curve-e01", "sm57s3-gpucheck-curve")):
        scratch, runs[key] = GC.smoke(ROOT, folder, name)
        res = scratch / "expertium" / TRACK / "result.json"
        r = json.loads(res.read_text()) if res.exists() else {}
        runs[key]["initial_parameter_sha256"] = r.get("initial_parameter_sha256")
        runs[key]["instant_loss_weight"] = (r.get("options") or {}).get("instant_loss_weight")
        scratches.append(scratch)
        print(f"[sm57s3-gpu] {key}: " + json.dumps({k: v for k, v in runs[key].items() if k != "stderr_tail"}), flush=True)
    b, c = runs["joint_this_tree"], runs["curve_this_tree"]
    s2j, s2i = stage2["joint_this_tree"], stage2["inst_this_tree"]
    e1 = json.loads((RUN72S / "epoch-01" / "expertium" / TRACK / "result.json").read_text())["initial_parameter_sha256"]
    checks = {
        "G1_joint_smoke_equals_stage2_joint_smoke_bitwise": b["exit"] == 0 and b["status"] == "complete"
        and b["master_sha256"] is not None and b["master_sha256"] == s2j["master_sha256"]
        and b["trained_parameter_sha256"] == s2j["trained_parameter_sha256"] and b["instant_loss_weight"] is None,
        "G2_curve_only_smoke_trains_with_instant_weight0": c["exit"] == 0 and c["status"] == "complete"
        and c["instant_loss_weight"] == 0.0 and c["curve_loss_weight"] is None
        and c["master_sha256"] not in (None, b["master_sha256"], s2i["master_sha256"]),
        "G3_same_fresh_init_equals_stage2_and_sm72s_epoch1": b["initial_parameter_sha256"] == c["initial_parameter_sha256"]
        == s2j["initial_parameter_sha256"] == e1}
    ok = all(checks.values())
    if ok:
        for s in scratches:
            shutil.rmtree(s, ignore_errors=True)
    out = {"ok": ok, "checks": checks, "runs": runs, "stage2_joint_smoke": {k: s2j[k] for k in ("master_sha256", "trained_parameter_sha256")},
           "sm72s_epoch1_initial_parameter_sha256": e1, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"[sm57s3-gpu] {'ALL OK' if ok else 'FAIL'}: " + ", ".join(f"{k} {'OK' if v else 'FAIL'}" for k, v in checks.items())
          + f"; peak allocated MiB joint {b['peak_allocated_mib']}, curve-only {c['peak_allocated_mib']}; nvidia-smi max "
            f"{b['nvidia_smi_max_used_mib']} / {c['nvidia_smi_max_used_mib']} (baseline {b['nvidia_smi_baseline_mib']})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
