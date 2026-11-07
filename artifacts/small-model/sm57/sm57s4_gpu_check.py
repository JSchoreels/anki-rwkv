#!/usr/bin/env python
"""The instant-only small model (SM57) stage 4: the GPU check (first ticket, ~2-3 min; DESIGN.md, "Stage 4 plan").

The trainer's smoke mode (2 users x 512 reviews, scratch copies; sm57_gpu_check.smoke), this tree:
  H1 stage 1's instant-only epoch-26 folder: the trained weights (master.pth sha256 and trained_parameter_sha256) bit for
     bit equal to stage 1's own GPU check of the same smoke (gpu/check-gpu-sm57.json, inst_this_tree, 11-06): the GPU and
     this tree (with the 1,500-user scorer option) still reproduce stage 1;
  H2 the stage-4 epoch-27 folder: complete, finite, curve_loss_weight 0.0 recorded, initial parameters == stage 1's
     epoch-26 trained parameters (the chain), weights differ from H1's;
  memory: result.json peak allocated / reserved MiB and the nvidia-smi max used MiB of each smoke.

  python -X utf8 artifacts/small-model/sm57/sm57s4_gpu_check.py --out artifacts/small-model/sm57/gpu/check-gpu-sm57-stage4.json
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sm57_gpu_check as G  # noqa: E402
import sm57_stage4 as S  # noqa: E402


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "gpu" / "check-gpu-sm57-stage4.json"))
    args = ap.parse_args()
    stage1 = json.loads((HERE / "gpu" / "check-gpu-sm57.json").read_text())["runs"]["inst_this_tree"]
    e26 = S.result_of(S.E26)
    runs, scratches = {}, []
    for key, folder, name in (("inst_e26_this_tree", S.E26, "sm57s4-gpucheck-e26"),
                              ("inst_e27_this_tree", S.folder(27), "sm57s4-gpucheck-e27")):
        scratch, runs[key] = G.smoke(G.ROOT, folder, name)
        res = scratch / "expertium" / G.TRACK / "result.json"
        runs[key]["initial_parameter_sha256"] = json.loads(res.read_text()).get("initial_parameter_sha256") if res.exists() else None
        scratches.append(scratch)
        print(f"[sm57s4-gpu] {key}: " + json.dumps({k: v for k, v in runs[key].items() if k != "stderr_tail"}), flush=True)
    a, b = runs["inst_e26_this_tree"], runs["inst_e27_this_tree"]
    checks = {
        "H1_e26_smoke_equals_stage1_gpu_check_bitwise": a["exit"] == 0 and a["status"] == "complete"
        and a["master_sha256"] == stage1["master_sha256"] and a["trained_parameter_sha256"] == stage1["trained_parameter_sha256"],
        "H2_e27_smoke_trains_weight0_from_e26": b["exit"] == 0 and b["status"] == "complete" and b["curve_loss_weight"] == 0.0
        and b["initial_parameter_sha256"] == e26["trained_parameter_sha256"] and b["master_sha256"] not in (None, a["master_sha256"])}
    ok = all(checks.values())
    if ok:
        for s in scratches:
            shutil.rmtree(s, ignore_errors=True)
    out = {"ok": ok, "checks": checks, "runs": runs, "stage1_reference": {k: stage1[k] for k in ("master_sha256", "trained_parameter_sha256")},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"[sm57s4-gpu] {'ALL OK' if ok else 'FAIL'}: " + ", ".join(f"{k} {'OK' if v else 'FAIL'}" for k, v in checks.items())
          + f"; peak allocated MiB {a['peak_allocated_mib']} / {b['peak_allocated_mib']}; nvidia-smi max "
            f"{a['nvidia_smi_max_used_mib']} / {b['nvidia_smi_max_used_mib']} (baseline {a['nvidia_smi_baseline_mib']})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
