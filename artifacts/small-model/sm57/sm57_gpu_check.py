#!/usr/bin/env python
"""The instant-only small model (SM57): the GPU check (first ticket, ~5 min; DESIGN.md in this folder).

  python -X utf8 artifacts/small-model/sm57/sm57_gpu_check.py --out artifacts/small-model/sm57/gpu/check-gpu-sm57.json

The trainer's smoke mode (scripts/expertium_decay.py run(smoke=True): the folder's options and start checkpoint on the
trainer's smoke rows, 2 users x 512 reviews), on scratch copies that are kept until compared:
  G1 the joint epoch-26 folder trained by the wt/sm92 tree (BASE, before the option) and by this tree: complete, the
     trained weights (master.pth sha256 and result.json trained_parameter_sha256) bit for bit equal;
  G2 the instant-only epoch-26 folder trained by this tree: complete, finite, result.json records curve_loss_weight 0.0,
     the weights differ from G1's;
  memory: result.json peak allocated / reserved MiB and the nvidia-smi max used MiB during each smoke.
Scratch folders: <tree>/artifacts/small-model/smoke/sm57-gpucheck-{joint,inst}, deleted after the comparison.
"""
from argparse import ArgumentParser
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE_TREE = Path("C:/Users/jonat/workspace/wt/sm92")
TRACK = "E89-F65e-F77a"
SM = ROOT / "artifacts/small-model"


def nvidia_used():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return int(out.split()[0])
    except Exception:   # noqa: BLE001
        return None


def smoke(tree, folder, name):
    scratch = tree / "artifacts/small-model/smoke" / name
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    decay = json.loads((folder / "decay.json").read_text())
    dist = decay["options"].get("distillation")
    if isinstance(dist, dict):
        decay["options"]["distillation"] = {k: v for k, v in dist.items() if k != "cache"}
    (scratch / "decay.json").write_text(json.dumps(decay, indent=1), encoding="utf-8")
    protocol = json.loads((folder / "protocol.json").read_text())
    protocol.get("full_run", {}).pop("lr_resume", None)
    protocol["full_run"]["control_dir"] = (scratch / "control").relative_to(tree).as_posix()
    (scratch / "protocol.json").write_text(json.dumps(protocol, indent=1), encoding="utf-8")
    code = ("import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "from scripts.expertium_decay import run; raise SystemExit(run(Path(sys.argv[2]), smoke=True))")
    env = dict(os.environ, RWKV_DELETED_CARDS="neighbour", PYTHONPATH="")
    peak, stop = [nvidia_used() or 0], threading.Event()

    def sample():
        while not stop.wait(2):
            v = nvidia_used()
            if v is not None:
                peak.append(v)
    t = threading.Thread(target=sample, daemon=True)
    t.start()
    started = time.perf_counter()
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", code, str(tree), str(scratch)], cwd=tree, env=env,
                       capture_output=True, text=True)
    stop.set()
    t.join()
    case = scratch / "expertium" / TRACK
    res = json.loads((case / "result.json").read_text()) if (case / "result.json").exists() else {}
    master = case / "master.pth"
    return scratch, {"tree": str(tree), "folder": folder.name, "exit": r.returncode, "status": res.get("status"),
                     "updates": res.get("updates"), "reviews": res.get("review_presentations"),
                     "seconds": round(time.perf_counter() - started, 1),
                     "master_sha256": hashlib.sha256(master.read_bytes()).hexdigest() if master.exists() else None,
                     "trained_parameter_sha256": res.get("trained_parameter_sha256"),
                     "curve_loss_weight": res.get("curve_loss_weight"), "error": res.get("error"),
                     "peak_allocated_mib": res.get("peak_allocated_mib"), "peak_reserved_mib": res.get("peak_reserved_mib"),
                     "nvidia_smi_baseline_mib": peak[0], "nvidia_smi_max_used_mib": max(peak),
                     "stderr_tail": r.stderr[-1500:] if r.returncode else ""}


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "gpu" / "check-gpu-sm57.json"))
    args = ap.parse_args()
    runs, scratches = {}, []
    for key, tree, folder, name in (("joint_base_tree", BASE_TREE, SM / "sm57-joint-e26", "sm57-gpucheck-joint"),
                                    ("joint_this_tree", ROOT, SM / "sm57-joint-e26", "sm57-gpucheck-joint"),
                                    ("inst_this_tree", ROOT, SM / "sm57-inst-e26", "sm57-gpucheck-inst")):
        scratch, runs[key] = smoke(tree, folder, name)
        scratches.append(scratch)
        print(f"[sm57-gpu] {key}: " + json.dumps({k: v for k, v in runs[key].items() if k != "stderr_tail"}), flush=True)
    a, b, c = runs["joint_base_tree"], runs["joint_this_tree"], runs["inst_this_tree"]
    checks = {
        "G1_joint_smoke_base_tree_equals_this_tree_bitwise": all(x["exit"] == 0 and x["status"] == "complete" for x in (a, b))
        and a["master_sha256"] is not None and a["master_sha256"] == b["master_sha256"]
        and a["trained_parameter_sha256"] == b["trained_parameter_sha256"] and b["curve_loss_weight"] is None,
        "G2_instant_only_smoke_trains_with_weight0": c["exit"] == 0 and c["status"] == "complete" and c["curve_loss_weight"] == 0.0
        and c["master_sha256"] not in (None, b["master_sha256"])}
    ok = all(checks.values())
    if ok:
        for s in scratches:
            shutil.rmtree(s, ignore_errors=True)
    out = {"ok": ok, "checks": checks, "runs": runs, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"[sm57-gpu] {'ALL OK' if ok else 'FAIL'}: " + ", ".join(f"{k} {'OK' if v else 'FAIL'}" for k, v in checks.items())
          + f"; peak allocated MiB joint {b['peak_allocated_mib']}, instant-only {c['peak_allocated_mib']}; nvidia-smi max "
            f"{b['nvidia_smi_max_used_mib']} / {c['nvidia_smi_max_used_mib']} (baseline {b['nvidia_smi_baseline_mib']})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
