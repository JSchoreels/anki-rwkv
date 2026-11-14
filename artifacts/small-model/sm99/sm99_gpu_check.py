#!/usr/bin/env python
"""The show-time instant-only big model (SM99): the GPU pair check (one queue ticket, ~10 min; DESIGN.md in this folder).
Adapted from the instant-only small model (SM57)'s sm57_gpu_check.py to the big architecture.

  python -X utf8 artifacts/small-model/sm99/sm99_gpu_check.py [--out artifacts/small-model/sm99/gpu/check-gpu-sm99.json]

Trainer smokes (scripts/expertium_decay.py run(smoke=True): the folder's options and start checkpoint on the trainer's
smoke rows, 2 users x 512 reviews) on scratch copies, kept until compared:
  G1 the show-time big model (SM91 step 2)'s own epoch folder (wt/sm91s2/.../sm91s2-big-showtime, read only) trained
     by the wt/sm91s2 tree (the tree it trained with, 11d6c1a4's sources) and by this tree: complete, the trained
     weights (master.pth sha256 and result.json trained_parameter_sha256) bit for bit equal, no curve_loss_weight;
  G2 epoch 1 of SM99 (sm99-big-inst-e1) on this tree: complete, finite, result.json records curve_loss_weight 0.0, and
     the trained weights differ from G2a = the same folder with the weight removed (so the option reaches the trainer);
Scorer pair:
  G3 this tree's scorer on the GPU (cap 0.8), the show-time big model (SM91 step 2)'s master.pth, users 4676 and 2388,
     w8+s8, --split-existing --deleted-cards neighbour, e2s-query and normal inputs: the benchmark blocks the rules read
     (benchmark, benchmark_existing, benchmark_deleted: LogLoss, RMSE, AUC, sizes) and the label counts equal to its
     committed rows (gpu/sm91s2-inputs.jsonl, scored on the GPU from 11d6c1a4); the raw float means (curve / instant
     log loss) within 1e-6 (GPU reduction order; the first run saw 1.5e-8 on one user; on CPU K5 is bitwise).
Memory: result.json peak allocated / reserved MiB and the nvidia-smi max used MiB during each smoke.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import sm99_epoch  # noqa: E402

BASE_TREE = Path("C:/Users/jonat/workspace/wt/sm91s2")
SM91_ROWS = ROOT / "artifacts/small-model/sm91s2/gpu/sm91s2-inputs.jsonl"
USERS_G3 = (4676, 2388)


def nvidia_used():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return int(out.split()[0])
    except Exception:   # noqa: BLE001
        return None


class Peak:
    def __init__(self):
        self.values, self.stop = [nvidia_used() or 0], threading.Event()
        self.t = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.wait(2):
            v = nvidia_used()
            if v is not None:
                self.values.append(v)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.t.join()


def smoke(tree, folder, name, drop_weight=False):
    scratch = tree / "artifacts/small-model/smoke" / name
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    decay = json.loads((folder / "decay.json").read_text())
    if drop_weight:
        decay["options"].pop("curve_loss_weight")
    (scratch / "decay.json").write_text(json.dumps(decay, indent=1), encoding="utf-8")
    protocol = json.loads((folder / "protocol.json").read_text())
    protocol.get("full_run", {}).pop("lr_resume", None)
    protocol["full_run"]["control_dir"] = (scratch / "control").relative_to(tree).as_posix()
    (scratch / "protocol.json").write_text(json.dumps(protocol, indent=1), encoding="utf-8")
    code = ("import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "from scripts.expertium_decay import run; raise SystemExit(run(Path(sys.argv[2]), smoke=True))")
    env = dict(os.environ, RWKV_DELETED_CARDS="neighbour", PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    started = time.perf_counter()
    with Peak() as peak:
        r = subprocess.run([sys.executable, "-X", "utf8", "-c", code, str(tree), str(scratch)], cwd=tree, env=env,
                           capture_output=True, text=True)
    case = scratch / "expertium" / decay["track"]
    res = json.loads((case / "result.json").read_text()) if (case / "result.json").exists() else {}
    master = case / "master.pth"
    return scratch, {"tree": str(tree), "folder": folder.name, "drop_weight": drop_weight, "exit": r.returncode,
                     "status": res.get("status"), "updates": res.get("updates"), "reviews": res.get("review_presentations"),
                     "seconds": round(time.perf_counter() - started, 1),
                     "master_sha256": hashlib.sha256(master.read_bytes()).hexdigest() if master.exists() else None,
                     "trained_parameter_sha256": res.get("trained_parameter_sha256"),
                     "curve_loss_weight": res.get("curve_loss_weight"), "error": res.get("error"),
                     "peak_allocated_mib": res.get("peak_allocated_mib"), "peak_reserved_mib": res.get("peak_reserved_mib"),
                     "nvidia_smi_baseline_mib": peak.values[0], "nvidia_smi_max_used_mib": max(peak.values),
                     "stderr_tail": r.stderr[-1500:] if r.returncode else ""}


def g3(work):
    users = work / "users-g3.json"
    users.write_text(json.dumps({"users": list(USERS_G3)}))
    env = {k: v for k, v in os.environ.items() if k not in ("RWKV_DELETED_CARDS", "EXPERTIUM_END_TO_START")}
    env.update(HELDOUT_GPU_MEMORY_FRACTION="0.8", PYTHONDONTWRITEBYTECODE="1")
    committed = {}
    for line in open(SM91_ROWS, encoding="utf-8"):
        x = json.loads(line)
        if x["user_id"] in USERS_G3 and "benchmark" in x:
            committed[(x["label"], x["user_id"])] = x
    out, res, ok = work / "g3-rows.jsonl", {}, True
    out.unlink(missing_ok=True)
    with Peak() as peak:
        for inputs in ("e2s-query", "normal"):
            cmd = [sys.executable, "-X", "utf8", str(ROOT / "scripts/small_model_heldout.py"), "--pool", "732", "--users-file", str(users),
                   "--model", f"sm91s2-big-showtime=fork::{sm99_epoch.REF_CK}", "--only", "sm91s2-big-showtime/w8+s8",
                   "--inputs", inputs, "--split-existing", "--deleted-cards", "neighbour", "--out", str(out)]
            r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
            if r.returncode:
                ok = False
                res[f"exit_{inputs}"] = {"exit": r.returncode, "tail": (r.stdout + r.stderr)[-1500:]}
    keys = ("benchmark", "benchmark_existing", "benchmark_deleted", "reviews", "curve_labels", "instant_labels")
    floats = ("curve_log_loss", "instant_log_loss", "curve_raw_log_loss", "instant_rating_cross_entropy")
    got = [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
    pairs = {}
    for x in got:
        if "benchmark" not in x:
            continue
        c = committed.get((x["label"], x["user_id"]))
        diff = None if c is None else max(abs(x[k] - c[k]) for k in floats)
        same = c is not None and all(x.get(k) == c.get(k) for k in keys) and diff <= 1e-6
        pairs[f"{x['label']}/{x['user_id']}"] = {"benchmark_identical": c is not None and all(x.get(k) == c.get(k) for k in keys),
                                                  "max_abs_raw_mean_diff": diff, "instant_log_loss": x.get("instant_log_loss"),
                                                  "committed": None if c is None else c.get("instant_log_loss")}
        ok &= same
    ok &= len(pairs) == 2 * len(USERS_G3)
    res.update(pairs=pairs, nvidia_smi_max_used_mib=max(peak.values))
    return ok, res


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "gpu" / "check-gpu-sm99.json"))
    args = ap.parse_args()
    work = HERE / "work"
    work.mkdir(exist_ok=True)
    runs, scratches = {}, []
    e1 = sm99_epoch.folder(1)
    for key, tree, folder, name, drop in (("sm91s2_base_tree", BASE_TREE, sm99_epoch.REF, "sm99-gpucheck-sm91s2", False),
                                          ("sm91s2_this_tree", ROOT, sm99_epoch.REF, "sm99-gpucheck-sm91s2", False),
                                          ("e1_no_weight_this_tree", ROOT, e1, "sm99-gpucheck-e1-noweight", True),
                                          ("e1_this_tree", ROOT, e1, "sm99-gpucheck-e1", False)):
        scratch, runs[key] = smoke(tree, folder, name, drop)
        scratches.append(scratch)
        print(f"[sm99-gpu] {key}: " + json.dumps({k: v for k, v in runs[key].items() if k != "stderr_tail"}), flush=True)
    a, b, c, d = (runs[k] for k in ("sm91s2_base_tree", "sm91s2_this_tree", "e1_no_weight_this_tree", "e1_this_tree"))
    g3_ok, g3_res = g3(work)
    done = lambda x: x["exit"] == 0 and x["status"] == "complete"  # noqa: E731
    checks = {
        "G1_sm91s2_epoch_smoke_its_tree_equals_this_tree_bitwise": done(a) and done(b) and a["master_sha256"] is not None
        and a["master_sha256"] == b["master_sha256"] and a["trained_parameter_sha256"] == b["trained_parameter_sha256"]
        and a["curve_loss_weight"] is None and b["curve_loss_weight"] is None,
        "G2_epoch1_smoke_trains_with_weight0": done(c) and done(d) and d["curve_loss_weight"] == 0.0 and c["curve_loss_weight"] is None
        and d["master_sha256"] not in (None, c["master_sha256"]),
        "G3_gpu_scorer_rows_equal_committed_sm91s2_rows": g3_ok}
    ok = all(checks.values())
    if ok:
        for s in scratches:
            shutil.rmtree(s, ignore_errors=True)
    out = {"ok": ok, "checks": checks, "runs": runs, "scorer": g3_res, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"[sm99-gpu] {'ALL OK' if ok else 'FAIL'}: " + ", ".join(f"{k} {'OK' if v else 'FAIL'}" for k, v in checks.items())
          + f"; peak allocated MiB {b['peak_allocated_mib']} / {d['peak_allocated_mib']}; nvidia-smi max "
            f"{b['nvidia_smi_max_used_mib']} / {d['nvidia_smi_max_used_mib']} (baseline {b['nvidia_smi_baseline_mib']}); "
            f"scorer {g3_res.get('nvidia_smi_max_used_mib')}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
