#!/usr/bin/env python
"""Compact state for Tiny (SM129): CPU proofs before any GPU ticket (~3-8 min, < 2 GB RAM; the launcher runs it under
memguard.ps1) -> check-cpu-sm129.json.

  K1  the serialized bytes and the round trip (sm129_bytes.py): ALL OK
  K2  this tree's scorer (the show-time two-head cool-down for Tiny (SM124)'s tree + the second integer-state test
      (SM49)'s scorer commit 639e7ba1, three files) reproduces SM124's own tree on CPU for the unchanged variants
      (w8+s8, w8+s4c8n, w8+s4c8nhx8): every logged metric of 2 users of the 732 set (<= 4,200 reviews), show-time
      inputs, leak-free flags, SM124's final model (w64-c1), bitwise
  K3  the card-only arm (w8+s4hgx4c8hx8n) and the compact arm (w8+s4c8nhx8): the CUDA dispatch run on CPU tensors
      (--emulate-gpu-path: state-VQ slots, IntScoreCodec, the kernel's float32 twin) == the CPU scorer (Hooks path,
      int_row_quant) within 1e-5 benchmark LogLoss (SM49's check rule); the card-only arm differs from the compact arm
  K4  the rule's selftest (sm129_rule.py selftest) passes
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PY = sys.executable
W = Path("C:/Users/jonat/workspace/wt")
T124 = W / "sm124"
CK124 = W / "sm124/artifacts/small-model/sm124-c1-e29st-dec2ep32k-t91s2-p2/expertium/E89-F65e-F77a/master.pth"
USERS = W / "sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json"
OLD = ("w8+s8", "w8+s4c8n", "w8+s4c8nhx8")
NEW = ("w8+s4c8nhx8", "w8+s4hgx4c8hx8n")
COUNT = int(os.environ.get("SM129_CHECK_USERS", "2"))
METRICS = ("curve_log_loss", "curve_raw_log_loss", "instant_log_loss", "instant_rating_cross_entropy")


def score(tree, out, variants, emulate=False):
    env = dict(os.environ, PYTHONPATH="", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env.pop("RWKV_DELETED_CARDS", None)
    cmd = [PY, "-X", "utf8", str(Path(tree) / "scripts/small_model_heldout.py"), "--device", "cpu", "--pool", "732",
           "--users-file", str(USERS), "--max-reviews", "4200", "--count", str(COUNT), "--model", f"t={'expertium'}:w64-c1:{CK124}",
           "--variants", ",".join(variants), "--inputs", "e2s-query", "--split-existing", "--deleted-cards", "neighbour",
           "--out", str(out), "--time-limit", "3000"]
    if emulate:
        cmd.append("--emulate-gpu-path")
    r = subprocess.run(cmd, cwd=tree, capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError(f"scorer failed in {tree}: {r.stderr[-2000:]}")
    return {(x["user_id"], x["variant"]): x for x in map(json.loads, Path(out).read_text().splitlines()) if "benchmark" in x}


def vals(r):
    return [r[m] for m in METRICS] + [r["benchmark"][h][k] for h in ("ahead", "instant")
                                      for k in ("LogLoss", "RMSE", "RMSE(bins)", "AUC", "size")]


def main():
    t0 = time.perf_counter()
    rep, fails = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}, []

    def check(name, ok, **detail):
        rep[name] = dict(detail, ok=bool(ok))
        if not ok:
            fails.append(name)
        print(("PASS " if ok else "FAIL ") + name, flush=True)

    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm129_bytes.py")], cwd=ROOT, capture_output=True, text=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    check("K1_bytes_round_trip", r.returncode == 0, tail=r.stdout.strip().splitlines()[-3:])
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        a = score(T124, t / "old.jsonl", OLD)
        b = score(ROOT, t / "new.jsonl", OLD)
        keys = sorted(a)
        check("K2_unchanged_variants_bitwise_vs_sm124_tree", len(keys) == COUNT * len(OLD) and keys == sorted(b)
              and all(vals(a[k]) == vals(b[k]) for k in keys),
              users=sorted({k[0] for k in keys}), reviews=sorted({a[k]["reviews"] for k in keys}), variants=list(OLD),
              equal={f"{u}/{v}": vals(a[(u, v)]) == vals(b.get((u, v), {"benchmark": {}})) if (u, v) in b else False for u, v in keys})
        cpu = score(ROOT, t / "cpu.jsonl", NEW)
        emu = score(ROOT, t / "emu.jsonl", NEW, emulate=True)
        ks = sorted(cpu)
        worst = max(abs(cpu[k]["benchmark"][h]["LogLoss"] - emu[k]["benchmark"][h]["LogLoss"]) for k in ks for h in ("ahead", "instant"))
        users = sorted({k[0] for k in ks})
        differs = any(vals(cpu[(u, NEW[1])]) != vals(cpu[(u, NEW[0])]) for u in users)
        check("K3_gpu_path_on_cpu_vs_cpu", len(ks) == COUNT * len(NEW) and sorted(emu) == ks and worst <= 1e-5 and differs,
              max_abs_dlogloss_benchmark=worst, card_only_differs_from_compact=differs,
              entity_bits={v: cpu[(users[0], v)].get("state_entity_bits") for v in NEW},
              dlogloss_card_only_minus_compact={f"{u}/{h}": round(cpu[(u, NEW[1])]["benchmark"][h]["LogLoss"]
                                                                   - cpu[(u, NEW[0])]["benchmark"][h]["LogLoss"], 6)
                                                for u in users for h in ("ahead", "instant")})
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm129_rule.py"), "selftest"], cwd=ROOT, capture_output=True, text=True)
    check("K4_rule_selftest", r.returncode == 0, tail=(r.stdout.strip().splitlines() or [r.stderr[-400:]])[-1][-900:])
    rep["fails"], rep["all_ok"], rep["seconds"] = fails, not fails, round(time.perf_counter() - t0, 1)
    (HERE / "check-cpu-sm129.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print("ALL OK" if not fails else f"FAILED: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
