#!/usr/bin/env python
"""The show-time error sensitivity test (SM100): the GPU scoring runner (ONE queue ticket, launch-sm100.sh).

For each input variant, in this order: e2s-query (shift 0), e2s-query+60s, e2s-query+10s, e2s-query+300s, normal
  scripts/small_model_heldout.py exactly as the show-time instant-only small model (SM96)'s launcher scored it
  (--pool 732 --users-file eval-users-732-fit.json (723 users) --only sm96-inst-e31st-st-dec1ep32k/w8+s8 --time-limit 1800
  --split-existing --deleted-cards neighbour, RWKV_DELETED_CARDS / EXPERTIUM_END_TO_START unset), plus --dump gpu/dump;
  pass 1 at HELDOUT_GPU_MEMORY_FRACTION 0.8; if users are still missing, one memory-only retry pass at 0.92 (same numerics).
Exit 0 when every variant has 723 users; 3 when some variant has fewer (the rule needs the 722 rule users of shift 0,
+10 s and +60 s); the scorer's own exit code if it failed. STOP file (gpu/STOP) honoured between passes.

  python sm100_score.py [--dry-run]
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
G = HERE / "gpu"
ROWS = G / "sm100-rows.jsonl"
DUMP = G / "dump"
STOP = G / "STOP"
SC = WT / "scripts" / "small_model_heldout.py"
CKPT = "C:/Users/jonat/workspace/wt/sm96/artifacts/small-model/sm96-inst-e31st-st-dec-1ep32k-muon/expertium/E89-F65e-F77a/master.pth"
USERS = "C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json"
CLBL = "sm96-inst-e31st-st-dec1ep32k"
VARIANTS = ("e2s-query", "e2s-query+60s", "e2s-query+10s", "e2s-query+300s", "normal")
EXPECT = 723


def label(v):
    return f"{CLBL}/w8+s8" + ("" if v == "normal" else f"@{v}")


def have(v):
    if not ROWS.exists():
        return 0
    lb = label(v)
    return len({r["user_id"] for r in map(json.loads, ROWS.read_text(encoding="utf-8").splitlines())
                if r.get("label") == lb and "benchmark" in r})


def log(msg):
    print(f"[sm100-score {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    dry = "--dry-run" in sys.argv
    G.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("RWKV_DELETED_CARDS", "EXPERTIUM_END_TO_START")}
    short = False
    for v in VARIANTS:
        for cap in ("0.8", "0.92"):
            n = have(v)
            if n >= EXPECT:
                break
            if STOP.exists():
                log("STOP file present"); return 1
            cmd = [sys.executable, "-X", "utf8", str(SC), "--pool", "732", "--users-file", USERS, "--model", f"{CLBL}=expertium:w64:{CKPT}",
                   "--only", f"{CLBL}/w8+s8", "--inputs", v, "--time-limit", "1800", "--split-existing", "--deleted-cards", "neighbour",
                   "--out", str(ROWS), "--dump", str(DUMP)]
            log(f"{v}: {n} of {EXPECT} users done; pass at cap {cap}")
            if dry:
                print("    " + " ".join(cmd)); break
            t0 = time.time()
            rc = subprocess.run(cmd, cwd=str(WT), env=env | {"HELDOUT_GPU_MEMORY_FRACTION": cap}).returncode
            log(f"{v}: exit {rc}, {have(v)} of {EXPECT} users, {(time.time() - t0) / 60:.1f} min")
            if rc:
                return rc
        if not dry and have(v) < EXPECT:
            log(f"{v}: only {have(v)} of {EXPECT} users after the 0.92 pass")
            short = True
    return 3 if short else 0


if __name__ == "__main__":
    raise SystemExit(main())
