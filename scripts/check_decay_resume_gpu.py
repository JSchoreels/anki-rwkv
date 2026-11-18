#!/usr/bin/env python
"""GPU check of the resumable decay (one queue ticket, ~15 min): the same small decay twice from one source epoch,
  A uninterrupted, B stopped with `expertium_decay.py stop` once half of its users are trained, then `run` again
  (resume). Compares, update by update (user, offset): batch seed, learning rate and loss, plus the final weights.
  B must stop with a snapshot, resume from it, skip its done users, end at the same learning rate as A, and match A's
  loss per update (bitwise when no user was in flight at the stop) and its final weights.

  python scripts/check_decay_resume_gpu.py --source-run RUN --epoch K --extra-options JSON --out DIR [--fraction 0.003]
  (prepare arguments as the real decay: --fresh-optimizer --train-tier final --evaluate none --seed 1 are fixed here;
  DIR under artifacts/, removed and rebuilt; writes DIR/check.json; exit 1 on a failure)"""
from argparse import ArgumentParser
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEC = ROOT / "scripts/expertium_decay.py"
TRIM = ROOT / "scripts/check_replay_label_windows.py"
CHECKS = []


def record(name, ok, detail=None):
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + json.dumps(detail) if detail is not None else ''}", flush=True)


def py(*args, check=True):
    code = subprocess.call([sys.executable, "-X", "utf8", *map(str, args)])
    if check and code:
        raise SystemExit(f"{args[0]} {args[1]} exited {code}")
    return code


def updates(case):
    return [json.loads(line) for line in (case / "updates.jsonl").read_text().splitlines() if line.strip()]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--extra-options", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.003)
    parser.add_argument("--snapshot-users", type=int, default=4)
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to(ROOT / "artifacts"):
        parser.error("--out must be under this checkout's artifacts/")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    started = time.time()
    dirs = {}
    for arm in ("A", "B"):
        d = dirs[arm] = out / arm
        py(DEC, "prepare", "--source-run", args.source_run, "--epoch", args.epoch, "--out", d.relative_to(ROOT),
           "--fraction", args.fraction, "--shape", "one_minus_sin", "--evaluate", "none", "--seed", 1, "--fresh-optimizer",
           "--train-tier", "final", "--extra-options", args.extra_options, "--snapshot-users", args.snapshot_users)
        py(TRIM, "--decay", d, "--trim-unlabelled-tail")
    pa, pb = (json.loads((dirs[a] / "protocol.json").read_text()) for a in "AB")
    record("A and B train the same rows", pa["rows"] == pb["rows"], {"users": len(pa["rows"])})
    track = json.loads((dirs["A"] / "decay.json").read_text())["track"]
    case = {a: dirs[a] / "expertium" / track for a in "AB"}

    t = time.time()
    record("A runs uninterrupted", py(DEC, "run", "--out", dirs["A"].relative_to(ROOT), check=False) == 0)
    minutes_a = (time.time() - t) / 60

    half = len(pb["rows"]) // 2
    t = time.time()
    proc = subprocess.Popen([sys.executable, "-X", "utf8", str(DEC), "run", "--out", str(dirs["B"].relative_to(ROOT))])
    requested = None
    while proc.poll() is None:
        try:
            progress = json.loads((case["B"] / "result.json").read_text()).get("progress_users", 0)
        except (OSError, ValueError):
            progress = 0
        if requested is None and progress >= half:
            py(DEC, "stop", "--out", dirs["B"].relative_to(ROOT))
            requested = progress
        time.sleep(0.5)
    stopped = json.loads((case["B"] / "result.json").read_text())
    record("B stops on request with a snapshot", proc.returncode != 0 and stopped["status"] == "stopped"
           and (case["B"] / "snapshots/latest.json").exists(),
           {"exit": proc.returncode, "status": stopped["status"], "requested_at_users": requested,
            "snapshot": stopped.get("latest_snapshot")})
    record("B resumes and completes", py(DEC, "run", "--out", dirs["B"].relative_to(ROOT), check=False) == 0)
    minutes_b = (time.time() - t) / 60
    resume = json.loads((dirs["B"] / "resume.json").read_text())
    resumed = json.loads((dirs["B"] / "protocol.json").read_text())
    record("resume skipped the done users", len(resumed["rows"]) == len(pb["rows"]) - len(resume["done_users"]) > 0,
           {"done": len(resume["done_users"]), "left": len(resumed["rows"]), "lr_resume": resumed["full_run"]["lr_resume"]})

    ua = {(u["user_id"], u["offset"]): u for u in updates(case["A"])}
    first = updates(dirs["B"] / "attempts/attempt-1/expertium" / track)
    second = updates(case["B"])
    done = set(resume["done_users"])
    kept = [u for u in first if u["user_id"] in done] + second  # updates of users in flight at the stop were retrained
    keys = [(u["user_id"], u["offset"]) for u in kept]
    # Deferred reads log an update only after the next ones are queued: the snapshot's last updates are applied but
    # not in attempt 1's log. They must be exactly the snapshot's unlogged count, all of done users.
    snap = json.loads((dirs["B"] / "attempts/attempt-1/expertium" / track / "snapshots/latest.json").read_text())
    unlogged = [k for k in ua if k not in set(keys)]
    record("B's kept updates cover A's exactly once (applied but unlogged at the stop: done users only)",
           len(set(keys)) == len(keys) and set(keys) <= set(ua) and len(unlogged) == snap["updates"] - len(first)
           and all(k[0] in done for k in unlogged),
           {"A": len(ua), "B kept": len(keys), "unlogged": len(unlogged), "snapshot_updates": snap["updates"],
            "attempt_1_logged": len(first), "B retrained": len(first) - sum(u["user_id"] in done for u in first)})
    common = [(u, ua[k]) for u, k in zip(kept, keys) if k in ua]
    lr_gap = max(abs(u["lr"] - a["lr"]) for u, a in common)
    record("same batch seeds", all(u["seed"] == a["seed"] for u, a in common))
    record("same learning rate per update", lr_gap <= 1e-12 * 7e-4, {"max_gap": lr_gap})
    record("same final learning rate", abs(second[-1]["lr"] - updates(case["A"])[-1]["lr"]) <= 1e-12,
           {"A": updates(case["A"])[-1]["lr"], "B": second[-1]["lr"]})
    loss_gap = max(abs(u["loss"] - a["loss"]) for u, a in common)
    record("loss per update matches A", loss_gap <= 1e-3, {"max_gap": loss_gap,
                                                           "bitwise": all(u["loss"] == a["loss"] for u, a in common)})
    import torch
    wa, wb = (torch.load(case[a] / "master.pth", map_location="cpu", weights_only=True) for a in "AB")
    gap = max(float((wa[k] - wb[k]).abs().max()) for k in wa)
    record("final weights match A", gap <= 1e-3, {"max_abs_gap": gap, "bitwise": all(torch.equal(wa[k], wb[k]) for k in wa)})
    report = {"source": f"{args.source_run} epoch {args.epoch}", "fraction": args.fraction, "users": len(pa["rows"]),
              "minutes_A": round(minutes_a, 1), "minutes_B_with_stop_and_resume": round(minutes_b, 1),
              "minutes_total": round((time.time() - started) / 60, 1), "checks": CHECKS,
              "all_pass": all(c["ok"] for c in CHECKS)}
    (out / "check.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print("ALL PASS" if report["all_pass"] else "FAILED", flush=True)
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
