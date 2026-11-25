#!/usr/bin/env python
"""State-energy penalty (SM171): prepare the arms of the screen on six-layer Tiny two-head (SM130) (design.md "Recipe").

Every arm starts from the release model's final master.pth AND optimizer.pth (the end of its own cool-down), trains the
same fixed subset of that model's own cool-down users (random.Random(171).shuffle of the release folder's rows sorted
by user id, whole users, until the review budget R is reached) with the release folder's options, loss and targets
unchanged, and the learning rate 0 -> 7e-5 over 64 updates, then one_minus_sin to 0 over the R reviews. Each arm adds
one track option, "state_vq": {"recipe": "sm171-energy", "lambda": L} (sm171_run.py installs the hook; L = 0 is the
control: the stock trainer's arithmetic, plus a no-grad diagnostic). `pair` proves that the arms differ only by L.

  python sm171_prepare.py arm  --arm ctl --lam 0      --root C:/Users/jonat/workspace/wt/sm171-run [--budget N]
  python sm171_prepare.py arm  --arm lo  --lam 0.003  --root ...
  python sm171_prepare.py pair --arms ctl,lo,hi --root ...
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

W = Path("C:/Users/jonat/workspace/wt")
SEED = 171
WARMUP = 64
PEAK = 7e-5
SNAPSHOT_USERS = 100
RELEASE = W / "sm130/artifacts/small-model/sm130-th-dec2ep32k-p2"   # six-layer Tiny two-head (SM130)
TRACK = "E89-F65e-F77a"
SHA = "48a5e724cf38b53a32d570d4d034791f1da2cfcbe878572ff03ad434d7fb75af"   # models.sh of SM148
BUDGET = 1 << 26


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)


def reviews(rows):
    return sum(r["train_range"][1] - r["train_range"][0] for r in rows)


def folder_name(arm):
    return f"sm171-tiny-{arm}"


def subset(rows, budget):
    order = sorted(rows, key=lambda r: r["user_id"])
    random.Random(SEED).shuffle(order)
    out, total = [], 0
    for row in order:
        if total >= budget:
            break
        if row["train_range"][0] != 0:
            raise SystemExit(f"user {row['user_id']} does not start at review 0 (state carry needs whole histories)")
        out.append(row)
        total += row["train_range"][1] - row["train_range"][0]
    return out


def arm(args):
    root = Path(args.root).resolve()
    case = RELEASE / "expertium" / TRACK
    result = json.loads((case / "result.json").read_text())
    if result.get("status") != "complete" or result.get("smoke_only"):
        raise SystemExit(f"{case}: not a complete run")
    protocol = json.loads((RELEASE / "protocol.json").read_text())
    decay = json.loads((RELEASE / "decay.json").read_text())
    master, optimizer = case / "master.pth", case / "optimizer.pth"
    have = sha256(master)
    if have != SHA:
        raise SystemExit(f"{master}: sha256 {have} is not the release model's {SHA}")
    out = root / "artifacts" / "small-model" / folder_name(args.arm)
    if out.exists():
        raise SystemExit(f"{out} exists")
    rows = subset(protocol["rows"], args.budget)
    r = reviews(rows)
    options = dict(decay["options"])
    if "state_vq" in options:
        raise SystemExit("the release options already hold state_vq")
    options["state_vq"] = {"recipe": "sm171-energy", "lambda": float(args.lam)}
    old = protocol["full_run"]
    full_run = {"epoch": old.get("epoch", 0), "evaluate": False, "init_master": master.as_posix(),
                "init_optimizer": optimizer.as_posix(),
                "optimizer_state": "continued from the release model's end (its own cool-down's final state)",
                "update_offset": 0,
                "lr_schedule": {"peak": PEAK, "warmup_updates": WARMUP, "decay_start": 0.0, "decay_end": 1.0,
                                "shape": "one_minus_sin", "final": 0.0},
                "evaluation_set": "none", "snapshot_users": SNAPSHOT_USERS,
                "control_dir": (out / "control").relative_to(root).as_posix()}
    if "benchmark_db" in old:
        full_run["benchmark_db"] = old["benchmark_db"]
    new_protocol = dict(protocol, rows=rows, full_run=full_run)
    new_decay = {k: v for k, v in decay.items() if k not in ("sm130",)}
    lr = f"0 -> {PEAK:g} over {WARMUP} updates, then one_minus_sin to 0 over {r:,} reviews"
    new_decay.update(training_users=len(rows), training_reviews=r, training_reviews_after_trims=r,
                     fraction_of_epoch_reviews=r / reviews(protocol["rows"]), lr=lr,
                     created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), options=options,
                     source_case=str(case), start=str(RELEASE), start_master_sha256=have,
                     sm171={"arm": args.arm, "lambda": float(args.lam), "model": "sm130th",
                            "what": ("control: the identical cool-down, penalty off (lambda 0)" if args.lam == 0 else
                                     f"state-energy penalty, lambda {args.lam:g} (rank-2 dropped energy share)"),
                            "users": len(rows), "reviews": r, "review_budget": args.budget, "subset_seed": SEED,
                            "lr": lr, "start_optimizer_sha256": sha256(optimizer),
                            "release_update_offset_end": result.get("update_offset_end"),
                            "release_trained_parameter_sha256": result.get("trained_parameter_sha256")})
    out.mkdir(parents=True)
    atomic_json(out / "decay.json", new_decay)
    atomic_json(out / "protocol.json", new_protocol)
    print(f"[sm171-prepare] {out.name}: {len(rows):,} users, {r:,} reviews (budget {args.budget:,}), peak {PEAK:g}, "
          f"from {master.parent.name} {have[:12]}, lambda {args.lam:g}")
    return 0


def pair(args):
    root = Path(args.root).resolve()
    arms = args.arms.split(",")
    folders = {a: root / "artifacts" / "small-model" / folder_name(a) for a in arms}
    decays = {a: json.loads((f / "decay.json").read_text()) for a, f in folders.items()}
    protos = {a: json.loads((f / "protocol.json").read_text()) for a, f in folders.items()}
    problems = []
    release_options = json.loads((RELEASE / "decay.json").read_text())["options"]
    base = None
    for a in arms:
        o = dict(decays[a]["options"])
        sv = o.pop("state_vq", None)
        if not isinstance(sv, dict) or sv.get("recipe") != "sm171-energy" or set(sv) != {"recipe", "lambda"}:
            problems.append(f"{a}: state_vq {sv}")
        if o != release_options:
            problems.append(f"{a}: options beyond state_vq differ from the release folder's")
        if decays[a]["start_master_sha256"] != SHA:
            problems.append(f"{a}: start sha")
        fr = dict(protos[a]["full_run"])
        fr.pop("control_dir")
        key = (json.dumps(protos[a]["rows"]), json.dumps(fr, sort_keys=True),
               json.dumps({k: v for k, v in protos[a].items() if k not in ("rows", "full_run")}, sort_keys=True))
        if base is None:
            base = key
        elif key != base:
            problems.append(f"{a}: rows / full_run / protocol differ from {arms[0]}")
    lams = {a: decays[a]["options"]["state_vq"]["lambda"] for a in arms}
    print(json.dumps({"arms": {a: folders[a].name for a in arms}, "lambda": lams, "users": len(protos[arms[0]]["rows"]),
                      "reviews": reviews(protos[arms[0]]["rows"]), "ok": not problems, "problems": problems}))
    return 0 if not problems else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("arm")
    s.add_argument("--arm", required=True)
    s.add_argument("--lam", type=float, required=True)
    s.add_argument("--budget", type=int, default=BUDGET)
    s.add_argument("--root", required=True)
    s = sub.add_parser("pair")
    s.add_argument("--arms", required=True)
    s.add_argument("--root", required=True)
    args = p.parse_args()
    return arm(args) if args.cmd == "arm" else pair(args)


if __name__ == "__main__":
    raise SystemExit(main())
