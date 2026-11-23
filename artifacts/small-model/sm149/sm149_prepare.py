#!/usr/bin/env python
"""Rank-2 low-rank state cool-down (SM149): prepare the two arms of one release model (design.md "Recipe").

Both arms start from the release model's final master.pth AND optimizer.pth (the end of its own cool-down), train the
same fixed subset of that model's own cool-down users (random.Random(149).shuffle of the release folder's rows, whole
users, until the size's review budget R is reached) with the release folder's options, loss and targets unchanged,
and the learning rate 0 -> peak over 64 updates, then one_minus_sin to 0 over the R reviews. The QAT arm adds one
track option, "state_vq": {"recipe": "lr2q8x8"} (sm149_run.py installs the rank-2 straight-through hook); the control
arm is the same folder without it. `pair` proves that the two folders differ only there.

  python sm149_prepare.py arm --model sm118 --arm qat     --root C:/Users/jonat/workspace/wt/sm149-run
  python sm149_prepare.py arm --model sm118 --arm control --root C:/Users/jonat/workspace/wt/sm149-run
  python sm149_prepare.py pair --model sm118 --root C:/Users/jonat/workspace/wt/sm149-run
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

W = Path("C:/Users/jonat/workspace/wt")
SEED = 149
RECIPE = "lr2q8x8"
WARMUP = 64
SNAPSHOT_USERS = 100
# tag -> (release run folder, track, size, peak LR, review budget R); the six release models of models.sh (SM148)
MODELS = {
    "sm118": (W / "sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2", "E89-F65e-F77a", "small", 7e-5, 1 << 26),
    "sm113": (W / "sm113/artifacts/small-model/sm113-inst-e35st-st-dec2ep32k-t99-p2", "E89-F65e-F77a", "small", 7e-5,
              1 << 26),
    "sm130th": (W / "sm130/artifacts/small-model/sm130-th-dec2ep32k-p2", "E89-F65e-F77a", "tiny", 7e-5, 1 << 26),
    "sm130io": (W / "sm130/artifacts/small-model/sm130-io-dec2ep32k-p2", "E89-F65e-F77a", "tiny", 7e-5, 1 << 26),
    "sm91s2": (W / "sm91s2/artifacts/small-model/sm91s2-big-showtime", "big-showtime", "big", 3.5e-5, 1 << 25),
    "sm99": (W / "sm99/artifacts/small-model/sm99-big-inst-e2", "big-inst", "big", 3.5e-5, 1 << 25),
}
SHA = {   # models.sh of the low-rank int8 states experiment (SM148)
    "sm118": "b4fe1175d25a7e6abff70e87ac0af1f4d76db9622d54f7cb53196fbf35117eb6",
    "sm113": "e4574e2aa8b2cfe6360dbd86f2a75b71d23ea81677b00ffcc9c3abce6b4cb736",
    "sm130th": "48a5e724cf38b53a32d570d4d034791f1da2cfcbe878572ff03ad434d7fb75af",
    "sm130io": "23f037282379199bd113a5297737c920fcbefd702b7b76eecc0095b8a31186a2",
    "sm91s2": "67ee8e9b0d0169733242c40646e0d2e5637845a25775762002ffc4e6396f9590",
    "sm99": "c49e45d698194da033caa1231f598f875922a4a2a3e921714a3fd433d37e5768",
}


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


def folder_name(tag, arm):
    return f"sm149-{tag}-{'qat' if arm == 'qat' else 'ctl'}"


def release(tag):
    folder, track, size, peak, budget = MODELS[tag]
    case = folder / "expertium" / track
    result = json.loads((case / "result.json").read_text())
    if result.get("status") != "complete" or result.get("smoke_only"):
        raise SystemExit(f"{case}: not a complete run")
    prepared = folder / "protocol.prepared.json"   # a resumed run rewrote protocol.json to the users it had left
    protocol = json.loads((prepared if prepared.exists() else folder / "protocol.json").read_text())
    decay = json.loads((folder / "decay.json").read_text())
    return folder, track, size, peak, budget, case, result, protocol, decay


def subset(rows, budget):
    """Whole users of random.Random(SEED).shuffle(rows) until `budget` reviews are reached (the last one whole)."""
    order = sorted(rows, key=lambda r: r["user_id"])   # independent of the release folder's own order
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
    folder, track, size, peak, budget, case, result, protocol, decay = release(args.model)
    master, optimizer = case / "master.pth", case / "optimizer.pth"
    have = sha256(master)
    if have != SHA[args.model]:
        raise SystemExit(f"{master}: sha256 {have} is not the release model's {SHA[args.model]}")
    out = root / "artifacts" / "small-model" / folder_name(args.model, args.arm)
    if out.exists():
        raise SystemExit(f"{out} exists")
    rows = subset(protocol["rows"], budget)
    r = reviews(rows)
    options = dict(decay["options"])
    if "state_vq" in options:
        raise SystemExit(f"{folder}: the release options already hold state_vq")
    if args.arm == "qat":
        options["state_vq"] = {"recipe": RECIPE}
    old = protocol["full_run"]
    full_run = {"epoch": old.get("epoch", 0), "evaluate": False, "init_master": master.as_posix(),
                "init_optimizer": optimizer.as_posix(),
                "optimizer_state": "continued from the release model's end (its own cool-down's final state)",
                "update_offset": 0,
                "lr_schedule": {"peak": peak, "warmup_updates": WARMUP, "decay_start": 0.0, "decay_end": 1.0,
                                "shape": "one_minus_sin", "final": 0.0},
                "evaluation_set": "none", "snapshot_users": SNAPSHOT_USERS,
                "control_dir": (out / "control").relative_to(root).as_posix()}
    if "benchmark_db" in old:
        full_run["benchmark_db"] = old["benchmark_db"]
    new_protocol = dict(protocol, rows=rows, full_run=full_run)
    new_decay = {k: v for k, v in decay.items() if k not in ("sm118", "sm113", "sm115", "sm130")}
    new_decay.update(training_users=len(rows), training_reviews=r, training_reviews_after_trims=r,
                     fraction_of_epoch_reviews=r / reviews(protocol["rows"]),
                     lr=f"0 -> {peak:g} over {WARMUP} updates, then one_minus_sin to 0 over {r:,} reviews",
                     created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), options=options,
                     source_case=str(case), start=str(folder), start_master_sha256=have,
                     sm149={"arm": args.arm, "model": args.model, "size": size,
                            "what": ("rank-2 low-rank state QAT cool-down (straight-through)" if args.arm == "qat" else
                                     "control: the identical cool-down with QAT off"),
                            "users": len(rows), "reviews": r, "review_budget": budget, "subset_seed": SEED,
                            "lr": f"0 -> {peak:g} over {WARMUP} updates, then one_minus_sin to 0 over {r:,} reviews",
                            "start_optimizer_sha256": sha256(optimizer),
                            "release_update_offset_end": result.get("update_offset_end"),
                            "release_trained_parameter_sha256": result.get("trained_parameter_sha256")})
    out.mkdir(parents=True)
    atomic_json(out / "decay.json", new_decay)
    atomic_json(out / "protocol.json", new_protocol)
    print(f"[sm149-prepare] {out.name}: {len(rows):,} users, {r:,} reviews (budget {budget:,}), peak {peak:g}, "
          f"from {master.parent.name} {have[:12]}, state_vq {options.get('state_vq')}")
    return 0


def pair(args):
    root = Path(args.root).resolve()
    a, b = (root / "artifacts" / "small-model" / folder_name(args.model, x) for x in ("qat", "control"))
    da, db = (json.loads((d / "decay.json").read_text()) for d in (a, b))
    pa, pb = (json.loads((d / "protocol.json").read_text()) for d in (a, b))
    oa, ob = dict(da["options"]), dict(db["options"])
    problems = []
    if oa.pop("state_vq", None) != {"recipe": RECIPE}:
        problems.append("QAT arm without state_vq lr2q8x8")
    if "state_vq" in ob:
        problems.append("control arm has state_vq")
    if oa != ob:
        problems.append("options differ beyond state_vq")
    if pa["rows"] != pb["rows"]:
        problems.append("rows differ")
    fa, fb = dict(pa["full_run"]), dict(pb["full_run"])
    fa.pop("control_dir"), fb.pop("control_dir")
    if fa != fb:
        problems.append("full_run differs beyond control_dir")
    if {k: v for k, v in pa.items() if k not in ("rows", "full_run")} != {k: v for k, v in pb.items()
                                                                          if k not in ("rows", "full_run")}:
        problems.append("protocol differs")
    folder, *_rest = MODELS[args.model]
    release_options = json.loads((folder / "decay.json").read_text())["options"]
    if ob != release_options:
        problems.append("control options are not the release folder's")
    for d in (da, db):
        if d["start_master_sha256"] != SHA[args.model]:
            problems.append("start sha")
    print(json.dumps({"model": args.model, "qat": a.name, "control": b.name, "users": len(pa["rows"]),
                      "reviews": reviews(pa["rows"]), "ok": not problems, "problems": problems}))
    return 0 if not problems else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("arm")
    s.add_argument("--model", choices=sorted(MODELS), required=True)
    s.add_argument("--arm", choices=("qat", "control"), required=True)
    s.add_argument("--root", required=True)
    s = sub.add_parser("pair")
    s.add_argument("--model", choices=sorted(MODELS), required=True)
    s.add_argument("--root", required=True)
    args = p.parse_args()
    return arm(args) if args.cmd == "arm" else pair(args)


if __name__ == "__main__":
    raise SystemExit(main())
