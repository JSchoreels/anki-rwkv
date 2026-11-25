#!/usr/bin/env python
"""Sharpness-aware cool-down (SM172): prepare the SAM arms on six-layer Tiny two-head (SM130) (design.md "Recipe").

Every arm is the control of the state-energy penalty screen (SM171, `sm171-tiny-ctl`: the release model's final
master.pth AND optimizer.pth, the same 944 users / 67,310,673 reviews in the same order, the release options, the
learning rate 0 -> 7e-5 over 64 updates then one_minus_sin to 0) with one difference: its "state_vq" track option is
{"recipe": "sm172-sam", "lambda": 0.0, "rho": R} instead of {"recipe": "sm171-energy", "lambda": 0.0}. sm172_run.py
installs the SM171 hook at lambda 0 (the control's code path) plus SAM of radius R. `arm` builds the folder from
SM171's own subset code (sm171_prepare.subset, seed 171); `pair` proves that each arm equals the control's decay.json
and protocol.json except that option (and the arm's own control_dir).

  python sm172_prepare.py arm  --arm r20 --rho 0.2  --root C:/Users/jonat/workspace/wt/sm172-run
  python sm172_prepare.py pair --arms r05,r20 --root C:/Users/jonat/workspace/wt/sm172-run
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sm171"))
import sm171_prepare as base  # noqa: E402

CONTROL = Path("C:/Users/jonat/workspace/wt/sm171-run/artifacts/small-model/sm171-tiny-ctl")   # read only
CONTROL_SV = {"recipe": "sm171-energy", "lambda": 0.0}


def folder_name(arm):
    return f"sm172-tiny-{arm}"


def arm(args):
    root = Path(args.root).resolve()
    case = base.RELEASE / "expertium" / base.TRACK
    result = json.loads((case / "result.json").read_text())
    protocol = json.loads((base.RELEASE / "protocol.json").read_text())
    decay = json.loads((base.RELEASE / "decay.json").read_text())
    master, optimizer = case / "master.pth", case / "optimizer.pth"
    have = base.sha256(master)
    if have != base.SHA:
        raise SystemExit(f"{master}: sha256 {have} is not the release model's {base.SHA}")
    out = root / "artifacts" / "small-model" / folder_name(args.arm)
    if out.exists():
        raise SystemExit(f"{out} exists")
    rows = base.subset(protocol["rows"], base.BUDGET)
    r = base.reviews(rows)
    options = dict(decay["options"])
    if "state_vq" in options:
        raise SystemExit("the release options already hold state_vq")
    options["state_vq"] = {"recipe": "sm172-sam", "lambda": 0.0, "rho": float(args.rho)}
    old = protocol["full_run"]
    full_run = {"epoch": old.get("epoch", 0), "evaluate": False, "init_master": master.as_posix(),
                "init_optimizer": optimizer.as_posix(),
                "optimizer_state": "continued from the release model's end (its own cool-down's final state)",
                "update_offset": 0,
                "lr_schedule": {"peak": base.PEAK, "warmup_updates": base.WARMUP, "decay_start": 0.0, "decay_end": 1.0,
                                "shape": "one_minus_sin", "final": 0.0},
                "evaluation_set": "none", "snapshot_users": base.SNAPSHOT_USERS,
                "control_dir": (out / "control").relative_to(root).as_posix()}
    if "benchmark_db" in old:
        full_run["benchmark_db"] = old["benchmark_db"]
    new_protocol = dict(protocol, rows=rows, full_run=full_run)
    new_decay = {k: v for k, v in decay.items() if k not in ("sm130",)}
    lr = f"0 -> {base.PEAK:g} over {base.WARMUP} updates, then one_minus_sin to 0 over {r:,} reviews"
    new_decay.update(training_users=len(rows), training_reviews=r, training_reviews_after_trims=r,
                     fraction_of_epoch_reviews=r / base.reviews(protocol["rows"]), lr=lr,
                     created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), options=options,
                     source_case=str(case), start=str(base.RELEASE), start_master_sha256=have,
                     sm172={"arm": args.arm, "rho": float(args.rho), "model": "sm130th",
                            "what": f"sharpness-aware cool-down, plain SAM radius {args.rho:g}; control = SM171 "
                                    "sm171-tiny-ctl (its rows are reused)",
                            "users": len(rows), "reviews": r, "review_budget": base.BUDGET, "subset_seed": base.SEED,
                            "lr": lr, "start_optimizer_sha256": base.sha256(optimizer),
                            "release_update_offset_end": result.get("update_offset_end"),
                            "release_trained_parameter_sha256": result.get("trained_parameter_sha256")})
    out.mkdir(parents=True)
    base.atomic_json(out / "decay.json", new_decay)
    base.atomic_json(out / "protocol.json", new_protocol)
    print(f"[sm172-prepare] {out.name}: {len(rows):,} users, {r:,} reviews, peak {base.PEAK:g}, "
          f"from {master.parent.name} {have[:12]}, rho {args.rho:g}")
    return 0


def pair(args):
    root = Path(args.root).resolve()
    arms = args.arms.split(",")
    cd = json.loads((CONTROL / "decay.json").read_text())
    cp = json.loads((CONTROL / "protocol.json").read_text())
    problems = []
    if cd["options"].get("state_vq") != CONTROL_SV:
        problems.append(f"control state_vq {cd['options'].get('state_vq')}")
    rhos = {}
    for a in arms:
        f = root / "artifacts" / "small-model" / folder_name(a)
        d = json.loads((f / "decay.json").read_text())
        p = json.loads((f / "protocol.json").read_text())
        o, co = dict(d["options"]), dict(cd["options"])
        sv = o.pop("state_vq", None)
        co.pop("state_vq", None)
        if not isinstance(sv, dict) or sv.get("recipe") != "sm172-sam" or set(sv) != {"recipe", "lambda", "rho"} \
                or sv["lambda"] != 0.0 or not sv["rho"] > 0:
            problems.append(f"{a}: state_vq {sv}")
        else:
            rhos[a] = sv["rho"]
        if o != co:
            problems.append(f"{a}: options beyond state_vq differ from the control's")
        for k in ("start_master_sha256", "training_users", "training_reviews", "lr", "source_case", "start"):
            if d.get(k) != cd.get(k):
                problems.append(f"{a}: decay.json {k} differs from the control's")
        if d.get("sm172", {}).get("start_optimizer_sha256") != cd.get("sm171", {}).get("start_optimizer_sha256"):
            problems.append(f"{a}: start optimizer sha differs from the control's")
        fr, cfr = dict(p["full_run"]), dict(cp["full_run"])
        fr.pop("control_dir")
        cfr.pop("control_dir")
        if json.dumps(p["rows"]) != json.dumps(cp["rows"]):
            problems.append(f"{a}: rows differ from the control's")
        if json.dumps(fr, sort_keys=True) != json.dumps(cfr, sort_keys=True):
            problems.append(f"{a}: full_run differs from the control's")
        rest = {k: v for k, v in p.items() if k not in ("rows", "full_run")}
        crest = {k: v for k, v in cp.items() if k not in ("rows", "full_run")}
        if json.dumps(rest, sort_keys=True) != json.dumps(crest, sort_keys=True):
            problems.append(f"{a}: protocol differs from the control's")
    print(json.dumps({"arms": {a: folder_name(a) for a in arms}, "rho": rhos, "control": str(CONTROL),
                      "users": len(cp["rows"]), "reviews": base.reviews(cp["rows"]), "ok": not problems,
                      "problems": problems}))
    return 0 if not problems else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("arm")
    s.add_argument("--arm", required=True)
    s.add_argument("--rho", type=float, required=True)
    s.add_argument("--root", required=True)
    s = sub.add_parser("pair")
    s.add_argument("--arms", required=True)
    s.add_argument("--root", required=True)
    args = p.parse_args()
    return arm(args) if args.cmd == "arm" else pair(args)


if __name__ == "__main__":
    raise SystemExit(main())
