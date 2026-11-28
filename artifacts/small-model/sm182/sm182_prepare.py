#!/usr/bin/env python
"""AUC ceiling (SM182): prepare the arms of the screen on six-layer Tiny two-head (SM130) (design.md "Recipe").

Every arm is the state-energy penalty screen's (SM171) control cool-down with one track option added, "pair_rank"
(the gated pairwise ranking term of the run tree, expertium.model.variant_model.apply_pair_rank): the same 944 users
and 67.3M reviews, the same learning rate (0 -> 7e-5 over 64 updates, then one_minus_sin to 0), the same seed, the same
start (the release model's final master.pth and optimizer.pth, used in place) and the release folder's options
(SM171's control also had "state_vq" with lambda 0, proven bitwise equal to the stock trainer; it is dropped here).

  python sm182_prepare.py arm  --arm r  --weight 1.0  --logloss 0.0  --root C:/Users/jonat/workspace/wt/sm182-run
  python sm182_prepare.py arm  --arm m  --weight <lambda_M> --logloss 1.0 --root ...
  python sm182_prepare.py arm  --arm ra --weight 1.0  --logloss 0.02 --root ...    (fallback of R, design.md)
  python sm182_prepare.py pair --arms r,m,ra --root ...     (the arms vs SM171's control: rows, schedule, options)
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

W = Path("C:/Users/jonat/workspace/wt")
RELEASE = W / "sm130/artifacts/small-model/sm130-th-dec2ep32k-p2"   # six-layer Tiny two-head (SM130)
TRACK = "E89-F65e-F77a"
SHA = "48a5e724cf38b53a32d570d4d034791f1da2cfcbe878572ff03ad434d7fb75af"
CTL = W / "sm171-run/artifacts/small-model/sm171-tiny-ctl"           # SM171's control (lambda 0)
PAIRS = 65536


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


def control_options():
    ctl_decay = json.loads((CTL / "decay.json").read_text())
    release_options = json.loads((RELEASE / "decay.json").read_text())["options"]
    options = dict(ctl_decay["options"])
    sv = options.pop("state_vq")
    if sv != {"recipe": "sm171-energy", "lambda": 0.0} or options != release_options:
        raise SystemExit("SM171's control options are not the release options plus state_vq lambda 0")
    return ctl_decay, options


def arm(args):
    root = Path(args.root).resolve()
    case = RELEASE / "expertium" / TRACK
    have = sha256(case / "master.pth")
    if have != SHA:
        raise SystemExit(f"release master sha256 {have} is not {SHA}")
    out = root / "artifacts" / "small-model" / f"sm182-tiny-{args.arm}"
    if out.exists():
        raise SystemExit(f"{out} exists")
    ctl_decay, options = control_options()
    ctl_protocol = json.loads((CTL / "protocol.json").read_text())
    spec = {"weight": float(args.weight), "logloss": float(args.logloss), "pairs": PAIRS}
    options["pair_rank"] = spec
    out.mkdir(parents=True)
    full_run = dict(ctl_protocol["full_run"], control_dir=(out / "control").relative_to(root).as_posix())
    protocol = dict(ctl_protocol, full_run=full_run)
    decay = {k: v for k, v in ctl_decay.items() if k != "sm171"}
    decay.update(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), options=options,
                 sm182={"arm": args.arm, "pair_rank": spec, "control": str(CTL),
                        "release_master_sha256": have, "release_optimizer_sha256": sha256(case / "optimizer.pth")})
    atomic_json(out / "decay.json", decay)
    atomic_json(out / "protocol.json", protocol)
    print(f"[sm182-prepare] {out.name}: pair_rank {spec}, {len(protocol['rows']):,} users, "
          f"{decay['training_reviews']:,} reviews, lr {decay['lr']}")
    return 0


def pair(args):
    root = Path(args.root).resolve()
    ctl_d = json.loads((CTL / "decay.json").read_text())
    ctl_p = json.loads((CTL / "protocol.json").read_text())
    _, base = control_options()
    problems, info = [], {}
    for a in args.arms.split(","):
        f = root / "artifacts" / "small-model" / f"sm182-tiny-{a}"
        d = json.loads((f / "decay.json").read_text())
        pr = json.loads((f / "protocol.json").read_text())
        o = dict(d["options"])
        spec = o.pop("pair_rank", None)
        if o != base:
            problems.append(f"{a}: options other than pair_rank differ from SM171's control without state_vq")
        if not spec or spec != d["sm182"]["pair_rank"]:
            problems.append(f"{a}: pair_rank missing or not the recorded one")
        if pr["rows"] != ctl_p["rows"]:
            problems.append(f"{a}: rows differ from SM171's control")
        if {k: v for k, v in pr["full_run"].items() if k != "control_dir"} != \
                {k: v for k, v in ctl_p["full_run"].items() if k != "control_dir"}:
            problems.append(f"{a}: full_run (start, schedule, epoch, snapshots) differs from SM171's control")
        if {k: v for k, v in pr.items() if k != "full_run"} != {k: v for k, v in ctl_p.items() if k != "full_run"}:
            problems.append(f"{a}: protocol fields differ from SM171's control")
        for k in ("training_users", "training_reviews", "lr", "start", "start_master_sha256"):
            if d.get(k) != ctl_d.get(k):
                problems.append(f"{a}: {k} differs")
        if sha256(pr["full_run"]["init_master"]) != SHA:
            problems.append(f"{a}: start master is not the release model")
        info[a] = spec
    print(json.dumps({"arms": info, "users": len(ctl_p["rows"]), "reviews": ctl_d["training_reviews"],
                      "ok": not problems, "problems": problems}))
    return 0 if not problems else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("arm")
    s.add_argument("--arm", required=True)
    s.add_argument("--weight", required=True, type=float)
    s.add_argument("--logloss", required=True, type=float)
    s.add_argument("--root", required=True)
    s = sub.add_parser("pair")
    s.add_argument("--arms", required=True)
    s.add_argument("--root", required=True)
    args = p.parse_args()
    return arm(args) if args.cmd == "arm" else pair(args)


if __name__ == "__main__":
    raise SystemExit(main())
