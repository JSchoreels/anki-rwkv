#!/usr/bin/env python
"""Session inputs (SM179): prepare the arms of the screen on six-layer Tiny two-head (SM130) (design.md "Recipe").

Every arm is the state-energy penalty screen's (SM171) control cool-down, with only the inputs changed: the same 944
users and 67.3M reviews, the same learning rate (0 -> 7e-5 over 64 updates, then one_minus_sin to 0), the same seed and
the release folder's options unchanged (SM171's control also had "state_vq" with lambda 0, proven bitwise equal to the
stock trainer; the arms here run the stock trainer). The start is the release model's final master.pth and
optimizer.pth with n zero columns inserted at input index 24 of features2card.0.weight (and of its Muon momentum), so
that step 0 computes exactly what the release model computes.

  python sm179_prepare.py arm  --arm ses --inputs session --root C:/Users/jonat/workspace/wt/sm179-run
  python sm179_prepare.py arm  --arm st  --inputs state   --root ...
  python sm179_prepare.py pair --arms ses,st --root ...     (the arms vs SM171's control: rows, schedule, options)
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

W = Path("C:/Users/jonat/workspace/wt")
RELEASE = W / "sm130/artifacts/small-model/sm130-th-dec2ep32k-p2"   # six-layer Tiny two-head (SM130)
TRACK = "E89-F65e-F77a"
SHA = "48a5e724cf38b53a32d570d4d034791f1da2cfcbe878572ff03ad434d7fb75af"
CTL = W / "sm171-run/artifacts/small-model/sm171-tiny-ctl"           # SM171's control (lambda 0)
DIMS = {"session": 6, "state": 4}
AT = 24   # the extra columns follow the 24 deterministic features (rwkv/extra_inputs.py)
WEIGHT = "features2card.0.weight"


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


def widen(t, n):
    """Insert n zero columns at index AT of a (rows, 92) tensor."""
    if t.dim() != 2 or t.shape[1] != 92:
        raise ValueError(f"expected a (rows, 92) tensor, got {tuple(t.shape)}")
    z = torch.zeros(t.shape[0], n, dtype=t.dtype)
    return torch.cat([t[:, :AT], z, t[:, AT:]], 1).contiguous()


def padded_start(out, n):
    case = RELEASE / "expertium" / TRACK
    master = torch.load(case / "master.pth", map_location="cpu", weights_only=True)
    opt = torch.load(case / "optimizer.pth", map_location="cpu", weights_only=True)
    w = master[WEIGHT]
    master[WEIGHT] = widen(w, n)
    hits = []
    for group in ("adamw", "muon"):
        for index, state in opt[group]["state"].items():
            for key, value in state.items():
                if torch.is_tensor(value) and value.dim() == 2 and value.shape[1] == 92:
                    if value.shape != w.shape:
                        raise ValueError(f"{group} {index} {key}: unexpected shape {tuple(value.shape)}")
                    state[key] = widen(value, n)
                    hits.append(f"{group}[{index}].{key}")
    if hits != ["muon[37].momentum_buffer"]:
        raise ValueError(f"expected exactly the first Linear's Muon momentum, got {hits}")
    start = out / "start"
    start.mkdir(parents=True)
    torch.save(master, start / "master.pth")
    torch.save(opt, start / "optimizer.pth")
    return start, hits


def arm(args):
    root = Path(args.root).resolve()
    n = DIMS[args.inputs]
    case = RELEASE / "expertium" / TRACK
    have = sha256(case / "master.pth")
    if have != SHA:
        raise SystemExit(f"release master sha256 {have} is not {SHA}")
    out = root / "artifacts" / "small-model" / f"sm179-tiny-{args.arm}"
    if out.exists():
        raise SystemExit(f"{out} exists")
    ctl_decay = json.loads((CTL / "decay.json").read_text())
    ctl_protocol = json.loads((CTL / "protocol.json").read_text())
    release_options = json.loads((RELEASE / "decay.json").read_text())["options"]
    options = dict(ctl_decay["options"])
    sv = options.pop("state_vq")
    if sv != {"recipe": "sm171-energy", "lambda": 0.0} or options != release_options:
        raise SystemExit("SM171's control options are not the release options plus state_vq lambda 0")
    start, hits = padded_start(out, n)
    full_run = dict(ctl_protocol["full_run"])
    full_run.update(init_master=(start / "master.pth").as_posix(), init_optimizer=(start / "optimizer.pth").as_posix(),
                    optimizer_state="the release model's end (its own cool-down's final state), first Linear widened by "
                                    f"{n} zero input columns at index {AT} (weight and Muon momentum)",
                    control_dir=(out / "control").relative_to(root).as_posix())
    protocol = dict(ctl_protocol, full_run=full_run)
    decay = {k: v for k, v in ctl_decay.items() if k != "sm171"}
    decay.update(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), options=options,
                 sm179={"arm": args.arm, "inputs": args.inputs, "extra_dims": n, "insert_at": AT,
                        "env": {"SM179_INPUTS": args.inputs}, "control": str(CTL),
                        "start_master_sha256": sha256(start / "master.pth"),
                        "start_optimizer_sha256": sha256(start / "optimizer.pth"), "widened": [WEIGHT] + hits,
                        "release_master_sha256": have, "release_optimizer_sha256": sha256(case / "optimizer.pth")})
    atomic_json(out / "decay.json", decay)
    atomic_json(out / "protocol.json", protocol)
    print(f"[sm179-prepare] {out.name}: inputs {args.inputs} (+{n} columns at {AT}), "
          f"{len(protocol['rows']):,} users, {decay['training_reviews']:,} reviews, lr {decay['lr']}")
    return 0


def pair(args):
    root = Path(args.root).resolve()
    ctl_d = json.loads((CTL / "decay.json").read_text())
    ctl_p = json.loads((CTL / "protocol.json").read_text())
    problems, info = [], {}
    for a in args.arms.split(","):
        f = root / "artifacts" / "small-model" / f"sm179-tiny-{a}"
        d = json.loads((f / "decay.json").read_text())
        pr = json.loads((f / "protocol.json").read_text())
        o = dict(ctl_d["options"])
        o.pop("state_vq")
        if d["options"] != o:
            problems.append(f"{a}: options differ from SM171's control without state_vq")
        if pr["rows"] != ctl_p["rows"]:
            problems.append(f"{a}: rows differ from SM171's control")
        keep = ("init_master", "init_optimizer", "optimizer_state", "control_dir")
        if {k: v for k, v in pr["full_run"].items() if k not in keep} != {k: v for k, v in ctl_p["full_run"].items() if k not in keep}:
            problems.append(f"{a}: full_run (schedule, epoch, snapshots) differs from SM171's control")
        if {k: v for k, v in pr.items() if k not in ("full_run",)} != {k: v for k, v in ctl_p.items() if k not in ("full_run",)}:
            problems.append(f"{a}: protocol fields differ from SM171's control")
        for k in ("training_users", "training_reviews", "lr", "training_seed"):
            if d.get(k) != ctl_d.get(k):
                problems.append(f"{a}: {k} differs")
        start = Path(pr["full_run"]["init_master"])
        if sha256(start) != d["sm179"]["start_master_sha256"]:
            problems.append(f"{a}: start master changed")
        info[a] = {"inputs": d["sm179"]["inputs"], "dims": d["sm179"]["extra_dims"]}
    print(json.dumps({"arms": info, "users": len(ctl_p["rows"]), "reviews": ctl_d["training_reviews"],
                      "ok": not problems, "problems": problems}))
    return 0 if not problems else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("arm")
    s.add_argument("--arm", required=True)
    s.add_argument("--inputs", required=True, choices=sorted(DIMS))
    s.add_argument("--root", required=True)
    s = sub.add_parser("pair")
    s.add_argument("--arms", required=True)
    s.add_argument("--root", required=True)
    args = p.parse_args()
    return arm(args) if args.cmd == "arm" else pair(args)


if __name__ == "__main__":
    raise SystemExit(main())
