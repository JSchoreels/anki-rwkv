#!/usr/bin/env python
"""Session inputs confirmation (SM181): prepare the Small arm (design.md "Axis 2").

The arm `sm181-small-ses` is the rank-2 cool-down test's (SM149) control cool-down of Small two-head (SM118), with only
the inputs changed (SM179_INPUTS=session, 6 columns): the same 794 users and 67,256,032 reviews, the same learning rate
(0 -> 7e-5 over 64 updates, then one_minus_sin to 0), the same seed 2025 and the release folder's options unchanged
(SM149's control ran the stock trainer, no track option). The start is the release model's final master.pth and
optimizer.pth with 6 zero columns inserted at input index 24 of features2card.0.weight (and of its Muon momentum), so
that step 0 computes exactly what the release model computes (as SM179 did for six-layer Tiny two-head (SM130)).

  python sm181_prepare.py arm  --root C:/Users/jonat/workspace/wt/sm179-run
  python sm181_prepare.py pair --root C:/Users/jonat/workspace/wt/sm179-run   (the arm vs SM149's control)
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

W = Path("C:/Users/jonat/workspace/wt")
RELEASE = W / "sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2"   # Small two-head (SM118)
TRACK = "E89-F65e-F77a"
SHA = "b4fe1175d25a7e6abff70e87ac0af1f4d76db9622d54f7cb53196fbf35117eb6"   # models.sh of SM148 / SM164
CTL = W / "sm149-run/artifacts/small-model/sm149-sm118-ctl"                # SM149's control (codec off, stock trainer)
ARM = "sm181-small-ses"
INPUTS = "session"
N = 6
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
    wide = [k for k, v in master.items() if v.dim() == 2 and v.shape[1] == 92]
    if wide != [WEIGHT]:
        raise ValueError(f"expected exactly {WEIGHT} with 92 input columns, got {wide}")
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
    if len(hits) != 1 or not hits[0].startswith("muon[") or not hits[0].endswith(".momentum_buffer"):
        raise ValueError(f"expected exactly the first Linear's Muon momentum, got {hits}")
    start = out / "start"
    start.mkdir(parents=True)
    torch.save(master, start / "master.pth")
    torch.save(opt, start / "optimizer.pth")
    return start, hits


def arm(args):
    root = Path(args.root).resolve()
    case = RELEASE / "expertium" / TRACK
    have = sha256(case / "master.pth")
    if have != SHA:
        raise SystemExit(f"release master sha256 {have} is not {SHA}")
    out = root / "artifacts" / "small-model" / ARM
    if out.exists():
        raise SystemExit(f"{out} exists")
    ctl_decay = json.loads((CTL / "decay.json").read_text())
    ctl_protocol = json.loads((CTL / "protocol.json").read_text())
    release_options = json.loads((RELEASE / "decay.json").read_text())["options"]
    if ctl_decay["options"] != release_options:
        raise SystemExit("SM149's control options are not the release options")
    if Path(ctl_protocol["full_run"]["init_master"]).resolve() != (case / "master.pth").resolve():
        raise SystemExit("SM149's control did not start from the release master")
    start, hits = padded_start(out, N)
    full_run = dict(ctl_protocol["full_run"])
    full_run.update(init_master=(start / "master.pth").as_posix(), init_optimizer=(start / "optimizer.pth").as_posix(),
                    optimizer_state="the release model's end (its own cool-down's final state), first Linear widened by "
                                    f"{N} zero input columns at index {AT} (weight and Muon momentum)",
                    control_dir=(out / "control").relative_to(root).as_posix())
    protocol = dict(ctl_protocol, full_run=full_run)
    decay = {k: v for k, v in ctl_decay.items() if k != "sm149"}
    decay.update(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), options=dict(ctl_decay["options"]),
                 sm181={"arm": "small-ses", "inputs": INPUTS, "extra_dims": N, "insert_at": AT,
                        "env": {"SM179_INPUTS": INPUTS}, "control": str(CTL),
                        "start_master_sha256": sha256(start / "master.pth"),
                        "start_optimizer_sha256": sha256(start / "optimizer.pth"), "widened": [WEIGHT] + hits,
                        "release_master_sha256": have, "release_optimizer_sha256": sha256(case / "optimizer.pth")})
    atomic_json(out / "decay.json", decay)
    atomic_json(out / "protocol.json", protocol)
    print(f"[sm181-prepare] {out.name}: inputs {INPUTS} (+{N} columns at {AT}, {hits}), "
          f"{len(protocol['rows']):,} users, {decay['training_reviews']:,} reviews, lr {decay['lr']}")
    return 0


def pair(args):
    root = Path(args.root).resolve()
    ctl_d = json.loads((CTL / "decay.json").read_text())
    ctl_p = json.loads((CTL / "protocol.json").read_text())
    problems = []
    f = root / "artifacts" / "small-model" / ARM
    d = json.loads((f / "decay.json").read_text())
    pr = json.loads((f / "protocol.json").read_text())
    if d["options"] != ctl_d["options"]:
        problems.append("options differ from SM149's control")
    if pr["rows"] != ctl_p["rows"]:
        problems.append("rows differ from SM149's control")
    keep = ("init_master", "init_optimizer", "optimizer_state", "control_dir")
    if {k: v for k, v in pr["full_run"].items() if k not in keep} != {k: v for k, v in ctl_p["full_run"].items() if k not in keep}:
        problems.append("full_run (schedule, epoch, snapshots) differs from SM149's control")
    if {k: v for k, v in pr.items() if k != "full_run"} != {k: v for k, v in ctl_p.items() if k != "full_run"}:
        problems.append("protocol fields differ from SM149's control")
    for k in ("training_users", "training_reviews", "lr"):
        if d.get(k) != ctl_d.get(k):
            problems.append(f"{k} differs")
    if d["options"].get("training_seed") != ctl_d["options"].get("training_seed"):
        problems.append("training seed differs")
    if sha256(Path(pr["full_run"]["init_master"])) != d["sm181"]["start_master_sha256"]:
        problems.append("start master changed")
    print(json.dumps({"arm": ARM, "inputs": d["sm181"]["inputs"], "users": len(ctl_p["rows"]),
                      "reviews": ctl_d["training_reviews"], "seed": ctl_d["options"].get("training_seed"),
                      "ok": not problems, "problems": problems}))
    return 0 if not problems else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("arm", "pair"):
        s = sub.add_parser(name)
        s.add_argument("--root", required=True)
    args = p.parse_args()
    return arm(args) if args.cmd == "arm" else pair(args)


if __name__ == "__main__":
    raise SystemExit(main())
