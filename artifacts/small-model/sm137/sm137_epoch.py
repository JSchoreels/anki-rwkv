#!/usr/bin/env python
"""Trained smaller state for Tiny (SM137): the first constant-LR epoch after the lane cut (the pilot epoch), CPU only.
Each folder it writes is trained by scripts/expertium_decay.py run (launch-sm137.sh).

The epoch is the six-layer Tiny with longer recovery (SM135)'s first epoch step with ONE change, the architecture:
  io  epoch 42 = exactly SM135's Instant-Only epoch 42 (the run's order rule over the 8,000 train users, AdamW 7e-4
      constant, 2048 x 16 lanes, leak-free replay KD from the big model (1 leak-free epoch) alpha 0.9, instant head only,
      show-time inputs), start = the lane cut of the six-layer Tiny (SM130)'s epoch 41 (sm137_lanes.py mask), at SM130
      epoch 41's update offset. SM135 trained that epoch from the uncut epoch 41 (sm135-io-e42-st, read only): it is the
      matched control of the pilot (same data order, same start weights except the cut lanes, same options).
  th  epoch 33 = SM135's two-head epoch-33 step (never trained by SM135), start = the lane cut of SM130's epoch 32.

  prepare --chain io|th --lanes v16k8   write sm137-{chain}-{lanes}-e{N}-st/{protocol.json, decay.json}
  pair --chain io|th --lanes v16k8      the folder == SM135's epoch-N folder (io: the trained one on disk; th: what
                                        sm135_epoch.build would write) except the architecture, the start (init_master /
                                        init_optimizer and their sha256), control_dir, start, note and created_utc. Exit 1
                                        otherwise.
"""
from argparse import ArgumentParser
import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
SM = ROOT / "artifacts/small-model"
W = Path("C:/Users/jonat/workspace/wt")
TRACK = "E89-F65e-F77a"
ARCH = "w64-c1-d2p1u1"
FIRST = {"io": 42, "th": 33}
CONTROL = {"io": W / "sm135/artifacts/small-model/sm135-io-e42-st", "th": None}
ALLOWED = {"protocol": {"/full_run/init_master", "/full_run/init_optimizer", "/full_run/control_dir"},
           "decay": {"/options/architecture", "/architecture", "/start", "/note", "/init_master_sha256",
                     "/init_optimizer_sha256", "/created_utc"}}


def sm135():
    spec = importlib.util.spec_from_file_location("sm135_epoch", SM / "sm135/sm135_epoch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lanes_of(text):
    m = re.fullmatch(r"v([0-9]+)k([0-9]+)", text)
    if not m:
        raise SystemExit(f"--lanes {text!r}: expected v<V>k<K>, e.g. v16k8")
    return int(m.group(1)), int(m.group(2))


def cut_dir(chain, lanes):
    return SM / f"sm137-{chain}-{lanes}-e{FIRST[chain] - 1}"


def folder(chain, lanes):
    return SM / f"sm137-{chain}-{lanes}-e{FIRST[chain]}-st"


def build(chain, lanes):
    e = sm135()
    v, k = lanes_of(lanes)
    arch = f"{ARCH}-v{v}k{k}"
    cut = cut_dir(chain, lanes)
    rec = json.loads((cut / "extend.json").read_text())
    src = e.CHAINS[chain]
    if rec.get("architecture") != arch or rec.get("chain") != chain or rec.get("source_master_sha256") != src["src_master_sha"] \
            or rec.get("source_optimizer_sha256") != src["src_optimizer_sha"] or rec.get("master_sha256") != e.sha256(cut / "master.pth") \
            or rec.get("optimizer_sha256") != e.sha256(cut / "optimizer.pth") or not rec.get("optimizer_states_equal_after_load") \
            or not rec.get("dropped_rows_and_moments_zero"):
        raise SystemExit(f"{cut.name}: not the pinned lane cut of {src['name']} at {arch}")
    protocol, decay = e.build(chain, FIRST[chain])
    out = folder(chain, lanes)
    protocol["full_run"].update(init_master=(cut / "master.pth").as_posix(), init_optimizer=(cut / "optimizer.pth").as_posix(),
                                control_dir=(out / "control").relative_to(ROOT).as_posix())
    decay["options"]["architecture"] = arch
    decay.update(architecture=arch, start=cut.name, init_master_sha256=rec["master_sha256"],
                 init_optimizer_sha256=rec["optimizer_sha256"],
                 note=f"SM137 ({chain}): the six-layer Tiny with longer recovery (SM135)'s {chain} epoch step, epoch "
                      f"{FIRST[chain]} (the run's order rule), architecture {arch} (card / note state cut to {v} value x {k} "
                      f"key lanes per head), start = {cut.name} (the lane cut of {src['name']})")
    return protocol, decay


def prepare(args):
    out = folder(args.chain, args.lanes)
    if out.exists():
        raise SystemExit(f"{out} exists")
    protocol, decay = build(args.chain, args.lanes)
    decay["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    e = sm135()
    e.atomic_json(tmp / "protocol.json", protocol)
    e.atomic_json(tmp / "decay.json", decay)
    tmp.rename(out)
    print(json.dumps({k: decay[k] for k in ("epoch", "architecture", "training_users", "training_reviews", "start", "note")}))
    return 0


def pair(args):
    e = sm135()
    out = folder(args.chain, args.lanes)
    mine = {"protocol": e.proto(out) if out.exists() else build(args.chain, args.lanes)[0],
            "decay": e.decay_of(out) if out.exists() else build(args.chain, args.lanes)[1]}
    ctl = CONTROL[args.chain]
    if ctl is not None:
        ref = {"protocol": e.proto(ctl), "decay": e.decay_of(ctl)}
        what = ctl.name + " (trained, on disk)"
    else:
        p, d = e.build(args.chain, FIRST[args.chain])
        ref, what = {"protocol": p, "decay": d}, f"sm135_epoch.build({args.chain!r}, {FIRST[args.chain]})"
    bad, differ = [], []
    for name in ("protocol", "decay"):
        for path, x, y in e.leaves(mine[name], ref[name]):
            if x != y:
                (differ if path in ALLOWED[name] else bad).append(f"{name}{path}")
    v, k = lanes_of(args.lanes)
    if mine["decay"].get("architecture") != f"{ARCH}-v{v}k{k}" or mine["decay"]["options"].get("architecture") != f"{ARCH}-v{v}k{k}":
        bad.append("architecture is not the lane architecture")
    print(json.dumps({"folder": out.name, "reference": what, "differs_where_allowed": sorted(differ), "bad": bad[:20]}))
    print(f"[sm137-epoch] pair {args.chain} {args.lanes}: {'OK' if not bad else 'FAILED'}")
    return 0 if not bad else 1


def main():
    p = ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "pair"):
        s = sub.add_parser(name)
        s.add_argument("--chain", choices=FIRST, required=True)
        s.add_argument("--lanes", default="v16k8")
    args = p.parse_args()
    return {"prepare": prepare, "pair": pair}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
