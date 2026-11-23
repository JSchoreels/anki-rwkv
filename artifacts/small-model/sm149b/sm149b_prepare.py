#!/usr/bin/env python
"""Rank-2 states with a tail-energy penalty (SM149b): prepare the new arm (design.md "Arms").

The arm is SM149's QAT arm of Small two-head (SM118) (sm149_prepare.py arm --arm qat: the same start weights and Muon
state, users, seed, review budget, learning-rate schedule and options) with the penalty in its state_vq option:
{"recipe": "lr2q8x8", "tail_lambda": <lambda>, "tail_stride": 8}. SM149's own prepare code builds the folder (only
its folder name is changed); `pair` proves that the folder differs from SM149's trained QAT arm folder
(wt/sm149-run/artifacts/small-model/sm149-sm118-qat) only by those two option keys and the sm149b note.

  python sm149b_prepare.py arm  --lam 1.0 --root C:/Users/jonat/workspace/wt/sm149-run     -> sm149b-sm118-tail1
  python sm149b_prepare.py pair --lam 1.0 --root C:/Users/jonat/workspace/wt/sm149-run
  (the pre-registered follow-up, only after a partial result: --lam 3.0 -> sm149b-sm118-tail3)
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sm149_prepare", HERE.parent / "sm149" / "sm149_prepare.py")
P = importlib.util.module_from_spec(spec)
sys.modules["sm149_prepare"] = P
spec.loader.exec_module(P)

MODEL = "sm118"
STRIDE = 8
SM149_QAT_FOLDER = "sm149-sm118-qat"


def name(lam):
    return f"sm149b-{MODEL}-tail{lam:g}"


def arm(args):
    root = Path(args.root).resolve()
    P.folder_name = lambda tag, arm_: name(args.lam)
    rc = P.arm(argparse.Namespace(model=MODEL, arm="qat", root=str(root)))
    if rc:
        return rc
    out = root / "artifacts" / "small-model" / name(args.lam)
    decay = json.loads((out / "decay.json").read_text())
    decay["options"]["state_vq"] = {"recipe": P.RECIPE, "tail_lambda": args.lam, "tail_stride": STRIDE}
    decay["sm149b"] = {"what": "SM149's rank-2 QAT arm plus the tail-energy penalty (design.md)", "lambda": args.lam,
                       "stride": STRIDE, "same_as": SM149_QAT_FOLDER}
    decay["sm149"]["what"] = "rank-2 low-rank state QAT cool-down (straight-through) + tail-energy penalty (SM149b)"
    P.atomic_json(out / "decay.json", decay)
    print(f"[sm149b-prepare] {out.name}: state_vq {decay['options']['state_vq']}")
    return 0


def pair(args):
    root = Path(args.root).resolve()
    a = root / "artifacts" / "small-model" / name(args.lam)
    b = root / "artifacts" / "small-model" / SM149_QAT_FOLDER
    da, db = (json.loads((d / "decay.json").read_text()) for d in (a, b))
    pa, pb = (json.loads((d / "protocol.json").read_text()) for d in (a, b))
    oa, ob = dict(da["options"]), dict(db["options"])
    problems = []
    sa, sb = oa.pop("state_vq"), ob.pop("state_vq")
    if sa != {"recipe": P.RECIPE, "tail_lambda": args.lam, "tail_stride": STRIDE}:
        problems.append(f"new arm state_vq {sa}")
    if sb != {"recipe": P.RECIPE}:
        problems.append(f"SM149 QAT arm state_vq {sb}")
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
    for key in ("start_master_sha256", "training_reviews", "training_users", "lr", "source_case", "start"):
        if da.get(key) != db.get(key):
            problems.append(f"decay.json {key} differs")
    if da["start_master_sha256"] != P.SHA[MODEL]:
        problems.append("start sha")
    print(json.dumps({"new": a.name, "sm149_qat": b.name, "users": len(pa["rows"]), "reviews": P.reviews(pa["rows"]),
                      "ok": not problems, "problems": problems}))
    return 0 if not problems else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd in ("arm", "pair"):
        s = sub.add_parser(cmd)
        s.add_argument("--lam", type=float, default=1.0)
        s.add_argument("--root", required=True)
    args = p.parse_args()
    return arm(args) if args.cmd == "arm" else pair(args)


if __name__ == "__main__":
    raise SystemExit(main())
