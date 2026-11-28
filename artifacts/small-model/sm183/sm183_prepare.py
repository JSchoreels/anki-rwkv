#!/usr/bin/env python
"""Tiny cool-down with session inputs (SM183): prepare the training folders in the run tree (design.md "Recipe").

Every folder is a copy of the release cool-down of six-layer Tiny (SM130) (the two passes `sm130-<chain>-dec2ep32k-p1`
/ `-p2`, chain th = two-head, io = Instant-Only) with only the start changed:
  p1   SM130's pass-1 protocol.json and decay.json byte for byte in content (same 8,000 users, same rows, same order,
       same lr_resume = first half of one one_minus_sin decay 7e-4 -> 0 over 2R, same seed, options and teacher
       targets, fresh Muon), with
         arm ses: init_master = SM130's own pass-1 start (th: recovery epoch 32, io: epoch 41) with 6 zero columns
                  inserted at input index 24 of features2card.0.weight (the other weights bit for bit);
                  SM179_INPUTS=session;
         arm ctl: init_master unchanged (SM130's start), SM179_INPUTS unset (used only if the control reproduction
                  check fails, design.md "Control");
         ctlcheck: the ctl arm in a separate folder for the control reproduction check (first 400 users);
  p2   sm130_prepare.py p2's code: the arm's p1 end (master.pth and optimizer.pth, Muon state continued, update offset
       continued), random.Random(2).shuffle of p1's rows, lr_resume = second half of the 2R decay.

  python sm183_prepare.py p1 --chain th --arm ses --root <run tree>
  python sm183_prepare.py p1 --chain th --arm ctlcheck --root <run tree>
  python sm183_prepare.py p2 --chain th --arm ses --root <run tree>          (after p1 is complete)
  python sm183_prepare.py pair --root <run tree>                             (every prepared folder vs SM130's)
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch

W = Path("C:/Users/jonat/workspace/wt")
SM = W / "sm130/artifacts/small-model"
TRACK = "E89-F65e-F77a"
# SM130's pass-1 start masters (decay.json start_master_sha256) and release masters (pass-2 ends)
START_SHA = {"th": "635f1a12893b4d87c7c946e6311103de3aa519417535919fb4c5e13314ea6a46"}
RELEASE_SHA = {"th": "48a5e724cf38b53a32d570d4d034791f1da2cfcbe878572ff03ad434d7fb75af"}
INPUTS = {"ses": "session", "ctl": "", "ctlcheck": ""}
N = 6
AT = 24   # the extra columns follow the 24 deterministic features (rwkv/extra_inputs.py)
WEIGHT = "features2card.0.weight"
P2_SHUFFLE_SEED = 2
FREE = ("init_master", "init_optimizer", "optimizer_state", "control_dir")


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


def src(chain, p):
    return SM / f"sm130-{chain}-dec2ep32k-p{p}"


def name(chain, arm, p):
    return f"sm183-{chain}-ctlcheck" if arm == "ctlcheck" else f"sm183-{chain}-{arm}-p{p}"


def widen(t):
    if t.dim() != 2 or t.shape[1] != 92:
        raise ValueError(f"expected a (rows, 92) tensor, got {tuple(t.shape)}")
    z = torch.zeros(t.shape[0], N, dtype=t.dtype)
    return torch.cat([t[:, :AT], z, t[:, AT:]], 1).contiguous()


def p1(args):
    root = Path(args.root).resolve()
    s = src(args.chain, 1)
    decay = json.loads((s / "decay.json").read_text())
    protocol = json.loads((s / "protocol.json").read_text())
    if (s / "protocol.prepared.json").exists() or (s / "attempts").exists():
        raise SystemExit(f"{s} was resumed; its protocol.json is not the prepared one")
    start = Path(protocol["full_run"]["init_master"])
    have = sha256(start)
    if have != decay["start_master_sha256"] or (args.chain in START_SHA and have != START_SHA[args.chain]):
        raise SystemExit(f"{start}: sha256 {have} is not SM130's pass-1 start")
    if protocol["full_run"].get("optimizer_state") != "fresh" or "init_optimizer" in protocol["full_run"]:
        raise SystemExit(f"{s}: pass 1 is not a fresh-optimizer start")
    out = root / "artifacts" / "small-model" / name(args.chain, args.arm, 1)
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.mkdir(parents=True)
    full_run = dict(protocol["full_run"], control_dir=(out / "control").relative_to(root).as_posix())
    info = {"chain": args.chain, "arm": args.arm, "pass": 1, "inputs": INPUTS[args.arm], "source": str(s),
            "sm130_start": str(start), "sm130_start_sha256": have}
    if args.arm == "ses":
        master = torch.load(start, map_location="cpu", weights_only=True)
        master[WEIGHT] = widen(master[WEIGHT])
        (out / "start").mkdir()
        torch.save(master, out / "start" / "master.pth")
        full_run["init_master"] = (out / "start" / "master.pth").as_posix()
        info.update(extra_dims=N, insert_at=AT, widened=[WEIGHT], start_master_sha256=sha256(out / "start" / "master.pth"))
    protocol = dict(protocol, full_run=full_run)
    decay = dict(decay, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), sm183=info)
    atomic_json(out / "decay.json", decay)
    atomic_json(out / "protocol.json", protocol)
    print(f"[sm183-prepare] {out.name}: {len(protocol['rows']):,} users, {reviews(protocol['rows']):,} reviews, "
          f"inputs {INPUTS[args.arm] or '(unset)'}, start {Path(full_run['init_master']).name} of {Path(full_run['init_master']).parent.name}, "
          f"lr_resume {json.dumps(full_run.get('lr_resume'))}")
    return 0


def p2(args):
    """sm130_prepare.py p2 with the names changed."""
    root = Path(args.root).resolve()
    a = root / "artifacts" / "small-model" / name(args.chain, args.arm, 1)
    out = root / "artifacts" / "small-model" / name(args.chain, args.arm, 2)
    if out.exists():
        raise SystemExit(f"{out} exists")
    case = a / "expertium" / TRACK
    result = json.loads((case / "result.json").read_text())
    if result.get("status") != "complete" or result.get("smoke_only"):
        raise SystemExit(f"{case} is not complete")
    for f in ("master.pth", "optimizer.pth"):
        if not (case / f).exists():
            raise SystemExit(f"{case / f} missing")
    q = a / "protocol.prepared.json"   # a resumed p1 rewrote protocol.json to the users it had left
    protocol = json.loads((q if q.exists() else a / "protocol.json").read_text())
    decay = json.loads((a / "decay.json").read_text())
    r = reviews(protocol["rows"])
    seg = protocol["full_run"].get("lr_resume")
    if seg != {"total_reviews": 2 * r, "reviews_offset": 0, "segment": True}:
        raise SystemExit(f"{a}: p1 lr_resume {seg} is not the first half of a 2R decay")
    if int(result.get("review_presentations", -1)) != r and not (a / "resume.json").exists():
        raise SystemExit(f"{a}: trained {result.get('review_presentations')} reviews, expected {r:,}")
    rows = list(protocol["rows"])
    random.Random(P2_SHUFFLE_SEED).shuffle(rows)
    full_run = dict(protocol["full_run"])
    full_run.pop("resumed", None)
    full_run.update(init_master=(case / "master.pth").as_posix(), init_optimizer=(case / "optimizer.pth").as_posix(),
                    optimizer_state="continued from pass 1 (Muon state of p1's end)",
                    update_offset=int(result["update_offset_end"]),
                    lr_resume={"total_reviews": 2 * r, "reviews_offset": r},
                    control_dir=(out / "control").relative_to(root).as_posix())
    protocol = dict(protocol, rows=rows, full_run=full_run)
    info = dict(decay["sm183"], **{"pass": 2, "p1": str(a), "start_master_sha256": sha256(case / "master.pth"),
                                   "start_optimizer_sha256": sha256(case / "optimizer.pth"),
                                   "p1_resumed": (a / "resume.json").exists(),
                                   "pass1_trained_parameter_sha256": result.get("trained_parameter_sha256")})
    s2 = json.loads((src(args.chain, 2) / "decay.json").read_text())
    decay = dict(decay, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), source_case=str(case),
                 source_update_offset_end=int(result["update_offset_end"]), start=str(a),
                 start_master_sha256=info["start_master_sha256"], sm130=s2.get("sm130"), sm183=info)
    out.mkdir(parents=True)
    atomic_json(out / "decay.json", decay)
    atomic_json(out / "protocol.json", protocol)
    print(f"[sm183-prepare] p2 {out.name}: {len(rows):,} users, {r:,} reviews, from {case.parent.parent.name} master "
          f"{info['start_master_sha256'][:12]}, update offset {full_run['update_offset']}, lr_resume {json.dumps(full_run['lr_resume'])}")
    return 0


def check_folder(f, p, problems, warnings):
    """One prepared folder against SM130's folder of the same pass."""
    d = json.loads((f / "decay.json").read_text())
    pr = json.loads((f / "protocol.json").read_text())
    if (f / "protocol.prepared.json").exists():   # a resumed folder: compare the prepared protocol
        pr = json.loads((f / "protocol.prepared.json").read_text())
    chain = d["sm183"]["chain"]
    s = src(chain, p)
    sd = json.loads((s / "decay.json").read_text())
    sp = json.loads((s / "protocol.json").read_text())
    tag = f.name
    if d["options"] != sd["options"]:
        problems.append(f"{tag}: options differ from {s.name}")
    if pr["rows"] != sp["rows"]:
        problems.append(f"{tag}: rows (users, ranges or order) differ from {s.name}")
    free = FREE + (("update_offset",) if p == 2 else ())
    if {k: v for k, v in pr["full_run"].items() if k not in free} != {k: v for k, v in sp["full_run"].items() if k not in free}:
        problems.append(f"{tag}: full_run (schedule, lr_resume, epoch, snapshots) differs from {s.name}")
    if {k: v for k, v in pr.items() if k != "full_run"} != {k: v for k, v in sp.items() if k != "full_run"}:
        problems.append(f"{tag}: protocol fields differ from {s.name}")
    for k in ("training_users", "training_reviews", "training_reviews_after_trims", "lr", "shuffle_seed", "track", "model",
              "architecture"):
        if d.get(k) != sd.get(k):
            problems.append(f"{tag}: decay {k} differs")
    if p == 1:
        if pr["full_run"].get("optimizer_state") != "fresh" or "init_optimizer" in pr["full_run"]:
            problems.append(f"{tag}: pass 1 must start with a fresh optimizer")
        start = Path(pr["full_run"]["init_master"])
        if d["sm183"]["arm"] == "ses":
            if sha256(start) != d["sm183"]["start_master_sha256"]:
                problems.append(f"{tag}: start master changed")
            a = torch.load(start, map_location="cpu", weights_only=True)
            b = torch.load(sp["full_run"]["init_master"], map_location="cpu", weights_only=True)
            if set(a) != set(b) or any(not torch.equal(a[k], b[k]) for k in b if k != WEIGHT):
                problems.append(f"{tag}: the padded start differs from SM130's start outside {WEIGHT}")
            w, v = a[WEIGHT], b[WEIGHT]
            if not (w.shape[1] == v.shape[1] + N and torch.equal(w[:, :AT], v[:, :AT]) and torch.equal(w[:, AT + N:], v[:, AT:])
                    and not w[:, AT:AT + N].any()):
                problems.append(f"{tag}: {WEIGHT} is not SM130's with {N} zero columns at {AT}")
        elif pr["full_run"]["init_master"] != sp["full_run"]["init_master"]:
            problems.append(f"{tag}: the control start is not SM130's")
    else:
        so = sp["full_run"]["update_offset"]
        if pr["full_run"]["update_offset"] != so:
            msg = f"{tag}: update offset {pr['full_run']['update_offset']} vs SM130's {so}"
            (warnings if d["sm183"].get("p1_resumed") else problems).append(msg)
    return {"chain": chain, "arm": d["sm183"]["arm"], "pass": p, "inputs": d["sm183"]["inputs"] or "(unset)",
            "users": len(pr["rows"]), "reviews": reviews(pr["rows"])}


def pair(args):
    root = Path(args.root).resolve()
    problems, warnings, info = [], [], {}
    for f in sorted((root / "artifacts" / "small-model").glob("sm183-*")):
        if not (f / "decay.json").exists():
            continue
        d = json.loads((f / "decay.json").read_text())
        info[f.name] = check_folder(f, int(d["sm183"]["pass"]), problems, warnings)
    print(json.dumps({"folders": info, "ok": bool(info) and not problems, "problems": problems, "warnings": warnings}))
    return 0 if info and not problems else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("p1", "p2"):
        s = sub.add_parser(c)
        s.add_argument("--chain", required=True, choices=["th", "io"])
        s.add_argument("--arm", required=True, choices=sorted(INPUTS) if c == "p1" else ["ses", "ctl"])
        s.add_argument("--root", required=True)
    s = sub.add_parser("pair")
    s.add_argument("--root", required=True)
    args = p.parse_args()
    return {"p1": p1, "p2": p2, "pair": pair}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
