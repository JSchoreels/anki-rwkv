#!/usr/bin/env python
"""The longer cool-down for Tiny (SM115): the two-epoch cool-down as two chained one-epoch decay folders
(the code of the longer big instant-only teacher (SM113)'s sm113_prepare.py; only the names and the note key differ).

One one_minus_sin decay 7e-4 -> 0 over 2 x R reviews (R = 581,942,724, the 8,000 users after the one trim) is split at
its middle into two folders that each train the 8,000 users once:
  p1  learning rate over [0, R) of the 2R schedule (7e-4 -> 0.293 x 7e-4), fresh Muon, from the Tiny Instant-Only
      model (SM101)'s constant-LR epoch 38 (w64-c1): the folder `sm77_warm.py cooldown --epoch 25 --arch w64-c1` prepares
      with SM113's extra options (the show-time instant-only big model (SM99)'s show-time targets), plus
      full_run.lr_resume = {"total_reviews": 2R, "reviews_offset": 0, "segment": true}.
  p2  learning rate over [R, 2R) (0.293 x 7e-4 -> 0), from p1's final master.pth AND optimizer.pth (Muon state
      continued, update offset continued), the same rows in a new order (random.Random(2).shuffle of p1's rows),
      lr_resume = {"total_reviews": 2R, "reviews_offset": R}.
Everything else is p1's (sm115_pair.py).

  python sm115_prepare.py p1 --dec DIR1                  (after sm77_warm cooldown + the label-window trim; once)
  python sm115_prepare.py p2 --p1 DIR1 --out DIR2        (after p1 is complete)
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRACK = "E89-F65e-F77a"
P2_SHUFFLE_SEED = 2


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


def p1(args):
    d = Path(args.dec).resolve()
    if (d / "protocol.prepared.json").exists() or (d / "attempts").exists() or (d / "expertium").exists():
        raise SystemExit(f"{d} has already run; p1 is set before the first run only")
    protocol = json.loads((d / "protocol.json").read_text())
    decay = json.loads((d / "decay.json").read_text())
    if not decay.get("trimmed_unlabelled_tails"):
        raise SystemExit(f"{d}: the label-window trim pass has not run")
    r = reviews(protocol["rows"])
    if r != decay["training_reviews_after_trims"]:
        raise SystemExit(f"{d}: rows hold {r:,} reviews, decay.json says {decay['training_reviews_after_trims']:,}")
    want = {"total_reviews": 2 * r, "reviews_offset": 0, "segment": True}
    have = protocol["full_run"].get("lr_resume")
    if have == want:
        print(f"[sm115-prepare] p1 already set: {json.dumps(want)}")
        return 0
    if have is not None:
        raise SystemExit(f"{d}: unexpected lr_resume {have}")
    protocol["full_run"]["lr_resume"] = want
    decay["sm115"] = {"two_epoch_cooldown": "pass 1 of 2", "reviews_per_pass": r, "schedule_total_reviews": 2 * r,
                      "lr": "7e-4 -> 0.2929 x 7e-4 (one_minus_sin over the first half of 2R)",
                      "set_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_json(d / "protocol.json", protocol)
    atomic_json(d / "decay.json", decay)
    print(f"[sm115-prepare] p1 {d.name}: {len(protocol['rows']):,} users, {r:,} reviews, lr_resume {json.dumps(want)}")
    return 0


def p2(args):
    a, out = Path(args.p1).resolve(), Path(args.out).resolve()
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
                    control_dir=(out / "control").relative_to(ROOT).as_posix())
    protocol = dict(protocol, rows=rows, full_run=full_run)
    decay = dict(decay, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), source_case=str(case),
                 source_update_offset_end=int(result["update_offset_end"]), start=str(a),
                 start_master_sha256=sha256(case / "master.pth"),
                 sm115={"two_epoch_cooldown": "pass 2 of 2", "reviews_per_pass": r, "schedule_total_reviews": 2 * r,
                        "lr": "0.2929 x 7e-4 -> 0 (one_minus_sin over the second half of 2R)",
                        "order": f"random.Random({P2_SHUFFLE_SEED}).shuffle of pass 1's rows",
                        "start_optimizer_sha256": sha256(case / "optimizer.pth"),
                        "pass1_trained_parameter_sha256": result.get("trained_parameter_sha256")})
    out.mkdir(parents=True)
    atomic_json(out / "decay.json", decay)
    atomic_json(out / "protocol.json", protocol)
    print(f"[sm115-prepare] p2 {out.name}: {len(rows):,} users, {r:,} reviews, from {case.name} master "
          f"{decay['start_master_sha256'][:12]}, update offset {full_run['update_offset']}, lr_resume {json.dumps(full_run['lr_resume'])}")
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("p1"); s.add_argument("--dec", required=True)
    s = sub.add_parser("p2"); s.add_argument("--p1", required=True); s.add_argument("--out", required=True)
    args = p.parse_args()
    return {"p1": p1, "p2": p2}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
