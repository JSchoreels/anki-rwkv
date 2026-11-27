#!/usr/bin/env python
"""4-bit weights (SM180): prepare the QAT arm and its control for one release model (design.md "Step 2: QAT").

The recipe is the SM171 short fine-tune (sm171_prepare.py), unchanged except the one track option: start = the release
model's final master.pth AND optimizer.pth, the release folder's options / loss / targets, whole users of
random.Random(171).shuffle of the release rows (sorted by user id) until 2^26 reviews, learning rate 0 -> 7e-5 over 64
updates then one_minus_sin to 0, snapshots every 100 users. The QAT arm adds
"state_vq": {"recipe": "weight-qat", "bits": 4, "group": 32}; the control has no extra option. For six-layer Tiny
two-head (SM130) the SM171 control (wt/sm171-run/artifacts/small-model/sm171-tiny-ctl) is the control: `pair` proves
it differs from the arm only by the option (its own option was the SM171 hook at weight 0, the stock arithmetic).

  python sm180_prepare.py arm  --model sm130th --arm qat --root C:/Users/jonat/workspace/wt/sm180-run
  python sm180_prepare.py arm  --model sm118   --arm ctl --root ...
  python sm180_prepare.py pair --model sm118 --root ...      (exit 0 = the pair differs only by the QAT option)
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
TRACK = "E89-F65e-F77a"
BUDGET = 1 << 26
QAT = {"recipe": "weight-qat", "bits": 4, "group": 32}
RELEASES = {   # key: (release folder, master sha256 (models.sh), plain name)
    "sm118": (W / "sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2",
              "b4fe1175d25a7e6abff70e87ac0af1f4d76db9622d54f7cb53196fbf35117eb6", "Small two-head (SM118)"),
    "sm113": (W / "sm113/artifacts/small-model/sm113-inst-e35st-st-dec2ep32k-t99-p2",
              "e4574e2aa8b2cfe6360dbd86f2a75b71d23ea81677b00ffcc9c3abce6b4cb736", "Small Instant-Only (SM113)"),
    "sm130th": (W / "sm130/artifacts/small-model/sm130-th-dec2ep32k-p2",
                "48a5e724cf38b53a32d570d4d034791f1da2cfcbe878572ff03ad434d7fb75af", "six-layer Tiny two-head (SM130)"),
    "sm130io": (W / "sm130/artifacts/small-model/sm130-io-dec2ep32k-p2",
                "23f037282379199bd113a5297737c920fcbefd702b7b76eecc0095b8a31186a2", "six-layer Tiny Instant-Only (SM130)"),
}
SM171_CTL = W / "sm171-run/artifacts/small-model/sm171-tiny-ctl"
SM171_CTL_SHA = "fe233f0c10281fdc43344cb081dc3edb6de706e1d8e9f863be3e5b72cf1e4eb4"


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


def folder(root, key, arm):
    if key == "sm130th" and arm == "ctl":
        return SM171_CTL
    return Path(root) / "artifacts" / "small-model" / f"sm180-{key}-{arm}"


def subset(rows, budget):   # sm171_prepare.subset, verbatim
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
    release, want, name = RELEASES[args.model]
    if args.model == "sm130th" and args.arm == "ctl":
        raise SystemExit("six-layer Tiny two-head (SM130): the SM171 control is reused, nothing to prepare")
    case = release / "expertium" / TRACK
    result = json.loads((case / "result.json").read_text())
    if result.get("status") != "complete" or result.get("smoke_only"):
        raise SystemExit(f"{case}: not a complete run")
    protocol = json.loads((release / "protocol.json").read_text())
    decay = json.loads((release / "decay.json").read_text())
    master, optimizer = case / "master.pth", case / "optimizer.pth"
    have = sha256(master)
    if have != want:
        raise SystemExit(f"{master}: sha256 {have} is not the release model's {want}")
    out = folder(args.root, args.model, args.arm)
    if out.exists():
        raise SystemExit(f"{out} exists")
    rows = subset(protocol["rows"], BUDGET)
    r = reviews(rows)
    options = dict(decay["options"])
    if "state_vq" in options:
        raise SystemExit("the release options already hold state_vq")
    if args.arm == "qat":
        options["state_vq"] = dict(QAT)
    old = protocol["full_run"]
    root = Path(args.root).resolve()
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
    new_decay = {k: v for k, v in decay.items() if k not in ("sm130", "sm118", "sm113")}
    lr = f"0 -> {PEAK:g} over {WARMUP} updates, then one_minus_sin to 0 over {r:,} reviews"
    new_decay.update(training_users=len(rows), training_reviews=r, training_reviews_after_trims=r,
                     fraction_of_epoch_reviews=r / reviews(protocol["rows"]), lr=lr,
                     created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), options=options,
                     source_case=str(case), start=str(release), start_master_sha256=have,
                     sm180={"arm": args.arm, "model": args.model, "name": name,
                            "what": ("QAT: 4-bit weights, groups of 32, straight-through" if args.arm == "qat" else
                                     "control: the identical fine-tune without the QAT option"),
                            "users": len(rows), "reviews": r, "review_budget": BUDGET, "subset_seed": SEED, "lr": lr,
                            "start_optimizer_sha256": sha256(optimizer),
                            "release_update_offset_end": result.get("update_offset_end"),
                            "release_trained_parameter_sha256": result.get("trained_parameter_sha256")})
    out.mkdir(parents=True)
    atomic_json(out / "decay.json", new_decay)
    atomic_json(out / "protocol.json", new_protocol)
    print(f"[sm180-prepare] {out.name}: {len(rows):,} users, {r:,} reviews, peak {PEAK:g}, from {master.parent.name} "
          f"{have[:12]}, {'QAT ' + json.dumps(QAT) if args.arm == 'qat' else 'control'}")
    return 0


def pair(args):
    release, want, name = RELEASES[args.model]
    fq, fc = folder(args.root, args.model, "qat"), folder(args.root, args.model, "ctl")
    dq, dc = (json.loads((f / "decay.json").read_text()) for f in (fq, fc))
    pq, pc = (json.loads((f / "protocol.json").read_text()) for f in (fq, fc))
    release_options = json.loads((release / "decay.json").read_text())["options"]
    problems = []
    oq, oc = dict(dq["options"]), dict(dc["options"])
    if oq.pop("state_vq", None) != QAT:
        problems.append("arm: state_vq is not the QAT option")
    sv = oc.pop("state_vq", None)
    if fc == SM171_CTL:
        if sv != {"recipe": "sm171-energy", "lambda": 0.0}:
            problems.append(f"SM171 control: state_vq {sv} (expected the weight-0 hook)")
        ctl_master = fc / "expertium" / TRACK / "master.pth"
        if not ctl_master.exists() or sha256(ctl_master) != SM171_CTL_SHA:
            problems.append("SM171 control: master.pth missing or sha256 changed")
    elif sv is not None:
        problems.append(f"control: state_vq {sv}")
    if oq != release_options or oc != release_options:
        problems.append("options beyond state_vq differ from the release folder's")
    if dq["start_master_sha256"] != want or dc["start_master_sha256"] != want:
        problems.append("start sha256")
    if dq.get("lr") != dc.get("lr") or dq.get("training_reviews") != dc.get("training_reviews"):
        problems.append("lr or training_reviews differ")
    frq, frc = dict(pq["full_run"]), dict(pc["full_run"])
    frq.pop("control_dir"), frc.pop("control_dir")
    if json.dumps(pq["rows"]) != json.dumps(pc["rows"]):
        problems.append("rows differ")
    if json.dumps(frq, sort_keys=True) != json.dumps(frc, sort_keys=True):
        problems.append("full_run differs")
    rest = lambda p: json.dumps({k: v for k, v in p.items() if k not in ("rows", "full_run")}, sort_keys=True)  # noqa
    if rest(pq) != rest(pc):
        problems.append("other protocol keys differ")
    print(json.dumps({"model": name, "arm": str(fq), "control": str(fc), "users": len(pq["rows"]),
                      "reviews": reviews(pq["rows"]), "ok": not problems, "problems": problems}))
    return 0 if not problems else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("arm")
    s.add_argument("--model", required=True, choices=sorted(RELEASES))
    s.add_argument("--arm", required=True, choices=("qat", "ctl"))
    s.add_argument("--root", required=True)
    s = sub.add_parser("pair")
    s.add_argument("--model", required=True, choices=sorted(RELEASES))
    s.add_argument("--root", required=True)
    args = p.parse_args()
    return arm(args) if args.cmd == "arm" else pair(args)


if __name__ == "__main__":
    raise SystemExit(main())
