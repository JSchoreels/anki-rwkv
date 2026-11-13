"""The longer big instant-only teacher (SM113): pair checks of the two cool-down folders (prints one line; exit 1 on any
other difference). Same leaf rule as the big instant-only teacher (SM110)'s sm110_pair.py.

  python sm113_pair.py p1 --arm <pass-1 folder> --ref110 <SM110's cool-down> --ref98 <SM98's cool-down>
  python sm113_pair.py p2 --arm <pass-2 folder> --p1 <pass-1 folder>

p1: the pass-1 folder == SM110's cool-down (protocol.prepared.json, i.e. as prepared) except full_run control_dir
    (folder name) and full_run lr_resume, which must be {"total_reviews": 2R, "reviews_offset": 0, "segment": true}
    with R = the rows' reviews; decay.json equal except created_utc, the "sm113" note and strings that only differ by
    the folder name. So: SM110's start (SM98's epoch 35), users, order, seeds, alphas, curve loss weight 0, show-time
    flag, trims, AND the same SM99 targets (teacher + replay folder). Then, as SM110 proved it, == SM98's cool-down except
    the teacher + replay folder (+ the same lr_resume / note / paths).
p2: the pass-2 folder == the pass-1 folder except: rows = random.Random(2).shuffle of pass 1's rows (the same rows, a
    new order); full_run init_master / init_optimizer = pass 1's final master.pth / optimizer.pth (sha256 recorded),
    optimizer_state, update_offset = pass 1's update_offset_end, lr_resume = {"total_reviews": 2R, "reviews_offset": R},
    control_dir; decay.json except created_utc, source_case, source_update_offset_end, start, start_master_sha256 (==
    sha256 of pass 1's master), the "sm113" note and folder-name strings. Options (targets, alphas, optimizer, ...) equal.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

TRACK = "E89-F65e-F77a"


def leaves(x, y, path=""):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            yield from leaves(x.get(k), y.get(k), f"{path}/{k}")
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (u, v) in enumerate(zip(x, y)):
            yield from leaves(u, v, f"{path}[{i}]")
    elif x != y:
        yield path, x, y


def proto(d):
    q = d / "protocol.prepared.json"   # a resumed run rewrites protocol.json to the users left
    return json.loads((q if q.exists() else d / "protocol.json").read_text())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def reviews(rows):
    return sum(r["train_range"][1] - r["train_range"][0] for r in rows)


def same_but(a, b, pa, pb, da, db, protocol_ok, decay_ok):
    fail = []
    for path, x, y in leaves(pa, pb):
        if path == "/full_run/control_dir" and Path(str(x)).parts[-2:] == (a.name, "control") and \
                Path(str(y)).parts[-2:] == (b.name, "control"):
            continue
        if path in protocol_ok or any(path.startswith(p + "/") for p in protocol_ok):
            continue
        fail.append(f"protocol {path}: {str(x)[:100]} vs {str(y)[:100]}")
    for path, x, y in leaves(da, db):
        if path in decay_ok or any(path.startswith(p + "/") for p in decay_ok):
            continue
        if isinstance(x, str) and isinstance(y, str) and x.replace(a.name, b.name) == y:
            continue
        fail.append(f"decay {path}: {str(x)[:100]} vs {str(y)[:100]}")
    return fail


def pair_p1(a, r110, r98):
    pa, p110, p98 = proto(a), proto(r110), proto(r98)
    da, d110, d98 = (json.loads((d / "decay.json").read_text()) for d in (a, r110, r98))
    r = reviews(pa["rows"])
    fail = []
    if pa["full_run"].get("lr_resume") != {"total_reviews": 2 * r, "reviews_offset": 0, "segment": True}:
        fail.append(f"lr_resume {pa['full_run'].get('lr_resume')} is not the first half of a 2 x {r:,} decay")
    if "lr_resume" in p110["full_run"] or "lr_resume" in p98["full_run"]:
        fail.append("a reference's prepared protocol has an lr_resume")
    fail += same_but(a, r110, pa, p110, da, d110, {"/full_run/lr_resume"}, {"/created_utc", "/sm113"})
    fail += [f"vs SM98: {x}" for x in same_but(a, r98, pa, p98, da, d98, {"/full_run/lr_resume"},
                                               {"/created_utc", "/sm113", "/options/distillation/teacher",
                                                "/options/distillation/replay"})]
    k, k98 = da["options"]["distillation"], d98["options"]["distillation"]
    for key in ("teacher", "replay"):
        if Path(k[key]).resolve() == Path(k98[key]).resolve():
            fail.append(f"distillation {key} is SM98's ({k[key]})")
    if da["options"].get("show_time_intervals") is not True or da["options"].get("curve_loss_weight") != 0.0:
        fail.append("show_time_intervals is not true or curve_loss_weight is not 0.0")
    info = (f"{len(pa['rows']):,} users, {r:,} reviews per pass, schedule over {2 * r:,}, update offset "
            f"{pa['full_run'].get('update_offset')}, start {str(da.get('start_master_sha256'))[:12]}, trims "
            f"{[t['user_id'] for t in da.get('trimmed_unlabelled_tails', [])]}")
    return fail, info


def pair_p2(b, a):
    pb, pa = proto(b), proto(a)
    db, da = (json.loads((d / "decay.json").read_text()) for d in (b, a))
    case = a / "expertium" / TRACK
    result = json.loads((case / "result.json").read_text())
    r = reviews(pa["rows"])
    fail = []
    want = list(pa["rows"])
    random.Random(2).shuffle(want)
    if pb["rows"] != want:
        fail.append("rows are not random.Random(2).shuffle of pass 1's rows")
    if sorted(json.dumps(x, sort_keys=True) for x in pb["rows"]) != sorted(json.dumps(x, sort_keys=True) for x in pa["rows"]):
        fail.append("rows are not the same multiset as pass 1's")
    fr = pb["full_run"]
    checks = {"init_master": Path(fr.get("init_master", "")).resolve() == (case / "master.pth").resolve(),
              "init_optimizer": Path(fr.get("init_optimizer", "")).resolve() == (case / "optimizer.pth").resolve(),
              "update_offset": fr.get("update_offset") == result.get("update_offset_end"),
              "lr_resume": fr.get("lr_resume") == {"total_reviews": 2 * r, "reviews_offset": r},
              "start_master_sha256": db.get("start_master_sha256") == sha256(case / "master.pth"),
              "start_optimizer_sha256": (db.get("sm113") or {}).get("start_optimizer_sha256") == sha256(case / "optimizer.pth"),
              "pass 1 complete": result.get("status") == "complete" and not result.get("smoke_only")}
    fail += [f"{k} wrong" for k, ok in checks.items() if not ok]
    fail += same_but(b, a, dict(pb, rows=None), dict(pa, rows=None), db, da,
                     {"/full_run/init_master", "/full_run/init_optimizer", "/full_run/optimizer_state",
                      "/full_run/update_offset", "/full_run/lr_resume"},
                     {"/created_utc", "/sm113", "/source_case", "/source_update_offset_end", "/start", "/start_master_sha256"})
    info = (f"{len(pb['rows']):,} users (new order), {r:,} reviews, from pass 1's end (master {str(db.get('start_master_sha256'))[:12]}, "
            f"update offset {fr.get('update_offset')}), lr_resume {json.dumps(fr.get('lr_resume'))}")
    return fail, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["p1", "p2"])
    p.add_argument("--arm", required=True)
    p.add_argument("--ref110", default="")
    p.add_argument("--ref98", default="")
    p.add_argument("--p1", default="")
    args = p.parse_args()
    if args.mode == "p1":
        fail, info = pair_p1(Path(args.arm), Path(args.ref110), Path(args.ref98))
        what = "== SM110's cool-down except lr_resume (first half of a 2-epoch decay) and paths; == SM98's except also the targets"
    else:
        fail, info = pair_p2(Path(args.arm), Path(args.p1))
        what = "== pass 1 except the start (pass 1's end), the new order and lr_resume (second half)"
    if fail:
        print(f"[sm113-pair] {args.mode} REFUSE: " + "; ".join(fail[:12]))
        return 1
    print(f"[sm113-pair] {args.mode} OK: {Path(args.arm).name} {what} ({info})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
