"""The show-time two-head cool-down for Small (SM118): pair checks of the cool-down folders (prints one line; exit 1 on
any other difference). The leaf rule and the p2 check are the longer cool-down for Tiny (SM115)'s sm115_pair.py (= the
longer big instant-only teacher (SM113)'s sm113_pair.py).

  python sm118_pair.py recipe --arm <recipe folder> --ref92 <SM92's cool-down>
  python sm118_pair.py p1 --arm <pass-1 folder> --ref92 <SM92's cool-down> --ref113 <SM113's pass-1 folder>
  python sm118_pair.py p2 --arm <pass-2 folder> --p1 <pass-1 folder>

recipe: a folder this tree prepared from the show-time retrain of the release (SM92)'s constant-LR epoch 26 with SM92's
    own extra options (the old targets, never trained) == SM92's cool-down (protocol.prepared.json, i.e. as prepared)
    except full_run control_dir and decay created_utc (folder-name strings aside). So this tree prepares SM92's
    cool-down exactly: start, users, order, seeds, alphas, options, trims.
p1: the pass-1 folder == SM92's cool-down except full_run control_dir, full_run lr_resume ({"total_reviews": 2R,
    "reviews_offset": 0, "segment": true}, R = the rows' reviews), the targets (distillation teacher + replay: the
    show-time big model (SM91s2) and the new 8,000-user show-time folder) and the "sm118" note. So: SM92's start (its
    constant-LR epoch 26), w64, both heads trained (no curve_loss_weight), update offset, users, order, seeds, alphas,
    the show-time flag, trims.
    AND == SM113's pass-1 folder except control_dir, the start (full_run init_master / update_offset; decay start,
    source_case, source_update_offset_end, start_master_sha256, the sm77 note), the curve loss weight (SM113: 0.0, here
    absent = 1.0), the targets and the note keys; each start field must equal SM92's cool-down's. So: SM113's schedule
    (lr_resume), alphas, options, architecture (w64), users, order, seeds and trims.
p2: the pass-2 folder == the pass-1 folder except: rows = random.Random(2).shuffle of pass 1's rows (the same rows, a
    new order); full_run init_master / init_optimizer = pass 1's final master.pth / optimizer.pth (sha256 recorded),
    optimizer_state, update_offset = pass 1's update_offset_end, lr_resume = {"total_reviews": 2R, "reviews_offset": R},
    control_dir; decay.json except created_utc, source_case, source_update_offset_end, start, start_master_sha256 (==
    sha256 of pass 1's master), the "sm118" note and folder-name strings. Options (targets, alphas, optimizer, ...) equal.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

TRACK = "E89-F65e-F77a"
TEACHER = "C:/Users/jonat/workspace/wt/sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth"
REPLAY = "C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/sm91s2-big-showtime-final8k-neighbour-e2s"


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


def info_of(a, pa, da):
    r = reviews(pa["rows"])
    return (f"{len(pa['rows']):,} users, {r:,} reviews per pass, update offset {pa['full_run'].get('update_offset')}, start "
            f"{str(da.get('start_master_sha256'))[:12]}, arch {da.get('architecture')}, trims "
            f"{[t['user_id'] for t in da.get('trimmed_unlabelled_tails', [])]}")


def pair_recipe(a, r92):
    pa, p92 = proto(a), proto(r92)
    da, d92 = (json.loads((d / "decay.json").read_text()) for d in (a, r92))
    fail = same_but(a, r92, pa, p92, da, d92, set(), {"/created_utc"})
    if "lr_resume" in pa["full_run"] or "lr_resume" in p92["full_run"]:
        fail.append("an lr_resume is set")
    return fail, info_of(a, pa, da)


def pair_p1(a, r92, r113):
    pa, p92, p113 = proto(a), proto(r92), proto(r113)
    da, d92, d113 = (json.loads((d / "decay.json").read_text()) for d in (a, r92, r113))
    r = reviews(pa["rows"])
    fail = []
    if pa["full_run"].get("lr_resume") != {"total_reviews": 2 * r, "reviews_offset": 0, "segment": True}:
        fail.append(f"lr_resume {pa['full_run'].get('lr_resume')} is not the first half of a 2 x {r:,} decay")
    if "lr_resume" in p92["full_run"]:
        fail.append("SM92's prepared protocol has an lr_resume")
    if p113["full_run"].get("lr_resume") != pa["full_run"].get("lr_resume"):
        fail.append("SM113's pass 1 has another lr_resume")
    fail += [f"vs SM92: {x}" for x in same_but(a, r92, pa, p92, da, d92, {"/full_run/lr_resume"},
                                               {"/created_utc", "/sm118", "/options/distillation/teacher",
                                                "/options/distillation/replay"})]
    start = {"/full_run/init_master", "/full_run/update_offset"}
    fail += [f"vs SM113: {x}" for x in same_but(a, r113, pa, p113, da, d113, start,
                                                {"/created_utc", "/sm113", "/sm118", "/sm77", "/source_case",
                                                 "/source_update_offset_end", "/start", "/start_master_sha256",
                                                 "/options/curve_loss_weight", "/options/distillation/teacher",
                                                 "/options/distillation/replay"})]
    for key in ("init_master", "update_offset"):
        if pa["full_run"].get(key) != p92["full_run"].get(key):
            fail.append(f"full_run {key} is not SM92's cool-down's")
    for key in ("architecture", "sm77", "source_case", "source_update_offset_end", "start", "start_master_sha256"):
        if da.get(key) != d92.get(key):
            fail.append(f"decay {key} is not SM92's cool-down's")
    if da.get("architecture") != "w64" or da["options"].get("architecture") != "w64":
        fail.append("architecture is not w64")
    k = da["options"]["distillation"]
    if Path(k["teacher"]).resolve() != Path(TEACHER).resolve() or Path(k["replay"]).resolve() != Path(REPLAY).resolve():
        fail.append(f"the targets are not SM91s2's 8,000-user show-time targets ({k.get('teacher')}, {k.get('replay')})")
    if da["options"].get("show_time_intervals") is not True or "curve_loss_weight" in da["options"]:
        fail.append("show_time_intervals is not true or a curve_loss_weight is set (both heads must train)")
    if d113["options"].get("curve_loss_weight") != 0.0:
        fail.append("SM113's pass 1 is not the instant-only recipe it should be")
    return fail, info_of(a, pa, da) + f", schedule over {2 * r:,}"


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
              "start_optimizer_sha256": (db.get("sm118") or {}).get("start_optimizer_sha256") == sha256(case / "optimizer.pth"),
              "pass 1 complete": result.get("status") == "complete" and not result.get("smoke_only")}
    fail += [f"{k} wrong" for k, ok in checks.items() if not ok]
    fail += same_but(b, a, dict(pb, rows=None), dict(pa, rows=None), db, da,
                     {"/full_run/init_master", "/full_run/init_optimizer", "/full_run/optimizer_state",
                      "/full_run/update_offset", "/full_run/lr_resume"},
                     {"/created_utc", "/sm118", "/source_case", "/source_update_offset_end", "/start", "/start_master_sha256"})
    info = (f"{len(pb['rows']):,} users (new order), {r:,} reviews, from pass 1's end (master {str(db.get('start_master_sha256'))[:12]}, "
            f"update offset {fr.get('update_offset')}), lr_resume {json.dumps(fr.get('lr_resume'))}")
    return fail, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["recipe", "p1", "p2"])
    p.add_argument("--arm", required=True)
    p.add_argument("--ref92", default="")
    p.add_argument("--ref113", default="")
    p.add_argument("--p1", default="")
    args = p.parse_args()
    if args.mode == "recipe":
        fail, info = pair_recipe(Path(args.arm), Path(args.ref92))
        what = "== SM92's cool-down except paths (this tree prepares it exactly)"
    elif args.mode == "p1":
        fail, info = pair_p1(Path(args.arm), Path(args.ref92), Path(args.ref113))
        what = ("== SM92's cool-down except lr_resume (first half of a 2-epoch decay), the targets (SM91s2's) and paths; "
                "== SM113's pass 1 except the start (SM92's epoch 26), the curve loss weight (both heads), the targets and paths")
    else:
        fail, info = pair_p2(Path(args.arm), Path(args.p1))
        what = "== pass 1 except the start (pass 1's end), the new order and lr_resume (second half)"
    if fail:
        print(f"[sm118-pair] {args.mode} REFUSE: " + "; ".join(fail[:12]))
        return 1
    print(f"[sm118-pair] {args.mode} OK: {Path(args.arm).name} {what} ({info})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
