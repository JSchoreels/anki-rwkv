"""Trained smaller state for Tiny (SM137): pair checks of the cool-down folders (prints one line; exit 1 on any other
difference). The code is the six-layer Tiny with longer recovery (SM135)'s sm135_pair.py (= the six-layer Tiny (SM130)'s
sm130_pair.py) with the note key "sm135" -> "sm137" and the architecture w64-c1-d2p1u1 -> w64-c1-d2p1u1-v{V}k{K} (--lanes,
default v16k8); the start is SM137's lane epoch (Instant-Only sm137-io-v16k8-e42-st, two-head sm137-th-v16k8-e33-st).
NEW in SM137: p1 is ALSO checked against the six-layer Tiny (SM130)'s own pass 1 (sm130-{chain}-dec2ep32k-p1) with the
same generic leaf rule, so the folder is SM130's cool-down except the start and the lane architecture. SM135's text follows.

  python sm137_pair.py p1 --chain th|io --arm <pass-1 folder> --start <the lane epoch folder> [--lanes v16k8]
  python sm137_pair.py p2 --arm <pass-2 folder> --p1 <pass-1 folder>
  python sm137_pair.py selfcheck

p1 (two-head, chain th): the pass-1 folder == the show-time two-head cool-down for Tiny (SM124)'s pass 1 (as prepared)
    except control_dir, the start (full_run init_master / update_offset; decay start, source_case,
    source_update_offset_end, start_master_sha256, the sm77 note), the architecture (w64-c1-d2p1u1-v16k8 here, w64-c1
    there) and the note keys. The start must be the given epoch's trained master at its update_offset_end, and its
    sha256 is recorded. So: SM124's users, order, seeds, trims, the 2R schedule (lr_resume), the show-time big model
    (SM91s2)'s targets, alphas, both heads, show-time inputs, 32k windows, fresh Muon, snapshots every 400 users.
p1 (Instant-Only, chain io): the same against the longer cool-down for Tiny (SM115)'s pass 1 (the show-time Instant-Only
    big model (SM99)'s targets, curve_loss_weight 0).
p2: the pass-2 folder == the pass-1 folder except: rows = random.Random(2).shuffle of pass 1's rows; full_run
    init_master / init_optimizer = pass 1's final master.pth / optimizer.pth, optimizer_state, update_offset = pass 1's
    update_offset_end, lr_resume = {"total_reviews": 2R, "reviews_offset": R}, control_dir; decay.json except created_utc,
    source_case, source_update_offset_end, start, start_master_sha256 (== sha256 of pass 1's master), the "sm137" note.
selfcheck: the generic p1 leaf rule accepts SM124's pass 1 against the show-time two-head cool-down for Small (SM118)'s
    pass 1, SM115's pass 1 against the longer big instant-only teacher (SM113)'s pass 1 (the same rule with w64 there),
    and SM130's two passes 1 against SM124's / SM115's.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

TRACK = "E89-F65e-F77a"
ARCH = "w64-c1-d2p1u1-v16k8"   # set from --lanes in main()
W = Path("C:/Users/jonat/workspace/wt")
REFS = {"th": {"p1": W / "sm124/artifacts/small-model/sm124-c1-e29st-dec2ep32k-t91s2-p1", "note": "/sm124", "arch": "w64-c1",
               "name": "the show-time two-head cool-down for Tiny (SM124)'s pass 1",
               "teacher": "C:/Users/jonat/workspace/wt/sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth",
               "replay": "C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/sm91s2-big-showtime-final8k-neighbour-e2s",
               "curve_loss_weight": None},
        "io": {"p1": W / "sm115/artifacts/small-model/sm115-c1-e38st-st-dec2ep32k-t99-p1", "note": "/sm115", "arch": "w64-c1",
               "name": "the longer cool-down for Tiny (SM115)'s pass 1",
               "teacher": "C:/Users/jonat/workspace/wt/sm99/artifacts/small-model/sm99-big-inst-e2/expertium/big-inst/master.pth",
               "replay": "C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/sm99-big-inst-final8k-neighbour-e2s",
               "curve_loss_weight": 0.0}}
SM130_P1 = {c: W / f"sm130/artifacts/small-model/sm130-{c}-dec2ep32k-p1" for c in ("th", "io")}
SELF = {"th": (W / "sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p1", "/sm118"),
        "io": (W / "sm113/artifacts/small-model/sm113-inst-e35st-st-dec2ep32k-t99-p1", "/sm113")}


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


START_PROTO = {"/full_run/init_master", "/full_run/update_offset"}
START_DECAY = {"/created_utc", "/sm77", "/source_case", "/source_update_offset_end", "/start", "/start_master_sha256",
               "/architecture", "/options/architecture"}


def generic(a, ref, note_a, note_ref):
    pa, pr = proto(a), proto(ref)
    da, dr = (json.loads((d / "decay.json").read_text()) for d in (a, ref))
    return same_but(a, ref, pa, pr, da, dr, START_PROTO, START_DECAY | {note_a, note_ref}), pa, da


def pair_p1(chain, a, start):
    ref = REFS[chain]
    fail, pa, da = generic(a, ref["p1"], "/sm137", ref["note"])
    r = reviews(pa["rows"])
    case = start / "expertium" / TRACK
    result = json.loads((case / "result.json").read_text())
    if pa["full_run"].get("lr_resume") != {"total_reviews": 2 * r, "reviews_offset": 0, "segment": True}:
        fail.append(f"lr_resume {pa['full_run'].get('lr_resume')} is not the first half of a 2 x {r:,} decay")
    if Path(pa["full_run"].get("init_master", "")).resolve() != (case / "master.pth").resolve() and \
            Path(pa["full_run"].get("init_master", "")).resolve() != (ROOT_OF(a) / pa["full_run"].get("init_master", "")).resolve():
        fail.append(f"init_master {pa['full_run'].get('init_master')} is not {case / 'master.pth'}")
    if pa["full_run"].get("update_offset") != result.get("update_offset_end"):
        fail.append(f"update_offset {pa['full_run'].get('update_offset')} is not the start's end {result.get('update_offset_end')}")
    if da.get("start_master_sha256") != sha256(case / "master.pth"):
        fail.append("start_master_sha256 is not the start's master")
    if result.get("status") != "complete" or result.get("smoke_only"):
        fail.append(f"{start.name} is not a complete epoch")
    if da.get("architecture") != ARCH or da["options"].get("architecture") != ARCH:
        fail.append(f"architecture is not {ARCH}")
    rd = json.loads((ref["p1"] / "decay.json").read_text())
    if rd.get("architecture") != ref["arch"]:
        fail.append(f"the reference pass 1 is not {ref['arch']}")
    k = da["options"]["distillation"]
    if Path(k["teacher"]).resolve() != Path(ref["teacher"]).resolve() or Path(k["replay"]).resolve() != Path(ref["replay"]).resolve():
        fail.append(f"the targets are not the reference's ({k.get('teacher')}, {k.get('replay')})")
    if da["options"].get("show_time_intervals") is not True or da["options"].get("curve_loss_weight") != ref["curve_loss_weight"]:
        fail.append("show_time_intervals is not true or curve_loss_weight is not the reference's")
    f130, _, _ = generic(a, SM130_P1[chain], "/sm137", "/sm130")
    fail += [f"vs SM130's pass 1: {x}" for x in f130]
    d130 = json.loads((SM130_P1[chain] / "decay.json").read_text())
    if d130.get("architecture") + "-" + ARCH.rsplit("-", 1)[1] != ARCH:
        fail.append(f"SM130's pass 1 is {d130.get('architecture')}, not the base of {ARCH}")
    info = (f"{len(pa['rows']):,} users, {r:,} reviews per pass, schedule over {2 * r:,}, update offset "
            f"{pa['full_run'].get('update_offset')}, start {str(da.get('start_master_sha256'))[:12]}, arch {da.get('architecture')}, "
            f"trims {[t['user_id'] for t in da.get('trimmed_unlabelled_tails', [])]}")
    return fail, info


def ROOT_OF(d):
    return next(q for q in Path(d).resolve().parents if (q / "scripts").is_dir() and (q / "artifacts").is_dir())


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
              "start_optimizer_sha256": (db.get("sm137") or {}).get("start_optimizer_sha256") == sha256(case / "optimizer.pth"),
              "pass 1 complete": result.get("status") == "complete" and not result.get("smoke_only")}
    fail += [f"{k} wrong" for k, ok in checks.items() if not ok]
    fail += same_but(b, a, dict(pb, rows=None), dict(pa, rows=None), db, da,
                     {"/full_run/init_master", "/full_run/init_optimizer", "/full_run/optimizer_state",
                      "/full_run/update_offset", "/full_run/lr_resume"},
                     {"/created_utc", "/sm137", "/source_case", "/source_update_offset_end", "/start", "/start_master_sha256"})
    info = (f"{len(pb['rows']):,} users (new order), {r:,} reviews, from pass 1's end (master {str(db.get('start_master_sha256'))[:12]}, "
            f"update offset {fr.get('update_offset')}), lr_resume {json.dumps(fr.get('lr_resume'))}")
    return fail, info


def selfcheck():
    bad = 0
    for chain, (ref, note) in SELF.items():
        fail, _, _ = generic(REFS[chain]["p1"], ref, REFS[chain]["note"], note)
        print(f"[sm137-pair] selfcheck {REFS[chain]['p1'].name} vs {ref.name}: {'OK' if not fail else 'REFUSE ' + '; '.join(fail[:6])}")
        bad += bool(fail)
    for chain in ("th", "io"):
        fail, _, _ = generic(SM130_P1[chain], REFS[chain]["p1"], "/sm130", REFS[chain]["note"])
        print(f"[sm137-pair] selfcheck {SM130_P1[chain].name} vs {REFS[chain]['p1'].name}: {'OK' if not fail else 'REFUSE ' + '; '.join(fail[:6])}")
        bad += bool(fail)
    return 1 if bad else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["p1", "p2", "selfcheck"])
    p.add_argument("--chain", choices=sorted(REFS), default="th")
    p.add_argument("--arm", default="")
    p.add_argument("--start", default="")
    p.add_argument("--p1", default="")
    p.add_argument("--lanes", default="v16k8")
    args = p.parse_args()
    global ARCH
    ARCH = "w64-c1-d2p1u1-" + args.lanes
    if args.mode == "selfcheck":
        return selfcheck()
    if args.mode == "p1":
        fail, info = pair_p1(args.chain, Path(args.arm), Path(args.start))
        what = f"== {REFS[args.chain]['name']} and SM130's pass 1 except the start ({Path(args.start).name}), the architecture ({ARCH}) and paths"
    else:
        fail, info = pair_p2(Path(args.arm), Path(args.p1))
        what = "== pass 1 except the start (pass 1's end), the new order and lr_resume (second half)"
    if fail:
        print(f"[sm137-pair] {args.mode} {args.chain} REFUSE: " + "; ".join(fail[:12]))
        return 1
    print(f"[sm137-pair] {args.mode} {args.chain} OK: {Path(args.arm).name} {what} ({info})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
