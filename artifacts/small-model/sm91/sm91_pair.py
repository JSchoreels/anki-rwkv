"""Show-time interval retrain (SM91): the arm must equal the reference except the show-time inputs flag, the start
weights and its own paths. Prints one line; exit 1 on any other difference.

  epoch  --arm <arm epoch folder> --ref <SM72s run epoch-21 folder>
         protocol.json equal except full_run init_master / init_optimizer / control_dir / benchmark_db (absolute vs
         relative; the start = SM72s epoch 20 itself, checked by path); decay.json options = SM72s run.json options plus
         exactly {"show_time_intervals": true}
  decay  --arm <arm cool-down folder> --ref <reference cool-down folder (tiny architecture additions (SM77) reference)>
         protocol.json equal except full_run init_master and control_dir (folder name); decay.json equal except
         created_utc, options/show_time_intervals (true in the arm, absent in the reference), the start fields
         (source_case, start, start_master_sha256, the sm77 note) and folder names; same users, order, update offset,
         seeds, schedule, architecture (w64)
  --same-start  (decay) the start fields must be equal too (a cool-down prepared from the reference's own start)
"""
import argparse
import json
from pathlib import Path

FLAG = "show_time_intervals"
START_FIELDS = {"/source_case", "/start", "/start_master_sha256", "/sm77", "/created_utc"}


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


def epoch(a, b):
    pa, pb = proto(a), proto(b)
    fail = []
    for path, x, y in leaves(pa, pb):
        if path in ("/full_run/init_master", "/full_run/init_optimizer", "/full_run/control_dir", "/full_run/benchmark_db"):
            continue
        fail.append(f"protocol {path}: {str(x)[:100]} vs {str(y)[:100]}")
    root = next(q for q in b.resolve().parents if (q / "scripts").is_dir() and (q / "artifacts").is_dir())
    for key in ("benchmark_db", "init_master", "init_optimizer"):
        if Path(pa["full_run"][key]).resolve() != (root / pb["full_run"][key]).resolve():
            fail.append(f"{key} {pa['full_run'][key]} vs {pb['full_run'][key]}")
    oa = dict(json.loads((a / "decay.json").read_text())["options"])
    run = dict(json.loads((b.parent / "run.json").read_text())["options"])
    if oa.pop(FLAG, None) is not True or FLAG in run:
        fail.append(f"{FLAG} is not true in the arm only")
    for path, x, y in leaves(oa, run):
        fail.append(f"options {path}: {str(x)[:100]} vs {str(y)[:100]}")
    return fail, (f"{len(pa['rows']):,} users, update offset {pa['full_run'].get('update_offset')}, start "
                  f"{Path(pa['full_run']['init_master']).parents[2].name}")


def decay(a, b, same_start):
    pa, pb = proto(a), proto(b)
    da, db = (json.loads((d / "decay.json").read_text()) for d in (a, b))
    fail = []
    for path, x, y in leaves(pa, pb):
        if path == "/full_run/init_master" and not same_start:
            continue
        if path == "/full_run/control_dir" and Path(str(x)) == Path(str(x)).parents[1] / a.name / "control" and \
                Path(str(y)).parts[-2:] == (b.name, "control"):
            continue
        fail.append(f"protocol {path}: {str(x)[:100]} vs {str(y)[:100]}")
    if da["options"].get(FLAG) is not True or FLAG in db["options"]:
        fail.append(f"{FLAG} is not true in the arm only")
    for path, x, y in leaves(da, db):
        if path == f"/options/{FLAG}" or path == "/created_utc" or (path in START_FIELDS and not same_start):
            continue
        if isinstance(x, str) and isinstance(y, str) and x.replace(a.name, b.name) == y:
            continue
        fail.append(f"decay {path}: {str(x)[:100]} vs {str(y)[:100]}")
    if da.get("source_update_offset_end") != db.get("source_update_offset_end"):
        fail.append("start update offsets differ")
    return fail, (f"{len(pa['rows']):,} users, update offset {pa['full_run'].get('update_offset')}, "
                  f"{da.get('training_reviews'):,} reviews, start {str(da.get('start_master_sha256'))[:12]} vs "
                  f"{str(db.get('start_master_sha256'))[:12]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=("epoch", "decay"))
    p.add_argument("--arm", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--same-start", action="store_true")
    args = p.parse_args()
    a, b = Path(args.arm), Path(args.ref)
    fail, info = epoch(a, b) if args.kind == "epoch" else decay(a, b, args.same_start)
    if fail:
        print(f"[sm91-pair] {args.kind} REFUSE: " + "; ".join(fail[:12]))
        return 1
    print(f"[sm91-pair] {args.kind} OK: {a.name} == {b.name} except the show-time inputs flag, the start weights and paths ({info})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
