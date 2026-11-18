"""Smaller model, card stream 1 layer (SM73 stage 2): the arm must equal the reference except the architecture, the start
weights and its own paths. Prints one line; exit 1 on any other difference.

  epoch  --arm <arm epoch folder> --ref <SM72s run epoch-21 folder>
         protocol.json equal except full_run init_master / init_optimizer / control_dir / benchmark_db (absolute vs
         relative); decay.json options = SM72s run.json options except "architecture" (w64-c1 vs w64)
  decay  --arm <arm cool-down folder> --ref <reference cool-down folder (tiny architecture additions (SM77) reference)>
         protocol.json equal except full_run init_master and control_dir (folder name); decay.json equal except
         created_utc, options/architecture + architecture (w64-c1 vs w64), the start fields (source_case, start,
         start_master_sha256, the sm77 note) and folder names; same users, order, update offset, seeds, schedule.
  --self-test  (decay) allow arm and reference to have the same architecture (checks the script on a w64 copy)
"""
import argparse
import json
from pathlib import Path

ARCH = "w64-c1"
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
    if Path(pa["full_run"]["benchmark_db"]).resolve() != (root / pb["full_run"]["benchmark_db"]).resolve():
        fail.append(f"benchmark_db {pa['full_run']['benchmark_db']} vs {pb['full_run']['benchmark_db']}")
    oa = dict(json.loads((a / "decay.json").read_text())["options"])
    run = dict(json.loads((b.parent / "run.json").read_text())["options"])
    if oa.pop("architecture", None) != ARCH or run.pop("architecture", None) != "w64":
        fail.append("architectures are not w64-c1 / w64")
    for path, x, y in leaves(oa, run):
        fail.append(f"options {path}: {str(x)[:100]} vs {str(y)[:100]}")
    return fail, (f"{len(pa['rows']):,} users, update offset {pa['full_run'].get('update_offset')}, start "
                  f"{Path(pa['full_run']['init_master']).parent.name}")


def decay(a, b, self_test):
    pa, pb = proto(a), proto(b)
    da, db = (json.loads((d / "decay.json").read_text()) for d in (a, b))
    fail = []
    for path, x, y in leaves(pa, pb):
        if path == "/full_run/init_master":
            continue
        if path == "/full_run/control_dir" and Path(str(x)) == Path(str(x)).parents[1] / a.name / "control" and                 Path(str(y)).parts[-2:] == (b.name, "control"):
            continue
        fail.append(f"protocol {path}: {str(x)[:100]} vs {str(y)[:100]}")
    want = ("w64", "w64") if self_test else (ARCH, "w64")
    if (da["options"].get("architecture"), db["options"].get("architecture")) != want or \
            (da.get("architecture"), db.get("architecture")) != want:
        fail.append(f"architectures {da['options'].get('architecture')} / {db['options'].get('architecture')}, want {want}")
    for path, x, y in leaves(da, db):
        if path in START_FIELDS or path in ("/options/architecture", "/architecture"):
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
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    a, b = Path(args.arm), Path(args.ref)
    fail, info = epoch(a, b) if args.kind == "epoch" else decay(a, b, args.self_test)
    if fail:
        print(f"[sm73s2-pair] {args.kind} REFUSE: " + "; ".join(fail[:12]))
        return 1
    print(f"[sm73s2-pair] {args.kind} OK: {a.name} == {b.name} except the architecture, the start weights and paths ({info})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
