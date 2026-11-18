"""Show-time interval retrain (SM91) step 3: the new cool-down must equal step 1's arm cool-down except the KD replay
targets (distillation "teacher" + "replay": the show-time big model and its show-time targets) and its own folder name.
Prints one line; exit 1 on any other difference.

  python artifacts/small-model/sm91s3/sm91s3_pair.py --arm <new cool-down> --ref <step 1's arm cool-down> [--targets]

protocol (protocol.prepared.json if present, else protocol.json): equal except full_run control_dir (folder name);
decay.json: equal except created_utc, options/distillation/teacher, options/distillation/replay and strings that only
differ by the folder name. Same start (source_case, start, start_master_sha256), users, order, update offset, seeds,
schedule, architecture, show-time flag. The teacher and the replay folder must both differ from the reference's.
--targets  also: the arm's replay manifest is an SM11 export of the arm's teacher (teacher_sha256 == sha256 of the file),
           inputs "e2s-query", deleted_cards "neighbour", same dataset path / revision as the reference's targets.
"""
import argparse
import hashlib
import json
from pathlib import Path

ALLOWED = {"/created_utc", "/options/distillation/teacher", "/options/distillation/replay"}


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


def pair(a, b, targets):
    pa, pb = proto(a), proto(b)
    da, db = (json.loads((d / "decay.json").read_text()) for d in (a, b))
    fail = []
    for path, x, y in leaves(pa, pb):
        if path == "/full_run/control_dir" and Path(str(x)).parts[-2:] == (a.name, "control") and \
                Path(str(y)).parts[-2:] == (b.name, "control"):
            continue
        fail.append(f"protocol {path}: {str(x)[:100]} vs {str(y)[:100]}")
    for path, x, y in leaves(da, db):
        if path in ALLOWED:
            continue
        if isinstance(x, str) and isinstance(y, str) and x.replace(a.name, b.name) == y:
            continue
        fail.append(f"decay {path}: {str(x)[:100]} vs {str(y)[:100]}")
    ka, kb = da["options"]["distillation"], db["options"]["distillation"]
    for key in ("teacher", "replay"):
        if Path(ka[key]).resolve() == Path(kb[key]).resolve():
            fail.append(f"distillation {key} is the reference's ({ka[key]})")
    if da["options"].get("show_time_intervals") is not True:
        fail.append("show_time_intervals is not true")
    info = ""
    if targets:
        ma = json.loads((Path(ka["replay"]) / "manifest.json").read_text())
        mb = json.loads((Path(kb["replay"]) / "manifest.json").read_text())
        teacher_sha = sha256(ka["teacher"])
        if ma.get("teacher_sha256") != teacher_sha:
            fail.append(f"replay manifest teacher {str(ma.get('teacher_sha256'))[:12]} != sha256 of {ka['teacher']} {teacher_sha[:12]}")
        if ma.get("inputs") != "e2s-query" or "inputs" in mb:
            fail.append(f"replay inputs {ma.get('inputs')} (want e2s-query) vs reference {mb.get('inputs', 'normal')}")
        for key in ("deleted_cards", "dataset_path", "dataset_revision", "format", "kind", "layout", "teacher_precision",
                    "encoding_seed", "max_pass_reviews", "burn_in_reviews"):
            if ma.get(key) != mb.get(key):
                fail.append(f"replay manifest {key}: {ma.get(key)} vs {mb.get(key)}")
        info = f"; targets: teacher {teacher_sha[:12]}, inputs {ma.get('inputs')}, deleted_cards {ma.get('deleted_cards')}"
    return fail, (f"{len(pa['rows']):,} users, update offset {pa['full_run'].get('update_offset')}, "
                  f"{da.get('training_reviews'):,} reviews, start {str(da.get('start_master_sha256'))[:12]}{info}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--targets", action="store_true")
    args = p.parse_args()
    a, b = Path(args.arm), Path(args.ref)
    fail, info = pair(a, b, args.targets)
    if fail:
        print(f"[sm91s3-pair] REFUSE: " + "; ".join(fail[:12]))
        return 1
    print(f"[sm91s3-pair] OK: {a.name} == {b.name} except the KD replay targets (teacher + replay folder) and paths ({info})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
