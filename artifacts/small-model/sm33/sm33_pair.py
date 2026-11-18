"""Grade-order sanity (SM33): the arm cool-down must equal the tiny architecture additions (SM77) reference cool-down
except the grade_order option (decay.json options) and its own folder paths: same start weights, users, order, update
offset, seeds, schedule and every other option. Prints one line; exit 1 on any other difference.

  python artifacts/small-model/sm33/sm33_pair.py --arm <arm folder> --ref <reference folder>
"""
import argparse
import json
from pathlib import Path


def leaves(x, y, path=""):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            yield from leaves(x.get(k), y.get(k), f"{path}/{k}")
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (u, v) in enumerate(zip(x, y)):
            yield from leaves(u, v, f"{path}[{i}]")
    elif x != y:
        yield path, x, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--ref", required=True)
    args = p.parse_args()
    a, b = Path(args.arm), Path(args.ref)

    def proto(d):
        q = d / "protocol.prepared.json"   # a resumed cool-down rewrites protocol.json to the users left
        return json.loads((q if q.exists() else d / "protocol.json").read_text())

    pa, pb = proto(a), proto(b)
    da, db = (json.loads((d / "decay.json").read_text()) for d in (a, b))
    fail = []
    for path, x, y in leaves(pa, pb):
        if path == "/full_run/control_dir" and str(x).replace(a.name, b.name) == str(y):
            continue
        fail.append(f"protocol {path}: {str(x)[:100]} vs {str(y)[:100]}")
    grade = da["options"].get("grade_order")
    for path, x, y in leaves(da, db):
        if path == "/created_utc" or path == "/options/grade_order":
            continue
        if isinstance(x, str) and isinstance(y, str) and x.replace(a.name, b.name) == y:
            continue
        fail.append(f"decay {path}: {str(x)[:100]} vs {str(y)[:100]}")
    if not grade or db["options"].get("grade_order"):
        fail.append(f"grade_order must be on in the arm only (arm {grade}, reference {db['options'].get('grade_order')})")
    if fail:
        print("[sm33-pair] decay REFUSE: " + "; ".join(fail))
        return 1
    print(f"[sm33-pair] decay OK: {a.name} == {b.name} except grade_order {json.dumps(grade)} and paths "
          f"({len(pa['rows']):,} users, same order and start weights {str(da.get('start_master_sha256'))[:12]}, update offset "
          f"{pa['full_run'].get('update_offset')}, {da.get('training_reviews'):,} reviews)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
