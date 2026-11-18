"""The smaller instant-only model (SM101): cool-down config proof. The instant-only model, stage 2 (SM57)'s `sm57_pair.py
decay` rule (read only, wt/sm57) with one more allowed difference, the architecture: the arm is "w64-c1" (card stream
1 layer) in both decay.json fields, the reference "w64". Prints one line; exit 1 on any other difference.

  decay --arm <cool-down folder> --ref <reference cool-down folder> [--same-start] [--ref-arch w64|w64-c1]
        protocol.json equal except full_run init_master and update_offset (the start; the offset only enters the 5-update
        LR warm-up, both are > 500k so the factor is 1; with --same-start they must be equal too) and control_dir (folder
        name); decay.json equal except created_utc, the architecture (w64-c1 vs w64), the start fields and folder names;
        fresh Muon; same users, order, seeds, schedule, KD replay targets, options.
"""
import argparse
import json
from pathlib import Path

ARCH, BASE = "w64-c1", "w64"
START_FIELDS = {"/source_case", "/source_update_offset_end", "/start", "/start_master_sha256", "/sm77", "/created_utc"}
ARCH_PATHS = {"/architecture", "/options/architecture"}
WARMUP = 5


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


def decay(a, b, same_start, base=BASE):
    pa, pb = proto(a), proto(b)
    da, db = (json.loads((d / "decay.json").read_text()) for d in (a, b))
    fail = []
    for path, x, y in leaves(pa, pb):
        if path in ("/full_run/init_master", "/full_run/update_offset") and not same_start:
            continue
        if path == "/full_run/init_master" and Path(str(x)).resolve() == Path(str(y)).resolve():
            continue
        if path == "/full_run/control_dir" and Path(str(x)) == Path(str(x)).parents[1] / a.name / "control" and \
                Path(str(y)).parts[-2:] == (b.name, "control"):
            continue
        fail.append(f"protocol {path}: {str(x)[:100]} vs {str(y)[:100]}")
    if min(pa["full_run"]["update_offset"], pb["full_run"]["update_offset"]) < WARMUP:
        fail.append("an update offset inside the LR warm-up")
    if pa["full_run"].get("optimizer_state") != "fresh" or "init_optimizer" in pa["full_run"]:
        fail.append("the cool-down does not start a fresh Muon state")
    if (da.get("architecture"), da["options"].get("architecture")) != (ARCH, ARCH):
        fail.append(f"arm architecture {da.get('architecture')} / {da['options'].get('architecture')}, want {ARCH}")
    if (db.get("architecture", "w64"), db["options"].get("architecture", "w64")) != (base, base):
        fail.append(f"reference architecture {db.get('architecture')} / {db['options'].get('architecture')}, want {base}")
    for path, x, y in leaves(da, db):
        if path in ARCH_PATHS or path == "/created_utc" or (path in START_FIELDS and not same_start):
            continue
        if isinstance(x, str) and isinstance(y, str) and x.replace(a.name, b.name) == y:
            continue
        fail.append(f"decay {path}: {str(x)[:100]} vs {str(y)[:100]}")
    return fail, (f"{len(pa['rows']):,} users, {da.get('training_reviews_after_trims', da.get('training_reviews')):,} reviews, "
                  f"update offset {pa['full_run'].get('update_offset')} vs {pb['full_run'].get('update_offset')}, start "
                  f"{str(da.get('start_master_sha256'))[:12]} vs {str(db.get('start_master_sha256'))[:12]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=("decay",))
    p.add_argument("--arm", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--same-start", action="store_true")
    p.add_argument("--ref-arch", default=BASE, help="the reference's architecture (w64; w64-c1 for the recipe proof)")
    args = p.parse_args()
    a, b = Path(args.arm), Path(args.ref)
    fail, info = decay(a, b, args.same_start, args.ref_arch)
    if fail:
        print(f"[sm101-pair] decay REFUSE: " + "; ".join(fail[:12]))
        return 1
    print(f"[sm101-pair] decay OK: {a.name} == {b.name} except the architecture (w64-c1 vs {args.ref_arch}), the start weights and paths ({info})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
