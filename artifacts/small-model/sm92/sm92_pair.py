"""Show-time retrain of the release (SM92): config proofs. The new epoch 26 and the new cool-down must equal the
release's own recipe (the leak-free small model (SM72s) epochs 21-25 and its one-epoch 32k Muon cool-down from epoch
25, SM63) except the show-time inputs flag, the start weights, the epoch index and paths. Prints one line; exit 1 on
any other difference.

  epoch  --arm <epoch-26 folder> --ref <SM72s run epoch-25 folder>
         protocol.json equal to epoch 25's except: rows = the run's own order rule for epoch 26
         (random.Random(split_seed * 1000 + 26) over run.json train_users, review counts, epoch 25's label-window trims;
         recomputed here), full_run epoch 26 (was 25), init_master / init_optimizer = SM72s epoch 25's master.pth /
         optimizer.pth (checked by path), update_offset = epoch 25's update_offset_end, control_dir / benchmark_db
         (paths); decay.json options = SM72s run.json options plus exactly {"show_time_intervals": true}.
  decay  --arm <cool-down folder> --ref <the release's cool-down folder>
         protocol.json equal except full_run init_master, update_offset (start weights; the offset only enters the
         5-update LR warm-up, both are > 500k so the factor is 1) and control_dir (folder name); decay.json equal
         except created_utc, options/show_time_intervals (true in the arm, absent in the release), the start fields
         (source_case, source_update_offset_end, start, start_master_sha256, sm77, architecture = "w64") and folder
         names; same users, order, seeds, LR schedule (7e-4 -> 0 one_minus_sin over the epoch), optimizer (fresh Muon),
         KD replay targets, options.
  --same-start  (decay) the start fields must be equal too (a cool-down prepared from the release's own start); the
                descriptive fields sm77_warm.py cooldown adds (architecture, sm77, start, start_master_sha256) are
                absent in the release and not compared
"""
import argparse
import json
from pathlib import Path
import random

FLAG = "show_time_intervals"
START_FIELDS = {"/source_case", "/source_update_offset_end", "/start", "/start_master_sha256", "/sm77", "/architecture",
                "/created_utc"}
WARMUP = 5
ADDED = {"/architecture", "/sm77", "/start", "/start_master_sha256"}   # written by sm77_warm.py cooldown only (absent in the release)


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


def checkout(p):
    return next(q for q in p.resolve().parents if (q / "scripts").is_dir() and (q / "artifacts").is_dir())


def epoch(a, b):
    pa, pb = proto(a), proto(b)
    run = b.parent
    config = json.loads((run / "run.json").read_text())
    result = json.loads((b / config["model"] / config["track"] / "result.json").read_text())
    fail = []
    n = pb["full_run"]["epoch"] + 1
    if pa["full_run"].get("epoch") != n:
        fail.append(f"epoch {pa['full_run'].get('epoch')} is not {n}")
    if pa["full_run"].get("update_offset") != result["update_offset_end"]:
        fail.append(f"update_offset {pa['full_run'].get('update_offset')} vs epoch {n - 1} end {result['update_offset_end']}")
    order = config["train_users"][:]
    random.Random(config["split_seed"] * 1000 + n).shuffle(order)
    trims = {t["user_id"]: t["to"] for t in pb.get("label_window_trims", [])}
    want = [{"user_id": u, "train_range": [0, trims.get(u, config["review_counts"][str(u)])]} for u in order]
    got = [{"user_id": r["user_id"], "train_range": r["train_range"]} for r in pa["rows"]]
    if got != want:
        fail.append(f"rows are not the run's epoch-{n} order rule ({len(got)} vs {len(want)} rows)")
    if any(set(r) - {"user_id", "train_range"} != set(q) - {"user_id", "train_range"} for r, q in zip(pa["rows"], pb["rows"])):
        fail.append("row fields differ from epoch 25's")
    for path, x, y in leaves(pa, pb):
        if path.startswith("/rows") or path in ("/full_run/init_master", "/full_run/init_optimizer", "/full_run/control_dir",
                                                "/full_run/benchmark_db", "/full_run/epoch", "/full_run/update_offset"):
            continue
        fail.append(f"protocol {path}: {str(x)[:100]} vs {str(y)[:100]}")
    case = (b / config["model"] / config["track"]).resolve()
    root = checkout(b)
    for key, want_path in (("init_master", case / "master.pth"), ("init_optimizer", case / "optimizer.pth"),
                           ("benchmark_db", root / pb["full_run"]["benchmark_db"])):
        if Path(pa["full_run"][key]).resolve() != Path(want_path).resolve():
            fail.append(f"{key} {pa['full_run'][key]} vs {want_path}")
    oa = dict(json.loads((a / "decay.json").read_text())["options"])
    ro = dict(config["options"])
    if oa.pop(FLAG, None) is not True or FLAG in ro:
        fail.append(f"{FLAG} is not true in the arm only")
    for path, x, y in leaves(oa, ro):
        fail.append(f"options {path}: {str(x)[:100]} vs {str(y)[:100]}")
    return fail, (f"epoch {n}, {len(pa['rows']):,} users in the run's epoch-{n} order, update offset "
                  f"{pa['full_run'].get('update_offset')}, start {case.parents[1].name}")


def decay(a, b, same_start):
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
    if da["options"].get(FLAG) is not True or FLAG in db["options"]:
        fail.append(f"{FLAG} is not true in the arm only")
    if da.get("architecture", "w64") != "w64" or da["options"].get("architecture") != "w64":
        fail.append(f"architecture {da.get('architecture')} / {da['options'].get('architecture')}")
    for path, x, y in leaves(da, db):
        if path == f"/options/{FLAG}" or path == "/created_utc" or (path in START_FIELDS and not same_start):
            continue
        if path in ADDED and y is None:
            continue
        if isinstance(x, str) and isinstance(y, str) and x.replace(a.name, b.name) == y:
            continue
        fail.append(f"decay {path}: {str(x)[:100]} vs {str(y)[:100]}")
    return fail, (f"{len(pa['rows']):,} users, {da.get('training_reviews_after_trims', da.get('training_reviews')):,} reviews, "
                  f"update offset {pa['full_run'].get('update_offset')} vs {pb['full_run'].get('update_offset')}, start "
                  f"{str(da.get('start_master_sha256'))[:12]}")


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
        print(f"[sm92-pair] {args.kind} REFUSE: " + "; ".join(fail[:12]))
        return 1
    print(f"[sm92-pair] {args.kind} OK: {a.name} == {b.name} except the show-time inputs flag, the start weights, the "
          f"epoch index and paths ({info})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
