#!/usr/bin/env python
"""The show-time instant-only small model (SM96): epoch 31, one constant-LR instant-only epoch on show-time inputs before
the cool-down (user 2026-11-09 00:20: the way the show-time retrain of the release (SM92) trained its epoch 26 on
show-time inputs before its own cool-down). CPU only; the folder it writes is trained by scripts/expertium_decay.py run.

Epoch 31 = the instant-only small model (SM57) stage 4's epoch recipe (sm57_stage4.py build, epochs 27..30: the leak-free
small model (SM72s) run's order rule random.Random(split_seed * 1000 + N) over its 8,000 train users, epoch 25's
label-window trims, AdamW 7e-4 constant, started from the previous epoch's weights + AdamW state + update offset,
options = stage 1's instant-only epoch 26 options = SM72s run.json + {"curve_loss_weight": 0.0}) with N = 31, started from
stage 4's epoch 30 (wt/sm57/artifacts/small-model/sm57s4-inst-e30, read only), plus exactly {"show_time_intervals": true}
in the options (the show-time retrain of the release (SM92)'s switch: its epoch 26 options = SM72s run.json + that flag).
Folder artifacts/small-model/sm96-inst-e31-st (in the sm96 worktree).

  prepare          write sm96-inst-e31-st/{protocol.json, decay.json} (refuses if it exists)
  pair             epoch 31 == stage 4's epoch 30 except the order rule's epoch index (rows), the start (epoch 30's
                   files, hashes, update offset), paths, notes and the input switch; options == SM92's epoch 26
                   options + {"curve_loss_weight": 0.0}. Exit 1 otherwise.
"""
from argparse import ArgumentParser
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                                       # the sm96 worktree
sys.path.insert(0, str(ROOT))
SM = ROOT / "artifacts/small-model"
TRACK = "E89-F65e-F77a"
S57 = Path("C:/Users/jonat/workspace/wt/sm57/artifacts/small-model")   # read only
E26 = S57 / "sm57-inst-e26"                                  # stage 1's instant-only epoch 26 (the stage 4 options base)
E29 = S57 / "sm57s4-inst-e29"
E30 = S57 / "sm57s4-inst-e30"                                # stage 4's last constant-LR epoch: the start
SM92E26 = Path("C:/Users/jonat/workspace/wt/sm92/artifacts/small-model/sm92-e26")   # read only
RUN72S = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run")
N = 31
E31 = SM / "sm96-inst-e31-st"
FLAG = {"show_time_intervals": True}                          # the input switch (on top of stage 4's curve_loss_weight 0)
STAGE4_FLAG = {"curve_loss_weight": 0.0}


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_config():
    return json.loads((RUN72S / "run.json").read_text())


def order(n, config):
    users = config["train_users"][:]
    random.Random(config["split_seed"] * 1000 + n).shuffle(users)
    return users


def result_of(d):
    p = Path(d) / "expertium" / TRACK / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def complete(d):
    r = result_of(d)
    return bool(r) and r.get("status") == "complete" and not r.get("smoke_only") and (Path(d) / "expertium" / TRACK / "master.pth").exists()


def proto(d):
    q = Path(d) / "protocol.prepared.json"   # a resumed run rewrites protocol.json to the users left
    return json.loads((q if q.exists() else Path(d) / "protocol.json").read_text())


def build(config=None):
    """sm57_stage4.build(31, prev=E30) with ROOT = the sm96 worktree and FLAG added to the options."""
    from scripts.run_full_training import rows_for
    config = config or run_config()
    if not complete(E30):
        raise SystemExit(f"{E30.name} is not a complete epoch")
    previous = proto(E30)
    protocol = copy.deepcopy(previous)
    rows = rows_for(order(N, config), config["review_counts"])
    trims = {t["user_id"]: t for t in previous.get("label_window_trims", [])}
    for row in rows:
        t = trims.get(row["user_id"])
        if t:
            row["train_range"] = [0, t["to"]]
    protocol["rows"] = rows
    case = E30 / "expertium" / TRACK
    full = protocol["full_run"]
    for key in ("lr_resume", "resumed", "optimizer_state"):
        full.pop(key, None)
    full.update(epoch=N, init_master=(case / "master.pth").as_posix(), init_optimizer=(case / "optimizer.pth").as_posix(),
                update_offset=result_of(E30)["update_offset_end"], control_dir=(E31 / "control").relative_to(ROOT).as_posix())
    base = json.loads((E26 / "decay.json").read_text())
    options = {**base["options"], **FLAG}
    decay = {"kind": "sm77 constant-LR epoch", "source_run": base["source_run"], "source_epoch": N - 1, "epoch": N,
             "model": base["model"], "track": base["track"], "options": options, "architecture": "w64",
             "training_users": len(rows), "training_reviews": sum(r["train_range"][1] - r["train_range"][0] for r in rows),
             "lr": base["lr"], "start": E30.name,
             "note": f"SM96: SM57 stage 4's epoch recipe, epoch {N} (the run's order rule), start = {E30.name}, extra options "
                     f"{json.dumps({**STAGE4_FLAG, **FLAG}, sort_keys=True)} (show-time inputs)",
             "init_master_sha256": sha256(case / "master.pth"), "init_optimizer_sha256": sha256(case / "optimizer.pth")}
    return protocol, decay


def prepare(_args):
    if E31.exists():
        raise SystemExit(f"{E31} exists")
    protocol, decay = build()
    decay["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    E31.mkdir(parents=True)
    atomic_json(E31 / "protocol.json", protocol)
    atomic_json(E31 / "decay.json", decay)
    print(json.dumps({k: decay[k] for k in ("epoch", "training_users", "training_reviews", "start", "note")}))
    return 0


def leaves(x, y, path=""):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            yield from leaves(x.get(k), y.get(k), f"{path}/{k}")
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (u, v) in enumerate(zip(x, y)):
            yield from leaves(u, v, f"{path}[{i}]")
    elif x != y:
        yield path, x, y


PROTO_SKIP = {"/full_run/epoch", "/full_run/init_master", "/full_run/init_optimizer", "/full_run/update_offset",
              "/full_run/control_dir"}
DECAY_SKIP = {"/source_epoch", "/epoch", "/start", "/note", "/init_master_sha256", "/init_optimizer_sha256", "/created_utc"}


def pair_fail(d=E31):
    """Epoch 31 vs stage 4's epoch 30 (both built by the same rule): every leaf equal except
    protocol: rows (= the run's epoch-31 order rule with epoch 25's trims; the order differs from epochs 21..30),
              full_run epoch / init_master / init_optimizer / update_offset / control_dir (= epoch 30's files and end);
    decay.json: source_epoch, epoch, start, note, created_utc, start hashes (= epoch 30's files),
              options.show_time_intervals (true in epoch 31 only).
    Plus: epoch 30 itself started the same way from epoch 29; epoch 31's options == SM92 epoch 26's + curve_loss_weight 0."""
    config = run_config()
    d = Path(d)
    p, p30 = proto(d), proto(E30)
    dd, d30 = (json.loads((x / "decay.json").read_text()) for x in (d, E30))
    fail, diffs = [], {"protocol": [], "decay": []}
    for path, x, y in leaves(p, p30):
        if path.startswith("/rows"):
            continue
        if path in PROTO_SKIP:
            diffs["protocol"].append(path)
            continue
        fail.append(f"protocol {path}: {str(x)[:100]} vs {str(y)[:100]}")
    for path, x, y in leaves(dd, d30):
        if path in DECAY_SKIP or (path == "/options/show_time_intervals" and x is True and y is None):
            diffs["decay"].append(path)
            continue
        fail.append(f"decay {path}: {str(x)[:100]} vs {str(y)[:100]}")
    trims = {t["user_id"]: t["to"] for t in p30.get("label_window_trims", [])}
    want = [{"user_id": u, "train_range": [0, trims.get(u, config["review_counts"][str(u)])],
             "available_reviews": config["review_counts"][str(u)],
             "validation_range": [0, config["review_counts"][str(u)]]} for u in order(N, config)]
    if p["rows"] != want:
        fail.append(f"rows are not the run's epoch-{N} order rule with epoch 25's trims")
    if {tuple(r["user_id"] for r in p["rows"])} & {tuple(order(k, config)) for k in range(21, N)}:
        fail.append("the epoch order repeats an earlier epoch")
    if sorted(r["user_id"] for r in p["rows"]) != sorted(r["user_id"] for r in p30["rows"]):
        fail.append("the user set differs from epoch 30's")
    full, case = p["full_run"], E30 / "expertium" / TRACK
    r30 = result_of(E30) or {}
    if full.get("epoch") != N or dd.get("epoch") != N or dd.get("source_epoch") != N - 1:
        fail.append(f"epoch {full.get('epoch')} / {dd.get('epoch')} / source {dd.get('source_epoch')}")
    if full.get("init_master") != (case / "master.pth").as_posix() or full.get("init_optimizer") != (case / "optimizer.pth").as_posix():
        fail.append("epoch 31 does not start from stage 4's epoch 30 files")
    if r30.get("status") != "complete" or full.get("update_offset") != r30.get("update_offset_end"):
        fail.append(f"update_offset {full.get('update_offset')} vs epoch 30 end {r30.get('update_offset_end')}")
    if dd.get("init_master_sha256") != sha256(case / "master.pth") or dd.get("init_optimizer_sha256") != sha256(case / "optimizer.pth"):
        fail.append("the recorded start hashes are not epoch 30's files")
    if full.get("control_dir") != (d / "control").relative_to(ROOT).as_posix():
        fail.append(f"control_dir {full.get('control_dir')}")
    # epoch 30 started from epoch 29 by the same rule (so "the same step" is the stage 4 step)
    f30 = p30["full_run"]
    c29 = E29 / "expertium" / TRACK
    if f30.get("init_master") != (c29 / "master.pth").as_posix() or f30.get("update_offset") != (result_of(E29) or {}).get("update_offset_end"):
        fail.append("stage 4's epoch 30 does not start from its epoch 29 (reference broken)")
    # the input switch applied as in the show-time retrain of the release (SM92)'s epoch 26
    o92 = json.loads((SM92E26 / "decay.json").read_text())["options"]
    if dd["options"] != {**o92, **STAGE4_FLAG} or "curve_loss_weight" in o92 or o92.get("show_time_intervals") is not True:
        fail.append("options are not SM92 epoch 26's + curve_loss_weight 0")
    if any(dd.get(k) != d30.get(k) for k in ("training_users", "training_reviews", "lr", "kind", "model", "track", "architecture")):
        fail.append("users / reviews / lr / kind differ from epoch 30")
    info = (f"{len(p['rows']):,} users, {dd['training_reviews']:,} reviews, epoch {N}, start {E30.name} "
            f"(sha256 {dd['init_master_sha256'][:12]}), offset {full.get('update_offset')}, "
            f"differences: protocol {sorted(set(diffs['protocol']))} + rows order; decay {sorted(set(diffs['decay']))}")
    return fail, info


def pair(_args):
    fail, info = pair_fail()
    if fail:
        print(f"[sm96-epoch] pair REFUSE: " + "; ".join(fail[:12]))
        return 1
    print(f"[sm96-epoch] pair OK: {E31.name} == stage 4's {E30.name} step except the input switch {json.dumps(FLAG)} and "
          f"the epoch index / start / paths ({info})")
    return 0


def main():
    p = ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("pair")
    a = p.parse_args()
    return {"prepare": prepare, "pair": pair}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
