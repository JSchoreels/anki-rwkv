#!/usr/bin/env python
"""The show-time instant-only model, stage 2 (SM98): constant-LR instant-only epochs 32..35 on show-time inputs before the
cool-down. CPU only; each folder it writes is trained by scripts/expertium_decay.py run (the launcher).

Epoch N (32..35) = exactly the recipe of the show-time instant-only small model (SM96)'s epoch 31 (sm96_epoch.py build:
the instant-only small model (SM57) stage 4's epoch recipe = the leak-free small model (SM72s) run's order rule
random.Random(split_seed * 1000 + N) over its 8,000 train users, epoch 25's label-window trims, AdamW 7e-4 constant,
from the previous epoch's weights + AdamW state + update offset, options = stage 4's + {"show_time_intervals": true})
with the epoch index advanced: epoch 32 starts from SM96's epoch 31 (wt/sm96/artifacts/small-model/sm96-inst-e31-st,
read only, master sha256 pinned), epoch N > 32 from epoch N - 1 of this run. Options are copied from the previous epoch
unchanged (so every epoch has epoch 31's options). Folders artifacts/small-model/sm98-inst-e{N}-st (sm98 worktree).

  prepare --epoch N   write sm98-inst-e{N}-st/{protocol.json, decay.json} (refuses if it exists or N - 1 is not complete)
  pair --epoch N      epoch N == epoch N - 1 except the order rule's epoch index (rows), the start (N - 1's files, hashes,
                      update offset), paths, notes; N - 1 itself started from N - 2 by the same rule; options == SM96
                      epoch 31's == the show-time retrain of the release (SM92) epoch 26's + {"curve_loss_weight": 0.0}.
                      Exit 1 otherwise.
  selfcheck           the same generic pair rule accepts the instant-only small model (SM57) stage 4's real consecutive
                      constant-LR epochs (30 vs 29, 29 vs 28): the rule is the step stage 4 actually took.
"""
from argparse import ArgumentParser
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                                       # the sm98 worktree
sys.path.insert(0, str(ROOT))
SM = ROOT / "artifacts/small-model"
TRACK = "E89-F65e-F77a"
W = Path("C:/Users/jonat/workspace/wt")
S57 = W / "sm57/artifacts/small-model"                       # read only
S96 = W / "sm96/artifacts/small-model"                       # read only
SM92E26 = W / "sm92/artifacts/small-model/sm92-e26"          # read only
RUN72S = W / "sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run"
E31 = S96 / "sm96-inst-e31-st"
E31_MASTER_SHA = "f49c36fa5bff146d062ded4cca2b474665906a9444993b268a90d9caae1e86c7"     # SM96 epoch 31 master.pth
E31_OPTIMIZER_SHA = "845d2cb2aafd7161872a01e8ece6d6aa220c188f59cb250eeee0c5d58411ea3d"
E31_OFFSET_END = 684418
FIRST, LAST = 32, 35
STAGE4_FLAG = {"curve_loss_weight": 0.0}
SWITCH = {"show_time_intervals": True}


def folder(n):
    if n <= 30:
        return S57 / f"sm57s4-inst-e{n}"
    if n == 31:
        return E31
    return SM / f"sm98-inst-e{n}-st"


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def decay_of(d):
    return json.loads((Path(d) / "decay.json").read_text())


def checkout(p):
    return next(q for q in Path(p).resolve().parents if (q / "scripts").is_dir() and (q / "artifacts").is_dir())


def build(n, config=None, prev=None, out=None):
    """sm96_epoch.build with N = n and the start = epoch n - 1 (options copied from it unchanged). prev / out override the
    folders (sm98_check.py's simulated chain only)."""
    from scripts.run_full_training import rows_for
    config = config or run_config()
    prev, out = Path(prev or folder(n - 1)), Path(out or folder(n))
    if not complete(prev):
        raise SystemExit(f"{prev.name} is not a complete epoch")
    if n == FIRST and sha256(prev / "expertium" / TRACK / "master.pth") != E31_MASTER_SHA:
        raise SystemExit(f"{prev.name} master.pth is not SM96's epoch 31 (sha256 pin)")
    previous = proto(prev)
    protocol = copy.deepcopy(previous)
    rows = rows_for(order(n, config), config["review_counts"])
    trims = {t["user_id"]: t for t in previous.get("label_window_trims", [])}
    for row in rows:
        t = trims.get(row["user_id"])
        if t:
            row["train_range"] = [0, t["to"]]
    protocol["rows"] = rows
    case = prev / "expertium" / TRACK
    full = protocol["full_run"]
    for key in ("lr_resume", "resumed", "optimizer_state"):
        full.pop(key, None)
    full.update(epoch=n, init_master=(case / "master.pth").as_posix(), init_optimizer=(case / "optimizer.pth").as_posix(),
                update_offset=result_of(prev)["update_offset_end"], control_dir=(out / "control").relative_to(ROOT).as_posix())
    decay = copy.deepcopy(decay_of(prev))
    decay.pop("created_utc", None)
    decay.update(source_epoch=n - 1, epoch=n, training_users=len(rows),
                 training_reviews=sum(r["train_range"][1] - r["train_range"][0] for r in rows), start=prev.name,
                 note=f"SM98: SM96 epoch 31's recipe (SM57 stage 4's epoch recipe + show-time inputs), epoch {n} (the run's "
                      f"order rule), start = {prev.name}, extra options "
                      f"{json.dumps({**STAGE4_FLAG, **SWITCH}, sort_keys=True)} (show-time inputs)",
                 init_master_sha256=sha256(case / "master.pth"), init_optimizer_sha256=sha256(case / "optimizer.pth"))
    return protocol, decay


def prepare(args):
    n = args.epoch
    if not FIRST <= n <= LAST:
        raise SystemExit(f"epoch {n} is outside {FIRST}..{LAST}")
    out = folder(n)
    if out.exists():
        raise SystemExit(f"{out} exists")
    protocol, decay = build(n)
    decay["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    atomic_json(tmp / "protocol.json", protocol)
    atomic_json(tmp / "decay.json", decay)
    tmp.rename(out)
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


def same_path(a, b):
    return Path(str(a)).resolve() == Path(str(b)).resolve()


def pair_fail(n, d=None, prev=None, prevprev=None, sm98=True):
    """Epoch n (folder d) vs epoch n - 1 (prev), both built by the same rule: every leaf equal except
    protocol: rows (= the run's epoch-n order rule with epoch 25's trims), full_run epoch / init_master / init_optimizer /
              update_offset / control_dir (= prev's files and end);
    decay.json: source_epoch, epoch, start, note, created_utc, start hashes (= prev's files). No option may differ.
    Plus: prev started from prevprev the same way; sm98=True also pins epoch 32's start (SM96 epoch 31's sha256) and
    requires the options to be SM96 epoch 31's == SM92 epoch 26's + curve_loss_weight 0."""
    config = run_config()
    d, prev, prevprev = Path(d or folder(n)), Path(prev or folder(n - 1)), Path(prevprev or folder(n - 2))
    p, pp = proto(d), proto(prev)
    dd, dp = decay_of(d), decay_of(prev)
    fail, diffs = [], {"protocol": [], "decay": []}
    for path, x, y in leaves(p, pp):
        if path.startswith("/rows"):
            continue
        if path in PROTO_SKIP:
            diffs["protocol"].append(path)
            continue
        fail.append(f"protocol {path}: {str(x)[:100]} vs {str(y)[:100]}")
    for path, x, y in leaves(dd, dp):
        if path in DECAY_SKIP:
            diffs["decay"].append(path)
            continue
        fail.append(f"decay {path}: {str(x)[:100]} vs {str(y)[:100]}")
    trims = {t["user_id"]: t["to"] for t in pp.get("label_window_trims", [])}
    want = [{"user_id": u, "train_range": [0, trims.get(u, config["review_counts"][str(u)])],
             "available_reviews": config["review_counts"][str(u)],
             "validation_range": [0, config["review_counts"][str(u)]]} for u in order(n, config)]
    if p["rows"] != want:
        fail.append(f"rows are not the run's epoch-{n} order rule with epoch 25's trims")
    if {tuple(r["user_id"] for r in p["rows"])} & {tuple(order(k, config)) for k in range(21, n)}:
        fail.append("the epoch order repeats an earlier epoch")
    if sorted(r["user_id"] for r in p["rows"]) != sorted(r["user_id"] for r in pp["rows"]):
        fail.append(f"the user set differs from epoch {n - 1}'s")
    full, case = p["full_run"], prev / "expertium" / TRACK
    rp = result_of(prev) or {}
    if full.get("epoch") != n or dd.get("epoch") != n or dd.get("source_epoch") != n - 1:
        fail.append(f"epoch {full.get('epoch')} / {dd.get('epoch')} / source {dd.get('source_epoch')}")
    if not (same_path(full.get("init_master"), case / "master.pth") and same_path(full.get("init_optimizer"), case / "optimizer.pth")):
        fail.append(f"epoch {n} does not start from epoch {n - 1}'s files ({prev.name})")
    if not complete(prev) or full.get("update_offset") != rp.get("update_offset_end"):
        fail.append(f"update_offset {full.get('update_offset')} vs epoch {n - 1} end {rp.get('update_offset_end')}")
    m_sha, o_sha = sha256(case / "master.pth"), sha256(case / "optimizer.pth")
    if dd.get("init_master_sha256") != m_sha or dd.get("init_optimizer_sha256") != o_sha:
        fail.append(f"the recorded start hashes are not epoch {n - 1}'s files")
    if full.get("control_dir") != (d / "control").relative_to(checkout(d)).as_posix():
        fail.append(f"control_dir {full.get('control_dir')}")
    if dd.get("start") != prev.name:
        fail.append(f"start {dd.get('start')} is not {prev.name}")
    # prev started from prevprev by the same rule (so "the same step" is the step prev took)
    fp, cpp = pp["full_run"], prevprev / "expertium" / TRACK
    if fp.get("epoch") != n - 1 or not same_path(fp.get("init_master"), cpp / "master.pth") or \
            fp.get("update_offset") != (result_of(prevprev) or {}).get("update_offset_end"):
        fail.append(f"epoch {n - 1} does not start from epoch {n - 2} ({prevprev.name}) (reference broken)")
    if any(dd.get(k) != dp.get(k) for k in ("training_users", "training_reviews", "lr", "kind", "model", "track", "architecture")):
        fail.append(f"users / reviews / lr / kind differ from epoch {n - 1}")
    if sm98:
        if n == FIRST and (m_sha != E31_MASTER_SHA or o_sha != E31_OPTIMIZER_SHA or rp.get("update_offset_end") != E31_OFFSET_END):
            fail.append("epoch 32 does not start from SM96 epoch 31's pinned files / offset")
        o92 = decay_of(SM92E26)["options"]
        if dd["options"] != decay_of(E31)["options"] or dd["options"] != {**o92, **STAGE4_FLAG} or \
                o92.get("show_time_intervals") is not True or "curve_loss_weight" in o92:
            fail.append("options are not SM96 epoch 31's (== SM92 epoch 26's + curve_loss_weight 0)")
    info = (f"{len(p['rows']):,} users, {dd['training_reviews']:,} reviews, epoch {n}, start {prev.name} "
            f"(sha256 {dd['init_master_sha256'][:12]}), offset {full.get('update_offset')}, "
            f"differences: protocol {sorted(set(diffs['protocol']))} + rows order; decay {sorted(set(diffs['decay']))}")
    return fail, info


def pair(args):
    fail, info = pair_fail(args.epoch)
    d = folder(args.epoch)
    if fail:
        print(f"[sm98-epoch] pair REFUSE epoch {args.epoch}: " + "; ".join(fail[:12]))
        return 1
    print(f"[sm98-epoch] pair OK: {d.name} == {folder(args.epoch - 1).name}'s step (SM96 epoch 31's recipe) except the "
          f"epoch index / start / paths ({info})")
    return 0


def selfcheck(_args):
    bad = 0
    for n in (30, 29):
        fail, info = pair_fail(n, sm98=False)
        print(f"[sm98-epoch] selfcheck stage 4 epoch {n} vs {n - 1}: {'OK' if not fail else 'REFUSE ' + '; '.join(fail[:6])} ({info})")
        bad += bool(fail)
    return 1 if bad else 0


def main():
    p = ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "pair"):
        s = sub.add_parser(name)
        s.add_argument("--epoch", type=int, required=True)
    sub.add_parser("selfcheck")
    a = p.parse_args()
    return {"prepare": prepare, "pair": pair, "selfcheck": selfcheck}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
