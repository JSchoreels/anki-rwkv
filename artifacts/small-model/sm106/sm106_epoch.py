#!/usr/bin/env python
"""The Tiny two-head model (SM106): constant-LR two-head epochs 27..29 on show-time inputs with the card stream cut to
1 layer (expertium.variants "w64-c1"). CPU only; each folder it writes is trained by scripts/expertium_decay.py run (the
launcher).

Epoch N (27..29) = exactly the recipe of the show-time retrain (SM92)'s epoch 26 (the leak-free small model (SM72s) run's
order rule random.Random(split_seed * 1000 + N) over its 8,000 train users, epoch 25's label-window trims, AdamW 7e-4
constant, options = SM92 epoch 26's: both heads (no curve_loss_weight key = the trainer's default), leak-free replay KD
from the big model (1 leak-free epoch), "show_time_intervals": true) with the epoch index advanced and ONE change, the
architecture: options/architecture and decay.json architecture = "w64-c1".
Epoch 27 starts from the cut checkpoint artifacts/small-model/sm106-cut-e26 (scripts/sm73s2.py cut of SM92's epoch 26: the
card stream's second layer removed, every other tensor and its AdamW state copied bit for bit) at SM92 epoch 26's update
offset; epoch N > 27 from epoch N - 1 of this run (weights + AdamW state + update offset, options unchanged).
This is the Tiny Instant-Only model (SM101)'s sm101_epoch.py with the chain moved (SM92 epoch 26 instead of the Small
Instant-Only model (SM98)'s epoch 35) and without the curve_loss_weight 0 override.

  prepare --epoch N   write sm106-c1-e{N}-st/{protocol.json, decay.json} (refuses if it exists or N - 1 is not complete)
  pair --epoch N      epoch N == epoch N - 1 except the order rule's epoch index (rows), the start, paths, notes, and for
                      N = 27 the architecture (w64 -> w64-c1) and the start = the cut of epoch 26 (sha256 pinned to its
                      extend.json, whose source is SM92 epoch 26's pinned master.pth / optimizer.pth). Exit 1 otherwise.
  selfcheck           the same generic pair rule accepts the Small Instant-Only model (SM98)'s real epochs 35 vs 34 and
                      34 vs 33 (read only, wt/sm98).
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
ROOT = HERE.parents[2]                                       # the sm106 worktree
sys.path.insert(0, str(ROOT))
SM = ROOT / "artifacts/small-model"
TRACK = "E89-F65e-F77a"
W = Path("C:/Users/jonat/workspace/wt")
S98 = W / "sm98/artifacts/small-model"                       # read only (selfcheck)
SM92E26 = W / "sm92/artifacts/small-model/sm92-e26"          # read only
RUN72S = W / "sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run"
E26_MASTER_SHA = "5a629e4e932ac6fddf1ada714fc33d9881aac8b8b1490ad6949193b5785e79cc"     # SM92 epoch 26 master.pth
E26_OPTIMIZER_SHA = "db5ebc37396bdab2fcfaf6e87ecf320fd1089c719bcd2377678097a3939d78b6"
E26_OFFSET_END = 574028
CUT = SM / "sm106-cut-e26"
ARCH, BASE_ARCH = "w64-c1", "w64"
FIRST, LAST = 27, 29
SWITCH = {"show_time_intervals": True}
ARCH_PATHS = {"/options/architecture", "/architecture"}


def folder(n):
    if n <= 25:
        return RUN72S / f"epoch-{n}"
    if n == 26:
        return SM92E26
    return SM / f"sm106-c1-e{n}-st"


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


def cut_record(cut=None):
    """The cut checkpoint (scripts/sm73s2.py cut of SM92 epoch 26) and its proof of origin; raises if it is not that."""
    cut = Path(cut or CUT)
    rec = json.loads((cut / "extend.json").read_text())
    bad = []
    if rec.get("init") != "cut" or rec.get("architecture") != ARCH or rec.get("removed_prefixes") != ["rwkv_modules.0.blocks.1.*"]:
        bad.append("not a w64-c1 cut of the card stream's second layer")
    if rec.get("source_master_sha256") != E26_MASTER_SHA or rec.get("source_optimizer_sha256") != E26_OPTIMIZER_SHA:
        bad.append("its source is not SM92 epoch 26's pinned files")
    if rec.get("master_sha256") != sha256(cut / "master.pth") or rec.get("optimizer_sha256") != sha256(cut / "optimizer.pth"):
        bad.append("its files do not match extend.json")
    if not rec.get("optimizer_states_equal_after_load"):
        bad.append("its AdamW state did not reload bitwise")
    if bad:
        raise SystemExit(f"{cut.name}: " + "; ".join(bad))
    return rec


def build(n, config=None, prev=None, out=None, cut=None):
    """SM92 epoch 26's epoch build with N = n, the start = epoch n - 1 (n > 27) or the cut of SM92 epoch 26 (n = 27:
    architecture -> w64-c1). prev / out / cut override the folders (sm106_check.py's simulated chain only)."""
    from scripts.run_full_training import rows_for
    config = config or run_config()
    prev, out = Path(prev or folder(n - 1)), Path(out or folder(n))
    if not complete(prev):
        raise SystemExit(f"{prev.name} is not a complete epoch")
    case = prev / "expertium" / TRACK
    if n == FIRST:
        if sha256(case / "master.pth") != E26_MASTER_SHA or sha256(case / "optimizer.pth") != E26_OPTIMIZER_SHA:
            raise SystemExit(f"{prev.name} is not SM92's epoch 26 (sha256 pin)")
        cut_record(cut)
        start = Path(cut or CUT)
    else:
        start = case
    previous = proto(prev)
    protocol = copy.deepcopy(previous)
    rows = rows_for(order(n, config), config["review_counts"])
    trims = {t["user_id"]: t for t in previous.get("label_window_trims", [])}
    for row in rows:
        t = trims.get(row["user_id"])
        if t:
            row["train_range"] = [0, t["to"]]
    protocol["rows"] = rows
    full = protocol["full_run"]
    for key in ("lr_resume", "resumed", "optimizer_state"):
        full.pop(key, None)
    full.update(epoch=n, init_master=(start / "master.pth").as_posix(), init_optimizer=(start / "optimizer.pth").as_posix(),
                update_offset=result_of(prev)["update_offset_end"], control_dir=(out / "control").relative_to(ROOT).as_posix())
    decay = copy.deepcopy(decay_of(prev))
    decay.pop("created_utc", None)
    if n == FIRST:
        decay["options"]["architecture"] = ARCH
        decay["architecture"] = ARCH
    decay.update(source_epoch=n - 1, epoch=n, training_users=len(rows),
                 training_reviews=sum(r["train_range"][1] - r["train_range"][0] for r in rows), start=prev.name,
                 note=f"SM106: SM92 epoch 26's recipe (SM72s's epoch protocol + show-time inputs, both heads), epoch {n} "
                      f"(the run's order rule), architecture {ARCH} (card stream 1 layer), start = "
                      f"{start.name if n == FIRST else prev.name}"
                      f"{' (SM92 epoch 26 cut: rwkv_modules.0.blocks.1.* removed, AdamW state re-indexed)' if n == FIRST else ''}, "
                      f"extra options {json.dumps(SWITCH, sort_keys=True)} (show-time inputs)",
                 init_master_sha256=sha256(start / "master.pth"), init_optimizer_sha256=sha256(start / "optimizer.pth"))
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
    print(json.dumps({k: decay[k] for k in ("epoch", "architecture", "training_users", "training_reviews", "start", "note")}))
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


def start_of(n, prev, cut=None, first=FIRST):
    """The files epoch n must start from: the cut (n = 27) or epoch n - 1's trained files."""
    return Path(cut or CUT) if n == first else Path(prev) / "expertium" / TRACK


def pair_fail(n, d=None, prev=None, prevprev=None, sm106=True, cut=None):
    """Epoch n (folder d) vs epoch n - 1 (prev): every leaf equal except
    protocol: rows (= the run's epoch-n order rule with epoch 25's trims), full_run epoch / init_master / init_optimizer /
              update_offset / control_dir (= the start's files and prev's end offset);
    decay.json: source_epoch, epoch, start, note, created_utc, start hashes; for n = 27 also the architecture (w64 in
              SM92's epoch 26, w64-c1 here, both fields). No other option may differ.
    Plus: prev started from prevprev the same way (for n = 28: epoch 27 started from the cut at epoch 26's end offset);
    sm106=True also requires the options to be SM92 epoch 26's (both heads: no curve_loss_weight key, show-time inputs)
    with the architecture w64-c1. sm106=False (selfcheck on another chain) uses no cut and no pins."""
    config = run_config()
    d, prev, prevprev = Path(d or folder(n)), Path(prev or folder(n - 1)), Path(prevprev or folder(n - 2))
    first = FIRST if sm106 else -1
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
        if sm106 and n == FIRST and path in ARCH_PATHS and (x, y) == (ARCH, BASE_ARCH):
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
    full, start = p["full_run"], start_of(n, prev, cut, first)
    rp = result_of(prev) or {}
    if full.get("epoch") != n or dd.get("epoch") != n or dd.get("source_epoch") != n - 1:
        fail.append(f"epoch {full.get('epoch')} / {dd.get('epoch')} / source {dd.get('source_epoch')}")
    if not (same_path(full.get("init_master"), start / "master.pth") and same_path(full.get("init_optimizer"), start / "optimizer.pth")):
        fail.append(f"epoch {n} does not start from {start}")
    if not complete(prev) or full.get("update_offset") != rp.get("update_offset_end"):
        fail.append(f"update_offset {full.get('update_offset')} vs epoch {n - 1} end {rp.get('update_offset_end')}")
    m_sha, o_sha = sha256(start / "master.pth"), sha256(start / "optimizer.pth")
    if dd.get("init_master_sha256") != m_sha or dd.get("init_optimizer_sha256") != o_sha:
        fail.append(f"the recorded start hashes are not {start}'s files")
    if full.get("control_dir") != (d / "control").relative_to(checkout(d)).as_posix():
        fail.append(f"control_dir {full.get('control_dir')}")
    if dd.get("start") != prev.name:
        fail.append(f"start {dd.get('start')} is not {prev.name}")
    # prev started from prevprev by the same rule (so "the same step" is the step prev took)
    fp, cpp = pp["full_run"], start_of(n - 1, prevprev, cut, first)
    if fp.get("epoch") != n - 1 or not same_path(fp.get("init_master"), cpp / "master.pth") or \
            fp.get("update_offset") != (result_of(prevprev) or {}).get("update_offset_end"):
        fail.append(f"epoch {n - 1} does not start from {cpp} at epoch {n - 2}'s end (reference broken)")
    if any(dd.get(k) != dp.get(k) for k in ("training_users", "training_reviews", "lr", "kind", "model", "track")):
        fail.append(f"users / reviews / lr / kind differ from epoch {n - 1}")
    if sm106:
        if n == FIRST:
            try:
                rec = cut_record(cut)
                if m_sha != rec["master_sha256"] or o_sha != rec["optimizer_sha256"]:
                    fail.append("epoch 27 does not start from the pinned cut")
            except SystemExit as error:
                fail.append(str(error))
            if sha256(prev / "expertium" / TRACK / "master.pth") != E26_MASTER_SHA or rp.get("update_offset_end") != E26_OFFSET_END:
                fail.append("epoch 27's previous epoch is not SM92 epoch 26 (pinned sha256 / offset)")
        o92 = decay_of(SM92E26)["options"]
        if dd["options"] != {**o92, "architecture": ARCH} or o92.get("architecture") != BASE_ARCH or \
                o92.get("show_time_intervals") is not True or "curve_loss_weight" in o92 or \
                "curve_loss_weight" in dd["options"] or dd.get("architecture") != ARCH:
            fail.append("options are not SM92 epoch 26's (both heads, show-time inputs) with architecture w64-c1")
    info = (f"{len(p['rows']):,} users, {dd['training_reviews']:,} reviews, epoch {n}, arch {dd.get('architecture')}, "
            f"start {start.parent.parent.name if n != first else start.name} (sha256 {dd['init_master_sha256'][:12]}), "
            f"offset {full.get('update_offset')}, differences: protocol {sorted(set(diffs['protocol']))} + rows order; "
            f"decay {sorted(set(diffs['decay']))}")
    return fail, info


def pair(args):
    fail, info = pair_fail(args.epoch)
    d = folder(args.epoch)
    if fail:
        print(f"[sm106-epoch] pair REFUSE epoch {args.epoch}: " + "; ".join(fail[:12]))
        return 1
    print(f"[sm106-epoch] pair OK: {d.name} == {folder(args.epoch - 1).name}'s step (SM92 epoch 26's recipe) except the "
          f"epoch index / start / paths{' / architecture w64-c1' if args.epoch == FIRST else ''} ({info})")
    return 0


def selfcheck(_args):
    bad = 0
    f98 = lambda k: S98 / f"sm98-inst-e{k}-st"   # noqa: E731
    for n in (35, 34):
        fail, info = pair_fail(n, d=f98(n), prev=f98(n - 1), prevprev=f98(n - 2), sm106=False)
        print(f"[sm106-epoch] selfcheck SM98 epoch {n} vs {n - 1}: {'OK' if not fail else 'REFUSE ' + '; '.join(fail[:6])} ({info})")
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
