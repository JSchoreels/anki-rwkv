#!/usr/bin/env python
"""Six-layer Tiny with longer recovery (SM135): up to four more constant-LR recovery epochs per chain, continuing the
six-layer Tiny (SM130)'s chains from their last recovery epoch. CPU only; each folder it writes is trained by
scripts/expertium_decay.py run (launch-sm135.sh). The code is the six-layer Tiny (SM130)'s sm130_epoch.py with the cut
removed (the start is an already trained six-layer epoch, so the architecture does not change at any step).

Chains (read only sources in wt/sm130):
  io  Instant-Only: epochs 42..45 = exactly SM130's Instant-Only epoch step (the leak-free small model (SM72s) run's order
      rule random.Random(split_seed * 1000 + N) over its 8,000 train users, epoch 25's label-window trims, AdamW 7e-4
      constant, 2048 x 16 lanes, leak-free replay KD from the big model (1 leak-free epoch) alpha 0.9, instant head only
      (curve_loss_weight 0), "show_time_intervals": true, w64-c1-d2p1u1) with the epoch index advanced. Epoch 42 starts
      from SM130's epoch 41 (sm130-io-e41-st, master sha256 af16139e28bb) at its update offset end.
  th  two-head: epochs 33..36 = SM130's two-head epoch step (both heads); epoch 33 starts from SM130's epoch 32
      (sm130-th-e32-st, 635f1a12893b).
Epoch N > FIRST starts from epoch N - 1 of this run (weights + AdamW state + update offset, options unchanged).

  prepare --chain io|th --epoch N    write sm135-{chain}-e{N}-st/{protocol.json, decay.json} (refuses if it exists or
                                     N - 1 is not complete)
  pair --chain io|th --epoch N       epoch N == epoch N - 1 except the order rule's epoch index (rows), the start, paths,
                                     notes. For N = FIRST the previous epoch must be SM130's pinned last recovery epoch.
                                     Exit 1 otherwise.
  selfcheck                          the same generic pair rule accepts SM130's real epochs 41 vs 40 and 32 vs 31
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
ROOT = HERE.parents[2]                                       # the sm135 worktree
sys.path.insert(0, str(ROOT))
SM = ROOT / "artifacts/small-model"
TRACK = "E89-F65e-F77a"
W = Path("C:/Users/jonat/workspace/wt")
RUN72S = W / "sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run"
ARCH = "w64-c1-d2p1u1"
SRC = W / "sm130/artifacts/small-model"
CHAINS = {
    "io": {"first": 42, "last": 45,
           "src_master_sha": "af16139e28bbd322f42790bfed25d5be2c62a92351a9045b14d3e6b66e7c96ff",   # SM130 io epoch 41
           "src_optimizer_sha": "c7d00f0ef2e25694b5470b960f59ee647c1efa282af211b1a9b213a3a42ed48f",
           "name": "the six-layer Tiny (SM130)'s Instant-Only epoch 41", "curve_loss_weight": 0.0},
    "th": {"first": 33, "last": 36,
           "src_master_sha": "635f1a12893b4d87c7c946e6311103de3aa519417535919fb4c5e13314ea6a46",   # SM130 th epoch 32
           "src_optimizer_sha": "33fe84765252fade25805958cd62d5abb9b53ebd81c3749f070c2d11b4b013d0",
           "name": "the six-layer Tiny (SM130)'s two-head epoch 32", "curve_loss_weight": None},
}


def folder(chain, n):
    if n < CHAINS[chain]["first"]:
        return SRC / f"sm130-{chain}-e{n}-st"
    return SM / f"sm135-{chain}-e{n}-st"


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


def source_ok(chain):
    """The previous epoch of epoch FIRST: SM130's last recovery epoch, pinned by its master and optimizer sha256."""
    c = CHAINS[chain]
    case = folder(chain, c["first"] - 1) / "expertium" / TRACK
    bad = []
    if not complete(folder(chain, c["first"] - 1)):
        bad.append("not complete")
    elif sha256(case / "master.pth") != c["src_master_sha"] or sha256(case / "optimizer.pth") != c["src_optimizer_sha"]:
        bad.append("master / optimizer sha256 are not the pinned ones")
    return bad


def build(chain, n, config=None):
    from scripts.run_full_training import rows_for
    c = CHAINS[chain]
    config = config or run_config()
    prev, out = folder(chain, n - 1), folder(chain, n)
    if not complete(prev):
        raise SystemExit(f"{prev.name} is not a complete epoch")
    if n == c["first"] and source_ok(chain):
        raise SystemExit(f"{prev.name} is not {c['name']}: " + "; ".join(source_ok(chain)))
    start = prev / "expertium" / TRACK
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
    decay.update(source_epoch=n - 1, epoch=n, training_users=len(rows),
                 training_reviews=sum(r["train_range"][1] - r["train_range"][0] for r in rows), start=prev.name,
                 note=f"SM135 ({chain}): the six-layer Tiny (SM130)'s {chain} epoch step, epoch {n} (the run's order rule), "
                      f"architecture {ARCH}, start = {prev.name}"
                      f"{' (SM130 last recovery epoch, pinned)' if n == c['first'] else ''}",
                 init_master_sha256=sha256(start / "master.pth"), init_optimizer_sha256=sha256(start / "optimizer.pth"))
    return protocol, decay


def prepare(args):
    c, n = CHAINS[args.chain], args.epoch
    if not c["first"] <= n <= c["last"]:
        raise SystemExit(f"epoch {n} is outside {c['first']}..{c['last']}")
    out = folder(args.chain, n)
    if out.exists():
        raise SystemExit(f"{out} exists")
    protocol, decay = build(args.chain, n)
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


def pair_fail(chain, n, d=None, prev=None, strict=True):
    """Epoch n (folder d) vs epoch n - 1 (prev): every leaf equal except the rows (= the run's epoch-n order rule with
    epoch 25's trims), full_run epoch / init_master / init_optimizer / update_offset / control_dir, decay source_epoch /
    epoch / start / note / created_utc / start hashes. strict: options = SM130's last recovery epoch's options
    (w64-c1-d2p1u1, show-time inputs, the chain's curve_loss_weight), and for n = FIRST the previous epoch is SM130's
    pinned last recovery epoch. strict=False (selfcheck on SM130's real epochs) skips the strict part."""
    c = CHAINS[chain]
    config = run_config()
    d = Path(d or folder(chain, n))
    prev = Path(prev or folder(chain, n - 1))
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
    full = p["full_run"]
    start = prev / "expertium" / TRACK
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
    if any(dd.get(k) != dp.get(k) for k in ("training_users", "training_reviews", "lr", "kind", "model", "track")):
        fail.append(f"users / reviews / lr / kind differ from epoch {n - 1}")
    if strict:
        if n == c["first"]:
            fail += [f"epoch {n}'s previous epoch: {b}" for b in source_ok(chain)]
        src_opts = decay_of(folder(chain, c["first"] - 1))["options"]
        if dd["options"] != src_opts or dd.get("architecture") != ARCH or src_opts.get("architecture") != ARCH \
                or src_opts.get("show_time_intervals") is not True \
                or dd["options"].get("curve_loss_weight") != c["curve_loss_weight"]:
            fail.append(f"options are not {c['name']}'s options ({ARCH}, show-time, the chain's curve_loss_weight)")
    info = (f"{len(p['rows']):,} users, {dd['training_reviews']:,} reviews, epoch {n}, arch {dd.get('architecture')}, "
            f"start {start.parent.parent.name} (sha256 {dd['init_master_sha256'][:12]}), "
            f"offset {full.get('update_offset')}, differences: protocol {sorted(set(diffs['protocol']))} + rows order; "
            f"decay {sorted(set(diffs['decay']))}")
    return fail, info


def pair(args):
    fail, info = pair_fail(args.chain, args.epoch)
    d = folder(args.chain, args.epoch)
    if fail:
        print(f"[sm135-epoch] pair REFUSE {args.chain} epoch {args.epoch}: " + "; ".join(fail[:12]))
        return 1
    print(f"[sm135-epoch] pair OK: {d.name} == {folder(args.chain, args.epoch - 1).name}'s step except the epoch index / "
          f"start / paths ({info})")
    return 0


def selfcheck(_args):
    bad = 0
    for chain, n in (("io", 41), ("th", 32)):
        d, prev = SRC / f"sm130-{chain}-e{n}-st", SRC / f"sm130-{chain}-e{n - 1}-st"
        fail, info = pair_fail(chain, n, d=d, prev=prev, strict=False)
        print(f"[sm135-epoch] selfcheck {chain} {d.name} vs {prev.name}: "
              f"{'OK' if not fail else 'REFUSE ' + '; '.join(fail[:6])} ({info})")
        bad += bool(fail)
    for chain in CHAINS:
        b = source_ok(chain)
        print(f"[sm135-epoch] selfcheck {chain} start {folder(chain, CHAINS[chain]['first'] - 1).name}: {'OK' if not b else '; '.join(b)}")
        bad += bool(b)
    return 1 if bad else 0


def main():
    p = ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "pair"):
        s = sub.add_parser(name)
        s.add_argument("--chain", choices=sorted(CHAINS), required=True)
        s.add_argument("--epoch", type=int, required=True)
    sub.add_parser("selfcheck")
    a = p.parse_args()
    return {"prepare": prepare, "pair": pair, "selfcheck": selfcheck}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
