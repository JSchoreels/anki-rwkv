#!/usr/bin/env python
"""Six-layer Tiny (SM130): the three constant-LR recovery epochs after the cut, for both chains. CPU only; each folder
it writes is trained by scripts/expertium_decay.py run (launch-sm130.sh). The code is the Tiny two-head model (SM106)'s
sm106_epoch.py (= the Tiny Instant-Only model (SM101)'s sm101_epoch.py) with the chain made a parameter.

Chains (read only sources):
  th  two-head: the Tiny two-head model (SM106)'s chain. Epochs 30, 31, 32 = exactly SM106's epoch-29 step (the leak-free
      small model (SM72s) run's order rule random.Random(split_seed * 1000 + N) over its 8,000 train users, epoch 25's
      label-window trims, AdamW 7e-4 constant, 2048 x 16 lanes, leak-free replay KD from the big model (1 leak-free
      epoch), both heads, "show_time_intervals": true) with the epoch index advanced and ONE change, the architecture
      (options/architecture and decay.json architecture = "w64-c1-d2p1u1"). Epoch 30 starts from the cut of SM106's
      epoch 29 (artifacts/small-model/sm130-th-cut-e29, sm130_cut.py) at SM106 epoch 29's update offset.
  io  Instant-Only: the Tiny Instant-Only model (SM101)'s chain. Epochs 39, 40, 41 = SM101's epoch-38 step
      (curve_loss_weight 0.0, show-time inputs; otherwise as above) with the architecture changed; epoch 39 starts from
      the cut of SM101's epoch 38 (artifacts/small-model/sm130-io-cut-e38).
Epoch N > FIRST starts from epoch N - 1 of this run (weights + AdamW state + update offset, options unchanged).

  prepare --chain th|io --epoch N    write sm130-{chain}-e{N}-st/{protocol.json, decay.json} (refuses if it exists or
                                     N - 1 is not complete)
  pair --chain th|io --epoch N       epoch N == epoch N - 1 except the order rule's epoch index (rows), the start, paths,
                                     notes, and for N = FIRST the architecture (w64-c1 -> w64-c1-d2p1u1) and the start =
                                     the pinned cut of the source epoch. Exit 1 otherwise.
  selfcheck                          the same generic pair rule accepts SM106's real epochs 29 vs 28 and SM101's 38 vs 37
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
ROOT = HERE.parents[2]                                       # the sm130 worktree
sys.path.insert(0, str(ROOT))
SM = ROOT / "artifacts/small-model"
TRACK = "E89-F65e-F77a"
W = Path("C:/Users/jonat/workspace/wt")
RUN72S = W / "sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run"
ARCH, BASE_ARCH = "w64-c1-d2p1u1", "w64-c1"
ARCH_PATHS = {"/options/architecture", "/architecture"}
CHAINS = {
    "th": {"first": 30, "last": 32, "src": W / "sm106/artifacts/small-model", "src_fmt": "sm106-c1-e{n}-st",
           "src_master_sha": "b3a694ca14cbd7a2291c7400203d5f9774bbdabc5d3e572a8408a196e5cb7be9",   # SM106 epoch 29
           "cut": SM / "sm130-th-cut-e29", "name": "the Tiny two-head model (SM106)", "curve_loss_weight": None},
    "io": {"first": 39, "last": 41, "src": W / "sm101/artifacts/small-model", "src_fmt": "sm101-c1-e{n}-st",
           "src_master_sha": "e6ea69124af00a3dcf2325047ae34580aaf96051d963ff6130dd06bde02d396c",   # SM101 epoch 38
           "cut": SM / "sm130-io-cut-e38", "name": "the Tiny Instant-Only model (SM101)", "curve_loss_weight": 0.0},
}


def folder(chain, n):
    c = CHAINS[chain]
    if n < c["first"]:
        return c["src"] / c["src_fmt"].format(n=n)
    return SM / f"sm130-{chain}-e{n}-st"


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


def cut_record(chain):
    c = CHAINS[chain]
    cut = c["cut"]
    rec = json.loads((cut / "extend.json").read_text())
    bad = []
    if rec.get("init") != "cut" or rec.get("architecture") != ARCH or rec.get("source_architecture") != BASE_ARCH:
        bad.append(f"not a {BASE_ARCH} -> {ARCH} cut")
    if rec.get("source_master_sha256") != c["src_master_sha"]:
        bad.append(f"its source is not {c['name']}'s pinned epoch")
    if rec.get("master_sha256") != sha256(cut / "master.pth") or rec.get("optimizer_sha256") != sha256(cut / "optimizer.pth"):
        bad.append("its files do not match extend.json")
    if not rec.get("optimizer_states_equal_after_load") or rec.get("parameters") != 185_057:
        bad.append("its AdamW state did not reload bitwise or the parameter count is not 185,057")
    if bad:
        raise SystemExit(f"{cut.name}: " + "; ".join(bad))
    src_case = folder(chain, CHAINS[chain]["first"] - 1) / "expertium" / TRACK
    if sha256(src_case / "optimizer.pth") != rec.get("source_optimizer_sha256"):
        bad.append("the source optimizer is not the cut's")
    if bad:
        raise SystemExit(f"{cut.name}: " + "; ".join(bad))
    return rec


def build(chain, n, config=None):
    from scripts.run_full_training import rows_for
    c = CHAINS[chain]
    config = config or run_config()
    prev, out = folder(chain, n - 1), folder(chain, n)
    if not complete(prev):
        raise SystemExit(f"{prev.name} is not a complete epoch")
    case = prev / "expertium" / TRACK
    if n == c["first"]:
        if sha256(case / "master.pth") != c["src_master_sha"]:
            raise SystemExit(f"{prev.name} is not {c['name']}'s pinned epoch (sha256)")
        cut_record(chain)
        start = c["cut"]
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
    if n == c["first"]:
        decay["options"]["architecture"] = ARCH
        decay["architecture"] = ARCH
    decay.update(source_epoch=n - 1, epoch=n, training_users=len(rows),
                 training_reviews=sum(r["train_range"][1] - r["train_range"][0] for r in rows), start=prev.name,
                 note=f"SM130 ({chain}): {c['name']}'s epoch step, epoch {n} (the run's order rule), architecture {ARCH} "
                      f"(six-layer Tiny: deck 2 / preset 1 / user 1 layers), start = "
                      f"{start.name if n == c['first'] else prev.name}"
                      f"{' (the pinned cut: trailing deck/preset/user layers removed, AdamW state re-indexed)' if n == c['first'] else ''}",
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


def pair_fail(chain, n, d=None, prev=None, prevprev=None, strict=True):
    """Epoch n (folder d) vs epoch n - 1 (prev): every leaf equal except the rows (= the run's epoch-n order rule with
    epoch 25's trims), full_run epoch / init_master / init_optimizer / update_offset / control_dir, decay source_epoch /
    epoch / start / note / created_utc / start hashes; for n = FIRST also the architecture (w64-c1 -> w64-c1-d2p1u1).
    strict=False (selfcheck on the source chains' real epochs) skips the cut and the architecture rules."""
    c = CHAINS[chain]
    config = run_config()
    d = Path(d or folder(chain, n))
    prev, prevprev = Path(prev or folder(chain, n - 1)), Path(prevprev or folder(chain, n - 2))
    first = c["first"] if strict else -1
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
        if strict and n == first and path in ARCH_PATHS and (x, y) == (ARCH, BASE_ARCH):
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
    start = c["cut"] if n == first else prev / "expertium" / TRACK
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
        if n == first:
            try:
                rec = cut_record(chain)
                if m_sha != rec["master_sha256"] or o_sha != rec["optimizer_sha256"]:
                    fail.append(f"epoch {n} does not start from the pinned cut")
            except SystemExit as error:
                fail.append(str(error))
            if sha256(prev / "expertium" / TRACK / "master.pth") != c["src_master_sha"]:
                fail.append(f"epoch {n}'s previous epoch is not {c['name']}'s pinned epoch")
        src_opts = decay_of(folder(chain, c["first"] - 1))["options"]
        want_opts = {**src_opts, "architecture": ARCH}
        if dd["options"] != want_opts or src_opts.get("architecture") != BASE_ARCH or dd.get("architecture") != ARCH \
                or src_opts.get("show_time_intervals") is not True \
                or dd["options"].get("curve_loss_weight") != c["curve_loss_weight"]:
            fail.append(f"options are not {c['name']}'s epoch-{c['first'] - 1} options with architecture {ARCH}")
    info = (f"{len(p['rows']):,} users, {dd['training_reviews']:,} reviews, epoch {n}, arch {dd.get('architecture')}, "
            f"start {start.name if n == first else start.parent.parent.name} (sha256 {dd['init_master_sha256'][:12]}), "
            f"offset {full.get('update_offset')}, differences: protocol {sorted(set(diffs['protocol']))} + rows order; "
            f"decay {sorted(set(diffs['decay']))}")
    return fail, info


def pair(args):
    fail, info = pair_fail(args.chain, args.epoch)
    d = folder(args.chain, args.epoch)
    if fail:
        print(f"[sm130-epoch] pair REFUSE {args.chain} epoch {args.epoch}: " + "; ".join(fail[:12]))
        return 1
    print(f"[sm130-epoch] pair OK: {d.name} == {folder(args.chain, args.epoch - 1).name}'s step except the epoch index / "
          f"start / paths{' / architecture ' + ARCH if args.epoch == CHAINS[args.chain]['first'] else ''} ({info})")
    return 0


def selfcheck(_args):
    bad = 0
    for chain, n in (("th", 29), ("io", 38)):
        fail, info = pair_fail(chain, n, strict=False)
        print(f"[sm130-epoch] selfcheck {chain} {folder(chain, n).name} vs {folder(chain, n - 1).name}: "
              f"{'OK' if not fail else 'REFUSE ' + '; '.join(fail[:6])} ({info})")
        bad += bool(fail)
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
