#!/usr/bin/env python
"""The show-time instant-only big model (SM99): prepare and pair-check its two big-model epochs (CPU, ~2 min each).

Each epoch is the show-time big model (SM91 step 2)'s own epoch recipe (scripts/sm72_big_epoch.py: the fork's
training users 5000-10000, whole histories, fresh AdamW, warmup 200 updates, 3.5e-4 -> 0 one_minus_sin over the epoch,
2 strided lanes x 8,192 carried state, budgeted carry table 768 MiB, GPU cap 0.8, RWKV_DELETED_CARDS=neighbour, the 27
label-window trims, snapshots every 50 users, show-time inputs) with exactly two changes:
  - "curve_loss_weight": 0.0 (instant-only training, as the instant-only small model (SM57) and its show-time chain);
  - a new user order per epoch: shuffle seed 2 (epoch 1) and 3 (epoch 2) instead of 1 (same users, same train ranges).
Epoch 1 starts from the show-time big model (SM91 step 2)'s master.pth; epoch 2 from epoch 1's.

  prepare --epoch N   sm72_big_epoch.py prepare + sm72_label_windows.py --trim (the label-window pre-flight)
  pair --epoch N      the folder == the show-time big model (SM91 step 2)'s prepared epoch folder except the allowed
                      fields (and, for N = 2, == epoch 1's folder except start, order, paths); writes
                      gpu/pair-e<N>.json. Refuses a folder that has started training (protocol.json is rewritten on a
                      resume); the launcher then relies on the saved gpu/pair-e<N>.json.
  negatives           the pair check on mutated copies must refuse each (used by sm99_check.py)
"""
from argparse import ArgumentParser
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
G = HERE / "gpu"
PY = sys.executable
SM = ROOT / "artifacts/small-model"
REF = Path("C:/Users/jonat/workspace/wt/sm91s2/artifacts/small-model/sm91s2-big-showtime")   # read only
REF_CK = REF / "expertium/big-showtime/master.pth"                                             # the show-time big model
REF_SHA = "4f31860d3cdeba2a79bb627d4aad7a2f620c4a20ca09e75c2c5f5748cb8fd63b"                 # its trained_parameter_sha256
REF_FILE_SHA = "67ee8e9b0d0169733242c40646e0d2e5637845a25775762002ffc4e6396f9590"            # sha256 of that master.pth
RUN8K = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/full-run/run-8k-20261013/run.json")
TRACK = "big-inst"
SEEDS = {1: 2, 2: 3}
EXTRA = {"show_time_intervals": True, "curve_loss_weight": 0.0}
SNAPSHOT_USERS = 50


def folder(n):
    return SM / f"sm99-big-inst-e{n}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def complete(d):
    r = d / "expertium" / TRACK / "result.json"
    if not r.exists():
        return False
    res = json.loads(r.read_text())
    return res.get("status") == "complete" and not res.get("smoke_only")


def start_of(n):
    if n == 1:
        return REF_CK
    prev = folder(n - 1)
    if not complete(prev):
        raise SystemExit(f"[sm99-epoch] epoch {n - 1} is not complete ({prev})")
    return prev / "expertium" / TRACK / "master.pth"


def prepare(n):
    out = folder(n)
    if out.exists():
        raise SystemExit(f"[sm99-epoch] {out} exists")
    env = dict(os.environ, RWKV_DELETED_CARDS="neighbour")
    cmd = [PY, "-X", "utf8", str(ROOT / "scripts/sm72_big_epoch.py"), "prepare", "--out", out.relative_to(ROOT).as_posix(),
           "--init", str(start_of(n)), "--track", TRACK, "--chunk-reviews", "8192", "--batch-chunks", "2",
           "--snapshot-users", str(SNAPSHOT_USERS), "--seed", str(SEEDS[n]), "--extra-options", json.dumps(EXTRA)]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    G.mkdir(exist_ok=True)
    subprocess.run([PY, "-X", "utf8", str(ROOT / "scripts/sm72_label_windows.py"), "--decay", str(out), "--trim",
                    "--out", str(G / f"check-label-windows-sm99-e{n}.json")], cwd=ROOT, env=env, check=True)
    d = json.loads((out / "decay.json").read_text())
    print(f"[sm99-epoch] prepared epoch {n}: {out.name}, start {start_of(n)}, seed {SEEDS[n]}, "
          f"{d['training_users']} users, {d.get('training_reviews_after_trims'):,} reviews after {d['label_windows']}")
    return 0


def load(d):
    """decay.json + the prepared protocol (protocol.prepared.json once a resume has rewritten protocol.json)."""
    p = d / "protocol.prepared.json"
    p = p if p.exists() else d / "protocol.json"
    x = {"decay": json.loads((d / "decay.json").read_text()), "protocol": json.loads(p.read_text())}
    u = d / "protocol.untrimmed.json"
    x["untrimmed"] = json.loads(u.read_text()) if u.exists() else None
    return x


def by_user(rows):
    return sorted(rows, key=lambda r: r["user_id"])


def order(seed):
    counts = json.loads(RUN8K.read_text())["review_counts"]
    users = [u for u in range(5000, 10001) if str(u) in counts]
    random.Random(seed).shuffle(users)
    return users


def problems(arm, ref, start, seed, ref_is_sm91s2=True):
    """What differs between `arm` and `ref` (loaded folders) beyond the allowed fields; [] = the pair holds."""
    out = []
    a, b = arm["decay"], ref["decay"]
    allowed = {"created_utc", "source_case", "source_sha256", "track", "extra_options", "shuffle_seed",
               "trimmed_unlabelled_tails"}
    keys = sorted(k for k in set(a) | set(b) if k != "options" and k not in allowed and a.get(k) != b.get(k))
    if keys:
        out.append(f"decay.json differs in {keys}")
    tails = lambda x: sorted(x.get("trimmed_unlabelled_tails") or [], key=lambda t: t["user_id"])  # noqa: E731
    if tails(a) != tails(b) or not tails(a):
        out.append("label-window trims differ from the reference (or none applied)")
    oa, ob = a["options"], b["options"]
    okeys = sorted(k for k in set(oa) | set(ob) if oa.get(k) != ob.get(k))
    want = ["curve_loss_weight"] if ref_is_sm91s2 else []
    if okeys != want or oa.get("curve_loss_weight") != 0.0 or type(oa.get("curve_loss_weight")) is not float \
            or oa.get("show_time_intervals") is not True or a.get("extra_options") != EXTRA:
        out.append(f"options differ in {okeys} (want {want}), curve_loss_weight {oa.get('curve_loss_weight')!r}, "
                   f"show_time_intervals {oa.get('show_time_intervals')!r}, extra_options {a.get('extra_options')}")
    if a.get("shuffle_seed") != seed or a.get("track") != TRACK or a.get("deleted_cards") != "neighbour" \
            or a.get("model") != "expertium" or "label_windows" not in a:
        out.append(f"seed / track / deleted_cards / model / label windows: {a.get('shuffle_seed')} {a.get('track')} "
                   f"{a.get('deleted_cards')} {a.get('model')} {a.get('label_windows')}")
    pa, pb = arm["protocol"], ref["protocol"]
    if by_user(pa["rows"]) != by_user(pb["rows"]):
        out.append("protocol rows (users, trimmed train ranges) differ")
    if [r["user_id"] for r in pa["rows"]] != order(seed):
        out.append(f"row order is not the shuffle of users 5000-10000 with seed {seed}")
    if [r["user_id"] for r in pa["rows"]] == [r["user_id"] for r in pb["rows"]]:
        out.append("row order equals the reference's (the seed did not change it)")
    top = sorted(k for k in set(pa) | set(pb) if k not in ("full_run", "rows") and pa.get(k) != pb.get(k))
    if top:
        out.append(f"protocol.json differs in {top}")
    fa, fb = pa["full_run"], pb["full_run"]
    fkeys = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
    if not set(fkeys) <= {"init_master", "control_dir"}:
        out.append(f"full_run differs in {fkeys}")
    if fa.get("optimizer_state") != "fresh" or fa.get("update_offset") != 0 or "lr_resume" in fa \
            or fa.get("snapshot_users") != SNAPSHOT_USERS or fa.get("lr_schedule", {}).get("peak") != 3.5e-4:
        out.append(f"not a fresh start of the recipe: {fa}")
    if Path(fa["init_master"]).resolve() != Path(start).resolve() or Path(a["source_case"]).resolve() != Path(start).resolve():
        out.append(f"start {fa['init_master']} is not {start}")
    ua, ub = arm["untrimmed"], ref["untrimmed"]
    if ua is None or ub is None or by_user(ua["rows"]) != by_user(ub["rows"]):
        out.append("protocol.untrimmed.json rows differ (or missing)")
    return out


def pair(n, write=True):
    d = folder(n)
    if (d / "expertium").exists() or (d / "attempts").exists() or (d / "protocol.prepared.json").exists():
        saved = G / f"pair-e{n}.json"
        ok = saved.exists() and json.loads(saved.read_text()).get("ok") is True
        print(f"[sm99-epoch] epoch {n} has started training; saved pair check {'OK' if ok else 'MISSING / FAILED'} ({saved.name})")
        return 0 if ok else 1
    start = start_of(n)
    arm = load(d)
    res = {"epoch": n, "folder": d.name, "start": str(start), "start_sha256": sha256(start), "seed": SEEDS[n]}
    res["vs_sm91s2"] = problems(arm, load(REF), start, SEEDS[n])
    if arm["decay"]["source_sha256"] != res["start_sha256"]:
        res["vs_sm91s2"].append("decay.json source_sha256 != the start file's sha256")
    if n == 1:
        ref_res = json.loads((REF_CK.parent / "result.json").read_text())
        if ref_res.get("trained_parameter_sha256") != REF_SHA or ref_res.get("status") != "complete"                 or res["start_sha256"] != REF_FILE_SHA:
            res["vs_sm91s2"].append("the show-time big model (SM91 step 2)'s result.json changed")
    else:
        prev = load(folder(n - 1))
        e = copy.deepcopy(prev)   # epoch 1 as the reference: allowed = start, seed (order), paths
        e["decay"]["options"] = dict(e["decay"]["options"])
        res["vs_epoch1"] = problems(arm, e, start, SEEDS[n], ref_is_sm91s2=False)
        prev_res = json.loads((folder(n - 1) / "expertium" / TRACK / "result.json").read_text())
        if prev_res.get("curve_loss_weight") != 0.0:
            res["vs_epoch1"].append(f"epoch {n - 1} did not train with curve_loss_weight 0.0 ({prev_res.get('curve_loss_weight')})")
    res["ok"] = not res["vs_sm91s2"] and not res.get("vs_epoch1")
    if write:
        G.mkdir(exist_ok=True)
        (G / f"pair-e{n}.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"[sm99-epoch] pair epoch {n}: {'OK' if res['ok'] else 'FAIL'} {json.dumps({k: v for k, v in res.items() if k.startswith('vs_')})}")
    return 0 if res["ok"] else 1


def negatives():
    """Mutated copies of epoch 1 that the pair check must refuse."""
    arm, ref = load(folder(1)), load(REF)
    cases = {}

    def m(name, f):
        x = copy.deepcopy(arm)
        f(x)
        cases[name] = bool(problems(x, ref, REF_CK, SEEDS[1]))
    cases["sm91s2_against_itself"] = bool(problems(copy.deepcopy(ref), ref, REF_CK, SEEDS[1]))
    m("curve_weight_1", lambda x: x["decay"]["options"].update(curve_loss_weight=1.0))
    m("curve_weight_int_0", lambda x: x["decay"]["options"].update(curve_loss_weight=0))
    m("no_show_time", lambda x: x["decay"]["options"].update(show_time_intervals=False))
    m("other_chunk", lambda x: x["decay"]["options"].update(chunk_reviews=4096))
    m("seed_1_order", lambda x: x["protocol"].update(rows=list(ref["protocol"]["rows"])))
    m("dropped_user", lambda x: x["protocol"].update(rows=x["protocol"]["rows"][1:]))
    m("other_peak", lambda x: x["protocol"]["full_run"]["lr_schedule"].update(peak=7e-4))
    m("other_start", lambda x: x["protocol"]["full_run"].update(init_master=str(folder(1) / "x.pth")))
    m("missing_trim", lambda x: x["decay"].update(trimmed_unlabelled_tails=x["decay"]["trimmed_unlabelled_tails"][1:]))
    m("snapshot_400", lambda x: x["protocol"]["full_run"].update(snapshot_users=400))
    ok = all(cases.values()) and not problems(arm, ref, REF_CK, SEEDS[1])
    print(f"[sm99-epoch] negatives {'OK' if ok else 'FAIL'}: {json.dumps(cases)}")
    return ok, cases


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("prepare", "pair", "negatives"))
    ap.add_argument("--epoch", type=int, choices=(1, 2), default=1)
    args = ap.parse_args()
    if args.command == "prepare":
        return prepare(args.epoch)
    if args.command == "pair":
        return pair(args.epoch)
    return 0 if negatives()[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
