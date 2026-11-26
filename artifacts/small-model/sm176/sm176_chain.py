#!/usr/bin/env python
"""How many users for an MIT Small (SM176): the training chain of every arm (design.md). It is the chain of How many
users for MIT weights (SM174) with ONE change: the architecture is Small's (`w64`, the Small two-head model (SM118),
331,511 parameters) instead of the six-layer Tiny's (`w64-c1-d2p1u1`). Arms: the 8,000-user teacher control first, then
1,000 / 2,000 / 4,000 users without a teacher (500 and 8,000 without a teacher dropped: GPU budget). CPU only, seconds;
every folder it writes is trained by scripts/expertium_decay.py run of the run tree (launch-sm176.sh).

SM174's text, unchanged except the architecture: trained FROM SCRATCH (fresh init, training_seed 2025) on a
nested random subset of the 8,000 training users of the leak-free small model (SM72s) run, both heads, show-time
inputs, leak-free (RWKV_DELETED_CARDS=neighbour), 2048-review windows x 16 strided lanes, AdamW 7e-4. No teacher (plain
LogLoss on the real outcomes) except the control arm n8000-kd, which keeps the usual replay teacher of the constant-LR
epochs (the big model (1 leak-free epoch), alpha 0.9, decay_alpha 0.5). Everything else is the six-layer Tiny two-head
(SM130) epoch-30 folder (template.json, SM174's file unchanged) with options.architecture = "w64"; check verifies
that the result equals the Small chain's own epoch-26 folder (the show-time retrain of the release (SM92) e26, the
start of the Small two-head model (SM118)) except the per-folder keys and the removed teacher.

Rule (fixed before any run): review budget B = 4 x 581,942,724 (four 8k epochs) per arm. Epochs E = round(B / R_n)
(R_n = the subset's reviews per epoch); decay epochs D_max = max(1, round(0.15 E)); constant-LR epochs at most
C_max = E - D_max. The held-out slice (users.json "slice": validation users with id >= 5000, never trained on, not in
the 732 set) is scored every k = max(1, round(290M / R_n)) constant epochs and at C_max. The constant phase stops at
the first eval point whose slice loss (mean of the review-weighted curve and instant LogLoss) is not at least 0.05%
below the best earlier eval point, or at C_max. The decay then starts from the best eval point (epoch b) and runs
D = max(1, round(D_max x b / C_max)) epochs, one_minus_sin 7e-4 -> 0 over exactly D x R_n reviews (chained one-epoch
segments, lr_resume "segment"). The last segment's master.pth is the arm's model.

  init                 write users.json and template.json (refuses to overwrite a different one)
  plan                 the per-arm numbers (E, D_max, C_max, k, eval points, reviews, GPU ETA)
  next --arm A         prepare what is needed and print the next action of arm A, one line:
                         TRAIN <folder> <minutes> | EVAL <tag> <master> | SCORE <tag> <master> | DONE <tag> <master>
  slice --arm A        the slice losses of A's eval points so far (JSON)
  check                CPU checks: users nested and disjoint from the 732 set and the slice; every prepared folder of
                       every arm equals the template except the listed keys (pair rule); a full dry walk of the
                       state machine on fake slice losses (early stop and budget end)
"""
from argparse import ArgumentParser
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
RUN = Path(os.environ.get("SM176_RUN", "C:/Users/jonat/workspace/wt/sm174-run"))   # SM174's run tree, unedited, reused
RUN_HEAD = "cea0f931460996830736bdaa06e3a7d66418479b"
OUT = RUN / "artifacts/small-model/sm176"
G = HERE / "gpu"
TEMPLATE_SRC = Path("C:/Users/jonat/workspace/wt/sm130/artifacts/small-model/sm130-th-e30-st")
SMALL_SRC = Path("C:/Users/jonat/workspace/wt/sm92/artifacts/small-model/sm92-e26")   # the Small chain's epoch 26
RUN72S = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run")
U732POOL = Path("C:/Users/jonat/workspace/wt/sm130/artifacts/small-model/eval-users-732.json")
TRACK = "E89-F65e-F77a"
ARCH = "w64"                            # Small (SM118); SM174 had "w64-c1-d2p1u1"
EPOCH8K = 581_942_724
BUDGET = 4 * EPOCH8K
DECAY_SHARE = 0.15
EVAL_REVIEWS = 290_000_000
MIN_GAIN = 0.0005                       # 0.05%
SPEED = 87_000                          # reviews/s of Small 2048-window epochs (SM92 e26, SM98 e32-e34: 86.9k-87.4k)
EPOCH_OVERHEAD_S = 30
SUBSET_ARMS = {"n0500": (500, False), "n1000": (1000, False), "n2000": (2000, False), "n4000": (4000, False),
               "n8000": (8000, False), "n8000-kd": (8000, True)}     # SM174's users.json (reused unchanged)
ARMS = {"n8000-kd": (8000, True), "n1000": (1000, False), "n2000": (2000, False), "n4000": (4000, False)}
ORDER = list(ARMS)                      # the reference first, then smallest to largest
SUBSET_SEED, ORDER_SEED, SLICE_SEED = 174, 174_000, 1741
SLICE_MAX_USER_REVIEWS, SLICE_REVIEWS = 120_000, 6_000_000
SLICE_ROWS = G / "sm176-slice-rows.jsonl"
ROWS732 = G / "sm176-rows-732.jsonl"
VARIANT = "w8+s8hx8"
E2S = "@e2s-query"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- users and template

def build_users():
    run = load(RUN72S / "run.json")
    counts = run["review_counts"]
    perm = sorted(run["train_users"])
    random.Random(SUBSET_SEED).shuffle(perm)
    subsets = {arm: sorted(perm[:n]) for arm, (n, _) in SUBSET_ARMS.items()}
    pool732 = set(load(U732POOL)["users"])
    cand = sorted(u for u in run["validation_users"] if u not in pool732 and counts[str(u)] <= SLICE_MAX_USER_REVIEWS)
    random.Random(SLICE_SEED).shuffle(cand)
    slice_users, total = [], 0
    for u in cand:
        if total >= SLICE_REVIEWS:
            break
        slice_users.append(u)
        total += counts[str(u)]
    return {"note": "SM174: nested random subsets of the leak-free small model (SM72s) run's 8,000 train users "
                    f"(sorted ids shuffled with random.Random({SUBSET_SEED}), prefixes); slice = validation users not in "
                    f"the 732 pool (ids >= 5000), at most {SLICE_MAX_USER_REVIEWS:,} reviews each, shuffled with "
                    f"random.Random({SLICE_SEED}), taken until {SLICE_REVIEWS:,} reviews. Public benchmark user ids only.",
            "split": str(RUN72S / "run.json"), "split_sha256": sha256(RUN72S / "run.json"),
            "subset_order": perm, "subsets": subsets, "users": sorted(slice_users), "slice_reviews": total}


def build_template():
    protocol = load(TEMPLATE_SRC / "protocol.json")
    decay = load(TEMPLATE_SRC / "decay.json")
    protocol = {k: v for k, v in protocol.items() if k not in ("rows", "evaluation_rows")}
    return {"source": str(TEMPLATE_SRC), "protocol_sha256": sha256(TEMPLATE_SRC / "protocol.json"),
            "decay_sha256": sha256(TEMPLATE_SRC / "decay.json"), "protocol": protocol, "decay": decay}


def init(args):
    for name, value in (("users.json", build_users()), ("template.json", build_template())):
        path = HERE / name
        if path.exists():
            if load(path) != json.loads(json.dumps(value)):
                raise SystemExit(f"{path} exists and differs from a fresh build")
            print(f"{name}: unchanged")
        else:
            atomic_json(path, value)
            print(f"{name}: written")
    return 0


def users():
    return load(HERE / "users.json")


def template():
    return load(HERE / "template.json")


def counts():
    return load(RUN72S / "run.json")["review_counts"]


def trims():
    return {t["user_id"]: t for t in template()["protocol"].get("label_window_trims", [])}


def rows_for(user_list, cnt, tr):
    rows = []
    for u in user_list:
        n = cnt[str(u)]
        end = tr[u]["to"] if u in tr else n
        rows.append({"user_id": u, "available_reviews": n, "train_range": [0, end], "validation_range": [0, n]})
    return rows


# ---------------------------------------------------------------- plan

def plan(arm, cnt=None, us=None, tr=None):
    cnt = cnt or counts()
    us = us or users()
    tr = trims() if tr is None else tr
    subset = us["subsets"][arm]
    r = sum(row["train_range"][1] for row in rows_for(subset, cnt, tr))
    e = max(2, round(BUDGET / r))
    d_max = max(1, round(DECAY_SHARE * e))
    c_max = e - d_max
    k = max(1, round(EVAL_REVIEWS / r))
    points = sorted(set(list(range(k, c_max + 1, k)) + [c_max]))
    return {"arm": arm, "users": len(subset), "teacher": ARMS[arm][1], "reviews_per_epoch": r, "epochs": e,
            "decay_epochs_max": d_max, "constant_epochs_max": c_max, "eval_every": k, "eval_points": points,
            "budget_reviews": e * r, "epoch_minutes": r / SPEED / 60 + EPOCH_OVERHEAD_S / 60}


def cmd_plan(args):
    total = 0
    for arm in ORDER:
        p = plan(arm)
        h = p["epochs"] * p["epoch_minutes"] / 60
        total += h
        print(f"{arm}: {p['users']:,} users, {p['reviews_per_epoch']:,} reviews/epoch, E {p['epochs']} = C_max "
              f"{p['constant_epochs_max']} + D_max {p['decay_epochs_max']}, eval every {p['eval_every']} -> "
              f"{len(p['eval_points'])} points {p['eval_points']}, {p['budget_reviews'] / 1e9:.3f}B reviews, "
              f"~{p['epoch_minutes']:.1f} min/epoch, training ~{h:.1f} h (upper bound)")
    print(f"training total ~{total:.1f} h (no early stop), + slice evals + 732 scoring")
    return 0


# ---------------------------------------------------------------- folders

def folder(arm, kind, j):
    return OUT / arm / f"{kind}{j:02d}"


def case(d):
    return Path(d) / "expertium" / TRACK


def result_of(d):
    p = case(d) / "result.json"
    return load(p) if p.exists() else None


def complete(d):
    r = result_of(d)
    return bool(r) and r.get("status") == "complete" and not r.get("smoke_only") and (case(d) / "master.pth").exists()


def order_rows(arm, epoch_index, cnt, us, tr):
    subset = us["subsets"][arm][:]
    random.Random(ORDER_SEED + epoch_index).shuffle(subset)
    return rows_for(subset, cnt, tr)


def build_folder(arm, kind, j, start, update_offset, decay_epochs=None, best=None, cnt=None, us=None, tmpl=None):
    """protocol.json + decay.json of epoch e{j} (kind "e") or decay segment d{j} (kind "d") of `arm`."""
    cnt = cnt or counts()
    us = us or users()
    tmpl = tmpl or template()
    tr = {t["user_id"]: t for t in tmpl["protocol"].get("label_window_trims", [])}
    n, teacher = ARMS[arm]
    p = plan(arm, cnt, us, tr)
    out = folder(arm, kind, j)
    epoch_index = j if kind == "e" else best + j          # the order seed's epoch index: decay segments continue it
    protocol = copy.deepcopy(tmpl["protocol"])
    protocol.pop("accuracy_loop", None)
    protocol["rows"] = order_rows(arm, epoch_index, cnt, us, tr)
    protocol["evaluation_rows"] = []
    protocol["label_window_trims"] = [t for t in tmpl["protocol"].get("label_window_trims", [])
                                      if t["user_id"] in set(us["subsets"][arm])]
    full = protocol["full_run"]
    for key in ("init_master", "init_optimizer", "update_offset", "lr_resume", "resumed", "optimizer_state", "lr_schedule"):
        full.pop(key, None)
    full.update(epoch=epoch_index, control_dir=(out / "control").relative_to(RUN).as_posix(), update_offset=int(update_offset))
    if start is not None:
        full.update(init_master=(Path(start) / "master.pth").as_posix(), init_optimizer=(Path(start) / "optimizer.pth").as_posix())
    reviews = sum(row["train_range"][1] - row["train_range"][0] for row in protocol["rows"])
    assert reviews == p["reviews_per_epoch"]
    if kind == "d":
        full["lr_schedule"] = {"decay_start": 0.0, "decay_end": 1.0, "shape": "one_minus_sin", "final": 0.0}
        full["lr_resume"] = {"total_reviews": decay_epochs * reviews, "reviews_offset": (j - 1) * reviews, "segment": True}
    decay = copy.deepcopy(tmpl["decay"])
    for key in ("created_utc", "init_master_sha256", "init_optimizer_sha256", "start", "source_run", "source_epoch"):
        decay.pop(key, None)
    decay["options"]["architecture"] = ARCH
    decay["architecture"] = ARCH
    if not teacher:
        decay["options"].pop("distillation", None)
    lr = ("7e-4 constant after the 5-update warmup (R01 default)" if kind == "e" else
          f"one_minus_sin 7e-4 -> 0 over {decay_epochs} epochs x {reviews:,} reviews, segment {j} of {decay_epochs}")
    decay.update(kind=f"sm176 {'from-scratch constant-LR epoch' if kind == 'e' else 'decay segment'}", epoch=epoch_index,
                 arm=arm, arm_users=n, teacher=bool(teacher), training_users=len(protocol["rows"]), training_reviews=reviews,
                 lr=lr, start=(str(start) if start is not None else f"fresh init (training_seed {decay['options']['training_seed']})"),
                 template=tmpl["source"], template_decay_sha256=tmpl["decay_sha256"],
                 note=f"SM176 {arm}: Small ({ARCH}) from scratch on {n:,} users "
                      f"({'with the usual replay teacher' if teacher else 'no teacher'}), {kind}{j:02d}")
    if start is not None:
        decay.update(init_master_sha256=sha256(Path(start) / "master.pth"), init_optimizer_sha256=sha256(Path(start) / "optimizer.pth"))
    return protocol, decay


def prepare(arm, kind, j, start, update_offset, **kw):
    out = folder(arm, kind, j)
    if out.exists():
        return out
    protocol, decay = build_folder(arm, kind, j, start, update_offset, **kw)
    decay["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    atomic_json(tmp / "protocol.json", protocol)
    atomic_json(tmp / "decay.json", decay)
    tmp.rename(out)
    return out


# ---------------------------------------------------------------- slice losses and the state machine

def tag(arm, kind, j):
    return f"sm176-{arm}-{kind}{j:02d}"


def read_rows(path):
    rows = {}
    if Path(path).exists():
        for raw in open(path, "rb"):
            try:
                r = json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if "benchmark" in r and "error" not in r:
                rows.setdefault(r["label"], {})[r["user_id"]] = r
    return rows


def slice_loss(rows, label, slice_users):
    got = rows.get(label, {})
    if not set(slice_users) <= set(got):
        return None
    out = {}
    for head in ("ahead", "instant"):
        num = sum(got[u]["benchmark"][head]["LogLoss"] * got[u]["benchmark"][head]["size"] for u in slice_users)
        den = sum(got[u]["benchmark"][head]["size"] for u in slice_users)
        out[head] = num / den
    out["mean"] = (out["ahead"] + out["instant"]) / 2
    return out


def state_path(arm):
    return G / f"{arm}-state.json"


def decide(p, losses):
    """losses: {eval point: mean slice loss} for the points reached so far (in order). Returns None (continue) or the
    stop decision."""
    seen = [e for e in p["eval_points"] if e in losses]
    if not seen:
        return None
    last = seen[-1]
    best_prev = min((losses[e] for e in seen[:-1]), default=None)
    reason = None
    if best_prev is not None and losses[last] > best_prev * (1 - MIN_GAIN):
        reason = f"early stop: eval point {last} is not 0.05% below the best earlier point"
    elif last == p["constant_epochs_max"]:
        reason = "constant-LR budget reached (C_max)"
    if reason is None:
        return None
    best = min(seen, key=lambda e: (losses[e], e))
    d = max(1, round(p["decay_epochs_max"] * best / p["constant_epochs_max"]))
    return {"reason": reason, "last_eval_point": last, "best_epoch": best, "decay_epochs": d,
            "slice_mean_loss": {str(e): losses[e] for e in seen}}


def next_action(arm, rows_slice=None, rows732=None, write=True):
    cnt, us, tmpl = counts(), users(), template()
    p = plan(arm, cnt, us)
    sp = state_path(arm)
    st = load(sp) if sp.exists() else None
    rows_slice = read_rows(SLICE_ROWS) if rows_slice is None else rows_slice
    if st is None:
        losses, offset, start = {}, 0, None
        for e in range(1, p["constant_epochs_max"] + 1):
            f = folder(arm, "e", e)
            if not complete(f):
                prepare(arm, "e", e, start, offset, cnt=cnt, us=us, tmpl=tmpl)
                return f"TRAIN {f.as_posix()} {math.ceil(p['epoch_minutes'] + 2)}"
            if e in p["eval_points"]:
                label = f"{tag(arm, 'e', e)}/{VARIANT}{E2S}"
                loss = slice_loss(rows_slice, label, us["users"])
                if loss is None:
                    return f"EVAL {tag(arm, 'e', e)} {(case(f) / 'master.pth').as_posix()}"
                losses[e] = loss["mean"]
                st = decide(p, losses)
                if st is not None:
                    break
            offset, start = result_of(f)["update_offset_end"], case(f)
        if st is None:
            raise SystemExit(f"{arm}: no stop decision after C_max (bug)")
        st.update(arm=arm, plan=p, decided_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        if write:
            atomic_json(sp, st)
    best, d = st["best_epoch"], st["decay_epochs"]
    start = case(folder(arm, "e", best))
    offset = result_of(folder(arm, "e", best))["update_offset_end"]
    for j in range(1, d + 1):
        f = folder(arm, "d", j)
        if not complete(f):
            prepare(arm, "d", j, start, offset, decay_epochs=d, best=best, cnt=cnt, us=us, tmpl=tmpl)
            return f"TRAIN {f.as_posix()} {math.ceil(p['epoch_minutes'] + 2)}"
        offset, start = result_of(f)["update_offset_end"], case(f)
    final = case(folder(arm, "d", d)) / "master.pth"
    rows732 = read_rows(ROWS732) if rows732 is None else rows732
    have = len(rows732.get(f"sm176-{arm}/{VARIANT}{E2S}", {}))
    return f"{'DONE' if have >= 720 else 'SCORE'} sm176-{arm} {final.as_posix()}"


def cmd_next(args):
    print(next_action(args.arm))
    return 0


def cmd_slice(args):
    us, rows = users(), read_rows(SLICE_ROWS)
    p = plan(args.arm)
    out = {}
    for e in p["eval_points"]:
        loss = slice_loss(rows, f"{tag(args.arm, 'e', e)}/{VARIANT}{E2S}", us["users"])
        if loss:
            out[e] = loss
    print(json.dumps(out, indent=1))
    return 0


# ---------------------------------------------------------------- checks

def leaves(x, y, path=""):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            yield from leaves(x.get(k), y.get(k), f"{path}/{k}")
    elif x != y:
        yield path


def check(args):
    ok = True

    def claim(cond, text):
        nonlocal ok
        ok &= bool(cond)
        print(("OK   " if cond else "FAIL ") + text)

    us, cnt, tmpl = users(), counts(), template()
    fresh = build_users()
    claim(fresh == us, "users.json equals a fresh build")
    claim(sha256(HERE / "users.json") == sha256(HERE.parent / "sm174/users.json"), "users.json is SM174's file (same subsets, same slice)")
    claim(sha256(HERE / "template.json") == sha256(HERE.parent / "sm174/template.json"), "template.json is SM174's file")
    claim(load(TEMPLATE_SRC / "protocol.json")["full_run"]["epoch"] == 30 and sha256(TEMPLATE_SRC / "decay.json") == tmpl["decay_sha256"],
          "template.json is the pinned SM130 epoch-30 folder")
    s = us["subsets"]
    claim(all(set(s[a]) <= set(s[b]) for a, b in zip(list(SUBSET_ARMS)[:4], list(SUBSET_ARMS)[1:5])) and s["n8000"] == s["n8000-kd"], "subsets nested")
    run = load(RUN72S / "run.json")
    claim(set(s["n8000"]) == set(run["train_users"]), "n8000 = all 8,000 train users")
    pool732 = set(load(U732POOL)["users"])
    claim(not (set(us["users"]) & pool732) and not (set(us["users"]) & set(run["train_users"])),
          f"slice ({len(us['users'])} users, {us['slice_reviews']:,} reviews) disjoint from the 732 pool and from training")
    claim(sum(r["train_range"][1] for r in rows_for(s["n8000"], cnt, trims())) == EPOCH8K, "8k epoch = 581,942,724 reviews (trim applied)")
    # pair rule on built (not written) folders: only these keys may differ from the template
    allowed_p = {"/rows", "/evaluation_rows", "/label_window_trims", "/full_run/epoch", "/full_run/control_dir",
                 "/full_run/update_offset", "/full_run/init_master", "/full_run/init_optimizer", "/full_run/lr_schedule",
                 "/full_run/lr_resume", "/accuracy_loop"}
    allowed_d = {"/kind", "/epoch", "/arm", "/arm_users", "/teacher", "/training_users", "/training_reviews", "/lr", "/start",
                 "/template", "/template_decay_sha256", "/note", "/created_utc", "/init_master_sha256",
                 "/init_optimizer_sha256", "/source_run", "/source_epoch", "/architecture", "/options/architecture"}
    small = load(SMALL_SRC / "decay.json")
    claim(small["options"]["architecture"] == ARCH == small["architecture"] and small["kind"] == "sm77 constant-LR epoch",
          f"the Small chain's epoch-26 folder (SM92 e26) is a {ARCH} constant-LR epoch")
    tprot = load(TEMPLATE_SRC / "protocol.json")
    tdec = load(TEMPLATE_SRC / "decay.json")
    for arm in ORDER:
        for kind, j, start in (("e", 1, None), ("d", 2, None)):
            prot, dec = build_folder(arm, kind, j, start, 0, decay_epochs=3, best=4, cnt=cnt, us=us, tmpl=tmpl)
            dp = {p for p in leaves(tprot, prot) if not p.startswith("/rows")} - allowed_p
            ddiff = set(leaves(tdec, dec)) - allowed_d
            want_opt = set() if ARMS[arm][1] else {"/options/distillation"}
            want_small = dict(small["options"])
            if not ARMS[arm][1]:
                want_small.pop("distillation", None)
            claim(dec["options"] == want_small and dec["architecture"] == ARCH,
                  f"{arm} {kind}{j:02d}: options equal the Small chain's epoch 26 (SM92 e26) "
                  f"{'exactly' if ARMS[arm][1] else 'minus the teacher'}; architecture {dec['options']['architecture']}")
            claim(dp == set() and ddiff == want_opt, f"{arm} {kind}{j:02d}: differs from the template only in allowed keys "
                  f"(protocol extra {sorted(dp)}, decay extra {sorted(ddiff - want_opt)}, distillation "
                  f"{'kept' if ARMS[arm][1] else 'removed'})")
            claim("init_master" not in prot["full_run"] and prot["full_run"]["update_offset"] == 0 or kind == "d",
                  f"{arm} {kind}{j:02d}: fresh init, update offset 0" if kind == "e" else f"{arm} d02: segment")
            if kind == "d":
                lr = prot["full_run"]["lr_resume"]
                claim(lr == {"total_reviews": 3 * plan(arm)["reviews_per_epoch"], "reviews_offset": plan(arm)["reviews_per_epoch"], "segment": True},
                      f"{arm} d02 of 3: lr_resume {lr}")
            rows = prot["rows"]
            claim(sorted(r["user_id"] for r in rows) == sorted(s[arm]) and len(rows) == ARMS[arm][0], f"{arm} {kind}{j:02d}: rows = the subset")
    # state machine dry walk on fake losses
    p = plan("n1000")
    lo = {e: 1.0 - 0.01 * i for i, e in enumerate(p["eval_points"])}
    st = None
    for e in p["eval_points"]:
        st = decide(p, {x: lo[x] for x in p["eval_points"] if x <= e})
        if st:
            break
    claim(st and st["best_epoch"] == p["constant_epochs_max"] and st["decay_epochs"] == p["decay_epochs_max"],
          f"dry walk, always improving: stop at C_max {p['constant_epochs_max']}, decay {p['decay_epochs_max']} ({st and st['reason']})")
    fl = dict(lo)
    pts = p["eval_points"]
    fl[pts[3]] = fl[pts[2]] * (1 - 0.0004)
    st = None
    for e in pts:
        st = decide(p, {x: fl[x] for x in pts if x <= e})
        if st:
            break
    claim(st and st["last_eval_point"] == pts[3] and st["best_epoch"] == pts[3] and st["decay_epochs"] == max(1, round(p["decay_epochs_max"] * pts[3] / p["constant_epochs_max"])),
          f"dry walk, gain 0.04% at point {pts[3]}: early stop, best {st and st['best_epoch']}, decay {st and st['decay_epochs']}")
    fl[pts[3]] = fl[pts[2]] * 1.001
    st = None
    for e in pts:
        st = decide(p, {x: fl[x] for x in pts if x <= e})
        if st:
            break
    claim(st and st["best_epoch"] == pts[2], f"dry walk, worse at point {pts[3]}: decay from the best point {pts[2]}")
    print("ALL OK" if ok else "CHECK FAILED")
    return 0 if ok else 1


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("init", "plan", "next", "slice", "check"))
    ap.add_argument("--arm", choices=ORDER)
    args = ap.parse_args()
    if args.command in ("next", "slice") and not args.arm:
        ap.error("--arm needed")
    return {"init": init, "plan": cmd_plan, "next": cmd_next, "slice": cmd_slice, "check": check}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
