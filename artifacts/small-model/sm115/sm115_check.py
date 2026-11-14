#!/usr/bin/env python
"""The longer cool-down for Tiny (SM115): CPU proofs before any GPU ticket (design.md in this folder; ~1-3 min).

  python -X utf8 artifacts/small-model/sm115/sm115_check.py [--out check-cpu-sm115.json] [--only C4,C5]

The run: the longer big instant-only teacher (SM113)'s recipe (one one_minus_sin decay over 2R reviews as two chained
folders, the show-time instant-only big model (SM99)'s show-time targets) applied to the Tiny Instant-Only model (SM101):
start = SM101's constant-LR epoch 38 (w64-c1), where SM101's own one-epoch cool-down started. No source change: the
code is SM113's launch tree.

  C1 code: scripts/ expertium/ rwkv/ == d99fe356 (SM113's launch commit) except the commit-id strings the history
     rewrite follow-up (f561e1c0) changed in scripts/sm91s2.py and scripts/sm91s3.py (not imported by the trainer or the
     scorer); and == c99330a7 (the tree SM101 trained and scored with) except exactly the two files of the SM113 change
     (6 lines added, 1 removed in each) and those two files.
  C2 the old paths are unchanged, A/B against c99330a7's own two files (SM113's C2, re-run here): resume_point on fake
     decays writes byte-identical protocol.json / resume.json; CatchUp's learning rate and KD switch are equal at every
     probed review for the fake decays and for SM101's, SM98's and SM110's real cool-downs (SM101's protocol.json was
     resumed twice); scripts/check_decay_resume.py passes in this tree.
  C3 the segment schedule (synthetic, R = 2,000 reviews), SM113's C3 re-run: pass 1 + pass 2 == one 2R decay at every
     review, also across a snapshot resume inside either pass; bad segments refused.
  C4 inputs, read only: the teacher sha256; SM101's epoch 38 sha256 == SM101's cool-down start; SM101's cool-down
     complete; the targets folder == SM113's pass 1's == SM110's (manifest sha256 == SM110's copy; SM110's 8,000-user
     target check ok); the early-stop reference: SM101's cool-down snapshot after 4,094 users (attempt-2 users-01200)
     master sha256 pinned, its cumulative reviews 290,914,508 and its last update's learning rate 0.2931 x peak.
  C5 the real pass-1 folder: sm115_pair.py p1 OK (== SM101's cool-down except lr_resume, the targets and paths; == SM113's
     pass 1 except the start and the architecture); 9 mutated copies refused; its CatchUp: total 2R, lr 7e-4 after the
     warmup, 0.2929 x 7e-4 at R; Muon fresh-state check (gpu/check-muon-fresh-p1.json) passed; label-window check ok.
  C6 pass 2 on a fake pass-1 end (fake result.json + master/optimizer bytes): sm115_prepare.py p2 builds a folder that
     sm115_pair.py p2 accepts; 4 mutated copies refused; its learning rate continues pass 1's.
  C7 rule: sm115_rule.py selftest.
  C8 breaks: sm115_block.py on fixed times (code == sm113_block.py's == sm110_block.py's) and on the real quiet.json now.
  C9 disk: >= 5 GB free on C: (two cool-downs of snapshots, ~3 MB each).
"""
from argparse import ArgumentParser
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SM = ROOT / "artifacts/small-model"
PY = sys.executable
CODE = "c99330a7"
W = Path("C:/Users/jonat/workspace/wt")
TRACK = "E89-F65e-F77a"
TEACHER = W / "sm99/artifacts/small-model/sm99-big-inst-e2/expertium/big-inst/master.pth"
TEACHER_SHA = "c49e45d698194da033caa1231f598f875922a4a2a3e921714a3fd433d37e5768"
E38 = W / "sm101/artifacts/small-model/sm101-c1-e38-st"
E38_SHA = "e6ea69124af00a3dcf2325047ae34580aaf96051d963ff6130dd06bde02d396c"
DEC101 = W / "sm101/artifacts/small-model/sm101-c1-e38st-st-dec-1ep32k-muon"
SNAP101 = DEC101 / "attempts/attempt-2/expertium" / TRACK / "snapshots/users-01200"
SNAP101_SHA = "3ab9e629e7d740cbd4b3f8c8f40f680e028dd6fbd7790c25e83e083f780a9c35"
DEC98 = W / "sm98/artifacts/small-model/sm98-inst-e35st-st-dec-1ep32k-muon"
S110 = W / "sm110/artifacts/small-model/sm110"
DEC110 = W / "sm110/artifacts/small-model/sm110-inst-e35st-st-dec-1ep32k-muon-t99"
P113 = W / "sm113/artifacts/small-model/sm113-inst-e35st-st-dec2ep32k-t99-p1"
REPLAY = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/sm99-big-inst-final8k-neighbour-e2s")
P1 = SM / "sm115-c1-e38st-st-dec2ep32k-t99-p1"
QUIET = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json")
LAUNCH113 = "d99fe356"
REWRITE = {"scripts/sm91s2.py": (5, 5), "scripts/sm91s3.py": (2, 2)}
CHANGED = {"scripts/expertium_decay.py": (6, 1), "scripts/expertium_options.py": (6, 1), **REWRITE}
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))


class Report:
    def __init__(self):
        self.checks, self.ok = {}, True

    def check(self, name, ok, **detail):
        self.checks[name] = {"ok": bool(ok), **detail}
        self.ok &= bool(ok)
        print(f"[sm115-check] {name}: {'OK' if ok else 'FAIL'} {json.dumps(detail, default=str)[:400]}", flush=True)


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def old_module(name, rel, tmp):
    """c99330a7's own copy of a file, loaded as a separate module (its imports resolve to this tree's packages)."""
    src = git("show", f"{CODE}:{rel}")
    path = Path(tmp) / f"{name}.py"
    path.write_text(src + "\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "ROOT"):
        mod.ROOT = ROOT   # the copy lives in a temp folder; its paths are this tree's
    return mod


def c1(rep):
    def numstat(ref):
        out = {}
        for line in git("diff", "--numstat", ref, "--", "scripts", "expertium", "rwkv").splitlines():
            a, d, f = line.split("\t")
            out[f] = (int(a), int(d))
        return out
    s113, s99330 = numstat(LAUNCH113), numstat(CODE)
    ids = {"45a56e3a", "c4fc3b84", "fa5a9f25", "d8b1f2c5", "4f39eb92", "63029f40"}
    changed = [l for l in git("diff", "-U0", LAUNCH113, "--", *REWRITE).splitlines()
               if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    rewrite_only = bool(changed) and all(any(i in l for i in ids) for l in changed)
    untracked = git("ls-files", "--others", "--exclude-standard", "--", "scripts", "expertium", "rwkv")
    rep.check("C1_code_sm113_launch_tree", s113 == REWRITE and rewrite_only and s99330 == CHANGED and not untracked,
              vs_sm113_launch=s113, rewrite_lines_only_commit_ids=rewrite_only, vs_c99330a7=s99330,
              untracked=untracked.splitlines()[:5])


def lr_probe(catch, total, points=64):
    out = []
    for k in range(points + 1):
        r = min(total, (total * k) // points)
        out.append((catch.learning_rate(100, 1000, r), bool(catch.schedule.decaying(r + catch.reviews_offset)) if catch.schedule else None))
    return out


def c2(rep, tmp):
    import scripts.expertium_decay as D
    from scripts.expertium_options import CatchUp
    OD = old_module("old_expertium_decay", "scripts/expertium_decay.py", tmp)
    OO = old_module("old_expertium_options", "scripts/expertium_options.py", tmp)
    trk = "check-track"

    def fake(out, users):
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        rows = [{"user_id": u, "available_reviews": n, "train_range": [0, n], "validation_range": [0, n]} for u, n in users]
        fr = {"epoch": 10, "evaluate": False, "init_master": "source/master.pth", "update_offset": 1000,
              "lr_schedule": {"decay_start": 0.0, "decay_end": 1.0, "shape": "one_minus_sin", "final": 0.0},
              "optimizer_state": "fresh", "snapshot_users": 2, "control_dir": (out / "control").relative_to(ROOT).as_posix()}
        D.atomic_json(out / "decay.json", {"track": trk, "options": {}})
        D.atomic_json(out / "protocol.json", {"rows": rows, "evaluation_rows": [], "full_run": fr})

    def interrupt(out, done, end, snapshot=True):
        case = out / D.MODEL / trk
        case.mkdir(parents=True)
        D.atomic_json(case / "result.json", {"status": "stopped"})
        if snapshot:
            f = case / "snapshots" / f"users-{len(done):05d}"
            f.mkdir(parents=True)
            (f / "master.pth").write_bytes(b"m"); (f / "optimizer.pth").write_bytes(b"o")
            meta = {"reason": "periodic", "done_users": done, "update_offset_end": end, "folder": f.relative_to(ROOT).as_posix()}
            D.atomic_json(f / "snapshot.json", meta); D.atomic_json(case / "snapshots" / "latest.json", meta)

    base = ROOT / "artifacts/small-model/smoke"
    same_files, lr_same, steps = True, True, []
    users = [(11, 300), (12, 500), (13, 200), (14, 400), (15, 600)]
    for tag, mod in (("old", OD), ("new", D)):
        fake(base / f"sm115-c2-{tag}", users)
    for done, end, snap in (([11, 12], 1040, True), ([13], 1060, True), ([], 0, False)):
        outs = {}
        for tag, mod in (("old", OD), ("new", D)):
            out = base / f"sm115-c2-{tag}"
            interrupt(out, done, end, snap)
            mod.resume_point(out, {"track": trk})
            outs[tag] = {f: (out / f).read_text().replace(f"sm115-c2-{tag}", "X") for f in ("protocol.json", "resume.json", "protocol.prepared.json")}
        eq = outs["old"] == outs["new"]
        same_files &= eq
        p = json.loads((base / "sm115-c2-new" / "protocol.json").read_text())
        a, b = OO.CatchUp({}, p["full_run"], p["rows"]), CatchUp({}, p["full_run"], p["rows"])
        t = sum(r["train_range"][1] - r["train_range"][0] for r in p["rows"])
        lr_same &= lr_probe(a, t) == lr_probe(b, t)
        steps.append({"done": done, "files_equal": eq, "lr_resume": p["full_run"].get("lr_resume")})
    real = {}
    for name, f in (("sm101_prepared", DEC101 / "protocol.prepared.json"), ("sm101_protocol_resumed_twice", DEC101 / "protocol.json"),
                    ("sm98_prepared", DEC98 / "protocol.prepared.json"), ("sm98_protocol", DEC98 / "protocol.json"),
                    ("sm110_prepared", DEC110 / "protocol.prepared.json"), ("sm110_protocol_resumed", DEC110 / "protocol.json")):
        if not f.exists():
            real[name] = "missing"; continue
        p = json.loads(f.read_text())
        a, b = OO.CatchUp({}, p["full_run"], p["rows"]), CatchUp({}, p["full_run"], p["rows"])
        t = sum(r["train_range"][1] - r["train_range"][0] for r in p["rows"])
        real[name] = {"equal": lr_probe(a, t, 200) == lr_probe(b, t, 200), "lr_resume": p["full_run"].get("lr_resume"),
                      "total": b.schedule.total}
        lr_same &= real[name]["equal"]
    for tag in ("old", "new"):
        shutil.rmtree(base / f"sm115-c2-{tag}", ignore_errors=True)
    r = subprocess.run([PY, "-X", "utf8", str(ROOT / "scripts/check_decay_resume.py")], capture_output=True, text=True,
                       cwd=str(ROOT), env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    lines = r.stdout.splitlines()
    passes, fails = sum(x.startswith("PASS") for x in lines), [x for x in lines if x.startswith("FAIL")]
    rep.check("C2_old_paths_unchanged", same_files and lr_same and all(isinstance(real[k], dict) for k in ("sm101_prepared", "sm101_protocol_resumed_twice", "sm98_protocol", "sm110_prepared", "sm110_protocol_resumed"))
              and r.returncode == 0 and passes > 10 and not fails, steps=steps, real=real,
              check_decay_resume={"exit": r.returncode, "pass": passes, "fail": fails[:5]})


def c3(rep):
    import scripts.expertium_decay as D
    from scripts.expertium_options import CatchUp
    sched = {"decay_start": 0.0, "decay_end": 1.0, "shape": "one_minus_sin", "final": 0.0}
    rows = [{"user_id": u, "train_range": [0, n]} for u, n in ((1, 700), (2, 500), (3, 800))]
    R = 2000
    whole = CatchUp({}, {"lr_schedule": sched}, rows + [{"user_id": u + 10, "train_range": [0, n]} for u, n in ((1, 700), (2, 500), (3, 800))])
    p1 = CatchUp({}, {"lr_schedule": sched, "lr_resume": {"total_reviews": 2 * R, "reviews_offset": 0, "segment": True}}, rows)
    p2 = CatchUp({}, {"lr_schedule": sched, "lr_resume": {"total_reviews": 2 * R, "reviews_offset": R}}, rows)
    ok1 = all(p1.learning_rate(50, 1000, r) == whole.learning_rate(50, 1000, r) for r in range(0, R + 1, 25))
    ok2 = all(p2.learning_rate(50, 1000, r) == whole.learning_rate(50, 1000, R + r) for r in range(0, R + 1, 25))
    end1, end2 = p1.learning_rate(50, 1000, R), p2.learning_rate(50, 1000, R)
    kd = all(p.schedule.decaying(0 + p.reviews_offset) for p in (p1, p2))
    # a snapshot resume inside each pass (resume_point on a fake folder)
    base = ROOT / "artifacts/small-model/smoke/sm115-c3"
    res = {}
    for tag, lr in (("p1", {"total_reviews": 2 * R, "reviews_offset": 0, "segment": True}), ("p2", {"total_reviews": 2 * R, "reviews_offset": R})):
        out = base / tag
        shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
        fr = {"lr_schedule": sched, "lr_resume": lr, "update_offset": 5000, "init_master": "x", "snapshot_users": 1,
              "control_dir": (out / "control").relative_to(ROOT).as_posix()}
        D.atomic_json(out / "decay.json", {"track": "t", "options": {}})
        D.atomic_json(out / "protocol.json", {"rows": rows, "evaluation_rows": [], "full_run": fr})
        case = out / D.MODEL / "t"; f = case / "snapshots/users-00001"; f.mkdir(parents=True)
        D.atomic_json(case / "result.json", {"status": "stopped"})
        (f / "master.pth").write_bytes(b"m"); (f / "optimizer.pth").write_bytes(b"o")
        meta = {"done_users": [1], "update_offset_end": 5100, "folder": f.relative_to(ROOT).as_posix()}
        D.atomic_json(f / "snapshot.json", meta); D.atomic_json(case / "snapshots/latest.json", meta)
        D.resume_point(out, {"track": "t"})
        p = json.loads((out / "protocol.json").read_text())
        c = CatchUp({}, p["full_run"], p["rows"])
        off = lr["reviews_offset"]
        res[tag] = {"lr_resume": p["full_run"]["lr_resume"], "rows": [x["user_id"] for x in p["rows"]],
                    "lr_equal": all(c.learning_rate(50, 1000, r) == whole.learning_rate(50, 1000, off + 700 + r) for r in range(0, 1301, 25))}
    shutil.rmtree(base, ignore_errors=True)
    refused = {}
    for name, lr, rr in (("segment_too_long", {"total_reviews": R + 100, "reviews_offset": 200, "segment": True}, rows),
                         ("segment_flag_not_true", {"total_reviews": 2 * R, "reviews_offset": 0, "segment": "yes"}, rows),
                         ("no_segment_short", {"total_reviews": 2 * R, "reviews_offset": 0}, rows),
                         ("negative_offset", {"total_reviews": 2 * R, "reviews_offset": -1, "segment": True}, rows)):
        try:
            CatchUp({}, {"lr_schedule": sched, "lr_resume": lr}, rr); refused[name] = False
        except ValueError:
            refused[name] = True
    ok = (ok1 and ok2 and abs(end1 / 7e-4 - (1 - math.sin(math.pi / 4))) < 1e-12 and end2 == 0.0 and kd
          and res["p1"]["lr_resume"] == {"total_reviews": 2 * R, "reviews_offset": 700, "segment": True}
          and res["p2"]["lr_resume"] == {"total_reviews": 2 * R, "reviews_offset": R + 700}
          and all(v["lr_equal"] and v["rows"] == [2, 3] for v in res.values()) and all(refused.values()))
    rep.check("C3_segment_schedule", ok, pass1_equal=ok1, pass2_equal=ok2, pass1_end_lr_over_peak=end1 / 7e-4,
              pass2_end_lr=end2, resumes=res, refused=refused)


def c4(rep):
    d101 = json.loads((DEC101 / "decay.json").read_text())
    d113 = json.loads((P113 / "decay.json").read_text())
    d110 = json.loads((DEC110 / "decay.json").read_text())
    r101 = json.loads((DEC101 / "expertium" / TRACK / "result.json").read_text())
    tchk = json.loads((S110 / "gpu/check-targets-sm110.json").read_text())
    man = sha256(REPLAY / "manifest.json")
    t, e38 = sha256(TEACHER), sha256(E38 / "expertium" / TRACK / "master.pth")
    snap = json.loads((SNAP101 / "snapshot.json").read_text())
    stop1 = json.loads((DEC101 / "attempts/attempt-1/expertium" / TRACK / "snapshots/users-02894/snapshot.json").read_text())
    upd = [json.loads(x) for x in open(DEC101 / "attempts/attempt-2/expertium" / TRACK / "updates.jsonl", encoding="utf-8") if x.strip()]
    n_upd = int(snap["updates"])
    lr_snap = upd[n_upd - 1]["lr"] / 7e-4
    cum = int(stop1["review_presentations"]) + int(snap["review_presentations"])
    total = int(d101["training_reviews_after_trims"])
    done = set(stop1["done_users"]) | set(snap["done_users"])   # attempt 1 (stopped for the break) + attempt 2's own users
    snap_ok = (sha256(SNAP101 / "master.pth") == SNAP101_SHA and len(done) == 4094 == len(stop1["done_users"]) + len(snap["done_users"])
               and stop1["reason"] == "stop" and len(stop1["done_users"]) == 2894 and cum == 290_914_508
               and abs(lr_snap - 0.29306) < 1e-4 and abs(lr_snap - (1 - math.sin(math.pi / 2 * cum / total))) < 2e-4)
    ok = (t == TEACHER_SHA and e38 == E38_SHA == d101["start_master_sha256"] and r101.get("status") == "complete"
          and man == sha256(S110 / "gpu/replay-manifest.json") and tchk.get("ok") is True and tchk.get("users_ok") == 8000
          and Path(d113["options"]["distillation"]["replay"]).resolve() == REPLAY.resolve()
          == Path(d110["options"]["distillation"]["replay"]).resolve() and snap_ok)
    rep.check("C4_inputs_pinned", ok, teacher=t[:12], epoch38=e38[:12], sm101_cooldown=r101.get("status"),
              replay_manifest=man[:12], sm110_target_check={k: tchk.get(k) for k in ("ok", "users_ok", "failures_count")},
              early_stop_reference={"snapshot": SNAP101.relative_to(DEC101).as_posix(), "master": SNAP101_SHA[:12],
                                    "users_done": len(done), "reviews_done": cum, "of": total,
                                    "fraction": round(cum / total, 5), "last_update_lr_over_peak": round(lr_snap, 5),
                                    "formula_lr_over_peak": round(1 - math.sin(math.pi / 2 * cum / total), 5), "ok": snap_ok})


def c5(rep, tmp):
    import sm115_pair
    from scripts.expertium_options import CatchUp
    fail, info = sm115_pair.pair_p1(P1, DEC101, P113)
    neg = {}

    def mutate(name, fn):
        d = Path(tmp) / f"mut-p1-{name}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
        p = json.loads((P1 / "protocol.json").read_text())
        y = json.loads((P1 / "decay.json").read_text())
        fn(p, y)
        (d / "protocol.json").write_text(json.dumps(p)); (d / "decay.json").write_text(json.dumps(y))
        f2, _ = sm115_pair.pair_p1(d, DEC101, P113)
        neg[name] = bool(f2)
    mutate("lr_offset", lambda p, y: p["full_run"]["lr_resume"].update(reviews_offset=1))
    mutate("no_segment", lambda p, y: p["full_run"]["lr_resume"].pop("segment"))
    mutate("old_targets", lambda p, y: y["options"]["distillation"].update(replay=json.loads((DEC101 / "decay.json").read_text())["options"]["distillation"]["replay"],
                                                                            teacher=json.loads((DEC101 / "decay.json").read_text())["options"]["distillation"]["teacher"]))
    mutate("arch_w64", lambda p, y: (y.update(architecture="w64"), y["options"].update(architecture="w64")))
    mutate("sm113_start", lambda p, y: (p["full_run"].update(init_master=json.loads((P113 / "protocol.prepared.json").read_text())["full_run"]["init_master"]),
                                        y.update(start_master_sha256=json.loads((P113 / "decay.json").read_text())["start_master_sha256"])))
    mutate("decay_alpha", lambda p, y: y["options"]["distillation"].update(decay_alpha=0.6))
    mutate("curve_weight", lambda p, y: y["options"].update(curve_loss_weight=1.0))
    mutate("order", lambda p, y: p["rows"].insert(0, p["rows"].pop(5)))
    mutate("start_sha", lambda p, y: y.update(start_master_sha256="0" * 64))
    p = json.loads((P1 / "protocol.json").read_text())
    c = CatchUp(json.loads((P1 / "decay.json").read_text())["options"], p["full_run"], p["rows"])
    R = sum(r["train_range"][1] - r["train_range"][0] for r in p["rows"])
    lr0, lrR = c.learning_rate(10, p["full_run"]["update_offset"], 0), c.learning_rate(10, p["full_run"]["update_offset"], R)
    muon = (HERE / "gpu/check-muon-fresh-p1.json")
    mu = json.loads(muon.read_text()) if muon.exists() else {}
    lw = json.loads((HERE / "gpu/check-replay-label-windows-p1.json").read_text())
    mu_ok = mu.get("passed", mu.get("ok", mu.get("all_ok"))) is True
    ok = (not fail and all(neg.values()) and c.schedule.total == 2 * R == 1_163_885_448 and lr0 == 7e-4
          and abs(lrR / 7e-4 - (1 - math.sin(math.pi / 4))) < 1e-9 and mu_ok and lw.get("ok") is True)
    rep.check("C5_pass1_folder", ok, pair=fail[:4] or info, mutated_refused=neg, schedule_total=c.schedule.total,
              lr_start=lr0, lr_at_R_over_peak=lrR / 7e-4, muon_fresh=mu_ok, label_windows=lw.get("ok"),
              trims=[t["user_id"] for t in json.loads((P1 / "decay.json").read_text()).get("trimmed_unlabelled_tails", [])])


def c6(rep, tmp):
    import sm115_pair
    from scripts.expertium_options import CatchUp
    base = ROOT / "artifacts/small-model/smoke"
    a, b = base / "sm115-c6-p1", base / "sm115-c6-p2"
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)
    a.mkdir(parents=True)
    for f in ("protocol.json", "decay.json"):
        shutil.copy(P1 / f, a / f)
    case = a / "expertium" / TRACK
    case.mkdir(parents=True)
    p = json.loads((a / "protocol.json").read_text())
    p["full_run"]["control_dir"] = (a / "control").relative_to(ROOT).as_posix()   # the copy's own control folder
    (a / "protocol.json").write_text(json.dumps(p, indent=1), encoding="utf-8")
    R = sum(r["train_range"][1] - r["train_range"][0] for r in p["rows"])
    (case / "master.pth").write_bytes(b"fake master"); (case / "optimizer.pth").write_bytes(b"fake optimizer")
    json.dump({"status": "complete", "update_offset_end": 856000, "review_presentations": R,
               "trained_parameter_sha256": "f" * 64}, open(case / "result.json", "w"))
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm115_prepare.py"), "p2", "--p1", str(a), "--out", str(b)],
                       capture_output=True, text=True)
    fail, info = sm115_pair.pair_p2(b, a) if r.returncode == 0 else (["prepare failed: " + r.stderr[-300:]], "")
    neg = {}

    def mutate(name, fn):
        d = base / f"sm115-c6-mut-{name}"
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(b, d)
        q = json.loads((d / "protocol.json").read_text()); y = json.loads((d / "decay.json").read_text())
        fn(q, y)
        (d / "protocol.json").write_text(json.dumps(q)); (d / "decay.json").write_text(json.dumps(y))
        f2, _ = sm115_pair.pair_p2(d, a)
        neg[name] = bool(f2)
        shutil.rmtree(d, ignore_errors=True)
    mutate("same_order", lambda q, y: q.update(rows=json.loads((a / "protocol.json").read_text())["rows"]))
    mutate("lr_offset", lambda q, y: q["full_run"]["lr_resume"].update(reviews_offset=0))
    mutate("other_start", lambda q, y: q["full_run"].update(init_master=str(E38 / "expertium" / TRACK / "master.pth")))
    mutate("teacher", lambda q, y: y["options"]["distillation"].update(decay_alpha=0.9))
    ok = r.returncode == 0 and not fail and all(neg.values())
    lr = {}
    if ok:
        q = json.loads((b / "protocol.json").read_text())
        c2 = CatchUp(json.loads((b / "decay.json").read_text())["options"], q["full_run"], q["rows"])
        c1_ = CatchUp(json.loads((a / "decay.json").read_text())["options"], p["full_run"], p["rows"])
        off = q["full_run"]["update_offset"]
        lr = {"p1_at_R": c1_.learning_rate(10, off, R), "p2_at_0": c2.learning_rate(0, off, 0), "p2_end": c2.learning_rate(0, off, R),
              "p2_mid_vs_formula": c2.learning_rate(0, off, R // 2) / 7e-4 - (1 - math.sin(math.pi / 2 * 0.75))}
        ok = lr["p1_at_R"] == lr["p2_at_0"] and lr["p2_end"] == 0.0 and abs(lr["p2_mid_vs_formula"]) < 1e-9
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)
    rep.check("C6_pass2_prepare_and_pair", ok, prepare=r.stdout.strip()[-300:], pair=fail[:4] or info, mutated_refused=neg, lr=lr)


def c7(rep):
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm115_rule.py"), "selftest"], capture_output=True, text=True)
    rep.check("C7_rule_selftest", r.returncode == 0 and r.stdout.strip().endswith("OK"), out=r.stdout.strip()[-300:])


def c8(rep, tmp):
    def block(now, quiet):
        return subprocess.run([PY, "-X", "utf8", str(HERE / "sm115_block.py"), "--now", now, "--quiet", str(quiet)],
                              capture_output=True, text=True).stdout.strip()
    empty = Path(tmp) / "quiet-empty.json"
    empty.write_text("[]", encoding="utf-8")
    got = {"fri_16:30": block("2026-11-13T16:30", empty), "fri_20:30": block("2026-11-13T20:30", empty),
           "sat_01:00": block("2026-11-14T01:00", empty), "sun_01:00": block("2026-11-15T01:00", empty),
           "sun_16:30": block("2026-11-15T16:30", empty), "sat_15:30": block("2026-11-14T15:30", empty),
           "sun_15:00": block("2026-11-15T15:00", empty), "real_now": block(time.strftime("%Y-%m-%dT%H:%M"), QUIET)}
    want = {"fri_16:30": "NEXT 210", "fri_20:30": "NOW 16260", "sat_01:00": "NEXT 1140", "sun_01:00": "NEXT 900",
            "sun_16:30": "NOW 30660", "sat_15:30": "NEXT 270", "sun_15:00": "NEXT 60"}
    src = (HERE / "sm115_block.py").read_text(encoding="utf-8").split('"""\nimport argparse', 1)[1]
    ref = (S110 / "sm110_block.py").read_text(encoding="utf-8").split('"""\nimport argparse', 1)[1]
    rep.check("C8_breaks", all(got[k] == v for k, v in want.items()) and src == ref and got["real_now"][:4] in ("NEXT", "NOW "),
              got=got, code_equals_sm110_block=src == ref)


def c9(rep):
    free = shutil.disk_usage("C:/").free / 2 ** 30
    rep.check("C9_disk", free >= 5, free_gib=round(free, 1))


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "check-cpu-sm115.json"))
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    rep = Report()
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        for name, fn in (("C1", lambda: c1(rep)), ("C2", lambda: c2(rep, tmp)), ("C3", lambda: c3(rep)), ("C4", lambda: c4(rep)),
                         ("C5", lambda: c5(rep, tmp)), ("C6", lambda: c6(rep, tmp)), ("C7", lambda: c7(rep)),
                         ("C8", lambda: c8(rep, tmp)), ("C9", lambda: c9(rep))):
            if only is None or name in only:
                try:
                    fn()
                except Exception as error:   # a crashed check is a failed check
                    import traceback
                    rep.check(f"{name}_crashed", False, error=repr(error)[:300], where=traceback.format_exc()[-500:])
    out = {"commit": git("rev-parse", "--short", "HEAD"), "minutes": round((time.perf_counter() - t0) / 60, 1),
           "all_ok": rep.ok and only is None, "partial": sorted(only) if only else None, "checks": rep.checks}
    if only is None:
        Path(args.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"[sm115-check] {'ALL OK' if out['all_ok'] else ('partial ' + ('OK' if rep.ok else 'FAILED'))} in {out['minutes']} min")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
