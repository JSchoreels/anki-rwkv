#!/usr/bin/env python
"""The longer big instant-only teacher (SM113): CPU proofs before any GPU ticket (design.md in this folder; ~1-3 min).

  python -X utf8 artifacts/small-model/sm113/sm113_check.py [--out check-cpu-sm113.json] [--only C4,C5]

The run: the big instant-only teacher (SM110)'s cool-down (from the Small Instant-Only model (SM98)'s epoch 35, the show-time
instant-only big model (SM99)'s show-time targets) stretched over two passes of the 8,000 users: one one_minus_sin decay
over 2R reviews, as two chained folders (sm113_prepare.py). One small source change makes that possible
(scripts/expertium_options.py: an lr_resume "segment" may end short of its total; scripts/expertium_decay.py
resume_point: a resumed folder that is itself a part of a longer decay keeps its place in it).

  C1 code: scripts/ expertium/ rwkv/ == 1e22b62c (the trees SM96 / SM98 / SM99 / SM101 / SM110 trained and scored with)
     except exactly the two files of the SM113 change (6 lines added, 1 removed in each).
  C2 the old paths are unchanged, A/B against 1e22b62c's own two files: resume_point on fake decays (snapshot resume, a
     second interruption, a stop before any snapshot) writes byte-identical protocol.json / resume.json; CatchUp's
     learning rate and KD switch are equal at every probed review for the prepared and resumed protocols of the fake
     decays, of SM98's and of SM110's real cool-downs (SM110's protocol.json is a resumed one); and the existing check
     scripts/check_decay_resume.py passes in this tree.
  C3 the segment schedule (synthetic, R = 2,000 reviews): pass 1 (lr_resume {2R, 0, segment}) and pass 2 (lr_resume
     {2R, R}) give exactly the learning rate of ONE decay over 2R at every review; pass 1 ends at 0.2929 x peak, pass 2
     at 0; a snapshot resume inside either pass keeps that (lr_resume {2R, done} / {2R, R + done}, segment kept in pass
     1 only); a segment that does not fit, or a non-True segment flag, is refused.
  C4 inputs, read only: the teacher sha256; SM98's epoch 35 sha256 == SM98's / SM110's cool-down start; SM98's cool-down
     complete; the targets folder == SM110's (manifest sha256 == SM110's copy; SM110's 8,000-user target check ok); the
     SM110 reference rows in ref/ == wt/sm110's files.
  C5 the real pass-1 folder: sm113_pair.py p1 OK (== SM110's cool-down except lr_resume and paths; == SM98's except also
     the targets); 7 mutated copies refused; its CatchUp: total 2R, lr 7e-4 after the warmup, 0.2929 x 7e-4 at R;
     Muon fresh-state check (gpu/check-muon-fresh-p1.json) passed; label-window check ok.
  C6 pass 2 on a fake pass-1 end (fake result.json + master/optimizer bytes): sm113_prepare.py p2 builds a folder that
     sm113_pair.py p2 accepts; 4 mutated copies refused; its learning rate continues pass 1's (at 0 == the 2R schedule at
     R; 0 at the end).
  C7 rule: sm113_rule.py selftest.
  C8 breaks: sm113_block.py on fixed times (code == sm110_block.py's) and on the real quiet.json now.
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
CODE = "1e22b62c"
W = Path("C:/Users/jonat/workspace/wt")
TRACK = "E89-F65e-F77a"
TEACHER = W / "sm99/artifacts/small-model/sm99-big-inst-e2/expertium/big-inst/master.pth"
TEACHER_SHA = "c49e45d698194da033caa1231f598f875922a4a2a3e921714a3fd433d37e5768"
E35 = W / "sm98/artifacts/small-model/sm98-inst-e35-st"
E35_SHA = "f7577d14ba102ab9ce1a4e995f9176854f1aa719a627209244d54ec02536ebd4"
DEC98 = W / "sm98/artifacts/small-model/sm98-inst-e35st-st-dec-1ep32k-muon"
S110 = W / "sm110/artifacts/small-model/sm110"
DEC110 = W / "sm110/artifacts/small-model/sm110-inst-e35st-st-dec-1ep32k-muon-t99"
REPLAY = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/sm99-big-inst-final8k-neighbour-e2s")
P1 = SM / "sm113-inst-e35st-st-dec2ep32k-t99-p1"
QUIET = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json")
CHANGED = {"scripts/expertium_decay.py": (6, 1), "scripts/expertium_options.py": (6, 1)}
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))


class Report:
    def __init__(self):
        self.checks, self.ok = {}, True

    def check(self, name, ok, **detail):
        self.checks[name] = {"ok": bool(ok), **detail}
        self.ok &= bool(ok)
        print(f"[sm113-check] {name}: {'OK' if ok else 'FAIL'} {json.dumps(detail, default=str)[:400]}", flush=True)


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def old_module(name, rel, tmp):
    """1e22b62c's own copy of a file, loaded as a separate module (its imports resolve to this tree's packages)."""
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
    stat = {}
    for line in git("diff", "--numstat", CODE, "--", "scripts", "expertium", "rwkv").splitlines():
        a, d, f = line.split("\t")
        stat[f] = (int(a), int(d))
    untracked = git("ls-files", "--others", "--exclude-standard", "--", "scripts", "expertium", "rwkv")
    rep.check("C1_code_1e22b62c_plus_the_sm113_change", stat == CHANGED and not untracked, diff=stat,
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
        fake(base / f"sm113-c2-{tag}", users)
    for done, end, snap in (([11, 12], 1040, True), ([13], 1060, True), ([], 0, False)):
        outs = {}
        for tag, mod in (("old", OD), ("new", D)):
            out = base / f"sm113-c2-{tag}"
            interrupt(out, done, end, snap)
            mod.resume_point(out, {"track": trk})
            outs[tag] = {f: (out / f).read_text().replace(f"sm113-c2-{tag}", "X") for f in ("protocol.json", "resume.json", "protocol.prepared.json")}
        eq = outs["old"] == outs["new"]
        same_files &= eq
        p = json.loads((base / "sm113-c2-new" / "protocol.json").read_text())
        a, b = OO.CatchUp({}, p["full_run"], p["rows"]), CatchUp({}, p["full_run"], p["rows"])
        t = sum(r["train_range"][1] - r["train_range"][0] for r in p["rows"])
        lr_same &= lr_probe(a, t) == lr_probe(b, t)
        steps.append({"done": done, "files_equal": eq, "lr_resume": p["full_run"].get("lr_resume")})
    real = {}
    for name, f in (("sm98_prepared", DEC98 / "protocol.prepared.json"), ("sm98_protocol", DEC98 / "protocol.json"),
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
        shutil.rmtree(base / f"sm113-c2-{tag}", ignore_errors=True)
    r = subprocess.run([PY, "-X", "utf8", str(ROOT / "scripts/check_decay_resume.py")], capture_output=True, text=True,
                       cwd=str(ROOT), env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    lines = r.stdout.splitlines()
    passes, fails = sum(x.startswith("PASS") for x in lines), [x for x in lines if x.startswith("FAIL")]
    rep.check("C2_old_paths_unchanged", same_files and lr_same and all(isinstance(real[k], dict) for k in ("sm98_protocol", "sm110_prepared", "sm110_protocol_resumed"))
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
    base = ROOT / "artifacts/small-model/smoke/sm113-c3"
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
    d98 = json.loads((DEC98 / "decay.json").read_text())
    d110 = json.loads((DEC110 / "decay.json").read_text())
    r98 = json.loads((DEC98 / "expertium" / TRACK / "result.json").read_text())
    tchk = json.loads((S110 / "gpu/check-targets-sm110.json").read_text())
    man = sha256(REPLAY / "manifest.json")
    refs = {f: sha256(HERE / "ref" / f) == sha256(S110 / "gpu" / f) for f in ("sm110-rows.jsonl", "sm110-rows-1500.jsonl", "sm110-mid-rows.jsonl")}
    t, e35 = sha256(TEACHER), sha256(E35 / "expertium" / TRACK / "master.pth")
    ok = (t == TEACHER_SHA and e35 == E35_SHA == d98["start_master_sha256"] == d110["start_master_sha256"]
          and r98.get("status") == "complete" and man == sha256(S110 / "gpu/replay-manifest.json")
          and tchk.get("ok") is True and tchk.get("users_ok") == 8000 and all(refs.values())
          and Path(d110["options"]["distillation"]["replay"]).resolve() == REPLAY.resolve())
    rep.check("C4_inputs_pinned", ok, teacher=t[:12], epoch35=e35[:12], sm98_cooldown=r98.get("status"),
              replay_manifest=man[:12], sm110_target_check={k: tchk.get(k) for k in ("ok", "users_ok", "failures_count")},
              ref_rows_equal=refs)


def c5(rep, tmp):
    import sm113_pair
    from scripts.expertium_options import CatchUp
    fail, info = sm113_pair.pair_p1(P1, DEC110, DEC98)
    neg = {}

    def mutate(name, fn):
        d = Path(tmp) / f"mut-p1-{name}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
        p = json.loads((P1 / "protocol.json").read_text())
        y = json.loads((P1 / "decay.json").read_text())
        fn(p, y)
        (d / "protocol.json").write_text(json.dumps(p)); (d / "decay.json").write_text(json.dumps(y))
        f2, _ = sm113_pair.pair_p1(d, DEC110, DEC98)
        neg[name] = bool(f2)
    mutate("lr_offset", lambda p, y: p["full_run"]["lr_resume"].update(reviews_offset=1))
    mutate("no_segment", lambda p, y: p["full_run"]["lr_resume"].pop("segment"))
    mutate("old_targets", lambda p, y: y["options"]["distillation"].update(replay=json.loads((DEC98 / "decay.json").read_text())["options"]["distillation"]["replay"],
                                                                            teacher=json.loads((DEC98 / "decay.json").read_text())["options"]["distillation"]["teacher"]))
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
    import sm113_pair
    from scripts.expertium_options import CatchUp
    base = ROOT / "artifacts/small-model/smoke"
    a, b = base / "sm113-c6-p1", base / "sm113-c6-p2"
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
    json.dump({"status": "complete", "update_offset_end": 790000, "review_presentations": R,
               "trained_parameter_sha256": "f" * 64}, open(case / "result.json", "w"))
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm113_prepare.py"), "p2", "--p1", str(a), "--out", str(b)],
                       capture_output=True, text=True)
    fail, info = sm113_pair.pair_p2(b, a) if r.returncode == 0 else (["prepare failed: " + r.stderr[-300:]], "")
    neg = {}

    def mutate(name, fn):
        d = base / f"sm113-c6-mut-{name}"
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(b, d)
        q = json.loads((d / "protocol.json").read_text()); y = json.loads((d / "decay.json").read_text())
        fn(q, y)
        (d / "protocol.json").write_text(json.dumps(q)); (d / "decay.json").write_text(json.dumps(y))
        f2, _ = sm113_pair.pair_p2(d, a)
        neg[name] = bool(f2)
        shutil.rmtree(d, ignore_errors=True)
    mutate("same_order", lambda q, y: q.update(rows=json.loads((a / "protocol.json").read_text())["rows"]))
    mutate("lr_offset", lambda q, y: q["full_run"]["lr_resume"].update(reviews_offset=0))
    mutate("other_start", lambda q, y: q["full_run"].update(init_master=str(E35 / "expertium" / TRACK / "master.pth")))
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
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm113_rule.py"), "selftest"], capture_output=True, text=True)
    rep.check("C7_rule_selftest", r.returncode == 0 and r.stdout.strip().endswith("OK"), out=r.stdout.strip()[-300:])


def c8(rep, tmp):
    def block(now, quiet):
        return subprocess.run([PY, "-X", "utf8", str(HERE / "sm113_block.py"), "--now", now, "--quiet", str(quiet)],
                              capture_output=True, text=True).stdout.strip()
    empty = Path(tmp) / "quiet-empty.json"
    empty.write_text("[]", encoding="utf-8")
    got = {"fri_16:30": block("2026-11-13T16:30", empty), "fri_20:30": block("2026-11-13T20:30", empty),
           "sat_01:00": block("2026-11-14T01:00", empty), "sun_01:00": block("2026-11-15T01:00", empty),
           "sun_16:30": block("2026-11-15T16:30", empty), "real_now": block(time.strftime("%Y-%m-%dT%H:%M"), QUIET)}
    want = {"fri_16:30": "NEXT 210", "fri_20:30": "NOW 16260", "sat_01:00": "NEXT 1140", "sun_01:00": "NEXT 900",
            "sun_16:30": "NOW 30660"}
    src = (HERE / "sm113_block.py").read_text(encoding="utf-8").split('"""\nimport argparse', 1)[1]
    ref = (S110 / "sm110_block.py").read_text(encoding="utf-8").split('"""\nimport argparse', 1)[1]
    rep.check("C8_breaks", all(got[k] == v for k, v in want.items()) and src == ref and got["real_now"][:4] in ("NEXT", "NOW "),
              got=got, code_equals_sm110_block=src == ref)


def c9(rep):
    free = shutil.disk_usage("C:/").free / 2 ** 30
    rep.check("C9_disk", free >= 5, free_gib=round(free, 1))


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "check-cpu-sm113.json"))
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
    print(f"[sm113-check] {'ALL OK' if out['all_ok'] else ('partial ' + ('OK' if rep.ok else 'FAILED'))} in {out['minutes']} min")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
