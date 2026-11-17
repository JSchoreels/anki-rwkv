#!/usr/bin/env python
"""The show-time two-head cool-down for Tiny (SM124): CPU proofs before the launch (design.md in this folder; ~5-10 min).
The checks are the show-time two-head cool-down for Small (SM118)'s sm118_check.py with the start, the reference and the
architecture changed to the Tiny two-head model (SM106)'s, and the export checks replaced by checks of the reused
targets.

  python -X utf8 artifacts/small-model/sm124/sm124_check.py [--out check-cpu-sm124.json] [--only C4,C5]

The run: the recipe of the show-time two-head cool-down for Small (SM118) (one one_minus_sin decay over 2R reviews as two
chained folders) applied to the Tiny two-head model: start = the Tiny two-head model (SM106)'s constant-LR epoch 29
(w64-c1), where SM106's own one-epoch cool-down started; both heads trained; the show-time big model (SM91s2)'s
show-time targets for the 8,000 cool-down users (exported by SM118 on 11-16, reused read only). No source change: the
code is SM115's launch tree (= SM118's).

  C1 code: scripts/ expertium/ rwkv/ == 112c4082 (SM115's launch commit, SM118's tree) and nothing uncommitted there;
     against c99330a7 (the tree SM106 trained and scored with) exactly these files differ: the 2R segment schedule of
     the longer big instant-only teacher (SM113) (expertium_decay.py, expertium_options.py) and the history-rewrite
     commit-id strings (sm91s2.py, sm91s3.py); wt/sm106 == c99330a7 except the two rewrite files, clean.
  C2 the old paths are unchanged, A/B against SM106's tree:
     C2a trainer slice on CPU (sm124_probe.py = SM118's probe) in wt/sm106 and in this tree: 2 small users of SM106's
         cool-down, from SM106's epoch 29 (w64-c1), SM106's cool-down options (show-time inputs, fresh Muon), with and
         without KD from SM106's replay targets: per update loss bits, instant bits, every gradient, final weights bit for
         bit; plus, in this tree only, the same slice with the new targets (finite, the curve head trains, the loss
         differs from the old targets').
     C2b CatchUp's learning rate / KD switch and resume_point's files, c99330a7's expertium_decay.py / expertium_options.py
         vs this tree's, on fake decays and on SM106's real cool-down, the recipe and pass 1; check_decay_resume.py.
  C3 the segment schedule (synthetic, R = 2,000 reviews), SM118's C3: pass 1 + pass 2 == one 2R decay at every review,
     also across a snapshot resume inside either pass; bad segments refused.
  C4 inputs, read only: the teacher sha256; SM106's epoch 29 sha256 == SM106's cool-down start; SM106's cool-down
     complete in one attempt, its final master sha256 pinned; the early-stop reference: SM106's cool-down snapshot
     users-04000 (master sha256 pinned, 4,000 users, 284,944,410 reviews, last update's learning rate 0.3045 x peak);
     the 8k run file's users == SM106's cool-down users.
  C5 the reused targets: `sm11_replay_export.py status` complete for the 8k run on show-time inputs; manifest == the
     one SM118 committed (gpu/replay-manifest.json of sm118); SM118's targets check (all 8,000 users, ok) committed; no
     target file changed after SM118's pass 1 started (mtimes); SM118's `targets` check run again now on all 8,000 users
     (shape, NaN pattern, finite, rating sums vs the old targets; the 1,991 users of SM91 step 3 within tolerance).
  C6 folders: the recipe (this tree's sm77_warm cooldown from SM106's epoch 29 with SM106's own extra options, never
     trained) == SM106's cool-down except paths (sm124_pair.py recipe); pass 1 (prepared) passes sm124_pair.py p1 (vs
     SM106's cool-down and SM118's pass 1); SM106's cool-down as pass 1 refused; the smoke folder == pass 1 except
     control_dir and its note; 11 mutated copies refused; CatchUp total 2R, 7e-4 after the warm-up, 0.2929 x 7e-4 at R;
     Muon fresh-state check; label-window checks of the recipe and pass 1 (trim 5599); the GPU smoke of the smoke folder
     passed (gpu/smoke0.out).
  C7 pass 2 on a fake pass-1 end (fake result.json + master/optimizer bytes): sm124_prepare.py p2 builds a folder that
     sm124_pair.py p2 accepts; 4 mutated copies refused; its learning rate continues pass 1's.
  C8 rule: sm124_rule.py selftest.
  C9 the scorer on CPU (users 4676 and 2388 of the 732 set, w8+s8, show-time inputs, --split-existing --deleted-cards
     neighbour, w64-c1): SM106's final master.pth reproduces SM106's committed rows (curve and instant within 1e-5, the
     per-user tolerance of AGENT.md: the committed GPU rows are rounded to 6 decimals, so CPU vs GPU can flip the last
     digit, seen 1.0e-6 on user 4676 in the trial run); the
     reference snapshot has the final master's keys, shapes and dtypes and scores the same users (finite, different).
  C10 breaks: sm124_block.py on fixed times (code == sm118_block.py's) and on the real quiet.json now.
  C11 resources: >= 10 GB free on C: (no new targets; snapshots ~3 MB each).
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
CODE106 = "c99330a7"     # SM106's (= SM101's) trainer and scorer tree
LAUNCH115 = "112c4082"   # SM115's launch commit (SM118's tree)
W = Path("C:/Users/jonat/workspace/wt")
TREE106 = W / "sm106"
TRACK = "E89-F65e-F77a"
ARCH = "w64-c1"
CACHE = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay")
TEACHER = W / "sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth"
TEACHER_SHA = "67ee8e9b0d0169733242c40646e0d2e5637845a25775762002ffc4e6396f9590"
S106 = W / "sm106/artifacts/small-model"
E29 = S106 / "sm106-c1-e29-st"
E29_SHA = "b3a694ca14cbd7a2291c7400203d5f9774bbdabc5d3e572a8408a196e5cb7be9"
DEC106 = S106 / "sm106-c1-e29st-st-dec-1ep32k-muon"
FINAL106_SHA = "e9d0d0a0b3368a920666c5cf89fdde402ca3cfd2465bd3688085bac90c8e3b4f"
SNAP106 = DEC106 / "expertium" / TRACK / "snapshots/users-04000"
SNAP106_SHA = "957b9118e944ea4422dd08fd0123d503db19ea4b8f18f8d348848a0434aaccfe"
P118 = W / "sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p1"
RUN8K = W / "sm55/artifacts/small-model/sm55-final8k-run.json"
OLD = CACHE / "big-leakfree-1ep-final8k-neighbour"          # SM106's own targets
ST91 = CACHE / "big-showtime-cool1991-neighbour-e2s"        # SM91 step 3: the same teacher, 1,991 users
NEW = CACHE / "sm91s2-big-showtime-final8k-neighbour-e2s"   # SM118's export, reused
RECIPE = SM / "sm124-recipe-e29-dec1ep32k"
P1 = SM / "sm124-c1-e29st-dec2ep32k-t91s2-p1"
SMOKE0 = SM / "sm124-smoke0-c1-e29st-dec2ep32k-t91s2"
G118 = SM / "sm118/gpu"
ROWS106 = SM / "sm106/gpu/sm106-rows.jsonl"
LBL106 = "sm106-c1-e29st-st-dec1ep32k/w8+s8@e2s-query"
USERS_C9 = (4676, 2388)
QUIET = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json")
DATA = r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"
DIFF106 = {"scripts/expertium_decay.py": (6, 1), "scripts/expertium_options.py": (6, 1), "scripts/sm91s2.py": (5, 5),
           "scripts/sm91s3.py": (2, 2)}
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))


class Report:
    def __init__(self):
        self.checks, self.ok = {}, True

    def check(self, name, ok, **detail):
        self.checks[name] = {"ok": bool(ok), **detail}
        self.ok &= bool(ok)
        print(f"[sm124-check] {name}: {'OK' if ok else 'FAIL'} {json.dumps(detail, default=str)[:400]}", flush=True)


def git(*args, root=ROOT):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def env_cpu(**extra):
    env = {k: v for k, v in os.environ.items() if k not in ("RWKV_DELETED_CARDS", "EXPERTIUM_END_TO_START", "SM11_INPUTS",
                                                            "HELDOUT_GPU_MEMORY_FRACTION", "SM11_GPU_MEMORY_FRACTION")}
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    env.update(extra)
    return env


def old_module(name, rel, tmp):
    """SM106's tree's own copy of a file, loaded as a separate module (its imports resolve to this tree's packages)."""
    src = git("show", f"{CODE106}:{rel}")
    path = Path(tmp) / f"{name}.py"
    path.write_text(src + "\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "ROOT"):
        mod.ROOT = ROOT   # the copy lives in a temp folder; its paths are this tree's
    return mod


def c1(rep):
    trees = {p: (git("rev-parse", f"HEAD:{p}"), git("rev-parse", f"{LAUNCH115}:{p}")) for p in ("scripts", "expertium", "rwkv")}
    dirty = git("status", "--porcelain", "--", "scripts", "expertium", "rwkv")
    num = {}
    for line in git("diff", "--numstat", CODE106, "HEAD", "--", "scripts", "expertium", "rwkv").splitlines():
        a, d, f = line.split("\t")
        num[f] = (int(a), int(d))
    t106 = {p: git("rev-parse", f"HEAD:{p}", root=TREE106) for p in ("expertium", "rwkv")}
    t106_ok = all(t106[p] == git("rev-parse", f"{CODE106}:{p}") for p in t106)
    s106 = [x.split("\t")[2] for x in git("diff", "--numstat", CODE106, "HEAD", "--", "scripts", root=TREE106).splitlines()]
    dirty106 = git("status", "--porcelain", "--", "scripts", "expertium", "rwkv", root=TREE106)
    ok = all(a == b for a, b in trees.values()) and not dirty and num == DIFF106 and t106_ok and \
        sorted(s106) == ["scripts/sm91s2.py", "scripts/sm91s3.py"] and not dirty106
    rep.check("C1_code_sm115_launch_tree", ok, trees={k: v[0][:12] for k, v in trees.items()}, uncommitted=dirty.splitlines()[:10],
              vs_sm106_tree=num, wt_sm106_equals_its_tree_except=s106, wt_sm106_uncommitted=dirty106.splitlines()[:5])


def trainer_probe(tree, spec, tmp):
    spec_file = Path(tmp) / f"spec-{hashlib.sha256(json.dumps(spec).encode()).hexdigest()[:12]}.json"
    spec_file.write_text(json.dumps(spec), encoding="utf-8")
    env = dict(os.environ, RWKV_DELETED_CARDS="neighbour", CUDA_VISIBLE_DEVICES="", GPU_BATCH1_DATA_PATH=DATA,
               PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    env.pop("RWKV_NO_JIT", None)
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm124_probe.py"), str(tree), str(spec_file)], cwd=tree,
                       capture_output=True, text=True, env=env)
    line = next((x for x in r.stdout.splitlines() if x.startswith("PROBE ")), None)
    if line is None:
        raise RuntimeError(f"probe failed in {tree}: {r.stderr[-3000:]}")
    return json.loads(line[6:])


def lr_probe(catch, total, points=64):
    out = []
    for k in range(points + 1):
        r = min(total, (total * k) // points)
        out.append((catch.learning_rate(100, 1000, r), bool(catch.schedule.decaying(r + catch.reviews_offset)) if catch.schedule else None))
    return out


def c2(rep, tmp):
    # C2a: the trainer slice, SM106's tree vs this tree
    protocol = json.loads((DEC106 / "protocol.json").read_text())
    small = sorted((r for r in protocol["rows"] if 300 <= r["train_range"][1] <= 1500), key=lambda r: r["train_range"][1])
    rows = [small[0], next(r for r in small[len(small) // 2:] if r["user_id"] != small[0]["user_id"])]
    opts = json.loads((DEC106 / "decay.json").read_text())["options"]
    options = {k: v for k, v in opts.items() if k not in ("packed_ipc", "distillation")}
    options.update(chunk_reviews=128, batch_chunks=2)
    ck = E29 / "expertium" / TRACK / "master.pth"
    keys = ("user", "offset", "loss_hex", "imm_hex", "curve_hex", "p_digest", "grad_digest", "zero_grad")
    view = lambda p: [{k: u[k] for k in keys} for u in p["updates"]]  # noqa: E731
    sched = {"lr_schedule": protocol["full_run"]["lr_schedule"]}
    res = {}
    kd_old = {k: v for k, v in opts["distillation"].items() if k != "cache"}
    kd_new = dict(kd_old, teacher=TEACHER.as_posix(), replay=NEW.as_posix())
    for label, extra, trees in (("muon_cooldown_show_time", {}, (TREE106, ROOT)),
                                ("muon_cooldown_show_time_kd_replay", {"distillation": kd_old}, (TREE106, ROOT)),
                                ("muon_cooldown_show_time_kd_new_targets_this_tree", {"distillation": kd_new}, (ROOT,))):
        spec = {"arch": ARCH, "checkpoint": str(ck), "options": {**options, **extra}, "rows": rows, "span": 256}
        if extra:   # KD replay targets with the cool-down's schedule (KD switch -> decay alpha 0.5)
            spec.update(load_teacher=True, full_run=sched)
        try:
            got = [trainer_probe(t, spec, tmp) for t in trees]
        except RuntimeError as error:
            res[label] = {"ok": False, "error": str(error)[-600:]}
            continue
        b = got[-1]
        n = len(b["updates"])
        same = len(got) == 1 or (view(got[0]) == view(b) and got[0]["weights"] == b["weights"])
        res[label] = {"ok": n > len(rows) and same and b["jit"] and b["optimizer"] == "HybridOptimizer"
                      and all(u["finite"] for u in b["updates"])
                      and not any(p.startswith("head_w.") for u in b["updates"] for p in u["zero_grad"]),   # the curve head trains
                      "trees": [str(t) for t in trees], "updates": n, "optimizer": b["optimizer"], "scripted": b["jit"],
                      "weights": b["weights"][:16], "losses": [float.fromhex(u["loss_hex"]) for u in b["updates"]],
                      "curve_avg": [float.fromhex(u["curve_hex"]) for u in b["updates"]][:4]}
    kd = res.get("muon_cooldown_show_time_kd_replay", {})
    plain = res.get("muon_cooldown_show_time", {})
    new = res.get("muon_cooldown_show_time_kd_new_targets_this_tree", {})
    kd_differs = kd.get("ok") and plain.get("ok") and kd["losses"] != plain["losses"]
    new_differs = kd.get("ok") and new.get("ok") and kd["losses"] != new["losses"]
    rep.check("C2a_trainer_sm106_tree_equals_this_tree", plain.get("ok", False) and kd.get("ok", False) and new.get("ok", False)
              and bool(kd_differs) and bool(new_differs), users=[r["user_id"] for r in rows], probes=res,
              kd_changes_the_loss=kd_differs, new_targets_change_the_loss=new_differs)

    # C2b: CatchUp / resume_point, c99330a7's copies vs this tree's
    import scripts.expertium_decay as D
    from scripts.expertium_options import CatchUp
    OD = old_module("old_expertium_decay", "scripts/expertium_decay.py", tmp)
    OO = old_module("old_expertium_options", "scripts/expertium_options.py", tmp)
    trk = "check-track"

    def fake(out, users):
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        rr = [{"user_id": u, "available_reviews": n, "train_range": [0, n], "validation_range": [0, n]} for u, n in users]
        fr = {"epoch": 10, "evaluate": False, "init_master": "source/master.pth", "update_offset": 1000,
              "lr_schedule": {"decay_start": 0.0, "decay_end": 1.0, "shape": "one_minus_sin", "final": 0.0},
              "optimizer_state": "fresh", "snapshot_users": 2, "control_dir": (out / "control").relative_to(ROOT).as_posix()}
        D.atomic_json(out / "decay.json", {"track": trk, "options": {}})
        D.atomic_json(out / "protocol.json", {"rows": rr, "evaluation_rows": [], "full_run": fr})

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
    for tag in ("old", "new"):
        fake(base / f"sm124-c2-{tag}", users)
    for done, end, snap in (([11, 12], 1040, True), ([13], 1060, True), ([], 0, False)):
        outs = {}
        for tag, mod in (("old", OD), ("new", D)):
            out = base / f"sm124-c2-{tag}"
            interrupt(out, done, end, snap)
            mod.resume_point(out, {"track": trk})
            outs[tag] = {f: (out / f).read_text().replace(f"sm124-c2-{tag}", "X") for f in ("protocol.json", "resume.json", "protocol.prepared.json")}
        eq = outs["old"] == outs["new"]
        same_files &= eq
        p = json.loads((base / "sm124-c2-new" / "protocol.json").read_text())
        a, b = OO.CatchUp({}, p["full_run"], p["rows"]), CatchUp({}, p["full_run"], p["rows"])
        t = sum(r["train_range"][1] - r["train_range"][0] for r in p["rows"])
        lr_same &= lr_probe(a, t) == lr_probe(b, t)
        steps.append({"done": done, "files_equal": eq, "lr_resume": p["full_run"].get("lr_resume")})
    real = {}
    opt = json.loads((DEC106 / "decay.json").read_text())["options"]
    for name, f in (("sm106_cooldown", DEC106 / "protocol.json"), ("recipe", RECIPE / "protocol.json")):
        p = json.loads(f.read_text())
        a, b = OO.CatchUp(opt, p["full_run"], p["rows"]), CatchUp(opt, p["full_run"], p["rows"])
        t = sum(r["train_range"][1] - r["train_range"][0] for r in p["rows"])
        real[name] = {"equal": lr_probe(a, t, 200) == lr_probe(b, t, 200), "lr_resume": p["full_run"].get("lr_resume"),
                      "total": b.schedule.total}
        lr_same &= real[name]["equal"]
    for tag in ("old", "new"):
        shutil.rmtree(base / f"sm124-c2-{tag}", ignore_errors=True)
    r = subprocess.run([PY, "-X", "utf8", str(ROOT / "scripts/check_decay_resume.py")], capture_output=True, text=True,
                       cwd=str(ROOT), env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    lines = r.stdout.splitlines()
    passes, fails = sum(x.startswith("PASS") for x in lines), [x for x in lines if x.startswith("FAIL")]
    rep.check("C2b_schedule_and_resume_unchanged", same_files and lr_same and r.returncode == 0 and passes > 10 and not fails,
              steps=steps, real=real, check_decay_resume={"exit": r.returncode, "pass": passes, "fail": fails[:5]})


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
    base = ROOT / "artifacts/small-model/smoke/sm124-c3"
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
    for name, lr in (("segment_too_long", {"total_reviews": R + 100, "reviews_offset": 200, "segment": True}),
                     ("segment_flag_not_true", {"total_reviews": 2 * R, "reviews_offset": 0, "segment": "yes"}),
                     ("no_segment_short", {"total_reviews": 2 * R, "reviews_offset": 0}),
                     ("negative_offset", {"total_reviews": 2 * R, "reviews_offset": -1, "segment": True})):
        try:
            CatchUp({}, {"lr_schedule": sched, "lr_resume": lr}, rows); refused[name] = False
        except ValueError:
            refused[name] = True
    ok = (ok1 and ok2 and abs(end1 / 7e-4 - (1 - math.sin(math.pi / 4))) < 1e-12 and end2 == 0.0 and kd
          and res["p1"]["lr_resume"] == {"total_reviews": 2 * R, "reviews_offset": 700, "segment": True}
          and res["p2"]["lr_resume"] == {"total_reviews": 2 * R, "reviews_offset": R + 700}
          and all(v["lr_equal"] and v["rows"] == [2, 3] for v in res.values()) and all(refused.values()))
    rep.check("C3_segment_schedule", ok, pass1_equal=ok1, pass2_equal=ok2, pass1_end_lr_over_peak=end1 / 7e-4,
              pass2_end_lr=end2, resumes=res, refused=refused)


def c4(rep):
    d106 = json.loads((DEC106 / "decay.json").read_text())
    r106 = json.loads((DEC106 / "expertium" / TRACK / "result.json").read_text())
    t, e29 = sha256(TEACHER), sha256(E29 / "expertium" / TRACK / "master.pth")
    final = sha256(DEC106 / "expertium" / TRACK / "master.pth")
    snap = json.loads((SNAP106 / "snapshot.json").read_text())
    upd = [json.loads(x) for x in open(DEC106 / "expertium" / TRACK / "updates.jsonl", encoding="utf-8") if x.strip()]
    lr_snap = upd[int(snap["updates"]) - 1]["lr"] / 7e-4
    rp = int(snap["review_presentations"])
    total = int(d106["training_reviews_after_trims"])
    one_attempt = not (DEC106 / "attempts").exists() and not (DEC106 / "resume.json").exists() and not (DEC106 / "protocol.prepared.json").exists()
    snap_ok = (sha256(SNAP106 / "master.pth") == SNAP106_SHA and len(snap["done_users"]) == 4000 and rp == 284_944_410
               and one_attempt and abs(lr_snap - 0.30450) < 1e-4 and abs(lr_snap - (1 - math.sin(math.pi / 2 * rp / total))) < 2e-4)
    run = json.loads(RUN8K.read_text())
    rtotal = sum(run["review_counts"][str(u)] for u in run["train_users"])
    users106 = {r["user_id"] for r in json.loads((DEC106 / "protocol.json").read_text())["rows"]}
    ok = (t == TEACHER_SHA and e29 == E29_SHA == d106["start_master_sha256"] and d106.get("architecture") == ARCH
          and r106.get("status") == "complete" and not r106.get("smoke_only") and final == FINAL106_SHA and snap_ok
          and len(run["train_users"]) == 8000 and rtotal == 581_942_738 and set(run["train_users"]) == users106)
    rep.check("C4_inputs_pinned", ok, teacher=t[:12], epoch29=e29[:12], sm106_cooldown=r106.get("status"), sm106_final=final[:12],
              sm106_cooldown_one_attempt=one_attempt,
              early_stop_reference={"snapshot": SNAP106.relative_to(DEC106).as_posix(), "master": SNAP106_SHA[:12],
                                    "users_done": len(snap["done_users"]), "reviews_done": rp, "of": total, "fraction": round(rp / total, 5),
                                    "last_update_lr_over_peak": round(lr_snap, 5),
                                    "formula_lr_over_peak": round(1 - math.sin(math.pi / 2 * rp / total), 5), "ok": snap_ok},
              run_users=len(run["train_users"]), run_reviews=rtotal)


def c5(rep, work):
    r = subprocess.run([PY, "-X", "utf8", str(ROOT / "scripts/sm11_replay_export.py"), "status", "--out", str(NEW), "--run", str(RUN8K),
                        "--inputs", "e2s-query"], cwd=ROOT, env=env_cpu(RWKV_DELETED_CARDS="neighbour"), capture_output=True, text=True)
    status = {"exit": r.returncode, "tail": (r.stdout + r.stderr)[-300:]}
    man_now = json.loads((NEW / "manifest.json").read_text())
    man118 = json.loads((G118 / "replay-manifest.json").read_text())
    chk118 = json.loads((G118 / "check-targets-sm118.json").read_text())
    seg = json.loads((G118 / "segments.jsonl").read_text().splitlines()[0])
    p1_start = int(seg["start_epoch_s"])
    files = sorted((NEW / "users").glob("*.npz"))
    run = json.loads(RUN8K.read_text())
    names = {int(p.stem) for p in files}
    newest = max(p.stat().st_mtime for p in files)
    size = sum(p.stat().st_size for p in files)
    out = work / "check-targets-reused.json"
    t0 = time.perf_counter()
    rr = subprocess.run([PY, "-X", "utf8", str(SM / "sm118/sm118_check.py"), "targets", "--out", str(out)], cwd=ROOT,
                        env=env_cpu(), capture_output=True, text=True)
    again = json.loads(out.read_text()) if out.exists() else {}
    ok = (r.returncode == 0 and man_now == man118 and chk118.get("ok") is True and chk118.get("users_ok") == 8000
          and names == set(run["train_users"]) and newest < p1_start and rr.returncode == 0 and again.get("ok") is True
          and again.get("users_ok") == 8000 and (again.get("vs_sm91s3_same_teacher") or {}).get("users") == 1991
          and seg["folder"].endswith("sm118-e26-dec2ep32k-t91s2-p1"))
    rep.check("C5_reused_targets", ok, status=status, manifest_equals_sm118=man_now == man118, teacher=man_now.get("teacher_sha256", "")[:12],
              sm118_targets_check_ok=chk118.get("ok"), users=len(names), gib=round(size / 2 ** 30, 2),
              newest_file_before_sm118_pass1_start=newest < p1_start,
              newest_file=time.strftime("%Y-%m-%d %H:%M", time.localtime(newest)),
              sm118_pass1_start=time.strftime("%Y-%m-%d %H:%M", time.localtime(p1_start)),
              targets_check_again={k: again.get(k) for k in ("users", "users_ok", "failures_count", "vs_sm91s3_same_teacher", "ok")},
              targets_check_seconds=round(time.perf_counter() - t0, 1))


def mutated_pair_p1(tmp, src, fn, name):
    import sm124_pair
    d = Path(tmp) / f"mut-p1-{name}" / src.name
    shutil.rmtree(d.parent, ignore_errors=True)
    d.mkdir(parents=True)
    p = json.loads((src / "protocol.json").read_text())
    y = json.loads((src / "decay.json").read_text())
    p["full_run"]["control_dir"] = (SM / d.name / "control").relative_to(ROOT).as_posix()
    fn(p, y)
    (d / "protocol.json").write_text(json.dumps(p)); (d / "decay.json").write_text(json.dumps(y))
    return sm124_pair.pair_p1(d, DEC106, P118)


def c6(rep, tmp):
    import sm124_pair
    from scripts.expertium_options import CatchUp
    frec, irec = sm124_pair.pair_recipe(RECIPE, DEC106)
    self_ref, _ = sm124_pair.pair_p1(DEC106, DEC106, P118)
    fail, info = sm124_pair.pair_p1(P1, DEC106, P118)
    fail_copy, _ = mutated_pair_p1(tmp, P1, lambda p, y: None, "as_is")
    p_1, p_0 = (json.loads((d / "protocol.json").read_text()) for d in (P1, SMOKE0))
    y_1, y_0 = (json.loads((d / "decay.json").read_text()) for d in (P1, SMOKE0))
    smoke_same = (dict(p_1, full_run={k: v for k, v in p_1["full_run"].items() if k != "control_dir"})
                  == dict(p_0, full_run={k: v for k, v in p_0["full_run"].items() if k != "control_dir"})
                  and {k: v for k, v in y_0.items() if k != "sm124_smoke0"} == y_1 and "sm124_smoke0" in y_0)
    d106 = json.loads((DEC106 / "decay.json").read_text())
    neg = {}
    for name, fn in (("lr_offset", lambda p, y: p["full_run"]["lr_resume"].update(reviews_offset=1)),
                     ("no_segment", lambda p, y: p["full_run"]["lr_resume"].pop("segment")),
                     ("old_targets", lambda p, y: y["options"].update(distillation=d106["options"]["distillation"])),
                     ("st91_replay", lambda p, y: y["options"]["distillation"].update(replay=str(ST91))),
                     ("curve_weight_0", lambda p, y: y["options"].update(curve_loss_weight=0.0)),
                     ("curve_weight_1", lambda p, y: y["options"].update(curve_loss_weight=1.0)),
                     ("decay_alpha", lambda p, y: y["options"]["distillation"].update(decay_alpha=0.6)),
                     ("show_time_off", lambda p, y: y["options"].pop("show_time_intervals")),
                     ("order", lambda p, y: p["rows"].insert(0, p["rows"].pop(5))),
                     ("arch_w64", lambda p, y: (y.update(architecture="w64"), y["options"].update(architecture="w64"))),
                     ("sm118_start", lambda p, y: (p["full_run"].update(init_master=json.loads((P118 / "protocol.json").read_text())["full_run"]["init_master"]),
                                                   y.update(start_master_sha256=json.loads((P118 / "decay.json").read_text())["start_master_sha256"])))):
        f2, _ = mutated_pair_p1(tmp, P1, fn, name)
        neg[name] = bool(f2)
    c = CatchUp(y_1["options"], p_1["full_run"], p_1["rows"])
    R = sum(r["train_range"][1] - r["train_range"][0] for r in p_1["rows"])
    lr0, lrR = c.learning_rate(10, p_1["full_run"]["update_offset"], 0), c.learning_rate(10, p_1["full_run"]["update_offset"], R)
    mu = subprocess.run([PY, "-X", "utf8", str(ROOT / "scripts/check_muon_fresh_decay.py"), "--decay", str(SMOKE0),
                         "--out", str(HERE / "gpu/check-muon-fresh-smoke0.json")], capture_output=True, text=True,
                        env=env_cpu(RWKV_DELETED_CARDS="neighbour"))
    lw = {k: json.loads((HERE / f"gpu/check-replay-label-windows-{k}.json").read_text()) for k in ("recipe", "p1")}
    trims = {k: [t["user_id"] for t in json.loads((d / "decay.json").read_text()).get("trimmed_unlabelled_tails", [])] for k, d in
             (("recipe", RECIPE), ("p1", P1), ("sm106", DEC106))}
    sm = [json.loads(x) for x in (HERE / "gpu/smoke0.out").read_text().splitlines() if x.startswith('{"smoke"')]
    smoke_ok = bool(sm) and sm[-1].get("exit") == 0 and sm[-1].get("status") == "complete" and sm[-1].get("reviews") == 1024 \
        and sm[-1].get("smoke") == SMOKE0.name
    ok = (not frec and bool(self_ref) and not fail and not fail_copy and smoke_same and all(neg.values())
          and c.schedule.total == 2 * R == 1_163_885_448 and lr0 == 7e-4 and abs(lrR / 7e-4 - (1 - math.sin(math.pi / 4))) < 1e-9
          and mu.returncode == 0 and all(v.get("ok") is True for v in lw.values()) and trims == {"recipe": [5599], "p1": [5599], "sm106": [5599]}
          and smoke_ok)
    rep.check("C6_recipe_pass1_and_smoke_folders", ok, recipe=frec[:4] or irec, sm106_as_pass1_refused=bool(self_ref),
              pass1=fail[:4] or info, smoke0_equals_pass1_except_control_dir_and_note=smoke_same, mutated_refused=neg,
              schedule_total=c.schedule.total, lr_start=lr0, lr_at_R_over_peak=lrR / 7e-4, muon_fresh_exit=mu.returncode,
              label_windows={k: v.get("ok") for k, v in lw.items()}, trims=trims, gpu_smoke0=sm[-1] if sm else None)


def c7(rep, tmp):
    import sm124_pair
    from scripts.expertium_options import CatchUp
    base = ROOT / "artifacts/small-model/smoke"
    a, b = base / "sm124-c7-p1", base / "sm124-c7-p2"
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)
    a.mkdir(parents=True)
    for f in ("protocol.json", "decay.json"):
        shutil.copy(P1 / f, a / f)
    case = a / "expertium" / TRACK
    case.mkdir(parents=True)
    p = json.loads((a / "protocol.json").read_text())
    p["full_run"]["control_dir"] = (a / "control").relative_to(ROOT).as_posix()
    (a / "protocol.json").write_text(json.dumps(p, indent=1), encoding="utf-8")
    R = sum(r["train_range"][1] - r["train_range"][0] for r in p["rows"])
    (case / "master.pth").write_bytes(b"fake master"); (case / "optimizer.pth").write_bytes(b"fake optimizer")
    json.dump({"status": "complete", "update_offset_end": 662340, "review_presentations": R,
               "trained_parameter_sha256": "f" * 64}, open(case / "result.json", "w"))
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm124_prepare.py"), "p2", "--p1", str(a), "--out", str(b)],
                       capture_output=True, text=True)
    fail, info = sm124_pair.pair_p2(b, a) if r.returncode == 0 else (["prepare failed: " + r.stderr[-300:]], "")
    neg = {}

    def mutate(name, fn):
        d = base / f"sm124-c7-mut-{name}"
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(b, d)
        q = json.loads((d / "protocol.json").read_text()); y = json.loads((d / "decay.json").read_text())
        fn(q, y)
        (d / "protocol.json").write_text(json.dumps(q)); (d / "decay.json").write_text(json.dumps(y))
        f2, _ = sm124_pair.pair_p2(d, a)
        neg[name] = bool(f2)
        shutil.rmtree(d, ignore_errors=True)
    mutate("same_order", lambda q, y: q.update(rows=json.loads((a / "protocol.json").read_text())["rows"]))
    mutate("lr_offset", lambda q, y: q["full_run"]["lr_resume"].update(reviews_offset=0))
    mutate("other_start", lambda q, y: q["full_run"].update(init_master=str(E29 / "expertium" / TRACK / "master.pth")))
    mutate("decay_alpha", lambda q, y: y["options"]["distillation"].update(decay_alpha=0.9))
    ok = r.returncode == 0 and not fail and all(neg.values())
    lr = {}
    if ok:
        q = json.loads((b / "protocol.json").read_text())
        c2_ = CatchUp(json.loads((b / "decay.json").read_text())["options"], q["full_run"], q["rows"])
        c1_ = CatchUp(json.loads((a / "decay.json").read_text())["options"], p["full_run"], p["rows"])
        off = q["full_run"]["update_offset"]
        lr = {"p1_at_R": c1_.learning_rate(10, off, R), "p2_at_0": c2_.learning_rate(0, off, 0), "p2_end": c2_.learning_rate(0, off, R),
              "p2_mid_vs_formula": c2_.learning_rate(0, off, R // 2) / 7e-4 - (1 - math.sin(math.pi / 2 * 0.75))}
        ok = lr["p1_at_R"] == lr["p2_at_0"] and lr["p2_end"] == 0.0 and abs(lr["p2_mid_vs_formula"]) < 1e-9
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)
    rep.check("C7_pass2_prepare_and_pair", ok, prepare=r.stdout.strip()[-300:], pair=fail[:4] or info, mutated_refused=neg, lr=lr)


def c8(rep):
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm124_rule.py"), "selftest"], capture_output=True, text=True)
    rep.check("C8_rule_selftest", r.returncode == 0 and r.stdout.strip().endswith("OK"), out=r.stdout.strip()[-400:], err=r.stderr[-400:])


def c9(rep, work):
    import torch
    users = work / "users-c9.json"
    users.write_text(json.dumps({"users": list(USERS_C9)}))
    final = DEC106 / "expertium" / TRACK / "master.pth"
    a = torch.load(final, map_location="cpu", weights_only=True)
    b = torch.load(SNAP106 / "master.pth", map_location="cpu", weights_only=True)
    same_layout = list(a) == list(b) and all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype for k in a)
    got = {}
    for name, ck in (("sm106-final", final), ("sm106-u4000", SNAP106 / "master.pth")):
        out = work / f"c9-{name}.jsonl"
        out.unlink(missing_ok=True)
        cmd = [PY, "-B", "-X", "utf8", str(ROOT / "scripts/small_model_heldout.py"), "--pool", "732", "--users-file", str(users),
               "--device", "cpu", "--model", f"{name}=expertium:{ARCH}:{ck}", "--only", f"{name}/w8+s8", "--inputs", "e2s-query",
               "--split-existing", "--deleted-cards", "neighbour", "--out", str(out)]
        r = subprocess.run(cmd, cwd=ROOT, env=env_cpu(), capture_output=True, text=True)
        rows = [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
        got[name] = {x["user_id"]: x["benchmark"] for x in rows if "benchmark" in x}
        if r.returncode or len(got[name]) != len(USERS_C9):
            got[name + "_error"] = (r.stdout + r.stderr)[-1200:]
    ref = {}
    for line in ROWS106.read_text(encoding="utf-8").splitlines():
        x = json.loads(line)
        if x.get("label") == LBL106 and x["user_id"] in USERS_C9 and "benchmark" in x:
            ref[x["user_id"]] = x["benchmark"]
    gaps, moved = {}, {}
    ok = same_layout and all(len(got.get(n, {})) == len(USERS_C9) for n in ("sm106-final", "sm106-u4000")) and len(ref) == len(USERS_C9)
    if ok:
        for u in USERS_C9:
            gaps[u] = max(abs(got["sm106-final"][u][h]["LogLoss"] - ref[u][h]["LogLoss"]) for h in ("ahead", "instant"))
            moved[u] = {h: (got["sm106-u4000"][u][h]["LogLoss"], got["sm106-final"][u][h]["LogLoss"]) for h in ("ahead", "instant")}
        ok = all(g <= 1e-5 for g in gaps.values()) and all(
            math.isfinite(v[h][0]) and v[h][0] != v[h][1] for v in moved.values() for h in ("ahead", "instant"))
    rep.check("C9_scorer_reproduces_sm106_rows_and_scores_the_snapshot", ok, same_state_dict_layout=same_layout,
              max_abs_logloss_vs_committed_rows=gaps, snapshot_vs_final=moved,
              errors={k: v for k, v in got.items() if k.endswith("_error")})


def c10(rep, tmp):
    def block(now, quiet):
        return subprocess.run([PY, "-X", "utf8", str(HERE / "sm124_block.py"), "--now", now, "--quiet", str(quiet)],
                              capture_output=True, text=True).stdout.strip()
    empty = Path(tmp) / "quiet-empty.json"
    empty.write_text("[]", encoding="utf-8")
    got = {"tue_12:00": block("2026-11-17T12:00", empty), "tue_20:30": block("2026-11-17T20:30", empty),
           "wed_01:00": block("2026-11-18T01:00", empty), "wed_19:00": block("2026-11-18T19:00", empty),
           "sat_15:00": block("2026-11-21T15:00", empty), "sun_15:00": block("2026-11-22T15:00", empty),
           "sun_16:30": block("2026-11-22T16:30", empty), "real_now": block(time.strftime("%Y-%m-%dT%H:%M"), QUIET)}
    want = {"tue_12:00": "NEXT 480", "tue_20:30": "NOW 16260", "wed_01:00": "NEXT 1140", "wed_19:00": "NEXT 60",
            "sat_15:00": "NEXT 300", "sun_15:00": "NEXT 60", "sun_16:30": "NOW 30660"}
    src = (HERE / "sm124_block.py").read_text(encoding="utf-8").split('"""\nimport argparse', 1)[1]
    ref = (SM / "sm118/sm118_block.py").read_text(encoding="utf-8").split('"""\nimport argparse', 1)[1]
    rep.check("C10_breaks", all(got[k] == v for k, v in want.items()) and src == ref and got["real_now"][:4] in ("NEXT", "NOW "),
              got=got, code_equals_sm118_block=src == ref)


def c11(rep):
    free = shutil.disk_usage("C:/").free / 2 ** 30
    rep.check("C11_disk", free >= 10, free_gib=round(free, 1))


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "check-cpu-sm124.json"))
    ap.add_argument("--work", default=str(HERE / "work"))
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    rep = Report()
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        for name, fn in (("C1", lambda: c1(rep)), ("C2", lambda: c2(rep, tmp)), ("C3", lambda: c3(rep)), ("C4", lambda: c4(rep)),
                         ("C5", lambda: c5(rep, work)), ("C6", lambda: c6(rep, tmp)), ("C7", lambda: c7(rep, tmp)),
                         ("C8", lambda: c8(rep)), ("C9", lambda: c9(rep, work)), ("C10", lambda: c10(rep, tmp)), ("C11", lambda: c11(rep))):
            if only is None or name in only:
                try:
                    fn()
                except Exception as error:   # a crashed check is a failed check
                    import traceback
                    rep.check(f"{name}_crashed", False, error=repr(error)[:300], where=traceback.format_exc()[-800:])
    out = {"commit": git("rev-parse", "--short", "HEAD"), "minutes": round((time.perf_counter() - t0) / 60, 1),
           "all_ok": rep.ok and only is None, "partial": sorted(only) if only else None, "checks": rep.checks}
    if only is None:
        Path(args.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"[sm124-check] {'ALL OK' if out['all_ok'] else ('partial ' + ('OK' if rep.ok else 'FAILED'))} in {out['minutes']} min")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
