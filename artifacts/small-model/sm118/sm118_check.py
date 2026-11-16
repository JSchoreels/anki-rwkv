#!/usr/bin/env python
"""The show-time two-head cool-down for Small (SM118): CPU proofs before any GPU ticket (design.md in this folder;
~3-8 min), and the check of the exported targets after the export (`targets`, run by the launcher).

  python -X utf8 artifacts/small-model/sm118/sm118_check.py [--out check-cpu-sm118.json] [--only C4,C5]
  python -X utf8 artifacts/small-model/sm118/sm118_check.py targets [--out gpu/check-targets-sm118.json]

The run: the longer cool-down recipe (one one_minus_sin decay over 2R reviews as two chained folders, the recipe of the
longer big instant-only teacher (SM113) and the longer cool-down for Tiny (SM115)) applied to the Small two-head model:
start = the show-time retrain of the release (SM92)'s constant-LR epoch 26 (w64), where SM92's own one-epoch cool-down
started; both heads trained; teacher targets of the show-time big model (SM91s2) on show-time inputs for the 8,000
cool-down users (a new export with scripts/sm11_replay_export.py --inputs e2s-query). No source change: the code is
SM115's launch tree.

  C1 code: scripts/ expertium/ rwkv/ == 112c4082 (SM115's launch commit) and nothing uncommitted or untracked there;
     against 9c125341 (the tree SM92 trained and scored with) exactly these files differ: the curve-loss-weight option
     of the instant-only small model (SM57) (variant_model.py, gpu_r01_training.py, gpu_r01_prefetch.py and part of
     expertium_options.py; default 1.0 proven bit for bit by SM57's C3 / C4), the 2R segment schedule of SM113
     (expertium_decay.py, expertium_options.py), the 1,500 pool option (eval_pools.py, small_model_heldout.py) and the
     history-rewrite commit-id strings (sm91s2.py, sm91s3.py).
  C2 the old paths are unchanged, A/B against SM92's tree:
     C2a trainer slice on CPU (sm118_probe.py, a copy of SM57's probe) in wt/sm92 (== 9c125341 except the two
         rewrite files) and in this tree: 2 small users of SM92's cool-down, from SM92's epoch 26, SM92's cool-down
         options (show-time inputs, fresh Muon), with and without KD from the old replay targets: per update loss bits,
         instant bits, every gradient, and the final weights bit for bit;
     C2b CatchUp's learning rate / KD switch and resume_point's files, 9c125341's expertium_decay.py / expertium_options.py
         vs this tree's, on fake decays and on SM92's real cool-down (as prepared and as resumed); check_decay_resume.py.
  C3 the segment schedule (synthetic, R = 2,000 reviews), SM115's C3: pass 1 + pass 2 == one 2R decay at every review,
     also across a snapshot resume inside either pass; bad segments refused.
  C4 inputs, read only: the teacher sha256 (== SM91 step 3's 1,991-user targets' manifest); SM92's epoch 26 sha256 ==
     SM92's cool-down start; SM92's cool-down complete, its final master sha256 pinned; the early-stop reference: SM92's
     cool-down snapshot users-00800 of its final attempt (resumed from attempt 1's users-03256, no later attempt):
     master sha256 pinned, 3,256 + 800 = 4,056 users, 288,456,043 reviews, last update's learning rate 0.2977 x peak; the
     8k run file == the old targets' users and reviews == SM92's cool-down users; the new targets folder does not exist
     yet or holds only users of the run.
  C5 the exporter on CPU, users 4001 and 8081 (cool-down users inside SM91 step 3's 1,991): the show-time big model
     (SM91s2) on show-time inputs: the manifest == SM91 step 3's manifest (no key differs); per user the NaN pattern ==
     the old targets' (SM92's) and SM91 step 3's; values == SM91 step 3's GPU targets within the exporter's tolerance
     (curve probability 2.2e-4, ratings one float16 step); the targets differ from the old ones (counts reported).
  C6 folders: the recipe (this tree's sm77_warm cooldown from SM92's epoch 26 with SM92's own extra options, never
     trained) == SM92's cool-down except paths (sm118_pair.py recipe); the pre-launch smoke folder == pass 1 as it will
     be prepared except the replay folder (SM91 step 3's 1,991 users): with the replay set to the new folder it passes
     sm118_pair.py p1 (vs SM92's cool-down and SM113's pass 1); 10 mutated copies refused; CatchUp total 2R, 7e-4 after
     the warm-up, 0.2929 x 7e-4 at R; Muon fresh-state check; the label-window check of the recipe; the GPU smoke of the
     smoke folder passed (gpu/smoke0.out).
  C7 pass 2 on a fake pass-1 end (fake result.json + master/optimizer bytes): sm118_prepare.py p2 builds a folder that
     sm118_pair.py p2 accepts; 4 mutated copies refused; its learning rate continues pass 1's.
  C8 rule: sm118_rule.py selftest (reproduces SM92's published rule pair and SM92 - SM91s2).
  C9 the scorer on CPU (users 4676 and 2388 of the 732 set, w8+s8, show-time inputs, --split-existing --deleted-cards
     neighbour): SM92's final master.pth reproduces SM92's committed rows (curve and instant within 1e-6); the reference
     snapshot has the final master's keys, shapes and dtypes and scores the same users (finite, different).
  C10 breaks: sm118_block.py on fixed times (code == sm115_block.py's) and on the real quiet.json now.
  C11 resources: >= 20 GB free on C: (new targets ~5.5 GB, snapshots ~3 MB each).
targets (after the export): for each of the 8,000 users, the new file's shape == the old targets', the NaN pattern ==
  the old targets' (curve and rating columns), every other value finite, ratings summing to 1 within float16 rounding;
  for the 1,991 users of SM91 step 3 (the same teacher), new == SM91 step 3's within the exporter's tolerance; summary
  of how far the targets moved from the old ones.
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
CODE92 = "9c125341"      # SM92's launch commit (its trainer and scorer tree)
LAUNCH115 = "112c4082"   # SM115's launch commit
W = Path("C:/Users/jonat/workspace/wt")
TREE92 = W / "sm92"
TRACK = "E89-F65e-F77a"
CACHE = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay")
TEACHER = W / "sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth"
TEACHER_SHA = "67ee8e9b0d0169733242c40646e0d2e5637845a25775762002ffc4e6396f9590"
S92 = W / "sm92/artifacts/small-model"
E26 = S92 / "sm92-e26"
E26_SHA = "5a629e4e932ac6fddf1ada714fc33d9881aac8b8b1490ad6949193b5785e79cc"
DEC92 = S92 / "sm92-e26-dec-1ep32k-muon"
FINAL92_SHA = "2091f8a879a56cf96a3df3b820fdacfa64c6467e8502a3a5eb17436524100d7f"
SNAP92 = DEC92 / "expertium" / TRACK / "snapshots/users-00800"
SNAP92_SHA = "e12a7ae7bbe5e439b32a37e14ba4de6f8b1472a0acba248f1b8e70984909acd2"
P113 = W / "sm113/artifacts/small-model/sm113-inst-e35st-st-dec2ep32k-t99-p1"
RUN8K = W / "sm55/artifacts/small-model/sm55-final8k-run.json"
OLD = CACHE / "big-leakfree-1ep-final8k-neighbour"          # SM92's own targets
ST91 = CACHE / "big-showtime-cool1991-neighbour-e2s"        # SM91 step 3: the same teacher, 1,991 users
NEW = CACHE / "sm91s2-big-showtime-final8k-neighbour-e2s"   # this run's export
RECIPE = SM / "sm118-recipe-e26-dec1ep32k"
SMOKE0 = SM / "sm118-smoke0-e26-dec2ep32k-st91"
ROWS92 = SM / "sm92/gpu/sm92-inputs.jsonl"
LBL92 = "sm92-e26-dec1ep32k/w8+s8@e2s-query"
USERS_C5, USERS_C9 = (4001, 8081), (4676, 2388)
QUIET = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json")
DATA = r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"
DIFF92 = {"expertium/model/variant_model.py": (11, 5), "scripts/eval_pools.py": (11, 4), "scripts/expertium_decay.py": (6, 1),
          "scripts/expertium_options.py": (8, 2), "scripts/gpu_r01_prefetch.py": (1, 1), "scripts/gpu_r01_training.py": (8, 0),
          "scripts/sm91s2.py": (5, 5), "scripts/sm91s3.py": (2, 2), "scripts/small_model_heldout.py": (4, 3)}
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))


class Report:
    def __init__(self):
        self.checks, self.ok = {}, True

    def check(self, name, ok, **detail):
        self.checks[name] = {"ok": bool(ok), **detail}
        self.ok &= bool(ok)
        print(f"[sm118-check] {name}: {'OK' if ok else 'FAIL'} {json.dumps(detail, default=str)[:400]}", flush=True)


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
    """SM92's tree's own copy of a file, loaded as a separate module (its imports resolve to this tree's packages)."""
    src = git("show", f"{CODE92}:{rel}")
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
    for line in git("diff", "--numstat", CODE92, "HEAD", "--", "scripts", "expertium", "rwkv").splitlines():
        a, d, f = line.split("\t")
        num[f] = (int(a), int(d))
    t92 = {p: git("rev-parse", f"HEAD:{p}", root=TREE92) for p in ("expertium", "rwkv")}
    t92_ok = all(t92[p] == git("rev-parse", f"{CODE92}:{p}") for p in t92)
    s92 = [x.split("\t")[2] for x in git("diff", "--numstat", CODE92, "HEAD", "--", "scripts", root=TREE92).splitlines()]
    dirty92 = git("status", "--porcelain", "--", "scripts", "expertium", "rwkv", root=TREE92)
    ok = all(a == b for a, b in trees.values()) and not dirty and num == DIFF92 and t92_ok and \
        sorted(s92) == ["scripts/sm91s2.py", "scripts/sm91s3.py"] and not dirty92
    rep.check("C1_code_sm115_launch_tree", ok, trees={k: v[0][:12] for k, v in trees.items()}, uncommitted=dirty.splitlines()[:10],
              vs_sm92_tree=num, wt_sm92_equals_its_launch_tree_except=s92, wt_sm92_uncommitted=dirty92.splitlines()[:5])


def trainer_probe(tree, spec, tmp):
    spec_file = Path(tmp) / f"spec-{hashlib.sha256(json.dumps(spec).encode()).hexdigest()[:12]}.json"
    spec_file.write_text(json.dumps(spec), encoding="utf-8")
    env = dict(os.environ, RWKV_DELETED_CARDS="neighbour", CUDA_VISIBLE_DEVICES="", GPU_BATCH1_DATA_PATH=DATA,
               PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    env.pop("RWKV_NO_JIT", None)
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm118_probe.py"), str(tree), str(spec_file)], cwd=tree,
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
    # C2a: the trainer slice, SM92's tree vs this tree
    protocol = json.loads((DEC92 / "protocol.prepared.json").read_text())
    small = sorted((r for r in protocol["rows"] if 300 <= r["train_range"][1] <= 1500), key=lambda r: r["train_range"][1])
    rows = [small[0], next(r for r in small[len(small) // 2:] if r["user_id"] != small[0]["user_id"])]
    opts = json.loads((DEC92 / "decay.json").read_text())["options"]
    options = {k: v for k, v in opts.items() if k not in ("packed_ipc", "distillation")}
    options.update(chunk_reviews=128, batch_chunks=2)
    ck = E26 / "expertium" / TRACK / "master.pth"
    keys = ("user", "offset", "loss_hex", "imm_hex", "curve_hex", "p_digest", "grad_digest", "zero_grad")
    view = lambda p: [{k: u[k] for k in keys} for u in p["updates"]]  # noqa: E731
    res = {}
    for label, extra in (("muon_cooldown_show_time", {}),
                         ("muon_cooldown_show_time_kd_replay", {"distillation": {k: v for k, v in opts["distillation"].items() if k != "cache"}})):
        spec = {"arch": "w64", "checkpoint": str(ck), "options": {**options, **extra}, "rows": rows, "span": 256}
        if extra:   # SM92's KD replay targets with the cool-down's schedule (KD switch -> decay alpha 0.5)
            spec.update(load_teacher=True, full_run={"lr_schedule": json.loads((DEC92 / "protocol.prepared.json").read_text())["full_run"]["lr_schedule"]})
        try:
            a, b = trainer_probe(TREE92, spec, tmp), trainer_probe(ROOT, spec, tmp)
        except RuntimeError as error:
            res[label] = {"ok": False, "error": str(error)[-600:]}
            continue
        n = len(a["updates"])
        res[label] = {"ok": n > len(rows) and view(a) == view(b) and a["weights"] == b["weights"] and a["jit"]
                      and a["optimizer"] == "HybridOptimizer" and all(u["finite"] for u in b["updates"])
                      and not any(p.startswith("head_w.") for u in b["updates"] for p in u["zero_grad"]),   # the curve head trains
                      "updates": n, "optimizer": b["optimizer"], "scripted": b["jit"], "weights": b["weights"][:16],
                      "losses": [float.fromhex(u["loss_hex"]) for u in b["updates"]],
                      "curve_avg": [float.fromhex(u["curve_hex"]) for u in b["updates"]][:4]}
    kd = res.get("muon_cooldown_show_time_kd_replay", {})
    plain = res.get("muon_cooldown_show_time", {})
    kd_differs = kd.get("ok") and plain.get("ok") and kd["losses"] != plain["losses"]
    rep.check("C2a_trainer_sm92_tree_equals_this_tree", plain.get("ok", False) and kd.get("ok", False) and bool(kd_differs),
              users=[r["user_id"] for r in rows], probes=res, kd_changes_the_loss=kd_differs)

    # C2b: CatchUp / resume_point, 9c125341's copies vs this tree's
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
        fake(base / f"sm118-c2-{tag}", users)
    for done, end, snap in (([11, 12], 1040, True), ([13], 1060, True), ([], 0, False)):
        outs = {}
        for tag, mod in (("old", OD), ("new", D)):
            out = base / f"sm118-c2-{tag}"
            interrupt(out, done, end, snap)
            mod.resume_point(out, {"track": trk})
            outs[tag] = {f: (out / f).read_text().replace(f"sm118-c2-{tag}", "X") for f in ("protocol.json", "resume.json", "protocol.prepared.json")}
        eq = outs["old"] == outs["new"]
        same_files &= eq
        p = json.loads((base / "sm118-c2-new" / "protocol.json").read_text())
        a, b = OO.CatchUp({}, p["full_run"], p["rows"]), CatchUp({}, p["full_run"], p["rows"])
        t = sum(r["train_range"][1] - r["train_range"][0] for r in p["rows"])
        lr_same &= lr_probe(a, t) == lr_probe(b, t)
        steps.append({"done": done, "files_equal": eq, "lr_resume": p["full_run"].get("lr_resume")})
    real = {}
    for name, f in (("sm92_prepared", DEC92 / "protocol.prepared.json"), ("sm92_protocol_resumed", DEC92 / "protocol.json"),
                    ("recipe", RECIPE / "protocol.json")):
        p = json.loads(f.read_text())
        opt = json.loads((DEC92 / "decay.json").read_text())["options"]
        a, b = OO.CatchUp(opt, p["full_run"], p["rows"]), CatchUp(opt, p["full_run"], p["rows"])
        t = sum(r["train_range"][1] - r["train_range"][0] for r in p["rows"])
        real[name] = {"equal": lr_probe(a, t, 200) == lr_probe(b, t, 200), "lr_resume": p["full_run"].get("lr_resume"),
                      "total": b.schedule.total}
        lr_same &= real[name]["equal"]
    for tag in ("old", "new"):
        shutil.rmtree(base / f"sm118-c2-{tag}", ignore_errors=True)
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
    base = ROOT / "artifacts/small-model/smoke/sm118-c3"
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
    d92 = json.loads((DEC92 / "decay.json").read_text())
    r92 = json.loads((DEC92 / "expertium" / TRACK / "result.json").read_text())
    t, e26 = sha256(TEACHER), sha256(E26 / "expertium" / TRACK / "master.pth")
    m91 = json.loads((ST91 / "manifest.json").read_text())
    final = sha256(DEC92 / "expertium" / TRACK / "master.pth")
    resume = json.loads((DEC92 / "resume.json").read_text())
    stop1 = json.loads((DEC92 / "attempts/attempt-1/expertium" / TRACK / "snapshots/users-03256/snapshot.json").read_text())
    snap = json.loads((SNAP92 / "snapshot.json").read_text())
    upd = [json.loads(x) for x in open(DEC92 / "expertium" / TRACK / "updates.jsonl", encoding="utf-8") if x.strip()]
    lr_snap = upd[int(snap["updates"]) - 1]["lr"] / 7e-4
    cum = int(stop1["review_presentations"]) + int(snap["review_presentations"])
    total = int(d92["training_reviews_after_trims"])
    done = set(stop1["done_users"]) | set(snap["done_users"])
    attempts = sorted(p.name for p in (DEC92 / "attempts").iterdir())
    snap_ok = (sha256(SNAP92 / "master.pth") == SNAP92_SHA and len(done) == 4056 == len(stop1["done_users"]) + len(snap["done_users"])
               and stop1["reason"] == "stop" and len(stop1["done_users"]) == 3256 and cum == 288_456_043
               and resume["folder"].endswith("attempt-1/expertium/E89-F65e-F77a/snapshots/users-03256") and attempts == ["attempt-1"]
               and abs(lr_snap - 0.29774) < 1e-4 and abs(lr_snap - (1 - math.sin(math.pi / 2 * cum / total))) < 2e-4)
    run = json.loads(RUN8K.read_text())
    old = json.loads((OLD / "status.json").read_text())
    rtotal = sum(run["review_counts"][str(u)] for u in run["train_users"])
    users92 = {r["user_id"] for r in json.loads((DEC92 / "protocol.prepared.json").read_text())["rows"]}
    st91_users = {int(p.stem) for p in (ST91 / "users").glob("*.npz")}
    new_users = {int(p.stem) for p in (NEW / "users").glob("*.npz")} if (NEW / "users").exists() else set()
    ok = (t == TEACHER_SHA == m91["teacher_sha256"] and e26 == E26_SHA == d92["start_master_sha256"]
          and r92.get("status") == "complete" and not r92.get("smoke_only") and final == FINAL92_SHA and snap_ok
          and len(run["train_users"]) == 8000 == old["users"] == old["complete_users"] and rtotal == old["reviews_total"] == 581_942_738
          and set(run["train_users"]) == users92 and st91_users <= users92 and len(st91_users) == 1991 and new_users <= users92)
    rep.check("C4_inputs_pinned", ok, teacher=t[:12], st91_manifest_teacher=m91["teacher_sha256"][:12], epoch26=e26[:12],
              sm92_cooldown=r92.get("status"), sm92_final=final[:12],
              early_stop_reference={"snapshot": SNAP92.relative_to(DEC92).as_posix(), "master": SNAP92_SHA[:12], "attempts": attempts,
                                    "users_done": len(done), "reviews_done": cum, "of": total, "fraction": round(cum / total, 5),
                                    "last_update_lr_over_peak": round(lr_snap, 5),
                                    "formula_lr_over_peak": round(1 - math.sin(math.pi / 2 * cum / total), 5), "ok": snap_ok},
              run_users=len(run["train_users"]), run_reviews=rtotal, st91_users=len(st91_users), new_targets_users_now=len(new_users))


def load_targets(folder, user):
    import numpy as np
    return np.load(Path(folder) / "users" / f"{user}.npz", allow_pickle=False)["targets"].astype("float32")


def target_gap(a, b):
    """the exporter's tolerance between two exports of the same teacher (SM110's C4b): curve probability, rating float16 steps."""
    import numpy as np
    g = ~np.isnan(b)
    prob = lambda t: np.clip(1 / (1 + np.exp(-t[:, 0])), 1e-5, 1 - 1e-5)  # noqa: E731  the exporter's decode()
    curve = float(np.abs(prob(a) - prob(b))[g[:, 0]].max()) if g[:, 0].any() else 0.0
    ulp = np.spacing(np.abs(b[:, 1:]).astype(np.float16)).astype(np.float32)
    steps = float((np.abs(a[:, 1:] - b[:, 1:]) / ulp)[g[:, 1:]].max()) if g[:, 1:].any() else 0.0
    return {"same_nan": bool((np.isnan(a) == np.isnan(b)).all()), "curve_max_abs": curve, "rating_max_float16_steps": steps,
            "bitwise": bool(np.array_equal(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9)))}


def c5(rep, work):
    import numpy as np
    out = work / "replay-sm91s2-e2s-cpu"
    shutil.rmtree(out, ignore_errors=True)
    cmd = [PY, "-B", "-X", "utf8", str(ROOT / "scripts/sm11_replay_export.py"), "export", "--out", str(out), "--run", str(RUN8K),
           "--teacher", TEACHER.as_posix(), "--inputs", "e2s-query", "--device", "cpu", "--workers", "1",
           "--users", ",".join(map(str, USERS_C5))]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT, env=env_cpu(RWKV_DELETED_CARDS="neighbour"), capture_output=True, text=True)
    run = {"exit": r.returncode, "seconds": round(time.perf_counter() - t0, 1), "tail": (r.stdout + r.stderr)[-400:]}
    ok, rows = r.returncode == 0, {}
    if ok:
        mn, m91 = (json.loads((f / "manifest.json").read_text()) for f in (out, ST91))
        diff = sorted(k for k in set(mn) | set(m91) if mn.get(k) != m91.get(k))
        ok &= diff == []
        for u in USERS_C5:
            new, old, st = load_targets(out, u), load_targets(OLD, u), load_targets(ST91, u)
            gap = target_gap(new, st)
            same_nan_old = bool((np.isnan(new) == np.isnan(old)).all())
            finite = bool(np.isfinite(new[~np.isnan(new)]).all())
            moved_old = int(np.nan_to_num(np.abs(new - old) > 1e-3).any(axis=-1).sum())
            ok &= gap["same_nan"] and same_nan_old and finite and gap["curve_max_abs"] <= 2.2e-4 and gap["rating_max_float16_steps"] <= 1.0 and moved_old > 0
            rows[u] = {"reviews": int(new.shape[0]), "cpu_vs_sm91s3_gpu": gap, "same_nan_as_old": same_nan_old, "finite": finite,
                       "rows_moved_vs_old_targets": moved_old,
                       "mean_abs_p_again_change_vs_old": round(float(np.nanmean(np.abs(new[:, 1] - old[:, 1]))), 5),
                       "mean_abs_curve_logit_change_vs_old": round(float(np.nanmean(np.abs(new[:, 0] - old[:, 0]))), 5)}
        rows["manifest_differences_vs_sm91s3"] = diff
    rep.check("C5_exporter_reproduces_sm91s3_targets_on_cpu", ok, run=run, users=rows)


def mutated_pair_p1(tmp, src, fn, name):
    import sm118_pair
    d = Path(tmp) / f"mut-p1-{name}" / src.name
    shutil.rmtree(d.parent, ignore_errors=True)
    d.mkdir(parents=True)
    p = json.loads((src / "protocol.json").read_text())
    y = json.loads((src / "decay.json").read_text())
    p["full_run"]["control_dir"] = (SM / d.name / "control").relative_to(ROOT).as_posix()
    y.pop("sm118_smoke0", None)
    y["options"]["distillation"]["replay"] = str(NEW).replace("\\", "/")
    fn(p, y)
    (d / "protocol.json").write_text(json.dumps(p)); (d / "decay.json").write_text(json.dumps(y))
    return sm118_pair.pair_p1(d, DEC92, P113)


def c6(rep, tmp):
    import sm118_pair
    from scripts.expertium_options import CatchUp
    frec, irec = sm118_pair.pair_recipe(RECIPE, DEC92)
    self_ref, _ = sm118_pair.pair_p1(DEC92, DEC92, P113)
    fail, info = mutated_pair_p1(tmp, SMOKE0, lambda p, y: None, "as_pass1")
    f0, _ = sm118_pair.pair_p1(SMOKE0, DEC92, P113)
    smoke_only_replay = bool(f0) and all("sm118_smoke0" in x or "targets are not" in x for x in f0)
    d92 = json.loads((DEC92 / "decay.json").read_text())
    neg = {}
    for name, fn in (("lr_offset", lambda p, y: p["full_run"]["lr_resume"].update(reviews_offset=1)),
                     ("no_segment", lambda p, y: p["full_run"]["lr_resume"].pop("segment")),
                     ("old_targets", lambda p, y: y["options"].update(distillation=d92["options"]["distillation"])),
                     ("st91_replay", lambda p, y: y["options"]["distillation"].update(replay=str(ST91))),
                     ("curve_weight_0", lambda p, y: y["options"].update(curve_loss_weight=0.0)),
                     ("curve_weight_1", lambda p, y: y["options"].update(curve_loss_weight=1.0)),
                     ("decay_alpha", lambda p, y: y["options"]["distillation"].update(decay_alpha=0.6)),
                     ("show_time_off", lambda p, y: y["options"].pop("show_time_intervals")),
                     ("order", lambda p, y: p["rows"].insert(0, p["rows"].pop(5))),
                     ("sm113_start", lambda p, y: (p["full_run"].update(init_master=json.loads((P113 / "protocol.prepared.json").read_text())["full_run"]["init_master"]),
                                                   y.update(start_master_sha256=json.loads((P113 / "decay.json").read_text())["start_master_sha256"])))):
        f2, _ = mutated_pair_p1(tmp, SMOKE0, fn, name)
        neg[name] = bool(f2)
    p = json.loads((SMOKE0 / "protocol.json").read_text())
    c = CatchUp(json.loads((SMOKE0 / "decay.json").read_text())["options"], p["full_run"], p["rows"])
    R = sum(r["train_range"][1] - r["train_range"][0] for r in p["rows"])
    lr0, lrR = c.learning_rate(10, p["full_run"]["update_offset"], 0), c.learning_rate(10, p["full_run"]["update_offset"], R)
    mu = subprocess.run([PY, "-X", "utf8", str(ROOT / "scripts/check_muon_fresh_decay.py"), "--decay", str(SMOKE0),
                         "--out", str(HERE / "gpu/check-muon-fresh-smoke0.json")], capture_output=True, text=True,
                        env=env_cpu(RWKV_DELETED_CARDS="neighbour"))
    lw = json.loads((HERE / "gpu/check-replay-label-windows-recipe.json").read_text())
    sm = [json.loads(x) for x in (HERE / "gpu/smoke0.out").read_text().splitlines() if x.startswith('{"smoke"')]
    smoke_ok = bool(sm) and sm[-1].get("exit") == 0 and sm[-1].get("status") == "complete" and sm[-1].get("reviews") == 1024
    ok = (not frec and bool(self_ref) and not fail and smoke_only_replay and all(neg.values()) and c.schedule.total == 2 * R == 1_163_885_448
          and lr0 == 7e-4 and abs(lrR / 7e-4 - (1 - math.sin(math.pi / 4))) < 1e-9 and mu.returncode == 0 and lw.get("ok") is True
          and lw.get("trims_recorded") == [5599] and smoke_ok)
    rep.check("C6_recipe_smoke_and_pass1_folders", ok, recipe=frec[:4] or irec, sm92_as_pass1_refused=bool(self_ref),
              smoke0_as_pass1=fail[:4] or info, smoke0_differs_only_by_replay_and_note=smoke_only_replay, mutated_refused=neg,
              schedule_total=c.schedule.total, lr_start=lr0, lr_at_R_over_peak=lrR / 7e-4, muon_fresh_exit=mu.returncode,
              label_windows=lw.get("ok"), trims=lw.get("trims_recorded"), gpu_smoke0=sm[-1] if sm else None)


def c7(rep, tmp):
    import sm118_pair
    from scripts.expertium_options import CatchUp
    base = ROOT / "artifacts/small-model/smoke"
    a, b = base / "sm118-c7-p1", base / "sm118-c7-p2"
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)
    a.mkdir(parents=True)
    for f in ("protocol.json", "decay.json"):
        shutil.copy(SMOKE0 / f, a / f)
    case = a / "expertium" / TRACK
    case.mkdir(parents=True)
    p = json.loads((a / "protocol.json").read_text())
    p["full_run"]["control_dir"] = (a / "control").relative_to(ROOT).as_posix()
    (a / "protocol.json").write_text(json.dumps(p, indent=1), encoding="utf-8")
    R = sum(r["train_range"][1] - r["train_range"][0] for r in p["rows"])
    (case / "master.pth").write_bytes(b"fake master"); (case / "optimizer.pth").write_bytes(b"fake optimizer")
    json.dump({"status": "complete", "update_offset_end": 596000, "review_presentations": R,
               "trained_parameter_sha256": "f" * 64}, open(case / "result.json", "w"))
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm118_prepare.py"), "p2", "--p1", str(a), "--out", str(b)],
                       capture_output=True, text=True)
    fail, info = sm118_pair.pair_p2(b, a) if r.returncode == 0 else (["prepare failed: " + r.stderr[-300:]], "")
    neg = {}

    def mutate(name, fn):
        d = base / f"sm118-c7-mut-{name}"
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(b, d)
        q = json.loads((d / "protocol.json").read_text()); y = json.loads((d / "decay.json").read_text())
        fn(q, y)
        (d / "protocol.json").write_text(json.dumps(q)); (d / "decay.json").write_text(json.dumps(y))
        f2, _ = sm118_pair.pair_p2(d, a)
        neg[name] = bool(f2)
        shutil.rmtree(d, ignore_errors=True)
    mutate("same_order", lambda q, y: q.update(rows=json.loads((a / "protocol.json").read_text())["rows"]))
    mutate("lr_offset", lambda q, y: q["full_run"]["lr_resume"].update(reviews_offset=0))
    mutate("other_start", lambda q, y: q["full_run"].update(init_master=str(E26 / "expertium" / TRACK / "master.pth")))
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
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm118_rule.py"), "selftest"], capture_output=True, text=True)
    rep.check("C8_rule_selftest", r.returncode == 0 and r.stdout.strip().endswith("OK"), out=r.stdout.strip()[-400:], err=r.stderr[-400:])


def c9(rep, work):
    import torch
    users = work / "users-c9.json"
    users.write_text(json.dumps({"users": list(USERS_C9)}))
    final = DEC92 / "expertium" / TRACK / "master.pth"
    a = torch.load(final, map_location="cpu", weights_only=True)
    b = torch.load(SNAP92 / "master.pth", map_location="cpu", weights_only=True)
    same_layout = list(a) == list(b) and all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype for k in a)
    got = {}
    for name, ck in (("sm92-final", final), ("sm92-u4056", SNAP92 / "master.pth")):
        out = work / f"c9-{name}.jsonl"
        out.unlink(missing_ok=True)
        cmd = [PY, "-B", "-X", "utf8", str(ROOT / "scripts/small_model_heldout.py"), "--pool", "732", "--users-file", str(users),
               "--device", "cpu", "--model", f"{name}=expertium:w64:{ck}", "--only", f"{name}/w8+s8", "--inputs", "e2s-query",
               "--split-existing", "--deleted-cards", "neighbour", "--out", str(out)]
        r = subprocess.run(cmd, cwd=ROOT, env=env_cpu(), capture_output=True, text=True)
        rows = [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
        got[name] = {x["user_id"]: x["benchmark"] for x in rows if "benchmark" in x}
        if r.returncode or len(got[name]) != len(USERS_C9):
            got[name + "_error"] = (r.stdout + r.stderr)[-1200:]
    ref = {}
    for line in ROWS92.read_text(encoding="utf-8").splitlines():
        x = json.loads(line)
        if x.get("label") == LBL92 and x["user_id"] in USERS_C9 and "benchmark" in x:
            ref[x["user_id"]] = x["benchmark"]
    gaps, moved = {}, {}
    ok = same_layout and all(len(got.get(n, {})) == len(USERS_C9) for n in ("sm92-final", "sm92-u4056")) and len(ref) == len(USERS_C9)
    if ok:
        for u in USERS_C9:
            gaps[u] = max(abs(got["sm92-final"][u][h]["LogLoss"] - ref[u][h]["LogLoss"]) for h in ("ahead", "instant"))
            moved[u] = {h: (got["sm92-u4056"][u][h]["LogLoss"], got["sm92-final"][u][h]["LogLoss"]) for h in ("ahead", "instant")}
        ok = all(g <= 1e-6 for g in gaps.values()) and all(
            math.isfinite(v[h][0]) and v[h][0] != v[h][1] for v in moved.values() for h in ("ahead", "instant"))
    rep.check("C9_scorer_reproduces_sm92_rows_and_scores_the_snapshot", ok, same_state_dict_layout=same_layout,
              max_abs_logloss_vs_committed_rows=gaps, snapshot_vs_final=moved,
              errors={k: v for k, v in got.items() if k.endswith("_error")})


def c10(rep, tmp):
    def block(now, quiet):
        return subprocess.run([PY, "-X", "utf8", str(HERE / "sm118_block.py"), "--now", now, "--quiet", str(quiet)],
                              capture_output=True, text=True).stdout.strip()
    empty = Path(tmp) / "quiet-empty.json"
    empty.write_text("[]", encoding="utf-8")
    got = {"mon_08:00": block("2026-11-16T08:00", empty), "mon_20:30": block("2026-11-16T20:30", empty),
           "tue_01:00": block("2026-11-17T01:00", empty), "tue_19:00": block("2026-11-17T19:00", empty),
           "sat_15:00": block("2026-11-21T15:00", empty), "sun_15:00": block("2026-11-22T15:00", empty),
           "sun_16:30": block("2026-11-22T16:30", empty), "real_now": block(time.strftime("%Y-%m-%dT%H:%M"), QUIET)}
    want = {"mon_08:00": "NEXT 720", "mon_20:30": "NOW 16260", "tue_01:00": "NEXT 1140", "tue_19:00": "NEXT 60",
            "sat_15:00": "NEXT 300", "sun_15:00": "NEXT 60", "sun_16:30": "NOW 30660"}
    src = (HERE / "sm118_block.py").read_text(encoding="utf-8").split('"""\nimport argparse', 1)[1]
    ref = (SM / "sm115/sm115_block.py").read_text(encoding="utf-8").split('"""\nimport argparse', 1)[1]
    rep.check("C10_breaks", all(got[k] == v for k, v in want.items()) and src == ref and got["real_now"][:4] in ("NEXT", "NOW "),
              got=got, code_equals_sm115_block=src == ref)


def c11(rep):
    free = shutil.disk_usage("C:/").free / 2 ** 30
    rep.check("C11_disk", free >= 20, free_gib=round(free, 1), new_targets_folder_exists=NEW.exists())


def targets(out_path, replay=NEW, users=None):
    """After the export: every user of the 8k run, new vs old targets (shape, NaN pattern, finite, rating sums), and new vs
    SM91 step 3's targets of the same teacher for its 1,991 users (the exporter's tolerance). --replay / --users only for
    testing the check on a small export (the launcher passes neither)."""
    import numpy as np
    run = json.loads(RUN8K.read_text())
    want = users or run["train_users"]
    st91 = {int(p.stem) for p in (ST91 / "users").glob("*.npz")}
    bad, n, moved_sum, moved_n, t0 = [], 0, 0.0, 0, time.perf_counter()
    same = {"users": 0, "bitwise": 0, "curve_max_abs": 0.0, "rating_max_float16_steps": 0.0}
    for u in want:
        try:
            new, old = load_targets(replay, u), load_targets(OLD, u)
        except (OSError, KeyError, ValueError) as error:
            bad.append({"user_id": u, "error": str(error)[:200]}); continue
        if new.shape != old.shape:
            bad.append({"user_id": u, "error": f"shape {new.shape} vs {old.shape}"}); continue
        if not (np.isnan(new) == np.isnan(old)).all():
            bad.append({"user_id": u, "error": "NaN pattern differs from the old targets"}); continue
        m = ~np.isnan(new)
        if not np.isfinite(new[m]).all():
            bad.append({"user_id": u, "error": "non-finite target"}); continue
        r = ~np.isnan(new[:, 1])
        if r.any() and np.abs(new[r, 1:].sum(-1) - 1).max() > 5e-3:
            bad.append({"user_id": u, "error": f"rating sums off by {float(np.abs(new[r, 1:].sum(-1) - 1).max()):.4f}"}); continue
        if u in st91:
            g = target_gap(new, load_targets(ST91, u))
            same["users"] += 1; same["bitwise"] += int(g["bitwise"])
            same["curve_max_abs"] = max(same["curve_max_abs"], g["curve_max_abs"])
            same["rating_max_float16_steps"] = max(same["rating_max_float16_steps"], g["rating_max_float16_steps"])
            if not g["same_nan"] or g["curve_max_abs"] > 2.2e-4 or g["rating_max_float16_steps"] > 1.0:
                bad.append({"user_id": u, "error": f"differs from SM91 step 3's targets of the same teacher: {g}"}); continue
        moved_sum += float(np.abs(new[r, 1] - old[r, 1]).sum()); moved_n += int(r.sum()); n += 1
    out = {"replay": str(replay), "users": len(want), "users_ok": n, "failures": bad[:50], "failures_count": len(bad),
           "vs_sm91s3_same_teacher": same, "mean_abs_p_again_change_vs_old": round(moved_sum / max(moved_n, 1), 5),
           "rating_rows": moved_n, "seconds": round(time.perf_counter() - t0, 1),
           "ok": not bad and n == len(want) and (users is not None or (n == 8000 and same["users"] == 1991))}
    Path(out_path).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[sm118-check] targets: {json.dumps({k: v for k, v in out.items() if k != 'failures'})}")
    return 0 if out["ok"] else 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "targets":
        ap = ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--out", default=str(HERE / "gpu" / "check-targets-sm118.json"))
        ap.add_argument("--replay", default=str(NEW))
        ap.add_argument("--users", default="")
        a = ap.parse_args()
        return targets(a.out, Path(a.replay), [int(u) for u in a.users.split(",")] if a.users else None)
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "check-cpu-sm118.json"))
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
    print(f"[sm118-check] {'ALL OK' if out['all_ok'] else ('partial ' + ('OK' if rep.ok else 'FAILED'))} in {out['minutes']} min")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
