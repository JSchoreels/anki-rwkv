#!/usr/bin/env python
"""The big instant-only teacher (SM110): CPU proofs before any GPU ticket (design.md in this folder; ~5-15 min), and the
check of the exported targets after the export (`targets`, run by the launcher).

  python -X utf8 artifacts/small-model/sm110/sm110_check.py [--out check-cpu-sm110.json] [--only C4,C5]
  python -X utf8 artifacts/small-model/sm110/sm110_check.py targets [--out gpu/check-targets-sm110.json]

The run: the Small Instant-Only model (SM98)'s cool-down rerun from its epoch 35 with only the KD replay targets swapped
for the show-time instant-only big model (SM99)'s, replayed on show-time inputs (scripts/sm11_replay_export.py --inputs
e2s-query, the exporter of the show-time interval retrain (SM91) step 3). No source file changes.

  C1 code: scripts/ expertium/ rwkv/ trees at HEAD == 1e22b62c (the trees SM96 / SM98 / SM99 / SM101 trained and scored
     with; the exporter's --inputs e2s-query is in them) and nothing uncommitted there.
  C2 inputs, all read only: the teacher (SM99's final epoch 2) sha256 == the release package's master.pth; SM98's epoch
     35 master sha256 == the start SM98's cool-down recorded; SM98's cool-down complete; SM98's cool-down snapshot
     users-04000 present (4,000 users done, both master.pth and snapshot.json); the 8k run file == the old targets' user
     set and review total (8,000 users, 581,942,738 reviews).
  C3 the cool-down recipe (sm77_warm.py cooldown from SM98's epoch 35 with extra-options.json, never trained) == SM98's
     cool-down except the teacher, the replay folder and paths (sm110_pair.py --untrimmed); mutated copies refused
     (decay alpha, start sha256, shuffle seed, curve loss weight, the show-time flag, the old targets).
  C4 the exporter on CPU, users 4001 and 8081 (in the cool-down and in the show-time teacher targets (SM91 step 3)):
     C4a SM99 on show-time inputs: the manifest == SM91 step 3's show-time manifest except the teacher and its sha256;
         per user the NaN pattern == the old targets' and SM91 step 3's (so the label windows and trims stay the same);
         every non-NaN value finite; the targets move against both (counts reported);
     C4b the show-time big model (SM91 step 2) on show-time inputs with this tree on CPU reproduces SM91 step 3's GPU
         targets for these users within the exporter's verify tolerance (decoded as ReplayTargets feeds the loss: curve
         probability 2.2e-4; ratings at most one float16 step apart, both tables being float16 roundings): the export path
         is the one that already ran.
  C5 the scorer on CPU (users 4676 and 2388 of the 732 set, w8+s8, show-time inputs, --split-existing --deleted-cards
     neighbour): SM98's final master.pth reproduces SM98's committed rows (instant and curve LogLoss within 1e-6); SM98's
     snapshot users-04000 has the final master's keys, shapes and dtypes and scores the same users (finite, different).
  C6 rule: sm110_rule.py selftest.
  C7 breaks: sm110_block.py on fixed times and on the real quiet.json (Thu 11-12 evening's break lifted).
  C8 resources: >= 20 GB free on C: for the new targets (~5.5 GB) and the cool-down (snapshots ~3 MB each).
targets (after the export): for each of the 8,000 users, the new file's shape == the old targets', the NaN pattern ==
  the old targets' (curve and rating columns), every other value finite, ratings summing to 1 within float16 rounding;
  summary of how far the new rating targets moved (mean abs change of P(again)).
"""
from argparse import ArgumentParser
import hashlib
import json
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
CACHE = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay")
TEACHER = W / "sm99/artifacts/small-model/sm99-big-inst-e2/expertium/big-inst/master.pth"
TEACHER_SHA = "c49e45d698194da033caa1231f598f875922a4a2a3e921714a3fd433d37e5768"
PACKAGE = W / "big-package/artifacts/models/sm99/master.pth"
SHOWBIG = W / "sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth"
SHOWBIG_SHA = "67ee8e9b0d0169733242c40646e0d2e5637845a25775762002ffc4e6396f9590"
TRACK = "E89-F65e-F77a"
E35 = W / "sm98/artifacts/small-model/sm98-inst-e35-st"
E35_SHA = "f7577d14ba102ab9ce1a4e995f9176854f1aa719a627209244d54ec02536ebd4"
DEC98 = W / "sm98/artifacts/small-model/sm98-inst-e35st-st-dec-1ep32k-muon"
SNAP98 = DEC98 / "expertium" / TRACK / "snapshots/users-04000"
RUN8K = W / "sm55/artifacts/small-model/sm55-final8k-run.json"
OLD = CACHE / "big-leakfree-1ep-final8k-neighbour"
ST91 = CACHE / "big-showtime-cool1991-neighbour-e2s"
NEW = CACHE / "sm99-big-inst-final8k-neighbour-e2s"
RECIPE = SM / "sm110-recipe-e35st-dec-1ep32k-muon-t99"
ROWS98 = SM / "sm98/gpu/sm98-rows.jsonl"
LBL98 = "sm98-inst-e35st-st-dec1ep32k/w8+s8@e2s-query"
U732 = W / "sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json"
USERS_C4, USERS_C5 = (4001, 8081), (4676, 2388)
QUIET = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json")


class Report:
    def __init__(self):
        self.checks, self.ok = {}, True

    def check(self, name, ok, **detail):
        self.checks[name] = {"ok": bool(ok), **detail}
        self.ok &= bool(ok)
        print(f"[sm110-check] {name}: {'OK' if ok else 'FAIL'} {json.dumps(detail, default=str)[:400]}", flush=True)


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


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


def c1(rep):
    trees = {p: (git("rev-parse", f"HEAD:{p}"), git("rev-parse", f"{CODE}:{p}")) for p in ("scripts", "expertium", "rwkv")}
    dirty = git("status", "--porcelain", "--", "scripts", "expertium", "rwkv")
    rep.check("C1_code_trees_equal_1e22b62c", all(a == b for a, b in trees.values()) and not dirty,
              trees={k: v[0][:12] for k, v in trees.items()}, uncommitted=dirty.splitlines()[:10])


def c2(rep):
    t = sha256(TEACHER)
    pkg = sha256(PACKAGE)
    e35 = sha256(E35 / "expertium" / TRACK / "master.pth")
    d98 = json.loads((DEC98 / "decay.json").read_text())
    r98 = json.loads((DEC98 / "expertium" / TRACK / "result.json").read_text())
    snap = json.loads((SNAP98 / "snapshot.json").read_text()) if (SNAP98 / "snapshot.json").exists() else {}
    run = json.loads(RUN8K.read_text())
    old = json.loads((OLD / "status.json").read_text())
    total = sum(run["review_counts"][str(u)] for u in run["train_users"])
    ok = (t == TEACHER_SHA == pkg and e35 == E35_SHA == d98["start_master_sha256"] and r98.get("status") == "complete"
          and not r98.get("smoke_only") and snap.get("progress_users") == 4000 and len(snap.get("done_users", [])) == 4000
          and (SNAP98 / "master.pth").exists() and len(run["train_users"]) == 8000 == old["users"] == old["complete_users"]
          and total == old["reviews_total"] == 581_942_738)
    rep.check("C2_inputs_pinned", ok, teacher=t[:12], package=pkg[:12], epoch35=e35[:12],
              sm98_cooldown_start=d98["start_master_sha256"][:12], sm98_cooldown_status=r98.get("status"),
              snapshot_users_04000={**{k: snap.get(k) for k in ("progress_users", "review_presentations", "update_offset_end")}, "done_users": len(snap.get("done_users", []))},
              run_users=len(run["train_users"]), run_reviews=total, old_targets=(old["users"], old["reviews_total"]))


def c3(rep, tmp):
    sys.path.insert(0, str(HERE))
    import sm110_pair
    fail, info = sm110_pair.pair(RECIPE, DEC98, untrimmed=True)
    negatives = {}

    def mutated(name, fn):
        d = Path(tmp) / f"neg-{name}" / RECIPE.name
        shutil.copytree(RECIPE, d, ignore=shutil.ignore_patterns("expertium", "control", "attempts"))
        dec = json.loads((d / "decay.json").read_text())
        fn(dec)
        (d / "decay.json").write_text(json.dumps(dec, indent=1))
        f, _ = sm110_pair.pair(d, DEC98, untrimmed=True)
        negatives[name] = bool(f)
    mutated("decay_alpha_0.9", lambda d: d["options"]["distillation"].update(decay_alpha=0.9))
    mutated("start_sha", lambda d: d.update(start_master_sha256="0" * 64))
    mutated("shuffle_seed", lambda d: d.update(shuffle_seed=2))
    mutated("curve_weight_1", lambda d: d["options"].update(curve_loss_weight=1.0))
    mutated("show_time_off", lambda d: d["options"].pop("show_time_intervals"))
    mutated("old_targets", lambda d: d["options"].update(distillation=json.loads((DEC98 / "decay.json").read_text())["options"]["distillation"]))
    self_ref, _ = sm110_pair.pair(DEC98, DEC98)
    rep.check("C3_recipe_equals_sm98_cooldown_except_targets", not fail and all(negatives.values()) and bool(self_ref),
              pair=info, failures=fail[:6], negatives_refused=negatives, sm98_against_itself_refused=bool(self_ref))


def load_targets(folder, user):
    import numpy as np
    return np.load(Path(folder) / "users" / f"{user}.npz", allow_pickle=False)["targets"].astype("float32")


def c4(rep, work):
    import numpy as np
    out_new, out_91 = work / "replay-sm99-e2s", work / "replay-sm91s2-e2s"
    users = ",".join(map(str, USERS_C4))
    res = {}
    for out, teacher in ((out_new, TEACHER), (out_91, SHOWBIG)):
        shutil.rmtree(out, ignore_errors=True)
        cmd = [PY, "-B", "-X", "utf8", str(ROOT / "scripts/sm11_replay_export.py"), "export", "--out", str(out), "--run", str(RUN8K),
               "--teacher", str(teacher), "--inputs", "e2s-query", "--device", "cpu", "--workers", "1", "--users", users]
        t0 = time.perf_counter()
        r = subprocess.run(cmd, cwd=ROOT, env=env_cpu(RWKV_DELETED_CARDS="neighbour"), capture_output=True, text=True)
        res[out.name] = {"exit": r.returncode, "seconds": round(time.perf_counter() - t0, 1), "tail": (r.stdout + r.stderr)[-600:]}
    ok_run = all(v["exit"] == 0 for v in res.values())
    rows, ok_a, ok_b = {}, ok_run, ok_run
    if ok_run:
        mn, m91 = (json.loads((f / "manifest.json").read_text()) for f in (out_new, ST91))
        diff = sorted(k for k in set(mn) | set(m91) if mn.get(k) != m91.get(k))
        ok_a &= diff == ["teacher", "teacher_sha256"] and mn["teacher_sha256"] == TEACHER_SHA and mn.get("inputs") == "e2s-query"
        for u in USERS_C4:
            new, old, st = load_targets(out_new, u), load_targets(OLD, u), load_targets(ST91, u)
            rep91 = load_targets(out_91, u)
            same_nan = bool((np.isnan(new) == np.isnan(old)).all() and (np.isnan(new) == np.isnan(st)).all())
            m = ~np.isnan(new)
            finite = bool(np.isfinite(new[m]).all())
            moved_old = int((np.abs(new - old) > 1e-3).any(axis=-1).sum())   # NaN compares False
            moved_91 = int(np.nan_to_num(np.abs(new[:, 1:] - st[:, 1:]) > 1e-3).any(axis=-1).sum())
            same91 = bool((np.isnan(rep91) == np.isnan(st)).all())
            g = ~np.isnan(st)
            prob = lambda t: np.clip(1 / (1 + np.exp(-t[:, 0])), 1e-5, 1 - 1e-5)  # noqa: E731  the exporter's decode()
            curve_gap = float(np.abs(prob(rep91) - prob(st))[g[:, 0]].max()) if g[:, 0].any() else 0.0
            rating_gap = float(np.abs(rep91[:, 1:] - st[:, 1:])[g[:, 1:]].max()) if g[:, 1:].any() else 0.0
            # both tables are float16 roundings of float32 values (CPU here, GPU there): allow one float16 step per value
            ulp = np.spacing(np.abs(st[:, 1:]).astype(np.float16)).astype(np.float32)
            rating_steps = float((np.abs(rep91[:, 1:] - st[:, 1:]) / ulp)[g[:, 1:]].max()) if g[:, 1:].any() else 0.0
            ok_a &= same_nan and finite and moved_91 > 0 and moved_old > 0
            ok_b &= same91 and curve_gap <= 2.2e-4 and rating_steps <= 1.0
            rows[u] = {"reviews": int(new.shape[0]), "labelled_rating_rows": int((~np.isnan(new[:, 1])).sum()),
                       "same_nan_pattern_as_old_and_sm91s3": same_nan, "finite": finite,
                       "rating_rows_moved_vs_sm91s3_targets": moved_91, "rows_moved_vs_old_targets": moved_old,
                       "mean_abs_p_again_change_vs_old": round(float(np.nanmean(np.abs(new[:, 1] - old[:, 1]))), 5),
                       "mean_abs_p_again_change_vs_sm91s3": round(float(np.nanmean(np.abs(new[:, 1] - st[:, 1]))), 5),
                       "sm91s2_cpu_vs_sm91s3_gpu": {"same_nan": same91, "curve_max_abs": curve_gap, "rating_max_abs": rating_gap,
                                                    "rating_max_float16_steps": rating_steps}}
        rows["manifest_differences_vs_sm91s3"] = diff
    rep.check("C4a_sm99_show_time_targets_on_cpu", ok_a, runs=res, users=rows)
    rep.check("C4b_this_tree_reproduces_sm91s3_gpu_targets", ok_b,
              users={u: rows.get(u, {}).get("sm91s2_cpu_vs_sm91s3_gpu") for u in USERS_C4})


def c5(rep, work):
    import torch
    users = work / "users-c5.json"
    users.write_text(json.dumps({"users": list(USERS_C5)}))
    final = DEC98 / "expertium" / TRACK / "master.pth"
    a = torch.load(final, map_location="cpu", weights_only=True)
    b = torch.load(SNAP98 / "master.pth", map_location="cpu", weights_only=True)
    same_layout = list(a) == list(b) and all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype for k in a)
    got = {}
    for name, ck in (("sm98-final", final), ("sm98-u4000", SNAP98 / "master.pth")):
        out = work / f"c5-{name}.jsonl"
        out.unlink(missing_ok=True)
        cmd = [PY, "-B", "-X", "utf8", str(ROOT / "scripts/small_model_heldout.py"), "--pool", "732", "--users-file", str(users),
               "--device", "cpu", "--model", f"{name}=expertium:w64:{ck}", "--only", f"{name}/w8+s8", "--inputs", "e2s-query",
               "--split-existing", "--deleted-cards", "neighbour", "--out", str(out)]
        r = subprocess.run(cmd, cwd=ROOT, env=env_cpu(), capture_output=True, text=True)
        rows = [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
        got[name] = {x["user_id"]: x["benchmark"] for x in rows if "benchmark" in x}
        if r.returncode or len(got[name]) != len(USERS_C5):
            got[name + "_error"] = (r.stdout + r.stderr)[-1200:]
    ref = {}
    for line in ROWS98.read_text(encoding="utf-8").splitlines():
        x = json.loads(line)
        if x.get("label") == LBL98 and x["user_id"] in USERS_C5 and "benchmark" in x:
            ref[x["user_id"]] = x["benchmark"]
    gaps, moved = {}, {}
    ok = same_layout and all(len(got.get(n, {})) == len(USERS_C5) for n in ("sm98-final", "sm98-u4000"))
    if ok:
        for u in USERS_C5:
            gaps[u] = max(abs(got["sm98-final"][u][h]["LogLoss"] - ref[u][h]["LogLoss"]) for h in ("ahead", "instant"))
            moved[u] = {h: (got["sm98-u4000"][u][h]["LogLoss"], got["sm98-final"][u][h]["LogLoss"]) for h in ("instant",)}
        ok = all(g <= 1e-6 for g in gaps.values()) and all(v["instant"][0] == v["instant"][0] and v["instant"][0] != v["instant"][1]
                                                             for v in moved.values())
    rep.check("C5_scorer_reproduces_sm98_rows_and_scores_the_snapshot", ok, same_state_dict_layout=same_layout,
              max_abs_logloss_vs_committed_rows=gaps, snapshot_vs_final_instant=moved,
              errors={k: v for k, v in got.items() if k.endswith("_error")})


def c6(rep):
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm110_rule.py"), "selftest"], capture_output=True, text=True)
    rep.check("C6_rule_selftest", r.returncode == 0, out=r.stdout.strip()[-1500:], err=r.stderr[-800:])


def c7(rep, tmp):
    def block(now, quiet=None):
        args = [PY, "-X", "utf8", str(HERE / "sm110_block.py"), "--now", now] + (["--quiet", str(quiet)] if quiet else [])
        return subprocess.run(args, capture_output=True, text=True).stdout.strip()
    empty = Path(tmp) / "quiet-empty.json"
    empty.write_text("[]", encoding="utf-8")
    q = Path(tmp) / "quiet-test.json"
    q.write_text(json.dumps([{"start": "2026-11-13T21:00", "end": "2026-11-14T01:00", "scope": "all"}]), encoding="utf-8")
    got = {"thu_12:00_default": block("2026-11-12T12:00", empty), "thu_21:00_default": block("2026-11-12T21:00", empty),
           "sun_15:00": block("2026-11-15T15:00", empty), "fri_20:30_quiet_21:00": block("2026-11-13T20:30", q),
           "real_thu_12:00": block("2026-11-12T12:00", QUIET), "real_thu_21:00": block("2026-11-12T21:00", QUIET),
           "real_fri_00:57": block("2026-11-13T00:57", QUIET), "real_fri_00:59": block("2026-11-13T00:59", QUIET),
           "real_fri_01:00": block("2026-11-13T01:00", QUIET), "real_now": block(time.strftime("%Y-%m-%dT%H:%M"), QUIET)}
    want = {"thu_12:00_default": "NEXT 480", "thu_21:00_default": "NOW 14460", "sun_15:00": "NEXT 60",
            "fri_20:30_quiet_21:00": "NEXT 30", "real_thu_12:00": "NEXT 779", "real_thu_21:00": "NEXT 239",
            "real_fri_00:57": "NEXT 2", "real_fri_00:59": "NOW 120", "real_fri_01:00": "NEXT 1140"}
    src = (HERE / "sm110_block.py").read_text(encoding="utf-8").split('"""\nimport argparse', 1)[1]
    ref = (W / "sm106/artifacts/small-model/sm106/sm106_block.py").read_text(encoding="utf-8").split('"""\nimport argparse', 1)[1]
    rep.check("C7_breaks", all(got[k] == v for k, v in want.items()) and src == ref, got=got, want=want,
              code_equals_sm106_block=src == ref)


def c8(rep):
    free = shutil.disk_usage("C:/").free / 2 ** 30
    rep.check("C8_disk", free >= 20, free_gib=round(free, 1), new_targets_folder_exists=NEW.exists())


def targets(out_path, replay=NEW, users=None):
    """After the export: every user of the 8k run, new vs old targets (shape, NaN pattern, finite, rating sums).
    --replay / --users only for testing the check on a small export (the launcher passes neither)."""
    import numpy as np
    run = json.loads(RUN8K.read_text())
    want = users or run["train_users"]
    bad, n, moved_sum, moved_n, t0 = [], 0, 0.0, 0, time.perf_counter()
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
        moved_sum += float(np.abs(new[r, 1] - old[r, 1]).sum()); moved_n += int(r.sum()); n += 1
    out = {"replay": str(replay), "users": len(want), "users_ok": n, "failures": bad[:50], "failures_count": len(bad),
           "mean_abs_p_again_change_vs_old": round(moved_sum / max(moved_n, 1), 5), "rating_rows": moved_n,
           "seconds": round(time.perf_counter() - t0, 1), "ok": not bad and n == len(want) and (users is not None or n == 8000)}
    Path(out_path).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[sm110-check] targets: {json.dumps({k: v for k, v in out.items() if k != 'failures'})}")
    return 0 if out["ok"] else 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "targets":
        ap = ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--out", default=str(HERE / "gpu" / "check-targets-sm110.json"))
        ap.add_argument("--replay", default=str(NEW))
        ap.add_argument("--users", default="")
        a = ap.parse_args()
        return targets(a.out, Path(a.replay), [int(u) for u in a.users.split(",")] if a.users else None)
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "check-cpu-sm110.json"))
    ap.add_argument("--work", default=str(HERE / "work"))
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    want = lambda c: only is None or c in only  # noqa: E731
    rep = Report()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        for name, fn in (("C1", lambda: c1(rep)), ("C2", lambda: c2(rep)), ("C3", lambda: c3(rep, tmp)),
                         ("C4", lambda: c4(rep, work)), ("C5", lambda: c5(rep, work)), ("C6", lambda: c6(rep)),
                         ("C7", lambda: c7(rep, tmp)), ("C8", lambda: c8(rep))):
            if want(name):
                try:
                    fn()
                except Exception as error:   # a crashed check is a failed check
                    rep.check(f"{name}_crashed", False, error=repr(error)[:600])
    out = {"commit": git("rev-parse", "--short", "HEAD"), "minutes": round((time.perf_counter() - t0) / 60, 1),
           "all_ok": rep.ok and only is None, "partial": sorted(only) if only else None, "checks": rep.checks}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"[sm110-check] {'ALL OK' if rep.ok else 'FAILED'}: " + ", ".join(f"{k} {'OK' if v['ok'] else 'FAIL'}" for k, v in rep.checks.items()))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
