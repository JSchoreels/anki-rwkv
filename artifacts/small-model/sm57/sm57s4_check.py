#!/usr/bin/env python
"""The instant-only small model (SM57) stage 4: CPU proofs (DESIGN.md, "Stage 4 plan") -> check-cpu-sm57-stage4.json.

  V1 the epoch builder (sm57_stage4.build) reproduces stage 1's instant-only epoch 26 from SM72s epoch 25 (protocol.json
     equal except benchmark_db, relative in the run vs resolved in stage 1: the same file; decay options equal), and the
     run's own epochs 22-25 from 21-24 (rows, trims, offsets, fields; paths compared resolved).
  V2 epoch 27 prepared for real (sm57s4-inst-e27) passes `pair`; refusals: epoch 26's rows, the flag removed, a wrong
     update offset, a start from SM72s epoch 25 instead of stage 1's epoch 26.
  V3 the epoch orders 21..30 are 10 distinct permutations of the same 8,000 train users.
  V4 sources: since stage 1's launch commit (08a9fbd8) only scripts/eval_pools.py and scripts/small_model_heldout.py
     change (the 1,500-user pool option of SM93, 24bd3c68); neither is in the trainer's SourceWatch set; the 732 pool is
     unchanged (pool_users(732) == eval-users-732.json, the 732 fit list inside it) and pool 1500 == eval-users-1500.json.
  V5 the cool-down prepared from stage 1's epoch 26 by this tree (sm77_warm.py cooldown --epoch 25 + the label-window
     trim pass) == stage 1's instant-only cool-down (sm57_pair.py decay --same-start, no flag difference): the same
     recipe and code path the launcher uses from epoch 30.
  V6 the rule and pool code reproduce the published pairs (sm57_stage4.py self-test).
  V7 the break logic on synthetic times (Saturday 20:00, Sunday 16:00, a quiet.json window replacing a night).
"""
from argparse import ArgumentParser
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import sm57_stage4 as S  # noqa: E402

PY = sys.executable
CHECKS = {}


def record(name, ok, **info):
    CHECKS[name] = {"ok": bool(ok), **info}
    print(f"[sm57s4-check] {name}: {'OK' if ok else 'FAIL'} {json.dumps(info)[:600]}", flush=True)


def norm(p, root):
    q = Path(p)
    return (q if q.is_absolute() else root / q).resolve().as_posix()


def v1():
    run = S.RUN72S
    root = run.parents[4]
    built, decay = S.build(26, prev=run / "epoch-25", out=S.E26)
    disk = json.loads((S.E26 / "protocol.json").read_text())
    diff = [p for p, x, y in S.leaves(built, disk) if not (p == "/full_run/benchmark_db" and norm(x, root) == norm(y, root))]
    d26 = json.loads((S.E26 / "decay.json").read_text())
    record("V1a_builder_reproduces_stage1_epoch26", not diff and decay["options"] == d26["options"]
           and decay["training_reviews"] == d26["training_reviews"] and decay["init_master_sha256"] == d26["init_master_sha256"]
           and decay["init_optimizer_sha256"] == d26["init_optimizer_sha256"], differences=diff[:5])
    bad = {}
    for n in range(22, 26):
        b, _ = S.build(n, prev=run / f"epoch-{n - 1:02d}", out=S.E26)
        d = json.loads((run / f"epoch-{n:02d}" / "protocol.json").read_text())
        diffs = []
        for p, x, y in S.leaves(b, d):
            if p in ("/full_run/init_master", "/full_run/init_optimizer", "/full_run/benchmark_db") and norm(x, root) == norm(y, root):
                continue
            if p == "/full_run/control_dir":
                continue
            diffs.append(p)
        if diffs:
            bad[n] = diffs[:5]
    record("V1b_builder_reproduces_sm72s_epochs_22_25", not bad, differences=bad)


def v2():
    d = S.folder(27)
    if not d.exists():
        r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm57_stage4.py"), "prepare", "--epoch", "27"], capture_output=True, text=True)
        if r.returncode:
            record("V2_epoch27_prepared_and_paired", False, error=r.stdout[-400:] + r.stderr[-400:])
            return
    fail, info = S.pair_fail(d, 27)
    record("V2a_epoch27_prepared_and_paired", not fail, info=info, failures=fail[:5])
    refusals = {}
    base_p = json.loads((d / "protocol.json").read_text())
    base_d = json.loads((d / "decay.json").read_text())
    p26 = json.loads((S.E26 / "protocol.json").read_text())

    def mutated(name, fp=None, fd=None, prev=None):
        t = S.SM / f"sm57s4-check-{name}"
        if t.exists():
            shutil.rmtree(t)
        t.mkdir()
        p, dd = json.loads(json.dumps(base_p)), json.loads(json.dumps(base_d))
        p["full_run"]["control_dir"] = (t / "control").relative_to(S.ROOT).as_posix()
        if fp:
            fp(p)
        if fd:
            fd(dd)
        S.atomic_json(t / "protocol.json", p)
        S.atomic_json(t / "decay.json", dd)
        fail, _ = S.pair_fail(t, 27, prev)
        shutil.rmtree(t)
        refusals[name] = bool(fail)

    mutated("rows26", fp=lambda p: p.__setitem__("rows", p26["rows"]))
    mutated("noflag", fd=lambda d: d["options"].pop("curve_loss_weight"))
    mutated("flag05", fd=lambda d: d["options"].__setitem__("curve_loss_weight", 0.5))
    mutated("offset", fp=lambda p: p["full_run"].__setitem__("update_offset", p["full_run"]["update_offset"] + 1))
    mutated("start25", fp=lambda p: p["full_run"].update(init_master=p26["full_run"]["init_master"],
                                                         init_optimizer=p26["full_run"]["init_optimizer"]))
    record("V2b_pair_refusals", all(refusals.values()), refusals=refusals)


def v3():
    config = S.run_config()
    orders = [tuple(S.order(n, config)) for n in range(21, 31)]
    same = all(sorted(o) == sorted(config["train_users"]) for o in orders)
    record("V3_orders_21_30_distinct_permutations", len(set(orders)) == 10 and same and len(orders[0]) == 8000,
           users=len(orders[0]))


def v4():
    out = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", "08a9fbd8", "--", "scripts", "expertium", "rwkv"],
                         capture_output=True, text=True).stdout.split()
    staged = subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--name-only", "08a9fbd8", "--", "scripts", "expertium", "rwkv"],
                            capture_output=True, text=True).stdout.split()
    changed = sorted(set(out) | set(staged))
    want = ["scripts/eval_pools.py", "scripts/small_model_heldout.py"]
    used = {"run_gpu_batch1.py", "gpu_batch1.py", "report_gpu_batch1.py", "benchmark_gpu_trial.py", "train_gpu_trial.py",
            "gpu_optimizations.py", "gpu_packed_encodings.py", "experiment_provenance.py", "trial_batches.py", "compare.py",
            "gpu_power.py"}
    src = (ROOT / "scripts/run_gpu_batch1.py").read_text(encoding="utf-8")
    watch_ok = all(f'"{n}"' in src for n in used) and not any(Path(w).name in used for w in want)
    from scripts.eval_pools import pool_users
    p732 = pool_users(732)
    f732 = json.loads((S.SM / "eval-users-732.json").read_text())["users"]
    fit = json.loads(Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json").read_text())["users"]
    p1500 = pool_users(1500)
    f1500 = json.loads(S.POOL1500.read_text())["users"]
    record("V4_sources_and_pools", changed == want and watch_ok and sorted(p732) == sorted(f732) and set(fit) <= set(p732)
           and sorted(p1500) == sorted(f1500) and len(p1500) == 1500 and set(p732) <= set(p1500),
           changed=changed, pool732=len(p732), pool1500=len(p1500))


def v5():
    t = S.SM / "sm57s4-check-dec"
    if t.exists():
        shutil.rmtree(t)
    extra = (HERE / "extra-options-instant.json").read_text()
    env = dict(os.environ, RWKV_DELETED_CARDS="neighbour")
    r = subprocess.run([PY, "-X", "utf8", str(ROOT / "scripts/sm77_warm.py"), "cooldown", "--epoch", "25", "--start", str(S.E26),
                        "--out", t.relative_to(ROOT).as_posix(), "--arch", "w64", "--fraction", "1.0", "--extra-options", extra],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    ok = r.returncode == 0
    trim = subprocess.run([PY, "-X", "utf8", str(ROOT / "scripts/check_replay_label_windows.py"), "--decay", str(t), "--trim-unlabelled-tail"],
                          capture_output=True, text=True, env=env, cwd=ROOT) if ok else None
    ok = ok and trim.returncode == 0
    pr = subprocess.run([PY, "-X", "utf8", str(HERE / "sm57_pair.py"), "decay", "--arm", str(t), "--ref", str(S.DEC1), "--same-start"],
                        capture_output=True, text=True, cwd=ROOT) if ok else None
    pr2 = subprocess.run([PY, "-X", "utf8", str(HERE / "sm57_pair.py"), "decay", "--arm", str(t),
                          "--ref", "C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon",
                          "--flag", json.dumps(S.FLAG)], capture_output=True, text=True, cwd=ROOT) if ok else None
    ok = ok and pr.returncode == 0 and pr2.returncode == 0
    record("V5_cooldown_prepare_reproduces_stage1", ok, prepare=r.stdout[-300:] + r.stderr[-300:] if r.returncode else r.stdout[-200:],
           vs_stage1=pr.stdout.strip()[-400:] if pr else None, vs_release=pr2.stdout.strip()[-300:] if pr2 else None)
    shutil.rmtree(t, ignore_errors=True)


def v6():
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm57_stage4.py"), "self-test"], capture_output=True, text=True, cwd=ROOT)
    record("V6_rule_self_test", r.returncode == 0, out=r.stdout.strip()[-400:])


def v7():
    from datetime import datetime
    tmp = Path(tempfile.mkdtemp())
    empty, moved = tmp / "q0.json", tmp / "q1.json"
    empty.write_text("[]", encoding="utf-8")
    moved.write_text(json.dumps([{"start": "2026-11-07T22:00", "end": "2026-11-08T00:30", "scope": "all"},
                                 {"start": "2026-11-07T12:00", "end": "2026-11-07T13:00", "scope": "timing"}]), encoding="utf-8")
    cases = [(empty, "2026-11-07T19:00", "NEXT 60"), (empty, "2026-11-07T21:00", f"NOW {4 * 3600 + 60}"),
             (empty, "2026-11-08T15:00", "NEXT 60"), (empty, "2026-11-08T17:00", f"NOW {8 * 3600 + 60}"),
             (empty, "2026-11-09T00:59", "NOW 120"), (empty, "2026-11-09T19:30", "NEXT 30"),
             (moved, "2026-11-07T20:30", "NEXT 90"), (moved, "2026-11-07T12:30", "NEXT 570"),
             (moved, "2026-11-08T00:40", "NEXT 920")]
    got = {f"{q.name} {t}": S.block_line(q, datetime.fromisoformat(t)) for q, _, t in [(c[0], None, c[1]) for c in cases]}
    bad = {k: v for (q, t, want), (k, v) in zip(cases, got.items()) if v != want}
    shutil.rmtree(tmp, ignore_errors=True)
    record("V7_break_logic", not bad, wrong=bad, cases=len(cases))


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "check-cpu-sm57-stage4.json"))
    args = ap.parse_args()
    t0 = time.time()
    for f in (v1, v2, v3, v4, v5, v6, v7):
        try:
            f()
        except Exception as error:   # noqa: BLE001
            record(f.__name__, False, error=repr(error)[:500])
    ok = all(c["ok"] for c in CHECKS.values())
    S.atomic_json(args.out, {"ok": ok, "checks": CHECKS, "seconds": round(time.time() - t0, 1),
                             "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(f"[sm57s4-check] {'ALL OK' if ok else 'FAIL'} ({len(CHECKS)} checks, {time.time() - t0:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
