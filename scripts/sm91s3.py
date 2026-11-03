#!/usr/bin/env python
"""Show-time interval retrain (SM91) step 3: CPU proofs before the teacher-target export and the cool-down
(artifacts/small-model/sm91s3/DESIGN.md).

New targets: the show-time big model (the big model (1 leak-free epoch) + one show-time epoch, step 2) replays the full
histories of the 1,991 users of step 1's quarter-epoch 32k Muon cool-down on show-time inputs
(scripts/sm11_replay_export.py --inputs e2s-query, RWKV_DELETED_CARDS=neighbour). New cool-down: step 1's arm
cool-down, rerun from step 1's arm epoch 21 with those targets; nothing else changes.

  cpu-check [--out artifacts/small-model/sm91s3/check-cpu-sm91s3.json]   (CPU, a few minutes, run by the launcher)
    E0 teacher: the show-time big model file loads in the exporter's fork model (float32); sha256 recorded.
    E1 exporter batches == the duration-leak scoring (SM31) scorer's batches, bitwise in every tensor, for 3 cool-down
       users (8,623 / 8,630 / 24,292 reviews): --inputs e2s-query vs sm31_inputs.samples(..., "e2s-query", "neighbour"),
       and --inputs normal vs the scorer's normal samples (the step 1 proofs showed the trainer's show-time rows ==
       those scorer rows, so the teacher reads exactly the student's inputs).
    E2 targets: the show-time big model on CPU over the exporter's batches (bitwise the scorer's, E1): the e2s-query
       table covers exactly the expected reviews, has the NaN pattern of the normal one (the same reviews get targets)
       and different curve and rating values (the transform reaches the targets).
    E3 the export command end to end on CPU (--device cpu --workers 1: the spawned loader worker path) for 2 users:
       written tables == E2's tables bitwise; manifest = the old targets' manifest except teacher, teacher_sha256
       (== the file) and "inputs": "e2s-query"; `verify --inputs e2s-query` passes, `verify` without it is refused.
       Normal inputs + the old teacher (the big model (1 leak-free epoch)) for 1 user: manifest == the old targets'
       manifest exactly, table == the old cache's table within the verify tolerance (curve 2.2e-4, rating 3.5e-4;
       the old one ran on the GPU), same NaN pattern: the flag-off exporter is the old exporter.
    E4 coverage plan: artifacts/small-model/sm91s3/cool-users-run.json lists exactly the new cool-down's 1,991 users
       with review counts == their available_reviews, from the old targets' dataset (path, revision); every
       train_range lies inside [0, available_reviews]. (After the export the launcher runs
       check_replay_label_windows.py on the new cool-down: every user's file present, rows == reviews, NaN pattern ==
       the export record, no unlabelled window.)
    E5 config: sm91s3_pair.py: the new cool-down == step 1's arm cool-down except the distillation teacher and replay
       folder and its own folder name (same start, users, order, update offset, schedule, seeds, options, flag);
       step 1's arm cool-down against itself is refused.
    E6 code: scripts/ rwkv/ expertium/ unchanged since step 1's launch commit 4f39eb92 (the commit that trained step 1's
       arm cool-down and scored its rows) except files the cool-down and the scorer never import
       (sm11_replay_export.py, sm65_rule.py, sm72_big_epoch.py, sm91s2.py, sm91s3.py).
    E7 rule: artifacts/small-model/sm91s3/sm91s3_rule.py self-test.
"""
from argparse import ArgumentParser
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
os.environ.setdefault("GPU_BATCH1_DATA_PATH", str(DATA))
os.environ.setdefault("RWKV_NO_JIT", "1")
PY = sys.executable
S = ROOT / "artifacts/small-model/sm91s3"
NEW = ROOT / "artifacts/small-model/sm91s3-arm-e21-dec-q32k-muon-stt"
REF = Path("C:/Users/jonat/workspace/wt/sm91/artifacts/small-model/sm91-arm-e21-dec-q32k-muon")      # read only
SHOWBIG = Path("C:/Users/jonat/workspace/wt/sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth")
BIG1 = Path("C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72-big-leakfree-1ep/expertium/big-leakfree-1ep/master.pth")
OLD = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/big-leakfree-1ep-final8k-neighbour")  # read only
RUN = S / "cool-users-run.json"
USERS = (8081, 4001, 5804)
STEP1_COMMIT = "4f39eb92"
UNUSED = {"scripts/sm11_replay_export.py", "scripts/sm65_rule.py", "scripts/sm72_big_epoch.py", "scripts/sm91s2.py",
          "scripts/sm91s3.py"}


class Report:
    def __init__(self):
        self.checks, self.ok = {}, True

    def check(self, name, ok, **detail):
        self.checks[name] = {"ok": bool(ok), **detail}
        self.ok &= bool(ok)
        print(f"[sm91s3-check] {name}: {'OK' if ok else 'FAIL'} {json.dumps(detail, default=str)[:300]}", flush=True)


def walk(x, path="batch"):
    import torch
    if isinstance(x, torch.Tensor):
        head = str((x.dtype, tuple(x.shape))).encode()
        yield path, hashlib.sha256(head + (x.detach().contiguous().view(torch.uint8).numpy().tobytes() if x.numel() else b"")).hexdigest()
    elif dataclasses.is_dataclass(x) and not isinstance(x, type):
        for f in dataclasses.fields(x):
            yield from walk(getattr(x, f.name), f"{path}.{f.name}")
    elif isinstance(x, dict):
        for k in sorted(x, key=str):
            yield from walk(x[k], f"{path}[{k}]")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from walk(v, f"{path}[{i}]")
    elif hasattr(x, "tolist") and hasattr(x, "dtype"):
        yield path, hashlib.sha256(repr((str(x.dtype), x.tolist())).encode()).hexdigest()
    elif hasattr(x, "__dict__") and not isinstance(x, type):
        for k in sorted(vars(x)):
            yield from walk(vars(x)[k], f"{path}.{k}")
    else:
        yield path, hashlib.sha256(repr(x).encode()).hexdigest()


def set_env(name, value):
    if value:
        os.environ[name] = value
    else:
        os.environ.pop(name, None)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scorer_batch(user, variant):
    from rwkv.prepare_batch import prepare
    from scripts import sm31_inputs
    set_env("RWKV_DELETED_CARDS", None)
    s = sm31_inputs.samples(DATA, user, variant, "neighbour")
    assert len(s) == 1
    return prepare([s[0]], seed=2025 + user)


def exporter_batch(user, variant):
    from scripts import sm11_replay_export as ex
    set_env("RWKV_DELETED_CARDS", "neighbour")
    set_env("SM11_INPUTS", variant)
    try:
        frame = ex.load_frame(DATA, user)
        batch, own = ex.prepare_pass(frame, user, 0, len(frame))
        return batch, own, ex.expected_masks(frame)
    finally:
        set_env("RWKV_DELETED_CARDS", None)


def table(model, batch, own, reviews):
    import numpy as np
    import torch
    from scripts import sm11_replay_export as ex
    with torch.inference_mode():
        stats = model.get_loss(batch)
    out = np.full((reviews, 5), np.nan, dtype=np.float16)
    ex.fill(out, stats, own, 0, reviews)
    return out


def e1_e2(rep, model):
    import numpy as np
    # the scorer first: the exporter's e2s-query patch stays installed in this process once used
    scorer = {(u, v): dict(walk(scorer_batch(u, v))) for u in USERS for v in ("normal", "e2s-query")}
    exp, tables = {}, {}
    for v in ("normal", "e2s-query"):
        for u in USERS:
            batch, own, masks = exporter_batch(u, v)
            exp[(u, v)] = dict(walk(batch))
            tables[(u, v)] = (table(model, batch, own, len(masks[0])), masks)
    for v in ("normal", "e2s-query"):
        diff = {u: sorted(k for k in set(scorer[(u, v)]) | set(exp[(u, v)]) if scorer[(u, v)].get(k) != exp[(u, v)].get(k))
                for u in USERS}
        rep.check(f"E1_exporter_batches_equal_scorer_{v}", not any(diff.values()) and all(len(exp[(u, v)]) > 5 for u in USERS),
                  fields=len(exp[(USERS[0], v)]), differing={u: d[:8] for u, d in diff.items() if d})
    moved = {u: sorted(k for k in exp[(u, "normal")] if exp[(u, "normal")][k] != exp[(u, "e2s-query")][k]) for u in USERS}
    rep.check("E1b_e2s_query_changes_the_batch", all(moved.values()), moved={u: m[:6] for u, m in moved.items()})
    # E2 (the scorer batches are bitwise the exporter's, E1, so the teacher's table is the same): NaN pattern, values move
    detail, ok = {}, True
    for u in USERS:
        te, (want_curve, want_rating) = tables[(u, "e2s-query")]
        tn, _ = tables[(u, "normal")]
        same_holes = np.array_equal(np.isnan(te), np.isnan(tn))
        cover = np.array_equal(~np.isnan(te[:, 0]), want_curve) and np.array_equal(~np.isnan(te[:, 1]), want_rating)
        changed_curve = int((np.nan_to_num(te[:, 0]) != np.nan_to_num(tn[:, 0])).sum())
        changed_rating = int((np.nan_to_num(te[:, 1:]) != np.nan_to_num(tn[:, 1:])).any(1).sum())
        good = same_holes and cover and changed_curve > 0 and changed_rating > 0
        ok &= good
        detail[u] = {"reviews": int(te.shape[0]), "curve_targets": int(want_curve.sum()), "rating_targets": int(want_rating.sum()),
                     "same_nan_pattern_as_normal": same_holes, "coverage_equals_masks": cover,
                     "curve_targets_changed": changed_curve, "rating_targets_changed": changed_rating}
    rep.check("E2_e2s_tables_cover_the_same_reviews_and_move", ok, users=detail)
    return tables


def e3(rep, tables, work):
    import numpy as np
    out_e = work / "replay-e2s"
    out_n = work / "replay-normal"
    users = USERS[:2]
    env = dict(os.environ, RWKV_DELETED_CARDS="neighbour", CUDA_VISIBLE_DEVICES="")
    env.pop("SM11_INPUTS", None)
    ex = str(ROOT / "scripts/sm11_replay_export.py")
    base = ["--run", str(RUN), "--data", str(DATA)]
    p = subprocess.run([PY, "-X", "utf8", ex, "export", "--out", str(out_e), *base, "--users", ",".join(map(str, users)),
                        "--teacher", SHOWBIG.as_posix(), "--device", "cpu", "--workers", "1", "--inputs", "e2s-query", "--threads", "2"],
                       capture_output=True, text=True, env=env)
    if p.returncode not in (0, 3):
        rep.check("E3_export_e2s_cpu", False, err=(p.stdout + p.stderr)[-1500:])
        return
    same, detail = True, {}
    for u in users:
        with np.load(out_e / "users" / f"{u}.npz") as f:
            got = f["targets"]
            meta = json.loads(f["meta"].tobytes().decode())
        want = tables[(u, "e2s-query")][0]
        eq = got.shape == want.shape and np.array_equal(got.view(np.uint16), want.view(np.uint16))
        same &= eq
        detail[u] = {"bitwise_equal": eq, "teacher_sha256": meta["teacher_sha256"][:12]}
    rep.check("E3a_export_e2s_tables_equal_e2_bitwise", same, users=detail)
    m_new = json.loads((out_e / "manifest.json").read_text())
    m_old = json.loads((OLD / "manifest.json").read_text())
    sha = sha256(SHOWBIG)
    diff = sorted(k for k in set(m_new) | set(m_old) if m_new.get(k) != m_old.get(k))
    rep.check("E3b_manifest_old_except_teacher_and_inputs",
              set(diff) == {"teacher", "teacher_sha256", "inputs"} and m_new["teacher_sha256"] == sha and m_new["inputs"] == "e2s-query"
              and m_new.get("deleted_cards") == "neighbour", differing=diff, teacher_sha256=sha)
    v_ok = subprocess.run([PY, "-X", "utf8", ex, "verify", "--out", str(out_e), *base, "--users", ",".join(map(str, users)),
                           "--teacher", SHOWBIG.as_posix(), "--inputs", "e2s-query", "--threads", "2"], capture_output=True, text=True, env=env)
    v_bad = subprocess.run([PY, "-X", "utf8", ex, "verify", "--out", str(out_e), *base, "--users", ",".join(map(str, users)),
                            "--teacher", SHOWBIG.as_posix(), "--threads", "2"], capture_output=True, text=True, env=env)
    rep.check("E3c_verify_e2s_passes_and_normal_verify_refused", v_ok.returncode == 0 and v_bad.returncode != 0,
              verify=v_ok.stdout.strip()[-300:], refused=(v_bad.stdout + v_bad.stderr).strip()[-200:])
    u = USERS[0]
    p = subprocess.run([PY, "-X", "utf8", ex, "export", "--out", str(out_n), *base, "--users", str(u), "--teacher", BIG1.as_posix(),
                        "--device", "cpu", "--workers", "1", "--threads", "2"], capture_output=True, text=True, env=env)
    if p.returncode not in (0, 3):
        rep.check("E3d_export_normal_cpu", False, err=(p.stdout + p.stderr)[-1500:])
        return
    m_n = json.loads((out_n / "manifest.json").read_text())
    with np.load(out_n / "users" / f"{u}.npz") as f:
        got = f["targets"].astype(np.float32)
    with np.load(OLD / "users" / f"{u}.npz") as f:
        old = f["targets"].astype(np.float32)
    holes = got.shape == old.shape and np.array_equal(np.isnan(got), np.isnan(old))
    valid = ~np.isnan(old[:, 0])
    pc = lambda t: 1 / (1 + np.exp(-t[:, 0]))  # noqa: E731
    cgap = float(np.abs(pc(got) - pc(old))[valid].max()) if holes else None
    rgap = float(np.nanmax(np.abs(got[:, 1:] - old[:, 1:]))) if holes else None
    rep.check("E3d_flag_off_export_reproduces_old_targets", m_n == m_old and holes and cgap <= 2.2e-4 and rgap <= 3.5e-4,
              manifest_equal=m_n == m_old, same_nan_pattern=holes, curve_max_abs=cgap, rating_max_abs=rgap, user=u)


def e4(rep):
    run = json.loads(RUN.read_text())
    rows = json.loads((NEW / "protocol.json").read_text())["rows"]
    old = json.loads((OLD / "manifest.json").read_text())
    by = {r["user_id"]: r for r in rows}
    ok = (sorted(run["train_users"]) == sorted(by) and len(run["train_users"]) == 1991
          and all(run["review_counts"][str(u)] == by[u]["available_reviews"] for u in by)
          and all(0 <= r["train_range"][0] <= r["train_range"][1] <= r["available_reviews"] for r in rows)
          and Path(run["dataset_path"]) == Path(old["dataset_path"]) and run["dataset_revision"] == old["dataset_revision"])
    rep.check("E4_export_users_equal_the_cooldown_users", ok, users=len(run["train_users"]),
              reviews=sum(run["review_counts"].values()), trained=sum(r["train_range"][1] - r["train_range"][0] for r in rows))


def e5(rep):
    pair = S / "sm91s3_pair.py"
    r = subprocess.run([PY, "-X", "utf8", str(pair), "--arm", str(NEW), "--ref", str(REF)], capture_output=True, text=True)
    rep.check("E5a_new_cooldown_equals_step1_arm_cooldown_except_targets", r.returncode == 0, out=r.stdout.strip()[-400:])
    r = subprocess.run([PY, "-X", "utf8", str(pair), "--arm", str(REF), "--ref", str(REF)], capture_output=True, text=True)
    rep.check("E5b_step1_against_itself_refused", r.returncode == 1, out=r.stdout.strip()[-300:])


def e6(rep):
    r = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", STEP1_COMMIT, "--", "scripts", "rwkv", "expertium"],
                       capture_output=True, text=True, check=True)
    changed = sorted(set(r.stdout.split()))
    other = [f for f in changed if f not in UNUSED]
    rep.check("E6_trainer_and_scorer_code_unchanged_since_step1", not other, changed=changed, other=other)


def e7(rep):
    r = subprocess.run([PY, "-X", "utf8", str(S / "sm91s3_rule.py"), "self-test"], capture_output=True, text=True)
    rep.check("E7_rule_self_test", r.returncode == 0, out=r.stdout.strip()[-800:], err=r.stderr[-500:])


def cpu_check(args):
    import torch
    torch.set_num_threads(2)
    from scripts import sm11_replay_export  # noqa: F401  (installs the state-quant model before rwkv.model is imported)
    from scripts import compress_quant as cq
    rep = Report()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    sha = sha256(SHOWBIG)
    base = {k: v.float() for k, v in torch.load(SHOWBIG, map_location="cpu", weights_only=True).items()}
    model = cq.set_state_bits(cq.load_model(state=base, reference_wkv=True), 0).eval()
    rep.check("E0_teacher_loads", True, path=str(SHOWBIG), sha256=sha, tensors=len(base))
    tables = e1_e2(rep, model)
    del model
    e3(rep, tables, work)
    e4(rep)
    e5(rep)
    e6(rep)
    e7(rep)
    from scripts.experiment_provenance import process_provenance
    out = {"ok": rep.ok, "checks": rep.checks, "users": list(USERS), "teacher_sha256": sha, "provenance": process_provenance()}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"[sm91s3-check] {'ALL OK' if rep.ok else 'FAIL'}: " + ", ".join(f"{k} {'OK' if v['ok'] else 'FAIL'}" for k, v in rep.checks.items()))
    return 0 if rep.ok else 1


def main():
    ap = ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("cpu-check")
    p.add_argument("--out", default=str(S / "check-cpu-sm91s3.json"))
    p.add_argument("--work", default=str(S / "work"))
    args = ap.parse_args()
    return cpu_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
