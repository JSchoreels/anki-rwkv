#!/usr/bin/env python
"""The instant-only small model (SM57) stage 3 (curve-only): CPU proofs before the GPU (DESIGN.md, "Stage 3 plan").

  python -X utf8 artifacts/small-model/sm57/sm57s3_check.py [--out artifacts/small-model/sm57/check-cpu-sm57-stage3.json]

  T1 same init: the trainer's fresh init for the curve-only options has the tensor hash of the joint options, equal to
     stage 2's joint (and instant-only) epoch-1 initial_parameter_sha256 and to SM72s epoch 1's.
  T2 epoch-1 folder (sm57_stage3.py pair): sm57s3-curve-e01 differs from stage 2's sm57s2-joint-e01 only by
     {"instant_loss_weight": 0.0}, the folder names and the stage name; refused: joint against joint, stage 2's
     instant-only folder, a copy with two rows swapped, a copy with the flag at 0.5.
  T3 the option path in the source: instant_loss_weight is applied by scripts/gpu_r01_training.py whenever it is not
     None (so 0.0 is applied), it is teacher-blind (scripts/expertium_options.TEACHER_BLIND) and an accepted batch
     option of the ordered prefetch (it cannot change the batches), and the model uses it only as IMMEDIATE_SCALE, the
     factor of the instant average (whose rating target holds the instant KD targets) in the loss and the loss tensor;
     no other use in scripts/ expertium/ rwkv/.
  T4 trainer slice on CPU (sm57s3_probe.py: the trainer's batch path with the run's options and state carry, the run's
     real leak-free replay KD on the CPU, CatchUp.loss, clip, fresh AdamW, TorchScript) from the fresh init on 2 small
     users of epoch 1's order:
     a  the probe without the teacher == stage 2's sm57_probe.py (proved against the base tree in stage 2) bit for bit;
     b  same batches: curve-only and joint see the same updates (user, offset, reviews, seed, batch tensors digest);
     c  same first forward: at update 1 every loss component and both heads' predictions are bit for bit equal;
     d  the loss: curve-only's loss == the sum of the curve terms (the loss's own order and constants, no instant term)
        at every update, bit for bit, and at update 1 == the curve terms of joint's loss on the same batch; joint's loss
        == the same formula with the instant term (the formula is faithful);
     e  the instant KD is gone: reversing the teacher's rating targets leaves curve-only's losses, gradients and final
        weights bit for bit unchanged (its instant average does change), while it changes joint's;
     f  the curve KD stays: flipping the teacher's curve targets changes curve-only's loss;
     g  gradients: at the last update curve-only's all-zero gradients = joint's plus the instant-head-only parameters,
        disjoint from the curve-head-only parameters that instant-only (stage 2's flag) zeroes (update 1 is recorded
        too, but from the fresh init most output layers start at zero, so its gradients are sparse in every arm);
     h  curve-only's weights are finite and differ from joint's; the weight read back is 0.0; TorchScript model.
  T5 rule: sm57_stage3.py rule-self-test (the stage-3 comparison reproduces stage 2's published epoch-5 pair).
  T6 breaks: sm57_stage2.windows on synthetic quiet files (stage 2's cases) and on the live quiet.json now.
"""
from argparse import ArgumentParser
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("sm57_stage3", HERE / "sm57_stage3.py")
S3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S3)
S2 = S3.S2
PY = sys.executable
DATA = r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"
ENV = dict(os.environ, RWKV_DELETED_CARDS="neighbour", CUDA_VISIBLE_DEVICES="", GPU_BATCH1_DATA_PATH=DATA)


class Report:
    def __init__(self):
        self.checks, self.ok = {}, True

    def check(self, name, ok, **detail):
        self.checks[name] = {"ok": bool(ok), **detail}
        self.ok &= bool(ok)
        print(f"[{'ok' if ok else 'FAIL'}] {name}", flush=True)


def sh(*args, cwd=ROOT):
    return subprocess.run([PY, "-X", "utf8", *map(str, args)], capture_output=True, text=True, env=ENV, cwd=cwd)


def probe(script, spec_, tmp):
    import hashlib
    f = Path(tmp) / f"spec-{hashlib.sha256(json.dumps(spec_, sort_keys=True).encode()).hexdigest()[:12]}.json"
    f.write_text(json.dumps(spec_), encoding="utf-8")
    env = dict(ENV, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    env.pop("RWKV_NO_JIT", None)
    r = subprocess.run([PY, "-X", "utf8", str(HERE / script), str(ROOT), str(f)], cwd=ROOT, capture_output=True, text=True, env=env)
    line = next((x for x in r.stdout.splitlines() if x.startswith("PROBE ")), None)
    if line is None:
        raise RuntimeError(f"{script} failed: {r.stderr[-3000:]}")
    return json.loads(line[6:])


def source_uses():
    out = []
    for sub in ("scripts", "expertium", "rwkv"):
        for p in sorted((ROOT / sub).rglob("*.py")):
            if p.name.startswith("check_instant_loss_weight"):
                continue   # SM26's own checks
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if "instant_loss_weight" in line or "IMMEDIATE_SCALE" in line:
                    out.append((p.relative_to(ROOT).as_posix(), i, line.strip()))
    return out


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "check-cpu-sm57-stage3.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rep = Report()
    config = S2.run_config()
    curve1, joint1, inst1 = S3.folder(1), S2.folder("joint", 1), S2.folder("inst", 1)

    # T1
    r = sh(HERE / "sm57_stage3.py", "init-hash")
    rep.check("T1_same_fresh_init_curve_joint_stage2_sm72s", r.returncode == 0, out=r.stdout.strip()[-800:], err=r.stderr[-400:])

    # T2
    r = sh(HERE / "sm57_stage3.py", "pair", "--epoch", "1")
    rep.check("T2a_epoch1_curve_differs_from_stage2_joint_only_by_flag", r.returncode == 0, out=r.stdout.strip()[-400:], err=r.stderr[-400:])
    r = sh(HERE / "sm57_stage3.py", "pair", "--epoch", "1", "--arm-dir", joint1)
    rep.check("T2b_joint_against_joint_refused", r.returncode == 1, out=r.stdout.strip()[-300:])
    r = sh(HERE / "sm57_stage3.py", "pair", "--epoch", "1", "--arm-dir", inst1)
    rep.check("T2c_stage2_instant_only_refused", r.returncode == 1, out=r.stdout.strip()[-300:])
    with tempfile.TemporaryDirectory() as tmp:
        swapped = Path(tmp) / "swapped"
        shutil.copytree(curve1, swapped)
        p = json.loads((swapped / "protocol.json").read_text())
        p["rows"][0], p["rows"][1] = p["rows"][1], p["rows"][0]
        (swapped / "protocol.json").write_text(json.dumps(p), encoding="utf-8")
        r = sh(HERE / "sm57_stage3.py", "pair", "--epoch", "1", "--arm-dir", swapped)
        rep.check("T2d_swapped_rows_refused", r.returncode == 1, out=r.stdout.strip()[-300:])
        half = Path(tmp) / "half"
        shutil.copytree(curve1, half)
        d = json.loads((half / "decay.json").read_text())
        d["options"]["instant_loss_weight"] = 0.5
        (half / "decay.json").write_text(json.dumps(d), encoding="utf-8")
        r = sh(HERE / "sm57_stage3.py", "pair", "--epoch", "1", "--arm-dir", half)
        rep.check("T2e_flag_0.5_refused", r.returncode == 1, out=r.stdout.strip()[-300:])

    # T3
    uses = source_uses()
    trainer = (ROOT / "scripts/gpu_r01_training.py").read_text(encoding="utf-8")
    model_src = (ROOT / "expertium/model/variant_model.py").read_text(encoding="utf-8")
    loss_body = model_src[model_src.index("def _get_loss_variant"):model_src.index("def get_loss(")]
    imm_lines = [x.strip() for x in loss_body.splitlines() if "IMMEDIATE_SCALE" in x]
    from scripts.expertium_options import TEACHER_BLIND
    prefetch = (ROOT / "scripts/gpu_r01_prefetch.py").read_text(encoding="utf-8")
    allowed = prefetch[prefetch.index("def training_batches"):prefetch.index("Unsupported R01 ordered-prefetch")]
    expected_files = {"scripts/gpu_r01_training.py", "scripts/expertium_options.py", "scripts/gpu_r01_prefetch.py",
                      "expertium/model/variant_model.py", "expertium/model/srs_model.py"}
    rep.check("T3_option_path_in_source",
              re.search(r'if options\.get\("instant_loss_weight"\) is not None:\s*\n(?:\s*#.*\n)?\s*if not hasattr\(base, "instant_loss_weight"\):'
                        r'\s*\n\s*raise ValueError\([^)]*\)\s*\n\s*base\.instant_loss_weight = float\(options\["instant_loss_weight"\]\)', trainer)
              is not None
              and "instant_loss_weight" in TEACHER_BLIND and '"instant_loss_weight"' in allowed
              and imm_lines == ["IMMEDIATE_SCALE = self.instant_loss_weight", "+ IMMEDIATE_SCALE * immediate_avg", "+ IMMEDIATE_SCALE * p_loss.detach()"]
              and "kd_alpha2 * kd_p2" in loss_body and loss_body.count("kd_p2") == 2
              and {u[0] for u in uses} <= expected_files,
              uses=[f"{a}:{b}: {c[:140]}" for a, b, c in uses], loss_lines=imm_lines)

    # T4
    import torch
    with tempfile.TemporaryDirectory() as tmp:
        _, ck = S2.fresh_hash(config["options"])
        fresh = Path(tmp) / "fresh.pth"
        torch.save(ck["model"], fresh)
        protocol = json.loads((joint1 / "protocol.json").read_text())
        rows = sorted((r for r in protocol["rows"] if r["train_range"][1] >= 300), key=lambda r: r["train_range"][1])[:2]
        plain = {k: v for k, v in config["options"].items() if k not in ("packed_ipc", "distillation")}
        plain.update(chunk_reviews=128, batch_chunks=2)
        kd = {k: v for k, v in config["options"].items() if k != "packed_ipc"}
        kd.update(chunk_reviews=128, batch_chunks=2)
        base = {"arch": "w64", "checkpoint": str(fresh), "rows": rows, "span": 256}
        old = probe("sm57_probe.py", {**base, "options": plain}, tmp)
        new = probe("sm57s3_probe.py", {**base, "options": plain}, tmp)
        joint = probe("sm57s3_probe.py", {**base, "options": kd, "teacher": True}, tmp)
        curve = probe("sm57s3_probe.py", {**base, "options": {**kd, **S3.FLAG}, "teacher": True}, tmp)
        joint_r = probe("sm57s3_probe.py", {**base, "options": kd, "teacher": True, "perturb": "rating"}, tmp)
        curve_r = probe("sm57s3_probe.py", {**base, "options": {**kd, **S3.FLAG}, "teacher": True, "perturb": "rating"}, tmp)
        curve_c = probe("sm57s3_probe.py", {**base, "options": {**kd, **S3.FLAG}, "teacher": True, "perturb": "curve"}, tmp)
        inst = probe("sm57s3_probe.py", {**base, "options": {**kd, **S2.FLAG}, "teacher": True}, tmp)

    keys = ("user", "offset", "loss_hex", "imm_hex", "p_digest", "grad_digest", "zero_grad")
    view = lambda p: [{k: u[k] for k in keys} for u in p["updates"]]  # noqa: E731
    n = len(joint["updates"])
    rep.check("T4a_probe_without_teacher_equals_stage2_probe", view(old) == view(new) and old["weights"] == new["weights"]
              and len(new["updates"]) > len(rows), updates=len(new["updates"]))
    bkeys = ("user", "offset", "reviews", "seed", "batch_digest")
    rep.check("T4b_same_batches", n > len(rows) and joint["teacher"] == "ReplayTargets"
              and [[u[k] for k in bkeys] for u in curve["updates"]] == [[u[k] for k in bkeys] for u in joint["updates"]],
              updates=n, rows=[r["user_id"] for r in rows], teacher_stats=joint["teacher_stats"])
    j0, c0 = joint["updates"][0], curve["updates"][0]
    rep.check("T4c_same_first_forward", j0["components_hex"] == c0["components_hex"] and j0["p_digest"] == c0["p_digest"]
              and j0["curve_digest"] == c0["curve_digest"], components={k: float.fromhex(v) for k, v in j0["components_hex"].items()})
    rep.check("T4d_curve_only_loss_is_exactly_the_curve_terms",
              all(u["finite"] and u["loss_hex"] == u["recomputed_curve_terms_hex"] for u in curve["updates"])
              and all(u["loss_hex"] == u["recomputed_full_hex"] for u in joint["updates"])
              and c0["loss_hex"] == j0["recomputed_curve_terms_hex"] and c0["loss_hex"] != j0["loss_hex"],
              first_update={"joint_loss": float.fromhex(j0["loss_hex"]), "joint_curve_terms": float.fromhex(j0["recomputed_curve_terms_hex"]),
                            "curve_only_loss": float.fromhex(c0["loss_hex"]), "instant_average": float.fromhex(j0["imm_hex"])},
              curve_only_losses=[float.fromhex(u["loss_hex"]) for u in curve["updates"]])
    lkeys = ("loss_hex", "grad_digest", "curve_digest", "zero_grad")
    rep.check("T4e_instant_kd_removed",
              [[u[k] for k in lkeys] for u in curve_r["updates"]] == [[u[k] for k in lkeys] for u in curve["updates"]]
              and curve_r["weights"] == curve["weights"]
              and curve_r["updates"][0]["imm_hex"] != c0["imm_hex"]
              and joint_r["updates"][0]["loss_hex"] != j0["loss_hex"] and joint_r["weights"] != joint["weights"],
              instant_average_first_update={"targets": float.fromhex(c0["imm_hex"]), "reversed_targets": float.fromhex(curve_r["updates"][0]["imm_hex"])},
              joint_loss_first_update={"targets": float.fromhex(j0["loss_hex"]), "reversed_targets": float.fromhex(joint_r["updates"][0]["loss_hex"])})
    rep.check("T4f_curve_kd_kept", curve_c["updates"][0]["loss_hex"] != c0["loss_hex"] and curve_c["weights"] != curve["weights"],
              curve_only_loss_first_update={"targets": float.fromhex(c0["loss_hex"]), "flipped_curve_targets": float.fromhex(curve_c["updates"][0]["loss_hex"])})
    # At the last update (the fresh init's zero-initialized output layers have moved, so joint's gradients are dense):
    zj, zc, zi = (set(x["updates"][-1]["zero_grad"]) for x in (joint, curve, inst))
    instant_only_params, curve_only_params = sorted(zc - zj), sorted(zi - zj)
    first = {k: sorted(set(x["updates"][0]["zero_grad"]) - set(j0["zero_grad"])) for k, x in (("curve_only", curve), ("instant_only", inst))}
    rep.check("T4g_gradients_instant_head_only_zeroed", zj <= zc and zj <= zi and instant_only_params and curve_only_params
              and not (set(instant_only_params) & set(curve_only_params)),
              last_update_joint_zero=sorted(zj), last_update_zeroed_by_curve_only=instant_only_params,
              last_update_zeroed_by_instant_only=curve_only_params, first_update_extra_zero=first,
              first_update_joint_zero_count=len(j0["zero_grad"]), parameters=len(joint["parameters"]))
    rep.check("T4h_weights_finite_differ_flag_read_back_jit", curve["weights_finite"] and curve["weights"] != joint["weights"]
              and curve["instant_loss_weight"] == 0.0 and curve["curve_loss_weight"] == 1.0 and joint["instant_loss_weight"] == 1.0
              and curve["jit"] and joint["jit"], optimizer=curve["optimizer"])

    # T5
    r = sh(HERE / "sm57_stage3.py", "rule-self-test")
    rep.check("T5_rule_self_test", r.returncode == 0, out=r.stdout.strip()[-600:], err=r.stderr[-400:])

    # T6
    with tempfile.TemporaryDirectory() as tmp:
        q = Path(tmp) / "quiet.json"

        def line(wins, now):
            q.write_text(json.dumps(wins), encoding="utf-8")
            return S2.block_line(q, datetime.fromisoformat(now))
        moved = [{"start": "2026-11-07T22:00", "end": "2026-11-08T00:41", "scope": "all"}]
        cases = {
            "empty_19:00_default_next_60": (line([], "2026-11-07T19:00"), "NEXT 60"),
            "empty_00:30_inside_default": (line([], "2026-11-08T00:30")[:3], "NOW"),
            "moved_19:00_next_180": (line(moved, "2026-11-07T19:00"), "NEXT 180"),
            "moved_21:00_free_next_60": (line(moved, "2026-11-07T21:00"), "NEXT 60"),
            "timing_scope_keeps_default": (line([{"start": "2026-11-07T22:00", "end": "2026-11-08T00:41", "scope": "timing"}],
                                                 "2026-11-07T19:00"), "NEXT 60"),
            "other_night_keeps_default": (line([{"start": "2026-11-06T22:00", "end": "2026-11-07T00:41", "scope": "all"}],
                                               "2026-11-07T19:00"), "NEXT 60"),
        }
        now = datetime.now()
        live = S2.block_line(S2.QUIET, now)
        to20 = int((now.replace(hour=20, minute=0, second=0, microsecond=0) - now).total_seconds() // 60)
        rep.check("T6_break_windows", all(got == want for got, want in cases.values())
                  and (now.hour >= 20 or now.hour < 1 or live == f"NEXT {to20}"),
                  cases={k: {"got": g, "want": w} for k, (g, w) in cases.items()}, live={"now": now.isoformat(timespec="seconds"), "block": live})

    from scripts.experiment_provenance import process_provenance
    out = {"ok": rep.ok, "checks": rep.checks, "seconds": round(time.perf_counter() - started, 1),
           "commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "dirty_source": subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "scripts", "expertium", "rwkv"],
                                          capture_output=True, text=True).stdout.strip(),
           "provenance": process_provenance()}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"[sm57s3-check] {'ALL OK' if rep.ok else 'FAIL'}: " + ", ".join(f"{k} {'OK' if v['ok'] else 'FAIL'}" for k, v in rep.checks.items()))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
