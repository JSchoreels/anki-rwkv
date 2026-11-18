#!/usr/bin/env python
"""The instant-only small model (SM57) stage 2: CPU proofs before the GPU (DESIGN.md, "Stage 2 plan").

  python -X utf8 artifacts/small-model/sm57/sm57s2_check.py [--out artifacts/small-model/sm57/check-cpu-sm57-stage2.json]

  S1 the 2k tier: 2,000 users inside the leak-free small model (SM72s) run's 8k train users, disjoint from the 732 set
     and its 3 long users; none of them has a label-window trim (SM72s's trims file covers all 8k users: only 5599);
     146,654,211 reviews per epoch (tiers.json).
  S2 same init: the trainer's fresh init (CatchUp.fresh_checkpoint(training_seed)) for the joint options and the
     instant-only options has the same tensor hash, equal to SM72s epoch 1's initial_parameter_sha256 (the release's
     own from-scratch start).
  S3 epoch-1 folders (sm57_stage2.py pair): the two arms differ only by {"curve_loss_weight": 0.0} and folder names, rows
     = the run's order rule over the 2k tier, joint options == SM72s run.json options, both from the fresh init;
     refused: joint against joint, an instant-only copy with two rows swapped, a copy with the flag at 0.5.
  S4 order rule: epochs 1..5 are 5 distinct permutations of the 2k tier.
  S5 trainer slice on CPU (sm57_probe.py, the trainer's batch path with the run's options and state carry, CatchUp.loss,
     clip, fresh AdamW, TorchScript) from the fresh init on 2 small users of epoch 1's order: the wt/sm92 tree (before
     the option) == this tree for the joint arm, every update and the final weights bit for bit; the instant-only arm
     sees the same updates (users, offsets) and the same first-update predictions (same init, same inputs), its loss is
     the instant average exactly, its weights differ.
  S6 rule: sm57_stage2.py rule-self-test (the stage-2 comparison reproduces stage 1's published pair on stage 1's rows).
  S7 breaks (sm57_stage2.windows on synthetic quiet files): the default 20:00-01:00 when quiet.json has nothing that
     night; a quiet.json window overlapping the night replaces the default; a scope "timing" window does not; another
     night's window does not.
"""
from argparse import ArgumentParser
from datetime import datetime
import importlib.util
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
spec = importlib.util.spec_from_file_location("sm57_stage2", HERE / "sm57_stage2.py")
S2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S2)
PY = sys.executable
BASE_TREE = Path("C:/Users/jonat/workspace/wt/sm92")
S72 = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s")
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


def trainer_probe(tree, spec_, tmp):
    import hashlib
    spec_file = Path(tmp) / f"spec-{hashlib.sha256(json.dumps(spec_).encode()).hexdigest()[:12]}.json"
    spec_file.write_text(json.dumps(spec_), encoding="utf-8")
    env = dict(ENV, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    env.pop("RWKV_NO_JIT", None)
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm57_probe.py"), str(tree), str(spec_file)], cwd=tree,
                       capture_output=True, text=True, env=env)
    line = next((x for x in r.stdout.splitlines() if x.startswith("PROBE ")), None)
    if line is None:
        raise RuntimeError(f"probe failed in {tree}: {r.stderr[-3000:]}")
    return json.loads(line[6:])


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "check-cpu-sm57-stage2.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rep = Report()
    config = S2.run_config()
    inst1, joint1 = S2.folder("inst", 1), S2.folder("joint", 1)

    # S1
    tiers = json.loads(S2.TIERS.read_text())["tiers"]["screen"]
    users = S2.screen_users()
    u732 = set(json.loads((S72 / "eval-users-732-fit.json").read_text())["users"])
    ulong = set(json.loads((S72 / "users-732-long.json").read_text())["users"])
    trims = json.loads(S2.TRIMS72S.read_text())
    total = sum(config["review_counts"][str(u)] for u in users)
    rep.check("S1_2k_tier_inside_run_disjoint_from_eval_no_trims", len(users) == len(set(users)) == 2000
              and set(users) <= set(config["train_users"]) and not (set(users) & (u732 | ulong))
              and not ({t["user_id"] for t in trims["trims"]} & set(users)) and total == tiers["train_reviews"] == 146654211
              and trims["chunk_reviews"] == config["options"]["chunk_reviews"],
              users=len(users), reviews=total, trims_users=[t["user_id"] for t in trims["trims"]])

    # S2
    hashes = {arm: S2.fresh_hash({**config["options"], **extra})[0] for arm, extra in S2.ARMS.items()}
    sm72s_e1 = json.loads((S2.RUN72S / "epoch-01" / "expertium" / S2.TRACK / "result.json").read_text())["initial_parameter_sha256"]
    rep.check("S2_same_fresh_init_both_arms_equals_sm72s_epoch1", hashes["inst"] == hashes["joint"] == sm72s_e1,
              hashes=hashes, sm72s_epoch1=sm72s_e1)

    # S3
    r = sh(HERE / "sm57_stage2.py", "pair", "--epoch", "1")
    rep.check("S3a_epoch1_arms_differ_only_by_flag", r.returncode == 0, out=r.stdout.strip()[-400:], err=r.stderr[-400:])
    with tempfile.TemporaryDirectory() as tmp:
        r = sh(HERE / "sm57_stage2.py", "pair", "--epoch", "1", "--arm-dir", joint1)
        rep.check("S3b_joint_against_joint_refused", r.returncode == 1, out=r.stdout.strip()[-300:])
        swapped = Path(tmp) / "swapped"
        shutil.copytree(inst1, swapped)
        p = json.loads((swapped / "protocol.json").read_text())
        p["rows"][0], p["rows"][1] = p["rows"][1], p["rows"][0]
        (swapped / "protocol.json").write_text(json.dumps(p), encoding="utf-8")
        r = sh(HERE / "sm57_stage2.py", "pair", "--epoch", "1", "--arm-dir", swapped)
        rep.check("S3c_swapped_rows_refused", r.returncode == 1, out=r.stdout.strip()[-300:])
        half = Path(tmp) / "half"
        shutil.copytree(inst1, half)
        d = json.loads((half / "decay.json").read_text())
        d["options"]["curve_loss_weight"] = 0.5
        (half / "decay.json").write_text(json.dumps(d), encoding="utf-8")
        r = sh(HERE / "sm57_stage2.py", "pair", "--epoch", "1", "--arm-dir", half)
        rep.check("S3d_flag_0.5_refused", r.returncode == 1, out=r.stdout.strip()[-300:])

    # S4
    orders = [tuple(S2.order(n, config)) for n in range(1, 6)]
    rep.check("S4_epoch_orders_distinct_permutations", len(set(orders)) == 5 and all(sorted(o) == sorted(users) for o in orders),
              first_users=[o[:3] for o in orders])

    # S5
    import torch
    with tempfile.TemporaryDirectory() as tmp:
        _, ck = S2.fresh_hash(config["options"])
        fresh = Path(tmp) / "fresh.pth"
        torch.save(ck["model"], fresh)
        protocol = json.loads((joint1 / "protocol.json").read_text())
        rows = sorted((r for r in protocol["rows"] if r["train_range"][1] >= 300), key=lambda r: r["train_range"][1])[:2]
        options = {k: v for k, v in config["options"].items() if k not in ("packed_ipc", "distillation")}
        options.update(chunk_reviews=128, batch_chunks=2)
        spec_ = {"arch": "w64", "checkpoint": str(fresh), "options": options, "rows": rows, "span": 256}
        base = trainer_probe(BASE_TREE, spec_, tmp)
        ours = trainer_probe(ROOT, spec_, tmp)
        inst = trainer_probe(ROOT, {**spec_, "options": {**options, **S2.FLAG}}, tmp)
        keys = ("user", "offset", "loss_hex", "imm_hex", "p_digest", "grad_digest", "zero_grad")
        view = lambda p: [{k: u[k] for k in keys} for u in p["updates"]]  # noqa: E731
        n = len(ours["updates"])
        rep.check("S5a_joint_from_fresh_init_base_tree_equals_this_tree", n > len(rows) and view(base) == view(ours)
                  and base["weights"] == ours["weights"] and ours["jit"], updates=n, optimizer=ours["optimizer"],
                  rows=[r["user_id"] for r in rows], losses=[float.fromhex(u["loss_hex"]) for u in ours["updates"]])
        rep.check("S5b_instant_only_same_batches_same_first_predictions_instant_loss",
                  [(u["user"], u["offset"]) for u in inst["updates"]] == [(u["user"], u["offset"]) for u in ours["updates"]]
                  and inst["updates"][0]["p_digest"] == ours["updates"][0]["p_digest"]
                  and inst["updates"][0]["loss_hex"] != ours["updates"][0]["loss_hex"]
                  and all(u["finite"] and u["loss_hex"] == u["imm_hex"] for u in inst["updates"])
                  and inst["weights"] != ours["weights"] and inst["weights_finite"],
                  losses=[float.fromhex(u["loss_hex"]) for u in inst["updates"]])

    # S6
    r = sh(HERE / "sm57_stage2.py", "rule-self-test")
    rep.check("S6_rule_self_test", r.returncode == 0, out=r.stdout.strip()[-600:], err=r.stderr[-400:])

    # S7
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
            "moved_00:50_free_after_00:41": (line(moved, "2026-11-08T00:50"), "NEXT 1150"),
            "timing_scope_keeps_default": (line([{"start": "2026-11-07T22:00", "end": "2026-11-08T00:41", "scope": "timing"}],
                                                 "2026-11-07T19:00"), "NEXT 60"),
            "other_night_keeps_default": (line([{"start": "2026-11-06T22:00", "end": "2026-11-07T00:41", "scope": "all"}],
                                               "2026-11-07T19:00"), "NEXT 60"),
            "afternoon_window_overlapping_nothing_keeps_default": (line([{"start": "2026-11-07T14:00", "end": "2026-11-07T15:00", "scope": "gpu"}],
                                                                        "2026-11-07T16:00"), "NEXT 240"),
            "lifted_night_window_ends_early": (line([{"start": "2026-11-07T20:00", "end": "2026-11-07T20:30", "scope": "all"}],
                                                    "2026-11-07T21:00"), "NEXT 1380"),
        }
        rep.check("S7_break_windows", all(got == want for got, want in cases.values()),
                  cases={k: {"got": g, "want": w} for k, (g, w) in cases.items()})

    from scripts.experiment_provenance import process_provenance
    out = {"ok": rep.ok, "checks": rep.checks, "seconds": round(time.perf_counter() - started, 1),
           "commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "dirty_source": subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "scripts", "expertium", "rwkv"],
                                          capture_output=True, text=True).stdout.strip(),
           "provenance": process_provenance()}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"[sm57s2-check] {'ALL OK' if rep.ok else 'FAIL'}: " + ", ".join(f"{k} {'OK' if v['ok'] else 'FAIL'}" for k, v in rep.checks.items()))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
