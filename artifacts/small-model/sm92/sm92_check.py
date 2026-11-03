#!/usr/bin/env python
"""Show-time retrain of the release (SM92): CPU proofs before the GPU (DESIGN.md in this folder).

  python -X utf8 artifacts/small-model/sm92/sm92_check.py [--out artifacts/small-model/sm92/check-cpu-sm92.json]

  C1 recipe: the leak-free small model (SM72s) epochs 21-25 protocols rebuilt by scripts/sm77_warm.py's order rule equal
     the protocols on disk (so the rebuilt epoch 26 follows the release's epoch 21-25 recipe).
  C2 epoch 26: sm92_pair.py epoch, the new epoch-26 folder vs SM72s epoch 25: only the flag, the start (epoch 25's
     master / AdamW state / update offset), the epoch index, the run's own epoch-26 user order and paths differ.
  C3 show-time transform: (a) the trainer / scorer files are byte-identical to commit cc7f5358, on which the
     show-time interval retrain (SM91) proofs passed (check-cpu-sm91.json: T1 flag moves only the interval columns,
     T2 trainer rows == scorer e2s-query rows bitwise, T3 flag off == the current trainer); (b) T1 + T2 re-run here
     on the epoch-26 protocol rows of 3 users with the run's options.
  C4 cool-down: a cool-down prepared by this tree from SM72s epoch 25 (the release's own start) with the flag and the
     label-window trim == the release's cool-down except the flag and folder names (sm92_pair.py decay --same-start);
     the release against itself is refused (no flag).
  C5 rule: sm92_rule.py self-test.
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
PY = sys.executable
RUN72S = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run")
RELEASE = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon")
ARM = ROOT / "artifacts/small-model/sm92-e26"
PROVEN = "cc7f5358"
FILES = ["scripts/gpu_r01_samples.py", "scripts/gpu_r01_prefetch.py", "scripts/sm31_inputs.py", "scripts/small_model_heldout.py",
         "scripts/expertium_decay.py", "scripts/gpu_r01_training.py", "scripts/expertium_options.py", "scripts/sm77_warm.py",
         "scripts/sm91.py", "rwkv", "expertium"]
ENV = dict(os.environ, RWKV_DELETED_CARDS="neighbour", CUDA_VISIBLE_DEVICES="")


def sh(*args):
    return subprocess.run([PY, "-X", "utf8", *map(str, args)], capture_output=True, text=True, env=ENV, cwd=ROOT)


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "check-cpu-sm92.json"))
    ap.add_argument("--work", default=str(HERE / "work"))
    args = ap.parse_args()
    from scripts.sm91 import Report, t1_t2, USERS
    rep = Report()
    pair = HERE / "sm92_pair.py"

    r = sh(ROOT / "scripts/sm77_warm.py", "check-protocol", "--epochs", "21,22,23,24,25")
    rep.check("C1_epochs_21_25_rebuilt_equal_disk", r.returncode == 0, out=r.stdout.strip()[-300:], err=r.stderr[-500:])

    r = sh(pair, "epoch", "--arm", ARM, "--ref", RUN72S / "epoch-25")
    rep.check("C2a_epoch26_equals_release_recipe_except_flag_start_index_paths", r.returncode == 0, out=r.stdout.strip()[-500:])
    r = sh(pair, "epoch", "--arm", ARM, "--ref", RUN72S / "epoch-24")
    rep.check("C2b_epoch26_against_epoch24_refused", r.returncode == 1, out=r.stdout.strip()[-200:])

    g = subprocess.run(["git", "-C", str(ROOT), "diff", "--stat", PROVEN, "HEAD", "--", *FILES], capture_output=True, text=True)
    proven = json.loads((ROOT / "artifacts/small-model/sm91/check-cpu-sm91.json").read_text())
    rep.check("C3a_trainer_scorer_files_equal_proven_commit", g.returncode == 0 and not g.stdout.strip() and proven.get("ok") is True,
              proven_commit=PROVEN, proven_checks={k: v["ok"] for k, v in proven["checks"].items()}, diff=g.stdout[-500:])
    config = json.loads((RUN72S / "run.json").read_text())
    protocol = json.loads((ARM / "protocol.json").read_text())
    by_user = {row["user_id"]: row for row in protocol["rows"]}
    import torch
    torch.set_num_threads(2)
    t1_t2(rep, [by_user[u] for u in USERS], dict(config["options"]))

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    test = work / "test-cooldown-from-sm72s-e25"
    if test.exists():
        shutil.rmtree(test)
    extra = json.loads((HERE / "extra-options.json").read_text())
    p = sh(ROOT / "scripts/sm77_warm.py", "cooldown", "--epoch", "25", "--start", RUN72S / "epoch-25",
           "--out", test.relative_to(ROOT).as_posix(), "--fraction", "1.0", "--extra-options", json.dumps(extra))
    ok = p.returncode == 0
    if ok:
        p = sh(ROOT / "scripts/check_replay_label_windows.py", "--decay", test, "--trim-unlabelled-tail")
        ok = p.returncode == 0
    if not ok:
        rep.check("C4_cooldown_prepare", False, err=(p.stdout + p.stderr)[-1500:])
    else:
        r = sh(pair, "decay", "--arm", test, "--ref", RELEASE, "--same-start")
        rep.check("C4a_flag_cooldown_equals_release_except_flag_and_names", r.returncode == 0, out=r.stdout.strip()[-500:])
        r = sh(pair, "decay", "--arm", RELEASE, "--ref", RELEASE, "--same-start")
        rep.check("C4b_release_against_itself_refused_without_flag", r.returncode == 1, out=r.stdout.strip()[-300:])

    r = sh(HERE / "sm92_rule.py", "self-test")
    rep.check("C5_rule_self_test", r.returncode == 0, out=r.stdout.strip()[-1200:], err=r.stderr[-800:])

    from scripts.experiment_provenance import process_provenance
    out = {"ok": rep.ok, "checks": rep.checks, "users": list(USERS), "provenance": process_provenance()}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"[sm92-check] {'ALL OK' if rep.ok else 'FAIL'}: " + ", ".join(f"{k} {'OK' if v['ok'] else 'FAIL'}" for k, v in rep.checks.items()))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
