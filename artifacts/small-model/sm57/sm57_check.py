#!/usr/bin/env python
"""The instant-only small model (SM57): CPU proofs before the GPU (DESIGN.md in this folder).

  python -X utf8 artifacts/small-model/sm57/sm57_check.py [--out artifacts/small-model/sm57/check-cpu-sm57.json]

  C1 recipe: the leak-free small model (SM72s) epochs 21-25 protocols rebuilt by scripts/sm77_warm.py's order rule equal
     the protocols on disk.
  C2 epoch 26 folders (sm57_pair.py): the joint arm == SM72s's epoch-26 recipe (no extra option), the instant-only arm ==
     the same + exactly {"curve_loss_weight": 0.0}, the two arms equal except the flag; wrong pairs refused.
  C3 loss, one real user's batch, a perturbed w64 student, eager and TorchScript, without and with KD targets
     (+ sync_free_loss): (a) option untouched == the loss code before SM57 (variant_model.py at BASE): loss, per-review
     loss tensor, every gradient bit for bit; (b) weight 1.0 == untouched bitwise; (c) weight 0.0: the training loss is
     exactly the instant average, the reported predictions / instant / curve averages are unchanged, the curve-only
     parameters (and only they) get newly zero gradients, and loss and gradients are linear in the weight
     (w 2 - w 1 == w 1 - w 0), i.e. the weight scales exactly the curve part.
  C4 trainer slice on CPU (sm57_probe.py: the trainer's batch path with the run's options and state carry, CatchUp.loss,
     clip 0.5, the trainer's optimizer; TorchScript like the trainer), 2 small users of SM72s epoch 26, from SM72s
     epoch 25: the BASE tree (wt/sm92) == this tree with the option absent == this tree with weight 1.0, per update
     (loss bits, every gradient) and the final weights bit for bit, with AdamW (epoch) and fresh Muon (cool-down);
     weight 0.0 trains (weights differ, finite, loss == instant average) and only curve-head parameters get newly zero
     gradients.
  C5 KD cache digest (TEACHER_BLIND) identical with and without the option; the scorer, the samples and the decay
     runner are byte-identical to BASE (only the loss, the trainer's option wiring, the batch-option allowlist and
     TEACHER_BLIND change).
  C6 cool-down: prepared by this tree from SM72s epoch 25 with the release's extras == the release (sm57_pair.py decay
     --same-start), and with the extras + the flag == the release except the flag; the release against itself with the
     flag refused.
  C7 rule: sm57_rule.py self-test.
"""
from argparse import ArgumentParser
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
PY = sys.executable
BASE = "22ca8edf"   # the show-time retrain of the release (SM92) branch tip this branch starts from
BASE_TREE = Path("C:/Users/jonat/workspace/wt/sm92")
RUN72S = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run")
RELEASE = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon")
TRACK = "E89-F65e-F77a"
SM = ROOT / "artifacts/small-model"
JOINT, INST = SM / "sm57-joint-e26", SM / "sm57-inst-e26"
FLAG = {"curve_loss_weight": 0.0}
DATA = r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"
TEACHER = Path(r"C:\Users\jonat\workspace\anki\qt\aqt\rwkv_inference\RWKV_trained_on_5000_10000.pth")
CHANGED = {"expertium/model/variant_model.py", "scripts/gpu_r01_training.py", "scripts/gpu_r01_prefetch.py",
           "scripts/expertium_options.py"}
UNCHANGED = ["scripts/small_model_heldout.py", "scripts/gpu_r01_samples.py", "scripts/expertium_decay.py",
             "scripts/sm77_warm.py", "scripts/sm31_inputs.py", "scripts/state_carry_train.py", "rwkv",
             "expertium/model/srs_model.py", "expertium/variants.py", "expertium/architecture.py"]
CURVE_HEAD = ("head_w.", "head_ahead_logits.", "ahead_linear.", "w_linear.", "gru_")   # the curve-only parameters of w64
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


def before_class(tmp):
    source = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE}:expertium/model/variant_model.py"],
                            capture_output=True, text=True, check=True).stdout
    assert "curve_loss_weight" not in source
    path = Path(tmp) / "variant_model_before_sm57.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("variant_model_before_sm57", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VariantSrsRWKV


LOSS_PROBE = r'''
import json, os, sys, importlib.util, hashlib
root, tmp, jit = sys.argv[1], sys.argv[2], sys.argv[3] == "jit"
sys.path.insert(0, root); os.chdir(root)
if jit: os.environ.pop("RWKV_NO_JIT", None)
else: os.environ["RWKV_NO_JIT"] = "1"
import torch
torch.set_num_threads(2)
torch._C._set_graph_executor_optimize(False)   # as scripts/expertium_decay.py run (jit_optimize false)
from rwkv.data_processing import get_rwkv_data
from rwkv.prepare_batch import prepare
from scripts.gpu_batch1 import make_samples
from expertium import variants
from expertium.model.variant_model import VariantSrsRWKV
from scripts import expertium_options as xo
spec = importlib.util.spec_from_file_location("before", os.path.join(tmp, "variant_model_before_sm57.py"))
before = importlib.util.module_from_spec(spec); spec.loader.exec_module(before)
frame = get_rwkv_data(__import__("pathlib").Path(os.environ["GPU_BATCH1_DATA_PATH"]), 7)
batch = prepare(make_samples(frame.iloc[:640], 7, 128), seed=2032)
kd = xo.Teacher({"teacher": sys.argv[4], "alpha": 0.9, "teacher_dtype": "float32"}, torch.device("cpu"), torch.float32).targets(batch)

def run(cls, weight, kd_mix, sync_free):
    torch.manual_seed(2025)
    m = cls(variants.variant("w64"))
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    m.train(); m.sync_free_loss = sync_free
    if weight is not None:
        m.curve_loss_weight = weight
    torch.manual_seed(11)
    s = m.get_loss(batch, kd_mix=kd_mix)
    s.average_loss.backward()
    names = [n for n, _ in m.named_parameters()]
    grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in m.parameters()]
    return s, grads, names, isinstance(m, torch.jit.ScriptModule) or any(isinstance(x, torch.jit.ScriptModule) for x in m.modules())

def same(a, b):
    return (torch.equal(a[0].average_loss, b[0].average_loss) and torch.equal(a[0].loss_tensor, b[0].loss_tensor)
            and len(a[1]) == len(b[1]) and all(torch.equal(x, y) for x, y in zip(a[1], b[1])))

res = {}
for label, kd_mix, sync_free in (("plain", None, False), ("kd + sync_free_loss", kd, True)):
    old = run(before.VariantSrsRWKV, None, kd_mix, sync_free)
    absent = run(VariantSrsRWKV, None, kd_mix, sync_free)
    one = run(VariantSrsRWKV, 1.0, kd_mix, sync_free)
    zero = run(VariantSrsRWKV, 0.0, kd_mix, sync_free)
    two = run(VariantSrsRWKV, 2.0, kd_mix, sync_free)
    zero_names = [n for n, g in zip(zero[2], zero[1]) if not bool(g.any())]
    absent_zero = [n for n, g in zip(absent[2], absent[1]) if not bool(g.any())]
    # linearity in the weight: grad(2) - grad(1) == grad(1) - grad(0) (the weight scales exactly the curve part)
    worst = 0.0
    for g0, g1, g2 in zip(zero[1], one[1], two[1]):
        sc = float(g1.abs().max())
        if sc > 0: worst = max(worst, float((g2 - 2 * g1 + g0).abs().max()) / sc)
    lin = abs((two[0].average_loss - one[0].average_loss) - (one[0].average_loss - zero[0].average_loss)).item()
    res[label] = {
        "scripted": absent[3],
        "a_untouched_equals_before": same(old, absent), "loss": absent[0].average_loss.item(),
        "b_one_equals_untouched": same(absent, one),
        "c_zero_loss_equals_instant_average": bool(torch.equal(zero[0].average_loss, zero[0].imm_avg)),
        "c_zero_loss": zero[0].average_loss.item(), "c_instant": zero[0].imm_avg.item(),
        "c_reports_unchanged": bool(torch.equal(absent[0].p_curve, zero[0].p_curve) and torch.equal(absent[0].p_imm_all, zero[0].p_imm_all)
                                    and torch.equal(absent[0].imm_avg, zero[0].imm_avg) and torch.equal(absent[0].ahead_avg, zero[0].ahead_avg)),
        "c_new_zero_grad_params": sorted(set(zero_names) - set(absent_zero)), "c_zero_grad_params_default": len(absent_zero),
        "c_grad_linearity_max_rel": worst, "c_loss_linearity_abs": lin,
        "parameters": len(zero[2]), "parameters_zero_grad": len(zero_names)}
print("LOSSPROBE " + json.dumps(res))
'''


def loss_probe(tmp, jit):
    probe = Path(tmp) / "sm57_loss_probe.py"
    probe.write_text(LOSS_PROBE, encoding="utf-8")
    env = dict(ENV, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([PY, "-X", "utf8", str(probe), str(ROOT), tmp, "jit" if jit else "eager", str(TEACHER)],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    line = next((x for x in r.stdout.splitlines() if x.startswith("LOSSPROBE ")), None)
    if line is None:
        raise RuntimeError(f"loss probe failed: {r.stderr[-3000:]}")
    return json.loads(line[10:])


def trainer_probe(tree, spec, tmp, jit=True):
    spec_file = Path(tmp) / f"spec-{hashlib.sha256(json.dumps(spec).encode()).hexdigest()[:12]}.json"
    spec_file.write_text(json.dumps(spec), encoding="utf-8")
    env = dict(ENV, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    if jit:
        env.pop("RWKV_NO_JIT", None)
    else:
        env["RWKV_NO_JIT"] = "1"
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm57_probe.py"), str(tree), str(spec_file)], cwd=tree,
                       capture_output=True, text=True, env=env)
    line = next((x for x in r.stdout.splitlines() if x.startswith("PROBE ")), None)
    if line is None:
        raise RuntimeError(f"probe failed in {tree}: {r.stderr[-3000:]}")
    return json.loads(line[6:])


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "check-cpu-sm57.json"))
    ap.add_argument("--work", default=str(HERE / "work"))
    ap.add_argument("--only", default="", help="comma list of C1..C7 (debugging); default all")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    want = lambda c: only is None or c in only  # noqa: E731
    started = time.perf_counter()
    rep = Report()
    pair = HERE / "sm57_pair.py"
    flag = json.dumps(FLAG)

    if want("C1"):
        r = sh(ROOT / "scripts/sm77_warm.py", "check-protocol", "--epochs", "21,22,23,24,25")
        rep.check("C1_epochs_21_25_rebuilt_equal_disk", r.returncode == 0, out=r.stdout.strip()[-300:], err=r.stderr[-500:])

    if want("C2"):
        for name, argv, code in (
                ("C2a_joint_epoch26_equals_recipe", ("epoch", "--arm", JOINT, "--ref", RUN72S / "epoch-25"), 0),
                ("C2b_instant_epoch26_equals_recipe_plus_flag", ("epoch", "--arm", INST, "--ref", RUN72S / "epoch-25", "--flag", flag), 0),
                ("C2c_instant_equals_joint_except_flag", ("arms", "--arm", INST, "--ref", JOINT, "--flag", flag), 0),
                ("C2d_instant_without_flag_refused", ("epoch", "--arm", INST, "--ref", RUN72S / "epoch-25"), 1),
                ("C2e_joint_against_epoch24_refused", ("epoch", "--arm", JOINT, "--ref", RUN72S / "epoch-24"), 1),
                ("C2f_joint_as_instant_refused", ("arms", "--arm", JOINT, "--ref", JOINT, "--flag", flag), 1)):
            r = sh(pair, *argv)
            rep.check(name, r.returncode == code, out=r.stdout.strip()[-400:], err=r.stderr[-300:])

    with tempfile.TemporaryDirectory() as tmp:
        before_class(tmp)
        if want("C3"):
            for mode, jit in (("eager", False), ("scripted", True)):
                res = loss_probe(tmp, jit)
                for label, x in res.items():
                    rep.check(f"C3_{mode}_{label}_a_untouched_equals_code_before_sm57", x["a_untouched_equals_before"], loss=x["loss"], scripted=x["scripted"])
                    rep.check(f"C3_{mode}_{label}_b_weight1_equals_untouched", x["b_one_equals_untouched"])
                    new = x["c_new_zero_grad_params"]
                    rep.check(f"C3_{mode}_{label}_c_weight0_instant_only", x["c_zero_loss_equals_instant_average"] and x["c_reports_unchanged"]
                                  and len(new) > 0 and all(n.startswith(CURVE_HEAD) for n in new)
                                  and x["c_grad_linearity_max_rel"] < 1e-4 and x["c_loss_linearity_abs"] < 1e-6,
                                  **{k: v for k, v in x.items() if k.startswith("c_") or k.startswith("param")})

        if want("C4"):
            protocol = json.loads((JOINT / "protocol.json").read_text())
            small = sorted((r for r in protocol["rows"] if 300 <= r["train_range"][1] <= 1500), key=lambda r: r["train_range"][1])
            rows = [small[0], next(r for r in small[len(small) // 2:] if r["user_id"] != small[0]["user_id"])]
            config = json.loads((RUN72S / "run.json").read_text())
            options = {k: v for k, v in config["options"].items() if k not in ("packed_ipc", "distillation")}
            options.update(chunk_reviews=128, batch_chunks=2)
            ck = RUN72S / "epoch-25" / "expertium" / TRACK / "master.pth"
            for opt_label, extra in (("adamw_epoch", {}), ("muon_cooldown", {"optimizer": "expertium-iter53"})):
                spec = {"arch": "w64", "checkpoint": str(ck), "options": {**options, **extra}, "rows": rows, "span": 256}
                base = trainer_probe(BASE_TREE, spec, tmp)
                ours = trainer_probe(ROOT, spec, tmp)
                one = trainer_probe(ROOT, {**spec, "options": {**spec["options"], "curve_loss_weight": 1.0}}, tmp)
                zero = trainer_probe(ROOT, {**spec, "options": {**spec["options"], "curve_loss_weight": 0.0}}, tmp)
                keys = ("user", "offset", "loss_hex", "imm_hex", "p_digest", "grad_digest", "zero_grad")
                view = lambda p: [{k: u[k] for k in keys} for u in p["updates"]]  # noqa: E731
                n = len(base["updates"])
                rep.check(f"C4_{opt_label}_base_tree_equals_this_tree_option_absent", n > len(rows) and view(base) == view(ours)
                          and base["weights"] == ours["weights"] and base["jit"], updates=n, optimizer=ours["optimizer"],
                          scripted=ours["jit"], weights=ours["weights"][:16], losses=[float.fromhex(u["loss_hex"]) for u in ours["updates"]])
                rep.check(f"C4_{opt_label}_weight1_equals_option_absent", view(one) == view(ours) and one["weights"] == ours["weights"])
                zsets = {tuple(u["zero_grad"]) for u in zero["updates"]}
                newz = set().union(*(set(u["zero_grad"]) - set(v["zero_grad"]) for u, v in zip(zero["updates"], ours["updates"])))
                rep.check(f"C4_{opt_label}_weight0_trains_instant_only", zero["weights"] != ours["weights"] and zero["weights_finite"]
                          and all(u["finite"] and u["loss_hex"] == u["imm_hex"] for u in zero["updates"])
                          and len(newz) > 0 and all(n.startswith(CURVE_HEAD) for n in newz),
                          new_zero_grad_params=sorted(newz),
                          losses=[float.fromhex(u["loss_hex"]) for u in zero["updates"]])

        if want("C5"):
            import torch
            from scripts import expertium_options as xo
            spec = {"teacher": str(TEACHER), "alpha": 0.9, "decay_alpha": 0.5, "teacher_dtype": "float32", "cache": tmp}
            track = {"architecture": "w64", "batch_chunks": 256, "fast_samples": True, "training_seed": 2025, "sync_free_loss": True}
            plain = xo.Teacher(spec, torch.device("cpu"), torch.float32, track)
            weighted = xo.Teacher(spec, torch.device("cpu"), torch.float32, {**track, "curve_loss_weight": 0.0})
            rep.check("C5a_kd_cache_digest_unchanged_by_option", plain.cache == weighted.cache, caches=[plain.cache.name, weighted.cache.name])
            g = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", BASE, "--", "scripts", "expertium", "rwkv"],
                               capture_output=True, text=True)
            changed = set(g.stdout.split())
            g2 = subprocess.run(["git", "-C", str(ROOT), "diff", "--stat", BASE, "--", *UNCHANGED], capture_output=True, text=True)
            rep.check("C5b_only_the_loss_option_files_changed_since_base", g.returncode == 0 and changed <= CHANGED and not g2.stdout.strip(),
                      changed=sorted(changed), base=BASE)

    if want("C6"):
        work = Path(args.work)
        work.mkdir(parents=True, exist_ok=True)
        extra = json.loads((HERE / "extra-options-joint.json").read_text())
        extra_i = json.loads((HERE / "extra-options-instant.json").read_text())
        rep.check("C6a_instant_extras_are_joint_extras_plus_flag", extra_i == {**extra, **FLAG} and set(extra_i) - set(extra) == set(FLAG),
                  joint=extra, instant=extra_i)
        for tag, ex, fl in (("joint", extra, "{}"), ("instant", extra_i, flag)):
            test = work / f"test-cooldown-{tag}-from-sm72s-e25"
            if test.exists():
                shutil.rmtree(test)
            p = sh(ROOT / "scripts/sm77_warm.py", "cooldown", "--epoch", "25", "--start", RUN72S / "epoch-25",
                   "--out", test.relative_to(ROOT).as_posix(), "--fraction", "1.0", "--extra-options", json.dumps(ex))
            ok = p.returncode == 0
            if ok:
                p = sh(ROOT / "scripts/check_replay_label_windows.py", "--decay", test, "--trim-unlabelled-tail")
                ok = p.returncode == 0
            if not ok:
                rep.check(f"C6_{tag}_cooldown_prepare", False, err=(p.stdout + p.stderr)[-1500:])
                continue
            r = sh(pair, "decay", "--arm", test, "--ref", RELEASE, "--same-start", "--flag", fl)
            rep.check(f"C6b_{tag}_cooldown_equals_release_except_flag_and_names", r.returncode == 0, out=r.stdout.strip()[-500:])
        r = sh(pair, "arms", "--arm", work / "test-cooldown-instant-from-sm72s-e25", "--ref", work / "test-cooldown-joint-from-sm72s-e25", "--flag", flag)
        rep.check("C6c_instant_cooldown_equals_joint_cooldown_except_flag", r.returncode == 0, out=r.stdout.strip()[-500:])
        r = sh(pair, "decay", "--arm", RELEASE, "--ref", RELEASE, "--same-start", "--flag", flag)
        rep.check("C6d_release_as_instant_refused", r.returncode == 1, out=r.stdout.strip()[-300:])

    if want("C7"):
        r = sh(HERE / "sm57_rule.py", "self-test")
        rep.check("C7_rule_self_test", r.returncode == 0, out=r.stdout.strip()[-1200:], err=r.stderr[-800:])

    from scripts.experiment_provenance import process_provenance
    out = {"ok": rep.ok, "checks": rep.checks, "seconds": round(time.perf_counter() - started, 1), "base": BASE,
           "commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "dirty_source": subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "scripts", "expertium", "rwkv"],
                                          capture_output=True, text=True).stdout.strip(),
           "provenance": process_provenance()}
    if only is None:
        Path(args.out).write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"[sm57-check] {'ALL OK' if rep.ok else 'FAIL'}: " + ", ".join(f"{k} {'OK' if v['ok'] else 'FAIL'}" for k, v in rep.checks.items()))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
