#!/usr/bin/env python
"""CPU checks for SM30, the review-number flag inputs (scripts/rep_flags.py; docs/SMALL_MODEL.md "Review-number flag
inputs (SM30)"). CPU only (CUDA hidden), 2 torch threads, eager modules (RWKV_NO_JIT) for bitwise comparisons.

  A. Option off = today: the training batch of the w64 runs' preparation path (columnar F30/F42 samples, batched id
     encodings, gpu_packed_features, packed IPC, chunk 2,048, materialised as on the GPU) and the w64 student's KD
     loss and every gradient are bit-identical to the same code at the base commit (exported with `git archive`),
     with the option absent and with "rep_flags": false.
  B. Flags right: against a pandas reference (raw revlog parquet, per-card cumcount over the full history) on 2 users,
     in mid-history 2,048-review windows, for the production and the reference sample builders; query rows carry the
     predicted review's flags. With the option on, every batch field except the 3 flag columns equals the option-off
     batch bit for bit, for 3 preparation paths (production, gpu_b01 packed without batched ids, rwkv prepare).
  C. Zero-init continuation: the ctx2048 epoch-6 checkpoint (92 inputs) widened with zero flag weights gives the same
     predictions, loss and shared-parameter gradients as the 92-input model, bit for bit (eval and train mode).
  D. One training step (AdamW state of epoch 6 widened, KD from the fork teacher): finite, flag weights move, every
     other parameter keeps its moments/step; CatchUp builds the rep-flags student from {"architecture": "w64",
     "rep_flags": true}; TorchScript (JIT) build of the rep-flags student runs.
  E. Teacher: handed exactly its 92 inputs (bit for bit) and returns the same targets; KD cache digest identical with
     and without the option and equal to the folder the ctx2048-cont run used.
  F. Scorer: scripts/small_model_heldout.py scores a widened checkpoint (ARCH w64-repflags) on 1 user on CPU, with the
     same metrics as the 92-input checkpoint (fp32 and w8+s8).
  G. Plumbing: prefetch allowlist, decay prepare carries the option, runbook allowlist parse.

  python scripts/check_rep_flags.py [--output artifacts/small-model/check-rep-flags.json] [--skip-scorer]
"""
from argparse import ArgumentParser
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("RWKV_NO_JIT", "1")
HERE = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("GPU_BATCH1_DATA_PATH", r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"))
os.environ["GPU_BATCH1_DATA_PATH"] = str(DATA)
TEACHER = Path(r"C:\Users\jonat\workspace\anki\qt\aqt\rwkv_inference\RWKV_trained_on_5000_10000.pth")
SMALL = Path(r"C:\Users\jonat\workspace\wt\small-model\artifacts\accuracy\experiments")  # read only
CONT = SMALL / "scale-w64kd-gate-cached-ctx2048-cont"
E6 = SMALL / "scale-w64kd-gate-cached-ctx2048/run/epoch-06/expertium/E89-F65e-F77a"
USERS = (8949, 3423)              # 40,030 and 31,008 training reviews (gate tier rows)
JOB = (8949, 32768, 4096)          # (user, offset, reviews): a real 32,768-review job start, 2 windows of 2,048


def run_options():
    return json.loads((CONT / "run" / "run.json").read_text())["options"]


def rows():
    protocol = json.loads((CONT / "run" / "epoch-07" / "protocol.json").read_text())
    return {row["user_id"]: row for row in protocol["rows"]}


def materialise(item_batch):
    import torch
    from scripts.gpu_b01 import map_batch, materialize
    return materialize(map_batch(item_batch, lambda x: x.clone()), torch.float32)


def production_batch(options, user=JOB[0], offset=JOB[1], reviews=JOB[2]):
    """The batch the w64 runs build: gpu_r01_prefetch picks gpu_preprocessing_stage.prepare_job for
    batched_id_encodings, on the split training history, then the GPU-side expansion (gpu_b01.materialize)."""
    from rwkv.data_processing import get_rwkv_data
    from scripts.gpu_batch1 import split_frame
    from scripts.gpu_preprocessing_stage import prepare_job
    row = rows()[user]
    section = split_frame(get_rwkv_data(DATA, user), row, "train")
    seed = options.get("training_seed", 2025) + user + offset
    item = prepare_job(row, offset, offset + reviews, seed, options, section.iloc[offset:offset + reviews])
    return materialise(item.batch)


def flat(value, path="batch"):
    import torch
    if isinstance(value, torch.Tensor):
        return {path: value}
    if isinstance(value, (list, tuple)):
        out = {}
        for i, x in enumerate(value):
            out.update(flat(x, f"{path}[{i}]"))
        return out
    if hasattr(value, "__dict__") and not isinstance(value, type):
        out = {}
        for key in sorted(vars(value)):
            out.update(flat(getattr(value, key), f"{path}.{key}"))
        return out
    if isinstance(value, (bool, int, float)):
        return {path: torch.tensor([value])}
    digest = __import__("hashlib").sha256(json.dumps(value, sort_keys=True, default=str).encode()).digest()
    return {path: torch.tensor(list(digest[:8]), dtype=torch.uint8)}


def kd_student(batch, options):
    """w64 student (seeded perturbation) + fp32 fork teacher targets: loss and every gradient."""
    import torch
    from expertium import variants
    from scripts import expertium_options as xo
    teacher = xo.Teacher({"teacher": str(TEACHER), "alpha": 0.9, "decay_alpha": 0.5, "teacher_dtype": "float32"},
                         torch.device("cpu"), torch.float32, options)
    targets = teacher.targets(batch)
    torch.manual_seed(2025)
    student = variants.build("w64")
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))
    student.train()
    torch.manual_seed(11)
    stats = student.get_loss(batch, kd_mix=targets)
    stats.average_loss.backward()
    out = {"loss": stats.average_loss.detach(), "teacher_curve": targets[0], "teacher_rating": targets[1]}
    out.update({f"grad.{name}": p.grad.detach().clone() for name, p in student.named_parameters() if p.grad is not None})
    return out


def dump(root, out):
    """Subprocess body for check A: option-off tensors from the source tree `root` (base commit or this tree)."""
    sys.path.insert(0, str(root))
    import torch
    torch.set_num_threads(2)
    tensors = {}
    variants_ = {"absent": {}} if root != HERE else {"absent": {}, "false": {"rep_flags": False}}
    for tag, extra in variants_.items():
        options = {**run_options(), **extra}
        batch = production_batch(options)
        tensors.update({f"{tag}/{k}": v for k, v in flat(batch).items()})
        tensors.update({f"{tag}/{k}": v for k, v in kd_student(batch, options).items()})
    torch.save(tensors, out)


def base_tree(ref, folder):
    commit = subprocess.run(["git", "-C", str(HERE), "merge-base", "HEAD", ref], capture_output=True, text=True,
                            check=True).stdout.strip()
    archive = subprocess.run(["git", "-C", str(HERE), "archive", "--format=tar", commit, "rwkv", "scripts", "expertium",
                              "features", "configs", "config.py", "utils.py"], capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(folder, filter="data") if hasattr(tarfile, "data_filter") else tar.extractall(folder)
    return commit


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/small-model/check-rep-flags.json"))
    parser.add_argument("--base-ref", default="small-model", help="option-off reference: merge-base of HEAD and this")
    parser.add_argument("--skip-scorer", action="store_true")
    parser.add_argument("--dump-root", type=Path, help=argparse_hidden())
    parser.add_argument("--dump-out", type=Path, help=argparse_hidden())
    args = parser.parse_args()
    if args.dump_root:
        dump(args.dump_root.resolve(), args.dump_out)
        return 0
    sys.path.insert(0, str(HERE))
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(2)
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare
    from expertium import variants
    from scripts import expertium_options as xo
    from scripts import rep_flags as rf
    from scripts.gpu_batch1 import make_samples, split_frame
    from scripts.gpu_r01_samples import samples_for
    started = time.perf_counter()
    results = {}

    def record(name, value, ok):
        results[name] = {"value": value, "ok": bool(ok)}
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {value}", flush=True)

    options = run_options()
    on = {**options, "rep_flags": True}

    # ---- A. option off = the base commit, every tensor
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        commit = base_tree(args.base_ref, folder / "base")
        outputs = {}
        for name, root in (("base", folder / "base"), ("this", HERE)):
            out = folder / f"{name}.pt"
            env = {**os.environ, "PYTHONPATH": str(root)}
            subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--dump-root", str(root),
                            "--dump-out", str(out)], check=True, env=env, cwd=str(root))
            outputs[name] = torch.load(out, weights_only=True)
        base, this = outputs["base"], outputs["this"]
        differing = []
        for tag in ("absent", "false"):
            keys = {k[len("absent/"):] for k in base}
            mine = {k[len(tag) + 1:] for k in this if k.startswith(tag + "/")}
            if keys != mine:
                differing.append(f"{tag}: key sets differ")
            for key in sorted(keys & mine):
                a, b = base["absent/" + key], this[f"{tag}/{key}"]
                if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                    differing.append(f"{tag}/{key}")
        grads = sum(1 for k in base if k.startswith("absent/grad."))
        record("A. option off (absent and false) = base commit: production batch + KD loss + every gradient, bitwise",
               {"base_commit": commit, "tensors": len(base), "gradients": grads, "batch_rows": list(base["absent/batch.start"].shape),
                "loss": float(base["absent/loss"]), "differing": differing[:8]},
               not differing and grads > 100)

    # ---- B. flags against a pandas reference on 2 users, mid-history windows
    table = rows()
    mismatch_first = 0
    for user in USERS:
        row = table[user]
        frame = get_rwkv_data(DATA, user)
        raw = pd.read_parquet(DATA / "revlogs" / f"user_id={user}")
        number = raw.groupby("card_id").cumcount().to_numpy() + 1     # full-history review number, row order = review_th
        expected_all = rf.flags_from_numbers(number)
        mismatch_first += int(((number == 1) != (raw["elapsed_days"].to_numpy() == -1)).sum())
        section = split_frame(frame, row, "train")
        offset = 32768 if len(section) > 36864 else 16384
        part = section.iloc[offset:offset + 4096]
        checks = {}
        for label, opts in (("production", on), ("reference", {"chunk_reviews": 2048, "rep_flags": True})):
            samples = samples_for(part, user, opts)
            bad = rows_checked = queries = local_differs = 0
            for sample in samples:
                ths = sample.review_ths.numpy().astype(np.int64)
                got = sample.card_features[:, 24:27].numpy()
                want = expected_all[ths - 1]
                bad += int((got != want).any(1).sum())
                rows_checked += len(ths)
                is_query = sample.card_features[:, 23].numpy() == 1
                queries += int(is_query.sum())
                bad += int((got[is_query].sum(1) != 1).sum())          # a query predicts a 2nd+ review
                # window-local numbering (what a window would see on its own) differs from the full history here:
                window = part[part["review_th"].between(ths.min(), ths.max())]
                local = window.groupby("card_id").cumcount().to_numpy() + 1
                local_flags = rf.flags_from_numbers(local)
                local_differs += int((local_flags != expected_all[window["review_th"].to_numpy() - 1]).any(1).sum())
            checks[label] = {"samples": len(samples), "rows": rows_checked, "query_rows": queries, "wrong": bad,
                             "rows_where_window_local_count_differs": local_differs}
        ok = all(c["wrong"] == 0 and c["rows"] > 4096 and c["query_rows"] > 0 for c in checks.values()) and \
            checks["production"]["rows_where_window_local_count_differs"] > 0
        record(f"B. user {user}: flags = pandas full-history reference in 2,048-review windows from offset {offset}",
               checks, ok)
    record("B. first review (i == 1) is exactly elapsed_days == -1 on these users", mismatch_first, mismatch_first == 0)

    def compare_paths(plain, flagged):
        diff = []
        fp, ff = flat(plain), flat(flagged)
        for key in sorted(set(fp) | set(ff)):
            a, b = fp.get(key), ff.get(key)
            if key == "batch.start":
                if b is None or not torch.equal(rf.strip(b), a) or b.shape[-1] != 95:
                    diff.append(key)
            elif a is None or b is None or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                diff.append(key)
        return diff

    plain_batch = production_batch(options)
    flag_batch = production_batch(on)
    paths = {"production (columnar, batched ids, gpu_packed, packed IPC)": compare_paths(plain_batch, flag_batch)}
    from scripts.gpu_b01 import prepare_job as b01_prepare
    row = table[JOB[0]]
    section = split_frame(get_rwkv_data(DATA, JOB[0]), row, "train").iloc[JOB[1]:JOB[1] + JOB[2]]
    b01 = {k: v for k, v in options.items() if k != "batched_id_encodings"}
    seed = 2025 + JOB[0] + JOB[1]
    paths["gpu_b01 packed, no batched ids"] = compare_paths(
        materialise(b01_prepare(row, JOB[1], JOB[1] + JOB[2], seed, b01, section).batch),
        materialise(b01_prepare(row, JOB[1], JOB[1] + JOB[2], seed, {**b01, "rep_flags": True}, section).batch))
    paths["reference make_samples + rwkv prepare"] = compare_paths(
        prepare(make_samples(section, JOB[0], 2048), seed=seed),
        prepare(rf.append_to_samples(make_samples(section, JOB[0], 2048), section), seed=seed))
    start = flag_batch.start
    query = start[:, 23] == 1
    first = start[:, 0] == float(np.float32((0 - 1.51) / 1.62))   # scaled_elapsed_days of a first review
    real = start.abs().sum(1) > 0
    alignment = {"query_rows_one_flag": bool((start[query, 24:27].sum(1) == 1).all()),
                 "first_review_rows_no_flag": bool((start[first & ~query & real, 24:27].sum(1) == 0).all()),
                 "flag_rows": [int(x) for x in start[:, 24:27].sum(0)]}
    record("B. option on: every batch field but the 3 flag columns equals option off, bitwise, on 3 paths; rows aligned",
           {**{k: v[:5] for k, v in paths.items()}, **alignment},
           not any(paths.values()) and alignment["query_rows_one_flag"] and alignment["first_review_rows_no_flag"])

    # ---- C. zero-init continuation from the 92-input epoch-6 checkpoint
    state92 = torch.load(E6 / "master.pth", map_location="cpu", weights_only=True)
    state95, widened = rf.widen_state(state92)
    m92, m95 = variants.build("w64"), variants.build("w64-repflags")
    m92.load_state_dict(state92, strict=True)
    m95.load_state_dict(state95, strict=True)
    with torch.no_grad():
        m92.eval(), m95.eval()
        a, b = m92.get_loss(plain_batch), m95.get_loss(flag_batch)
    same_eval = {k: torch.equal(getattr(a, k), getattr(b, k)) for k in ("p_curve", "p_imm_all", "average_loss")}
    m92.train(), m95.train()
    torch.manual_seed(5)
    a = m92.get_loss(plain_batch)
    a.average_loss.backward()
    torch.manual_seed(5)
    b = m95.get_loss(flag_batch)
    b.average_loss.backward()
    g92 = dict(m92.named_parameters())
    shared = [n for n, p in m95.named_parameters() if n in g92 and p.grad is not None]
    grads_equal = all(torch.equal(g92[n].grad, dict(m95.named_parameters())[n].grad) for n in shared)
    flag_grad = m95.features2card_rep_flags_weight.grad
    record("C. epoch-6 checkpoint + zero flag weights: eval predictions, train loss and shared gradients bitwise",
           {"widened": widened, "eval": same_eval, "train_loss": [a.average_loss.item(), b.average_loss.item()],
            "shared_gradients": len(shared), "shared_gradients_equal": grads_equal, "flag_grad_abs_sum": float(flag_grad.abs().sum())},
           widened and all(same_eval.values()) and torch.equal(a.average_loss, b.average_loss) and grads_equal
           and shared == [n for n, p in m92.named_parameters() if p.grad is not None] and float(flag_grad.abs().sum()) > 0)

    # ---- D. one training step with the widened AdamW state; CatchUp; JIT build
    from rwkv.train_rwkv import get_optimizer
    from argparse import Namespace
    master = variants.build("w64-repflags")
    master.load_state_dict(state95, strict=True)
    optimizer = get_optimizer(Namespace(PEAK_LR=7e-4), master)
    saved = torch.load(E6 / "optimizer.pth", map_location="cpu", weights_only=True)
    reference = get_optimizer(Namespace(PEAK_LR=7e-4), m92)
    old_names = [n for g in reference.param_groups for n in
                 [{id(p): n for n, p in m92.named_parameters()}[id(p)] for p in g["params"]]]
    optimizer.load_state_dict(rf.widen_optimizer_state(saved, optimizer, master))
    names = {id(p): n for n, p in master.named_parameters()}
    kept, with_state = True, 0
    for g in optimizer.param_groups:
        for p in g["params"]:
            if names[id(p)] == rf.PARAMETER:
                kept = kept and len(optimizer.state[p]) == 0
                continue
            old = saved["state"].get(old_names.index(names[id(p)]))
            if old is None:
                kept = kept and len(optimizer.state[p]) == 0
            else:
                with_state += 1
                kept = kept and all(torch.equal(optimizer.state[p][k], old[k]) for k in old)
    flag_param = master.features2card_rep_flags_weight
    teacher = xo.Teacher({"teacher": str(TEACHER), "alpha": 0.9, "decay_alpha": 0.5, "teacher_dtype": "float32"},
                         torch.device("cpu"), torch.float32, on)
    master.train()
    torch.manual_seed(3)
    stats = master.get_loss(flag_batch, kd_mix=teacher.targets(flag_batch))
    optimizer.zero_grad(set_to_none=True)
    stats.average_loss.backward()
    torch.nn.utils.clip_grad_norm_(master.parameters(), 1.0)
    old_step = float(next(iter(saved["state"].values()))["step"])
    optimizer.step()
    finite = torch.isfinite(stats.average_loss).item() and all(torch.isfinite(p).all() for p in master.parameters())
    steps = {"flag_weight": float(optimizer.state[flag_param]["step"]),
             "others": sorted({float(optimizer.state[p]["step"]) for g in optimizer.param_groups for p in g["params"]
                               if p is not flag_param and "step" in optimizer.state[p]})}
    record("D. one KD training step from epoch 6 (widened AdamW state): finite, flag weights move, moments kept",
           {"loss": round(stats.average_loss.detach().item(), 6), "flag_weight_abs_max": float(flag_param.abs().max()),
            "steps": steps, "epoch6_step": old_step, "moments_kept": kept, "parameters_with_state": with_state},
           finite and float(flag_param.abs().max()) > 0 and kept and steps["flag_weight"] == 1.0
           and steps["others"] == [old_step + 1])
    catch_up = xo.CatchUp(on, {}, [table[USERS[0]]])
    fresh = catch_up.fresh_checkpoint(2025)["model"]
    record("D. CatchUp({architecture: w64, rep_flags: true}) builds the rep-flags student, fresh flag weights zero",
           {"variant": catch_up.variant.name, "describe": catch_up.describe(master).get("rep_flags")},
           catch_up.variant.rep_flags and rf.PARAMETER in fresh and not fresh[rf.PARAMETER].any())
    code = chr(10).join([
        "import os, sys, torch",
        "os.environ.pop('RWKV_NO_JIT', None); sys.path.insert(0, r'%s'); torch.set_num_threads(2)",
        "from expertium import variants",
        "b = torch.load(r'%s', weights_only=False)",
        "master = variants.build('w64-repflags').float()",
        "child = variants.build('w64-repflags').selective_cast(torch.bfloat16)",
        "gen = torch.Generator().manual_seed(1)",
        "with torch.no_grad(): [p.add_(0.05 * torch.randn(p.shape, generator=gen)) for p in master.parameters()]",
        "child.copy_downcast_(master, torch.bfloat16); b.start = b.start.to(torch.bfloat16); child.train()",
        "s = child.get_loss(b); s.average_loss.backward(); g = child.features2card_rep_flags_weight",
        "print(type(child).__mro__[2].__name__, g.dtype, child.features2card[0].weight.dtype,",
        "      bool(torch.isfinite(s.average_loss)), bool(g.grad.abs().sum() > 0))"])
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "batch.pt"
        torch.save(prepare(rf.append_to_samples(make_samples(section.iloc[:512], JOB[0], 256), section.iloc[:512]),
                           seed=1), path)
        env = {k: v for k, v in os.environ.items() if k != "RWKV_NO_JIT"}
        jit = subprocess.run([sys.executable, "-c", code % (HERE, path)], capture_output=True, text=True, env=env)
    record("D. TorchScript (JIT) bf16 child of w64-repflags (as trained): runs, flag weight float32 and gets a gradient",
           (jit.stdout.strip() or jit.stderr[-400:]),
           jit.returncode == 0 and jit.stdout.strip() == "ScriptModule torch.float32 torch.bfloat16 True True")

    # ---- E. teacher inputs and the KD cache digest
    seen = {}

    class Spy:
        def __init__(self, model, tag):
            self.model, self.tag = model, tag

        def get_loss(self, batch):
            seen[self.tag] = batch.start.clone()
            return self.model.get_loss(batch)

    def spy(teacher_, tag):
        teacher_.model = Spy(teacher_.model, tag)
    spec = {"teacher": str(TEACHER), "alpha": 0.9, "decay_alpha": 0.5, "teacher_dtype": "float32"}
    t_off = xo.Teacher(spec, torch.device("cpu"), torch.float32, options)
    t_on = xo.Teacher(spec, torch.device("cpu"), torch.float32, on)
    spy(t_off, "off"), spy(t_on, "on")
    r_off, r_on = t_off.targets(plain_batch), t_on.targets(flag_batch)
    record("E. teacher gets exactly its 92 inputs (bitwise) and returns the same targets; the student batch keeps 95",
           {"teacher_input": list(seen["on"].shape), "student_input": list(flag_batch.start.shape)},
           torch.equal(seen["on"], seen["off"]) and seen["on"].shape[-1] == 92 and flag_batch.start.shape[-1] == 95
           and torch.equal(r_off[0], r_on[0]) and torch.equal(r_off[1], r_on[1]))
    recorded = json.loads((CONT / "run/epoch-08/expertium/E89-F65e-F77a/result.json").read_text())
    recorded = Path(recorded["expertium_catch_up"]["distillation"]["cache"]).name
    with tempfile.TemporaryDirectory() as folder:
        cached = {tag: xo.Teacher({**spec, "cache": folder}, torch.device("cpu"), torch.float32, o).cache.name
                  for tag, o in (("off", options), ("on", on))}
    digests = {"off": xo.cache_digest(options)[0], "on": xo.cache_digest(on)[0], "teacher_folders": cached,
               "ctx2048_cont_run": recorded}
    record("E. KD cache digest identical with and without rep_flags = the ctx2048-cont run's cache folder",
           digests, digests["off"] == digests["on"] and recorded == f"batches-{digests['on']}"
           and cached["off"] == cached["on"] == recorded)

    # ---- F. the scorer on a widened checkpoint
    if not args.skip_scorer:
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            widened_path = folder / "e6-repflags.pth"
            subprocess.run([sys.executable, str(HERE / "scripts/rep_flags.py"), "widen", "--in", str(E6 / "master.pth"),
                            "--out", str(widened_path)], check=True)
            out = folder / "heldout.jsonl"
            scorer = subprocess.run(
                [sys.executable, "-X", "utf8", str(HERE / "scripts/small_model_heldout.py"), "--device", "cpu",
                 "--count", "1", "--max-reviews", "4000", "--variants", "fp32,w8+s8", "--out", str(out),
                 "--model", f"rf=expertium:w64-repflags:{widened_path}", "--model", f"e6=expertium:w64:{E6 / 'master.pth'}"],
                capture_output=True, text=True, cwd=str(HERE))
            scored = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
        by = {r["label"]: r for r in scored if "benchmark" in r}
        same = {}
        for variant_ in ("fp32", "w8+s8"):
            a, b = by.get(f"rf/{variant_}"), by.get(f"e6/{variant_}")
            same[variant_] = bool(a and b and a["benchmark"] == b["benchmark"])
        user = scored[0]["user_id"] if scored else None
        record("F. small_model_heldout.py scores the widened checkpoint (w64-repflags) = the 92-input checkpoint",
               {"user": user, "labels": sorted(by), "identical_benchmark": same,
                "rf_fp32_logloss": by.get("rf/fp32", {}).get("benchmark", {}).get("ahead", {}).get("LogLoss"),
                "parameters": [by.get("rf/fp32", {}).get("parameters"), by.get("e6/fp32", {}).get("parameters")],
                "stderr_tail": "" if scorer.returncode == 0 else scorer.stderr[-600:]},
               scorer.returncode == 0 and len(by) == 4 and all(same.values()))

    # ---- G. plumbing
    import scripts.gpu_r01_prefetch as prefetch
    import re
    text = Path(prefetch.__file__).read_text(encoding="utf-8")
    allow = set(re.findall(r'"([a-z_0-9]+)"', text[text.index("set(options) - {"):text.index("or bool(options.get(\"preprocessing_cache\"))")]))
    record("G. prefetch allowlist has rep_flags (and still ends with chunk_reviews for check_long_chunks)",
           sorted(set(on) - allow), not (set(on) - allow) and '"chunk_reviews"}' in text)
    source = SMALL / "scale-w64kd-gate/run"
    if (source / "run.json").exists():
        import contextlib
        import shutil
        from scripts.expertium_decay import prepare as decay_prepare
        out = Path(tempfile.mkdtemp()) / "decay"
        with contextlib.redirect_stdout(io.StringIO()):
            decay_prepare(source, 3, out, 0.01, "one_minus_sin", "none", 1, {"rep_flags": True}, 0.0)
        carried = json.loads((out / "decay.json").read_text())["options"].get("rep_flags")
        shutil.rmtree(out.parent, ignore_errors=True)
        record("G. expertium_decay.prepare carries rep_flags into the decay options", carried, carried is True)

    results["seconds"] = round(time.perf_counter() - started, 1)
    failed = [name for name, value in results.items() if isinstance(value, dict) and not value["ok"]]
    results["failed"] = failed
    output = args.output if args.output.is_absolute() else HERE / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(f"{'FAILED ' + str(failed) if failed else 'all checks passed'} ({results['seconds']} s) -> {output}")
    return 1 if failed else 0


def argparse_hidden():
    import argparse
    return argparse.SUPPRESS


if __name__ == "__main__":
    sys.exit(main())
