#!/usr/bin/env python
"""Frozen-model answer-time head (SM82b): the response-time prediction (SM82) head trained on a FROZEN copy of the
released small model (the leak-free small model (SM72s) 25 epochs + the 1-epoch 32k Muon cool-down), so no grade
prediction can change.

How it is frozen: the trunk never trains at all. Each training user's history goes once through the frozen model on
the time scorer's own path (scripts/sm82_time.py score: full history, fp32, leak-free inputs, --deleted-cards
neighbour, id / day codes seed 2025 + user) under torch.no_grad, with every parameter requires_grad False; a forward
hook keeps the instant head's 128-wide hidden layer (head_p output, the head's input) at the instant rows that carry a
time label. Only the head (time_heads.0.linear, 128 -> 64, zero init, 8,256 values) is in the optimizer (plain AdamW),
so there is no trunk backward and no other weight can move. The written master.pth = the released master's tensors,
loaded again from its file, + the trained head.

Data: the users, histories and labels of the SM82 quarter-epoch cool-down (1,991 training users, 145.6M reviews, none
in the 732 evaluation pool), each user's first --cap reviews (a full-history forward of a longer user does not fit the
0.8 memory cap; 59 users are cut, 136.1M reviews kept). Time labels exactly as SM82 (scripts/expertium_options.py
time_labels: a recorded time on a later day than the card's previous review; the true grade's 16-bin distribution;
censored at a lower user cap). One pass, in a seeded user order, through a 2M-row shuffle buffer, minibatches of 8,192
rows, AdamW lr 2e-3 (2% warm-up, cosine to 0), no weight decay.

  plan        --protocol P --cap N --out DIR     CPU: user order + labelled rows per user (the LR schedule's length).
  train       --start EXT --plan DIR/plan.json --out DIR    the head training (GPU, or CPU for the proof); resumable.
  check       --ext EXT --out report.json        the CPU proof (see `check`).
  gradecompare --rows R --dump D ...             the grade check: released vs frozen-with-head scorer rows / dumps.
  decide      ...                                the rule, writes artifacts/small-model/sm82b/sm82b-decision.{md,json}.
"""
from argparse import ArgumentParser
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HERE = ROOT / "artifacts/small-model/sm82b"
ARCH = "w64-sm82"
BINS = 16
HEAD = "time_heads.0.linear."
S72 = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s")   # read only
U732 = S72 / "eval-users-732-fit.json"
SOLD = S72 / "gpu/small-lf-neighbour.jsonl"                                    # the released model's 732 scores
SLBL = "s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon"
SM82 = Path("C:/Users/jonat/workspace/wt/sm82/artifacts/small-model/sm82")    # read only
GRADE_FIELDS = ("reviews", "curve_labels", "instant_labels", "curve_log_loss", "curve_raw_log_loss", "instant_log_loss",
                "instant_rating_cross_entropy", "benchmark", "benchmark_existing", "benchmark_deleted")

from scripts.expertium_options import time_labels   # noqa: E402  the SM82 label definition (one copy)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def data_path():
    from scripts import compress_common as cc
    return cc.data_path()


# ------------------------------------------------------------------ plan
def _plan_user(args):
    data, user, cap = args
    import pandas as pd
    frame = pd.read_parquet(Path(data) / "revlogs" / f"user_id={user}", columns=["duration", "elapsed_days"])
    lower, _, ucap = time_labels(frame["duration"].to_numpy(), frame["elapsed_days"].to_numpy())
    return user, len(frame), int((lower[:cap] >= 0).sum()), int(ucap)


def plan(args):
    from multiprocessing import Pool
    protocol = json.loads(Path(args.protocol).read_text())
    rows = protocol["rows"]
    if any(r["train_range"] != [0, r["available_reviews"]] for r in rows):
        raise SystemExit("every protocol row must be a whole history")
    users = sorted(r["user_id"] for r in rows)
    pool732 = set(json.loads(U732.read_text())["users"])
    if pool732 & set(users):
        raise SystemExit(f"{len(pool732 & set(users))} training users are in the 732 evaluation pool")
    order = [int(u) for u in np.random.default_rng(args.seed).permutation(users)]
    data = str(data_path())
    with Pool(args.workers) as p:
        info = {u: (n, k, c) for u, n, k, c in p.map(_plan_user, [(data, u, args.cap) for u in order], chunksize=8)}
    expected = {r["user_id"]: r["available_reviews"] for r in rows}
    bad = [u for u in order if info[u][0] != expected[u]]
    if bad:
        raise SystemExit(f"revlog length differs from the protocol for {len(bad)} users ({bad[:5]})")
    out = Path(args.out)
    record = {"protocol": str(args.protocol), "protocol_sha256": sha256(args.protocol), "seed": args.seed, "cap": args.cap,
              "users": len(order), "reviews_all": int(sum(info[u][0] for u in order)),
              "reviews_used": int(sum(min(info[u][0], args.cap) for u in order)),
              "users_cut": int(sum(info[u][0] > args.cap for u in order)),
              "labelled_rows": int(sum(info[u][1] for u in order)),
              "order": [{"user_id": u, "reviews": info[u][0], "labelled": info[u][1], "user_cap_ms": info[u][2]} for u in order]}
    atomic_json(out / "plan.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "order"}))
    return 0


# ------------------------------------------------------------------ training
class PrefixBatches:
    """The time scorer's inputs (smh.LeakBatches: full history, deleted-card signals neutralised by the neighbour
    rule, id / day codes seed 2025 + user) cut to the first `cap` reviews, plus the revlog's grade and time labels."""

    def __init__(self, users, data, cap):
        self.users, self.data, self.cap = users, data, cap

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        started = time.perf_counter()
        try:
            import pandas as pd
            from rwkv.data_processing import get_rwkv_data
            from rwkv.prepare_batch import prepare
            from scripts import compress_common as cc
            from scripts import sm71_leak
            frame = get_rwkv_data(self.data, user)
            frame, _ = sm71_leak.neutralise(frame, "neighbour", self.data, user)
            full = len(frame)
            frame = frame.iloc[:self.cap]
            batch = prepare(cc.make_samples(frame, user, len(frame)), seed=2025 + user)
            rev = pd.read_parquet(Path(self.data) / "revlogs" / f"user_id={user}", columns=["rating", "duration", "elapsed_days"])
            lower, cens, _ = time_labels(rev["duration"].to_numpy(), rev["elapsed_days"].to_numpy())
            labels = {"rating": np.array(rev["rating"].to_numpy(np.int64)), "lower": np.array(lower), "cens": np.array(cens), "full": full}
        except Exception as error:  # recorded, not fatal
            return user, None, None, repr(error), time.perf_counter() - started
        return user, batch, labels, None, time.perf_counter() - started


def first(items):
    return items[0]


def time_loss_rows(logits, grade, lower, cens):
    """Per-row censored cross-entropy of the true grade's 16-bin distribution; the same formula as the variant's
    training loss (expertium/model/variant_model.py _get_loss_variant, SM82)."""
    import torch
    per = logits.float().view(-1, 4, BINS)[torch.arange(len(grade), device=logits.device), grade]
    log_p = torch.log_softmax(per, dim=-1)
    log_tail = torch.logcumsumexp(log_p.flip([-1]), dim=-1).flip([-1])
    index = lower.unsqueeze(-1)
    return -torch.where(cens.unsqueeze(-1) > 0, log_tail.gather(-1, index), log_p.gather(-1, index)).squeeze(-1)


def lr_at(step, total, peak, warmup_share):
    warm = max(1, round(warmup_share * total))
    if step < warm:
        return peak * (step + 1) / warm
    if step >= total:
        return 0.0
    return peak * 0.5 * (1.0 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def train(args):
    import torch
    from scripts import small_model_heldout as smh   # sets RWKV_NO_JIT; the scorer's long WKV
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("HELDOUT_GPU_MEMORY_FRACTION", "0.8")))
    else:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    start = Path(args.start)
    ext = json.loads((start / "extend.json").read_text())
    released_path = Path(ext["source_master"])
    released = torch.load(released_path, map_location="cpu", weights_only=True)
    state = {k: v.float() for k, v in torch.load(start / "master.pth", map_location="cpu", weights_only=True).items()}
    if any(not torch.equal(state[k], released[k].float()) for k in released) or \
            sorted(set(state) - set(released)) != [HEAD + "bias", HEAD + "weight"] or \
            any(bool(state[k].any()) for k in (HEAD + "bias", HEAD + "weight")):
        raise SystemExit(f"{start} is not the released master + a zero head")
    model = smh.build_expertium(ARCH, state).to(device)
    smh.set_expertium_state_bits(model, 0)   # fp32 state, as the time scorer
    for p in model.parameters():
        p.requires_grad_(False)
    head = model.time_heads[0].linear
    for p in head.parameters():
        p.requires_grad_(True)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    in_optimizer = sorted(n for n, p in model.named_parameters() if any(p is q for g in optimizer.param_groups for q in g["params"]))
    captured = {}
    model.head_p.register_forward_hook(lambda module, inputs, output: captured.__setitem__("x", output))

    plan_rec = json.loads(Path(args.plan).read_text())
    args.cap = int(plan_rec["cap"])   # the plan's prefix cap (its labelled-row counts use it)
    order = plan_rec["order"]
    if args.max_users:
        small = [r for r in order if r["reviews"] <= args.max_user_reviews][:args.max_users]
        # smoke: also the longest history (cut at the cap), so a memory problem shows before the long run
        order = small + ([max(order, key=lambda r: (r["reviews"], r["user_id"]))] if args.smoke_longest else [])
    total_rows = sum(r["labelled"] for r in order)
    total_steps = -(-total_rows // args.batch)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    resume = out / "resume.pt"
    gen = torch.Generator().manual_seed(args.seed)
    step, done, passes = 0, 0, 0
    buf = []          # [(x [n,128], grade, lower, cens)] on device
    stats = {"users": 0, "skipped": [], "rows": 0, "rows_planned_mismatch": [], "duration_input_nonzero": 0,
             "grade_out_of_range": 0, "duplicate_review_th": 0, "reviews_forward": 0, "loss_sum": 0.0, "loss_rows": 0,
             "first_loss": None, "pass_losses": []}
    if resume.exists():
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        if ck["plan_sha256"] != sha256(args.plan) or ck["config"] != vars_config(args):
            raise SystemExit(f"{resume} belongs to another plan or configuration")
        head.load_state_dict(ck["head"])
        optimizer.load_state_dict(ck["optimizer"])
        step, done, passes, stats = ck["step"], ck["done"], ck["passes"], ck["stats"]
        gen.set_state(ck["generator"])
        if ck["leftover"] is not None:
            buf = [tuple(t.to(device) for t in ck["leftover"])]
        print(f"[sm82b-train] resumed at user {done}/{len(order)}, step {step}/{total_steps}", flush=True)
    log_sum, log_rows, last_log = 0.0, 0, step

    def run_pass(final):
        nonlocal step, buf, passes, log_sum, log_rows, last_log
        if not buf:
            return
        x = torch.cat([b[0] for b in buf]); g = torch.cat([b[1] for b in buf])
        lo = torch.cat([b[2] for b in buf]); ce = torch.cat([b[3] for b in buf])
        perm = torch.randperm(len(x), generator=gen).to(device)
        full = len(x) // args.batch
        stop = len(x) if final else full * args.batch
        pass_sum, pass_rows = 0.0, 0
        for s in range(0, stop, args.batch):
            idx = perm[s:min(s + args.batch, stop)]
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, total_steps, args.lr, args.warmup)
            loss = time_loss_rows(head(x[idx]), g[idx], lo[idx], ce[idx]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            value = float(loss.detach())
            if stats["first_loss"] is None:
                stats["first_loss"] = value
                print(f"[sm82b-train] step 1: time loss {value:.4f} nats over {len(idx):,} rows (zero head: ln 16 = 2.7726)", flush=True)
            step += 1
            log_sum += value * len(idx); log_rows += len(idx); pass_sum += value * len(idx); pass_rows += len(idx)
            if step - last_log >= args.log_every:
                el = time.perf_counter() - started
                print(f"[sm82b-train] step {step}/{total_steps}: time loss {log_sum / max(log_rows, 1):.4f} nats (last "
                      f"{step - last_log} steps), lr {optimizer.param_groups[0]['lr']:.2e}, users {done}/{len(order)}, "
                      f"{stats['reviews_forward']:,} reviews, {el / 60:.1f} min this run", flush=True)
                log_sum, log_rows, last_log = 0.0, 0, step
        stats["loss_sum"] += pass_sum; stats["loss_rows"] += pass_rows
        stats["pass_losses"].append(round(pass_sum / max(pass_rows, 1), 5))
        rest = perm[stop:]
        buf = [(x[rest], g[rest], lo[rest], ce[rest])] if len(rest) else []
        passes += 1

    def save_resume():
        left = tuple(t.cpu() for t in buf[0]) if buf else None
        tmp = resume.with_suffix(".tmp")
        torch.save({"plan_sha256": sha256(args.plan), "config": vars_config(args), "head": head.state_dict(),
                    "optimizer": optimizer.state_dict(), "step": step, "done": done, "passes": passes, "stats": stats,
                    "generator": gen.get_state(), "leftover": left}, tmp)
        os.replace(tmp, resume)

    started = time.perf_counter()
    users = [r["user_id"] for r in order]
    planned = {r["user_id"]: r["labelled"] for r in order}
    dataset = PrefixBatches(users[done:], data_path(), args.cap)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.workers, collate_fn=first,
                                         prefetch_factor=1 if args.workers else None)
    print(f"[sm82b-train] {len(order)} users ({len(order) - done} to go), {total_rows:,} labelled rows planned, "
          f"{total_steps:,} steps of {args.batch:,}; trainable {trainable}; device {device}", flush=True)
    for user, batch, labels, error, prep in loader:
        if args.stop_after_passes and passes >= args.stop_after_passes:
            break
        if batch is None:
            stats["skipped"].append([user, error[:200]])
            done += 1
            continue
        try:
            gb = batch.to(device)
            with torch.no_grad():
                model.forward_batch_variant(gb.start, gb.sub_gather, gb.sub_gather_lens, gb.time_shift_selects,
                                            gb.skips, gb.num_data)
            x_all = captured.pop("x")[0].float()
            lab = gb.labels[0]
            q = (lab[:, 6] == 1) & (lab[:, 4] == 1)          # the instant rows (as the time scorer)
            qi = torch.nonzero(q).squeeze(-1)
            th = gb.label_review_th[0].long()[qi]
            lower = torch.as_tensor(labels["lower"], device=device)
            cens = torch.as_tensor(labels["cens"], device=device)
            rating = torch.as_tensor(labels["rating"], device=device)
            keep = (th >= 1) & (th <= len(labels["lower"]))
            keep &= lower[(th - 1).clamp(min=0)] >= 0
            qi, th = qi[keep], th[keep]
            stats["duplicate_review_th"] += int(len(th) - len(torch.unique(th)))
            stats["duration_input_nonzero"] += int((gb.start.reshape(-1, gb.start.shape[-1])[qi, 8] != 0).sum())   # the row's own time is not an input
            grade = rating[th - 1] - 1
            stats["grade_out_of_range"] += int(((grade < 0) | (grade > 3)).sum())
            rows = (x_all[qi].clone(), grade.clamp(0, 3), lower[th - 1], cens[th - 1])
            del gb, x_all
        except torch.OutOfMemoryError as oom:
            captured.clear()
            torch.cuda.empty_cache()
            stats["skipped"].append([user, f"OOM: {oom}"[:200]])
            done += 1
            continue
        if len(th) != planned[user]:
            stats["rows_planned_mismatch"].append([user, int(len(th)), planned[user]])
        stats["users"] += 1
        stats["rows"] += int(len(th))
        stats["reviews_forward"] += min(labels["full"], args.cap)
        buf.append(rows)
        done += 1
        if sum(len(b[0]) for b in buf) >= args.buffer_rows:
            run_pass(final=False)
            save_resume()
    finished = done >= len(order)
    if finished:
        run_pass(final=True)
    save_resume()
    trunk_same = all(torch.equal(v.detach().cpu(), released[k].float()) for k, v in model.state_dict().items() if k in released)
    result = {"status": "complete" if finished else "stopped", "users_done": done, "users": len(order), "step": step,
              "steps_planned": total_steps, "rows_planned": total_rows, "passes": passes, "batch": args.batch,
              "lr": args.lr, "warmup": args.warmup, "buffer_rows": args.buffer_rows, "cap": args.cap, "seed": args.seed,
              "optimizer": "AdamW (betas 0.9/0.999, eps 1e-8, no weight decay), head only", "trainable": trainable,
              "in_optimizer": in_optimizer, "trunk_unchanged_in_memory": trunk_same,
              "mean_time_loss": round(stats["loss_sum"] / max(stats["loss_rows"], 1), 5),
              "seconds_this_run": round(time.perf_counter() - started, 1), "device": str(device), "start": str(start),
              "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None,
              "released_master": str(released_path), "released_master_sha256": ext["source_master_sha256"],
              **{k: v for k, v in stats.items() if k not in ("loss_sum", "loss_rows")}}
    if finished:
        out_sd = {k: v.clone() for k, v in released.items()}
        out_sd.update({HEAD + "weight": head.weight.detach().cpu().float().clone(),
                       HEAD + "bias": head.bias.detach().cpu().float().clone()})
        torch.save(out_sd, out / "master.pth")
        result["master"] = str(out / "master.pth")
        result["master_sha256"] = sha256(out / "master.pth")
    atomic_json(out / "train.json", result)
    print(f"[sm82b-train] {result['status']}: {done}/{len(order)} users, {step:,} steps, {stats['rows']:,} rows, mean "
          f"time loss {result['mean_time_loss']:.4f}, skipped {len(stats['skipped'])}, trunk unchanged {trunk_same}, "
          f"{result['seconds_this_run'] / 60:.1f} min -> {out}", flush=True)
    return 0 if trunk_same and stats["duration_input_nonzero"] == 0 else 1


def vars_config(args):
    return {k: getattr(args, k) for k in ("batch", "lr", "warmup", "buffer_rows", "cap", "seed", "max_users", "max_user_reviews",
                                          "smoke_longest")}


# ------------------------------------------------------------------ CPU proof
GRADE_PROBE = r'''
import json, os, sys
root = sys.argv[1]
sys.path.insert(0, root)
os.chdir(root)
import torch
torch.set_num_threads(4)
from scripts import small_model_heldout as smh
from scripts import compress_common as cc
spec = json.loads(open(sys.argv[2]).read())
batches = smh.LeakBatches([spec["user"]], cc.data_path(), cc.run_info()["review_counts"], "neighbour", 0)
user, batch, error, _ = batches[0]
outs = {}
for tag, arch, path in (("released", "w64", spec["released"]), ("new", spec["arch"], spec["new"])):
    state = {k: v.float() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
    model = smh.build_expertium(arch, state)
    smh.set_expertium_state_bits(model, spec.get("bits", 0))
    with torch.inference_mode():
        outs[tag] = model.forward_batch_variant(batch.start, batch.sub_gather, batch.sub_gather_lens,
                                                batch.time_shift_selects, batch.skips, batch.num_data)
names = ["ahead_logits", "w", "w_log_p", "p_logits", "s_raw", "d_raw"]
eq = {n: bool(torch.equal(outs["released"][i], outs["new"][i])) for i, n in enumerate(names)}
t = outs["new"][6]
print("PROBE " + json.dumps({"user": user, "start_shape": list(batch.start.shape), "outputs_bitwise_equal": eq,
                             "time_logits_nonzero": bool(t.abs().max() > 0), "time_logits_shape": list(t.shape)}))
'''


def check(args):
    """CPU proof, before any GPU ticket:
    F1 frozen: a CPU training run on 3 small training users (several AdamW updates): the trainable parameters and the
       optimizer hold the head's 2 tensors only; the written master's every non-head tensor is bitwise the released
       master's (same keys, dtypes, values); the model's trunk in memory is unchanged; the head moved off zero; the
       first loss is ln 16 (zero head).
    F2 resume: the same run stopped after one buffer pass and resumed gives a bitwise identical head.
    F3 grade outputs: on full histories of 2 evaluation users (the scorer path, CPU fp32 and int8 state), every grade
       output of the trained frozen model (w64-sm82) is bitwise the released model's (w64); the time logits are non-zero.
    F4 no leak: the response-time prediction (SM82) T3 probe (scripts/sm82_time.py) on the trained head: changing one
       review's answer time changes no output (grade or time) up to and including its own instant row; training rows
       carry a zero duration input (the row's own time is never an input).
    F5 labels: every training row's time label is time_labels() of the review it predicts, its grade the revlog's;
       rows per user = the plan's.
    F6 loss = metric: the training loss per row equals the scorer's censored LogLoss (sm82_time.censored_ll).
    F7 scorer and export: scripts/small_model_heldout.py (CPU, w8+s8, leak-free) scores the released model and the
       trained frozen model on 2 users: every grade field and the per-review dump arrays are equal; the time scorer
       (sm82_time.py score, CPU) scores the trained head."""
    import tempfile
    import torch
    from scripts import sm82_time as t82
    started = time.perf_counter()
    report = {"checks": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    fails = []
    ext = Path(args.ext).resolve()
    released = Path(json.loads((ext / "extend.json").read_text())["source_master"])
    py = [sys.executable, "-X", "utf8", str(Path(__file__).resolve())]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env.pop("RWKV_DELETED_CARDS", None)
    (HERE / "work").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(HERE / "work")) as tmp:
        tmp = Path(tmp)
        plan_rec = json.loads(Path(args.plan).read_text())
        small = [r for r in plan_rec["order"] if 400 <= r["reviews"] <= 3000][:3]
        mini = dict(plan_rec, order=small)
        mini_plan = tmp / "plan.json"
        mini_plan.write_text(json.dumps(mini))
        common = ["--start", str(ext), "--plan", str(mini_plan), "--device", "cpu", "--batch", "128", "--buffer-rows", "600",
                  "--workers", "0", "--log-every", "5"]
        def run(cmd):
            r = subprocess.run(py + cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
            return r.returncode, r.stdout[-3000:] + r.stderr[-3000:]
        rc_a, log_a = run(["train", "--out", str(tmp / "a")] + common)
        rc_b1, log_b1 = run(["train", "--out", str(tmp / "b"), "--stop-after-passes", "1"] + common)
        rc_b2, log_b2 = run(["train", "--out", str(tmp / "b")] + common)
        ta = json.loads((tmp / "a/train.json").read_text()) if (tmp / "a/train.json").exists() else {}
        tb = json.loads((tmp / "b/train.json").read_text()) if (tmp / "b/train.json").exists() else {}
        if rc_a != 0 or not ta.get("master"):
            report["checks"]["F1_frozen"] = {"pass": False, "log": log_a}
            fails.append("F1")
        else:
            rel = torch.load(released, map_location="cpu", weights_only=True)
            new = torch.load(ta["master"], map_location="cpu", weights_only=True)
            same_keys = sorted(set(new) - set(rel)) == [HEAD + "bias", HEAD + "weight"] and not (set(rel) - set(new))
            bitwise = all(torch.equal(new[k], rel[k]) and new[k].dtype == rel[k].dtype for k in rel)
            moved = float(new[HEAD + "weight"].abs().max()), float(new[HEAD + "bias"].abs().max())
            f1 = (same_keys and bitwise and min(moved) > 0 and ta["trunk_unchanged_in_memory"]
                  and ta["trainable"] == [HEAD + "weight", HEAD + "bias"] and ta["in_optimizer"] == sorted([HEAD + "bias", HEAD + "weight"])
                  and ta["step"] >= 5 and abs(ta["first_loss"] - math.log(BINS)) < 1e-5)
            report["checks"]["F1_frozen"] = {"pass": f1, "users": [r["user_id"] for r in small], "steps": ta["step"],
                                             "rows": ta["rows"], "keys_released_plus_head": same_keys,
                                             "non_head_tensors_bitwise": bitwise, "head_abs_max": moved,
                                             "trainable": ta["trainable"], "in_optimizer": ta["in_optimizer"],
                                             "trunk_unchanged_in_memory": ta["trunk_unchanged_in_memory"],
                                             "first_loss": ta["first_loss"], "pass_losses": ta["pass_losses"]}
            fails += [] if f1 else ["F1"]
            f5 = (ta["rows_planned_mismatch"] == [] and ta["grade_out_of_range"] == 0 and ta["duplicate_review_th"] == 0
                  and not ta["skipped"] and ta["rows"] == sum(r["labelled"] for r in small))
            report["checks"]["F5_labels"] = {"pass": f5, "rows": ta["rows"], "planned": sum(r["labelled"] for r in small),
                                             "mismatch": ta["rows_planned_mismatch"], "grade_out_of_range": ta["grade_out_of_range"],
                                             "duplicate_review_th": ta["duplicate_review_th"], "skipped": ta["skipped"],
                                             "how": "labels are looked up by the row's review_th in time_labels() of the revlog "
                                                    "(the trainer's TimeTargets rule); rows = instant rows with a label"}
            fails += [] if f5 else ["F5"]
            if rc_b1 == 0 and rc_b2 == 0 and tb.get("master"):
                hb = torch.load(tb["master"], map_location="cpu", weights_only=True)
                f2 = all(torch.equal(new[HEAD + n], hb[HEAD + n]) for n in ("weight", "bias")) and tb["step"] == ta["step"]
            else:
                f2 = False
            report["checks"]["F2_resume_bitwise"] = {"pass": f2, "steps": [ta.get("step"), tb.get("step")],
                                                     "log": None if f2 else (log_b1 + log_b2)[-2000:]}
            fails += [] if f2 else ["F2"]
            # F3 grade outputs bitwise on 2 evaluation users (fp32 state and the int8 state of w8+s8's s8)
            pool = json.loads(U732.read_text())["users"]
            from scripts import compress_common as cc
            counts = cc.run_info()["review_counts"]
            eval_small = sorted((u for u in pool if 1500 <= counts[str(u)] <= 4000), key=lambda u: counts[str(u)])[:2]
            probes = []
            for user in eval_small:
                for bits in (0, 8):
                    probes.append(dict(t82.run_probe(ROOT, GRADE_PROBE, {"user": user, "released": str(released), "new": ta["master"],
                                                                         "arch": ARCH, "bits": bits, "deleted_cards_env": ""},
                                                     tmp, f"f3-{user}-{bits}"), bits=bits))
            f3 = all(all(p["outputs_bitwise_equal"].values()) and p["time_logits_nonzero"] for p in probes)
            report["checks"]["F3_grade_outputs_bitwise"] = {"pass": f3, "probes": probes}
            fails += [] if f3 else ["F3"]
            # F4 no leak: SM82's T3 probe on the trained head (a training user, full history)
            leak = t82.run_probe(ROOT, t82.LEAK_PROBE, {"arch": ARCH, "checkpoint": ta["master"], "user": small[1]["user_id"],
                                                        "picks": 6, "deleted_cards_env": ""}, tmp, "f4-leak")
            f4 = (all(r["outputs_up_to_its_query_bitwise_equal"] and r["query_row_duration_input_zero"] for r in leak)
                  and all(r["later_outputs_changed"] is not False for r in leak)
                  and all(len(r["input_rows_changed"]) == 1 and r["input_columns_changed"] == [8] and r["changed_rows_after_query"]
                          for r in leak)
                  and ta["duration_input_nonzero"] == 0)
            report["checks"]["F4_no_time_leak"] = {"pass": f4, "user": small[1]["user_id"], "picks": leak,
                                                   "training_rows_with_nonzero_duration_input": ta["duration_input_nonzero"]}
            fails += [] if f4 else ["F4"]
            # F7 scorer and export (CPU): grade rows and dumps, then the time scorer
            ufile = tmp / "users.json"
            ufile.write_text(json.dumps({"users": eval_small}))
            rows, dump = tmp / "grade.jsonl", tmp / "dump"
            cmd = [sys.executable, "-X", "utf8", str(ROOT / "scripts/small_model_heldout.py"), "--pool", "732", "--users-file", str(ufile),
                   "--model", f"rel=expertium:w64:{released}", "--model", f"new=expertium:{ARCH}:{ta['master']}",
                   "--only", "rel/w8+s8", "--only", "new/w8+s8", "--device", "cpu", "--workers", "0", "--split-existing",
                   "--deleted-cards", "neighbour", "--dump", str(dump), "--out", str(rows)]
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
            cmp = grade_compare(rows, dump, "rel", "new", eval_small, existing=None) if r.returncode == 0 and rows.exists() else {"pass": False}
            trows = tmp / "time.jsonl"
            r2 = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts/sm82_time.py"), "score", "--master", ta["master"],
                                 "--baselines", str(SM82 / "time-baselines.jsonl"), "--users-file", str(ufile), "--out", str(trows),
                                 "--device", "cpu", "--workers", "0"], capture_output=True, text=True, env=env, cwd=str(ROOT))
            tr = [json.loads(x) for x in trows.read_text().splitlines()] if trows.exists() else []
            time_ok = r2.returncode == 0 and len(tr) == 2 and all("error" not in x and x["baseline_same_reviews"] and
                                                                   math.isfinite(x["model"]["ll"]) for x in tr)
            f7 = cmp["pass"] and time_ok
            report["checks"]["F7_scorer_export"] = {"pass": f7, "grade": cmp, "scorer_exit": r.returncode,
                                                    "scorer_log": None if r.returncode == 0 else (r.stdout + r.stderr)[-2000:],
                                                    "time_scorer_exit": r2.returncode,
                                                    "time_rows": [{"user_id": x.get("user_id"), "reviews": x.get("reviews"),
                                                                   "ll_per_review": x["model"]["ll"] / max(x["model"]["n"], 1)}
                                                                  for x in tr if "model" in x],
                                                    "time_log": None if time_ok else (r2.stdout + r2.stderr)[-2000:]}
            fails += [] if f7 else ["F7"]
    # F6 training loss == the scorer's metric
    g = torch.Generator().manual_seed(0)
    n = 4096
    logits = torch.randn(n, 4 * BINS, generator=g, dtype=torch.float64) * 1.5
    grade = torch.randint(0, 4, (n,), generator=g)
    lower = torch.randint(0, BINS, (n,), generator=g)
    cens = (torch.rand(n, generator=g) < 0.2).long()
    ours = time_loss_rows(logits, grade, lower, cens).double().numpy()
    prob = torch.softmax(logits.view(n, 4, BINS)[torch.arange(n), grade], -1).numpy()
    theirs = t82.censored_ll(prob, lower.numpy(), cens.numpy())
    gap = float(np.abs(ours - theirs).max())
    f6 = gap < 1e-5 and float(prob.min()) > 1e-7
    report["checks"]["F6_loss_equals_metric"] = {"pass": f6, "rows": n, "max_abs_gap": gap, "min_prob": float(prob.min())}
    fails += [] if f6 else ["F6"]
    report.update(passed=not fails, fails=fails, seconds=round(time.perf_counter() - started, 1))
    atomic_json(args.out, report)
    print(f"[sm82b-check] {'PASS' if not fails else 'FAIL ' + ','.join(fails)} ({report['seconds']} s; {args.out})")
    return 0 if not fails else 1


# ------------------------------------------------------------------ grade check
def grade_compare(rows_path, dump, rel_tag, new_tag, users, existing=None, existing_label=None):
    """Released vs frozen-with-head scorer rows (every grade field equal) and per-review dumps (arrays equal); with
    `existing`, the released model's recorded rows too."""
    rows = [json.loads(x) for x in Path(rows_path).read_text().splitlines()]
    by = {(r["user_id"], r["label"]): r for r in rows if "label" in r}
    old = {}
    if existing is not None:
        old = {r["user_id"]: r for r in map(json.loads, Path(existing).read_text().splitlines()) if r.get("label") == existing_label}
    per, ok_all, ok_existing = [], True, True
    for u in users:
        a, b = by.get((u, f"{rel_tag}/w8+s8")), by.get((u, f"{new_tag}/w8+s8"))
        if a is None or b is None:
            per.append({"user_id": u, "missing": True}); ok_all = False
            continue
        fields = {f: a.get(f) == b.get(f) for f in GRADE_FIELDS}
        da = np.load(Path(dump) / f"{u}_{rel_tag}_w8+s8.npz"); db = np.load(Path(dump) / f"{u}_{new_tag}_w8+s8.npz")
        arrays = {k: bool(np.array_equal(da[k], db[k])) for k in da.files} if set(da.files) == set(db.files) else {"files": False}
        rec = {"user_id": u, "reviews": a.get("reviews"), "dump_rows": int(len(da["review_th"])), "fields_equal": all(fields.values()),
               "dump_arrays_equal": all(arrays.values()), "unequal_fields": [f for f, v in fields.items() if not v],
               "parameters": [a.get("parameters"), b.get("parameters")], "weight_bytes": [a.get("weight_bytes"), b.get("weight_bytes")]}
        ok_all &= rec["fields_equal"] and rec["dump_arrays_equal"]
        if existing is not None:
            e = old.get(u)
            ef = {f: (e or {}).get(f) == b.get(f) for f in GRADE_FIELDS}
            rec["existing_fields_equal"] = bool(e) and all(ef.values())
            rec["existing_unequal_fields"] = [f for f, v in ef.items() if not v]
            ok_existing &= rec["existing_fields_equal"]
        per.append(rec)
    out = {"pass": bool(ok_all), "users": users, "fields": list(GRADE_FIELDS), "per_user": per}
    if existing is not None:
        out["existing_equal"] = bool(ok_existing)
        out["existing"] = {"rows": str(existing), "label": existing_label}
    return out


def gradecompare(args):
    users = json.loads(Path(args.users_file).read_text())["users"]
    out = grade_compare(args.rows, args.dump, args.rel_tag, args.new_tag, users,
                        existing=args.existing or None, existing_label=args.existing_label or None)
    atomic_json(args.out, out)
    print(f"[sm82b-grade] {len(users)} users: released vs frozen with the head bitwise {out['pass']}; vs the recorded "
          f"released scores {out.get('existing_equal')}")
    return 0 if out["pass"] else 1


# ------------------------------------------------------------------ decision
def _time_arrays(trows, base, tusers):
    n = np.array([trows[u]["model"]["n"] for u in tusers], float)
    nu = np.array([trows[u]["model"]["n_unc"] for u in tusers], float)
    ll = {"model": np.array([trows[u]["model"]["ll"] for u in tusers]),
          "model_marginal": np.array([trows[u]["model_marginal"]["ll"] for u in tusers])}
    ae = {"model": np.array([trows[u]["model"]["ae"] for u in tusers]),
          "model_marginal": np.array([trows[u]["model_marginal"]["ae"] for u in tusers])}
    for b in ("card_last", "user_median"):
        ll[b] = np.array([base[u][b]["ll"] for u in tusers]); ae[b] = np.array([base[u][b]["ae"] for u in tusers])
    return n, nu, ll, ae


def _paired(a_ll, b_ll, a_ae, b_ae, n, nu, idx):
    row = {}
    for metric, x, y, w in (("time_logloss", a_ll, b_ll, n), ("mae_ln_s", a_ae, b_ae, nu)):
        delta = float((x - y).sum() / w.sum())
        boot = ((x - y)[idx]).sum(1) / w[idx].sum(1)
        row[metric] = {"delta": round(delta, 5), "ci": [round(float(np.percentile(boot, q)), 5) for q in (2.5, 97.5)],
                       "relative_pct": round(100 * delta / float(y.sum() / w.sum()), 2),
                       "users_better": int(((x - y) / np.maximum(w, 1) < 0).sum())}
    return row


def decide(args):
    from scripts import sm77_rule
    users732 = set(json.loads(U732.read_text())["users"])
    big = sm77_rule.load(*sm77_rule.BIG, users732)
    pool = sorted(users732 & set(big))
    load_rows = lambda p: {r["user_id"]: r for r in map(json.loads, Path(p).read_text().splitlines()) if "error" not in r} if Path(p).exists() else {}
    trows, srows = load_rows(args.time_rows), load_rows(args.sm82_time_rows)
    base = {r["user_id"]: r for r in map(json.loads, Path(args.baselines).read_text().splitlines())}
    tmissing = [u for u in pool if u not in trows or u not in srows]
    if tmissing:
        print(f"[sm82b-rule] time rows missing for {len(tmissing)} users ({tmissing[:5]})")
        return 2
    tusers = [u for u in pool if trows[u]["reviews"] > 0]
    mis = [u for u in tusers if not trows[u]["baseline_same_reviews"] or trows[u]["review_th_sha256"] != srows[u]["review_th_sha256"]]
    if mis:
        print(f"[sm82b-rule] REFUSE: {len(mis)} users' time-scored reviews differ from the baselines' or SM82's ({mis[:5]})")
        return 1
    idx = np.random.default_rng(0).integers(0, len(tusers), size=(2000, len(tusers)))
    n, nu, ll, ae = _time_arrays(trows, base, tusers)
    _, _, ll82, ae82 = _time_arrays(srows, base, tusers)
    tr = {"users": len(tusers), "reviews": int(n.sum()), "uncensored": int(nu.sum()),
          "time_logloss": {k: round(float(v.sum() / n.sum()), 5) for k, v in ll.items()},
          "mae_ln_s": {k: round(float(v.sum() / nu.sum()), 5) for k, v in ae.items()},
          "sm82_joint": {"time_logloss": round(float(ll82["model"].sum() / n.sum()), 5),
                         "time_logloss_marginal": round(float(ll82["model_marginal"].sum() / n.sum()), 5),
                         "mae_ln_s": round(float(ae82["model"].sum() / nu.sum()), 5),
                         "mae_ln_s_marginal": round(float(ae82["model_marginal"].sum() / nu.sum()), 5)}, "vs": {}}
    beats = True
    for b in ("card_last", "user_median"):
        row = _paired(ll["model"], ll[b], ae["model"], ae[b], n, nu, idx)
        beats &= all(row[m]["delta"] < 0 and row[m]["ci"][1] < 0 for m in row)
        tr["vs"][b] = row
    tr["vs"]["sm82_joint"] = _paired(ll["model"], ll82["model"], ae["model"], ae82["model"], n, nu, idx)
    by_grade = {}
    for g in range(4):
        key = f"g{g + 1}"
        c = sum(trows[u]["model"][key][0] for u in tusers)
        rec = {"reviews": c}
        for k in ("model", "card_last", "user_median"):
            rec[k] = round(sum((trows[u][k] if k == "model" else base[u][k])[key][1] for u in tusers) / max(c, 1), 4)
        rec["sm82_joint"] = round(sum(srows[u]["model"][key][1] for u in tusers) / max(c, 1), 4)
        under = [sum(trows[u]["under_5_5s"][g][i] for u in tusers) for i in range(3)]
        under82 = [sum(srows[u]["under_5_5s"][g][i] for u in tusers) for i in range(3)]
        rec["p_under_5_5s_predicted_vs_actual"] = [round(under[1] / max(under[0], 1), 4), round(under[2] / max(under[0], 1), 4)]
        rec["sm82_joint_p_under_5_5s_predicted"] = round(under82[1] / max(under82[0], 1), 4)
        by_grade[str(g + 1)] = rec
    tr["by_grade_time_logloss"] = by_grade
    gc = json.loads(Path(args.grade_check).read_text())
    grade_same = bool(gc["pass"])
    rel = sm77_rule.load(SOLD, f"{SLBL}/w8+s8", set(big))
    rg = sm77_rule.compare(rel, rel, big, [u for u in pool if u in rel])
    tj = json.loads(Path(args.train_json).read_text())
    res = {"grade": {"bitwise_equal_released": grade_same, "check_users": gc["users"], "fields": gc["fields"],
                     "existing_equal": gc.get("existing_equal"), "per_user": gc["per_user"],
                     "released_722": {h: {"logloss": rg[h]["arm"], "vs_big_pct": rg[h]["arm_vs_big_pct"]} for h in ("curve", "instant")},
                     "grade_check_file": str(args.grade_check)},
           "time": tr, "training": {k: tj.get(k) for k in ("users", "users_done", "rows", "step", "passes", "batch", "lr", "warmup",
                                                           "buffer_rows", "cap", "reviews_forward", "skipped", "optimizer",
                                                           "trainable", "mean_time_loss", "first_loss", "seconds_this_run",
                                                           "master", "master_sha256", "released_master", "released_master_sha256")},
           "time_beats_baselines": bool(beats), "grade_unchanged": grade_same}
    res["verdict"] = "pass" if grade_same and beats else ("fail-grade" if not grade_same else "fail-time")
    res["rule"] = {"grade": "grade unchanged: released model vs the frozen model with the head, scorer rows (every grade field) "
                            "and per-review dump arrays bitwise equal on the check users; the non-head weights are the released "
                            "master's bitwise (CPU proof F1)",
                   "time": "head beats both baselines on the censored time LogLoss and on the mean |ln error| (uncensored), 95% CI "
                           "of the review-weighted difference clear of zero (paired user bootstrap 2,000, seed 0)",
                   "time_reviews": "benchmark test reviews of the 722 users with a recorded time, on a later day than the card's "
                                   "previous review; conditional on the true grade (as response-time prediction (SM82))"}
    res.update(time_rows=str(args.time_rows), sm82_time_rows=str(args.sm82_time_rows), baselines=str(args.baselines))
    p = gc["per_user"][0] if gc["per_user"] else {}
    res["size"] = {"parameters": p.get("parameters"), "weight_bytes": p.get("weight_bytes")}
    HERE.mkdir(parents=True, exist_ok=True)
    atomic_json(HERE / "sm82b-decision.json", res)
    md = ["# Frozen-model answer-time head (SM82b): the answer-time head on the frozen released small model", "",
          "The response-time prediction (SM82) head (one zero-init Linear from the instant head's 128-wide layer to 4 grades x 16 "
          "log-time bins, censored cross-entropy of the true grade's bins) trained alone on a frozen copy of the released small "
          "model (the leak-free small model (SM72s) 25 epochs + the 1-epoch 32k Muon cool-down): every other weight is the "
          f"released master's, bit for bit. Training: the {tj.get('users_done'):,} users of the SM82 quarter-epoch cool-down, "
          f"each history's first {tj.get('cap'):,} reviews through the frozen model on the time scorer's path (fp32, leak-free "
          f"inputs), {tj.get('rows') or 0:,} labelled rows, one pass, {tj.get('step') or 0:,} AdamW steps of {tj.get('batch'):,} "
          f"rows (lr {tj.get('lr')}, cosine), head only; {(tj.get('seconds_this_run') or 0) / 60:.1f} min (last run segment).", "",
          f"Grade (unchanged): the scorer (w8+s8, leak-free) on {len(gc['users'])} users of the 732 set, the released model vs "
          "the frozen model with the head, in the same ticket: every grade field of the rows and the per-review dump arrays "
          f"(p_ahead, p_instant, y, bin, review_th) compared exactly. Recorded released scores ({SOLD.name}) compared too.", "",
          "| user | reviews | dump rows | rows and dumps bitwise equal | equal to the recorded released scores |", "|---|---|---|---|---|"]
    for r in gc["per_user"]:
        md.append(f"| {r['user_id']} | {r.get('reviews') or 0:,} | {r.get('dump_rows') or 0:,} | "
                  f"{r.get('fields_equal') and r.get('dump_arrays_equal')} | {r.get('existing_fields_equal')} |")
    md += ["", "| head | released model, 722 users (LogLoss) | vs the big model (1 leak-free epoch) | change |", "|---|---|---|---|"]
    for h in ("curve", "instant"):
        x = res["grade"]["released_722"][h]
        md.append(f"| {h} | {x['logloss']:.5f} | {x['vs_big_pct']:+.2f}% | {'none (bitwise)' if grade_same else 'NOT EQUAL'} |")
    md += ["", f"Time ({tr['users']} users, {tr['reviews']:,} test reviews with a recorded time on a later day than the card's "
           f"previous review, {tr['uncensored']:,} under the cap): censored 16-bin time LogLoss given the true grade, and the mean "
           "|ln error| of the predicted median (uncensored reviews). Baselines get a residual distribution per grade fit on the "
           "users' non-test reviews (same rows as SM82's).", "",
           "| predictor | time LogLoss | mean abs ln error |", "|---|---|---|"]
    j = tr["sm82_joint"]
    for label, a, b in (("frozen-model head (given the true grade)", tr["time_logloss"]["model"], tr["mae_ln_s"]["model"]),
                        ("frozen-model head (grade mixed by the grade head)", tr["time_logloss"]["model_marginal"], tr["mae_ln_s"]["model_marginal"]),
                        ("jointly trained head, response-time prediction (SM82) (given the true grade)", j["time_logloss"], j["mae_ln_s"]),
                        ("jointly trained head, response-time prediction (SM82) (grade mixed)", j["time_logloss_marginal"], j["mae_ln_s_marginal"]),
                        ("baseline: the card's last time", tr["time_logloss"]["card_last"], tr["mae_ln_s"]["card_last"]),
                        ("baseline: the user's running median per grade", tr["time_logloss"]["user_median"], tr["mae_ln_s"]["user_median"])):
        md.append(f"| {label} | {a:.4f} | {b:.4f} |")
    md += ["", "| frozen-model head vs | LogLoss delta [95% CI] | relative | abs ln error delta [95% CI] | relative |", "|---|---|---|---|---|"]
    for b, label in (("card_last", "the card's last time"), ("user_median", "the user's running median per grade"),
                     ("sm82_joint", "the jointly trained head (SM82): the cost of freezing")):
        x, y = tr["vs"][b]["time_logloss"], tr["vs"][b]["mae_ln_s"]
        md.append(f"| {label} | {x['delta']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}] | {x['relative_pct']:+.1f}% | "
                  f"{y['delta']:+.4f} [{y['ci'][0]:+.4f}, {y['ci'][1]:+.4f}] | {y['relative_pct']:+.1f}% |")
    md += ["", "| grade | reviews | frozen-model head LogLoss | jointly trained head (SM82) | card's last time | user's median per grade | "
           "P(under 5.5 s) predicted / actual | SM82 predicted |", "|---|---|---|---|---|---|---|---|"]
    for g, x in by_grade.items():
        md.append(f"| {g} | {x['reviews']:,} | {x['model']:.4f} | {x['sm82_joint']:.4f} | {x['card_last']:.4f} | {x['user_median']:.4f} | "
                  f"{x['p_under_5_5s_predicted_vs_actual'][0]:.3f} / {x['p_under_5_5s_predicted_vs_actual'][1]:.3f} | "
                  f"{x['sm82_joint_p_under_5_5s_predicted']:.3f} |")
    s = res["size"]
    if s["parameters"] and s["parameters"][0]:
        md += ["", f"Size: {s['parameters'][0]:,} -> {s['parameters'][1]:,} parameters; scorer w8 weight bytes "
               f"{s['weight_bytes'][0]:,} -> {s['weight_bytes'][1]:,} ({s['weight_bytes'][1] - s['weight_bytes'][0]:+,} B); state unchanged."]
    md += ["", f"Verdict: **{res['verdict']}** (grade unchanged bitwise: {grade_same}; head beats both baselines: {bool(beats)}; "
           f"vs the jointly trained head (SM82): LogLoss {tr['vs']['sm82_joint']['time_logloss']['delta']:+.4f}, "
           f"|ln err| {tr['vs']['sm82_joint']['mae_ln_s']['delta']:+.4f})."]
    (HERE / "sm82b-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    v = tr["vs"]["sm82_joint"]
    print(f"[sm82b-rule] {tr['users']} users: grade unchanged {grade_same}; time LogLoss {tr['time_logloss']['model']:.4f} vs card's "
          f"last {tr['time_logloss']['card_last']:.4f} / user median {tr['time_logloss']['user_median']:.4f} / joint (SM82) "
          f"{j['time_logloss']:.4f} ({v['time_logloss']['delta']:+.4f}), |ln err| {tr['mae_ln_s']['model']:.4f} vs "
          f"{tr['mae_ln_s']['card_last']:.4f} / {tr['mae_ln_s']['user_median']:.4f} / {j['mae_ln_s']:.4f}: {res['verdict'].upper()}")
    return 0


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--protocol", required=True)
    p.add_argument("--cap", type=int, default=400_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", required=True)
    p = sub.add_parser("train")
    p.add_argument("--start", required=True, help="sm82_time.py extend output: the released master + the zero head")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--warmup", type=float, default=0.02)
    p.add_argument("--buffer-rows", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--max-users", type=int, default=0, help="smoke: only the first N plan users with <= --max-user-reviews")
    p.add_argument("--max-user-reviews", type=int, default=20_000)
    p.add_argument("--smoke-longest", action="store_true", help="smoke: add the plan's longest user (cut at the cap)")
    p.add_argument("--stop-after-passes", type=int, default=0, help="check only: stop after N buffer passes")
    p = sub.add_parser("check")
    p.add_argument("--ext", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("gradecompare")
    p.add_argument("--rows", required=True)
    p.add_argument("--dump", required=True)
    p.add_argument("--rel-tag", required=True)
    p.add_argument("--new-tag", required=True)
    p.add_argument("--users-file", required=True)
    p.add_argument("--existing", default="")
    p.add_argument("--existing-label", default="")
    p.add_argument("--out", required=True)
    p = sub.add_parser("decide")
    p.add_argument("--time-rows", required=True)
    p.add_argument("--sm82-time-rows", default=str(SM82 / "gpu/sm82-time.jsonl"))
    p.add_argument("--baselines", default=str(SM82 / "time-baselines.jsonl"))
    p.add_argument("--grade-check", required=True)
    p.add_argument("--train-json", required=True)
    args = parser.parse_args()
    return {"plan": plan, "train": train, "check": check, "gradecompare": gradecompare, "decide": decide}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
