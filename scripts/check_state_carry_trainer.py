"""CPU check of SM21 v1 in the R01 trainer (scripts/state_carry_train.py; docs/SMALL_MODEL.md, SM21 work item (b)).

Real gate-tier users of the ctx2048 run (protocol rows, train split), the w64 e6 checkpoint, float32, CPU eager
(RWKV_NO_JIT=1) unless noted; the run's own options without the packed IPC transport and the teacher. No GPU.

  A  option absent = today: every batch the prefetch path builds (`training_batches` -> worker `prepare_job`), and
     the loss and every gradient of each user's first update (train mode, the trainer's dropout seed), from this tree
     and from the source tree before SM21 (`git archive` of the parent of the first "SM21 (a)" commit), sha256 equal;
  B  carry path, empty table = stock forward on the same (carry) batch: loss and every gradient bit for bit, train mode;
  C  consistent inputs: with state_carry, every window row of every update equals (bitwise) the row of the user's
     full-history sample built by the same path (one id/day encoding draw per user, user-level day features);
  D  exact carry through the trainer path: chunk 2,048, one window per update, fp32 table, eval mode: every row's
     rating probabilities and curve weights match the single full-history forward (|dp| < 1e-4); the deploy table
     (int8 card/note, fp16 deck/preset/user) and the "carry": false control (zero start) are reported for scale;
  E  v1 semantics (several windows per update read the table as of the update start, latest window wins):
     the trainer path equals the prototype (`scripts/state_carry.carry_forward` + `FreshCarry`, fp32) bit for bit on
     every update of a user, two windows per update;
  F  user boundary: the table is cleared at a new user's first update (its outputs equal a fresh StateCarry's bit for
     bit), and a user that does not start at offset 0 is refused;
  G  JIT (scripted model, the trainer's mode, subprocess): the hand-over from the scripted forward reaches
     StateCarry.streams for the student only; empty-table loss vs stock (max |diff|) and a carried update differs.
Prints ALL OK and writes artifacts/small-model/check-state-carry-trainer.json.

  .venv/Scripts/python.exe -X utf8 scripts/check_state_carry_trainer.py [--users 1340,8254]
"""
from argparse import ArgumentParser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = os.environ.get("GPU_BATCH1_DATA_PATH", r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
RUN = Path("C:/Users/jonat/workspace/wt/small-model/artifacts/accuracy/experiments/scale-w64kd-gate-cached-ctx2048")
CKPT = RUN / "run" / "epoch-06" / "expertium" / "E89-F65e-F77a" / "master.pth"

# Runs in a subprocess with sys.path[0] = the tree under test; prints one JSON line of digests.
PROBE = r'''
import hashlib, json, os, sys
root = sys.argv[1]
sys.path.insert(0, root)
os.chdir(root)
os.environ["RWKV_NO_JIT"] = "1"
import torch
torch.set_num_threads(2)
from concurrent.futures import ThreadPoolExecutor
from scripts.gpu_r01_prefetch import training_batches
from scripts.expertium_options import CatchUp
from expertium.variants import build
spec = json.loads(sys.argv[2])
options, rows, span, ckpt = spec["options"], spec["rows"], spec["span"], spec["checkpoint"]

def digest_batch(batch):
    h = hashlib.sha256()
    for name in ("start", "labels", "label_review_th"):
        h.update(getattr(batch, name).contiguous().numpy().tobytes())
    for group in (batch.sub_gather, batch.time_shift_selects, batch.skips):
        for splits in group:
            for t in splits:
                h.update(t.contiguous().numpy().tobytes())
    h.update(json.dumps([int(batch.num_data), [[int(v) for v in s] for s in batch.sub_gather_lens]]).encode())
    return h.hexdigest()

model = build("w64").float()
model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
catch_up = CatchUp(options, {}, rows)
out, seen = {"batches": [], "losses": []}, set()
# Materialize first: the in-process preparation thread reseeds the global RNG (the trainer's workers are processes).
with ThreadPoolExecutor(max_workers=1) as pool:
    prepared = list(training_batches(rows, span, options["training_seed"], pool, options, None, 2))
for row, item in prepared:
        batch = item.batch.to(torch.device("cpu"))
        out["batches"].append([row["user_id"], item.offset, item.seed, digest_batch(batch)])
        if row["user_id"] in seen:
            continue
        seen.add(row["user_id"])
        model.train()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(item.seed)
        stats = catch_up.loss(model, batch, 0, (row["user_id"], item.offset, item.reviews, item.seed), item.batch.labels)
        stats.average_loss.backward()
        h = hashlib.sha256(stats.average_loss.detach().numpy().tobytes())
        for p in model.parameters():
            h.update(b"-" if p.grad is None else p.grad.numpy().tobytes())
        out["losses"].append([row["user_id"], float(stats.average_loss), h.hexdigest()])
print("PROBE " + json.dumps(out))
'''


def base_revision():
    commits = subprocess.run(["git", "log", "--reverse", "--format=%H %s", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.splitlines()
    first = next(line.split()[0] for line in commits if line.split(" ", 1)[1].startswith("SM21 (a)"))
    return subprocess.run(["git", "rev-parse", f"{first}^"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def run_probe(tree, spec, tmp):
    probe = Path(tmp) / "probe.py"
    probe.write_text(PROBE, encoding="utf-8")
    env = dict(os.environ, GPU_BATCH1_DATA_PATH=DATA, RWKV_NO_JIT="1", PYTHONPATH="", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-X", "utf8", str(probe), str(tree), json.dumps(spec)], cwd=tree,
                            capture_output=True, text=True, env=env)
    line = next((x for x in result.stdout.splitlines() if x.startswith("PROBE ")), None)
    if line is None:
        raise RuntimeError(f"probe failed in {tree}: {result.stderr[-3000:]}")
    return json.loads(line[6:])


def tensors_digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(b"-" if t is None else t.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--users", default="1340,8254")
    parser.add_argument("--output", type=Path, default=Path("artifacts/small-model/check-state-carry-trainer.json"))
    parser.add_argument("--skip-base", action="store_true", help="skip A's comparison with the pre-SM21 tree")
    parser.add_argument("--jit-probe", action="store_true", help=argparse_suppress())
    args = parser.parse_args()
    if args.jit_probe:
        return jit_probe(args)
    os.environ["RWKV_NO_JIT"] = "1"
    os.environ["GPU_BATCH1_DATA_PATH"] = DATA
    sys.path.insert(0, str(ROOT))
    import torch
    torch.set_num_threads(4)
    from concurrent.futures import ThreadPoolExecutor
    from expertium.variants import build
    from expertium.model import variant_model
    from scripts.expertium_options import CatchUp
    from scripts.gpu_r01_prefetch import training_batches
    from scripts.state_carry_train import StateCarry
    from scripts import state_carry as proto

    started = time.perf_counter()
    users = [int(u) for u in args.users.split(",")]
    protocol = json.loads((RUN / "run" / "epoch-01" / "protocol.json").read_text())
    run_options = dict(json.loads((RUN / "experiment.json").read_text())["options"])
    plain = {k: v for k, v in run_options.items() if k not in ("packed_ipc", "distillation")}
    rows = [next(r for r in protocol["rows"] if r["user_id"] == u) for u in users]
    span = plain["batch_chunks"] * plain["chunk_reviews"]
    model = build("w64").float()
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True), strict=True)
    report = {"users": users, "reviews": {r["user_id"]: r["train_range"][1] - r["train_range"][0] for r in rows},
              "checkpoint": str(CKPT), "options": plain, "checks": {}}
    fails = []

    def items(options, span_reviews, user_rows=rows):
        with ThreadPoolExecutor(max_workers=1) as pool:
            for row, item in training_batches(user_rows, span_reviews, options["training_seed"], pool, options, None, 2):
                yield row, item, item.batch.to(torch.device("cpu"))

    def outputs(stats):
        return stats.p_imm_all.detach(), stats.w.detach()

    # ---------------- A: option absent = today (this tree vs the tree before SM21)
    if not args.skip_base:
        base = base_revision()
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "base"
            tree.mkdir()
            archive = subprocess.run(["git", "archive", base, "scripts", "rwkv", "expertium"], cwd=ROOT,
                                     capture_output=True, check=True).stdout
            subprocess.run(["tar", "-x", "-f", "-", "-C", str(tree)], input=archive, check=True)
            spec = {"options": plain, "rows": rows, "span": span, "checkpoint": str(CKPT)}
            ours = run_probe(ROOT, spec, tmp)
            theirs = run_probe(tree, spec, tmp)
        a = ours == theirs and len(ours["batches"]) > 0
        report["checks"]["A_option_absent_equals_before_sm21"] = {
            "pass": a, "base_revision": base, "batches": len(ours["batches"]), "losses": ours["losses"],
            "batch_digests_equal": ours["batches"] == theirs["batches"], "loss_gradient_digests_equal": ours["losses"] == theirs["losses"]}
        fails += [] if a else ["A"]

    carry_options = dict(plain, state_carry={"precision": "fp32"})

    # ---------------- B: empty table = stock forward on the carry batch, train mode, loss + gradients
    b_rows = []
    # materialized first: the in-process preparation thread reseeds the global RNG while it runs
    for row, item, batch in list(items(carry_options, span)):
        if item.offset != 0:
            continue
        results = []
        for carried in (False, True):
            model.train()
            model.zero_grad(set_to_none=True)
            torch.manual_seed(item.seed)
            if carried:
                catch_up = CatchUp(carry_options, {}, rows)
                catch_up.carry.attach(item, batch, row["user_id"])
                stats = catch_up.loss(model, batch, 0, (row["user_id"], item.offset, item.reviews, item.seed))
                calls = catch_up.carry.stats["streams_calls"]
            else:
                stats = model.get_loss(batch)
            stats.average_loss.backward()
            results.append(tensors_digest([stats.average_loss, *[p.grad for p in model.parameters()]]))
        b_rows.append({"user": row["user_id"], "stock": results[0], "carry_empty_table": results[1],
                       "streams_calls": calls, "equal": results[0] == results[1] and calls == 1})
    b = all(r["equal"] for r in b_rows) and len(b_rows) == len(users)
    report["checks"]["B_empty_table_equals_stock_bitwise"] = {"pass": b, "rows": b_rows}
    fails += [] if b else ["B"]
    model.zero_grad(set_to_none=True)
    model.eval()

    # ---------------- C + D: consistent inputs and exact carry, one 2,048-review window per update
    from rwkv.data_processing import get_rwkv_data
    from scripts.gpu_batch1 import split_frame
    from scripts.gpu_r01_samples import samples_for
    from scripts.state_carry_train import user_segment_features
    frames = {row["user_id"]: user_segment_features(split_frame(get_rwkv_data(Path(DATA), row["user_id"]), row, "train"))
              for row in rows}

    def row_keys(options, item, user):
        """(review_th, query) of every canonical row of a single-window batch, from the worker's own samples."""
        (sample,) = samples_for(frames[user].iloc[item.offset:item.offset + item.reviews], user, options)
        return (sample.review_ths.long() * 2 + sample.skips.long()).tolist()

    one = dict(carry_options, batch_chunks=1)
    c_rows, d_rows = [], []
    for row in rows:
        user = row["user_id"]
        reviews = row["train_range"][1] - row["train_range"][0]
        full_options = dict(one, chunk_reviews=reviews)
        _, full_item, full = next(items(full_options, reviews, [row]))
        with torch.no_grad():
            full_out = model.forward_batch_variant(full.start, full.sub_gather, full.sub_gather_lens,
                                                   full.time_shift_selects, full.skips, full.num_data)
        full_p, full_w = torch.softmax(full_out[3].float(), dim=-1)[0], full_out[1].float()[0]
        where = {k: i for i, k in enumerate(row_keys(full_options, full_item, user))}
        labelled = full.labels[0, :, 4] > 0
        windows = [(item, batch, torch.tensor([where[k] for k in row_keys(one, item, user)]))
                   for _, item, batch in items(one, plain["chunk_reviews"], [row])]
        mismatched_rows, compared_rows = 0, 0
        for item, batch, index in windows:
            n = index.numel()
            same = (batch.start[:n] == full.start.index_select(0, index)).all(dim=1)
            mismatched_rows += int((~same).sum())
            compared_rows += n
        c_rows.append({"user": user, "windows": len(windows), "rows": compared_rows, "mismatched_rows": mismatched_rows})
        results = {}
        for mode in ("fp32", "deploy", "none"):
            spec = {"precision": "fp32", "carry": False} if mode == "none" else {"precision": mode}
            catch_up = CatchUp(dict(one, state_carry=spec), {}, [row])
            max_dp, max_dw, gap, count = 0.0, 0.0, 0.0, 0
            for item, batch, index in windows:
                n = index.numel()
                catch_up.carry.attach(item, batch, user)
                with torch.no_grad():
                    stats = catch_up.loss(model, batch, 0)
                p, w = outputs(stats)
                dp = (p[0, :n].float() - full_p.index_select(0, index)).abs()
                max_dp = max(max_dp, float(dp.max()))
                max_dw = max(max_dw, float((w[0, :n].float() - full_w.index_select(0, index)).abs().max()))
                mask = labelled.index_select(0, index)
                gap += float(dp.max(dim=-1).values[mask].sum())
                count += int(mask.sum())
            results[mode] = {"max_abs_dp": max_dp, "max_abs_dw": max_dw, "mean_max_dp_labelled": gap / max(count, 1),
                             "table_bytes": catch_up.carry.table.nbytes()}
        d_rows.append({"user": user, "windows": len(windows), **results})
    c = all(r["mismatched_rows"] == 0 and r["rows"] > 0 for r in c_rows)
    d = all(r["fp32"]["max_abs_dp"] < 1e-4 and r["fp32"]["max_abs_dw"] < 1e-4 and r["windows"] > 1 for r in d_rows)
    report["checks"]["C_consistent_inputs_bitwise"] = {"pass": c, "rows": c_rows}
    report["checks"]["D_exact_carry_equals_full_history"] = {"pass": d, "rows": d_rows}
    fails += ([] if c else ["C"]) + ([] if d else ["D"])

    # ---------------- E: v1 (two windows per update) = prototype FreshCarry, bitwise
    two = dict(carry_options, batch_chunks=2)
    e_rows = []
    for row in rows:
        user = row["user_id"]
        frame = frames[user]
        catch_up = CatchUp(two, {}, [row])
        fresh = proto.FreshCarry()
        equal, updates, offsets = True, 0, []
        for _, item, batch in items(two, 2 * plain["chunk_reviews"], [row]):
            catch_up.carry.attach(item, batch, user)
            with torch.no_grad():
                stats = catch_up.loss(model, batch, 0)
                global_T = batch.start.size(0) // batch.num_data
                # the prototype's canonical ids from the frame (int64) and window numbers within the update
                samples = samples_for(frame.iloc[item.offset:item.offset + item.reviews], user, two)
                ids = proto.canonical_ids(samples, global_T, frame)
                theirs, finals = proto.carry_forward(model, batch, ids, proto.canonical_windows(samples, global_T), fresh)
                fresh.commit(finals)
            ours = (stats.p_imm_all, stats.w)
            theirs_p = torch.softmax(theirs[3], dim=-1)
            equal = equal and torch.equal(ours[0], theirs_p) and torch.equal(ours[1], theirs[1])
            updates += 1
            offsets.append(item.offset)
        e_rows.append({"user": user, "updates": updates, "offsets": offsets, "bitwise_equal": equal})
    e = all(r["bitwise_equal"] and r["updates"] > 1 for r in e_rows)
    report["checks"]["E_v1_equals_prototype_fresh_carry"] = {"pass": e, "rows": e_rows}
    fails += [] if e else ["E"]

    # ---------------- F: user boundary
    second = rows[-1]
    catch_up = CatchUp(one, {}, rows)
    first_items = list(items(one, plain["chunk_reviews"], rows))
    boundary_out, slots_after_attach, refused = None, None, False
    for row, item, batch in first_items:
        catch_up.carry.attach(item, batch, row["user_id"])
        if row["user_id"] == second["user_id"] and item.offset == 0:
            slots_after_attach = len(catch_up.carry.table.slots)
        with torch.no_grad():
            stats = catch_up.loss(model, batch, 0)
        if row["user_id"] == second["user_id"] and item.offset == 0:
            boundary_out = (stats.p_imm_all.clone(), stats.w.clone())
    fresh_up = CatchUp(one, {}, [second])
    _, item0, batch0 = next(items(one, plain["chunk_reviews"], [second]))
    fresh_up.carry.attach(item0, batch0, second["user_id"])
    with torch.no_grad():
        stats0 = fresh_up.loss(model, batch0, 0)
    try:
        late = [x for x in first_items if x[1].offset > 0][0]
        StateCarry(True).attach(late[1], late[2], late[0]["user_id"])
    except RuntimeError:
        refused = True
    f = (len(users) > 1 and slots_after_attach == 0 and boundary_out is not None
         and torch.equal(boundary_out[0], stats0.p_imm_all) and torch.equal(boundary_out[1], stats0.w) and refused)
    report["checks"]["F_user_boundary"] = {"pass": f, "slots_after_new_user_attach": slots_after_attach,
                                           "first_update_equals_fresh_table": boundary_out is not None and torch.equal(boundary_out[0], stats0.p_imm_all),
                                           "non_zero_offset_start_refused": refused}
    fails += [] if f else ["F"]
    assert variant_model.CARRY_HOOK[0] is None

    # ---------------- G: scripted (JIT) model, subprocess
    env = dict(os.environ, GPU_BATCH1_DATA_PATH=DATA, CUDA_VISIBLE_DEVICES="")
    env.pop("RWKV_NO_JIT", None)
    result = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--jit-probe",
                             "--users", str(users[0])], cwd=ROOT, capture_output=True, text=True, env=env)
    line = next((x for x in result.stdout.splitlines() if x.startswith("JIT ")), None)
    g_info = json.loads(line[4:]) if line else {"error": result.stderr[-2000:]}
    g = bool(line) and g_info["scripted"] and g_info["streams_calls"] == 2 and g_info["teacher_calls"] == 0 \
        and g_info["empty_table_max_abs_loss_diff"] < 1e-5 and g_info["carried_update_differs"]
    report["checks"]["G_jit_hand_over"] = {"pass": g, **g_info}
    fails += [] if g else ["G"]

    report["fails"] = fails
    report["passed"] = not fails
    report["seconds"] = round(time.perf_counter() - started, 1)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v.get("pass") for k, v in report["checks"].items()}))
    print("ALL OK" if not fails else f"FAILED: {fails}")
    return 0 if not fails else 1


def argparse_suppress():
    import argparse
    return argparse.SUPPRESS


def jit_probe(args):
    """Scripted student (no RWKV_NO_JIT): the scripted forward hands over to StateCarry.streams; a scripted teacher
    model in the same active context is not carried."""
    os.environ["GPU_BATCH1_DATA_PATH"] = DATA
    sys.path.insert(0, str(ROOT))
    import torch
    torch.set_num_threads(4)
    from concurrent.futures import ThreadPoolExecutor
    from expertium.variants import build
    from scripts.expertium_options import CatchUp
    from scripts.gpu_r01_prefetch import training_batches
    protocol = json.loads((RUN / "run" / "epoch-01" / "protocol.json").read_text())
    options = {k: v for k, v in json.loads((RUN / "experiment.json").read_text())["options"].items()
               if k not in ("packed_ipc", "distillation")}
    options = dict(options, batch_chunks=1, state_carry={"precision": "fp32"})
    row = next(r for r in protocol["rows"] if r["user_id"] == int(args.users))
    model = build("w64").float()
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    teacher = build("w64").float()
    teacher.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True), strict=True)
    teacher.eval()
    catch_up = CatchUp(options, {}, [row])
    with ThreadPoolExecutor(max_workers=1) as pool:
        batches = []
        for _, item in training_batches([row], options["chunk_reviews"], options["training_seed"], pool, options, None, 2):
            batches.append((item, item.batch.to(torch.device("cpu"))))
            if len(batches) == 2:
                break
    out = {"scripted": isinstance(model, torch.jit.ScriptModule)}
    with torch.no_grad():
        item, batch = batches[0]
        stock = model.get_loss(batch)
        catch_up.carry.attach(item, batch, row["user_id"])
        with catch_up.carry.active(model):
            carried = model.get_loss(batch)
            teacher.get_loss(batch)  # another model object: stock path
        out["empty_table_max_abs_loss_diff"] = float((stock.average_loss - carried.average_loss).abs())
        item, batch = batches[1]
        stock = model.get_loss(batch)
        catch_up.carry.attach(item, batch, row["user_id"])
        carried = catch_up.loss(model, batch, 0)
        out["carried_update_differs"] = not torch.equal(stock.p_imm_all, carried.p_imm_all)
        out["carried_update_max_abs_dp"] = float((stock.p_imm_all - carried.p_imm_all).abs().max())
    out["streams_calls"] = catch_up.carry.stats["streams_calls"]
    out["teacher_calls"] = out["streams_calls"] - 2
    print("JIT " + json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
