"""CPU check of SM21 v2, strided lanes (`"state_carry": {"lanes": "strided"}`, scripts/state_carry_train.py).

Same setup as scripts/check_state_carry_trainer.py (users 1340 and 8254 of the gate tier, the ctx2048 run's options
without the packed IPC transport and the teacher, the w64 e6 checkpoint, float32, CPU eager unless noted). No GPU.
These users have 3 and 5 windows of 2,048, so the lane checks use 2 and 3 lanes (batch_chunks) to get several updates
and ragged lanes; L16 = the production batch_chunks.

  A0 option absent = the tree before SM21 (check_state_carry_trainer A: every prefetch batch, first-update loss and
     gradients sha256 equal);
  A1 v1 unchanged: `true` (deploy) and {"precision": "fp32"} at 2 windows per update, every update in order with the
     carry (train mode, the trainer's dropout seed): batch digests, loss + gradient digests and describe() equal to
     the v1 tree (5a2455e, `git archive`);
  L  v2 layout: same update count, offsets, seeds and review counts as v1; update u holds window u of every lane
     (strided_layout; ragged rule in the state_carry_train docstring), item.carry_lanes; each slot's input rows equal
     the user's full-history sample rows of its window bitwise; at L16 (one update) the v2 batch = the v1 batch;
  B  v2 first update (all lane starts, empty table) = stock forward: loss and every gradient bitwise, train mode;
  D  v2 exact carry through the trainer path (fp32 table, eval mode) vs the user's single full-history forward:
     zero lane starts: every lane-0 row matches (|dp| < 1e-4), the other lanes are reported (lane-start windows and
     continuation windows separately); seeded lane starts (each lane's table seeded with the exact full-history state
     at its start, taken from v1 one-window-per-update fp32 carry): every row matches (|dp| < 1e-4); deploy table
     (int8 card/note, fp16 others, seeded from the deploy v1 run) reported;
  F  user boundary: the v2 table is cleared at the next user's first update (outputs equal a fresh table bit for bit);
     a v1 universe and an update without lanes are refused;
  G  JIT (scripted model, subprocess): the hand-over reaches StateCarry.streams for the student only at 2 lanes; empty
     table vs stock, and the carried second update differs.
Prints ALL OK and writes artifacts/small-model/check-state-carry-lanes.json.

  .venv/Scripts/python.exe -X utf8 scripts/check_state_carry_lanes.py [--users 1340,8254] [--skip-base]
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
sys.path.insert(0, str(ROOT))
from scripts.check_state_carry_trainer import CKPT, DATA, PROBE, RUN, base_revision, run_probe, tensors_digest  # noqa: E402

V1_REVISION = "5a2455e"  # branch point of sm21v2 on state-carry: SM21 v1 as checked (check-state-carry-trainer.json)

# Every update of every user in order with the carry; one JSON line of digests (runs in the tree under test).
PROBE_ALL = r'''
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
out = {"updates": []}
with ThreadPoolExecutor(max_workers=1) as pool:
    prepared = list(training_batches(rows, span, options["training_seed"], pool, options, None, 2))
for row, item in prepared:
    batch = item.batch.to(torch.device("cpu"))
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(item.seed)
    catch_up.carry.attach(item, batch, row["user_id"])
    stats = catch_up.loss(model, batch, 0, (row["user_id"], item.offset, item.reviews, item.seed), item.batch.labels)
    stats.average_loss.backward()
    h = hashlib.sha256(stats.average_loss.detach().numpy().tobytes())
    for p in model.parameters():
        h.update(b"-" if p.grad is None else p.grad.numpy().tobytes())
    out["updates"].append([row["user_id"], item.offset, item.reviews, item.seed, digest_batch(batch), h.hexdigest()])
out["describe"] = catch_up.carry.describe()
print("PROBE " + json.dumps(out))
'''


def seed_lane_starts(table, snapshots):
    """Check helper (also used by the GPU check): fill a v2 table right after its reset with, for every lane j in
    `snapshots`, the v1 per-entity slots `snapshots[j]` (the table state at lane j's first window); lanes without a
    snapshot keep zero. Exact lane starts make the v2 carry reproduce the full history on every row."""
    import torch
    for (stream, layer) in sorted({key for slots in snapshots.values() for key in slots}):
        E = table.universe[stream].numel()
        keys = table.keys[stream]
        lane = torch.div(keys, E, rounding_mode="floor")
        code = keys - lane * E
        template = next(slots[(stream, layer)] for slots in snapshots.values() if (stream, layer) in slots)
        # non-tensor entries (the SM32c "rank1" flag) are copied as they are
        slot = {name: t if not torch.is_tensor(t) else torch.zeros((keys.numel(), *t.shape[1:]), dtype=t.dtype,
                                                                  device=t.device)
                for name, t in template.items()}
        if slot["scale"] is not None:
            slot["scale"].fill_(1)
        for j, slots in snapshots.items():
            if (stream, layer) not in slots:
                continue
            mask = lane == j
            for name, t in slots[(stream, layer)].items():
                if torch.is_tensor(t):
                    slot[name][mask] = t.index_select(0, code[mask])
        table.slots[(stream, layer)] = slot


def clone_slots(table):
    return {key: {name: t.clone() if hasattr(t, "clone") else t for name, t in slot.items()}
            for key, slot in table.slots.items()}


def digest(batch):
    h = hashlib.sha256()
    for name in ("start", "labels", "label_review_th"):
        h.update(getattr(batch, name).contiguous().numpy().tobytes())
    for group in (batch.sub_gather, batch.time_shift_selects, batch.skips):
        for splits in group:
            for t in splits:
                h.update(t.contiguous().numpy().tobytes())
    h.update(json.dumps([int(batch.num_data), [[int(v) for v in s] for s in batch.sub_gather_lens]]).encode())
    return h.hexdigest()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--users", default="1340,8254")
    parser.add_argument("--output", type=Path, default=Path("artifacts/small-model/check-state-carry-lanes.json"))
    parser.add_argument("--skip-base", action="store_true", help="skip A0/A1 (the comparisons with other trees)")
    parser.add_argument("--jit-probe", action="store_true", help=__import__("argparse").SUPPRESS)
    # SM72: another architecture and checkpoint (e.g. --architecture fork --checkpoint <the fork .pth>); the run's
    # options get that architecture with the switches it cannot run removed (compatible_options). Defaults = w64 e6.
    parser.add_argument("--architecture", default="w64")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    if args.jit_probe:
        return jit_probe(args)
    os.environ["RWKV_NO_JIT"] = "1"
    os.environ["GPU_BATCH1_DATA_PATH"] = DATA
    import torch
    torch.set_num_threads(4)
    from concurrent.futures import ThreadPoolExecutor
    from expertium.variants import build
    from expertium.model import variant_model
    from rwkv.data_processing import get_rwkv_data
    from scripts.expertium_options import CatchUp
    from scripts.gpu_batch1 import split_frame
    from scripts.gpu_r01_prefetch import training_batches
    from scripts.gpu_r01_samples import samples_for
    from scripts.state_carry_train import StateCarry, strided_layout, user_segment_features

    started = time.perf_counter()
    users = [int(u) for u in args.users.split(",")]
    protocol = json.loads((RUN / "run" / "epoch-01" / "protocol.json").read_text())
    run_options = dict(json.loads((RUN / "experiment.json").read_text())["options"])
    plain = {k: v for k, v in run_options.items() if k not in ("packed_ipc", "distillation")}
    ckpt = args.checkpoint or str(CKPT)
    if args.architecture != "w64":
        from scripts.expertium_options import compatible_options
        plain, dropped = compatible_options(dict(plain, architecture=args.architecture))
    rows = [next(r for r in protocol["rows"] if r["user_id"] == u) for u in users]
    W = plain["chunk_reviews"]
    span = plain["batch_chunks"] * W
    model = build(args.architecture).float()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    report = {"users": users, "reviews": {r["user_id"]: r["train_range"][1] - r["train_range"][0] for r in rows},
              "checkpoint": ckpt, "architecture": args.architecture, "v1_revision": V1_REVISION, "checks": {}}
    if args.architecture != "w64":
        report["options_dropped"] = dropped
    fails = []

    def items(options, user_rows=rows):
        with ThreadPoolExecutor(max_workers=1) as pool:
            prepared = list(training_batches(user_rows, options["batch_chunks"] * W, options["training_seed"], pool,
                                             options, None, 2))
        return [(row, item, item.batch.to(torch.device("cpu"))) for row, item in prepared]

    def opts(lanes, spec):
        return dict(plain, batch_chunks=lanes, state_carry=spec)

    # ---------------- A0 / A1: option absent and v1 against their reference trees
    if not args.skip_base:
        with tempfile.TemporaryDirectory() as tmp:
            trees = {}
            for name, revision in (("base", base_revision()), ("v1", V1_REVISION)):
                tree = trees[name] = Path(tmp) / name
                tree.mkdir()
                archive = subprocess.run(["git", "archive", revision, "scripts", "rwkv", "expertium"], cwd=ROOT,
                                         capture_output=True, check=True).stdout
                subprocess.run(["tar", "-x", "-f", "-", "-C", str(tree)], input=archive, check=True)
            spec = {"options": plain, "rows": rows, "span": span, "checkpoint": str(CKPT)}
            ours, theirs = run_probe(ROOT, spec, tmp), run_probe(trees["base"], spec, tmp)
            a0 = ours == theirs and len(ours["batches"]) > 0
            report["checks"]["A0_option_absent_equals_before_sm21"] = {
                "pass": a0, "base_revision": base_revision(), "batches": len(ours["batches"]), "losses": ours["losses"]}
            fails += [] if a0 else ["A0"]
            a1_rows = []
            probe = Path(tmp) / "probe_all.py"
            probe.write_text(PROBE_ALL, encoding="utf-8")
            env = dict(os.environ, GPU_BATCH1_DATA_PATH=DATA, RWKV_NO_JIT="1", PYTHONPATH="", CUDA_VISIBLE_DEVICES="")
            for carry_spec in (True, {"precision": "fp32"}):
                spec = {"options": opts(2, carry_spec), "rows": rows, "span": 2 * W, "checkpoint": str(CKPT)}
                results = []
                for tree in (ROOT, trees["v1"]):
                    result = subprocess.run([sys.executable, "-X", "utf8", str(probe), str(tree), json.dumps(spec)],
                                            cwd=tree, capture_output=True, text=True, env=env)
                    line = next((x for x in result.stdout.splitlines() if x.startswith("PROBE ")), None)
                    if line is None:
                        raise RuntimeError(f"probe failed in {tree}: {result.stderr[-3000:]}")
                    results.append(json.loads(line[6:]))
                a1_rows.append({"spec": carry_spec, "updates": len(results[0]["updates"]),
                                "updates_equal": results[0]["updates"] == results[1]["updates"],
                                "describe_equal": results[0]["describe"] == results[1]["describe"],
                                "describe_mode": results[0]["describe"]["mode"]})
            a1 = all(r["updates_equal"] and r["describe_equal"] and r["updates"] > len(users) for r in a1_rows)
            report["checks"]["A1_v1_unchanged"] = {"pass": a1, "rows": a1_rows}
            fails += [] if a1 else ["A1"]

    # ---------------- full-history references
    frames = {row["user_id"]: user_segment_features(split_frame(get_rwkv_data(Path(DATA), row["user_id"]), row, "train"))
              for row in rows}

    def window_keys(options, user, window):
        (sample,) = samples_for(frames[user].iloc[window * W:(window + 1) * W], user, options)
        return (sample.review_ths.long() * 2 + sample.skips.long()).tolist()

    full = {}
    one = opts(1, {"precision": "fp32"})
    model.eval()
    for row in rows:
        user = row["user_id"]
        reviews = len(frames[user])
        full_options = dict(one, chunk_reviews=reviews)
        (_, _, batch), = _full_items(training_batches, ThreadPoolExecutor, full_options, reviews, row)
        with torch.no_grad():
            out = model.forward_batch_variant(batch.start, batch.sub_gather, batch.sub_gather_lens,
                                              batch.time_shift_selects, batch.skips, batch.num_data)
        (sample,) = samples_for(frames[user], user, full_options)
        keys = (sample.review_ths.long() * 2 + sample.skips.long()).tolist()
        full[user] = {"batch": batch, "p": torch.softmax(out[3].float(), dim=-1)[0], "w": out[1].float()[0],
                      "where": {k: i for i, k in enumerate(keys)}, "windows": -(-reviews // W)}

    def index_of(options, user, window):
        return torch.tensor([full[user]["where"][k] for k in window_keys(options, user, window)])

    # ---------------- L: layout
    l_rows = []
    for lanes in (2, 3, plain["batch_chunks"]):
        v1_items = items(opts(lanes, {"precision": "fp32"}))
        v2_items = items(opts(lanes, {"precision": "fp32", "lanes": "strided"}))
        for row in rows:
            user = row["user_id"]
            ones = [x for x in v1_items if x[0]["user_id"] == user]
            twos = [x for x in v2_items if x[0]["user_id"] == user]
            layout = strided_layout(full[user]["windows"], lanes)
            same_meta = [(i.offset, i.reviews, i.seed) for _, i, _ in ones] == [(i.offset, i.reviews, i.seed) for _, i, _ in twos]
            lanes_ok = [i.carry_lanes for _, i, _ in twos] == [[j for j, _ in update] for update in layout]
            rows_ok, compared = True, 0
            for (_, item, batch), update in zip(twos, layout):
                global_T = batch.start.size(0) // batch.num_data
                for b, (_, window) in enumerate(update):
                    index = index_of(opts(lanes, {"lanes": "strided"}), user, window)
                    n = index.numel()
                    rows_ok &= bool(torch.equal(batch.start[b * global_T:b * global_T + n],
                                                full[user]["batch"].start.index_select(0, index)))
                    compared += n
            entry = {"lanes": lanes, "user": user, "updates": len(twos), "layout": layout,
                     "same_updates_offsets_reviews_seeds_as_v1": same_meta and len(ones) == len(twos) == len(layout),
                     "carry_lanes_ok": lanes_ok, "rows_equal_full_history_rows": rows_ok, "rows": compared}
            if len(layout) == 1:
                entry["batch_equals_v1"] = digest(twos[0][2]) == digest(ones[0][2])
            l_rows.append(entry)
    ll = all(r["same_updates_offsets_reviews_seeds_as_v1"] and r["carry_lanes_ok"] and r["rows_equal_full_history_rows"]
             and r.get("batch_equals_v1", True) and r["rows"] > 0 for r in l_rows) and any(
        len(r["layout"]) > 2 for r in l_rows)
    report["checks"]["L_v2_layout"] = {"pass": ll, "rows": l_rows}
    fails += [] if ll else ["L"]

    # ---------------- B: v2 first update = stock, train mode
    b_rows = []
    v2_train = opts(2, {"precision": "fp32", "lanes": "strided"})
    for row, item, batch in items(v2_train):
        if item.offset != 0:
            continue
        results = []
        for carried in (False, True):
            model.train()
            model.zero_grad(set_to_none=True)
            torch.manual_seed(item.seed)
            if carried:
                catch_up = CatchUp(v2_train, {}, rows)
                catch_up.carry.attach(item, batch, row["user_id"])
                stats = catch_up.loss(model, batch, 0, (row["user_id"], item.offset, item.reviews, item.seed))
            else:
                stats = model.get_loss(batch)
            stats.average_loss.backward()
            results.append(tensors_digest([stats.average_loss, *[p.grad for p in model.parameters()]]))
        b_rows.append({"user": row["user_id"], "equal": results[0] == results[1], "windows": int(batch.num_data)})
    b = len(b_rows) == len(users) and all(r["equal"] for r in b_rows)
    report["checks"]["B_v2_first_update_equals_stock_bitwise"] = {"pass": b, "rows": b_rows}
    fails += [] if b else ["B"]
    model.zero_grad(set_to_none=True)
    model.eval()

    # ---------------- D: exact carry
    def v1_snapshots(user_row, precision, starts):
        """v1 one window per update: the table state before each lane-start window (= the exact full-history state
        of every entity seen so far, at that precision)."""
        catch_up = CatchUp(opts(1, {"precision": precision}), {}, [user_row])
        snaps = {}
        for window, (_, item, batch) in enumerate(items(opts(1, {"precision": precision}), [user_row])):
            if window in starts:
                snaps[starts[window]] = clone_slots(catch_up.carry.table)
            catch_up.carry.attach(item, batch, user_row["user_id"])
            with torch.no_grad():
                catch_up.loss(model, batch, 0)
        return snaps

    d_rows = []
    for lanes in (2, 3):
        for row in rows:
            user = row["user_id"]
            layout = strided_layout(full[user]["windows"], lanes)
            starts = {window: lane for lane, window in layout[0] if window > 0}
            entry = {"lanes": lanes, "user": user, "updates": len(layout)}
            for mode in ("zero_fp32", "seeded_fp32", "seeded_deploy", "zero_deploy"):
                precision = mode.split("_")[1]
                options = opts(lanes, {"precision": precision, "lanes": "strided"})
                snaps = v1_snapshots(row, precision, starts) if mode.startswith("seeded") else None
                catch_up = CatchUp(options, {}, [row])
                worst = {"lane0": 0.0, "lane_start_windows": 0.0, "continuation_windows": 0.0, "all": 0.0, "w": 0.0}
                for u, ((_, item, batch), update) in enumerate(zip(items(options, [row]), layout)):
                    catch_up.carry.attach(item, batch, user)
                    if u == 0 and snaps:
                        seed_lane_starts(catch_up.carry.table, snaps)
                    with torch.no_grad():
                        stats = catch_up.loss(model, batch, 0)
                    for b, (lane, window) in enumerate(update):
                        index = index_of(options, user, window)
                        n = index.numel()
                        dp = float((stats.p_imm_all[b, :n].float() - full[user]["p"].index_select(0, index)).abs().max())
                        dw = float((stats.w[b, :n].float() - full[user]["w"].index_select(0, index)).abs().max())
                        worst["all"] = max(worst["all"], dp)
                        worst["w"] = max(worst["w"], dw)
                        # the first lane of update 0 starts at window 0: its windows are the exact history prefix
                        group = ("lane0" if lane == layout[0][0][0] else
                                 "lane_start_windows" if u == 0 else "continuation_windows")
                        worst[group] = max(worst[group], dp)
                entry[mode] = {**{f"max_abs_dp_{k}": v for k, v in worst.items() if k != "w"}, "max_abs_dw": worst["w"],
                               "table_bytes": catch_up.carry.table.nbytes()}
            d_rows.append(entry)
    d = all(r["seeded_fp32"]["max_abs_dp_all"] < 1e-4 and r["seeded_fp32"]["max_abs_dw"] < 1e-4
            and r["zero_fp32"]["max_abs_dp_lane0"] < 1e-4 for r in d_rows) and any(r["updates"] > 2 for r in d_rows)
    report["checks"]["D_v2_exact_carry"] = {"pass": d, "rows": d_rows}
    fails += [] if d else ["D"]

    # ---------------- F: user boundary
    v2 = opts(2, {"precision": "fp32", "lanes": "strided"})
    all_items = items(v2)
    second = rows[-1]
    catch_up = CatchUp(v2, {}, rows)
    slots_after_attach, boundary_out = None, None
    for row, item, batch in all_items:
        catch_up.carry.attach(item, batch, row["user_id"])
        if row["user_id"] == second["user_id"] and item.offset == 0:
            slots_after_attach = len(catch_up.carry.table.slots)
        with torch.no_grad():
            stats = catch_up.loss(model, batch, 0)
        if row["user_id"] == second["user_id"] and item.offset == 0:
            boundary_out = (stats.p_imm_all.clone(), stats.w.clone())
    fresh = CatchUp(v2, {}, [second])
    _, item0, batch0 = items(v2, [second])[0]
    fresh.carry.attach(item0, batch0, second["user_id"])
    with torch.no_grad():
        stats0 = fresh.loss(model, batch0, 0)
    refusals = {}
    _, v1_item, v1_batch = items(opts(2, {"precision": "fp32"}), [second])[0]
    for name, (item, batch) in {"v1_universe": (v1_item, v1_batch), "missing_lanes": (v1_item, v1_batch)}.items():
        carry = StateCarry({"precision": "fp32", "lanes": "strided"})
        if name == "missing_lanes":
            carry.user = second["user_id"]  # skip the reset: the lane check itself must refuse
        try:
            carry.attach(item, batch, second["user_id"])
            refusals[name] = False
        except RuntimeError:
            refusals[name] = True
    f = (slots_after_attach == 0 and boundary_out is not None and torch.equal(boundary_out[0], stats0.p_imm_all)
         and torch.equal(boundary_out[1], stats0.w) and all(refusals.values()) and len(users) > 1)
    report["checks"]["F_user_boundary"] = {"pass": f, "slots_after_new_user_attach": slots_after_attach,
                                           "first_update_equals_fresh_table": boundary_out is not None
                                           and torch.equal(boundary_out[0], stats0.p_imm_all), "refused": refusals}
    fails += [] if f else ["F"]
    assert variant_model.CARRY_HOOK[0] is None

    # ---------------- G: scripted model
    env = dict(os.environ, GPU_BATCH1_DATA_PATH=DATA, CUDA_VISIBLE_DEVICES="")
    env.pop("RWKV_NO_JIT", None)
    result = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--jit-probe",
                             "--users", str(users[0]), "--architecture", args.architecture, "--checkpoint", ckpt],
                            cwd=ROOT, capture_output=True, text=True, env=env)
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


def _full_items(training_batches, executor, options, reviews, row):
    with executor(max_workers=1) as pool:
        prepared = list(training_batches([row], reviews, options["training_seed"], pool, options, None, 2))
    import torch
    return [(r, item, item.batch.to(torch.device("cpu"))) for r, item in prepared]


def jit_probe(args):
    """Scripted student, 2 lanes (user 1340: updates {0, 1} and {2}); a scripted teacher in the same active context
    is not carried."""
    os.environ["GPU_BATCH1_DATA_PATH"] = DATA
    import torch
    torch.set_num_threads(4)
    from concurrent.futures import ThreadPoolExecutor
    from expertium.variants import build
    from scripts.expertium_options import CatchUp
    from scripts.gpu_r01_prefetch import training_batches
    protocol = json.loads((RUN / "run" / "epoch-01" / "protocol.json").read_text())
    options = {k: v for k, v in json.loads((RUN / "experiment.json").read_text())["options"].items()
               if k not in ("packed_ipc", "distillation")}
    options = dict(options, batch_chunks=2, state_carry={"precision": "fp32", "lanes": "strided"})
    if args.architecture != "w64":
        from scripts.expertium_options import compatible_options
        options, _ = compatible_options(dict(options, architecture=args.architecture))
    ckpt = args.checkpoint or str(CKPT)
    row = next(r for r in protocol["rows"] if r["user_id"] == int(args.users))
    model = build(args.architecture).float()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    teacher = build(args.architecture).float()
    teacher.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    teacher.eval()
    catch_up = CatchUp(options, {}, [row])
    with ThreadPoolExecutor(max_workers=1) as pool:
        batches = [(item, item.batch.to(torch.device("cpu"))) for _, item in
                   training_batches([row], 2 * options["chunk_reviews"], options["training_seed"], pool, options, None, 2)]
    out = {"scripted": isinstance(model, torch.jit.ScriptModule), "updates": len(batches),
           "lanes": [item.carry_lanes for item, _ in batches]}
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
