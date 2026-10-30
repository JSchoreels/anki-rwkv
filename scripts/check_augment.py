#!/usr/bin/env python
"""CPU checks of the data augmentation (SM67) note split (scripts/augment.py; artifacts/small-model/sm67/DESIGN.md).

  cpu  --out report.json [--work DIR]     ~10-15 min, 2-4 threads, no GPU:
    A0  flag off, loader: get_rwkv_data frames of 2 users bitwise equal to the wt/sm72s tree (run read-only, no bytecode).
    T0  flag off, trainer path (the leak-free small model's (SM72s) cool-down options, replay KD, strided-lane carry,
        2 lanes x 128 reviews, 2 small users, from the epoch-25 weights): prepared batches, teacher targets, loss,
        outputs and every gradient bitwise equal to the wt/sm72s tree.
    T1  the "augment" option on, rows of users that are not split: the same bits as T0 (unsplit users of the arm
        train exactly as in the control).
    B   3 split users, each half vs the whole user: every kept card's own inputs (intervals, cumulative / sin / cos,
        duration, rating one-hot, state, deleted-card flags), ids, has_label and labels identical on 100% of the
        rows; the halves' labelled reviews sum to the user's; the remapped label review numbers equal the whole
        user's; which context columns change (share of rows).
    C   teacher targets: on the trainer path (2 halves x 768 reviews), ReplayTargets' remapped targets equal the
        direct lookup of the real user's file at the whole user's label review, on every labelled row; and the
        frame-level index remap of every labelled row of the 3 users.
    E   seeds: every pseudo item's seed - offset = training_seed + real user (id codes, day baseline, dropout); the
        consistent id codes of a half equal the whole user's for every shared id.
    D   held-out: expand_rows refuses a held-out user; the expansion of the control cool-down's 8,000 rows (from the
        epoch-25 cool-down's untrimmed protocol) has no held-out user (464 / 732 / 2,225 pools, validation, final
        test), keeps the order and the reviews; pseudo ids never collide with real ids; the scorer never reads
        "augment".
  pair --arm DIR --control DIR --out report.json   (after the launcher's prepare, ~2 min): the arm cool-down is the
       control's exactly (options, schedule, start, users and order before the split, reviews), except "augment" and
       the expanded rows; labelled reviews equal (label-window reports); no held-out user.
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
SM72S = Path("C:/Users/jonat/workspace/wt/sm72s")  # read only
RUN72S = SM72S / "artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run"
DEC25 = SM72S / "artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon"
DATA = "C:/Users/jonat/workspace/anki-rwkv/data/anki-revlogs-10k-original-20261011"
TRACK = "E89-F65e-F77a"
SPEC = {"kind": "note_split", "min_reviews": 65536, "share": 0.5, "seed": 67}
OWN = ["scaled_elapsed_days", "scaled_elapsed_days_cumulative", "scaled_elapsed_seconds", "elapsed_seconds_sin",
       "elapsed_seconds_cos", "scaled_elapsed_seconds_cumulative", "elapsed_seconds_cumulative_sin",
       "elapsed_seconds_cumulative_cos", "scaled_duration", "rating_1", "rating_2", "rating_3", "rating_4", "scaled_state",
       "note_id_is_nan", "deck_id_is_nan", "preset_id_is_nan", "elapsed_days", "elapsed_seconds", "duration", "rating",
       "state", "i", "y", "is_first_review", "card_id", "note_id", "deck_id", "preset_id", "user_id", "has_label",
       "label_elapsed_seconds", "label_elapsed_days", "label_y", "label_rating", "label_is_equalize"]
CONTEXT = ["diff_new_cards", "diff_reviews", "cum_new_cards_today", "cum_reviews_today", "day_offset_diff", "day_of_week",
           "scaled_diff_new_cards", "scaled_diff_reviews", "scaled_cum_new_cards_today", "scaled_cum_reviews_today",
           "scaled_day_offset_diff", "day_offset", "day_offset_first"]

PROBE = r'''
import hashlib, json, os, sys
root = sys.argv[1]
sys.path.insert(0, root)
os.chdir(root)
import numpy as np
import torch
torch.set_num_threads(2)
spec = json.loads(open(sys.argv[2]).read())

def digest(value, h):
    if isinstance(value, torch.Tensor):
        h.update(str((tuple(value.shape), str(value.dtype))).encode()); h.update(value.detach().contiguous().cpu().numpy().tobytes())
    elif isinstance(value, np.ndarray):
        h.update(str(value.dtype).encode()); h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, (list, tuple)):
        h.update(b"["); [digest(v, h) for v in value]; h.update(b"]")
    elif isinstance(value, dict):
        for k in sorted(value, key=str):
            h.update(str(k).encode()); digest(value[k], h)
    elif hasattr(value, "__dict__"):
        digest(vars(value), h)
    else:
        h.update(repr(value).encode())
    return h

if spec["mode"] == "frames":
    from rwkv.data_processing import get_rwkv_data
    from pathlib import Path
    out = {}
    for user in spec["users"]:
        df = get_rwkv_data(Path(spec["data"]), user)
        h = hashlib.sha256(json.dumps([list(map(str, df.columns)), [str(t) for t in df.dtypes]]).encode())
        for col in df.columns:
            h.update(np.ascontiguousarray(df[col].to_numpy()).tobytes())
        out[str(user)] = {"rows": len(df), "sha256": h.hexdigest()}
    print("PROBE " + json.dumps(out)); raise SystemExit

from concurrent.futures import ThreadPoolExecutor
from scripts.gpu_r01_prefetch import training_batches
from scripts.expertium_options import CatchUp
from expertium.variants import build
model = build("w64").float()
model.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu", weights_only=True), strict=True)
params = dict(model.named_parameters())
options = spec["options"]
catch_up = CatchUp(options, {}, spec["rows"])
catch_up.load_teacher(torch.device("cpu"), torch.float32)
with ThreadPoolExecutor(max_workers=1) as pool:
    prepared = list(training_batches(spec["rows"], spec["span"], options["training_seed"], pool, options, None, 2))
out = []
frames = {}
for row, item in prepared:
    batch = item.batch.to(torch.device("cpu"))
    key = (row["user_id"], item.offset, item.reviews, item.seed)
    rec = {"user": row["user_id"], "offset": item.offset, "reviews": item.reviews, "seed": item.seed,
           "lanes": list(getattr(item, "carry_lanes", []) or []), "batch_sha256": digest(item.batch, hashlib.sha256()).hexdigest()}
    curve, rating = catch_up.teacher.replay.targets(batch, key, item.batch.labels)
    rec["targets_sha256"] = digest([curve, rating], hashlib.sha256()).hexdigest()
    if spec.get("direct_targets") and "augment" in row:
        # direct lookup, independent of the replay remap: the half frame's own orig_review_th column -> real user's file
        from scripts.gpu_batch1 import row_frame
        uid = row["user_id"]
        if uid not in frames:
            frame = row_frame(row, orig=True)
            frames[uid] = np.concatenate([[0], frame["orig_review_th"].to_numpy()])
        orig = torch.from_numpy(frames[uid].astype("int64"))
        src = row["augment"]["source_user"]
        with np.load(os.path.join(options["distillation"]["replay"], "users", f"{src}.npz")) as stored:
            table = torch.from_numpy(stored["targets"]).float()
        labels = item.batch.labels
        th = item.batch.label_review_th.reshape(-1).long()
        crow = torch.nonzero(((1 - labels[:, :, 6]) * labels[:, :, 4]).reshape(-1), as_tuple=True)[0]
        qrow = torch.nonzero((labels[:, :, 6] * labels[:, :, 4]).reshape(-1), as_tuple=True)[0]
        want_c = torch.sigmoid(table[orig[th[crow]] - 1, 0]).clamp(1e-5, 1 - 1e-5)
        want_q = table[orig[th[qrow]] - 1, 1:]
        rec["direct_curve_equal"] = bool(torch.equal(curve.reshape(-1)[crow], want_c))
        rec["direct_rating_equal"] = bool(torch.equal(rating.reshape(-1, 4)[qrow], want_q))
        rec["labelled_rows"] = [int(crow.numel()), int(qrow.numel())]
        rec["seed_is_real_user"] = item.seed - item.offset == options["training_seed"] + src
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(item.seed)
    catch_up.carry.attach(item, batch, row["user_id"])
    stats = catch_up.loss(model, batch, 0, key, item.batch.labels)
    stats.average_loss.backward()
    h = hashlib.sha256()
    for name, p in params.items():
        h.update(b"-" if p.grad is None else p.grad.numpy().tobytes())
    rec.update(loss=stats.average_loss.item(), loss_hex=float(stats.average_loss).hex(),
               p_digest=hashlib.sha256(stats.p_imm_all.detach().numpy().tobytes()).hexdigest(), grad_digest=h.hexdigest())
    out.append(rec)
print("PROBE " + json.dumps(out))
'''


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_probe(tree, spec, tmp):
    probe = Path(tmp) / "sm67_probe.py"
    probe.write_text(PROBE, encoding="utf-8")
    spec_file = Path(tmp) / f"spec-{hashlib.sha256(json.dumps(spec).encode()).hexdigest()[:12]}.json"
    spec_file.write_text(json.dumps(spec), encoding="utf-8")
    env = dict(os.environ, GPU_BATCH1_DATA_PATH=DATA, PYTHONPATH="", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               RWKV_DELETED_CARDS="neighbour", RWKV_NO_JIT="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(probe), str(tree), str(spec_file)], cwd=tree,
                            capture_output=True, text=True, env=env)
    line = next((x for x in result.stdout.splitlines() if x.startswith("PROBE ")), None)
    if line is None:
        raise RuntimeError(f"probe failed in {tree}: {result.stderr[-3000:]}")
    return json.loads(line[6:])


def cool_down_options():
    options = json.loads((DEC25 / "decay.json").read_text())["options"]
    options = {k: v for k, v in options.items() if k != "packed_ipc"}
    options.update(chunk_reviews=128, batch_chunks=2)
    return options


def compare_halves(user, data, counts):
    """B: the whole user's frame vs its two halves (this tree, RWKV_DELETED_CARDS=neighbour)."""
    import numpy as np
    from rwkv.data_processing import get_rwkv_data
    from scripts.augment import encode, kept_review_ths
    from scripts.gpu_batch1 import row_frame
    full = get_rwkv_data(Path(data), user)
    rec = {"user": user, "reviews": len(full), "labelled": int(full["has_label"].sum()), "halves": []}
    covered = np.zeros(len(full), dtype=int)
    ok = True
    for part in (0, 1):
        kept = kept_review_ths(data, user, part, SPEC["seed"])
        row = {"user_id": encode(user, part), "available_reviews": int(kept.size), "train_range": [0, int(kept.size)],
               "validation_range": [0, int(kept.size)], "augment": {"kind": "note_split", "seed": SPEC["seed"], "source_user": user, "part": part}}
        half = row_frame(row, orig=True)
        orig = half["orig_review_th"].to_numpy()
        same_index = bool(np.array_equal(orig, kept))
        covered[orig - 1] += 1
        ref = full.iloc[orig - 1].reset_index(drop=True)
        own_diff = {}
        for col in OWN:
            a, b = half[col].to_numpy(), ref[col].to_numpy()
            differ = (a != b) & ~((a != a) & (b != b)) if a.dtype.kind == "f" else a != b
            if differ.any():
                own_diff[col] = float(differ.mean())
        # label review numbers: the half's (renumbered) label maps back to the whole user's label review
        lab = half["has_label"].to_numpy() == 1
        h_label = half["label_review_th"].to_numpy()[lab].astype(np.int64)
        f_label = ref["label_review_th"].to_numpy()[lab].astype(np.int64)
        remap_ok = bool(np.array_equal(kept[h_label - 1], f_label))
        cards_whole = bool(set(half["card_id"]).isdisjoint(set(full["card_id"].to_numpy()[np.setdiff1d(np.arange(len(full)), orig - 1)])))
        context = {}
        for col in CONTEXT:
            if col in half:
                a, b = half[col].to_numpy().astype(float), ref[col].to_numpy().astype(float)
                context[col] = round(float((~np.isclose(a, b, rtol=0, atol=1e-9)).mean()), 4)
        h = {"part": part, "reviews": len(half), "labelled": int(lab.sum()), "orig_review_th_equals_kept": same_index,
             "own_columns_differing": own_diff, "label_remap_equal": remap_ok, "cards_entirely_in_this_half": cards_whole,
             "context_columns_changed_share": context}
        ok &= same_index and not own_diff and remap_ok and cards_whole
        rec["halves"].append(h)
    rec["every_review_in_exactly_one_half"] = bool((covered == 1).all())
    rec["labels_conserved"] = sum(h["labelled"] for h in rec["halves"]) == rec["labelled"]
    rec["reviews_match_counts"] = len(full) == counts[str(user)]
    rec["pass"] = bool(ok and rec["every_review_in_exactly_one_half"] and rec["labels_conserved"] and rec["reviews_match_counts"])
    return rec


def cpu(args):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    os.environ["RWKV_DELETED_CARDS"] = "neighbour"
    os.environ["GPU_BATCH1_DATA_PATH"] = DATA
    started = time.perf_counter()
    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "spec": SPEC, "checks": {}}
    fails = []
    control = json.loads((DEC25 / "protocol.untrimmed.json").read_text())
    rows = control["rows"]
    counts = {str(r["user_id"]): r["available_reviews"] for r in rows}
    from scripts.augment import encode, expand_rows, held_out_users, selected_users, decode, PSEUDO_BASE
    held, never = held_out_users(), held_out_users(include_2225=True)
    chosen, eligible = selected_users({int(u): n for u, n in counts.items()}, SPEC, never)
    small = sorted((r for r in rows if 300 <= r["available_reviews"] <= 1500 and r["user_id"] not in chosen),
                   key=lambda r: r["available_reviews"])
    plain_rows = [small[0], small[len(small) // 2]]
    ck = str(RUN72S / "epoch-25" / "expertium" / TRACK / "master.pth")
    options = cool_down_options()
    t = time.perf_counter()
    arm_rows, aug = expand_rows(rows, SPEC, DATA, held, never, workers=4)  # the arm's rows (D below)
    expansion_seconds = round(time.perf_counter() - t, 1)
    with tempfile.TemporaryDirectory() as tmp:
        # A0 loader, flag off
        users = [r["user_id"] for r in plain_rows]
        spec = {"mode": "frames", "users": users, "data": DATA}
        a0 = run_probe(ROOT, spec, tmp) == run_probe(SM72S, spec, tmp)
        report["checks"]["A0_flag_off_frames_equal_sm72s_tree"] = {"pass": a0, "users": users}
        fails += [] if a0 else ["A0"]
        # T0 trainer path, flag off
        spec = {"mode": "train", "checkpoint": ck, "options": options, "rows": plain_rows, "span": 256}
        ours, theirs = run_probe(ROOT, spec, tmp), run_probe(SM72S, spec, tmp)
        t0 = ours == theirs and len(ours) > len(plain_rows)
        report["checks"]["T0_flag_off_trainer_equals_sm72s_tree"] = {
            "pass": t0, "updates": len(ours), "users": users, "losses": [u["loss"] for u in ours],
            "compared": "prepared batch bytes, teacher targets, loss, rating outputs, every gradient"}
        fails += [] if t0 else ["T0"]
        # T1 option on, unsplit rows
        arm = run_probe(ROOT, dict(spec, options=dict(options, augment=SPEC)), tmp)
        t1 = arm == theirs
        report["checks"]["T1_option_on_unsplit_users_equal_sm72s_tree"] = {"pass": t1, "updates": len(arm)}
        fails += [] if t1 else ["T1"]
        # C (trainer path) + E (seeds) on two halves of the smallest split user, 768 reviews each
        user = min(chosen, key=lambda u: counts[str(u)])
        expanded = [r for r in arm_rows if r.get("augment", {}).get("source_user") == user]
        pseudo = [dict(r, train_range=[0, 768]) for r in expanded]
        spec_c = {"mode": "train", "checkpoint": ck, "options": dict(options, augment=SPEC), "rows": pseudo, "span": 256,
                  "direct_targets": True}
        got = run_probe(ROOT, spec_c, tmp)
        c_ok = all(u["direct_curve_equal"] and u["direct_rating_equal"] for u in got) and len(got) >= 4
        e_ok = all(u["seed_is_real_user"] for u in got)
        finite = all(np.isfinite(u["loss"]) for u in got)
        report["checks"]["C_trainer_targets_equal_direct_lookup"] = {
            "pass": c_ok and finite, "user": user, "pseudo_users": [r["user_id"] for r in pseudo], "updates": len(got),
            "labelled_rows_curve_query": [u["labelled_rows"] for u in got], "losses": [u["loss"] for u in got]}
        report["checks"]["E_pseudo_item_seeds_are_the_real_users"] = {"pass": e_ok, "seeds": [u["seed"] for u in got]}
        fails += ([] if c_ok and finite else ["C"]) + ([] if e_ok else ["E"])
    # E: consistent id codes of a half == the whole user's for shared ids
    from scripts.state_carry_train import consistent_id_table, BATCH_STREAMS
    from rwkv.data_processing import get_rwkv_data
    from rwkv.config import ID_ENCODE_DIMS
    from scripts.gpu_batch1 import row_frame
    full = get_rwkv_data(Path(DATA), user)
    half = row_frame(expanded[0])
    seed = options["training_seed"] + user
    codes_equal = True
    for stream in BATCH_STREAMS:
        if stream == "user_id":
            continue
        ids_h = torch.from_numpy(half[stream].to_numpy().astype(np.int64))
        ids_f = torch.from_numpy(full[stream].to_numpy().astype(np.int64))
        th, ih = consistent_id_table(ids_h, ID_ENCODE_DIMS[stream], torch.float32, seed, stream)
        tf, iff = consistent_id_table(ids_f, ID_ENCODE_DIMS[stream], torch.float32, seed, stream)
        uh, uf = np.unique(ids_h.numpy()), np.unique(ids_f.numpy())
        codes_equal &= bool(torch.equal(th, tf[torch.from_numpy(np.searchsorted(uf, uh))]))
    report["checks"]["E_id_codes_half_equal_whole"] = {"pass": codes_equal, "user": user}
    fails += [] if codes_equal else ["E_codes"]
    # B: 3 split users of increasing size (the smallest, a median and a large one under 250k reviews)
    by_size = sorted((u for u in chosen if counts[str(u)] <= 250_000), key=lambda u: counts[str(u)])
    picks = [by_size[0], by_size[len(by_size) // 2], by_size[-1]] if not args.quick else [by_size[0]]
    b = [compare_halves(u, DATA, counts) for u in picks]
    report["checks"]["B_kept_cards_identical_labels_conserved"] = {"pass": all(r["pass"] for r in b), "users": b}
    fails += [] if all(r["pass"] for r in b) else ["B"]
    # C (index level, every labelled row of the 3 users) through ReplayTargets.kept_reviews
    from scripts.expertium_options import ReplayTargets
    rt = ReplayTargets(options["distillation"], SPEC)
    c2 = True
    for u in picks:
        f = get_rwkv_data(Path(DATA), u)
        for part in (0, 1):
            row = {"user_id": encode(u, part), "available_reviews": 0, "augment": {"kind": "note_split", "seed": 67, "source_user": u, "part": part}}
            kept = rt.kept_reviews(u, part).numpy()
            row["available_reviews"] = int(kept.size)
            h = row_frame(row, orig=True)
            lab = h["has_label"].to_numpy() == 1
            mapped = kept[h["label_review_th"].to_numpy()[lab].astype(np.int64) - 1]
            whole = f["label_review_th"].to_numpy()[h["orig_review_th"].to_numpy()[lab] - 1].astype(np.int64)
            c2 &= bool(np.array_equal(mapped, whole))
    report["checks"]["C_index_remap_equals_whole_user_labels"] = {"pass": c2, "users": picks}
    fails += [] if c2 else ["C_index"]
    # D held-out
    d = {}
    try:
        expand_rows([{"user_id": sorted(held)[0], "available_reviews": 70000, "train_range": [0, 70000],
                      "validation_range": [0, 70000]}], SPEC, DATA, held, never)
        d["held_out_row_refused"] = False
    except ValueError:
        d["held_out_row_refused"] = True
    d["selection_disjoint_from_every_evaluation_set"] = not (chosen & never)
    d["training_rows_in_464_732_validation_final_test"] = len({r["user_id"] for r in rows} & held)
    d["training_rows_in_2225_pool_trained_whole_as_in_control"] = len({r["user_id"] for r in rows} & (never - held))
    d["expansion_seconds"] = expansion_seconds
    real = [decode(r["user_id"])[0] for r in arm_rows]
    order = [u for i, u in enumerate(real) if i == 0 or u != real[i - 1]]
    d["order_kept"] = order == [r["user_id"] for r in rows]
    d["reviews_kept"] = sum(r["available_reviews"] for r in arm_rows) == sum(r["available_reviews"] for r in rows)
    d["real_users_disjoint_from_held_out"] = not (set(real) & held)
    d["pseudo_users_from_evaluation_users"] = len({r["augment"]["source_user"] for r in arm_rows if "augment" in r} & never)
    d["pseudo_ids_above_real"] = all(r["user_id"] >= PSEUDO_BASE for r in arm_rows if "augment" in r) and max(
        int(u) for u in counts) < PSEUDO_BASE
    d["held_out_checked"], d["never_split_checked"] = len(held), len(never)
    d["pools"] = "held out (refused): 464, 732, split validation + final test; never split: those + the 2,225 pool"
    d["scorer_mentions_augment"] = "augment" in (ROOT / "scripts/small_model_heldout.py").read_text(encoding="utf-8")
    d["expansion"] = aug
    d_ok = (d["held_out_row_refused"] and d["selection_disjoint_from_every_evaluation_set"] and d["order_kept"] and d["reviews_kept"]
            and d["real_users_disjoint_from_held_out"] and d["pseudo_ids_above_real"] and not d["scorer_mentions_augment"]
            and d["pseudo_users_from_evaluation_users"] == 0 and d["training_rows_in_464_732_validation_final_test"] == 0)
    report["checks"]["D_held_out_safety"] = {"pass": d_ok, **d}
    fails += [] if d_ok else ["D"]
    report["fails"] = fails
    report["result"] = "ALL OK" if not fails else "FAILED: " + ", ".join(fails)
    report["seconds"] = round(time.perf_counter() - started, 1)
    atomic_json(args.out, report)
    print(report["result"], f"({report['seconds']} s)")
    return 0 if not fails else 1


def pair(args):
    """The prepared arm vs the control cool-down."""
    arm, ctl = Path(args.arm), Path(args.control)
    da, dc = (json.loads((p / "decay.json").read_text()) for p in (arm, ctl))
    def first_protocol(folder):  # before trims and resumes: untrimmed, else the copy made at the first resume, else current
        for name in ("protocol.untrimmed.json", "protocol.prepared.json", "protocol.json"):
            if (folder / name).exists():
                return json.loads((folder / name).read_text())
    pa, pc = first_protocol(arm), first_protocol(ctl)
    from scripts.augment import decode, expand_rows, held_out_users
    fail = []
    held, never = held_out_users(), held_out_users(include_2225=True)
    oa = dict(da["options"])
    spec = oa.pop("augment", None)
    if spec is None:
        fail.append("the arm has no augment option")
    if oa != dc["options"]:
        fail.append(f"options differ: { {k: (oa.get(k), dc['options'].get(k)) for k in set(oa) | set(dc['options']) if oa.get(k) != dc['options'].get(k)} }")
    for k in ("source_run", "source_epoch", "source_update_offset_end", "model", "track", "fraction_of_epoch_reviews",
              "training_users", "training_reviews", "train_users_from", "shuffle_seed", "lr", "evaluation_set", "dropped_width32_options"):
        if da.get(k) != dc.get(k):
            fail.append(f"{k}: {str(da.get(k))[:120]} vs {str(dc.get(k))[:120]}")
    fa, fc = dict(pa["full_run"]), dict(pc["full_run"])
    for f in (fa, fc):
        f.pop("control_dir", None)
    if fa != fc:
        fail.append(f"full_run differs: { {k: (fa.get(k), fc.get(k)) for k in set(fa) | set(fc) if fa.get(k) != fc.get(k)} }")
    for k in set(pa) | set(pc):
        if k not in ("rows", "full_run") and pa.get(k) != pc.get(k):
            fail.append(f"protocol {k} differs")
    expected, report = expand_rows(pc["rows"], spec, pa["dataset_path"], held, never, workers=4) if spec else ([], {})
    if pa["rows"] != expected:
        fail.append("arm rows are not the expansion of the control's rows")
    real = {decode(r["user_id"])[0] for r in pa["rows"]}
    if real & held:
        fail.append(f"{len(real & held)} held-out users in the arm")
    split = {r["augment"]["source_user"] for r in pa["rows"] if "augment" in r}
    if split & never:
        fail.append(f"{len(split & never)} evaluation-set users split")
    if real != {r["user_id"] for r in pc["rows"]}:
        fail.append("the arm's real users differ from the control's")
    labels = {}
    for name, path in (("arm", args.arm_windows), ("control", args.control_windows)):
        if path and Path(path).exists():
            w = json.loads(Path(path).read_text())
            labels[name] = {"labelled_reviews": w.get("labelled_reviews"), "rating_labelled_reviews": w.get("rating_labelled_reviews"),
                            "reviews_in_train_ranges": w.get("reviews_in_train_ranges"), "ok": w.get("ok")}
    if "arm" in labels and "control" in labels:
        if labels["arm"]["labelled_reviews"] != labels["control"]["labelled_reviews"]:
            fail.append(f"labelled reviews differ: {labels['arm']} vs {labels['control']}")
        if labels["control"].get("rating_labelled_reviews") not in (None, labels["arm"]["rating_labelled_reviews"]):
            fail.append("rating labels differ")
    out = {"arm": str(arm), "control": str(ctl), "rows_arm": len(pa["rows"]), "rows_control": len(pc["rows"]),
           "expansion": report, "labels": labels, "failures": fail, "ok": not fail}
    atomic_json(args.out, out)
    print(("[sm67-pair] OK: " if not fail else "[sm67-pair] REFUSE: ") + (
        f"arm = the control cool-down except augment ({report.get('split_users')} users split into {report.get('pseudo_users')} halves, "
        f"{report.get('split_review_share', 0):.1%} of the reviews; {len(pa['rows']):,} rows), same reviews, start, schedule, order; "
        f"labelled reviews {labels.get('arm', {}).get('labelled_reviews')} = {labels.get('control', {}).get('labelled_reviews')}"
        if not fail else "; ".join(fail)))
    return 0 if not fail else 1


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cpu", "pair"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="cpu: one split user in B")
    parser.add_argument("--arm", type=Path)
    parser.add_argument("--control", type=Path, default=DEC25)
    parser.add_argument("--arm-windows", type=Path)
    parser.add_argument("--control-windows", type=Path)
    args = parser.parse_args()
    return cpu(args) if args.command == "cpu" else pair(args)


if __name__ == "__main__":
    raise SystemExit(main())
