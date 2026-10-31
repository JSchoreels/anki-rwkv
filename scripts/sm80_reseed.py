#!/usr/bin/env python
"""Data-order re-seed (SM80): CPU proof and decision (artifacts/small-model/sm80/HANDOFF.md).

The data augmentation run (SM67) found that every epoch of the leak-free small model (SM72s) sees bit-identical
inputs: `training_seed` is 2025 in every epoch and each window's seed is training_seed + user + offset, so the random
id codes, the day baseline and the dropout masks repeat; only the user order changes (split_seed x 1000 + epoch).
SM80 trains SM72s's epoch 21 from its epoch 20 with another training_seed (`scripts/sm77_warm.py epoch --data-seed`),
then the same quarter-epoch 32k Muon cool-down as the tiny architecture additions (SM77) reference (its own order and
seeds unchanged), and compares with that reference (SM72s's own epoch 21 + the same cool-down, already scored).

  check --arm DIR --off DIR [--users N] --out report.json
        CPU (~1-3 min). (P0) the epoch-21 protocol rebuilt from epoch 20 == SM72s's epoch-21 protocol on disk;
        (P1) option off: DIR's protocol == SM72s's epoch 21 except the start-weight / control paths, its options ==
        SM72s's run options exactly; (P2) option on: the arm's protocol == the option-off protocol (users, user order,
        review ranges, update offset, schedule), its decay.json == the option-off one except training_seed and the
        note / time, same start-weight hashes; (P3) batches of N training users with the real options (2048 x 16
        strided lanes): with the run's seed twice the batches are bit-identical, with the arm's seed the update
        layout (user, offset, reviews, lanes), the labels, the review numbers and the KD replay targets are
        bit-identical and only the seed-dependent encodings differ.
  decide --arm-rows FILE --arm-label L --ref-rows FILE --ref-label L
        the SM77 rule (scripts/sm77_rule.py: 722 users, points of % of the big model (1 leak-free epoch); go curve
        <= -0.20 or instant <= -0.15 with that head's CI clear of zero and the other head <= +0.03; stop if both
        > -0.10; grey otherwise); writes artifacts/small-model/sm80/sm80-decision.{json,md}; exit 2 if users miss.
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GPU_BATCH1_DATA_PATH", r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")

from scripts.sm77_warm import RUN72S, build_protocol  # noqa: E402

HERE = ROOT / "artifacts/small-model/sm80"
PATH_KEYS = ("init_master", "init_optimizer", "control_dir", "benchmark_db")
DECAY_FREE = ("created_utc", "note", "options")
# rwkv.prepare_batch / scripts.gpu_packed_encodings: the seed draws the id-code tables and the day baselines (and,
# through torch.manual_seed(item.seed), the dropout masks); everything else in a batch is a function of the data
SEEDED = {"packed[table0]", "packed[table1]", "packed[table2]", "packed[table3]", "packed[baselines]", "input[start]"}
SEED_FREE = {"start", "sub_gather", "time_shift_selects", "skips", "labels", "label_review_th", "packed[indices0]",
             "packed[indices1]", "packed[indices2]", "packed[indices3]", "packed[days]", "packed[first]",
             "packed[periods]", "packed[frequencies]", "packed[padded]"}


def strip(protocol):
    p = json.loads(json.dumps(protocol))
    for k in PATH_KEYS:
        p["full_run"].pop(k, None)
    return p


def tensors(batch):
    """Every tensor reachable from the batch (attributes, dicts, lists), by path."""
    import torch
    out = {}

    def walk(value, name, depth):
        if isinstance(value, torch.Tensor):
            out[name] = value
        elif depth > 6:
            return
        elif isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{name}[{k}]", depth + 1)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                walk(v, f"{name}[{i}]", depth + 1)
        elif hasattr(value, "__dict__"):
            for k, v in vars(value).items():
                walk(v, f"{name}.{k}" if name else k, depth + 1)
    walk(batch, "", 0)
    return out


def family(name):
    """'time_shift_selects[2][1]' -> 'time_shift_selects', 'packed[table0]' -> 'packed[table0]'."""
    import re
    return re.sub(r"\[\d+\]", "", name)


def batches(rows, options, seed):
    from concurrent.futures import ThreadPoolExecutor
    from scripts.gpu_r01_prefetch import training_batches
    span = options["batch_chunks"] * options["chunk_reviews"]
    with ThreadPoolExecutor(max_workers=1) as pool:
        return [(row, item) for row, item in training_batches(rows, span, seed, pool, options, None, 2)]


def check(args):
    import torch
    from scripts.expertium_options import ReplayTargets
    started = time.perf_counter()
    rep, fails = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}, []
    run = RUN72S
    config = json.loads((run / "run.json").read_text())
    disk = json.loads((run / "epoch-21" / "protocol.json").read_text())
    built, _ = build_protocol(run, 21)
    ok = built == disk
    rep["checks"]["P0_rebuilt_epoch21_equals_disk"] = ok
    fails += [] if ok else ["P0"]
    arm, off = Path(args.arm), Path(args.off)
    pa, po = (json.loads((d / "protocol.json").read_text()) for d in (arm, off))
    da, do = (json.loads((d / "decay.json").read_text()) for d in (arm, off))
    # P1 option off == SM72s's own epoch 21
    p1 = {"protocol_equals_epoch21_except_paths": strip(po) == strip(disk),
          "start_weights_are_epoch20": Path(po["full_run"]["init_master"]).resolve()
          == (run / "epoch-20" / config["model"] / config["track"] / "master.pth").resolve(),
          "options_equal_run_options": do["options"] == config["options"]}
    rep["checks"]["P1_option_off"] = p1
    fails += [f"P1 {k}" for k, v in p1.items() if not v]
    # P2 option on: only training_seed differs
    oa, oo = dict(da["options"]), dict(do["options"])
    seeds = (oa.pop("training_seed", None), oo.pop("training_seed", None))
    decay_diff = sorted(k for k in set(da) | set(do) if k not in DECAY_FREE and da.get(k) != do.get(k))
    p2 = {"protocol_equals_option_off_except_control_dir": strip(pa) == strip(po),
          "options_equal_except_training_seed": oa == oo, "training_seeds": list(seeds),
          "seed_changed": seeds[0] != seeds[1] and seeds[1] == config["options"]["training_seed"],
          "decay_json_equal_except_options_note_time": not decay_diff, "decay_json_differs_in": decay_diff,
          "users": len(pa["rows"]), "reviews": da["training_reviews"], "update_offset": pa["full_run"].get("update_offset")}
    rep["checks"]["P2_option_on"] = p2
    fails += [f"P2 {k}" for k in ("protocol_equals_option_off_except_control_dir", "options_equal_except_training_seed",
                                  "seed_changed", "decay_json_equal_except_options_note_time") if not p2[k]]
    # P3 batches: N training users of the epoch-21 protocol (smallest + mid + one with >= 2 updates), real options
    options = {k: v for k, v in config["options"].items() if k != "packed_ipc"}
    span = options["batch_chunks"] * options["chunk_reviews"]
    sizes = sorted(pa["rows"], key=lambda r: r["train_range"][1] - r["train_range"][0])
    picks = [sizes[0], sizes[len(sizes) // 2]] + [r for r in sizes if span < r["train_range"][1] - r["train_range"][0] <= 3 * span][:1]
    rows = [r for r in pa["rows"] if r["user_id"] in {p["user_id"] for p in picks[:args.users]}]  # epoch order
    replay = ReplayTargets(options["distillation"])
    base, again, new = (batches(rows, options, s) for s in (seeds[1], seeds[1], seeds[0]))

    def view(seq, with_targets):
        out = []
        for row, item in seq:
            b = item.batch.to(torch.device("cpu"))
            entry = {"user": row["user_id"], "offset": item.offset, "reviews": item.reviews, "seed": item.seed,
                     "lanes": list(getattr(item, "carry_lanes", []) or []),
                     # the worker's batch (card features in `start`, deferred id-code tables / day baselines in
                     # `packed`) and the model input built from it on the device (`input[start]`)
                     "tensors": {**tensors(item.batch), "input[start]": b.start}}
            if with_targets:
                entry["kd"] = replay.targets(b, (row["user_id"], item.offset, item.reviews, item.seed), item.batch.labels)
            out.append(entry)
        return out
    vb, vg, vn = view(base, True), view(again, False), view(new, True)
    layout = lambda v: [(e["user"], e["offset"], e["reviews"], e["lanes"]) for e in v]  # noqa: E731
    same = lambda x, y: x.shape == y.shape and x.dtype == y.dtype and torch.equal(x, y)  # noqa: E731
    determinism = all(set(a["tensors"]) == set(b["tensors"]) and all(same(a["tensors"][k], b["tensors"][k]) for k in a["tensors"])
                      for a, b in zip(vb, vg)) and layout(vb) == layout(vg)
    equal_fields, differing = set(), set()
    for a, b in zip(vb, vn):
        for k in sorted(set(a["tensors"]) | set(b["tensors"])):
            (equal_fields if k in a["tensors"] and k in b["tensors"] and same(a["tensors"][k], b["tensors"][k]) else differing).add(family(k))
    must_equal = [k for k in equal_fields | differing if k.startswith("label")]
    kd_equal = all(same(x, y) for a, b in zip(vb, vn) for x, y in zip(a["kd"], b["kd"]))
    seed_shift = {b["seed"] - a["seed"] for a, b in zip(vb, vn)}
    p3 = {"users": [r["user_id"] for r in rows], "updates": len(vb),
          "update_reviews": sum(e["reviews"] for e in vb),
          "same_seed_twice_bit_identical": determinism,
          "layout_equal (user, offset, reviews, lanes)": layout(vb) == layout(vn),
          "label_fields": sorted(must_equal), "label_fields_equal": all(k in equal_fields - differing for k in must_equal) and bool(must_equal),
          "kd_replay_targets_equal": kd_equal, "window_seed_shift": sorted(seed_shift),
          "equal_fields": sorted(equal_fields - differing), "differing_fields": sorted(differing),
          "reseed_changes_inputs": bool(differing),
          "seed_independent_inputs_equal": all(k in equal_fields - differing for k in SEED_FREE if k in equal_fields | differing),
          "only_seeded_fields_differ": differing <= SEEDED}
    rep["checks"]["P3_batches"] = p3
    fails += [f"P3 {k}" for k in ("same_seed_twice_bit_identical", "layout_equal (user, offset, reviews, lanes)",
                                  "label_fields_equal", "kd_replay_targets_equal", "reseed_changes_inputs",
                                  "seed_independent_inputs_equal", "only_seeded_fields_differ") if not p3[k]]
    if seed_shift != {seeds[0] - seeds[1]}:
        fails.append("P3 window seed shift")
    rep.update(fails=fails, verdict="PASS" if not fails else "FAIL", seconds=round(time.perf_counter() - started, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(f"[sm80-check] {rep['verdict']}: P0 {ok}; option off == SM72s epoch 21 {all(p1.values())}; option on: only "
          f"training_seed {seeds[1]} -> {seeds[0]} ({p2['users']:,} users, {p2['reviews']:,} reviews); batches of "
          f"{len(rows)} users / {len(vb)} updates: layout, labels, KD targets equal {p3['layout_equal (user, offset, reviews, lanes)'] and p3['label_fields_equal'] and kd_equal}, "
          f"differing {p3['differing_fields']}" + (f"; FAILS {fails}" if fails else ""))
    return 0 if not fails else 1


def decide(args):
    import scripts.sm77_rule as r
    users732 = set(json.loads(r.U732.read_text())["users"])
    long_users = set(json.loads(r.ULONG.read_text())["users"])
    big = r.load(*r.BIG, users732 | long_users)
    users = sorted(users732 & set(big))
    arm = r.load(Path(args.arm_rows), args.arm_label, set(big)) if Path(args.arm_rows).exists() else {}
    ref = r.load(Path(args.ref_rows), args.ref_label, set(big))
    missing = [u for u in users if u not in arm or u not in ref]
    if missing:
        print(f"[sm80-rule] {len(users) - len(missing)} of {len(users)} users scored in both arms (missing {missing[:10]})")
        return 2
    res = {"rule_pool": r.compare(arm, ref, big, users)}
    both = sorted(set(big) & set(arm) & set(ref))
    if len(both) > len(users):
        res["732+long"] = r.compare(arm, ref, big, both)
    res["verdict"], res["why"] = r.verdict(res["rule_pool"])
    res.update(arm_rows=str(args.arm_rows), arm_label=args.arm_label, ref_rows=str(args.ref_rows), ref_label=args.ref_label,
               big=f"{r.BIG[0]} {r.BIG[1]}",
               rule={"go": r.GO, "other_max": r.OTHER_MAX, "stop": r.STOP, "unit": "points of % of the big model (1 leak-free epoch)"})
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "sm80-decision.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    md = ["# Data-order re-seed (SM80): epoch 21 with a new training_seed vs the same recipe with the usual seed", "",
          "Arm: the leak-free small model (SM72s) epoch 20 + epoch 21 with another data-order seed + the quarter-epoch 32k "
          "Muon cool-down; reference: SM72s's own epoch 21 + the same cool-down (the tiny architecture additions (SM77) "
          "reference). Leak-free scoring, w8+s8, review-weighted LogLoss, user bootstrap 2,000 seed 0, 95% CI; deltas in "
          "points of % of the big model (1 leak-free epoch).", "",
          "| pool | users | head | arm | reference | delta | points [95% CI] | users better | arm vs big | reference vs big |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for pool in ("rule_pool", "732+long"):
        if pool in res:
            for h in r.HEADS.values():
                x = res[pool][h]
                md.append(f"| {'732 (rule)' if pool == 'rule_pool' else pool} | {res[pool]['users']} | {h} | {x['arm']:.5f} | "
                          f"{x['reference']:.5f} | {x['delta']:+.6f} | {x['points']:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}] | "
                          f"{x['users_better']} | {x['arm_vs_big_pct']:+.2f}% | {x['reference_vs_big_pct']:+.2f}% |")
    md += ["", "Same architecture and size (w64); only epoch 21's per-window seeds differ.", "",
           f"Verdict: **{res['verdict']}** ({res['why']})."]
    (HERE / "sm80-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    p = res["rule_pool"]
    print(f"[sm80-rule] {p['users']} users: curve {p['curve']['points']:+.2f} points {p['curve']['ci']} instant "
          f"{p['instant']['points']:+.2f} points {p['instant']['ci']} (arm - reference): {res['verdict'].upper()} ({res['why']})")
    return 0


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check")
    p.add_argument("--arm", required=True)
    p.add_argument("--off", required=True)
    p.add_argument("--users", type=int, default=3)
    p.add_argument("--out", default=str(HERE / "check-cpu-sm80.json"))
    p = sub.add_parser("decide")
    p.add_argument("--arm-rows", required=True)
    p.add_argument("--arm-label", required=True)
    p.add_argument("--ref-rows", required=True)
    p.add_argument("--ref-label", required=True)
    args = parser.parse_args()
    return check(args) if args.command == "check" else decide(args)


if __name__ == "__main__":
    raise SystemExit(main())
