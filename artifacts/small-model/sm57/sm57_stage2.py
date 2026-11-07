#!/usr/bin/env python
"""The instant-only small model (SM57) stage 2: from scratch at small scale, instant-only vs joint (DESIGN.md, "Stage 2
plan"). CPU only, except the folders it prepares are trained by scripts/expertium_decay.py run on the GPU.

Recipe: the factory's 2k tier (accuracy-loop tier "screen": 2,000 users nested in the 8k final tier, 146,654,211
reviews per epoch) with the leak-free small model (SM72s)'s own from-scratch recipe: its run.json options unchanged
(w64, 2,048-review windows x 16 strided state-carry lanes, leak-free replay KD alpha 0.9 from the big model (1 leak-free
epoch), AdamW 7e-4 constant after a 5-update warm-up, training_seed 2025 = the fresh init and the per-window seeds),
its epoch-1 protocol fields, the run's epoch order rule (random.Random(split_seed * 1000 + N) over the tier's users),
RWKV_DELETED_CARDS=neighbour. Epoch 1 starts from the trainer's fresh init (no init_master); epoch N > 1 from the arm's
own epoch N-1 (master + AdamW state + update offset). Arms: "joint" (no extra option) and "inst" (+ {"curve_loss_weight":
0.0}). Folders artifacts/small-model/sm57s2-{inst,joint}-eNN.

  prepare --arm inst|joint --epoch N        write the folder (protocol.json + decay.json); refuses if it exists
  pair --epoch N                            the two arms' epoch-N folders differ only by the flag (exit 1 otherwise)
  init-hash                                 the fresh init's tensor hash for both arms' options (and SM72s epoch 1's)
  block [--quiet FILE] [--now ISO]          "NOW <s>" inside a break, else "NEXT <min>" (quiet.json all/gpu + default
                                            nightly 20:00-01:00 unless quiet.json has a window overlapping that night)
  plan                                      after epoch 1 (both arms trained and scored): the final epoch F from the
                                            measured times (7 h cap), written once to gpu/plan-stage2.json
  rule [--rows FILE]                        -> sm57-stage2-decision.{json,md}; exit 2 while the final epoch rows miss
  rule-self-test                            the stage-2 comparison on the stage-1 rows reproduces sm57-decision.json
"""
from argparse import ArgumentParser
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from datetime import datetime, timedelta

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
SM = ROOT / "artifacts/small-model"
G = HERE / "gpu"
TRACK = "E89-F65e-F77a"
RUN72S = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run")
TRIMS72S = RUN72S.parent / "label-window-trims.json"
TIERS = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/accuracy/tiers.json")
QUIET = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json")
LABEL_DB = (RUN72S / "label-db").as_posix()   # as stage 1 (the trainer reads benchmark info read-only; evaluate false)
FLAG = {"curve_loss_weight": 0.0}
ARMS = {"inst": FLAG, "joint": {}}
MAX_EPOCHS = 5
CAP_MINUTES = 420           # ~7 h of GPU tickets in all
ROWS = G / "sm57s2-rows.jsonl"
PLAN = G / "plan-stage2.json"
DAILY = (20, 1)             # default nightly break 20:00 -> 01:00


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def folder(arm, n):
    return SM / f"sm57s2-{arm}-e{n:02d}"


def label(arm, n):
    return f"sm57s2-{arm}-e{n:02d}/w8+s8"


def run_config():
    return json.loads((RUN72S / "run.json").read_text())


def screen_users():
    tiers = json.loads(TIERS.read_text())
    users = list(tiers["tiers"]["screen"]["train_users"])
    if not set(users) <= set(run_config()["train_users"]):
        raise SystemExit("the screen tier is not inside the leak-free small model (SM72s) run's train users")
    return users


def order(n, config=None):
    config = config or run_config()
    users = screen_users()
    random.Random(config["split_seed"] * 1000 + n).shuffle(users)
    return users


def result_of(d):
    p = d / "expertium" / TRACK / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def complete(d):
    r = result_of(d)
    return bool(r) and r.get("status") == "complete" and not r.get("smoke_only") and (d / "expertium" / TRACK / "master.pth").exists()


# ------------------------------------------------------------------ prepare
def build(arm, n):
    from scripts.run_full_training import rows_for
    config = run_config()
    template = json.loads((RUN72S / "epoch-01" / "protocol.json").read_text())
    protocol = copy.deepcopy(template)
    rows = rows_for(order(n, config), config["review_counts"])
    trims = {t["user_id"]: t for t in json.loads(TRIMS72S.read_text())["trims"]}
    applied = []
    for row in rows:
        t = trims.get(row["user_id"])
        if t:
            row["train_range"] = [0, t["to"]]
            applied.append(t)
    protocol["rows"] = rows
    protocol["label_window_trims"] = applied
    out = folder(arm, n)
    full = protocol["full_run"]
    full.update(epoch=n, benchmark_db=LABEL_DB, control_dir=(out / "control").relative_to(ROOT).as_posix())
    for key in ("init_master", "init_optimizer", "update_offset"):
        full.pop(key, None)
    protocol["accuracy_loop"] = dict(template["accuracy_loop"], tier="screen",
                                     experiment=f"sm57 stage 2 ({arm}): the 2k tier from scratch, SM72s recipe")
    options = dict(config["options"])
    extra = ARMS[arm]
    clash = sorted(set(extra) & set(options))
    if clash:
        raise SystemExit(f"the flag would override the run's options {clash}")
    options.update(extra)
    decay = {"kind": "sm57 stage 2 constant-LR epoch (the 2k tier from scratch)", "source_run": RUN72S.as_posix(),
             "tier": "screen", "epoch": n, "arm": arm, "model": config["model"], "track": config["track"],
             "options": options, "architecture": "w64", "training_users": len(rows),
             "training_reviews": sum(r["train_range"][1] - r["train_range"][0] for r in rows),
             "lr": "7e-4 constant after the 5-update warm-up (no lr_schedule)",
             "note": f"SM57 stage 2 epoch {n}, arm {arm}, extra options {json.dumps(extra, sort_keys=True)}"}
    if n == 1:
        decay["start"] = "fresh init (CatchUp.fresh_checkpoint(training_seed))"
    else:
        prev = folder(arm, n - 1)
        if not complete(prev):
            raise SystemExit(f"{prev.name} is not a complete epoch")
        case = prev / "expertium" / TRACK
        full.update(init_master=(case / "master.pth").as_posix(), init_optimizer=(case / "optimizer.pth").as_posix(),
                    update_offset=result_of(prev)["update_offset_end"])
        decay.update(start=prev.name, init_master_sha256=sha256(case / "master.pth"),
                     init_optimizer_sha256=sha256(case / "optimizer.pth"))
    return protocol, decay


def prepare(args):
    out = folder(args.arm, args.epoch)
    if out.exists():
        raise SystemExit(f"{out} exists")
    protocol, decay = build(args.arm, args.epoch)
    decay["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.mkdir(parents=True)
    atomic_json(out / "protocol.json", protocol)
    atomic_json(out / "decay.json", decay)
    print(json.dumps({k: decay[k] for k in ("epoch", "arm", "training_users", "training_reviews", "start", "note")}))
    return 0


# ------------------------------------------------------------------ pair
def leaves(x, y, path=""):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            yield from leaves(x.get(k), y.get(k), f"{path}/{k}")
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (u, v) in enumerate(zip(x, y)):
            yield from leaves(u, v, f"{path}[{i}]")
    elif x != y:
        yield path, x, y


def proto(d):
    q = d / "protocol.prepared.json"   # a resumed run rewrites protocol.json to the users left
    return json.loads((q if q.exists() else d / "protocol.json").read_text())


def pair_fail(a, b, n):
    """a = the instant-only folder, b = the joint folder of epoch n."""
    config = run_config()
    pa, pb = proto(a), proto(b)
    da, db = (json.loads((d / "decay.json").read_text()) for d in (a, b))
    fail = []
    swap = lambda s: s.replace("sm57s2-inst-", "sm57s2-joint-").replace("(inst)", "(joint)") if isinstance(s, str) else s  # noqa: E731
    for path, x, y in leaves(pa, pb):
        if swap(x) != y:
            fail.append(f"protocol {path}: {str(x)[:100]} vs {str(y)[:100]}")
    want = order(n, config)
    for tag, p in (("inst", pa), ("joint", pb)):
        if [r["user_id"] for r in p["rows"]] != want:
            fail.append(f"{tag} rows are not the epoch-{n} order rule over the 2k tier")
        bad = [r["user_id"] for r in p["rows"] if r["train_range"] != [0, config["review_counts"][str(r["user_id"])]]
               and r["user_id"] not in {t["user_id"] for t in p.get("label_window_trims", [])}]
        if bad:
            fail.append(f"{tag} rows with a partial train_range {bad[:5]}")
        if p["full_run"].get("epoch") != n:
            fail.append(f"{tag} epoch {p['full_run'].get('epoch')}")
    if db["options"] != config["options"]:
        fail.append("joint options are not the leak-free small model (SM72s) run's options")
    for key, value in FLAG.items():
        if key in db["options"] or da["options"].get(key) != value:
            fail.append(f"{key} is not {value} in the instant-only arm only")
    if {k: v for k, v in da["options"].items() if k not in FLAG} != db["options"]:
        fail.append("the arms' options differ beyond the flag")
    for path, x, y in leaves(da, db):
        if path in ("/created_utc", "/init_master_sha256", "/init_optimizer_sha256") or path.startswith("/options/"):
            continue
        if path == "/arm" or swap(x) == y:
            continue
        if path == "/note" and isinstance(x, str) and isinstance(y, str) and \
                x.replace(json.dumps(FLAG, sort_keys=True), "{}").replace("arm inst", "arm joint") == y:
            continue
        fail.append(f"decay {path}: {str(x)[:100]} vs {str(y)[:100]}")
    for tag, d, p in (("inst", a, pa), ("joint", b, pb)):
        full = p["full_run"]
        if n == 1:
            if any(k in full for k in ("init_master", "init_optimizer", "update_offset")):
                fail.append(f"{tag} epoch 1 is not from the fresh init")
        else:
            prev = folder(tag, n - 1)
            case = (prev / "expertium" / TRACK).as_posix()
            if full.get("init_master") != f"{case}/master.pth" or full.get("init_optimizer") != f"{case}/optimizer.pth":
                fail.append(f"{tag} epoch {n} does not start from its own epoch {n - 1}")
            r = result_of(prev) or {}
            if full.get("update_offset") != r.get("update_offset_end"):
                fail.append(f"{tag} update_offset {full.get('update_offset')} vs epoch {n - 1} end {r.get('update_offset_end')}")
    if n > 1 and (result_of(folder("inst", n - 1)) or {}).get("updates") != (result_of(folder("joint", n - 1)) or {}).get("updates"):
        fail.append(f"epoch {n - 1} update counts differ between the arms")
    return fail, f"{len(pa['rows']):,} users, {da['training_reviews']:,} reviews, epoch {n}"


def pair(args):
    a, b = folder("inst", args.epoch), folder("joint", args.epoch)
    if args.arm_dir:
        a = Path(args.arm_dir)
    if args.ref_dir:
        b = Path(args.ref_dir)
    fail, info = pair_fail(a, b, args.epoch)
    if fail:
        print(f"[sm57s2-pair] epoch {args.epoch} REFUSE: " + "; ".join(fail[:12]))
        return 1
    print(f"[sm57s2-pair] epoch {args.epoch} OK: {a.name} == {b.name} except {json.dumps(FLAG)} and folder names ({info})")
    return 0


# ------------------------------------------------------------------ init hash
def fresh_hash(options):
    import torch
    from scripts.expertium_options import CatchUp
    from scripts.run_gpu_batch1 import tensor_hash
    torch.set_num_threads(2)
    ck = CatchUp(options, {}, [{"user_id": 0, "train_range": [0, 1]}]).fresh_checkpoint(options.get("training_seed", 2025))
    return tensor_hash(ck["model"]), ck


def init_hash(_args):
    config = run_config()
    out = {arm: fresh_hash({**config["options"], **extra})[0] for arm, extra in ARMS.items()}
    out["sm72s_epoch1_initial_parameter_sha256"] = json.loads(
        (RUN72S / "epoch-01" / "expertium" / TRACK / "result.json").read_text())["initial_parameter_sha256"]
    print(json.dumps(out))
    return 0 if len(set(out.values())) == 1 else 1


# ------------------------------------------------------------------ breaks
def windows(quiet_path, now):
    """Break windows: quiet.json (scope all/gpu) plus, for each night D, the default D 20:00 -> D+1 01:00 unless a
    quiet.json all/gpu window overlaps that night (the user moved the night's break: quiet.json's window replaces it)."""
    text = Path(quiet_path).read_text(encoding="utf-8")
    try:
        wins = [(w["start"], w["end"]) for w in json.loads(text) if w.get("scope", "all") in ("all", "gpu")]
    except (ValueError, KeyError, TypeError):   # malformed JSON: every start/end pair, scope assumed "all"
        wins = re.findall(r'"start"\s*:\s*"([^"]+)"[^{}]*?"end"\s*:\s*"([^"]+)"', text, re.S)
    quiet = [(datetime.fromisoformat(s), datetime.fromisoformat(e)) for s, e in wins]
    blocks = list(quiet)
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for d in range(-1, 14):
        s, e = day + timedelta(days=d, hours=DAILY[0]), day + timedelta(days=d + 1, hours=DAILY[1])
        if not any(qs < e and qe > s for qs, qe in quiet):
            blocks.append((s, e))
    blocks.sort()
    merged = []
    for s, e in blocks:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def block_line(quiet_path, now):
    try:
        merged = windows(quiet_path, now)
    except OSError as error:
        return f"ERR quiet.json unreadable: {error}"
    for s, e in merged:
        if s <= now < e:
            return f"NOW {int((e - now).total_seconds()) + 60}"
        if s > now:
            return f"NEXT {int((s - now).total_seconds() // 60)}"
    return "NEXT 100000"


def block(args):
    now = datetime.fromisoformat(args.now) if args.now else datetime.now()
    print(block_line(args.quiet, now))
    return 0


# ------------------------------------------------------------------ plan
def scored_epochs(final):
    return sorted({e for e in (1, 3, final) if e <= final})


def plan(_args):
    """Final epoch F: the largest of 5..1 whose projected GPU total stays within CAP_MINUTES, from the measured epoch-1
    segments (both arms) and the epoch-1 scoring tickets. Written once; later calls print the stored plan."""
    if PLAN.exists():
        print(PLAN.read_text().strip())
        return 0
    segs = [json.loads(x) for x in (G / "segments-stage2.jsonl").read_text().splitlines() if x.strip()]
    tickets = [json.loads(x) for x in (G / "tickets-stage2.jsonl").read_text().splitlines() if x.strip()]
    e1 = [s for s in segs if Path(s["folder"]).name in (folder("inst", 1).name, folder("joint", 1).name)]
    sc1 = [t for t in tickets if t["kind"] == "score" and t["epoch"] == 1]
    if not e1 or not sc1:
        raise SystemExit("plan: epoch-1 segments or scoring tickets missing")
    pair_min = sum(s["minutes"] for s in e1)                  # one epoch, both arms
    score_min = sum(t["minutes"] for t in sc1) / 2            # one checkpoint
    spent = sum(s["minutes"] for s in segs) + sum(t["minutes"] for t in tickets)
    final, projection = 1, {}
    for f in range(MAX_EPOCHS, 0, -1):
        left = (f - 1) * pair_min + 2 * score_min * len([e for e in scored_epochs(f) if e > 1])
        projection[f] = round(spent + left, 1)
        if spent + left <= CAP_MINUTES and final == 1:
            final = f
    out = {"final_epoch": final, "scored_epochs": scored_epochs(final), "epoch_pair_minutes": round(pair_min, 1),
           "scoring_minutes_per_checkpoint": round(score_min, 1), "spent_minutes": round(spent, 1),
           "projected_total_minutes": projection, "cap_minutes": CAP_MINUTES,
           "rule": "largest F <= 5 with spent + (F-1) x epoch pair + 2 x scoring x (scored epochs after 1) <= cap",
           "created": datetime.now().isoformat(timespec="seconds")}
    atomic_json(PLAN, out)
    print(json.dumps(out))
    return 0


# ------------------------------------------------------------------ rule
def rule_module():
    spec = importlib.util.spec_from_file_location("sm92_rule", ROOT / "artifacts/small-model/sm92/sm92_rule.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage1_verdict(x):
    """sm57_rule.verdict, the same rule: CI entirely below zero and at least 0.05 points."""
    if x["points_ci"][1] >= 0:
        return "stop", "CI not below zero"
    if x["points"] > -0.05:
        return "stop", "CI below zero but the gain is below the 0.05-point minimum"
    return "go", "CI below zero and at least the 0.05-point minimum"


def compare_epoch(R, rows, n, users, big, labels=None):
    labels = labels or {"inst": label("inst", n), "joint": label("joint", n)}
    s = {k: R.load(rows, v, set(big)) for k, v in labels.items()}
    if any(u not in s[k] for k in s for u in users):
        return None
    rel = R.load(*R.REL_NORMAL, set(big))
    return {"inst_minus_joint": R.compare(s["inst"], s["joint"], big, users),
            "inst_minus_big": R.compare(s["inst"], big, big, users),
            "joint_minus_big": R.compare(s["joint"], big, big, users),
            "joint_minus_release": R.compare(s["joint"], rel, big, users),
            "inst_minus_release": R.compare(s["inst"], rel, big, users)}


def training_rows():
    out = {}
    segs = [json.loads(x) for x in (G / "segments-stage2.jsonl").read_text().splitlines() if x.strip()] \
        if (G / "segments-stage2.jsonl").exists() else []
    for n in range(1, MAX_EPOCHS + 1):
        for arm in ARMS:
            d = folder(arm, n)
            r = result_of(d)
            if not r:
                continue
            mine = [s for s in segs if Path(s["folder"]).name == d.name]
            out[f"{arm} e{n:02d}"] = {"status": r.get("status"), "segments": len(mine),
                                      "wall_minutes": round(sum(s["minutes"] for s in mine), 1),
                                      "training_minutes": round(r.get("training_seconds", 0) / 60, 1),
                                      "reviews": r.get("review_presentations"), "reviews_per_second": round(r.get("reviews_per_second", 0)),
                                      "peak_allocated_mib": round(r.get("peak_allocated_mib") or 0), "peak_reserved_mib": round(r.get("peak_reserved_mib") or 0),
                                      "nvidia_smi_max_used_mib": max((s.get("nvidia_smi_max_used_mib") or 0 for s in mine), default=None),
                                      "nvidia_smi_baseline_mib": min((s.get("nvidia_smi_baseline_mib") or 10**9 for s in mine), default=None),
                                      "initial_parameter_sha256": r.get("initial_parameter_sha256"),
                                      "curve_loss_weight": r.get("curve_loss_weight"), "updates": r.get("updates")}
    return out


def rule(args):
    R = rule_module()
    users, big = R.pools()
    rows = Path(args.rows)
    final = json.loads(PLAN.read_text())["final_epoch"] if PLAN.exists() else MAX_EPOCHS
    res = {"pool": {"users": len(users)}, "final_epoch": final, "plan": json.loads(PLAN.read_text()) if PLAN.exists() else None,
           "epochs": {}}
    for n in range(1, MAX_EPOCHS + 1):
        c = compare_epoch(R, rows, n, users, big) if rows.exists() else None
        if c:
            res["epochs"][n] = c
    if final in res["epochs"]:
        res["verdict"], why = stage1_verdict(res["epochs"][final]["inst_minus_joint"]["instant"])
        res["why"] = f"epoch {final}: instant {R.fmt(res['epochs'][final]['inst_minus_joint']['instant'])}: {why}"
    else:
        res["verdict"], res["why"] = "pending", f"epoch {final} rows missing"
    if 1 in res["epochs"]:
        x = res["epochs"][1]["inst_minus_joint"]["instant"]
        res["epoch1_reading"] = ("instant-only leads at epoch 1" if x["points_ci"][1] < 0 else
                                 "joint leads at epoch 1 (the curve task helps instant early)" if x["points_ci"][0] > 0 else
                                 "no difference at epoch 1 (CI includes zero)")
    init = {k: v["initial_parameter_sha256"] for k, v in training_rows().items() if k.endswith("e01")}
    res["same_init"] = {"hashes": init, "equal": len(set(init.values())) == 1 if init else None}
    res["training"] = training_rows()
    res["rule"] = {"go_if": f"instant-only - joint at the final epoch ({final}), instant head: 95% CI upper < 0 and points <= -0.05",
                   "unit": "points = % of the big model (1 leak-free epoch) normal-input LogLoss, review-weighted, 722 users of the 732 set, paired user bootstrap 2,000 seed 0"}
    atomic_json(HERE / "sm57-stage2-decision.json", res)
    md = ["# The instant-only small model (SM57) stage 2: from scratch on the 2k tier, instant-only vs joint", "",
          "Both arms from the same fresh init (training_seed 2025), the leak-free small model (SM72s) recipe on the 2k tier "
          "(2,000 users, 2,048-review windows x 16 lanes, leak-free replay KD, AdamW 7e-4 constant), same order and seeds; "
          "the instant-only arm trains with every curve loss term at weight 0. Constant-LR checkpoints, no cool-down. Leak-free "
          "scoring, normal inputs, w8+s8, 722 users of the 732 set; points = % of the big model (1 leak-free epoch) LogLoss, "
          "paired user bootstrap 2,000, seed 0, 95% CI.", "",
          "| epoch | instant-only - joint, instant | users better / worse | instant-only - big, instant | joint - big, instant | joint - big, curve | joint - the release (SM63), instant / curve |",
          "|---|---|---|---|---|---|---|"]
    for n, c in sorted(res["epochs"].items()):
        x = c["inst_minus_joint"]["instant"]
        md.append(f"| {n} | {R.fmt(x)} | {x['users_better']} / {x['users_worse']} | {R.fmt(c['inst_minus_big']['instant'])} | "
                  f"{R.fmt(c['joint_minus_big']['instant'])} | {R.fmt(c['joint_minus_big']['curve'])} | "
                  f"{R.fmt(c['joint_minus_release']['instant'])} / {R.fmt(c['joint_minus_release']['curve'])} |")
    md += ["", "Training per arm and epoch:", "",
           "| arm / epoch | status | segments | wall min | training min | reviews/s | peak allocated / reserved MiB | nvidia-smi max / baseline MiB |",
           "|---|---|---|---|---|---|---|---|"]
    for k, t in res["training"].items():
        md.append(f"| {k} | {t['status']} | {t['segments']} | {t['wall_minutes']} | {t['training_minutes']} | {t['reviews_per_second']:,} | "
                  f"{t['peak_allocated_mib']:,} / {t['peak_reserved_mib']:,} | {t['nvidia_smi_max_used_mib']} / {t['nvidia_smi_baseline_mib']} |")
    md += ["", f"Same fresh init in both arms (initial_parameter_sha256): {res['same_init']['equal']}.",
           f"Epoch-1 reading: {res.get('epoch1_reading', 'pending')}.", "",
           f"Rule: {res['rule']['go_if']}. Verdict: **{res['verdict']}** ({res['why']})."]
    (HERE / "sm57-stage2-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    parts = [f"[sm57s2-rule] {len(users)} users: {res['verdict'].upper()} ({res['why']})"]
    for n, c in sorted(res["epochs"].items()):
        parts.append(f"e{n} inst-joint instant {R.fmt(c['inst_minus_joint']['instant'])}, joint curve vs big {R.fmt(c['joint_minus_big']['curve'])}")
    print("; ".join(parts))
    return 0 if res["verdict"] != "pending" else 2


def rule_self_test(_args):
    """compare_epoch on the stage-1 rows (labels sm57-inst/joint-dec1ep32k) reproduces sm57-decision.json's rule pair and
    verdict; synthetic: identical arms -> stop."""
    R = rule_module()
    users, big = R.pools()
    stage1 = json.loads((HERE / "sm57-decision.json").read_text())["rule_pool"]["inst_minus_joint"]
    c = compare_epoch(R, G / "sm57-rows.jsonl", 0, users, big,
                      {"inst": "sm57-inst-dec1ep32k/w8+s8", "joint": "sm57-joint-dec1ep32k/w8+s8"})
    fails = []
    for h in ("instant", "curve"):
        for k in ("points", "points_ci", "users_better", "users_worse"):
            if c["inst_minus_joint"][h][k] != stage1[h][k]:
                fails.append(f"{h} {k}: {c['inst_minus_joint'][h][k]} vs stage 1 {stage1[h][k]}")
    if stage1_verdict(c["inst_minus_joint"]["instant"])[0] != "go":
        fails.append("stage-1 rows are not go")
    same = compare_epoch(R, G / "sm57-rows.jsonl", 0, users, big,
                         {"inst": "sm57-joint-dec1ep32k/w8+s8", "joint": "sm57-joint-dec1ep32k/w8+s8"})
    if stage1_verdict(same["inst_minus_joint"]["instant"])[0] != "stop":
        fails.append("identical arms are not stop")
    print(json.dumps({"users": len(users), "stage1_reproduced": [c["inst_minus_joint"]["instant"]["points"],
                      c["inst_minus_joint"]["instant"]["points_ci"]], "ok": not fails, "failures": fails}))
    return 0 if not fails else 1


def main():
    p = ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--arm", choices=tuple(ARMS), required=True)
    a.add_argument("--epoch", type=int, required=True)
    a = sub.add_parser("pair")
    a.add_argument("--epoch", type=int, required=True)
    a.add_argument("--arm-dir", default="")
    a.add_argument("--ref-dir", default="")
    sub.add_parser("init-hash")
    a = sub.add_parser("block")
    a.add_argument("--quiet", default=str(QUIET))
    a.add_argument("--now", default="")
    sub.add_parser("plan")
    a = sub.add_parser("rule")
    a.add_argument("--rows", default=str(ROWS))
    sub.add_parser("rule-self-test")
    args = p.parse_args()
    return {"prepare": prepare, "pair": pair, "init-hash": init_hash, "block": block, "plan": plan, "rule": rule,
            "rule-self-test": rule_self_test}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
