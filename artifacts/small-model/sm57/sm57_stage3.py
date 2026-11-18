#!/usr/bin/env python
"""The instant-only small model (SM57) stage 3: the mirror arm, curve-only (DESIGN.md, "Stage 3 plan"). CPU only,
except the folders it prepares are trained by scripts/expertium_decay.py run on the GPU.

One new arm, "curve" = the stage-2 joint arm + {"instant_loss_weight": 0.0} (SM26's option: the instant term of the
training loss, which carries the instant KD targets, gets weight 0; the curve terms are unchanged). Everything else is
stage 2's joint arm: the 2k tier, the leak-free small model (SM72s) recipe and options, epoch order, seeds, the same
fresh init for epoch 1, epoch N > 1 from the arm's own epoch N-1. Folders artifacts/small-model/sm57s3-curve-eNN, built
from sm57_stage2.build("joint", N) with only the flag, the folder paths and the stage name changed. The comparison is
stage 2's joint arm (not retrained): its folders sm57s2-joint-eNN and its rows gpu/sm57s2-rows.jsonl.

  prepare --epoch N                         write sm57s3-curve-eNN (protocol.json + decay.json); refuses if it exists
  pair --epoch N [--arm-dir D] [--ref-dir D] curve eN differs from stage 2's joint eN only by the flag (exit 1 otherwise)
  init-hash                                 the fresh init's tensor hash for the curve options == joint's == SM72s e1's
  block [--quiet FILE] [--now ISO]          sm57_stage2.block (quiet.json all/gpu + the nightly 20:00-01:00)
  rule [--rows FILE]                        -> sm57-stage3-decision.{json,md}; exit 2 while the epoch-5 rows miss
  rule-self-test                            the stage-3 comparison on stage 2's rows reproduces sm57-stage2-decision.json
"""
from argparse import ArgumentParser
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("sm57_stage2", HERE / "sm57_stage2.py")
S2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S2)
SM, G, TRACK = S2.SM, S2.G, S2.TRACK
FLAG = {"instant_loss_weight": 0.0}
EPOCHS = 5
SCORED = (1, 3, 5)
ROWS = G / "sm57s3-rows.jsonl"
ROWS2 = S2.ROWS
STAGE2_NAME, STAGE3_NAME = "sm57 stage 2", "sm57 stage 3"


def folder(n):
    return SM / f"sm57s3-curve-e{n:02d}"


def label(n):
    return f"sm57s3-curve-e{n:02d}/w8+s8"


# ------------------------------------------------------------------ prepare
def build(n):
    protocol, decay = S2.build("joint", n)   # stage 2's own construction of the joint epoch-n folder
    out = folder(n)
    full = protocol["full_run"]
    full["control_dir"] = (out / "control").relative_to(ROOT).as_posix()
    loop = protocol["accuracy_loop"]
    loop["experiment"] = loop["experiment"].replace(f"{STAGE2_NAME} (joint)", f"{STAGE3_NAME} (curve)")
    clash = sorted(set(FLAG) & set(decay["options"]))
    if clash:
        raise SystemExit(f"the flag would override the run's options {clash}")
    decay["options"] = {**decay["options"], **FLAG}
    decay.update(kind=decay["kind"].replace(STAGE2_NAME, STAGE3_NAME), arm="curve",
                 note=f"SM57 stage 3 epoch {n}, arm curve, extra options {json.dumps(FLAG, sort_keys=True)}")
    if n > 1:
        prev = folder(n - 1)
        if not S2.complete(prev):
            raise SystemExit(f"{prev.name} is not a complete epoch")
        case = prev / "expertium" / TRACK
        full.update(init_master=(case / "master.pth").as_posix(), init_optimizer=(case / "optimizer.pth").as_posix(),
                    update_offset=S2.result_of(prev)["update_offset_end"])
        decay.update(start=prev.name, init_master_sha256=S2.sha256(case / "master.pth"),
                     init_optimizer_sha256=S2.sha256(case / "optimizer.pth"))
    return protocol, decay


def prepare(args):
    out = folder(args.epoch)
    if out.exists():
        raise SystemExit(f"{out} exists")
    protocol, decay = build(args.epoch)
    decay["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.mkdir(parents=True)
    S2.atomic_json(out / "protocol.json", protocol)
    S2.atomic_json(out / "decay.json", decay)
    print(json.dumps({k: decay[k] for k in ("epoch", "arm", "training_users", "training_reviews", "start", "note")}))
    return 0


# ------------------------------------------------------------------ pair
def swap(s):
    """A curve-arm string as the joint arm's (folder names, stage name)."""
    if not isinstance(s, str):
        return s
    return (s.replace("sm57s3-curve-", "sm57s2-joint-").replace(f"{STAGE3_NAME} (curve)", f"{STAGE2_NAME} (joint)")
            .replace(STAGE3_NAME, STAGE2_NAME))


def pair_fail(a, b, n):
    """a = the curve-only folder, b = stage 2's joint folder of epoch n."""
    config = S2.run_config()
    pa, pb = S2.proto(a), S2.proto(b)
    da, db = (json.loads((d / "decay.json").read_text()) for d in (a, b))
    fail = []
    for path, x, y in S2.leaves(pa, pb):
        if swap(x) != y:
            fail.append(f"protocol {path}: {str(x)[:100]} vs {str(y)[:100]}")
    want = S2.order(n, config)
    for tag, p in (("curve", pa), ("joint", pb)):
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
            fail.append(f"{key} is not {value} in the curve-only arm only")
    if {k: v for k, v in da["options"].items() if k not in FLAG} != db["options"]:
        fail.append("the arms' options differ beyond the flag")
    for path, x, y in S2.leaves(da, db):
        if path in ("/created_utc", "/init_master_sha256", "/init_optimizer_sha256") or path.startswith("/options/"):
            continue
        if path == "/arm" and (x, y) == ("curve", "joint"):
            continue
        if path == "/note" and isinstance(x, str) and isinstance(y, str) and \
                x.replace(json.dumps(FLAG, sort_keys=True), "{}").replace("stage 3", "stage 2").replace("arm curve", "arm joint") == y:
            continue
        if swap(x) != y:
            fail.append(f"decay {path}: {str(x)[:100]} vs {str(y)[:100]}")
    for tag, d, p, dec in (("curve", a, pa, da), ("joint", b, pb, db)):
        full = p["full_run"]
        if n == 1:
            if any(k in full for k in ("init_master", "init_optimizer", "update_offset")) or not str(dec.get("start", "")).startswith("fresh init"):
                fail.append(f"{tag} epoch 1 is not from the fresh init")
        else:
            prev = folder(n - 1) if tag == "curve" else S2.folder("joint", n - 1)
            case = (prev / "expertium" / TRACK).as_posix()
            if full.get("init_master") != f"{case}/master.pth" or full.get("init_optimizer") != f"{case}/optimizer.pth" \
                    or dec.get("start") != prev.name:
                fail.append(f"{tag} epoch {n} does not start from its own epoch {n - 1}")
            r = S2.result_of(prev) or {}
            if full.get("update_offset") != r.get("update_offset_end"):
                fail.append(f"{tag} update_offset {full.get('update_offset')} vs epoch {n - 1} end {r.get('update_offset_end')}")
    if n > 1:
        ua, ub = (S2.result_of(folder(n - 1)) or {}), (S2.result_of(S2.folder("joint", n - 1)) or {})
        if (ua.get("updates"), ua.get("update_offset_end")) != (ub.get("updates"), ub.get("update_offset_end")):
            fail.append(f"epoch {n - 1} update counts / offsets differ between the arms")
    return fail, f"{len(pa['rows']):,} users, {da['training_reviews']:,} reviews, epoch {n}"


def pair(args):
    a, b = folder(args.epoch), S2.folder("joint", args.epoch)
    if args.arm_dir:
        a = Path(args.arm_dir)
    if args.ref_dir:
        b = Path(args.ref_dir)
    fail, info = pair_fail(a, b, args.epoch)
    if fail:
        print(f"[sm57s3-pair] epoch {args.epoch} REFUSE: " + "; ".join(fail[:12]))
        return 1
    print(f"[sm57s3-pair] epoch {args.epoch} OK: {a.name} == {b.name} except {json.dumps(FLAG)}, folder names and the stage name ({info})")
    return 0


# ------------------------------------------------------------------ init hash
def init_hash(_args):
    config = S2.run_config()
    out = {"curve": S2.fresh_hash({**config["options"], **FLAG})[0], "joint": S2.fresh_hash(config["options"])[0],
           "stage2_joint_e01_initial_parameter_sha256": (S2.result_of(S2.folder("joint", 1)) or {}).get("initial_parameter_sha256"),
           "stage2_inst_e01_initial_parameter_sha256": (S2.result_of(S2.folder("inst", 1)) or {}).get("initial_parameter_sha256"),
           "sm72s_epoch1_initial_parameter_sha256": json.loads(
               (S2.RUN72S / "epoch-01" / "expertium" / TRACK / "result.json").read_text())["initial_parameter_sha256"]}
    print(json.dumps(out))
    return 0 if len(set(out.values())) == 1 else 1


def block(args):
    return S2.block(args)


# ------------------------------------------------------------------ rule
def verdict(x):
    """The pre-registered rule: curve-only - joint on the curve head, CI entirely below zero and at least 0.05 points."""
    return S2.stage1_verdict(x)


def compare_epoch(R, users, big, curve, joint, inst):
    """curve, joint, inst = (rows file, label). None if a user is missing in any of the three."""
    s = {k: R.load(src, lb, set(big)) for k, (src, lb) in (("curve", curve), ("joint", joint), ("inst", inst))}
    if any(u not in s[k] for k in s for u in users):
        return None
    return {"curve_minus_joint": R.compare(s["curve"], s["joint"], big, users),
            "inst_minus_joint": R.compare(s["inst"], s["joint"], big, users),
            "curve_minus_big": R.compare(s["curve"], big, big, users),
            "inst_minus_big": R.compare(s["inst"], big, big, users),
            "joint_minus_big": R.compare(s["joint"], big, big, users)}


def training_rows():
    out = {}
    segs = [json.loads(x) for x in (G / "segments-stage3.jsonl").read_text().splitlines() if x.strip()] \
        if (G / "segments-stage3.jsonl").exists() else []
    stage2 = json.loads((HERE / "sm57-stage2-decision.json").read_text())["training"]
    for n in range(1, EPOCHS + 1):
        d = folder(n)
        r = S2.result_of(d)
        if r:
            mine = [s for s in segs if Path(s["folder"]).name == d.name]
            out[f"curve e{n:02d}"] = {
                "status": r.get("status"), "segments": len(mine), "wall_minutes": round(sum(s["minutes"] for s in mine), 1),
                "training_minutes": round(r.get("training_seconds", 0) / 60, 1), "reviews": r.get("review_presentations"),
                "reviews_per_second": round(r.get("reviews_per_second", 0)),
                "peak_allocated_mib": round(r.get("peak_allocated_mib") or 0), "peak_reserved_mib": round(r.get("peak_reserved_mib") or 0),
                "nvidia_smi_max_used_mib": max((s.get("nvidia_smi_max_used_mib") or 0 for s in mine), default=None),
                "nvidia_smi_baseline_mib": min((s.get("nvidia_smi_baseline_mib") or 10**9 for s in mine), default=None),
                "initial_parameter_sha256": r.get("initial_parameter_sha256"),
                "instant_loss_weight": (r.get("options") or {}).get("instant_loss_weight"), "updates": r.get("updates"),
                "update_offset_end": r.get("update_offset_end")}
        if f"joint e{n:02d}" in stage2:
            out[f"joint e{n:02d} (stage 2)"] = stage2[f"joint e{n:02d}"]
    return out


def rule(args):
    R = S2.rule_module()
    users, big = R.pools()
    res = {"pool": {"users": len(users)}, "final_epoch": EPOCHS, "scored_epochs": list(SCORED), "epochs": {}}
    for n in SCORED:
        c = compare_epoch(R, users, big, (Path(args.rows), label(n)), (ROWS2, S2.label("joint", n)), (ROWS2, S2.label("inst", n))) \
            if Path(args.rows).exists() else None
        if c:
            res["epochs"][n] = c
    if EPOCHS in res["epochs"]:
        res["verdict"], why = verdict(res["epochs"][EPOCHS]["curve_minus_joint"]["curve"])
        res["why"] = f"epoch {EPOCHS}: curve {R.fmt(res['epochs'][EPOCHS]['curve_minus_joint']['curve'])}: {why}"
        both = res["epochs"][EPOCHS]
        res["two_models_beat_joint_on_both_heads"] = verdict(both["curve_minus_joint"]["curve"])[0] == "go" and \
            S2.stage1_verdict(both["inst_minus_joint"]["instant"])[0] == "go"
    else:
        res["verdict"], res["why"] = "pending", f"epoch {EPOCHS} rows missing"
    tr = training_rows()
    init = {k: v["initial_parameter_sha256"] for k, v in tr.items() if "e01" in k}
    res["same_init"] = {"hashes": init, "equal": len(set(init.values())) == 1 if len(init) == 2 else None}
    res["training"] = tr
    res["rule"] = {"go_if": f"curve-only - joint at epoch {EPOCHS}, curve head: 95% CI upper < 0 and points <= -0.05",
                   "unit": "points = % of the big model (1 leak-free epoch) normal-input LogLoss, review-weighted, 722 users of "
                           "the 732 set, paired user bootstrap 2,000 seed 0"}
    S2.atomic_json(HERE / "sm57-stage3-decision.json", res)
    md = ["# The instant-only small model (SM57) stage 3: the mirror arm, curve-only vs joint", "",
          "Curve-only = stage 2's joint arm with the instant loss term at weight 0 (`instant_loss_weight` 0.0, SM26's option; "
          "the instant KD targets live in that term), same fresh init, the 2k tier, the leak-free small model (SM72s) recipe, "
          "same users, order and seeds; the joint and instant-only arms are stage 2's (not retrained). Constant-LR checkpoints. "
          "Leak-free scoring, normal inputs, w8+s8, 722 users of the 732 set; points = % of the big model (1 leak-free epoch) "
          "LogLoss, paired user bootstrap 2,000, seed 0, 95% CI.", "",
          "| epoch | curve-only - joint, curve (the rule) | users better / worse | curve-only - joint, instant | curve-only - big, curve | joint - big, curve | curve-only - big, instant |",
          "|---|---|---|---|---|---|---|"]
    for n, c in sorted(res["epochs"].items()):
        x = c["curve_minus_joint"]
        md.append(f"| {n} | {R.fmt(x['curve'])} | {x['curve']['users_better']} / {x['curve']['users_worse']} | {R.fmt(x['instant'])} | "
                  f"{R.fmt(c['curve_minus_big']['curve'])} | {R.fmt(c['joint_minus_big']['curve'])} | {R.fmt(c['curve_minus_big']['instant'])} |")
    md += ["", "Two dedicated models (instant-only for the instant head, curve-only for the curve head) vs the joint model:", "",
           "| epoch | instant: instant-only - joint | curve: curve-only - joint | instant-only - big, instant | curve-only - big, curve | joint - big, instant / curve |",
           "|---|---|---|---|---|---|"]
    for n, c in sorted(res["epochs"].items()):
        md.append(f"| {n} | {R.fmt(c['inst_minus_joint']['instant'])} | {R.fmt(c['curve_minus_joint']['curve'])} | "
                  f"{R.fmt(c['inst_minus_big']['instant'])} | {R.fmt(c['curve_minus_big']['curve'])} | "
                  f"{R.fmt(c['joint_minus_big']['instant'])} / {R.fmt(c['joint_minus_big']['curve'])} |")
    md += ["", "Training per epoch (curve-only, and stage 2's joint arm for reference):", "",
           "| arm / epoch | status | segments | wall min | training min | reviews/s | peak allocated / reserved MiB | nvidia-smi max / baseline MiB |",
           "|---|---|---|---|---|---|---|---|"]
    for k, t in tr.items():
        md.append(f"| {k} | {t['status']} | {t['segments']} | {t['wall_minutes']} | {t['training_minutes']} | {t['reviews_per_second']:,} | "
                  f"{t['peak_allocated_mib']:,} / {t['peak_reserved_mib']:,} | {t['nvidia_smi_max_used_mib']} / {t['nvidia_smi_baseline_mib']} |")
    md += ["", f"Same fresh init as stage 2's joint arm (initial_parameter_sha256): {res['same_init']['equal']}."]
    if "two_models_beat_joint_on_both_heads" in res:
        md.append(f"Two dedicated models beat the joint model on both heads at epoch {EPOCHS} (both rules go): "
                  f"{res['two_models_beat_joint_on_both_heads']}.")
    md += ["", f"Rule: {res['rule']['go_if']}. Verdict: **{res['verdict']}** ({res['why']})."]
    (HERE / "sm57-stage3-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    parts = [f"[sm57s3-rule] {len(users)} users: {res['verdict'].upper()} ({res['why']})"]
    for n, c in sorted(res["epochs"].items()):
        parts.append(f"e{n} curve-joint curve {R.fmt(c['curve_minus_joint']['curve'])}, instant {R.fmt(c['curve_minus_joint']['instant'])}")
    print("; ".join(parts))
    return 0 if res["verdict"] != "pending" else 2


def rule_self_test(_args):
    """compare_epoch with stage 2's arms in the curve and joint slots (curve := instant-only e05) reproduces
    sm57-stage2-decision.json's epoch-5 pair on both heads; identical arms -> stop; a missing label -> None."""
    R = S2.rule_module()
    users, big = R.pools()
    s2 = json.loads((HERE / "sm57-stage2-decision.json").read_text())["epochs"]["5"]["inst_minus_joint"]
    c = compare_epoch(R, users, big, (ROWS2, S2.label("inst", 5)), (ROWS2, S2.label("joint", 5)), (ROWS2, S2.label("inst", 5)))
    fails = []
    for h in ("instant", "curve"):
        for k in ("points", "points_ci", "users_better", "users_worse"):
            for key in ("curve_minus_joint", "inst_minus_joint"):
                if c[key][h][k] != s2[h][k]:
                    fails.append(f"{key} {h} {k}: {c[key][h][k]} vs stage 2 {s2[h][k]}")
    same = compare_epoch(R, users, big, (ROWS2, S2.label("joint", 5)), (ROWS2, S2.label("joint", 5)), (ROWS2, S2.label("inst", 5)))
    if verdict(same["curve_minus_joint"]["curve"])[0] != "stop":
        fails.append("identical arms are not stop")
    if compare_epoch(R, users, big, (ROWS2, "no-such-label/w8+s8"), (ROWS2, S2.label("joint", 5)), (ROWS2, S2.label("inst", 5))) is not None:
        fails.append("a missing label is not None")
    print(json.dumps({"users": len(users), "stage2_e5_reproduced": {h: [c["curve_minus_joint"][h]["points"], c["curve_minus_joint"][h]["points_ci"]]
                                                                   for h in ("instant", "curve")}, "ok": not fails, "failures": fails}))
    return 0 if not fails else 1


def main():
    p = ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--epoch", type=int, required=True)
    a = sub.add_parser("pair")
    a.add_argument("--epoch", type=int, required=True)
    a.add_argument("--arm-dir", default="")
    a.add_argument("--ref-dir", default="")
    sub.add_parser("init-hash")
    a = sub.add_parser("block")
    a.add_argument("--quiet", default=str(S2.QUIET))
    a.add_argument("--now", default="")
    a = sub.add_parser("rule")
    a.add_argument("--rows", default=str(ROWS))
    sub.add_parser("rule-self-test")
    args = p.parse_args()
    return {"prepare": prepare, "pair": pair, "init-hash": init_hash, "block": block, "rule": rule,
            "rule-self-test": rule_self_test}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
