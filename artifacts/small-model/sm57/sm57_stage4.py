#!/usr/bin/env python
"""The instant-only small model (SM57) stage 4: a better dedicated instant model (DESIGN.md, "Stage 4 plan"). CPU only,
except the folders it prepares are trained by scripts/expertium_decay.py run on the GPU.

Candidate  stage 1's instant-only epoch 26 (the leak-free small model (SM72s) epoch 25 + one instant-only epoch,
           artifacts/small-model/sm57-inst-e26, reused, not retrained) + instant-only epochs 27..30 at the release's
           epoch recipe (SM72s run.json options + {"curve_loss_weight": 0.0}, the run's order rule
           random.Random(split_seed * 1000 + N) over its 8,000 train users, epoch 25's label-window trims, AdamW 7e-4
           constant, each epoch from its own previous epoch: weights, AdamW state, update offset) + the release's
           one-epoch 32k Muon cool-down (sm77_warm.py cooldown --epoch 25: the release's users / order / seeds / KD
           targets / schedule) from epoch 30, instant-only. Folders artifacts/small-model/sm57s4-inst-eNN and
           sm57s4-inst-e30-dec-1ep32k-muon. Label sm57s4-inst-e30-dec1ep32k/w8+s8.
Reference  stage 1's instant-only arm (the ship candidate): label sm57-inst-dec1ep32k/w8+s8 in gpu/sm57-rows.jsonl.

  prepare --epoch N        write sm57s4-inst-eNN (protocol.json + decay.json) from epoch N-1 (26 = stage 1's folder)
  pair --epoch N           the epoch-N folder is the recipe: exit 1 otherwise (see pair_fail)
  block [--quiet F --now ISO]   "NOW <s>" inside a break, else "NEXT <min>": quiet.json all/gpu + the default nightly
                           20:00-01:00 (Sundays from 16:00) unless quiet.json has a window overlapping that night
  rule [--rows F]          -> sm57-stage4-decision.{json,md}; exit 2 while the candidate's rule rows are missing
  pool1500                 (on go) candidate vs the release on the 1,500 validation users -> sm57-stage4-pool1500.{json,md}
  self-test                the rule and pool code reproduce the published stage-1 and 1,500-user pairs
"""
from argparse import ArgumentParser
import copy
import hashlib
import importlib.util
import json
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
S4 = HERE / "stage4"
TRACK = "E89-F65e-F77a"
RUN72S = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run")
QUIET = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json")
FLAG = {"curve_loss_weight": 0.0}
FIRST, LAST = 27, 30
E26 = SM / "sm57-inst-e26"                                  # stage 1's instant-only epoch 26 (reused)
DEC1 = SM / "sm57-inst-e26-dec-1ep32k-muon"                 # stage 1's instant-only cool-down
DEC = SM / f"sm57s4-inst-e{LAST}-dec-1ep32k-muon"
CAND = f"sm57s4-inst-e{LAST}-dec1ep32k/w8+s8"
STAGE1 = "sm57-inst-dec1ep32k/w8+s8"
ROWS = G / "sm57s4-rows.jsonl"
ROWS1500 = G / "sm57s4-rows-1500.jsonl"
STAGE1_ROWS = G / "sm57-rows.jsonl"
SM93_ROWS = S4 / "sm93-rows.jsonl"                          # copy of origin/sm93 3225c739 (the release + stage 1, 768 users)
SM93_USERS = S4 / "users-sm93-score.json"
SM93_DECISION = S4 / "sm93-decision.json"
POOL1500 = SM / "eval-users-1500.json"
SIZE = 0.05
TOL = 1e-5
DAILY = (20, 1)             # default nightly break 20:00 -> 01:00
SUNDAY_START = 16           # Sundays from 16:00 (user)
CONST = {26: "sm57s4-inst-e26/w8+s8", 28: "sm57s4-inst-e28/w8+s8", 30: "sm57s4-inst-e30/w8+s8"}   # constant-LR reads


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def folder(n):
    return E26 if n == 26 else SM / f"sm57s4-inst-e{n:02d}"


def run_config():
    return json.loads((RUN72S / "run.json").read_text())


def order(n, config=None):
    config = config or run_config()
    users = config["train_users"][:]
    random.Random(config["split_seed"] * 1000 + n).shuffle(users)
    return users


def result_of(d):
    p = Path(d) / "expertium" / TRACK / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def complete(d):
    r = result_of(d)
    return bool(r) and r.get("status") == "complete" and not r.get("smoke_only") and (Path(d) / "expertium" / TRACK / "master.pth").exists()


def proto(d):
    q = Path(d) / "protocol.prepared.json"   # a resumed run rewrites protocol.json to the users left
    return json.loads((q if q.exists() else Path(d) / "protocol.json").read_text())


# ------------------------------------------------------------------ prepare
def build(n, prev=None, out=None, config=None):
    """Epoch n's protocol and decay.json from the previous epoch folder `prev` (default: this stage's own epoch n-1,
    26 = stage 1's folder): sm77_warm.build_protocol's rule (the run's order rule, the previous protocol's fields and
    label-window trims, update offset = the previous epoch's end), started from `prev`'s trained weights + AdamW state."""
    from scripts.run_full_training import rows_for
    config = config or run_config()
    prev = Path(prev) if prev else folder(n - 1)
    out = Path(out) if out else folder(n)
    if not complete(prev):
        raise SystemExit(f"{prev.name} is not a complete epoch")
    previous = proto(prev)
    protocol = copy.deepcopy(previous)
    rows = rows_for(order(n, config), config["review_counts"])
    trims = {t["user_id"]: t for t in previous.get("label_window_trims", [])}
    for row in rows:
        t = trims.get(row["user_id"])
        if t:
            row["train_range"] = [0, t["to"]]
    protocol["rows"] = rows
    case = prev / "expertium" / TRACK
    full = protocol["full_run"]
    for key in ("lr_resume", "resumed", "optimizer_state"):
        full.pop(key, None)
    full.update(epoch=n, init_master=(case / "master.pth").as_posix(), init_optimizer=(case / "optimizer.pth").as_posix(),
                update_offset=result_of(prev)["update_offset_end"], control_dir=(out / "control").relative_to(ROOT).as_posix())
    base = json.loads((E26 / "decay.json").read_text())
    options = dict(base["options"])
    decay = {"kind": "sm77 constant-LR epoch", "source_run": base["source_run"], "source_epoch": n - 1, "epoch": n,
             "model": base["model"], "track": base["track"], "options": options, "architecture": "w64",
             "training_users": len(rows), "training_reviews": sum(r["train_range"][1] - r["train_range"][0] for r in rows),
             "lr": base["lr"], "start": prev.name,
             "note": f"SM57 stage 4: SM72s epoch {n} protocol (the run's order rule), start = {prev.name}, extra options "
                     f"{json.dumps(FLAG, sort_keys=True)}",
             "init_master_sha256": sha256(case / "master.pth"), "init_optimizer_sha256": sha256(case / "optimizer.pth")}
    return protocol, decay


def prepare(args):
    if not FIRST <= args.epoch <= LAST:
        raise SystemExit(f"stage 4 epochs are {FIRST}..{LAST}")
    out = folder(args.epoch)
    if out.exists():
        raise SystemExit(f"{out} exists")
    protocol, decay = build(args.epoch)
    decay["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.mkdir(parents=True)
    atomic_json(out / "protocol.json", protocol)
    atomic_json(out / "decay.json", decay)
    print(json.dumps({k: decay[k] for k in ("epoch", "training_users", "training_reviews", "start", "note")}))
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


SKIP_FULL = {"/full_run/epoch", "/full_run/init_master", "/full_run/init_optimizer", "/full_run/update_offset",
             "/full_run/control_dir"}


def pair_fail(d, n, prev=None):
    """d = the epoch-n folder. Its protocol == stage 1's epoch 26 protocol except rows / epoch / start / offset /
    control_dir; rows == the run's epoch-n order rule with epoch 25's trims (as epoch 26); start = its own epoch n-1
    (paths, hashes, update offset); options == stage 1's epoch 26 options == SM72s run.json options + exactly the flag;
    the order differs from every earlier epoch of the run (21..n-1)."""
    config = run_config()
    prev = Path(prev) if prev else folder(n - 1)
    p, p26 = proto(d), proto(E26)
    dd, d26 = (json.loads((x / "decay.json").read_text()) for x in (d, E26))
    fail = []
    for path, x, y in leaves(p, p26):
        if path.startswith("/rows") or path in SKIP_FULL:
            continue
        fail.append(f"protocol {path}: {str(x)[:100]} vs {str(y)[:100]}")
    trims = {t["user_id"]: t["to"] for t in p26.get("label_window_trims", [])}
    want = [{"user_id": u, "train_range": [0, trims.get(u, config["review_counts"][str(u)])],
             "available_reviews": config["review_counts"][str(u)],
             "validation_range": [0, config["review_counts"][str(u)]]} for u in order(n, config)]
    if p["rows"] != want:
        fail.append(f"rows are not the run's epoch-{n} order rule with epoch 25's trims")
    seen = {tuple(order(k, config)) for k in range(21, n)}
    if tuple(order(n, config)) in seen or len(seen) != n - 21:
        fail.append("the epoch order repeats an earlier epoch")
    full = p["full_run"]
    case = (prev / "expertium" / TRACK)
    if full.get("epoch") != n:
        fail.append(f"epoch {full.get('epoch')} is not {n}")
    if full.get("init_master") != (case / "master.pth").as_posix() or full.get("init_optimizer") != (case / "optimizer.pth").as_posix():
        fail.append(f"epoch {n} does not start from {prev.name}")
    r = result_of(prev) or {}
    if r.get("status") != "complete" or full.get("update_offset") != r.get("update_offset_end"):
        fail.append(f"update_offset {full.get('update_offset')} vs {prev.name} end {r.get('update_offset_end')} ({r.get('status')})")
    if dd.get("init_master_sha256") != sha256(case / "master.pth") or dd.get("init_optimizer_sha256") != sha256(case / "optimizer.pth"):
        fail.append("the recorded start hashes are not the previous epoch's files")
    if full.get("control_dir") != (d / "control").relative_to(ROOT).as_posix():
        fail.append(f"control_dir {full.get('control_dir')}")
    if dd["options"] != d26["options"]:
        fail.append("options differ from stage 1's instant-only epoch 26")
    if {k: v for k, v in d26["options"].items() if k not in FLAG} != config["options"] or \
            any(d26["options"].get(k) != v for k, v in FLAG.items()) or any(k in config["options"] for k in FLAG):
        fail.append("stage 1's epoch 26 options are not SM72s run.json + exactly the flag")
    for key in ("model", "track", "architecture", "lr", "kind"):
        if dd.get(key) != d26.get(key):
            fail.append(f"decay {key}: {dd.get(key)} vs {d26.get(key)}")
    if dd.get("training_users") != 8000 or dd.get("training_reviews") != d26.get("training_reviews"):
        fail.append(f"{dd.get('training_users')} users / {dd.get('training_reviews')} reviews vs epoch 26's {d26.get('training_reviews')}")
    return fail, f"{len(p['rows']):,} users, {dd['training_reviews']:,} reviews, epoch {n}, start {prev.name}, offset {full.get('update_offset')}"


def pair(args):
    d = Path(args.dir) if args.dir else folder(args.epoch)
    fail, info = pair_fail(d, args.epoch, args.prev or None)
    if fail:
        print(f"[sm57s4-pair] epoch {args.epoch} REFUSE: " + "; ".join(fail[:12]))
        return 1
    print(f"[sm57s4-pair] epoch {args.epoch} OK: {d.name} = the release's epoch recipe + {json.dumps(FLAG)} ({info})")
    return 0


# ------------------------------------------------------------------ breaks
def windows(quiet_path, now):
    """quiet.json (scope all/gpu) plus, for each night D, the default D 20:00 (Sunday 16:00) -> D+1 01:00 unless a
    quiet.json all/gpu window overlaps it (the user moved that night's break: quiet.json's window replaces it)."""
    text = Path(quiet_path).read_text(encoding="utf-8")
    try:
        wins = [(w["start"], w["end"]) for w in json.loads(text) if w.get("scope", "all") in ("all", "gpu")]
    except (ValueError, KeyError, TypeError):   # malformed JSON: every start/end pair, scope assumed "all"
        wins = re.findall(r'"start"\s*:\s*"([^"]+)"[^{}]*?"end"\s*:\s*"([^"]+)"', text, re.S)
    quiet = [(datetime.fromisoformat(s), datetime.fromisoformat(e)) for s, e in wins]
    blocks = list(quiet)
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for k in range(-1, 14):
        d = day + timedelta(days=k)
        start = SUNDAY_START if d.weekday() == 6 else DAILY[0]
        s, e = d + timedelta(hours=start), d + timedelta(days=1, hours=DAILY[1])
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


# ------------------------------------------------------------------ rule
def rule_module():
    spec = importlib.util.spec_from_file_location("sm92_rule", ROOT / "artifacts/small-model/sm92/sm92_rule.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verdict(x, unit="points"):
    if x["points_ci"][1] >= 0:
        return "stop", f"CI not below zero"
    if x["points"] > -SIZE:
        return "stop", f"CI below zero but the gain is below the {SIZE:.2f}-{unit} minimum"
    return "go", f"CI below zero and at least the {SIZE:.2f}-{unit} minimum"


def training_rows():
    segs = [json.loads(x) for x in (G / "segments-stage4.jsonl").read_text().splitlines() if x.strip()] \
        if (G / "segments-stage4.jsonl").exists() else []
    out = {}
    for name, d in [(f"epoch {n}", folder(n)) for n in range(FIRST, LAST + 1)] + [("cool-down", DEC)]:
        r = result_of(d)
        if not r:
            continue
        mine = [s for s in segs if Path(s["folder"]).name == d.name]
        out[name] = {"status": r.get("status"), "segments": len(mine), "wall_minutes": round(sum(s["minutes"] for s in mine), 1),
                     "training_minutes_last_segment": round(r.get("training_seconds", 0) / 60, 1),
                     "reviews_last_segment": r.get("review_presentations"), "reviews_per_second": round(r.get("reviews_per_second", 0)),
                     "peak_allocated_mib": round(r.get("peak_allocated_mib") or 0), "peak_reserved_mib": round(r.get("peak_reserved_mib") or 0),
                     "nvidia_smi_max_used_mib": max((s.get("nvidia_smi_max_used_mib") or 0 for s in mine), default=None),
                     "nvidia_smi_baseline_mib": min((s.get("nvidia_smi_baseline_mib") or 10**9 for s in mine), default=None),
                     "curve_loss_weight": r.get("curve_loss_weight"), "initial_parameter_sha256": r.get("initial_parameter_sha256"),
                     "trained_parameter_sha256": r.get("trained_parameter_sha256"), "updates": r.get("updates"),
                     "update_offset_end": r.get("update_offset_end")}
    return out


def chain_check():
    """Each epoch starts from the previous one's trained weights (initial == previous trained parameter hash)."""
    out = {}
    for n in range(FIRST, LAST + 1):
        a, b = result_of(folder(n - 1)), result_of(folder(n))
        if a and b and b.get("status") == "complete":
            out[n] = b.get("initial_parameter_sha256") == a.get("trained_parameter_sha256")
    return out


def compare_all(R, rows, users, big, stage1_rows=STAGE1_ROWS):
    have = set(big)
    s = {"cand": R.load(rows, CAND, have) if Path(rows).exists() else {},
         "stage1": R.load(stage1_rows, STAGE1, have), "release": R.load(*R.REL_NORMAL, have), "big": big}
    for n, lb in CONST.items():
        s[f"e{n}"] = R.load(rows, lb, have) if Path(rows).exists() else {}
    full = lambda k: all(u in s[k] for u in users)   # noqa: E731
    out = {}
    for x, y in (("cand", "stage1"), ("cand", "release"), ("stage1", "release"), ("cand", "big"), ("stage1", "big"),
                 ("e28", "e26"), ("e30", "e26"), ("e30", "e28"), ("cand", "e30"), ("stage1", "e26"), ("e30", "stage1")):
        if full(x) and full(y):
            out[f"{x}_minus_{y}"] = R.compare(s[x], s[y], big, users)
    return out, s


NAMES = {"cand_minus_stage1": "candidate - stage 1's instant-only arm (the rule)",
         "cand_minus_release": "candidate - the release (SM63)",
         "stage1_minus_release": "stage 1's instant-only arm - the release (reference)",
         "cand_minus_big": "candidate - the big model (1 leak-free epoch)",
         "stage1_minus_big": "stage 1's instant-only arm - the big model (1 leak-free epoch)",
         "e28_minus_e26": "constant-LR epoch 28 - epoch 26 (instant-only, no cool-down)",
         "e30_minus_e26": "constant-LR epoch 30 - epoch 26 (instant-only, no cool-down)",
         "e30_minus_e28": "constant-LR epoch 30 - epoch 28",
         "cand_minus_e30": "cool-down effect from epoch 30 (candidate - its epoch 30)",
         "stage1_minus_e26": "cool-down effect from epoch 26 (stage 1 - its epoch 26)",
         "e30_minus_stage1": "constant-LR epoch 30 - stage 1's cooled-down arm"}


def rule(args):
    R = rule_module()
    users, big = R.pools()
    rows = Path(args.rows)
    res = {"rule_pool": {"users": len(users)}}
    pairs, s = compare_all(R, rows, users, big)
    res["rule_pool"].update(pairs)
    if "cand_minus_stage1" in pairs:
        v, why = verdict(pairs["cand_minus_stage1"]["instant"])
        res["verdict"], res["why"] = v, f"instant {R.fmt(pairs['cand_minus_stage1']['instant'])}: {why}"
    else:
        res["verdict"], res["why"] = "pending", f"candidate rows: {sum(u in s['cand'] for u in users)} of {len(users)} rule users"
    if "cand_minus_big" in pairs:
        res["smaller_model_bar_instant"] = R.bar(pairs["cand_minus_big"])["instant"]
    both = sorted(set(big) & set(s["cand"]) & set(s["stage1"]) & set(s["release"]))
    if len(both) > len(users):
        res["732+long"] = {"users": len(both), "cand_minus_stage1": R.compare(s["cand"], s["stage1"], big, both),
                           "cand_minus_release": R.compare(s["cand"], s["release"], big, both)}
    res["training"] = training_rows()
    res["chain_initial_equals_previous_trained"] = chain_check()
    res["rule"] = {"go_if": "candidate - stage 1's instant-only arm, instant head, 95% CI upper < 0 and points <= -0.05",
                   "unit": "points = % of the big model (1 leak-free epoch) normal-input LogLoss, review-weighted, 722 users of the 732 set, paired user bootstrap 2,000 seed 0"}
    atomic_json(HERE / "sm57-stage4-decision.json", res)
    md = ["# The instant-only small model (SM57) stage 4: more instant-only epochs before the cool-down", "",
          f"Candidate: stage 1's instant-only epoch 26 + instant-only epochs {FIRST}-{LAST} (the release's epoch recipe) + the "
          "release's one-epoch 32k Muon cool-down, all with every curve loss term at weight 0. Reference: stage 1's instant-only "
          "arm (epoch 26 + the same cool-down; the ship candidate). Leak-free scoring, normal inputs, w8+s8, 722 users of the 732 "
          "set; points = % of the big model (1 leak-free epoch) LogLoss, paired user bootstrap 2,000, seed 0, 95% CI.", "",
          f"Verdict: **{res['verdict']}** ({res['why']}).", "",
          "| comparison | head | model | ref | points [95% CI] | users better / worse |", "|---|---|---|---|---|---|"]
    for key, name in NAMES.items():
        if key not in pairs:
            continue
        heads = ("instant",) if "cand" in key or "stage1" in key or key.startswith("e") else ("instant", "curve")
        for h in heads:
            x = pairs[key][h]
            md.append(f"| {name} | {h} | {x['new']:.5f} | {x['ref']:.5f} | {R.fmt(x)} | {x['users_better']} / {x['users_worse']} |")
    if "732+long" in res:
        p = res["732+long"]
        md += ["", f"732+long ({p['users']} users), instant: candidate - stage 1 {R.fmt(p['cand_minus_stage1']['instant'])}, "
               f"candidate - the release {R.fmt(p['cand_minus_release']['instant'])}."]
    if "smaller_model_bar_instant" in res:
        b = res["smaller_model_bar_instant"]
        md += ["", f"Smaller-model bar (instant <= +{b['limit']}% of the big model): {b['pct']:+.3f}% -> {'passes' if b['passes'] else 'fails'} "
               "(the curve head is untrained, never served)."]
    md += ["", "Training (wall minutes over all segments; reviews/s and memory of the last segment's result.json):", "",
           "| stage | status | segments | wall min | reviews/s | peak allocated / reserved MiB | nvidia-smi max / baseline MiB |",
           "|---|---|---|---|---|---|---|"]
    for k, t in res["training"].items():
        md.append(f"| {k} | {t['status']} | {t['segments']} | {t['wall_minutes']} | {t['reviews_per_second']:,} | "
                  f"{t['peak_allocated_mib']:,} / {t['peak_reserved_mib']:,} | {t['nvidia_smi_max_used_mib']} / {t['nvidia_smi_baseline_mib']} |")
    md += ["", f"Chain (each epoch's initial parameters == the previous epoch's trained parameters): {res['chain_initial_equals_previous_trained']}.",
           "", f"Rule: {res['rule']['go_if']} ({res['rule']['unit']})."]
    (HERE / "sm57-stage4-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    extra = [f"{k} {R.fmt(v['instant'])}" for k, v in pairs.items() if k != "cand_minus_stage1"]
    print(f"[sm57s4-rule] {len(users)} users: {res['verdict'].upper()} ({res['why']})" + ("; " + "; ".join(extra) if extra else ""))
    return 0 if res["verdict"] != "pending" else 2


# ------------------------------------------------------------------ 1,500 validation users
def merged_rows(sources, label, pool):
    from scripts.sm71_report import load as rows_load   # no user filter
    out = {}
    for src in sources:
        if Path(src).exists():
            for u, r in rows_load(src, label).items():
                if u in pool and u not in out:
                    out[u] = r
    return out


def pool_compare(cand_label, cand_sources, check_sources=None):
    """The SM93 comparison: candidate vs the release (and vs stage 1) on the 1,500 validation users, denominator = the
    release's own LogLoss (% of the release's LogLoss), paired user bootstrap 2,000, seed 0."""
    R = rule_module()
    pool = set(json.loads(POOL1500.read_text())["users"])
    spec = json.loads(SM93_USERS.read_text())
    rel = merged_rows([R.REL_NORMAL[0], SM93_ROWS], R.RELEASE_LABEL, pool)
    st1 = merged_rows([STAGE1_ROWS, SM93_ROWS], STAGE1, pool)
    cand = merged_rows(cand_sources, cand_label, pool)
    res = {"pool": "the 1,500 validation users (eval-users-1500.json)", "never_fit_skipped": spec["never_fit"]}
    if check_sources:   # the 6 check users re-scored with the 1,500 ticket == the 732-set rows (same tree, same model)
        a, b = (merged_rows([src], cand_label, pool) for src in check_sources)
        diffs = []
        for u in spec["check_users"]:
            if u in a and u in b:
                d = max(abs(a[u]["all"][h][0] - b[u]["all"][h][0]) for h in R.HEADS)
                diffs.append(float("inf") if any(a[u]["all"][h][1] != b[u]["all"][h][1] for h in R.HEADS) else d)
        res["row_check"] = {"users": len(diffs), "max_abs_logloss_diff": max(diffs) if diffs else None}
    expected = sorted(pool - set(spec["never_fit"]))
    users = sorted(set(cand) & set(rel) & set(st1))
    res["users"], res["expected"] = len(users), len(expected)
    res["missing"] = {"candidate": len([u for u in expected if u not in cand]), "release": len([u for u in expected if u not in rel])}
    if users:
        res["candidate_minus_release"] = R.compare(cand, rel, rel, users)
        res["candidate_minus_stage1"] = R.compare(cand, st1, rel, users)
        res["stage1_minus_release"] = R.compare(st1, rel, rel, users)
        u_old, u_new = [u for u in users if u < 5000], [u for u in users if u >= 5000]
        res["by_part"] = {"ids < 5000 (the 732 set)": R.compare(cand, rel, rel, u_old),
                          "ids >= 5000": R.compare(cand, rel, rel, u_new) if u_new else None}
    return res


def pool1500(_args):
    R = rule_module()
    res = pool_compare(CAND, [ROWS, ROWS1500], [ROWS, ROWS1500])
    rc = res.get("row_check", {})
    if rc.get("users", 0) < len(json.loads(SM93_USERS.read_text())["check_users"]):
        res["verdict"], res["why"] = "pending", "check users not re-scored yet"
    elif rc["max_abs_logloss_diff"] > TOL:
        res["verdict"], res["why"] = "invalid", f"re-scored check users differ by more than {TOL}: {rc}"
    elif res["users"] < res["expected"] - 12:
        res["verdict"], res["why"] = "pending", f"{res['users']} of {res['expected']} users scored"
    else:
        x = res["candidate_minus_release"]["instant"]
        v, why = verdict(x, "%")
        res["verdict"], res["why"] = v, f"instant {R.fmt(x)}: {why}"
    res["unit"] = "% of the release's LogLoss on the same users (1% of it = 1.003 points of the big model (1 leak-free epoch) on the 722 users)"
    res["bar"] = "candidate - the release, instant: 95% CI upper < 0 and gain >= 0.05 (% of the release's LogLoss): the SM93 ship bar"
    atomic_json(HERE / "sm57-stage4-pool1500.json", res)
    md = ["# The instant-only small model (SM57) stage 4: the candidate on the 1,500 validation users", "",
          f"Verdict: **{res['verdict']}** ({res.get('why', '')}). Unit: {res['unit']}. Bar: {res['bar']}.", "",
          f"Row check (6 users of the 732 set re-scored with the 1,500-user ticket): {json.dumps(res.get('row_check'))}", "",
          "| users | comparison | head | model | ref | % [95% CI] | users better / worse |", "|---|---|---|---|---|---|---|"]
    parts = [("all", "candidate - the release", res.get("candidate_minus_release")),
             ("all", "candidate - stage 1's instant-only arm", res.get("candidate_minus_stage1")),
             ("all", "stage 1 - the release (SM93 reference)", res.get("stage1_minus_release"))] + \
        [(k, "candidate - the release", v) for k, v in res.get("by_part", {}).items()]
    for name, what, p in parts:
        if not p:
            continue
        x = p["instant"]
        md.append(f"| {p['users']} ({name}) | {what} | instant | {x['new']:.5f} | {x['ref']:.5f} | {R.fmt(x)} | {x['users_better']} / {x['users_worse']} |")
    (HERE / "sm57-stage4-pool1500.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[sm57s4-pool1500] {res['users']} users: {res['verdict'].upper()} ({res.get('why', '')})")
    return 0 if res["verdict"] in ("go", "stop") else 2


# ------------------------------------------------------------------ self-test
def self_test(_args):
    """(1) compare_all with stage 1's rows as the 'candidate' reproduces sm57-decision.json's instant-only - the release
    pair; identical arms are stop. (2) pool_compare with stage 1's label as the candidate reproduces SM93's 1,500-user
    decision (users, instant pair)."""
    R = rule_module()
    users, big = R.pools()
    fails = []
    stage1 = json.loads((HERE / "sm57-decision.json").read_text())["rule_pool"]["inst_minus_release"]
    have = set(big)
    st1 = R.load(STAGE1_ROWS, STAGE1, have)
    rel = R.load(*R.REL_NORMAL, have)
    got = R.compare(st1, rel, big, users)
    for h in ("instant", "curve"):
        for k in ("points", "points_ci", "users_better", "users_worse"):
            if got[h][k] != stage1[h][k]:
                fails.append(f"stage 1 vs release {h} {k}: {got[h][k]} vs {stage1[h][k]}")
    if verdict(got["instant"])[0] != "go":
        fails.append("stage 1 vs the release is not go")
    if verdict(R.compare(st1, st1, big, users)["instant"])[0] != "stop":
        fails.append("identical arms are not stop")
    sm93 = json.loads(SM93_DECISION.read_text())
    p = pool_compare(STAGE1, [STAGE1_ROWS, SM93_ROWS])
    x, y = p.get("candidate_minus_release", {}).get("instant", {}), sm93["candidate_minus_release"]["instant"]
    for k in ("points", "points_ci", "users_better", "users_worse", "new", "ref"):
        if x.get(k) != y.get(k):
            fails.append(f"1,500 pool {k}: {x.get(k)} vs SM93 {y.get(k)}")
    if p["users"] != sm93["users"]:
        fails.append(f"1,500 pool users {p['users']} vs SM93 {sm93['users']}")
    out = {"ok": not fails, "failures": fails, "stage1_vs_release_722": [got["instant"]["points"], got["instant"]["points_ci"]],
           "pool1500_stage1_vs_release": [p["users"], x.get("points"), x.get("points_ci")]}
    print(json.dumps(out))
    return 0 if not fails else 1


def main():
    p = ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--epoch", type=int, required=True)
    a = sub.add_parser("pair")
    a.add_argument("--epoch", type=int, required=True)
    a.add_argument("--dir", default="")
    a.add_argument("--prev", default="")
    a = sub.add_parser("block")
    a.add_argument("--quiet", default=str(QUIET))
    a.add_argument("--now", default="")
    a = sub.add_parser("rule")
    a.add_argument("--rows", default=str(ROWS))
    sub.add_parser("pool1500")
    sub.add_parser("self-test")
    args = p.parse_args()
    return {"prepare": prepare, "pair": pair, "block": block, "rule": rule, "pool1500": pool1500,
            "self-test": self_test}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
