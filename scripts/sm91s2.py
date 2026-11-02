#!/usr/bin/env python
"""Show-time interval retrain (SM91) step 2: CPU proofs before the big-model epoch (artifacts/small-model/sm91s2/DESIGN.md).

The epoch: the big model (fork RWKV, 2,762,884 parameters) gets one added epoch from the big model (1 leak-free epoch)
with the leak-free retrain (SM72) recipe (scripts/sm72_big_epoch.py: users 5000-10000, shuffle seed 1, fresh AdamW,
warmup 200, 3.5e-4 -> 0 one_minus_sin, 2 strided lanes x 8,192 carried state, RWKV_DELETED_CARDS=neighbour), the only
change being the track option "show_time_intervals": true (scripts/gpu_r01_samples.py, scripts/gpu_r01_prefetch.py:
the duration-leak scoring (SM31) `e2s-query` inputs, as the show-time interval retrain (SM91) step 1 small model).

  cpu-check [--out artifacts/small-model/sm91s2/check-cpu-sm91s2.json]   (all on CPU, ~10-20 min, run by the launcher)
    T1 trainer samples, flag on vs off, with the big model's options (users 8632, 5984, 7885 of its training users,
       2,641 / 12,595 / 20,883 reviews): only interval columns 2-7 of instant query rows and the curve query time of
       update rows move, never longer (scripts/sm91.py t1_t2, unchanged).
    T2 trainer rows (flag on) == the duration-leak scoring (SM31) scorer's e2s-query rows bitwise; flag off == its
       normal rows; on - off == e2s-query - normal (same users, scripts/sm91.py t1_t2).
    T3 flag off == the current big-model trainer: full training batches (gpu_r01_prefetch.training_batches, the
       big model (1 leak-free epoch) recipe's options from wt/sm72 decay.json minus packed_ipc, span 2 x 8,192) of this
       tree and of the tree the better teacher (SM65) big-model epoch ran on (git archive 45a56e3a) are bitwise equal in
       every tensor (T3a); T3m the model side: the fork architecture with the big model (1 leak-free epoch) weights,
       every update in order with the carried state, train mode, the trainer's seeds: batch digest, loss and every
       gradient sha256 equal between the two trees (users 8632, 5625 at 1,024-review windows x 2 lanes, 2 updates each,
       so a CPU process stays far below 6 GB; the window size is a layout parameter, the code path is the epoch's);
       T3b / T3mb flag on: which batch tensors change and that the loss changes (informational).
    T4 config: this folder's decay.json / protocol.json == the big model (1 leak-free epoch) recipe
       (wt/sm72/artifacts/small-model/sm72-big-leakfree-1ep decay.json, protocol.prepared.json, protocol.untrimmed.json)
       except the flag, the start weights (sha256 = the big model (1 leak-free epoch) master.pth that the better teacher
       (SM65) load check passed), the track / paths and snapshot_users (400 -> 50); the label-window trims equal; the
       SM72 folder against itself is refused (no flag); nothing trained yet; outputs are new paths.
    T5 rule: artifacts/small-model/sm91s2/sm91s2_rule.py self-test (SM31's big-model show-time cost +0.205 / +0.508
       reproduced; a pair with itself 0 / stop; a synthetic model 0.2 points better on instant only: go).
    T6 scorer: the reused SM31 rows came from commit fa5a9f25; the scorer of this tree and of a git archive of
       fa5a9f25 give identical rows on CPU (users 4676, 2388 of the 732 set, the big model (1 leak-free epoch),
       w8+s8, --split-existing --deleted-cards neighbour, --inputs e2s-query and normal).
"""
from argparse import ArgumentParser
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import sm91  # noqa: E402  (Report, t1_t2, probe: the SM91 step 1 proofs, reused unchanged)

PY = sys.executable
S = ROOT / "artifacts/small-model/sm91s2"
OUT = ROOT / "artifacts/small-model/sm91s2-big-showtime"
TRACK = "big-showtime"
SM72OUT = Path("C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72-big-leakfree-1ep")  # read only
BIG1 = SM72OUT / "expertium/big-leakfree-1ep/master.pth"                                        # read only
SM65LOAD = Path("C:/Users/jonat/workspace/wt/sm65/artifacts/small-model/sm65/check-init-load-sm65.json")  # read only
REF_TRAINER = "45a56e3a"   # better teacher (SM65) launch commit: the last big-model epoch ran from this tree
REF_SCORER = "fa5a9f25"    # duration-leak scoring (SM31) rows' provenance commit (big-leakfree-1ep/w8+s8@e2s-query)
FLAG = "show_time_intervals"
USERS_T12 = (8632, 5984, 7885)   # big-model training users: 1 / 1 (2 lanes) / 2 updates at 2 x 8,192
USERS_T3M = (8632, 5625)         # 2,641 / 2,794 reviews: 3 windows of 1,024 -> 2 updates each at 2 lanes
USERS_T6 = (4676, 2388)          # 732-set users, 3,716 / 3,685 reviews
SNAPSHOT_USERS = 50

PROBE_MODEL = r'''
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
spec = json.loads(open(sys.argv[2]).read())
options, rows, span = spec["options"], spec["rows"], spec["span"]

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

model = build(options["architecture"]).float()
model.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu", weights_only=True), strict=True)
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
    out["updates"].append([row["user_id"], item.offset, item.reviews, item.seed, digest_batch(batch), h.hexdigest(),
                           float(stats.average_loss)])
print("PROBE " + json.dumps(out))
'''


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def archive(rev, dest, paths=("scripts", "expertium", "rwkv")):
    """git archive REV PATHS -> dest (once); the reference trees for T3 / T6."""
    if (dest / ".complete").exists():
        return dest
    data = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", rev, *paths], capture_output=True, check=True).stdout
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest)
    (dest / ".complete").write_text(rev + "\n")
    return dest


def big_rows(users):
    """Rows of the prepared (trimmed) protocol for these users (the epoch's own train ranges)."""
    by_user = {r["user_id"]: r for r in json.loads((OUT / "protocol.json").read_text())["rows"]}
    return [by_user[u] for u in users]


def sm72_options():
    return dict(json.loads((SM72OUT / "decay.json").read_text())["options"])


def probe_model(tree, rows, options, work, tag):
    spec = work / f"probe-model-{tag}.json"
    spec.write_text(json.dumps({"rows": rows, "span": options["batch_chunks"] * options["chunk_reviews"],
                                "options": options, "checkpoint": str(BIG1)}))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", RWKV_DELETED_CARDS="neighbour", PYTHONPATH="",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    r = subprocess.run([PY, "-B", "-X", "utf8", "-c", PROBE_MODEL, str(tree), str(spec)], cwd=tree,
                       capture_output=True, text=True, env=env)
    line = next((x for x in r.stdout.splitlines() if x.startswith("PROBE ")), None)
    if r.returncode or line is None:
        raise RuntimeError(f"model probe {tag} failed ({r.returncode}): {r.stderr[-3000:]}")
    return json.loads(line[6:])


def t3(rep, work, ref_tree):
    off = {k: v for k, v in sm72_options().items() if k != "packed_ipc"}
    on = dict(off, **{FLAG: True})
    rows = big_rows(USERS_T12)
    a = sm91.probe(ref_tree, rows, off, work, "sm65-off")
    b = sm91.probe(ROOT, rows, off, work, "sm91s2-off")
    c = sm91.probe(ROOT, rows, on, work, "sm91s2-on")
    same = len(a) == len(b) and all(x["fields"] == y["fields"] and x["offset"] == y["offset"] and x["reviews"] == y["reviews"]
                                    for x, y in zip(a, b))
    diff = sorted({k for x, y in zip(a, b) for k in set(x["fields"]) | set(y["fields"]) if x["fields"].get(k) != y["fields"].get(k)})
    rep.check("T3a_flag_off_batches_equal_current_big_model_trainer", same and len(a) >= len(rows), batches=len(a),
              updates=[(x["user"], x["offset"], x["reviews"]) for x in a], tensors_per_batch=[len(x["fields"]) for x in a],
              differing=diff[:20], reference_tree=f"{ref_tree} (git archive {REF_TRAINER})", span=2 * 8192)
    moved = sorted({k for x, y in zip(c, b) for k in set(x["fields"]) | set(y["fields"]) if x["fields"].get(k) != y["fields"].get(k)})
    rep.check("T3b_flag_on_batch_tensors_that_change", len(c) == len(b) and bool(moved)
              and not any(k.startswith(("batch.sub_gather", "batch.time_shift", "batch.skips", "carry")) for k in moved),
              changed=moved, note="the packed feature / label tensors only; permutations, skips and carry layout unchanged")
    # model side, small windows (memory): every update with the carry, loss + gradients
    small = dict(off, chunk_reviews=1024)
    mrows = big_rows(USERS_T3M)
    ma = probe_model(ref_tree, mrows, small, work, "sm65-off")
    mb = probe_model(ROOT, mrows, small, work, "sm91s2-off")
    mc = probe_model(ROOT, mrows, dict(small, **{FLAG: True}), work, "sm91s2-on")
    ua, ub, uc = ma["updates"], mb["updates"], mc["updates"]
    rep.check("T3m_flag_off_model_loss_and_gradients_equal_current_big_model_trainer",
              ua == ub and len(ua) >= 2 * len(mrows), updates=len(ua), per_update=[u[:4] + [u[6]] for u in ua],
              reference_tree=f"{ref_tree} (git archive {REF_TRAINER})", architecture=off["architecture"],
              checkpoint=str(BIG1), windows="1,024 x 2 strided lanes (carried state across updates)")
    rep.check("T3mb_flag_on_changes_the_loss", len(uc) == len(ub) and all(x[:4] == y[:4] for x, y in zip(uc, ub))
              and any(x[5] != y[5] for x, y in zip(uc, ub)),
              losses_on=[u[6] for u in uc], losses_off=[u[6] for u in ub], note="informational: same layout, other inputs")


def t4(rep):
    """The prepared folder == the big model (1 leak-free epoch) recipe except the allowed fields."""
    def diff(a_dir, b_dir, b_protocol="protocol.prepared.json"):
        problems = []
        a, b = json.loads((a_dir / "decay.json").read_text()), json.loads((b_dir / "decay.json").read_text())
        allowed = {"created_utc", "source_case", "source_sha256", "track", "extra_options"}
        keys = sorted(k for k in set(a) | set(b) if k != "options" and k not in allowed and a.get(k) != b.get(k))
        if keys:
            problems.append(f"decay.json differs in {keys}")
        oa, ob = a["options"], b["options"]
        okeys = sorted(k for k in set(oa) | set(ob) if oa.get(k) != ob.get(k))
        if okeys != [FLAG] or oa.get(FLAG) is not True or a.get("extra_options") != {FLAG: True}:
            problems.append(f"options differ in {okeys} (want exactly [{FLAG}] = true), extra_options {a.get('extra_options')}")
        pa, pb = json.loads((a_dir / "protocol.json").read_text()), json.loads((b_dir / b_protocol).read_text())
        if pa["rows"] != pb["rows"]:
            problems.append("protocol rows (users, order, trimmed train ranges) differ")
        top = sorted(k for k in set(pa) | set(pb) if k != "full_run" and pa.get(k) != pb.get(k))
        if top:
            problems.append(f"protocol.json differs in {top}")
        fa, fb = pa["full_run"], pb["full_run"]
        fkeys = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
        if not set(fkeys) <= {"init_master", "control_dir", "snapshot_users"}:
            problems.append(f"full_run differs in {fkeys}")
        ua, ub = a_dir / "protocol.untrimmed.json", b_dir / "protocol.untrimmed.json"
        if ua.exists() != ub.exists() or (ua.exists() and json.loads(ua.read_text())["rows"] != json.loads(ub.read_text())["rows"]):
            problems.append("protocol.untrimmed.json rows differ")
        return problems, fkeys, okeys
    problems, fkeys, okeys = diff(OUT, SM72OUT)
    a = json.loads((OUT / "decay.json").read_text())
    p = json.loads((OUT / "protocol.json").read_text())["full_run"]
    big_sha = sha256(BIG1)
    load = json.loads(SM65LOAD.read_text())
    if Path(p["init_master"]).resolve() != BIG1.resolve() or Path(a["source_case"]).resolve() != BIG1.resolve():
        problems.append(f"start weights {p['init_master']} are not {BIG1}")
    if a["source_sha256"] != big_sha or load.get("shipped_sha256") != big_sha or not load.get("pass"):
        problems.append(f"start sha256 {a['source_sha256'][:12]} vs file {big_sha[:12]} vs SM65 load check "
                        f"{str(load.get('shipped_sha256'))[:12]} (pass {load.get('pass')})")
    if p.get("snapshot_users") != SNAPSHOT_USERS or p.get("optimizer_state") != "fresh" or p.get("update_offset") != 0 or "lr_resume" in p:
        problems.append(f"full_run not a fresh start with snapshot_users {SNAPSHOT_USERS}: {p}")
    if a.get("track") != TRACK or a.get("deleted_cards") != "neighbour" or a.get("model") != "expertium":
        problems.append(f"track / deleted_cards / model: {a.get('track')} {a.get('deleted_cards')} {a.get('model')}")
    if "label_windows" not in a:
        problems.append("label-window pre-flight not applied")
    if (OUT / "expertium").exists() or (OUT / "attempts").exists():
        problems.append("the folder has already trained (expertium/ or attempts/ exists): T4 must run before the epoch")
    if OUT.resolve() == SM72OUT.resolve() or (OUT / "expertium" / TRACK / "master.pth").resolve() == BIG1.resolve():
        problems.append("output would overwrite the big model (1 leak-free epoch)")
    rep.check("T4a_epoch_equals_big_model_1_leak_free_epoch_recipe_except_flag_start_paths", not problems,
              problems=problems, full_run_differences=fkeys, option_differences=okeys, start_sha256=big_sha,
              users=len(json.loads((OUT / "protocol.json").read_text())["rows"]),
              training_reviews_after_trims=a.get("training_reviews_after_trims"), label_windows=a.get("label_windows"))
    self_problems, _, _ = diff(SM72OUT, SM72OUT)
    rep.check("T4b_sm72_against_itself_refused_without_flag", any("options differ" in x for x in self_problems),
              problems=self_problems)


def t5(rep):
    r = subprocess.run([PY, "-X", "utf8", str(S / "sm91s2_rule.py"), "self-test"], capture_output=True, text=True)
    rep.check("T5_rule_reproduces_sm31_big_model_cost_and_zero", r.returncode == 0, out=r.stdout.strip()[-900:], err=r.stderr[-800:])


def t6(rep, work, ref_tree):
    users = work / "users-t6.json"
    users.write_text(json.dumps({"users": list(USERS_T6)}))
    volatile = {"label", "tag", "checkpoint", "prepare_seconds", "model_seconds", "provenance_commit", "provenance_dirty"}
    env = {k: v for k, v in os.environ.items() if k not in ("RWKV_DELETED_CARDS", "EXPERTIUM_END_TO_START")}
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    results, ok = {}, True
    for inputs in ("e2s-query", "normal"):
        rows = {}
        for name, tree in (("sm31", ref_tree), ("sm91s2", ROOT)):
            out = work / f"t6-{name}-{inputs}.jsonl"
            out.unlink(missing_ok=True)
            cmd = [PY, "-B", "-X", "utf8", str(tree / "scripts/small_model_heldout.py"), "--pool", "732", "--users-file", str(users),
                   "--device", "cpu", "--model", f"big-leakfree-1ep=fork::{BIG1}", "--only", "big-leakfree-1ep/w8+s8",
                   "--inputs", inputs, "--split-existing", "--deleted-cards", "neighbour", "--out", str(out)]
            r = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True)
            got = [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
            rows[name] = {x["user_id"]: {k: v for k, v in x.items() if k not in volatile} for x in got if "benchmark" in x}
            if r.returncode or len(rows[name]) != len(USERS_T6):
                ok = False
                results[f"{name}/{inputs}"] = {"exit": r.returncode, "rows": len(rows[name]), "tail": (r.stdout + r.stderr)[-1500:]}
        same = rows["sm31"] == rows["sm91s2"] and len(rows["sm31"]) == len(USERS_T6)
        ok &= same
        results[inputs] = {"identical": same, "users": sorted(rows["sm91s2"]),
                           "curve_log_loss": {u: r.get("curve_log_loss") for u, r in rows["sm91s2"].items()}}
    rep.check("T6_scorer_rows_equal_sm31_scorer_commit", ok, reference_tree=f"{ref_tree} (git archive {REF_SCORER})", **results)


def cpu_check(args):
    import torch
    torch.set_num_threads(2)
    rep = sm91.Report()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    trainer_tree = archive(REF_TRAINER, work / f"ref-trainer-{REF_TRAINER}")
    # the scorer reads the pool definition (tiers.json -> the split) from its own tree at start-up
    scorer_tree = archive(REF_SCORER, work / f"ref-scorer-{REF_SCORER}",
                          ("scripts", "expertium", "rwkv", "artifacts/accuracy/tiers.json", "artifacts/small-model/eval-users-732.json"))
    t4(rep)   # first: config before anything else (cheap)
    options = sm72_options()
    if not args.skip_t12:
        sm91.t1_t2(rep, big_rows(USERS_T12), options)
    t3(rep, work, trainer_tree)
    t5(rep)
    t6(rep, work, scorer_tree)
    from scripts.experiment_provenance import process_provenance
    out = {"ok": rep.ok, "checks": rep.checks, "users": {"T1_T2_T3a": list(USERS_T12), "T3m": list(USERS_T3M), "T6": list(USERS_T6)},
           "provenance": process_provenance()}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"[sm91s2-check] {'ALL OK' if rep.ok else 'FAIL'}: " + ", ".join(f"{k} {'OK' if v['ok'] else 'FAIL'}" for k, v in rep.checks.items()))
    return 0 if rep.ok else 1


def main():
    ap = ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("cpu-check")
    p.add_argument("--out", default=str(S / "check-cpu-sm91s2.json"))
    p.add_argument("--work", default=str(S / "work"))
    p.add_argument("--skip-t12", action="store_true")
    args = ap.parse_args()
    return cpu_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
