#!/usr/bin/env python
"""The show-time instant-only big model (SM99): CPU proofs before any GPU ticket (DESIGN.md in this folder; ~15-30 min).

  python -X utf8 artifacts/small-model/sm99/sm99_check.py [--out check-cpu-sm99.json] [--only K3,K4]

The run: the show-time big model (SM91 step 2) + two big-model epochs on show-time inputs with "curve_loss_weight"
0.0 (sm99_epoch.py), each the big model's own recipe (fresh AdamW, warmup 200, 3.5e-4 -> 0 one_minus_sin).

  K1 code: scripts/ expertium/ rwkv/ trees at HEAD == c99330a7 (the trees the show-time instant-only small model
     (SM96), the show-time instant-only model, stage 2 (SM98) and the smaller instant-only model (SM101) trained and
     scored with) and nothing uncommitted there; the files changed since the show-time big model (SM91 step 2)'s launch
     commit 11d6c1a4 are listed (K3 / K5 prove they do not change its path).
  K2 epoch 1's prepared folder == the show-time big model (SM91 step 2)'s epoch folder except curve_loss_weight 0.0, the
     start (its master.pth, sha256 pinned), the shuffle seed (2: the row order only; same users, same trimmed ranges,
     same 27 label-window trims), track and paths (sm99_epoch.py pair); 11 mutated copies refused (sm99_epoch.py
     negatives).
  K3 trainer slice on CPU (sm99_probe.py = sm57_probe.py: the trainer's batch path with the epoch's options and state
     carry, CatchUp.loss, clip, the trainer's AdamW, TorchScript like the trainer), architecture "fork" from the show-time
     big model (SM91 step 2)'s weights, users 8632 and 5625 of the epoch at 1,024-review windows x 2 strided lanes (the
     window size is a layout parameter; the CPU process stays far below 6 GB):
       K3a the tree of 11d6c1a4 (git archive; the show-time big model (SM91 step 2) trained from it), options without
           the weight == this tree, options without the weight: per update the loss bits, the instant average bits,
           the instant predictions, every gradient, the zero-gradient set, and the final weights, bit for bit;
       K3b this tree with weight 1.0 == this tree without the weight, bit for bit;
       K3c this tree with weight 0.0 (the epoch's options): trains (weights differ, finite), every update's loss == its
           instant average bit for bit, the first update's instant predictions == K3b's (same forward), and only
           curve-only parameters (head_w., head_ahead_logits., ahead_linear., w_linear., gru_) get newly zero gradients.
  K4 show-time samples with the epoch's options (scripts/sm91.py t1_t2, users 8632, 5984, 7885): the flag moves only
     the instant query rows' interval columns 2-7 and the curve query times (never longer); the trainer rows equal the
     scorer's e2s-query rows bitwise.
  K5 scorer: this tree's small_model_heldout.py and the tree of 11d6c1a4 (git archive) give identical rows on CPU for
     the show-time big model (SM91 step 2) (users 4676, 2388 of the 732 set, w8+s8, --split-existing --deleted-cards
     neighbour, --inputs e2s-query and normal), so its committed rows (11d6c1a4) are comparable with this tree's rows.
  K6 rule: sm99_rule.py selftest.
  K7 breaks: sm99_block.py on fixed times (the daily 20:00-01:00, Sunday from 16:00, a quiet.json window replacing a
     night's default).
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
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import sm99_epoch  # noqa: E402

PY = sys.executable
CODE = "c99330a7"
REF_COMMIT = "11d6c1a4"
DATA = r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"
ENV = dict(os.environ, RWKV_DELETED_CARDS="neighbour", CUDA_VISIBLE_DEVICES="", GPU_BATCH1_DATA_PATH=DATA,
           OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
CURVE_HEAD = ("head_w.", "head_ahead_logits.", "ahead_linear.", "w_linear.", "gru_")
USERS_K3, USERS_K4, USERS_K5 = (8632, 5625), (8632, 5984, 7885), (4676, 2388)


class Report:
    def __init__(self):
        self.checks, self.ok = {}, True

    def check(self, name, ok, **detail):
        self.checks[name] = {"ok": bool(ok), **detail}
        self.ok &= bool(ok)
        print(f"[sm99-check] {name}: {'OK' if ok else 'FAIL'} {json.dumps(detail, default=str)[:400]}", flush=True)


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def archive(rev, dest, paths):
    if (dest / ".complete").exists():
        return dest
    data = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", rev, *paths], capture_output=True, check=True).stdout
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest)
    (dest / ".complete").write_text(rev + "\n")
    return dest


def k1(rep):
    trees = {p: (git("rev-parse", f"HEAD:{p}"), git("rev-parse", f"{CODE}:{p}")) for p in ("scripts", "expertium", "rwkv")}
    dirty = git("status", "--porcelain", "--", "scripts", "expertium", "rwkv")
    changed = git("diff", "--name-only", REF_COMMIT, "HEAD", "--", "scripts", "expertium", "rwkv").splitlines()
    rep.check("K1_code_trees_equal_c99330a7", all(a == b for a, b in trees.values()) and not dirty,
              trees={k: v[0][:12] for k, v in trees.items()}, uncommitted=dirty.splitlines()[:10],
              changed_since_sm91s2_launch_11d6c1a4=changed)


def k2(rep):
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm99_epoch.py"), "pair", "--epoch", "1"], capture_output=True, text=True)
    pair = json.loads((HERE / "gpu" / "pair-e1.json").read_text()) if (HERE / "gpu" / "pair-e1.json").exists() else {}
    ok_neg, cases = sm99_epoch.negatives()
    d = json.loads((sm99_epoch.folder(1) / "decay.json").read_text())
    rep.check("K2_epoch1_equals_sm91s2_epoch_except_weight_start_seed_paths", r.returncode == 0 and pair.get("ok") is True and ok_neg,
              pair=pair, negatives_refused=cases, out=r.stdout.strip()[-400:],
              training_reviews_after_trims=d.get("training_reviews_after_trims"), label_windows=d.get("label_windows"))


def probe(tree, spec, tmp):
    f = Path(tmp) / f"spec-{hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]}.json"
    f.write_text(json.dumps(spec), encoding="utf-8")
    env = dict(ENV, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    env.pop("RWKV_NO_JIT", None)
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm99_probe.py"), str(tree), str(f)], cwd=tree, capture_output=True,
                       text=True, env=env)
    line = next((x for x in r.stdout.splitlines() if x.startswith("PROBE ")), None)
    if line is None:
        raise RuntimeError(f"probe failed in {tree}: {r.stderr[-3000:]}")
    return json.loads(line[6:])


def k3(rep, work, ref_tree, tmp):
    d = json.loads((sm99_epoch.folder(1) / "decay.json").read_text())
    rows_all = {r["user_id"]: r for r in json.loads((sm99_epoch.folder(1) / "protocol.json").read_text())["rows"]}
    rows = [rows_all[u] for u in USERS_K3]
    epoch = {k: v for k, v in d["options"].items() if k != "packed_ipc"}
    epoch.update(chunk_reviews=1024)
    off = {k: v for k, v in epoch.items() if k != "curve_loss_weight"}
    spec = {"arch": "fork", "checkpoint": str(sm99_epoch.REF_CK), "rows": rows, "span": 2 * 1024, "lr": 3.5e-4}
    t0 = time.perf_counter()
    base = probe(ref_tree, {**spec, "options": off}, tmp)
    ours = probe(ROOT, {**spec, "options": off}, tmp)
    one = probe(ROOT, {**spec, "options": {**off, "curve_loss_weight": 1.0}}, tmp)
    zero = probe(ROOT, {**spec, "options": epoch}, tmp)
    keys = ("user", "offset", "loss_hex", "imm_hex", "p_digest", "grad_digest", "zero_grad")
    view = lambda p: [{k: u[k] for k in keys} for u in p["updates"]]  # noqa: E731
    n = len(ours["updates"])
    rep.check("K3a_sm91s2_tree_equals_this_tree_option_absent", n >= 2 * len(rows) and view(base) == view(ours)
              and base["weights"] == ours["weights"] and ours["jit"], updates=n, optimizer=ours["optimizer"], scripted=ours["jit"],
              reference_tree=f"{ref_tree} (git archive {REF_COMMIT})", weights=ours["weights"][:16],
              losses=[float.fromhex(u["loss_hex"]) for u in ours["updates"]], windows="1,024 x 2 strided lanes, carried state")
    rep.check("K3b_weight1_equals_option_absent", view(one) == view(ours) and one["weights"] == ours["weights"])
    newz = set().union(*(set(u["zero_grad"]) - set(v["zero_grad"]) for u, v in zip(zero["updates"], ours["updates"])))
    rep.check("K3c_weight0_trains_instant_only", len(zero["updates"]) == n and zero["weights"] != ours["weights"]
              and zero["weights_finite"] and all(u["finite"] and u["loss_hex"] == u["imm_hex"] for u in zero["updates"])
              and zero["updates"][0]["p_digest"] == ours["updates"][0]["p_digest"]
              and len(newz) > 0 and all(x.startswith(CURVE_HEAD) for x in newz),
              new_zero_grad_params=sorted(newz), losses=[float.fromhex(u["loss_hex"]) for u in zero["updates"]],
              instant_averages=[float.fromhex(u["imm_hex"]) for u in zero["updates"]], seconds=round(time.perf_counter() - t0))


def k4(rep):
    from scripts import sm91
    r = sm91.Report()
    d = json.loads((sm99_epoch.folder(1) / "decay.json").read_text())
    rows_all = {x["user_id"]: x for x in json.loads((sm99_epoch.folder(1) / "protocol.json").read_text())["rows"]}
    options = {k: v for k, v in d["options"].items() if k != "show_time_intervals"}   # t1_t2 sets the flag itself
    t0 = time.perf_counter()
    sm91.t1_t2(r, [rows_all[u] for u in USERS_K4], options)
    for name, c in r.checks.items():
        rep.check(f"K4_{name}", c["ok"], **{k: v for k, v in c.items() if k != "ok"})
    rep.checks[f"K4_{name}"]["seconds"] = round(time.perf_counter() - t0)


def k5(rep, work, ref_tree):
    users = work / "users-k5.json"
    users.write_text(json.dumps({"users": list(USERS_K5)}))
    volatile = {"label", "tag", "checkpoint", "prepare_seconds", "model_seconds", "provenance_commit", "provenance_dirty"}
    env = {k: v for k, v in os.environ.items() if k not in ("RWKV_DELETED_CARDS", "EXPERTIUM_END_TO_START")}
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    results, ok = {}, True
    for inputs in ("e2s-query", "normal"):
        rows = {}
        for name, tree in (("sm91s2", ref_tree), ("sm99", ROOT)):
            out = work / f"k5-{name}-{inputs}.jsonl"
            out.unlink(missing_ok=True)
            cmd = [PY, "-B", "-X", "utf8", str(tree / "scripts/small_model_heldout.py"), "--pool", "732", "--users-file", str(users),
                   "--device", "cpu", "--model", f"sm91s2-big-showtime=fork::{sm99_epoch.REF_CK}", "--only", "sm91s2-big-showtime/w8+s8",
                   "--inputs", inputs, "--split-existing", "--deleted-cards", "neighbour", "--out", str(out)]
            r = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True)
            got = [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
            rows[name] = {x["user_id"]: {k: v for k, v in x.items() if k not in volatile} for x in got if "benchmark" in x}
            if r.returncode or len(rows[name]) != len(USERS_K5):
                ok = False
                results[f"{name}/{inputs}"] = {"exit": r.returncode, "rows": len(rows[name]), "tail": (r.stdout + r.stderr)[-1500:]}
        same = rows["sm91s2"] == rows["sm99"] and len(rows["sm99"]) == len(USERS_K5)
        ok &= same
        results[inputs] = {"identical": same, "users": sorted(rows["sm99"]),
                           "instant_log_loss": {u: x.get("instant_log_loss") for u, x in rows["sm99"].items()}}
    rep.check("K5_scorer_rows_equal_sm91s2_scorer_commit", ok, reference_tree=f"{ref_tree} (git archive {REF_COMMIT})", **results)


def k6(rep):
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm99_rule.py"), "selftest"], capture_output=True, text=True)
    rep.check("K6_rule_selftest", r.returncode == 0, out=r.stdout.strip()[-1500:], err=r.stderr[-800:])


def k7(rep, tmp):
    def block(now, quiet=None):
        args = [PY, "-X", "utf8", str(HERE / "sm99_block.py"), "--now", now] + (["--quiet", str(quiet)] if quiet else [])
        return subprocess.run(args, capture_output=True, text=True).stdout.strip()
    q = Path(tmp) / "quiet-test.json"
    q.write_text(json.dumps([{"start": "2026-11-12T21:00", "end": "2026-11-13T01:00", "scope": "all"}]), encoding="utf-8")
    empty = Path(tmp) / "quiet-empty.json"
    empty.write_text("[]", encoding="utf-8")
    got = {"wed_19:00": block("2026-11-11T19:00", empty), "wed_21:00": block("2026-11-11T21:00", empty),
           "thu_00:59": block("2026-11-12T00:59", empty), "thu_01:00": block("2026-11-12T01:00", empty),
           "sun_15:00": block("2026-11-15T15:00", empty), "sun_17:00": block("2026-11-15T17:00", empty),
           "thu_20:30_quiet_21:00": block("2026-11-12T20:30", q), "thu_19:00_quiet_21:00": block("2026-11-12T19:00", q),
           "real_now": block(time.strftime("%Y-%m-%dT%H:%M"))}
    want = {"wed_19:00": "NEXT 60", "wed_21:00": "NOW 14460", "thu_00:59": "NOW 120", "thu_01:00": "NEXT 1140",
            "sun_15:00": "NEXT 60", "sun_17:00": "NOW 28860", "thu_20:30_quiet_21:00": "NEXT 30", "thu_19:00_quiet_21:00": "NEXT 120"}
    rep.check("K7_breaks", all(got[k] == v for k, v in want.items()), got=got, want=want)


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "check-cpu-sm99.json"))
    ap.add_argument("--work", default=str(HERE / "work"))
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    want = lambda c: only is None or c in only  # noqa: E731
    import torch
    torch.set_num_threads(2)
    rep = Report()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ref_tree = archive(REF_COMMIT, work / f"ref-{REF_COMMIT}",
                       ("scripts", "expertium", "rwkv", "artifacts/accuracy/tiers.json", "artifacts/small-model/eval-users-732.json"))
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        if want("K1"):
            k1(rep)
        if want("K2"):
            k2(rep)
        if want("K3"):
            k3(rep, work, ref_tree, tmp)
        if want("K4"):
            k4(rep)
        if want("K5"):
            k5(rep, work, ref_tree)
        if want("K6"):
            k6(rep)
        if want("K7"):
            k7(rep, tmp)
    from scripts.experiment_provenance import process_provenance
    out = {"all_ok": rep.ok and only is None, "ok": rep.ok, "only": sorted(only) if only else None, "checks": rep.checks,
           "seconds": round(time.perf_counter() - started), "provenance": process_provenance()}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"[sm99-check] {'ALL OK' if out['all_ok'] else ('OK (partial)' if rep.ok else 'FAIL')}: "
          + ", ".join(f"{k} {'OK' if v['ok'] else 'FAIL'}" for k, v in rep.checks.items()), flush=True)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
