#!/usr/bin/env python
"""Show-time interval retrain (SM91) step 1: CPU proofs before launch (artifacts/small-model/sm91/DESIGN.md).

The arm trains with the track option "show_time_intervals": true (scripts/gpu_r01_samples.py, scripts/gpu_r01_prefetch.py):
the duration-leak scoring (SM31) `e2s-query` inputs, applied as the scorer's transform (scripts/sm31_inputs.py).

  cpu-check --out artifacts/small-model/sm91/check-cpu-sm91.json   (all on CPU, a few minutes)
    T1 trainer samples, flag on vs off (state-carry sample path, the leak-free small model (SM72s) options, users of its
       epoch-21 protocol, deleted cards: neighbour): card features move only in the 6 interval columns (2-7) and only on
       instant query rows; global labels move only in label_elapsed_seconds (the curve query time) and only on update
       rows, never longer; ids, module permutations, review_ths, label_review_ths, day offsets and skips equal.
    T2 trainer vs scorer: on every training row (keyed by review_th, is_query), the flag-on trainer row equals the
       scorer's `e2s-query` row (sm31_inputs.samples) bitwise in every card-feature column and every global label except
       has_label (the train split masks it), wherever the flag-off trainer row equals the scorer's `normal` row; and the
       flag-on minus flag-off change equals the scorer's e2s-query minus normal change on every row and column.
    T3 flag off == the current trainer: full training batches (scripts/gpu_r01_prefetch.training_batches, the real
       options minus packed_ipc, span 16 x 2048) built by this tree with the flag off and by the smaller model, card stream
       1 layer (SM73 stage 2) tree (git archive of its commit) are bitwise equal, every tensor; flag on: which batch
       tensors change (informational; the label and feature tensors only).
    T4 config: the arm epoch folder == SM72s's epoch-21 protocol except the flag and paths (sm91_pair.py epoch); a w64
       cool-down prepared by this tree from SM72s epoch 21 with the flag == the reference cool-down (tiny architecture
       additions (SM77) reference) except the flag and folder names (sm91_pair.py decay); the reference against itself is
       refused (no flag).
    T5 rule: sm91_rule.py self-test (SM31's released small model e2s-query vs normal reproduced; a pair with itself = 0).
"""
from argparse import ArgumentParser
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = os.environ.get("GPU_BATCH1_DATA_PATH", r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
os.environ.setdefault("GPU_BATCH1_DATA_PATH", DATA)
RUN72S = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run")
DREF = Path("C:/Users/jonat/workspace/wt/sm77/artifacts/small-model/sm77-ref-e21-dec-q32k-muon")
FLAG = "show_time_intervals"
LABEL_COLUMNS = ["label_elapsed_seconds", "label_elapsed_days", "label_y", "label_rating", "has_label",
                 "label_is_equalize", "is_query"]
USERS = (2651, 8823, 2158)   # SM72s epoch-21 train users: ~3k, ~15k, ~32k reviews (one strided update each)

PROBE = r'''
import hashlib, json, os, sys
root = sys.argv[1]
sys.path.insert(0, root)
os.chdir(root)
import torch
torch.set_num_threads(2)
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from scripts.gpu_r01_prefetch import training_batches
spec = json.loads(open(sys.argv[2]).read())

def walk(x, path):
    if isinstance(x, torch.Tensor):
        yield path, hashlib.sha256(str((x.dtype, tuple(x.shape))).encode() + x.detach().contiguous().view(torch.uint8).numpy().tobytes() if x.numel() else str((x.dtype, tuple(x.shape))).encode()).hexdigest()
    elif dataclasses.is_dataclass(x):
        for f in dataclasses.fields(x):
            yield from walk(getattr(x, f.name), f"{path}.{f.name}")
    elif isinstance(x, dict):
        for k in sorted(x, key=str):
            yield from walk(x[k], f"{path}[{k}]")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from walk(v, f"{path}[{i}]")
    elif hasattr(x, "tolist") and hasattr(x, "dtype"):
        yield path, hashlib.sha256(repr((str(x.dtype), x.tolist())).encode()).hexdigest()
    else:
        yield path, hashlib.sha256(repr(x).encode()).hexdigest()

out = []
with ThreadPoolExecutor(max_workers=1) as pool:
    for row, item in training_batches(spec["rows"], spec["span"], spec["options"]["training_seed"], pool, spec["options"], None, 1):
        fields = dict(walk(item.batch, "batch"))
        for name in ("carry", "carry_lanes", "carry_universe"):
            if hasattr(item, name):
                fields.update(walk(getattr(item, name), name))
        out.append({"user": row["user_id"], "offset": item.offset, "reviews": item.reviews, "chunks": item.chunks, "seed": item.seed,
                    "fields": fields})
print("PROBE " + json.dumps(out))
'''


class Report:
    def __init__(self):
        self.checks, self.ok = {}, True

    def check(self, name, ok, **detail):
        self.checks[name] = {"ok": bool(ok), **detail}
        self.ok &= bool(ok)
        print(f"[sm91-check] {name}: {'OK' if ok else 'FAIL'} {json.dumps(detail, default=str)[:300]}", flush=True)


def set_deleted(value):
    if value:
        os.environ["RWKV_DELETED_CARDS"] = value
    else:
        os.environ.pop("RWKV_DELETED_CARDS", None)


def trainer_samples(user, row, options, flag):
    """The arm's worker path for one user: the full frame (deleted cards: neighbour, as RWKV_DELETED_CARDS does in
    training), the show-time frame change, the train split, user-level segment features, then samples_for."""
    from rwkv.data_processing import get_rwkv_data
    from scripts.gpu_batch1 import split_frame
    from scripts.gpu_r01_samples import samples_for, show_time_frame
    from scripts.state_carry_train import user_segment_features
    set_deleted("neighbour")
    try:
        frame = get_rwkv_data(Path(DATA), user)
    finally:
        set_deleted(None)
    opts = dict(options)
    if flag:
        opts[FLAG] = True
        frame = show_time_frame(frame)
    section = user_segment_features(split_frame(frame, row, "train"))
    return samples_for(section, user, opts)


def scorer_rows(user, variant):
    from scripts import sm31_inputs
    set_deleted(None)
    s = sm31_inputs.samples(Path(DATA), user, variant, "neighbour")
    assert len(s) == 1
    return s[0]


def keyed(samples):
    """(review_th, is_query) -> (card feature row, global label row) over a list of samples."""
    import numpy as np
    out = {}
    for s in samples:
        cf, gl, th = s.card_features.numpy(), s.global_labels.numpy(), s.review_ths.numpy()
        q = gl[:, LABEL_COLUMNS.index("is_query")]
        for i in range(len(th)):
            key = (int(th[i]), int(q[i]))
            if key in out:
                raise RuntimeError(f"duplicate row key {key}")
            out[key] = (cf[i], gl[i])
    return out


def other_tensors(a, b):
    import torch
    diff = []
    for name in ("review_ths", "label_review_ths", "day_offsets", "day_offsets_first", "skips"):
        if not torch.equal(getattr(a, name), getattr(b, name)):
            diff.append(name)
    for k in a.ids:
        if not torch.equal(a.ids[k], b.ids[k]):
            diff.append(f"ids.{k}")
    for k, m in a.modules.items():
        n = b.modules[k]
        if not (torch.equal(m.from_perm, n.from_perm) and torch.equal(m.to_perm, n.to_perm)
                and (m.split_len == n.split_len).all() and (m.split_B == n.split_B).all()):
            diff.append(f"modules.{k}")
    if a.length != b.length or a.start_th != b.start_th or a.end_th != b.end_th:
        diff.append("length/start/end")
    return diff


def t1_t2(rep, rows, options):
    import numpy as np
    ls, iq, hl = (LABEL_COLUMNS.index(c) for c in ("label_elapsed_seconds", "is_query", "has_label"))
    t1, t2 = [], []
    for row in rows:
        user = row["user_id"]
        off, on = trainer_samples(user, row, options, False), trainer_samples(user, row, options, True)
        # T1: which columns move, on which rows
        rec = {"user": user, "train_reviews": row["train_range"][1] - row["train_range"][0], "samples": len(on)}
        bad = []
        qcols, ucols, labs, qrows, crows, shortest = set(), set(), set(), 0, 0, 0.0
        if len(on) != len(off):
            bad.append("sample counts differ")
        for a, b in zip(on, off):
            other = other_tensors(a, b)
            if other:
                bad.append(f"other tensors changed {other}")
            cfa, cfb, gla, glb = a.card_features.numpy(), b.card_features.numpy(), a.global_labels.numpy(), b.global_labels.numpy()
            q = glb[:, iq] == 1
            ch = cfa != cfb
            qcols |= set(int(c) for c in np.flatnonzero(ch[q].any(0)))
            ucols |= set(int(c) for c in np.flatnonzero(ch[~q].any(0)))
            qrows += int(ch[q].any(1).sum())
            labs |= {LABEL_COLUMNS[c] for c in np.flatnonzero((gla != glb).any(0))}
            if (gla[q, ls] != glb[q, ls]).any():
                bad.append("query rows' label_elapsed_seconds changed")
            moved = gla[~q, ls] != glb[~q, ls]
            crows += int(moved.sum())
            if moved.any():
                shortest = max(shortest, float((glb[~q, ls] - gla[~q, ls]).max()))
            if (gla[~q, ls] > glb[~q, ls]).any():
                bad.append("a curve query time got longer")
        rec.update(query_columns=sorted(qcols), update_columns=sorted(ucols), label_columns=sorted(labs),
                   query_rows_moved=qrows, curve_query_times_moved=crows, curve_query_time_max_shortening_s=shortest)
        if not qcols <= {2, 3, 4, 5, 6, 7} or ucols or not labs <= {"label_elapsed_seconds"}:
            bad.append("columns outside the expected set")
        if not qrows or not crows:
            bad.append("nothing moved (user without answer times?)")
        rec["failures"] = bad
        t1.append(rec)
        # T2: trainer rows vs the scorer's rows, keyed by (review_th, is_query)
        s_norm, s_e2s = keyed([scorer_rows(user, "normal")]), keyed([scorer_rows(user, "e2s-query")])
        k_off, k_on = keyed(off), keyed(on)
        rec2 = {"user": user, "trainer_rows": len(k_on), "scorer_rows": len(s_e2s)}
        bad2, base_mismatch, compared, mism = [], {}, 0, 0
        if set(k_on) != set(k_off) or not set(k_on) <= set(s_e2s) or set(s_e2s) != set(s_norm):
            bad2.append("row keys differ")
        else:
            for key in k_on:
                (cf1, gl1), (cf0, gl0) = k_on[key], k_off[key]
                (sc1, sg1), (sc0, sg0) = s_e2s[key], s_norm[key]
                # where the flag-off trainer row equals the scorer's normal row, the flag-on row must equal e2s-query's
                for name, t1v, t0v, s1v, s0v in (("cf", cf1, cf0, sc1, sc0), ("gl", gl1, gl0, sg1, sg0)):
                    same0 = t0v == s0v
                    if name == "gl":
                        same0[hl] = False   # has_label: the train split masks the last labels (not compared)
                    for c in np.flatnonzero(~same0):
                        if not (name == "gl" and c == hl):
                            base_mismatch[f"{name}{c}"] = base_mismatch.get(f"{name}{c}", 0) + 1
                    compared += int(same0.sum())
                    if (t1v[same0] != s1v[same0]).any():
                        mism += 1
                    # the change itself: trainer (on - off) == scorer (e2s - normal), bitwise, every column
                    tm, sm = t1v != t0v, s1v != s0v
                    if name == "gl":
                        tm[hl] = sm[hl] = False
                    if not (tm == sm).all() or (t1v[tm] != s1v[sm]).any() or (t1v[tm] != s1v[tm]).any():
                        mism += 1
        rec2.update(values_compared=compared, rows_mismatched=mism,
                    flag_off_vs_scorer_normal_differs=base_mismatch)
        if mism:
            bad2.append(f"{mism} row comparisons differ")
        if base_mismatch:
            bad2.append(f"flag-off trainer rows differ from the scorer's normal rows in {base_mismatch}")
        rec2["failures"] = bad2
        t2.append(rec2)
        print(f"[sm91-check] user {user}: T1 {'OK' if not bad else bad} T2 {'OK' if not bad2 else bad2}", flush=True)
    rep.check("T1_trainer_flag_moves_only_query_intervals_and_curve_times", all(not r["failures"] for r in t1), users=t1)
    rep.check("T2_trainer_rows_equal_scorer_e2s_query_rows", all(not r["failures"] for r in t2), users=t2)


def probe(tree, rows, options, work, tag):
    spec = work / f"probe-{tag}.json"
    spec.write_text(json.dumps({"rows": rows, "span": options["batch_chunks"] * options["chunk_reviews"],
                                "options": options}))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", RWKV_DELETED_CARDS="neighbour", PYTHONPATH="",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    r = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", PROBE, str(tree), str(spec)], cwd=tree,
                       capture_output=True, text=True, env=env)
    line = next((x for x in r.stdout.splitlines() if x.startswith("PROBE ")), None)
    if r.returncode or line is None:
        raise RuntimeError(f"probe {tag} failed ({r.returncode}): {r.stderr[-3000:]}")
    return json.loads(line[6:])


def t3(rep, rows, options, ref_tree, work):
    off = {k: v for k, v in options.items() if k != "packed_ipc"}
    on = dict(off, **{FLAG: True})
    a, b, c = probe(ref_tree, rows, off, work, "ref-off"), probe(ROOT, rows, off, work, "sm91-off"), probe(ROOT, rows, on, work, "sm91-on")
    same = len(a) == len(b) and all(x["fields"] == y["fields"] and x["offset"] == y["offset"] and x["reviews"] == y["reviews"]
                                    for x, y in zip(a, b))
    diff = sorted({k for x, y in zip(a, b) for k in set(x["fields"]) | set(y["fields"]) if x["fields"].get(k) != y["fields"].get(k)})
    rep.check("T3_flag_off_batches_equal_current_trainer", same and len(a) >= len(rows), batches=len(a),
              tensors_per_batch=[len(x["fields"]) for x in a], differing=diff[:20], reference_tree=str(ref_tree))
    moved = sorted({k for x, y in zip(c, b) for k in set(x["fields"]) | set(y["fields"]) if x["fields"].get(k) != y["fields"].get(k)})
    rep.check("T3b_flag_on_batch_tensors_that_change", len(c) == len(b) and bool(moved), changed=moved,
              note="informational: the packed card-feature / label tensors; the permutations and id tensors must not be listed")


def t4(rep, work):
    pair = ROOT / "artifacts/small-model/sm91/sm91_pair.py"
    arm = ROOT / "artifacts/small-model/sm91-arm-e21"
    run = lambda *a: subprocess.run([sys.executable, "-X", "utf8", str(pair), *a], capture_output=True, text=True)  # noqa: E731
    r = run("epoch", "--arm", str(arm), "--ref", str(RUN72S / "epoch-21"))
    rep.check("T4a_arm_epoch_equals_sm72s_epoch21_except_flag_and_paths", r.returncode == 0, out=r.stdout.strip()[-400:])
    test = work / "test-cooldown-from-sm72s-e21"
    rel = test.relative_to(ROOT).as_posix()
    if not (test / "decay.json").exists():
        extra = json.loads((ROOT / "artifacts/small-model/sm91/extra-options.json").read_text())
        p = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts/sm77_warm.py"), "cooldown", "--epoch", "21",
                            "--start", str(RUN72S / "epoch-21"), "--out", rel, "--fraction", "0.25",
                            "--extra-options", json.dumps(dict(extra, **{FLAG: True}))],
                           capture_output=True, text=True, env=dict(os.environ, RWKV_DELETED_CARDS="neighbour", CUDA_VISIBLE_DEVICES=""))
        if p.returncode:
            rep.check("T4b_cooldown_prepare", False, err=p.stderr[-1500:])
            return
    r = run("decay", "--arm", str(test), "--ref", str(DREF), "--same-start")
    rep.check("T4b_flag_cooldown_equals_reference_except_flag_and_names", r.returncode == 0, out=r.stdout.strip()[-400:])
    r = run("decay", "--arm", str(DREF), "--ref", str(DREF), "--same-start")
    rep.check("T4c_reference_against_itself_refused_without_flag", r.returncode == 1, out=r.stdout.strip()[-300:])


def t5(rep):
    r = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "artifacts/small-model/sm91/sm91_rule.py"), "self-test"],
                       capture_output=True, text=True)
    rep.check("T5_rule_reproduces_sm31_and_zero", r.returncode == 0, out=r.stdout.strip()[-800:], err=r.stderr[-800:])


def cpu_check(args):
    import torch
    torch.set_num_threads(2)
    rep = Report()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    config = json.loads((RUN72S / "run.json").read_text())
    protocol = json.loads((RUN72S / "epoch-21" / "protocol.json").read_text())
    by_user = {r["user_id"]: r for r in protocol["rows"]}
    rows = [by_user[u] for u in USERS]
    options = dict(config["options"])
    if not args.skip_t12:
        t1_t2(rep, rows, options)
    t3(rep, rows, options, Path(args.ref_tree), work)
    t4(rep, work)
    t5(rep)
    from scripts.experiment_provenance import process_provenance
    out = {"ok": rep.ok, "checks": rep.checks, "users": list(USERS), "provenance": process_provenance()}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"[sm91-check] {'ALL OK' if rep.ok else 'FAIL'}: " + ", ".join(f"{k} {'OK' if v['ok'] else 'FAIL'}" for k, v in rep.checks.items()))
    return 0 if rep.ok else 1


def main():
    ap = ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("cpu-check")
    p.add_argument("--out", default=str(ROOT / "artifacts/small-model/sm91/check-cpu-sm91.json"))
    p.add_argument("--work", default=str(ROOT / "artifacts/small-model/sm91/work"))
    p.add_argument("--ref-tree", required=True, help="the current trainer's tree (git archive of the sm73s2 commit)")
    p.add_argument("--skip-t12", action="store_true")
    args = ap.parse_args()
    return cpu_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
