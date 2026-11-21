"""Per-user adaptation (SM143): proof checks.

  zero   --rows ADAPT.jsonl --scorer SCORER.jsonl [--tol 0]
         The 0-step adaptation rows against the stock scorer's rows (scripts/small_model_heldout.py, same label, users,
         inputs, device): instant/curve totals and the "benchmark" dict equal (|diff| <= tol; 0 = bit for bit on the
         same device), head_identity_max_abs and d_identity_max_abs <= tol, every variant's evaluation predictions equal
         to the base ones (zero_step_equal), causal entries present for every fold, y_equal true. Exit 1 on failure.
  smoke  --gpu GPU.jsonl --cpu CPU.jsonl [--tol 1e-5]
         GPU rows vs the CPU rows of the same run (0 steps): per user the base benchmark / totals within tol, the
         identity checks within tol, zero_step_equal everywhere. Prints "smoke OK" on success.
  steps  --rows ROWS [--tol 1e-5]
         Rows of a real-step run: no error row, identity checks within tol, a causal entry per fold (and per d fold),
         every variant has a step record in every adapted fold, d trained (windows > 0, train_seconds > 0), finite
         metrics. Prints "steps OK".
  pins   --models models.sh --pins design-pins.json [--write]
         sha256 of the checkpoints, the users files and the SM81 FSRS-7 files of the pilot users (one digest) equal the
         pinned values (--write records them).
"""
import hashlib
import math
import re
from pathlib import Path
import argparse
import json
import sys


def rows(path, keep_errors=False):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "error" in r and not keep_errors:
            continue
        out[(r["user_id"], r["label"])] = r
    return out


def flat(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


FIELDS = ("curve_labels", "instant_labels", "curve_log_loss", "curve_raw_log_loss", "instant_log_loss",
          "instant_rating_cross_entropy")


def compare(a, b, tol, fails, what):
    for f in FIELDS:
        if abs(a[f] - b[f]) > tol:
            fails.append(f"{what}: {f} {a[f]!r} vs {b[f]!r}")
    fa, fb = dict(flat(a["benchmark"])), dict(flat(b["benchmark"]))
    if set(fa) != set(fb):
        fails.append(f"{what}: benchmark keys differ")
    for k in fa:
        x, y = fa[k], fb.get(k)
        if x is None or y is None:
            if x != y:
                fails.append(f"{what}: benchmark {k} {x!r} vs {y!r}")
        elif abs(x - y) > tol:
            fails.append(f"{what}: benchmark {k} {x!r} vs {y!r}")


def own_checks(r, tol, fails, what):
    if r.get("head_identity_max_abs", 1) > tol:
        fails.append(f"{what}: head_identity_max_abs {r.get('head_identity_max_abs')}")
    if "d" in r["variants"] and r.get("d_identity_max_abs", 1) > tol:
        fails.append(f"{what}: d_identity_max_abs {r.get('d_identity_max_abs')}")
    if not all(r["zero_step_equal"].values()):
        fails.append(f"{what}: zero_step_equal {r['zero_step_equal']}")
    if not r.get("y_equal"):
        fails.append(f"{what}: y_equal false")
    if len(r.get("causal", [])) < len(r.get("folds", [])):
        fails.append(f"{what}: causal entries missing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("zero", "smoke", "steps", "pins"))
    ap.add_argument("--models")
    ap.add_argument("--pins")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--rows")
    ap.add_argument("--scorer")
    ap.add_argument("--gpu")
    ap.add_argument("--cpu")
    ap.add_argument("--tol", type=float, default=None)
    a = ap.parse_args()
    fails, n = [], 0
    if a.mode == "pins":
        return pins(a)
    if a.mode == "steps":
        tol = a.tol if a.tol is not None else 1e-5
        for line in open(a.rows, encoding="utf-8"):
            r = json.loads(line)
            n += 1
            what = f"user {r['user_id']}"
            if "error" in r:
                fails.append(f"{what}: error {r['error'][:120]}")
                continue
            if r.get("head_identity_max_abs", 1) > tol or ("d" in r["variants"] and r.get("d_identity_max_abs", 1) > tol):
                fails.append(f"{what}: identity {r.get('head_identity_max_abs')} {r.get('d_identity_max_abs')}")
            adapted = [f for f in r["folds"] if "skipped" not in f]
            if len(r["causal"]) < len(r["folds"]) + ("d" in r["variants"]) * len(adapted):
                fails.append(f"{what}: causal entries missing")
            for f in adapted:
                for v in r["variants"]:
                    if v not in f["variants"]:
                        fails.append(f"{what} fold {f['fold']}: no record for {v}")
                d = f["variants"].get("d")
                if d is not None and not (d["windows"] > 0 and d["train_seconds"] > 0):
                    fails.append(f"{what} fold {f['fold']}: d did not train")
            for k, m in r["metrics"].items():
                if not math.isfinite(m["ll_sum"]):
                    fails.append(f"{what}: {k} LogLoss not finite")
        if fails or n == 0:
            print(f"steps FAIL ({n} rows): " + "; ".join(fails[:12]))
            sys.exit(1)
        print(f"steps OK: {n} rows")
        return
    if a.mode == "zero":
        tol = a.tol if a.tol is not None else 0.0
        mine, ref = rows(a.rows), rows(a.scorer)
        for key, r in mine.items():
            s = ref.get(key)
            if s is None:
                fails.append(f"{key}: no scorer row")
                continue
            compare(r, s, tol, fails, str(key))
            own_checks(r, tol, fails, str(key))
            n += 1
    else:
        tol = a.tol if a.tol is not None else 1e-5
        g, c = rows(a.gpu), rows(a.cpu)
        for key, r in c.items():
            s = g.get(key)
            if s is None:
                fails.append(f"{key}: no GPU row")
                continue
            compare(s, r, tol, fails, str(key))
            own_checks(s, tol, fails, str(key) + " gpu")
            n += 1
    if fails or n == 0:
        print(f"{a.mode} FAIL ({n} rows): " + "; ".join(fails[:12]))
        sys.exit(1)
    print(f"{a.mode} OK: {n} rows within {tol}" if a.mode == "zero" else f"smoke OK: {n} rows within {tol}")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pins(a):
    here = Path(a.pins).parent
    text = Path(a.models).read_text(encoding="utf-8")
    w = re.search(r"^W=(.+)$", text, re.M).group(1)
    track = re.search(r"^TRACK=(.+)$", text, re.M).group(1)
    fsrs = Path(re.search(r"^FSRS=(.+)$", text, re.M).group(1).replace("$W", w))
    cur = {"models": {}, "users": {}}
    for m in re.finditer(r'^(SMALL|TINY|BIG)="([^"]+)"', text, re.M):
        tag, ka, path, digest, role = m.group(2).split("|")
        path = path.replace("$W", w).replace("$TRACK", track)
        got = sha(path)
        if got != digest:
            print(f"pins FAIL: {tag} checkpoint sha256 {got} != models.sh {digest}")
            sys.exit(1)
        cur["models"][tag] = got
    for name in ("users-smoke.json", "users-smoke-one.json", "users-pilot.json", "users-pilot-d.json"):
        cur["users"][name] = sha(here / name)
    users = json.loads((here / "users-pilot.json").read_text())["users"]
    h = hashlib.sha256()
    for u in sorted(set(users) | {2388, 4676}):
        h.update((f"{u}:" + sha(fsrs / f"{u}.npz") + "\n").encode())
    cur["fsrs7_files_digest"] = h.hexdigest()
    if a.write:
        Path(a.pins).write_text(json.dumps(cur, indent=1) + "\n", encoding="utf-8")
        print(f"pins written: {a.pins}")
        return
    old = json.loads(Path(a.pins).read_text(encoding="utf-8"))
    if old != cur:
        print(f"pins FAIL: {[k for k in cur if cur[k] != old.get(k)]} changed")
        sys.exit(1)
    print("pins OK")


if __name__ == "__main__":
    main()
