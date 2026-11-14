#!/usr/bin/env python
"""The show-time error sensitivity test (SM100): CPU equivalence checks of the shifted show-time inputs (before any GPU run).

On a few users of the 732 set (the scorer's own path: scripts/sm31_inputs.samples with --deleted-cards neighbour, then
rwkv.prepare_batch.prepare with the scorer's seed 2025 + user):
  C1  e2s-query of this branch == e2s-query of the show-time instant-only small model (SM96)'s code (sm96 commit e54fdb93,
      read with `git show`, not from wt/sm96), whole batch bit for bit (every tensor: dtype, shape, values).
  C2  e2s-query+0s == e2s-query, whole batch bit for bit.
  C3  for +10 s / +60 s / +300 s / -60 s, on the add_queries frames (the same frames the batch is built from):
      only the 8 query interval columns of query rows move (update rows and every other column identical);
      elapsed_seconds and elapsed_seconds_cumulative move by exactly the shift where the show-time interval is >= 0
      (negative shifts: floored at 0); the -1 sentinel is kept; the sin / cos / scaled columns equal the formulas
      applied to the moved values; the curve query's label_elapsed_seconds (apply_frame) moves the same way.
  C4  the shifted batches differ from the unshifted one (the shift reaches the model inputs), and only in the tensors the
      interval columns feed.
  C5  the scorer accepts the variant names (--inputs type) and refuses a malformed one.

  python artifacts/small-model/sm100/sm100_check.py   -> check-cpu-sm100.json (all_ok), exit 0 if all OK
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
os.environ.pop("RWKV_DELETED_CARDS", None)
os.environ.pop("EXPERTIUM_END_TO_START", None)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts import compress_common as cc  # noqa: E402
from scripts import sm31_inputs as new  # noqa: E402

USERS = (2388, 4676, 1712, 1286)        # the three shortest users of the 732 set + one median user (29,268 reviews)
SHIFTS = (10, 60, 300, -60)
SM96_COMMIT = "e54fdb93"
MODE = "neighbour"


def old_module():
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{SM96_COMMIT}:scripts/sm31_inputs.py"], check=True,
                         capture_output=True, text=True, encoding="utf-8").stdout
    mod = types.ModuleType("sm31_inputs_sm96")
    mod.__file__ = f"sm96:{SM96_COMMIT}:scripts/sm31_inputs.py"
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def flat(obj, prefix=""):
    """{path: tensor-or-scalar} for a batch (dataclass / object / dict / list / tensor)."""
    out = {}
    if isinstance(obj, torch.Tensor):
        out[prefix] = obj
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            out |= flat(obj[k], f"{prefix}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out |= flat(v, f"{prefix}[{i}]")
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        for k in sorted(vars(obj)):
            out |= flat(vars(obj)[k], f"{prefix}.{k}")
    else:
        out[prefix] = obj
    return out


def diff_paths(a, b):
    fa, fb = flat(a), flat(b)
    if fa.keys() != fb.keys():
        return ["<structure>"]
    bad = []
    for k in fa:
        x, y = fa[k], fb[k]
        if isinstance(x, torch.Tensor):
            if not (isinstance(y, torch.Tensor) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)):
                bad.append(k)
        elif not (x == y or (x != x and y != y)):
            bad.append(k)
    return bad


def batch(mod, data, user, variant):
    from rwkv.prepare_batch import prepare
    return prepare(mod.samples(data, user, variant, MODE), seed=2025 + user)


def query_frames(data, user):
    """The scorer's frame for e2s-query up to add_queries (one full-history section), with the unpatched add_queries."""
    from rwkv import data_processing as dp
    from scripts import sm71_leak
    frame = new.load_frame(data, user, "e2s-query")
    frame, _ = sm71_leak.neutralise(frame, MODE, data, user)
    section = dp.add_segment_features(frame.copy()).reset_index(drop=True)
    return frame, dp.add_queries(section.copy(), [])


def check_frames(frame, qdf, shift):
    from rwkv import data_processing as dp
    res = {}
    base = new._query_rows_show_time(qdf, 0)
    moved = new._query_rows_show_time(qdf, shift)
    q = qdf["is_query"].to_numpy() == 1
    cols = list(new.QUERY_INTERVAL_COLUMNS)
    other = [c for c in qdf.columns if c not in cols]
    res["other_columns_equal"] = bool(base[other].equals(moved[other]))
    res["update_rows_equal"] = bool(base.loc[~q, cols].equals(moved.loc[~q, cols]))
    es0 = base.loc[q, "elapsed_seconds"].to_numpy().astype("float64")
    es1 = moved.loc[q, "elapsed_seconds"].to_numpy().astype("float64")
    sent = es0 == -1
    want = np.where(sent, -1.0, np.maximum(es0 + shift, 0.0))
    res["query_rows"] = int(q.sum())
    res["sentinel_rows"] = int(sent.sum())
    res["elapsed_moved_exactly"] = bool(np.array_equal(es1, want.astype(base["elapsed_seconds"].dtype).astype("float64")))
    c0 = base.loc[q, "elapsed_seconds_cumulative"].to_numpy().astype("float64")
    c1 = moved.loc[q, "elapsed_seconds_cumulative"].to_numpy().astype("float64")
    res["cumulative_moved_by_interval_move"] = bool(np.allclose(c1 - c0, es1 - es0, rtol=0, atol=1e-6))
    day = 86400
    sub = moved.loc[q]
    forms = {"elapsed_seconds_sin": np.sin((es1 % day) * 2 * np.pi / day), "elapsed_seconds_cos": np.cos((es1 % day) * 2 * np.pi / day),
             "elapsed_seconds_cumulative_sin": np.sin((c1 % day) * 2 * np.pi / day),
             "elapsed_seconds_cumulative_cos": np.cos((c1 % day) * 2 * np.pi / day),
             "scaled_elapsed_seconds": np.asarray(dp.scale_elapsed_seconds(es1)),
             "scaled_elapsed_seconds_cumulative": np.asarray(dp.scale_elapsed_seconds_cumulative(c1))}
    res["derived_columns_match_formulas"] = bool(all(
        np.allclose(sub[c].to_numpy().astype("float64"), forms[c].astype(sub[c].dtype).astype("float64"), rtol=0, atol=1e-6)
        for c in forms))
    res["query_rows_changed"] = int((es1 != es0).sum())
    # curve query label (apply_frame on the full frame)
    l0 = new.apply_frame(frame, "e2s-query")["label_elapsed_seconds"].to_numpy().astype("float64")
    l1 = new.apply_frame(frame, f"e2s-query{shift:+d}s")["label_elapsed_seconds"].to_numpy().astype("float64")
    fill = l0 == np.float64(np.pi).astype(frame["label_elapsed_seconds"].dtype)
    lsent = l0 == -1
    lwant = np.where(fill | lsent, l0, np.maximum(l0 + shift, 0.0))
    res["curve_label_moved_exactly"] = bool(np.allclose(l1, lwant, rtol=0, atol=1e-6))
    keys = ("other_columns_equal", "update_rows_equal", "elapsed_moved_exactly", "cumulative_moved_by_interval_move",
            "derived_columns_match_formulas", "curve_label_moved_exactly")
    res["ok"] = all(res[k] for k in keys) and res["query_rows_changed"] > 0
    return res


def check_parse():
    ok = new.parse("e2s-query+60s") == ("e2s-query", 60) and new.parse("e2s-query-10s") == ("e2s-query", -10) \
        and new.parse("e2s-query") == ("e2s-query", 0) and new.parse("normal") == ("normal", 0)
    for bad in ("e2s-query+60", "e2s-query+1.5s", "e2s-all+10s", "foo"):
        try:
            new.parse(bad)
            ok = False
        except ValueError:
            pass
    r = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts/small_model_heldout.py"), "--inputs", "e2s-query+bad",
                        "--device", "cpu", "--count", "0"], capture_output=True, text=True, cwd=str(ROOT))
    refused = r.returncode == 2 and "--inputs" in r.stderr
    return {"parse_ok": ok, "scorer_refuses_malformed": refused, "ok": ok and refused}


def main():
    data = cc.data_path()
    old = old_module()
    out = {"users": list(USERS), "shifts": list(SHIFTS), "sm96_code": f"git show {SM96_COMMIT}:scripts/sm31_inputs.py",
           "deleted_cards": MODE, "per_user": {}}
    all_ok = True
    for user in USERS:
        u = {}
        b_new = batch(new, data, user, "e2s-query")
        b_old = batch(old, data, user, "e2s-query")
        d = diff_paths(b_new, b_old)
        u["C1_e2s_query_equals_sm96_code"] = {"ok": not d, "differing": d[:5], "tensors": len(flat(b_new))}
        d = diff_paths(batch(new, data, user, "e2s-query+0s"), b_new)
        u["C2_shift0_equals_e2s_query"] = {"ok": not d, "differing": d[:5]}
        frame, qdf = query_frames(data, user)
        u["C3_frames"] = {f"{s:+d}s": check_frames(frame, qdf, s) for s in SHIFTS}
        b60 = batch(new, data, user, "e2s-query+60s")
        d = diff_paths(b60, b_new)
        u["C4_shift60_batch_differs"] = {"ok": bool(d), "differing": d}
        b_norm = batch(new, data, user, "normal")
        u["normal_differs_from_e2s_query"] = bool(diff_paths(b_norm, b_new))
        ok = u["C1_e2s_query_equals_sm96_code"]["ok"] and u["C2_shift0_equals_e2s_query"]["ok"] \
            and all(x["ok"] for x in u["C3_frames"].values()) and u["C4_shift60_batch_differs"]["ok"]
        u["ok"] = ok
        all_ok &= ok
        out["per_user"][str(user)] = u
        print(f"[sm100-check] user {user}: C1 {u['C1_e2s_query_equals_sm96_code']['ok']} C2 {u['C2_shift0_equals_e2s_query']['ok']} "
              f"C3 {[x['ok'] for x in u['C3_frames'].values()]} C4 {u['C4_shift60_batch_differs']['ok']} "
              f"(+60 s moves {u['C4_shift60_batch_differs']['differing']}; {u['C3_frames']['+60s']['query_rows']} query rows, "
              f"{u['C3_frames']['+60s']['query_rows_changed']} moved)", flush=True)
    out["C5_names"] = check_parse()
    all_ok &= out["C5_names"]["ok"]
    out["all_ok"] = bool(all_ok)
    (HERE / "check-cpu-sm100.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[sm100-check] C5 {out['C5_names']}; ALL {'OK' if all_ok else 'FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
