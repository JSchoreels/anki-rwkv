"""SM72 CPU check of the opt-in training-input transform RWKV_DELETED_CARDS (rwkv/data_processing.get_rwkv_data).

For each user:
  off : RWKV_DELETED_CARDS unset -> the frame equals the frame of the reference get_rwkv_data (the sm71 commit's
        rwkv/data_processing.py, loaded from git) and the full-history sample (compress_common.make_samples) is
        tensor-for-tensor identical;
  on  : RWKV_DELETED_CARDS=neighbour -> the frame equals scripts/sm71_leak.neutralise(reference, "neighbour") (the
        scorer's --deleted-cards neighbour), and against the reference sample the changes are only on deleted-card
        rows, only in the 3 *_is_nan columns and the deck/preset ids; the counts are compared with SM71's
        artifacts/small-model/sm71/check-inputs.json ("neighbour" block).
usage: sm72_check.py USER [USER ...] --out JSON [--reference-commit sm71]
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts import compress_common as cc  # noqa: E402
from scripts import sm71_leak  # noqa: E402
from scripts.sm71_check import tensors  # noqa: E402

ENV = "RWKV_DELETED_CARDS"
KEYS = ("feature_rows_changed", "feature_changes_outside_deleted_rows", "feature_changes_outside_flag_columns",
        "ids_changed", "ids_changed_outside_deleted_rows", "other_sample_tensors_differ")


def reference_module(commit):
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:rwkv/data_processing.py"], check=True,
                         capture_output=True, text=True, encoding="utf-8").stdout
    path = Path(tempfile.mkdtemp()) / "data_processing_reference.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("data_processing_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same_samples(a, b):
    ta, tb = tensors(a), tensors(b)
    return sorted(ta) == sorted(tb) and all(torch.equal(ta[k], tb[k]) for k in ta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("users", type=int, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference-commit", default="sm71")
    ap.add_argument("--sm71", default=str(ROOT / "artifacts/small-model/sm71/check-inputs.json"))
    args = ap.parse_args()
    os.environ.pop(ENV, None)
    from rwkv import data_processing as dp
    ref_dp = reference_module(args.reference_commit)
    data = cc.data_path()
    flag_cols = [dp.CARD_FEATURE_COLUMNS.index(c) for c in ("note_id_is_nan", "deck_id_is_nan", "preset_id_is_nan")]
    sm71 = json.loads(Path(args.sm71).read_text(encoding="utf-8"))
    report, ok = {}, True
    for user in args.users:
        ref = ref_dp.get_rwkv_data(data, user)
        base = cc.make_samples(ref, user, len(ref))[0]
        os.environ.pop(ENV, None)
        off = dp.get_rwkv_data(data, user)
        rec = {"reviews": len(ref), "off_frame_equal": bool(off.equals(ref)),
               "off_sample_identical": same_samples(cc.make_samples(off, user, len(off))[0], base)}
        os.environ[ENV] = "neighbour"
        try:
            on = dp.get_rwkv_data(data, user)
        finally:
            os.environ.pop(ENV, None)
        expected, stats = sm71_leak.neutralise(ref, "neighbour", data, user)
        rec["on_frame_equals_sm71_neighbour"] = bool(on.equals(expected))
        changed_cols = [c for c in ref.columns if not ref[c].equals(on[c])]
        rec["on_frame_columns_changed"] = changed_cols
        s = cc.make_samples(on, user, len(on))[0]
        gone = set(ref.loc[ref["note_id_is_nan"] == 1, "card_id"].astype(np.int64).tolist())
        row_gone = torch.isin(base.ids["card_id"], torch.tensor(sorted(gone), dtype=torch.int64).to(torch.int32))
        rows, cols = torch.nonzero(s.card_features != base.card_features, as_tuple=True)
        r = {"stats": stats,
             "feature_rows_changed": int(rows.unique().numel()),
             "feature_changes_outside_deleted_rows": int((~row_gone[rows]).sum()),
             "feature_changes_outside_flag_columns": int((~torch.isin(cols, torch.tensor(flag_cols))).sum()),
             "ids_changed": {k: int((s.ids[k] != base.ids[k]).sum()) for k in base.ids},
             "ids_changed_outside_deleted_rows": {k: int(((s.ids[k] != base.ids[k]) & ~row_gone).sum()) for k in base.ids}}
        ref_t = tensors(base)
        r["other_sample_tensors_differ"] = [k for k, v in tensors(s).items()
                                            if not k.startswith((".card_features", ".ids")) and not torch.equal(v, ref_t[k])]
        rec["deleted_rows_in_sample"] = int(row_gone.sum())
        rec["neighbour"] = r
        want = sm71.get(str(user), {}).get("neighbour")
        if want is not None:
            rec["matches_sm71_check_inputs"] = {k: r[k] == want[k] for k in KEYS} | {"stats": r["stats"] == want["stats"]}
        rec["pass"] = (rec["off_frame_equal"] and rec["off_sample_identical"] and rec["on_frame_equals_sm71_neighbour"]
                       and set(changed_cols) <= {"note_id_is_nan", "deck_id_is_nan", "preset_id_is_nan", "deck_id", "preset_id"}
                       and r["feature_changes_outside_deleted_rows"] == 0 and r["feature_changes_outside_flag_columns"] == 0
                       and all(v == 0 for v in r["ids_changed_outside_deleted_rows"].values())
                       and r["ids_changed"]["card_id"] == r["ids_changed"]["note_id"] == r["ids_changed"]["user_id"] == 0
                       and (want is None or all(rec["matches_sm71_check_inputs"].values())))
        ok &= rec["pass"]
        report[user] = rec
        print(json.dumps({user: rec})[:1200], flush=True)
    report["all_pass"] = bool(ok)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print("ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
