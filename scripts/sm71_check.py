"""SM71 input checks on a few users: what --deleted-cards changes in the model inputs.

For each user: the get_rwkv_data frame with and without scripts/sm71_leak.neutralise, cut into the scorer's one
full-history sample (compress_common.make_samples). Checks that
  flags     : ids are bit-identical; card_features differ only in rows (reviews and queries) of deleted cards, and only
              in the three *_is_nan columns; the prepared batch (random id/day codes, seed 2025 + user) differs only in
              those same entries;
  neighbour : card/note/user ids identical; deck/preset ids differ only on deleted-card rows; card_features as flags.
Also counts affected reviews. usage: sm71_check.py USER [USER ...] --out JSON
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts import compress_common as cc  # noqa: E402
from scripts import sm71_leak  # noqa: E402


def tensors(obj, prefix=""):
    """Flatten the tensors of a sample / prepared batch into {path: tensor}."""
    out = {}
    if isinstance(obj, torch.Tensor):
        out[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(tensors(v, f"{prefix}.{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(tensors(v, f"{prefix}[{i}]"))
    elif hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            out.update(tensors(v, f"{prefix}.{k}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("users", type=int, nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from rwkv.data_processing import CARD_FEATURE_COLUMNS, get_rwkv_data
    from rwkv.prepare_batch import prepare
    data = cc.data_path()
    flag_cols = [CARD_FEATURE_COLUMNS.index(c) for c in ("note_id_is_nan", "deck_id_is_nan", "preset_id_is_nan")]
    report = {}
    for user in args.users:
        frame = get_rwkv_data(data, user)
        base = cc.make_samples(frame, user, len(frame))[0]
        gone_ths = sm71_leak.deleted_review_ths(data, user)
        rec = {"reviews": len(frame), "deleted_reviews_by_table": int(len(gone_ths))}
        gone_cards = set(frame.loc[frame["note_id_is_nan"] == 1, "card_id"].astype(np.int64).tolist())
        gone32 = torch.tensor(sorted(gone_cards), dtype=torch.int64).to(torch.int32)
        row_gone = torch.isin(base.ids["card_id"], gone32)
        rec["deleted_rows_in_sample"] = int(row_gone.sum())
        rec["sample_rows"] = int(base.card_features.shape[0])
        for mode in ("flags", "neighbour"):
            new_frame, stats = sm71_leak.neutralise(frame, mode, data, user)
            s = cc.make_samples(new_frame, user, len(new_frame))[0]
            r = {"stats": stats}
            diff = (s.card_features != base.card_features)
            rows, cols = torch.nonzero(diff, as_tuple=True)
            r["feature_rows_changed"] = int(rows.unique().numel())
            r["feature_changes_outside_deleted_rows"] = int((~row_gone[rows]).sum())
            r["feature_changes_outside_flag_columns"] = int((~torch.isin(cols, torch.tensor(flag_cols))).sum())
            r["ids_changed"] = {k: int((s.ids[k] != base.ids[k]).sum()) for k in base.ids}
            r["ids_changed_outside_deleted_rows"] = {k: int(((s.ids[k] != base.ids[k]) & ~row_gone).sum())
                                                     for k in base.ids}
            other = {k: v for k, v in tensors(s).items() if not k.startswith((".card_features", ".ids"))}
            ref = tensors(base)
            r["other_sample_tensors_differ"] = [k for k, v in other.items() if not torch.equal(v, ref[k])]
            if mode == "flags":
                pa = tensors(prepare([base], seed=2025 + user))
                pb = tensors(prepare([s], seed=2025 + user))
                bad = {}
                for k, v in pa.items():
                    w = pb[k]
                    if v.shape != w.shape:
                        bad[k] = "shape"
                    elif not torch.equal(v, w):
                        bad[k] = int((v != w).sum())
                r["prepared_tensors_differ"] = bad
            rec[mode] = r
        report[user] = rec
        print(json.dumps({user: rec})[:1500], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
