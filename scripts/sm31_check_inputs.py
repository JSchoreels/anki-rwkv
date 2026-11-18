#!/usr/bin/env python
"""CPU proof for the scorer's input variants (scripts/sm31_inputs.py; duration-leak scoring (SM31), robustness to
other apps (SM29)).

For each user (full history, --deleted-cards neighbour as in the leak-free scoring):
  1. `normal` through sm31_inputs.samples is bit-identical to the scorer's LeakBatches path (get_rwkv_data +
     neutralise + make_samples): every tensor of every sample.
  2. each variant against normal: which card-feature columns and global-label columns change, on update rows and on
     instant query rows; ids, module permutations, review_ths, label_review_ths, day offsets and skips must be equal,
     and so must every label except the curve query time of the e2s variants. Expected changes:
       e2s-query       query rows: the 6 interval columns only (2-7); update rows: nothing; global labels:
                       label_elapsed_seconds of update rows only (the curve query time), equal to e2s-all's
       e2s-all         interval columns 2-7 on any row; label_elapsed_seconds; and on query rows columns 2-4 equal
                       e2s-query's (the same show-time interval)
       passfail        rating one-hot 9-12 on update rows of Hard / Easy reviews only
       notime          scaled_duration (8) on update rows only (query rows already carry 0 there)
       passfail-notime the union of the two
usage: sm31_check_inputs.py --users 1 2 3 --out artifacts/small-model/sm31/check-cpu-sm31-inputs.json
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import compress_common as cc  # noqa: E402
from scripts import sm31_inputs  # noqa: E402
from scripts import sm71_leak  # noqa: E402
from rwkv.data_processing import CARD_FEATURE_COLUMNS, get_rwkv_data  # noqa: E402

LABEL_COLUMNS = ["label_elapsed_seconds", "label_elapsed_days", "label_y", "label_rating", "has_label",
                 "label_is_equalize", "is_query"]
EXPECT_CARD = {"e2s-query": ({2, 3, 4, 5, 6, 7}, set()), "e2s-all": ({2, 3, 4, 5, 6, 7}, {2, 3, 4, 5, 6, 7}),
               "passfail": (set(), {9, 10, 11, 12}), "notime": (set(), {8}),
               "passfail-notime": (set(), {8, 9, 10, 11, 12})}   # (query rows may change, update rows may change)


def leak_path_samples(data, user):
    frame = get_rwkv_data(data, user)
    frame, _ = sm71_leak.neutralise(frame, "neighbour", data, user)
    return cc.make_samples(frame, user, len(frame))


def tensors(sample):
    out = {"card_features": sample.card_features, "global_labels": sample.global_labels,
           "review_ths": sample.review_ths, "label_review_ths": sample.label_review_ths,
           "day_offsets": sample.day_offsets, "day_offsets_first": sample.day_offsets_first, "skips": sample.skips}
    for k, v in sample.ids.items():
        out[f"ids.{k}"] = v
    for k, m in sample.modules.items():
        out[f"modules.{k}.from_perm"] = m.from_perm
        out[f"modules.{k}.split_len"] = torch.as_tensor(m.split_len)
        out[f"modules.{k}.split_B"] = torch.as_tensor(m.split_B)
    return out


def bitwise_equal(a, b):
    ta, tb = tensors(a), tensors(b)
    return sorted(k for k in ta if ta[k].shape != tb[k].shape or not torch.equal(ta[k], tb[k]))


def check_user(data, user):
    rec, fail = {"user": user}, []
    base = sm31_inputs.samples(data, user, "normal", "neighbour")
    ref = leak_path_samples(data, user)
    assert len(base) == len(ref) == 1
    diff = bitwise_equal(base[0], ref[0])
    rec["normal_vs_scorer_path_differs"] = diff
    if diff:
        fail.append(f"user {user}: normal differs from the scorer path in {diff}")
    b = base[0]
    q = b.global_labels[:, LABEL_COLUMNS.index("is_query")].numpy() == 1
    rec["rows"], rec["query_rows"] = int(len(q)), int(q.sum())
    cf0, gl0 = b.card_features.numpy(), b.global_labels.numpy()
    got = {}
    for variant in EXPECT_CARD:
        s = sm31_inputs.samples(data, user, variant, "neighbour")[0]
        got[variant] = s
        cf, gl = s.card_features.numpy(), s.global_labels.numpy()
        if cf.shape != cf0.shape:
            fail.append(f"user {user} {variant}: shape {cf.shape} vs {cf0.shape}")
            continue
        other = [k for k in bitwise_equal(s, b) if k not in ("card_features", "global_labels")]
        ch = cf != cf0
        v = {"query_columns": sorted(int(c) for c in np.flatnonzero(ch[q].any(0))),
             "update_columns": sorted(int(c) for c in np.flatnonzero(ch[~q].any(0))),
             "query_rows_changed": int(ch[q].any(1).sum()), "update_rows_changed": int(ch[~q].any(1).sum()),
             "label_columns_changed": [LABEL_COLUMNS[c] for c in np.flatnonzero((gl != gl0).any(0))],
             "other_tensors_changed": other}
        v["query_column_names"] = [CARD_FEATURE_COLUMNS[c] for c in v["query_columns"]]
        v["update_column_names"] = [CARD_FEATURE_COLUMNS[c] for c in v["update_columns"]]
        rec[variant] = v
        allowed_q, allowed_u = EXPECT_CARD[variant]
        if other:
            fail.append(f"user {user} {variant}: other tensors changed {other}")
        if not set(v["query_columns"]) <= allowed_q or not set(v["update_columns"]) <= allowed_u:
            fail.append(f"user {user} {variant}: columns q{v['query_columns']} u{v['update_columns']} outside the expected set")
        want_labels = ["label_elapsed_seconds"] if variant.startswith("e2s") else []
        if not set(v["label_columns_changed"]) <= set(want_labels):
            fail.append(f"user {user} {variant}: labels changed {v['label_columns_changed']}")
        if variant.startswith("e2s"):
            ls = LABEL_COLUMNS.index("label_elapsed_seconds")
            if (gl[q, ls] != gl0[q, ls]).any():
                fail.append(f"user {user} {variant}: query rows' label_elapsed_seconds changed")
            moved = gl[~q, ls] != gl0[~q, ls]
            v["curve_query_times_moved"] = int(moved.sum())
            v["curve_query_time_max_shortening_s"] = float((gl0[~q, ls] - gl[~q, ls]).max()) if moved.any() else 0.0
            if (gl[~q, ls] > gl0[~q, ls]).any():
                fail.append(f"user {user} {variant}: a curve query time got longer")
        if variant in ("passfail", "passfail-notime"):
            onehot = cf0[:, 9:13]
            hard_easy = ~q & ((onehot[:, 1] == 1) | (onehot[:, 3] == 1))
            rows_changed = ch[:, 9:13].any(1)
            v["hard_easy_update_rows"] = int(hard_easy.sum())
            if not np.array_equal(rows_changed, hard_easy) or not (cf[~q][:, 9:13].sum(1) == 1).all() \
                    or not np.array_equal(cf[:, 9], cf0[:, 9]):
                fail.append(f"user {user} {variant}: rating one-hot changes are not exactly the Hard / Easy update rows")
    if "e2s-query" in got and "e2s-all" in got:
        a, e = got["e2s-query"], got["e2s-all"]
        ls = LABEL_COLUMNS.index("label_elapsed_seconds")
        same_curve = bool(torch.equal(a.global_labels[:, ls], e.global_labels[:, ls]))
        same_now = bool(torch.equal(a.card_features[q][:, 2:5], e.card_features[q][:, 2:5]))
        rec["e2s_query_vs_all"] = {"curve_query_times_equal": same_curve, "query_rows_current_interval_equal": same_now}
        if not same_curve or not same_now:
            fail.append(f"user {user}: e2s-query and e2s-all disagree on the show-time query interval ({rec['e2s_query_vs_all']})")
    return rec, fail


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--users", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = cc.data_path()
    recs, fails = [], []
    for user in a.users:
        rec, fail = check_user(data, user)
        recs.append(rec)
        fails += fail
        print(f"[sm31-check] user {user}: {rec['rows']} rows, {rec['query_rows']} query rows, "
              f"{'OK' if not fail else 'FAIL'}", flush=True)
    from scripts.experiment_provenance import process_provenance
    out = {"ok": not fails, "failures": fails, "users": recs, "provenance": process_provenance()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"[sm31-check] {'OK' if not fails else 'FAIL'}: {len(recs)} users; " + "; ".join(fails[:5]))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
