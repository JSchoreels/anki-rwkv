#!/usr/bin/env python
"""SM30: review-number flag inputs for the student (docs/SMALL_MODEL.md "Review-number flag inputs (SM30)").

Three card-level binary inputs per row: the card's 2nd review, 3rd review, 4th-or-later review (the 1st review is
already marked by elapsed time -1). They come from the frame's "i" column (the card's review number, counted by
rwkv.data_processing.get_rwkv_data over the user's FULL history), looked up by review_th, so a 2,048-review training
window starting mid-history carries the true numbers. Query rows keep "i" in add_queries (it is in the keep list),
so a query row carries the flags of the review it predicts: that number is known at the forecast point.

Layout. Opt-in only (track option "rep_flags": true, architecture field rep_flags). The samples get the 3 columns
after the 24 card features, so every preparation path (columnar samples, packed IPC, gpu_packed_features, batched id
encodings) carries them without change and the 68 encoding columns follow: pipeline rows are
[24 card | 3 flags | 68 encodings] = 95. The student reorders them to its inputs [the 92 | 3 flags]; its first layer
is the 92-input Linear plus a bias-free float32 [fc, 3] weight on the flags (`features2card_rep_flags_weight`), i.e.
one 95-input layer split in two, so zero flag weights leave every output bit-identical. The teacher gets `strip(start)`, exactly
its 92 inputs.

  python scripts/rep_flags.py widen --in <92-input master.pth> --out <95-input master.pth>
"""
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CARD_COLUMNS = 24          # len(rwkv.data_processing.CARD_FEATURE_COLUMNS)
FLAGS = 3                  # 2nd review, 3rd review, 4th-or-later review
FLAG_SLICE = slice(CARD_COLUMNS, CARD_COLUMNS + FLAGS)
MODEL_INPUTS = 92
PARAMETER = "features2card_rep_flags_weight"


def flags_from_numbers(numbers):
    """numbers = the card's review number (1 = first review) -> float32 [N, 3] (i == 2, i == 3, i >= 4)."""
    numbers = np.asarray(numbers)
    return np.stack([numbers == 2, numbers == 3, numbers >= 4], -1).astype(np.float32)


def append_to_samples(samples, section):
    """Append the flag columns to every RWKVSample built from `section` (a contiguous slice of the full-history
    frame of get_rwkv_data, so review_th runs without gaps). Rows are matched by review_th; query rows share the
    review_th (and "i") of the review they predict."""
    review_ths = section["review_th"].to_numpy()
    numbers = section["i"].to_numpy()
    if len(review_ths) and not np.array_equal(review_ths, np.arange(review_ths[0], review_ths[0] + len(review_ths))):
        raise ValueError("rep_flags needs a contiguous history slice (review_th without gaps)")
    first = int(review_ths[0]) if len(review_ths) else 0
    for sample in samples:
        index = sample.review_ths.numpy().astype(np.int64) - first
        if len(index) and (index.min() < 0 or index.max() >= len(numbers)):
            raise ValueError("sample rows outside the history slice")
        flags = torch.from_numpy(flags_from_numbers(numbers[index])).to(sample.card_features.dtype)
        sample.card_features = torch.cat([sample.card_features, flags.to(sample.card_features.device)], 1)
    return samples


def strip(start):
    """Pipeline rows [24 card | 3 flags | 68 encodings] -> the 92 model inputs, values unchanged."""
    return torch.cat([start[..., :CARD_COLUMNS], start[..., CARD_COLUMNS + FLAGS:]], -1)


def strip_batch(batch):
    return replace(batch, start=strip(batch.start))


def widen_state(state, fc_dim=None):
    """A 92-input state dict -> the rep-flags model's: adds the flag weight [fc, 3] at exactly zero. Returns
    (state, widened?). A state that already has it is returned unchanged."""
    if PARAMETER in state:
        return state, False
    first = state["features2card.0.weight"]
    if first.shape[1] != MODEL_INPUTS:
        raise ValueError(f"features2card.0.weight has {first.shape[1]} inputs, expected {MODEL_INPUTS}")
    out = dict(state)
    out[PARAMETER] = torch.zeros(first.shape[0] if fc_dim is None else fc_dim, FLAGS, dtype=first.dtype)
    return out, True


def widen_optimizer_state(saved, optimizer, master):
    """An optimizer state saved for the 92-input model -> the same state for `optimizer` over the rep-flags
    `master` (identical groups plus the flag weight). Every existing parameter keeps its moments and step; the flag
    weight gets no state, so AdamW starts it fresh (step 0, its own bias correction)."""
    names = {id(p): n for n, p in master.named_parameters()}
    new_groups = [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]
    old_groups = [[n for n in group if n != PARAMETER] for group in new_groups]
    if [len(g["params"]) for g in saved["param_groups"]] != [len(g) for g in old_groups]:
        raise ValueError("saved optimizer groups do not match the 92-input layout of this model")
    old_index, new_index, counter = {}, {}, 0
    for group in old_groups:
        for name in group:
            old_index[name], counter = counter, counter + 1
    counter = 0
    for group in new_groups:
        for name in group:
            new_index[name], counter = counter, counter + 1
    parameters = dict(master.named_parameters())
    state = {}
    for name, index in old_index.items():
        if index in saved["state"]:
            entry = saved["state"][index]
            for key in ("exp_avg", "exp_avg_sq"):
                if key in entry and entry[key].shape != parameters[name].shape:
                    raise ValueError(f"{name}: saved {key} {tuple(entry[key].shape)} != {tuple(parameters[name].shape)}")
            state[new_index[name]] = entry
    groups = [dict(saved_group, params=[new_index[n] for n in names_]) for saved_group, names_ in
              zip(saved["param_groups"], new_groups)]
    return {"state": state, "param_groups": groups}


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    widen = sub.add_parser("widen", help="92-input master.pth -> rep-flags master.pth (flag weights exactly zero)")
    widen.add_argument("--in", dest="source", type=Path, required=True)
    widen.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    state = torch.load(args.source, map_location="cpu", weights_only=True)
    state, changed = widen_state(state)
    if not changed:
        raise SystemExit(f"{args.source} already has {PARAMETER}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.out)
    print(f"[rep-flags] {args.source} -> {args.out}: added {PARAMETER} {tuple(state[PARAMETER].shape)} = 0")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
