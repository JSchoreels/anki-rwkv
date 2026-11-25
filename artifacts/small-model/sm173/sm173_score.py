#!/usr/bin/env python
"""Deleted-card note ids (SM173), phase 1b: the held-out scorer of the run tree (scripts/small_model_heldout.py, run tree
C:/Users/jonat/workspace/wt/sm173-run = e8ab44d4, detached, never edited) with the deleted cards' placeholder note ids
changed at inference. The arm is chosen by the environment variable SM173_ARM (inherited by the data-loader workers,
which re-import this file, so the patch is installed at import time):

  cur   : no patch (today's inputs: the float64 placeholder ID_PLACEHOLDER + card_id rounded to a multiple of 64, so
          deleted cards in one 64-wide card-id block share a note lane; every placeholder becomes -2^31 in the int32
          id tensor, so all deleted cards of the user share one note code).
  fix   : the fix: the placeholder note id built in int64 (ID_PLACEHOLDER + card_id exact, the note_id column int64):
          one note lane and one note code per deleted card. Changing the set of note codes re-draws the user's id / day
          codes (seed 2025 + user), a reseed for users with deleted cards.
  lanes : only the note lanes made unique per deleted card (_module_group_locations groups placeholder rows by card);
          the id tensor and so every random code are exactly today's (no reseed). The clean memory-mixing effect.
Users without deleted cards get bitwise the same inputs in every arm.

  cd C:/Users/jonat/workspace/wt/sm173-run
  SM173_ARM=fix python -X utf8 <this file> <small_model_heldout.py arguments>
"""
import os
from pathlib import Path
import sys

import numpy as np

CODE = Path(os.environ.get("SM173_CODE", "C:/Users/jonat/workspace/wt/sm173-run")).resolve()
sys.path.insert(0, str(CODE))
ARM = os.environ.get("SM173_ARM", "")
if ARM not in ("cur", "fix", "lanes"):
    raise SystemExit("set SM173_ARM to cur, fix or lanes")
if os.environ.get("RWKV_DELETED_CARDS"):
    raise SystemExit("unset RWKV_DELETED_CARDS: the scorer applies --deleted-cards itself (the fix arm needs the flags)")
PLACEHOLDER_MIN = 1e17   # placeholder note ids are about 3.14e17; the dataset's real note ids are small integers

from rwkv import data_processing as dp  # noqa: E402

if ARM == "fix":
    _get = dp.get_rwkv_data

    def get_rwkv_data(data_path, user_id, equalize_review_ths=[]):
        frame = _get(data_path, user_id, equalize_review_ths)
        missing = frame["note_id_is_nan"].to_numpy() == 1   # set exactly for cards missing from the cards table
        if missing.any():
            real = frame["note_id"].to_numpy()
            if (real[~missing] >= PLACEHOLDER_MIN).any() or (real[~missing] != np.round(real[~missing])).any():
                raise RuntimeError("sm173 fix: unexpected real note id")
            card = frame["card_id"].to_numpy().astype(np.int64)
            frame["note_id"] = np.where(missing, np.int64(dp.ID_PLACEHOLDER) + card, real.astype(np.int64))
        return frame

    dp.get_rwkv_data = get_rwkv_data   # sm31_inputs.load_frame / the scorer look it up at call time

if ARM == "lanes":
    import pandas as pd
    _groups = dp._module_group_locations

    def _module_group_locations(section_df, submodule):
        if submodule != "note_id":
            return _groups(section_df, submodule)
        values = section_df["note_id"].to_numpy()
        placeholder = values >= PLACEHOLDER_MIN
        if not placeholder.any():
            return _groups(section_df, submodule)
        key = np.where(placeholder, -1.0 - section_df["card_id"].to_numpy().astype(np.float64), values)
        return _groups(pd.DataFrame({"note_id": key, "index": section_df["index"].to_numpy()}), "note_id")

    dp._module_group_locations = _module_group_locations   # create_sample looks it up at call time

if __name__ == "__main__":
    os.chdir(CODE)
    sys.argv[0] = str(CODE / "scripts" / "small_model_heldout.py")
    from scripts import small_model_heldout
    print(f"[sm173] arm {ARM}, code tree {CODE}", flush=True)
    sys.exit(small_model_heldout.main())
