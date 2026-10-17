"""Score fork and small-model checkpoints on the user's own collection (CPU only, aggregate output).

Wraps scripts/small_model_heldout.py unchanged: user 1 = the collection converted on 2026-10-14
(anki-rwkv/artifacts/collection-eval/2026-10-14/collection-dataset, read-only; 173,690 scored reviews from its
collection-label-db, the same set as the fork's 0.4417 / 0.4366 reference). The collection's review types come from the
user's fork (not SM-2): 806 learning rows, 34,458 type-4 rows. Env COLL_PATCH:
  none     (default)
  nosteps  history review types relabelled as FSRS without steps (../feature-screen/sm25.py)
  nostate  the fork's review-type input zeroed (features2card.0.weight[:, 22] = 0)
Same arguments as the scorer, e.g. --device cpu --model st=expertium:w64:<master.pth> --only st/w8+s8 --out <jsonl>.
Only aggregate metrics leave this script; never pass --dump here (per-review collection data stays private).
"""
import os
import sys
from pathlib import Path

WT = Path(r"C:\Users\jonat\workspace\wt\small-model")
COLL = Path(r"C:\Users\jonat\workspace\anki-rwkv\artifacts\collection-eval\2026-10-14")
sys.path.insert(0, str(WT))
assert "--dump" not in sys.argv, "per-review collection predictions stay out of this script"
PATCH = os.environ.get("COLL_PATCH", "none")

import lmdb  # noqa: E402
import torch  # noqa: E402

if PATCH == "nosteps":
    sys.path.insert(0, str(WT / "artifacts" / "small-model" / "feature-screen"))
    os.environ.setdefault("SM25_MODE", "nosteps")
    import sm25  # noqa: E402,F401  (patches rwkv.data_processing.base_transform_state at import)
elif PATCH == "nostate":
    _load = torch.load

    def _nostate(path, *a, **k):
        s = _load(path, *a, **k)
        if str(path).endswith("RWKV_trained_on_5000_10000.pth") and "features2card.0.weight" in s:
            w = s["features2card.0.weight"].clone(); w[:, 22] = 0.0; s = dict(s, **{"features2card.0.weight": w})
        return s
    torch.load = _nostate

from scripts import compress_common as cc  # noqa: E402
from scripts import small_model_heldout as sm  # noqa: E402

_run_info = cc.run_info
cc.run_info = lambda: dict(_run_info(), review_counts={"1": 223670}, dataset_path=str(COLL / "collection-dataset"))
cc.data_path = lambda run=None: COLL / "collection-dataset"
sm.heldout_users = lambda counts: [1]
_open = lmdb.open
lmdb.open = lambda path, *a, **k: _open(str(COLL / "collection-label-db") if str(path).endswith("label-db") else path, *a, **k)

if __name__ == "__main__":
    sm.main()
