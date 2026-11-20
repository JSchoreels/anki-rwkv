#!/usr/bin/env python
"""SM136 CPU run: users of the main pool and the extension ordered by review count (smallest first) -> cpu/order.json
(user ids and revlog row counts only, public benchmark users). CPU, about 1 min."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from scripts import compress_common as cc  # noqa: E402
from scripts.small_model_heldout import card_prior_answers  # noqa: E402

data = cc.data_path(cc.run_info())
out = {}
for name, f in (("main", "users-sm136.json"), ("ext", "users-sm136-ext.json")):
    users = json.loads((HERE / f).read_text())["users"]
    out[name] = sorted(((int(len(card_prior_answers(data, u))), u) for u in users))
(HERE / "cpu" / "order.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
print({k: (v[0], v[len(v) // 2], v[-1]) for k, v in out.items()})
