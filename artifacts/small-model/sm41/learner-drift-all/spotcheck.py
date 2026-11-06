"""Spot-check of the fused run (SM41g confirmation): the fused rows of the spot-check users (learner-drift-all/spotcheck)
against (1) the existing no-drift rows (`check`) and (2) the 20-user run's separate improve / decline rows (same user and
split): the same lanes, every counter, largest relative gap."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts import sm41_learner_drift as LD   # noqa: E402

LD.ALL = True
LD.OUT = LD.HERE / "learner-drift-all" / "spotcheck"
LD.check(None)
for sc in ("improve", "decline"):
    old = LD.rows_by_key(str(LD.HERE / "learner-drift" / sc / "rows" / "*.jsonl"))
    new = LD.scen_rows(sc)
    worst, at, n = 0.0, "", 0
    for key, r in new.items():
        o = old.get(key)
        if o is None:
            continue
        for k, name in enumerate(r["lanes"]):
            if name not in o["lanes"]:
                continue
            j = o["lanes"].index(name)
            n += 1
            for key2 in o["acc"]:
                a, b = np.asarray(r["acc"][key2])[k], np.asarray(o["acc"][key2])[j]
                g = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1.0)))
                if g > worst:
                    worst, at = g, f"{key} {name} {key2}"
            for key2 in ("memorised_instant", "backlog"):
                g = abs(r[key2][k] - o[key2][j]) / max(abs(o[key2][j]), 1.0)
                if g > worst:
                    worst, at = g, f"{key} {name} {key2}"
    print(f"{sc} (fused view) vs the 20-user separate run: {n} lanes compared, largest relative gap {worst:.2e} {at}")
