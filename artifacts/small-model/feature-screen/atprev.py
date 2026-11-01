"""Add 'as of the card's previous review' versions of the user-level features (the curve is forecast there)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ORIG = Path(r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
for p in sorted(Path("fx").glob("*.npz")):
    z = dict(np.load(p))
    if "user_pass_ewma200_atprev" in z:
        continue
    card = pd.read_parquet(ORIG / "revlogs" / f"user_id={p.stem}", columns=["card_id"])["card_id"].to_numpy()
    idx = np.arange(len(card))
    prev = pd.Series(idx).groupby(card).shift(1).to_numpy()
    ok = ~np.isnan(prev); pi = np.where(ok, prev, 0).astype(int)
    for k in ("user_pass_ewma200", "user_pass_ewma5000", "user_logdur_ewma200", "session_position", "gap_any_review_s"):
        # value seen when the previous review was shown, plus that review's own outcome is known then; ewma_prev at
        # index pi excludes row pi, so use the value at pi + 1 (which includes it) when it exists
        nxt = np.minimum(pi + 1, len(card) - 1) if k.startswith("user_") else pi
        z[k + "_atprev"] = np.where(ok, z[k][nxt], np.nan)
    np.savez_compressed(p, **z)
print("ok")
