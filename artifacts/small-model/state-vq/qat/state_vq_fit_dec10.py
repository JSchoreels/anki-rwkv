"""Wrapper around scripts/state_vq_fit.py (unchanged) that points the student at dec10 itself (books fitted on its own states) and the output
at dec10fit/ (corpus under dec10fit/corpus). PROTOCOL stays the original student's (same 50 training users).
usage: state_vq_fit_dec10.py collect --users 50 --shard 0/2 | fit --shift-stages 3 --out codebooks-dec10.npz | refit-norms --codebooks ..."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts import state_vq_fit as f  # noqa: E402
DEC10 = Path("C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay/expertium/E89-F65e-F77a/master.pth")
f.STUDENT = DEC10  # PROTOCOL was computed from the original STUDENT at import: unchanged
f.OUT = ROOT / "artifacts/small-model/state-vq/dec10fit"
f.main()
