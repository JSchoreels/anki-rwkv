"""Wrapper around scripts/state_vq_fit.py (unchanged) that points the student at the SM32b QAT checkpoint and the output
at qat/refit/ (corpus under qat/refit/corpus). PROTOCOL stays the original student's (same 50 training users).
usage: state_vq_fit_qat.py collect --users 50 --shard 0/2 | fit --shift-stages 3 --out codebooks-qat-refit.npz | refit-norms --codebooks ..."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts import state_vq_fit as f  # noqa: E402
QAT = ROOT / "artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay-qat378x412p/expertium/E89-F65e-F77a/master.pth"
f.STUDENT = QAT  # PROTOCOL was computed from the original STUDENT at import: unchanged
f.OUT = ROOT / "artifacts/small-model/state-vq/qat/refit"
f.main()
