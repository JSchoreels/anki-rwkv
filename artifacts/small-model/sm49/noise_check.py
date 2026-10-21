"""SM49 diagnostic (CPU): how much do the int state variants move when every value entering the int rounding (WKV state
and stored token-shift vectors) gets a 1-ulp-scale relative perturbation (x * (1 + e), e = +-2^-23, random sign, seeded)?
The CUDA path differs from the CPU one by float reduction order only (~1e-7 relative), so this measures each format's
sensitivity to that noise through rounding flips, without a GPU. Rows go to noise-cpu.jsonl; compare with check-cpu.jsonl.
usage: noise_check.py [scorer args]  (same as scripts/small_model_heldout.py --device cpu ...)"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, "C:/Users/jonat/workspace/wt/sm49")
from scripts import small_model_heldout as sm  # noqa: E402
from scripts import state_vq  # noqa: E402

_quant = state_vq.int_row_quant
GEN = torch.Generator().manual_seed(0)


def int_row_quant(bits):
    q = _quant(bits)

    def noisy(x):
        sign = torch.randint(0, 2, x.shape, generator=GEN).to(x.dtype) * 2 - 1
        return q(x * (1 + sign * 2.0 ** -23))
    return noisy


state_vq.int_row_quant = int_row_quant
assert Path(sys.argv[sys.argv.index("--out") + 1]).name.startswith("noise"), "noise rows only"
sm.main()
