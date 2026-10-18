"""SM26 on the scripted (TorchScript) model, the training path: setting instant_loss_weight works and weight 1.0 == absent;
weight 2.0 adds exactly one more instant average. (check_instant_loss_weight.py runs eager, RWKV_NO_JIT=1, and missed
the 2026-10-18 04:19 failure: a None attribute cannot be set to a float on a ScriptModule.)"""
import os, sys
from pathlib import Path
ROOT = Path(r"C:\Users\jonat\workspace\wt\small-model"); sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
os.environ["GPU_BATCH1_DATA_PATH"] = r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"
os.environ.pop("RWKV_NO_JIT", None)
import torch; torch.set_num_threads(4)
from rwkv.data_processing import get_rwkv_data
from rwkv.prepare_batch import prepare
from scripts.gpu_batch1 import make_samples
from expertium import variants
from expertium.model.variant_model import VariantSrsRWKV
frame = get_rwkv_data(Path(os.environ["GPU_BATCH1_DATA_PATH"]), 7)
batch = prepare(make_samples(frame.iloc[:640], 7, 128), seed=2032)
out = {}
for w in (None, 1.0, 2.0):
    torch.manual_seed(2025); m = VariantSrsRWKV(variants.variant("w64"))
    print(type(m).__mro__[1].__name__, "scripted" if isinstance(m, torch.jit.ScriptModule) else "eager")
    if w is not None: m.instant_loss_weight = w
    m.train(); torch.manual_seed(11); s = m.get_loss(batch); out[w] = (s.average_loss.item(), s.imm_avg.item())
print(out)
ok = out[None][0] == out[1.0][0] and abs(out[2.0][0] - out[1.0][0] - out[1.0][1]) < 1e-5
print("JIT OK" if ok else "JIT FAIL")
