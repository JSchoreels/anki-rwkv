"""SM58: per-review predictions of one checkpoint on the user's own collection (full history as one sequence, float32,
CPU only, 2 threads, below-normal priority). Output: a parquet (review_th, rating, p_curve, p_instant) under
collection-cache/ (NEVER committed: per-card rows of the user's own collection; only the aggregate summary is).

  python sm58_collection.py --arch w64 --checkpoint master.pth --tag sm36
The big model's predictions are reused read-only from anki-rwkv/artifacts/collection-eval/2026-10-14/preds-model-B-full-s0.parquet.
Inputs are the same prepared batch as score_collection.py (prepared-full-s0.pt) in the main checkout.
"""
import argparse
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--arch", default="w64")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--tag", required=True)
parser.add_argument("--threads", type=int, default=2)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[n] = str(min(2, args.threads))
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MAIN_EVAL = Path(r"C:\Users\jonat\workspace\anki-rwkv\artifacts\collection-eval\2026-10-14")
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(min(2, args.threads))
assert not torch.cuda.is_available()
if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
from expertium.variants import build  # noqa: E402
from scripts.full_run_metrics import evaluate_with_predictions  # noqa: E402

out = HERE / "collection-cache" / f"preds-{args.tag}-full-s0.parquet"
out.parent.mkdir(exist_ok=True)
if out.exists():
    print(f"{out.name} exists; nothing to do")
    sys.exit(0)
batches = torch.load(MAIN_EVAL / "prepared-full-s0.pt", weights_only=False)
model = build(args.arch).float()
model.load_state_dict({k: v.float() for k, v in torch.load(args.checkpoint, map_location="cpu", weights_only=True).items()},
                      strict=True)
n_ref = 0
for module in model.modules():
    if hasattr(module, "use_reference_wkv"):
        module.use_reference_wkv = True
        n_ref += 1
# expertium mixers pick reference_rwkv7 on CPU tensors by themselves (no use_reference_wkv attribute)
model.eval()
t0 = time.perf_counter()
with torch.inference_mode():
    _, predictions = evaluate_with_predictions(model, ({"batch": b} for b in batches))
seconds = time.perf_counter() - t0
ths = sorted(predictions["rating"])
pd.DataFrame({"review_th": ths, "rating": [predictions["rating"][t] for t in ths],
              "p_curve": [predictions["ahead"].get(t) for t in ths],
              "p_instant": [predictions["instant"].get(t) for t in ths]}).to_parquet(out)
print(f"scored {args.tag}: {len(ths)} reviews, {seconds:.0f} s model time -> {out.name}", flush=True)
