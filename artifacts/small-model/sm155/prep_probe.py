"""Bounded-memory long-card scorer (SM155): peak memory of the batch preparation alone (get_rwkv_data + prepare, the
scorer's UserBatches) and of the stream engine's row layout, for one benchmark user. No model, no scoring.
usage: prep_probe.py USER"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import psutil  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)
from scripts import compress_quant as cq  # noqa: E402

cq.install_state_quant_model()
from scripts import compress_common as cc  # noqa: E402
from scripts import stream_scorer  # noqa: E402

proc = psutil.Process()
user = int(sys.argv[1])
t0 = time.time()
batch = cc.full_history_batch(cc.data_path(), user)
rss_batch = proc.memory_info().rss
layout = stream_scorer.Layout(batch)
rss_layout = proc.memory_info().rss
ents = [s["entities"] for s in layout.streams]
print(f"user {user}: rows {layout.n}, entities {ents}, after batch {rss_batch / 2**30:.2f} GB, after layout "
      f"{rss_layout / 2**30:.2f} GB (peak working set {proc.memory_info().peak_wset / 2**30:.2f} GB), "
      f"{time.time() - t0:.0f} s", flush=True)
