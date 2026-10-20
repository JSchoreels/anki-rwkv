"""Diagnostic: record every codeword pick of the +vq scorer path (torch loop and/or fused kernel) on one device.
usage: python vqdiag.py WT DEVICE CASE USER MODE OUT
  MODE torch: torch step loop for every stream (rank-1 with factor "sq" = the kernel's math; rank-2 eigh)
  MODE fused: the real scoring path (fused kernel picks recorded with record=True)
"""
import os
import sys
import time
from pathlib import Path
os.environ.setdefault("RWKV_NO_JIT", "1")
WT, DEVICE, CASE, USER, MODE, OUT = sys.argv[1:7]
USER = int(USER)
sys.path.insert(0, WT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scripts import check_state_vq_score_gpu as chk  # noqa: E402
from scripts import state_vq  # noqa: E402
from scripts import compress_common as cc  # noqa: E402

device = torch.device(DEVICE)
if device.type == "cuda":
    from scripts import compress_wkv_cuda
    compress_wkv_cuda.register_cuda()
    torch.cuda.set_per_process_memory_fraction(0.8)
else:
    torch.set_num_threads(4)
if os.environ.get("CASES_JSON"):
    import json as _json
    chk.CASES = [tuple(c) for c in _json.load(open(os.environ["CASES_JSON"]))]
case = [c for c in chk.CASES if c[0] == CASE][0]
_, ckpt, books_path, name, _ = case
model = chk.load_model(ckpt, device)
layout = {"slots": state_vq.install_scoring(model)}
books = state_vq.Codebooks.load(books_path)
codecs = state_vq.score_codecs(name, books, device, torch.float64)
if MODE == "torch":
    for c in codecs.values():
        c.score_kernel = None

LOG = []          # (kind, slot-ish tag, x float32 [N, D] or picks, picks)
CUR = {"tag": None}
orig_rvq = state_vq.rvq


def rvq_logged(x, books_, norms2=None, chunk=4096):
    acc, picks = state_vq.rvq_picks(x, books_, norms2, chunk)
    LOG.append(("rvq", CUR["tag"], x.detach().float().cpu().numpy(), picks.cpu().numpy().astype(np.int16)))
    return acc


state_vq.rvq = rvq_logged
WLOG = []
_orig_codec_wkv = state_vq.StateCodec.wkv


def codec_wkv_logged(self, S_):
    out_ = _orig_codec_wkv(self, S_)
    WLOG.append((S_.detach().cpu().numpy(), out_.detach().cpu().numpy()))
    return out_


if os.environ.get("LOG_WKV"):
    state_vq.StateCodec.wkv = codec_wkv_logged
orig_wkv, orig_shift = state_vq.score_wkv, state_vq.score_shift


def wkv_tag(*a):
    CUR["tag"] = f"wkv{a[-1]}"
    if MODE == "fused" and state_vq.SCORE[a[-1]].score_kernel is not None:
        kc = state_vq.SCORE[a[-1]].score_kernel
        r, k, v, w, a_, kd = (x.float().contiguous() for x in a[:6])
        skip = a[6].bool().contiguous()
        out, before, final, picks, cscale = kc.forward(r, k, v, w, a_, kd, skip, record=True)
        LOG.append(("kernel", CUR["tag"], None, picks.cpu().numpy().astype(np.int16)))
        return out.to(a[1].dtype)
    return orig_wkv(*a)


def shift_tag(x, sel, slot, role):
    CUR["tag"] = f"shift{slot}/{role}"
    return orig_shift(x, sel, slot, role)


from expertium.model import rwkv_model as erm  # noqa: E402
erm.vq_wkv, erm.vq_shift = wkv_tag, shift_tag
state_vq.set_scoring(layout["slots"], codecs)

import lmdb  # noqa: E402
from scripts.compress_metrics import read_labels  # noqa: E402
env = lmdb.open(str(cc.RUN / "label-db"), readonly=True, lock=False, map_size=10_000_000_000)
with env.begin() as txn:
    labels = read_labels(txn, USER)
batch = cc.full_history_batch(cc.data_path(cc.run_info()), USER).to(device)
t0 = time.time()
values, preds, bench = chk.evaluate(model, batch, labels, USER)
print("seconds", round(time.time() - t0, 1), "calls", len(LOG))
ref = chk.load_ref(Path(os.environ.get("REF_DIR", str(chk.OUT / "check-ref"))), CASE, USER)

cand = {"values": values, "preds": preds, "bench": bench, "exact": chk.exact_logloss(labels, preds)}
import pickle
pickle.dump(cand, open(OUT + ".cand.pkl", "wb"))
c = chk.compare(ref, cand, 1e-5)
print("vs CPU ref:", {h: (c[h]["d_exact_logloss"], c[h]["max_abs_dp"], c[h]["n_dp_over_1e-4"]) for h in ("ahead", "instant")},
      "ok", c["ok"])
arrays = {}
for i, (kind, tag, x, picks) in enumerate(LOG):
    arrays[f"{i}/{kind}/{tag}/picks"] = picks
    if x is not None:
        arrays[f"{i}/{kind}/{tag}/x"] = x
np.savez(OUT, **arrays)
if WLOG:
    np.savez(OUT + ".wkv.npz", **{f"{i}/{w}": a for i, (x, y) in enumerate(WLOG) for w, a in (("in", x), ("out", y))})
print("saved", OUT)
