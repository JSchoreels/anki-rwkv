"""Codebook state compression sweep (SM90): held-out coding error of WKV component 0 for the secondary knobs
(codebook size 8 / 10 / 12 bits per stage, 200k vs 600k fitting points, 20 vs 50 Lloyd iterations), proxy only.
Error = relative squared error of rank-1 heads (truncation removed: component 0 coded, exact norm) on the held-out
corpus users, per stage count; reported against bits (stages x index bits) and search cost (codewords x 32 MAC).
writes artifacts/small-model/sm90/knobs-sm90.json"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxy_sm90 import held_out, D  # noqa: E402
from scripts import state_vq as sv  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
FILES = {"b10 200k i20 (main fit)": ("codebooks-sm90.npz", 10), "b8 200k i20": ("knob-b8.npz", 8),
         "b12 200k i20": ("knob-b12.npz", 12), "b10 600k i20": ("knob-p600.npz", 10), "b10 200k i50": ("knob-i50.npz", 10)}


def main():
    _, A, _ = held_out(10, 8000)
    X = torch.cat([A["card"], A["note"]])
    exact = sv.StateCodec(sv.StreamRecipe(rank=1, stages=(0,), norm_bits=None, shift_m=0), sv.Codebooks()).wkv(X[:, None])[:, 0]
    rep = {}
    for name, (f, bits) in FILES.items():
        if not (D / f).exists():
            print(f"[knobs] {name}: {f} missing")
            continue
        books = sv.Codebooks.load(D / f)
        n = len(books.wkv[0]) if name != "b10 200k i20 (main fit)" else 10
        rows = []
        for s in range(1, n + 1):
            rec = sv.StreamRecipe(rank=1, stages=(s,), index_bits=bits, norm_bits=None, shift_m=0)
            out = sv.StateCodec(rec, books).wkv(X[:, None])[:, 0]
            e = float(((exact.double() - out.double()) ** 2).sum() / (exact.double() ** 2).sum())
            rows.append({"stages": s, "bits": s * bits, "search_macs": s * (2 ** bits) * 32, "rel_err_vs_rank1": e})
        rep[name] = rows
        print(f"[knobs] {name}: " + ", ".join(f"{r['bits']}b:{r['rel_err_vs_rank1']:.3e}" for r in rows), flush=True)
    (Path(__file__).resolve().parent / "knobs-sm90.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
