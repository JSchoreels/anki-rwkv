"""Codebook state compression sweep (SM90): pick the WKV books, 200k vs 900k fitting points (pre-registered in
DESIGN.md "Fitting-data knob"): held-out relative squared WKV error of every ladder recipe's coded stream; the 900k
books win if they are lower for both 1x recipes (vqC5902 card heads and vqN3020 note heads).
usage: choose_books.py codebooks-sm90-200k.npz codebooks-big.npz -> books-choice.json; prints the winner's file name"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import torch  # noqa: E402
from dataclasses import replace  # noqa: E402

from proxy_sm90 import held_out, wkv_err, D, HERE  # noqa: E402
from scripts import state_vq as sv  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
LADDER = ["vqC5902", "vqC4430", "vqC2958", "vqC1934", "vqC1470", "vqN3964", "vqN3020", "vqN1524"]


def main():
    a_name, b_name = sys.argv[1], sys.argv[2]
    _, A, _ = held_out(10, 8000)
    books = {a_name: sv.Codebooks.load(D / a_name), b_name: sv.Codebooks.load(D / b_name)}
    rep = {}
    for r in LADDER:
        stream = "card" if r.startswith("vqC") else "note"
        rec = replace(sv.recipe(r)[0 if stream == "card" else 1], shift_m=0)
        rep[r] = {name: wkv_err(A[stream], rec, b) for name, b in books.items()}
        print(f"[books] {r} {stream}: " + ", ".join(f"{k} {v:.3e}" for k, v in rep[r].items()), flush=True)
    win = b_name if all(rep[r][b_name] < rep[r][a_name] for r in ("vqC5902", "vqN3020")) else a_name
    (HERE / "books-choice.json").write_text(json.dumps({"errors": rep, "winner": win}, indent=1))
    print(win)


if __name__ == "__main__":
    main()
