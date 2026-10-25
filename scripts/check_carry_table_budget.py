"""SM72 check (CPU, seconds): the budgeted carry table ("table_gpu_mib", scripts/state_carry_train.py) stores and
returns exactly what today's table does, on the device and in host memory.

Random commit/lookup sequences over every stream kind (int8 card/note rows, fp16 deck, fp32 precision, rank-1 packed),
with repeated entities per update (the latest window wins), go through three tables: today's (no budget), budgeted on
the device (budget large) and budgeted in host memory (budget 0). Every lookup must be bit-identical.

  .venv/Scripts/python.exe scripts/check_carry_table_budget.py [--device cuda]
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.state_carry_train import BATCH_STREAMS, CarryTable  # noqa: E402


def run(device, precision, rank1, seed):
    g = torch.Generator().manual_seed(seed)
    universe = {stream: np.arange(0, 400, 3 + k, dtype=np.int64) for k, stream in enumerate(BATCH_STREAMS)}
    tables = [CarryTable(precision), CarryTable(precision, 1 << 20), CarryTable(precision, 0)]
    for t in tables:
        t.reset(universe, device)
        if t.gpu_mib is not None:
            t.place(0 if t.gpu_mib else 1)
        if rank1:
            t.fp32_streams = {2}
    assert not tables[1].host and tables[2].host
    H, K, C = 4, 8, 32
    worst = 0.0
    for update in range(6):
        for stream in range(len(BATCH_STREAMS)):
            E = tables[0].size[stream]
            for layer in range(2):
                n = int(torch.randint(1, 3 * E, (1,), generator=g))
                codes = torch.randint(0, E, (n,), generator=g).to(device)
                windows = torch.randint(0, 4, (n,), generator=g).to(device)
                tm = torch.randn(n, C, generator=g).to(device)
                if rank1 and stream == 2:
                    S = (torch.randn(n, H, K, 1, generator=g) * torch.randn(n, H, 1, K, generator=g)).to(device)
                else:
                    S = torch.randn(n, H, K, K, generator=g).to(device)
                cm = None if stream == 4 else torch.randn(n, C, generator=g).to(device)
                half = n // 2
                records = [(codes[:half], windows[:half], (tm[:half], S[:half], None if cm is None else cm[:half])),
                           (codes[half:], windows[half:], (tm[half:], S[half:], None if cm is None else cm[half:]))]
                looks = []
                for t in tables:
                    t.commit(stream, layer, records, 4)
                    query = torch.randint(0, E, (50,), generator=torch.Generator().manual_seed(update * 100 + stream)).to(device)
                    looks.append(t.lookup(stream, layer, query, torch.float32))
                for other in looks[1:]:
                    for a, b in zip(looks[0], other):
                        if a is None:
                            assert b is None
                            continue
                        assert a.device == b.device and a.dtype == b.dtype, (a.device, b.device, a.dtype, b.dtype)
                        if not torch.equal(a, b):
                            worst = max(worst, float((a.float() - b.float()).abs().max()))
                            raise AssertionError(f"lookup differs (precision {precision}, stream {stream}, layer {layer})")
    host_bytes = tables[2].nbytes()
    assert host_bytes == tables[0].nbytes() and all(
        t.device.type == "cpu" for slot in tables[2].slots.values() for t in slot.values() if isinstance(t, torch.Tensor))
    return {"precision": precision, "rank1": rank1, "identical": True, "table_bytes": host_bytes}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="artifacts/small-model/sm72/check-carry-table-budget.json")
    args = parser.parse_args()
    rows = [run(args.device, p, r, 7 + i) for i, (p, r) in enumerate((("deploy", False), ("deploy", True), ("fp32", False)))]
    report = {"device": args.device, "cases": rows, "pass": all(r["identical"] for r in rows)}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
