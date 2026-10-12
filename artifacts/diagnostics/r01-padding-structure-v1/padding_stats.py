"""CPU-only: real vs padded hierarchy positions per stream/split on real E33-shaped R01 batches."""
import json, os, sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
STEP = int(sys.argv[3]) if len(sys.argv) > 3 else 8
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
cohort = json.loads((ROOT / "artifacts/retraining/gpu-retrain-r01/cohorts-user-holdout-v1.json").read_text(encoding="utf-8"))
os.environ["GPU_BATCH1_DATA_PATH"] = str(ROOT / cohort["dataset_path"])
import torch
torch.set_num_threads(4)
from rwkv.data_processing import get_rwkv_data
from scripts.gpu_batch1 import split_frame
from scripts.gpu_b01 import prepare_job

protocol = json.loads((ROOT / "artifacts/retraining/gpu-retrain-r01/iteration-0033/protocol.json").read_text(encoding="utf-8"))
rows = protocol["rows"]
options = {"prefetch_users": True, "gpu_packed_features": True, "fast_samples": True, "columnar_samples": True, "batch_chunks": 512}
span = 512 * 128
jobs = []
for row in rows:
    length = row["train_range"][1] - row["train_range"][0]
    for offset in range(0, length, span):
        jobs.append((row, offset, min(length, offset + span)))
picked = jobs[::STEP]
names = ["card", "deck", "note", "preset", "global"]
result = {"jobs_total": len(jobs), "picked": len(picked), "updates": []}
tot = {n: [0, 0, 0, 0] for n in names}  # padded, real, splits, rows
for row, offset, end in picked:
    frame = split_frame(get_rwkv_data(Path(os.environ["GPU_BATCH1_DATA_PATH"]), row["user_id"]), row, "train")
    item = prepare_job(row, offset, end, 2025 + row["user_id"] + offset, options, frame.iloc[offset:end])
    b = item.batch
    upd = {"user": row["user_id"], "reviews": end - offset, "streams": {}}
    for name, gathers, lens in zip(names, b.sub_gather, b.sub_gather_lens):
        splits = []
        for g, L in zip(gathers, lens):
            g = g.view(-1, L)
            real = int((g >= 0).sum())
            splits.append([int(L), int(g.size(0)), int(g.numel()), real])
            tot[name][0] += g.numel(); tot[name][1] += real; tot[name][2] += 1; tot[name][3] += g.size(0)
        upd["streams"][name] = splits
    upd["start_rows"] = int(b.start.size(0)) if hasattr(b, "start") and torch.is_tensor(b.start) else None
    result["updates"].append(upd)
    print(row["user_id"], end - offset, {n: len(upd["streams"][n]) for n in names}, flush=True)
result["totals"] = {n: {"padded": v[0], "real": v[1], "ratio": v[0] / max(v[1], 1), "splits": v[2], "rows": v[3]} for n, v in tot.items()}
OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
print(json.dumps(result["totals"], indent=1))
