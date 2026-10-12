"""CUDA check: F26 shift-gather backward equals deterministic PyTorch bit for bit."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Use a new workspace artifact directory")
    cohort = json.loads((ROOT / "artifacts/retraining/gpu-retrain-r01/cohorts-user-holdout-v1.json").read_text(encoding="utf-8"))
    os.environ["GPU_BATCH1_DATA_PATH"] = str(ROOT / cohort["dataset_path"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    from rwkv.data_processing import get_rwkv_data
    from rwkv.model.exact_index import ExactShiftGather, shift_backward
    from scripts.experiment_provenance import process_provenance
    from scripts.gpu_batch1 import split_frame
    from scripts.gpu_b01 import prepare_job, transfer_batches
    options = {"prefetch_users": True, "preparation_workers": 8, "gpu_packed_features": True, "packed_ipc": True}
    rows = sorted(cohort["stages"]["100"]["rows"], key=lambda row: row["available_reviews"])
    picks = [(rows[-1], 0, 32768), (rows[-3], 32768, 65536), (rows[len(rows) // 2], 0, 32768), (rows[0], 0, 32768), (rows[10], 0, 32768)]
    jobs = []
    for row, offset, end in picks:
        frame = split_frame(get_rwkv_data(Path(os.environ["GPU_BATCH1_DATA_PATH"]), row["user_id"]), row, "train")
        end = min(end, len(frame))
        jobs.append((row, prepare_job(row, offset, end, 2025 + row["user_id"] + offset, options, frame.iloc[offset:end])))
    result = {"provenance": process_provenance(), "torch": torch.__version__, "cases": [], "status": "running",
              "scope": "real Expertium time-shift indices; random BF16/FP32 gradients with zeros and signed zeros; compared with deterministic torch.gather backward"}
    generator = torch.Generator(device="cuda").manual_seed(26)
    try:
        for (row, _), (_, item, batch) in zip(jobs, transfer_batches(iter(jobs), torch.bfloat16, False)):
            for module, (selects, lengths) in enumerate(zip(batch.time_shift_selects, batch.sub_gather_lens)):
                for select, length in zip(selects, lengths):
                    index = select.view(-1, length).long()
                    B, T = index.shape
                    counts = torch.bincount((torch.arange(B, device=index.device)[:, None] * T + index).reshape(-1).cpu())
                    for dtype in (torch.bfloat16, torch.float32):
                        x = torch.randn((B, T, 32), device="cuda", dtype=dtype, generator=generator).requires_grad_()
                        grad = torch.randn((B, T, 32), device="cuda", dtype=dtype, generator=generator)
                        grad[torch.rand(grad.shape, device="cuda", generator=generator) < 0.2] = 0.0
                        grad[torch.rand(grad.shape, device="cuda", generator=generator) < 0.05] = -0.0
                        torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, 32)).backward(grad)
                        reference = x.grad
                        mine = shift_backward(grad, index)
                        bits = torch.int16 if dtype == torch.bfloat16 else torch.int32
                        x2 = x.detach().clone().requires_grad_()
                        ExactShiftGather.apply(x2, index).backward(grad)
                        result["cases"].append({"user_id": row["user_id"], "module": module, "B": B, "T": T, "dtype": str(dtype),
                                                "max_sources_per_position": int(counts.max()),
                                                "bitwise_equal": bool(torch.equal(reference.view(bits), mine.view(bits))),
                                                "autograd_bitwise_equal": bool(torch.equal(reference.view(bits), x2.grad.view(bits)))})
        cases = result["cases"]
        result["status"] = "passed" if cases and all(c["bitwise_equal"] and c["autograd_bitwise_equal"] for c in cases) else "failed"
        result["summary"] = {"cases": len(cases), "equal": sum(c["bitwise_equal"] for c in cases),
                             "cases_with_32_or_more_sources": sum(c["max_sources_per_position"] >= 32 for c in cases),
                             "max_sources": max(c["max_sources_per_position"] for c in cases)}
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        output.mkdir(parents=True)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "summary": result.get("summary"), "error": result.get("error")}))


if __name__ == "__main__":
    main()
