"""Audit actual synthetic updates/checkpoints; no GPU launches or quality claims."""

import hashlib
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance


def main():
    provenance = process_provenance()
    import torch
    torch.set_num_threads(1)
    cases = ("integration-expertium-joint-v2", "integration-expertium-curve", "integration-expertium-instant",
             "integration-baseline-width-cosine", "integration-baseline-depth-wsd", "integration-target-stop",
             "integration-final-confirmations", "integration-final-joint-v2")
    rows = []
    for name in cases:
        folder = ROOT / "artifacts/benchmarks/gpu-future" / name
        result = json.loads((folder / "result.json").read_text())
        assert result["status"] == "complete" and result["config"]["synthetic_only"]
        expected = 1 if name == "integration-target-stop" else 2
        assert result["updates"] == expected and result["raw_review_presentations"] == 2 * expected
        assert result["maximum_supported_head_weight_change"] > 0
        assert all(math.isfinite(result[key]) and result[key] > 0 for key in ("training_seconds", "elapsed_seconds", "reviews_per_second"))
        task = result["config"]["task"]
        saved = result["provenance"]
        with ZipFile(ROOT / saved["source_archive"]) as archive:
            assert hashlib.sha256(archive.read("source-manifest.json")).hexdigest() == saved["source_tree_sha256"]
        for entry in result["history"]:
            state = torch.load(folder / f"checkpoint-{entry['update']:06d}.pth", weights_only=True, map_location="cpu")
            assert sum(value.numel() for value in state.values()) == result["model"]["parameters"]
            assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in state.values())
            for metric in ("curve", "instant"):
                supported = task in ("joint", metric)
                assert entry["metrics"][metric + "_evaluated_users"] == (1 if supported else 0)
                for aggregation in ("label_weighted", "user_average"):
                    value = entry["metrics"][metric + "_log_loss_" + aggregation]
                    assert (value is not None and math.isfinite(value) and value >= 0) if supported else value is None
            if task == "curve":
                assert not any(key.startswith(("head_p.", "p_linear.")) for key in state)
            if task == "instant":
                assert not any(key.startswith(("head_w.", "w_linear.", "ahead_linear.", "head_ahead_logits.")) for key in state)
        rows.append({"case": name, "updates": expected, "checkpoint_count": len(result["history"]),
                     "source_tree_sha256": saved["source_tree_sha256"], "status": "audited; synthetic only"})
    # Curve binary LL must retain large wrong-prediction losses, as the original
    # BCE-with-logits evaluator does. A probability clamp must not hide them.
    from scripts.train_gpu_future import evaluate_batches
    from scripts.gpu_future_models import TaskStatistics
    from scripts.gpu_training_targets import compare_budgets
    from rwkv.model.srs_model import PreparedBatch
    logits = torch.tensor([[-25.0, 25.0]])
    labels = torch.zeros(1, 2, 7)
    labels[..., 2] = torch.tensor([[1.0, 0.0]])
    batch = PreparedBatch(1, torch.zeros(2, 92), [], [], [], [], labels, torch.tensor([[1, 2]]))
    class Predictor(torch.nn.Module):
        def get_loss(self, batch):
            return TaskStatistics(torch.tensor(0.0), None, None, logits.sigmoid(), None, None, 2, 0,
                                  torch.ones(1, 2), torch.zeros(1, 2), torch.zeros(1, 2), batch.label_review_th, logits)
    scores = evaluate_batches(Predictor(), [batch])
    assert abs(scores["curve_log_loss"] - 25) < 1e-5
    control = json.loads((ROOT / "artifacts/benchmarks/gpu-future/integration-final-joint-v2/result.json").read_text())
    candidate = json.loads((ROOT / "artifacts/benchmarks/gpu-future/integration-expertium-curve/result.json").read_text())
    budgets = compare_budgets(control["history"], candidate["history"])
    for axis in ("training_seconds", "elapsed_seconds", "raw_review_presentations"):
        for side in ("control", "candidate"):
            entry = budgets[axis][side]
            assert entry is None or entry[axis] <= budgets[axis]["common_limit"]
    result = {"status": "passed", "analysis_provenance": provenance, "cases": rows,
              "scope": "synthetic checkpoints, finite changed FP32 weights, supported-task masks, recorded source archives and observed budget limits; not speed/quality validation",
              "curve_evaluation_large_wrong_logits": "passed; log loss 25 retained", "observed_budgets": budgets}
    target = ROOT / "artifacts/benchmarks/gpu-future/integration-audit.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Audited {len(rows)} synthetic runs: finite checkpoints, changed heads and correct task reporting")


if __name__ == "__main__":
    main()
