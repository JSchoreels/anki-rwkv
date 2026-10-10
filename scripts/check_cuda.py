"""Verify the compiled recurrence and both models on the local NVIDIA GPU."""

from argparse import Namespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from rwkv.model.rwkv_ops import RWKV7_WKV, reference_rwkv7
from rwkv.model.srs_model import PreparedBatch
from rwkv.train_rwkv import CLIP, get_optimizer, transfer_child_grad_to_master
from scripts.compare import (BASELINE_CONFIG, SMALL_CONFIG, BaselineModel, SmallModel,
                             require_device, training_models)


def synthetic_batch(device, dtype):
    """Two independent sequences with both heads labelled and query skips."""
    batch_size, length = 2, 35
    labels = torch.zeros(batch_size, length, 7, device=device)
    labels[:, :, 0] = 100.0
    labels[:, :, 2] = 1.0
    labels[:, :, 3] = 3.0
    labels[:, :, 4] = 1.0
    labels[:, 1::2, 6] = 1.0
    skips = labels[:, :, 6].bool()
    gather = torch.arange(batch_size * length, device=device)
    shifts = (((torch.arange(length, device=device) - 1).clamp_min(0) // 2) * 2).repeat(batch_size, 1)
    return PreparedBatch(
        num_data=batch_size, start=torch.randn(batch_size * length, 92, device=device, dtype=dtype) * 0.1,
        sub_gather=[[gather] for _ in range(5)], sub_gather_lens=[[length] for _ in range(5)],
        time_shift_selects=[[shifts.flatten()] for _ in range(5)],
        skips=[[skips.flatten()] for _ in range(5)], labels=labels,
        label_review_th=torch.arange(1, length + 1, device=device).repeat(batch_size, 1))


def check_recurrence(device, dtype, heads):
    torch.manual_seed(2025)
    shape = (2, 35, heads, 32)  # crosses the 32-step state checkpoint boundary
    cpu_inputs = [(torch.randn(shape) * 0.05).to(dtype) for _ in range(6)]
    cpu_inputs[3] = torch.full(shape, 0.9, dtype=torch.float32)
    skip = torch.zeros(2, 35, dtype=torch.bool)
    skip[:, 3::7] = True
    expected_inputs = [value.clone().requires_grad_() for value in cpu_inputs]
    expected = reference_rwkv7(*expected_inputs, skip)
    output_grad = torch.randn(shape) * 0.1
    (expected.float() * output_grad).sum().backward()
    actual_inputs = [value.to(device).requires_grad_() for value in cpu_inputs]
    actual = RWKV7_WKV.apply(*actual_inputs, skip.to(device))
    (actual.float() * output_grad.to(device)).sum().backward()
    # BF16 rounds native backward results; float32 has a tighter parity bound.
    atol, rtol = (2e-5, 2e-3) if dtype == torch.float32 else (2e-4, 5e-2)
    torch.testing.assert_close(actual.cpu(), expected.detach(), atol=atol, rtol=rtol)
    for actual_input, expected_input in zip(actual_inputs, expected_inputs):
        assert actual_input.grad is not None and torch.isfinite(actual_input.grad).all()
        torch.testing.assert_close(actual_input.grad.cpu(), expected_input.grad, atol=atol, rtol=rtol)


def check_training(device, dtype):
    batch = synthetic_batch(device, dtype)
    for name, model_class, architecture in [("baseline", BaselineModel, BASELINE_CONFIG),
                                             ("expertium_iter36", SmallModel, SMALL_CONFIG)]:
        master, model = training_models(model_class, architecture, device, dtype, 2025)
        optimizer = get_optimizer(Namespace(PEAK_LR=7e-4), master)
        initial = master.w_linear.weight.detach().clone()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            if model is not master:
                model.zero_grad(set_to_none=True)
                model.copy_downcast_(master, dtype)
            stats = model.get_loss(batch)
            assert stats is not None and torch.isfinite(stats.average_loss)
            assert stats.ahead_n == 36 and stats.imm_n == 34
            stats.average_loss.backward()
            if model is not master:
                transfer_child_grad_to_master(master, model)
            norm = torch.nn.utils.clip_grad_norm_(master.parameters(), CLIP, error_if_nonfinite=True)
            assert torch.isfinite(norm)
            optimizer.step()
        assert (master.w_linear.weight.detach() - initial).abs().max() > 0
        assert all(param.dtype == torch.float32 and torch.isfinite(param).all() for param in master.parameters())
        print(f"{name}: two finite {dtype} training updates with float32 master weights passed.", flush=True)


def main():
    device = require_device({"DEVICE": "cuda:0", "DTYPE": "bfloat16"})
    print(f"GPU: {torch.cuda.get_device_name(device)}; capability {torch.cuda.get_device_capability(device)}; "
          f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}", flush=True)
    for dtype in [torch.float32, torch.bfloat16]:
        for heads in [1, 4]:
            check_recurrence(device, dtype, heads)
        check_training(device, dtype)
    torch.cuda.synchronize(device)
    print("CUDA forward/backward parity and both model training checks passed.")


if __name__ == "__main__":
    main()
