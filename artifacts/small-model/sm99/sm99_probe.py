"""The show-time instant-only big model (SM99): a few real training updates on CPU in one source tree (called by
sm99_check.py). A copy of the instant-only small model (SM57)'s sm57_probe.py, unchanged except this docstring; the
big model runs through it with spec["arch"] = "fork" and spec["lr"] = 3.5e-4.

  python sm99_probe.py <tree> <spec.json>

The trainer's batch path (scripts.gpu_r01_prefetch.training_batches with the run's options, state carry lanes), the
trainer's loss (scripts.expertium_options.CatchUp.loss), clip_grad_norm_(CLIP) and the trainer's optimizer
(CatchUp.optimizer: the rwkv AdamW, or the cool-down's Muon) for every update of spec["rows"], from spec["checkpoint"].
The curve_loss_weight option is applied the way scripts/gpu_r01_training.py applies it (only when present in the
options). Prints one line "PROBE <json>": per update the loss bits, the instant average bits, a digest of every
gradient, the names of parameters with an all-zero gradient; the final weights' digest.
"""
import hashlib
import json
import os
import sys

root = sys.argv[1]
sys.path.insert(0, root)
os.chdir(root)
import torch  # noqa: E402

torch.set_num_threads(2)
torch._C._set_graph_executor_optimize(False)   # as scripts/expertium_decay.py run (jit_optimize false)
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from scripts.gpu_r01_prefetch import training_batches  # noqa: E402
from scripts.expertium_options import CatchUp  # noqa: E402
from expertium.variants import build  # noqa: E402
from rwkv.train_rwkv import CLIP  # noqa: E402

spec = json.loads(open(sys.argv[2]).read())
options = spec["options"]
model = build(spec["arch"]).float()
model.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu", weights_only=True), strict=True)
if options.get("curve_loss_weight") is not None:   # as scripts/gpu_r01_training.py
    model.curve_loss_weight = float(options["curve_loss_weight"])
params = dict(model.named_parameters())
catch_up = CatchUp(options, {}, spec["rows"])
optimizer = catch_up.optimizer(model)
for group in optimizer.param_groups:
    group["lr"] = spec.get("lr", 7e-4)
with ThreadPoolExecutor(max_workers=1) as pool:
    prepared = list(training_batches(spec["rows"], spec["span"], options["training_seed"], pool, options, None, 2))
out = []
for row, item in prepared:
    batch = item.batch.to(torch.device("cpu"))
    model.train()
    optimizer.zero_grad(set_to_none=False)
    model.zero_grad(set_to_none=True)
    torch.manual_seed(item.seed)
    catch_up.carry.attach(item, batch, row["user_id"])
    stats = catch_up.loss(model, batch, 0, (row["user_id"], item.offset, item.reviews, item.seed), item.batch.labels)
    stats.average_loss.backward()
    h = hashlib.sha256()
    zero = []
    for name, p in params.items():
        g = p.grad
        h.update(b"-" if g is None else g.detach().numpy().tobytes())
        if g is None or not bool(g.any()):
            zero.append(name)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    optimizer.step()
    out.append({"user": row["user_id"], "offset": item.offset, "loss_hex": float(stats.average_loss).hex(),
                "imm_hex": float(stats.imm_avg).hex(), "curve_hex": float(stats.ahead_avg).hex(),
                "p_digest": hashlib.sha256(stats.p_imm_all.detach().numpy().tobytes()).hexdigest(),
                "grad_digest": h.hexdigest(), "grad_norm": float(norm), "zero_grad": zero,
                "finite": bool(torch.isfinite(stats.average_loss))})
w = hashlib.sha256()
for name, t in model.state_dict().items():
    w.update(name.encode())
    w.update(t.detach().contiguous().numpy().tobytes())
finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
print("PROBE " + json.dumps({"updates": out, "weights": w.hexdigest(), "weights_finite": finite,
                             "jit": isinstance(model, torch.jit.ScriptModule) or any(
                                 isinstance(m, torch.jit.ScriptModule) for m in model.modules()),
                             "optimizer": type(optimizer).__name__, "parameters": list(params)}))
