"""The instant-only small model (SM57) stage 3: a few real training updates on CPU in one source tree (called by
sm57s3_check.py). sm57_probe.py (stages 1-2) plus: the instant_loss_weight option applied the way
scripts/gpu_r01_training.py applies it, the run's real leak-free replay KD (CatchUp.load_teacher on the CPU) when
spec["teacher"] is true, optional teacher-target perturbations, and per update every loss component's bits and a digest
of the batch.

  python sm57s3_probe.py <tree> <spec.json>

spec: arch, checkpoint, options, rows, span, [lr], [teacher: bool], [perturb: "" | "rating" | "curve"]
  perturb "rating": the teacher's instant (rating) targets are reversed along the 4 ratings (a different distribution
  on every query row); "curve": the teacher's curve targets become 1 - target on every labelled row.
Prints one line "PROBE <json>".
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
if options.get("instant_loss_weight") is not None:   # as scripts/gpu_r01_training.py (SM26)
    model.instant_loss_weight = float(options["instant_loss_weight"])
if options.get("curve_loss_weight") is not None:     # as scripts/gpu_r01_training.py (SM57)
    model.curve_loss_weight = float(options["curve_loss_weight"])
params = dict(model.named_parameters())
catch_up = CatchUp(options, {}, spec["rows"])
if spec.get("teacher"):
    catch_up.load_teacher(torch.device("cpu"), torch.float32)
    original = catch_up.teacher.targets
    perturb = spec.get("perturb", "")

    def targets(*args, **kwargs):
        curve, rating, alpha = original(*args, **kwargs)
        if perturb == "rating":
            rating = rating.flip(-1).contiguous()
        elif perturb == "curve":
            curve = torch.where(curve != 0, 1.0 - curve, curve)
        return curve, rating, alpha
    catch_up.teacher.targets = targets
optimizer = catch_up.optimizer(model)
for group in optimizer.param_groups:
    group["lr"] = spec.get("lr", 7e-4)
with ThreadPoolExecutor(max_workers=1) as pool:
    prepared = list(training_batches(spec["rows"], spec["span"], options["training_seed"], pool, options, None, 2))


def digest(*tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(b"-" if t is None else t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


COMPONENTS = ("ahead_avg", "imm_avg", "ahead_raw_avg", "w_loss_avg", "ahead_logits_mag_loss_avg", "ahead_logits_diff_loss_avg")
out = []
for row, item in prepared:
    batch = item.batch.to(torch.device("cpu"))
    batch_digest = digest(*(getattr(item.batch, k) for k in ("start", "labels", "label_review_th")
                            if isinstance(getattr(item.batch, k, None), torch.Tensor)))
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
    c = {k: getattr(stats, k).detach().float() for k in COMPONENTS}
    a, i, r, w, m, d = (c[k] for k in COMPONENTS)
    # The loss's own order and constants (expertium/model/variant_model.py _get_loss_variant), with CURVE_SCALE 1.0:
    full = 0.5 * a + float(model.instant_loss_weight) * i + 0.5 * r + 1e-5 * w + 1e-4 * m + 1e-3 * d
    curve_terms = 0.5 * a + 0.5 * r + 1e-5 * w + 1e-4 * m + 1e-3 * d   # the same sum without the instant term
    out.append({"user": row["user_id"], "offset": item.offset, "reviews": item.reviews, "seed": item.seed,
                "batch_digest": batch_digest, "loss_hex": float(stats.average_loss).hex(),
                "components_hex": {k: float(v).hex() for k, v in c.items()},
                "recomputed_full_hex": float(full).hex(), "recomputed_curve_terms_hex": float(curve_terms).hex(),
                "imm_hex": float(stats.imm_avg).hex(), "curve_hex": float(stats.ahead_avg).hex(),
                "p_digest": digest(stats.p_imm_all), "curve_digest": digest(stats.p_curve),
                "grad_digest": h.hexdigest(), "grad_norm": float(norm), "zero_grad": zero,
                "finite": bool(torch.isfinite(stats.average_loss))})
wd = hashlib.sha256()
for name, t in model.state_dict().items():
    wd.update(name.encode())
    wd.update(t.detach().contiguous().numpy().tobytes())
finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
print("PROBE " + json.dumps({"updates": out, "weights": wd.hexdigest(), "weights_finite": finite,
                             "instant_loss_weight": float(model.instant_loss_weight),
                             "curve_loss_weight": float(model.curve_loss_weight),
                             "teacher": None if catch_up.teacher is None else type(catch_up.teacher.replay).__name__,
                             "teacher_stats": None if catch_up.teacher is None else dict(catch_up.teacher.cache_stats),
                             "jit": isinstance(model, torch.jit.ScriptModule) or any(
                                 isinstance(m, torch.jit.ScriptModule) for m in model.modules()),
                             "optimizer": type(optimizer).__name__, "parameters": list(params)}))
