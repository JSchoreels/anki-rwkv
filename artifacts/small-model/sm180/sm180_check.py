#!/usr/bin/env python
"""4-bit weights (SM180): CPU proofs of the gated code addition (design.md "The gated code addition"), run in the run
tree (C:/Users/jonat/workspace/wt/sm180-run). CPU only, well under 1 GB of RAM.

  python sm180_check.py   -> check-cpu-sm180.json (all_ok)

C1 labels: w4g32 / w4g16 parse to w4-g32 / w4-g16 with the state format unchanged; w8 / w4 / fp32 parse as before.
C2 scorer weights: for Small (w64) and six-layer Tiny (w64-c1-d2p1u1) release weights, the scorer's w4 state dicts
   (quantize_state_dict over linear_weight_names) equal fake_quant of every 2-D .weight and leave every other tensor
   bitwise equal; the int4 codes are in -7..7 with at most 16 distinct values per group; w8 unchanged vs a direct
   int8 per-row re-implementation (the weights.int8.bin rule).
C3 QAT on the trainer's master / child pair (Tiny, CPU eager, float32 master, bfloat16 child, MasterChildSync): the
   plain downcast is unchanged before install; install selects exactly the scorer's quantized list; after install every
   downcast writes the scorer's w4g32 weights (cast to the child dtype) into the child 2-D weights, leaves every other
   child tensor as the plain downcast and the master untouched; the gradient transfer to the master is unchanged
   (straight-through); the flat-buffer sync and float32 training (child = master) are refused; uninstall restores.
C4 an unknown key is refused; another state_vq recipe never reaches weight_qat.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = Path(os.environ.get("SM180_CODE", "C:/Users/jonat/workspace/wt/sm180-run")).resolve()
sys.path.insert(0, str(CODE))
os.environ.setdefault("RWKV_NO_JIT", "1")

import torch  # noqa: E402

from scripts import compress_quant as cq  # noqa: E402
from scripts import small_model_heldout as smh  # noqa: E402
from scripts import weight_qat  # noqa: E402

W = "C:/Users/jonat/workspace/wt"
MODELS = {"small": ("w64", W + "/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/E89-F65e-F77a/master.pth"),
          "tiny": ("w64-c1-d2p1u1", W + "/sm130/artifacts/small-model/sm130-th-dec2ep32k-p2/expertium/E89-F65e-F77a/master.pth")}
OUT = {}


def record(name, value, ok):
    OUT[name] = {"value": value, "ok": bool(ok)}
    print(f"[sm180-check] {'OK  ' if ok else 'FAIL'} {name}: {value}", flush=True)


def c1():
    for label, want in (("w4g32+lr2.16.4q6x8", "w4-g32"), ("w4g16+lr2.16.4q6x8", "w4-g16"),
                        ("w4g32+lr2.16.4q8x8", "w4-g32"), ("w4g16+lr2.16.4q8x8", "w4-g16"),
                        ("w8+lr2.16.4q6x8", "w8-pc"), ("w8+lr2.16.4q8x8", "w8-pc"), ("w8+s8hx8", "w8-pc"),
                        ("w4+s8hx8", "w4-g32"), ("fp32", "fp32")):
        scheme, state = smh.parse_variant(label)
        same_state = True
        if "+" in label:
            ref = smh.parse_variant("w8+" + label.split("+", 1)[1])[1]
            same_state = type(state) is type(ref) and (getattr(state, "fields", None) is None
                                                       or state.fields() == ref.fields()) and (
                not isinstance(state, tuple) or tuple(state) == tuple(ref))
        record(f"C1 parse {label}", f"{scheme} {type(state).__name__}", scheme == want and same_state)
    record("C1 scheme table", {k: cq.WEIGHT_SCHEMES[k] for k in ("w8-pc", "w4-g32", "w4-g16")},
           cq.WEIGHT_SCHEMES["w8-pc"] == (8, None) and cq.WEIGHT_SCHEMES["w4-g32"] == (4, 32)
           and cq.WEIGHT_SCHEMES["w4-g16"] == (4, 16))


def int8_rows(w):
    scale = w.abs().amax(dim=1, keepdim=True) / 127
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.clamp(torch.round(w / scale), -127, 127) * scale


def c2(size, arch, path):
    base = {k: v.float() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
    names = cq.linear_weight_names(base)
    two_d = [k for k, v in base.items() if k.endswith(".weight") and v.dim() == 2]
    record(f"C2 {size} quantized list = every 2-D .weight", len(names), names == two_d)
    w8 = cq.quantize_state_dict(base, names, *cq.WEIGHT_SCHEMES["w8-pc"])
    record(f"C2 {size} w8 = int8 per row (weights.int8.bin rule)",
           max(float((w8[n] - int8_rows(base[n])).abs().max()) for n in names),
           all(torch.equal(w8[n], int8_rows(base[n])) for n in names))
    for scheme in ("w4-g32", "w4-g16"):
        bits, group = cq.WEIGHT_SCHEMES[scheme]
        q = cq.quantize_state_dict(base, names, bits, group)
        others = all(torch.equal(q[k], base[k]) for k in base if k not in set(names))
        same = all(torch.equal(q[n], cq.fake_quant(base[n], bits, group)) for n in names)
        worst_levels, codes_ok = 0, True
        for n in names:
            w = base[n]
            out_f, in_f = w.shape
            g = min(group, in_f) if in_f >= group else in_f
            for start in range(0, in_f, group):
                blk_w, blk_q = w[:, start:start + group], q[n][:, start:start + group]
                scale = blk_w.abs().amax(dim=1, keepdim=True) / 7
                scale = torch.where(scale > 0, scale, torch.ones_like(scale))
                codes = blk_q / scale
                codes_ok &= bool((codes.round() - codes).abs().max() < 1e-4) and bool(codes.abs().max() <= 7 + 1e-4)
                worst_levels = max(worst_levels, max(len(torch.unique(r.round())) for r in codes))
            del g
        changed = sum(int(not torch.equal(q[n], base[n])) for n in names)   # rows of 1-2 values are exact at int4
        nbytes = cq.weight_bytes(base, names, bits, group)["total"]
        record(f"C2 {size} {scheme}: fake_quant on the list, other tensors bitwise equal, codes in -7..7",
               {"tensors_changed": changed, "max_levels_per_row_group": worst_levels, "bytes": nbytes,
                "bytes_w8": cq.weight_bytes(base, names, 8, None)["total"]},
               others and same and codes_ok and worst_levels <= 15 and changed > 0.5 * len(names))
    return base, names


def c3(arch, base, names):
    """The trainer's master / child pair on CPU (eager): float32 master, bfloat16 selectively cast child, the release
    sync (MasterChildSync, grouped_sync)."""
    from expertium.variants import build
    from scripts.gpu_r01_update import MasterChildSync
    master = build(arch).float()
    master.load_state_dict(base, strict=True)
    child = build(arch).selective_cast(torch.bfloat16)
    child.copy_downcast_(master, torch.bfloat16)
    plain = {k: v.clone() for k, v in child.state_dict().items()}
    sync = MasterChildSync(master, child)
    sync.downcast()
    record("C3 plain grouped downcast = copy_downcast_ (before install)", len(plain),
           all(torch.equal(child.state_dict()[k], plain[k]) for k in plain))
    info = weight_qat.install(child, {"recipe": "weight-qat", "bits": 4, "group": 32}, "cpu")
    qnames = sorted(n for n, p in child.named_parameters() if id(p) in weight_qat.QUANT)
    record("C3 install selects exactly the scorer's quantized list", info["quantized_tensors"], qnames == sorted(names))
    sync.downcast()
    after = dict(child.named_parameters())
    q = cq.quantize_state_dict(base, names, 4, 32)
    quant_ok = all(torch.equal(after[n].data, q[n].to(after[n].dtype)) for n in names)
    other_ok = all(torch.equal(after[n].data, plain[n]) for n in after if n not in set(names))
    master_ok = all(torch.equal(master.state_dict()[k], base[k]) for k in base)
    record("C3 after install, every downcast: child 2-D weights = scorer w4g32 weight cast to the child dtype; other "
           "child tensors = the plain downcast; master (latent) unchanged",
           {"downcasts": info["downcasts"], "tensors_last_downcast": info.get("tensors_last_downcast"),
            "child_dtypes": sorted({str(after[n].dtype) for n in names})},
           quant_ok and other_ok and master_ok and info["downcasts"] == 1 and info["tensors_last_downcast"] == len(names))
    # straight-through: the gradient is computed at the child (quantized) weights and the trainer's transfer copies it
    # unchanged onto the float32 master; check that transfer on one synthetic gradient
    for p in child.parameters():
        p.grad = torch.ones_like(p) * 0.5
    sync.transfer_grads()
    record("C3 gradient transfer child -> master unchanged (STE: applied to the latent weights)", 0,
           all(m.grad is not None and torch.equal(m.grad, torch.full_like(m, 0.5)) for m, _c in sync.pairs))
    from scripts import gpu_r01_flat
    try:
        gpu_r01_flat.FlatMasterChildSync(master, child, None)
        refused = False
    except ValueError:
        refused = True
    record("C3 flat-buffer sync refused while installed", refused, refused)
    shared = build(arch).float()
    shared.load_state_dict(base)
    weight_qat.install(shared, {"recipe": "weight-qat", "bits": 4, "group": 32})
    try:
        weight_qat.apply([(p, p) for p in shared.parameters()])
        guarded = False
    except ValueError:
        guarded = True
    record("C3 float32 training (child is master) refused", guarded, guarded)
    weight_qat.uninstall()
    child.copy_downcast_(master, torch.bfloat16)
    sync.downcast()
    record("C3 uninstall: downcast is the plain copy again", 0,
           all(torch.equal(child.state_dict()[k], plain[k]) for k in plain))


def c4():
    try:
        weight_qat.install(torch.nn.Linear(4, 4), {"recipe": "weight-qat", "bits": 4, "group": 32, "oops": 1})
        refused = False
    except ValueError:
        refused = True
    record("C4 unknown key refused", refused, refused)
    weight_qat.uninstall()
    import inspect
    from scripts import state_vq
    src = inspect.getsource(state_vq.install_training)
    head = src.split("global LEARNED", 1)[1].split("from expertium.model import state_vq_hooks", 1)[0]
    gated = 'value.get("recipe") == "weight-qat"' in head and head.count("return") == 1
    record("C4 state_vq.install_training: weight_qat reached only for recipe weight-qat", gated, gated)


def main():
    with torch.no_grad():
        c1()
    base, names = None, None
    for size, (arch, path) in MODELS.items():
        with torch.no_grad():
            b, n = c2(size, arch, path)
        if size == "tiny":
            base, names = b, n
    c3(MODELS["tiny"][0], base, names)
    c4()
    import subprocess
    commit = subprocess.run(["git", "-C", str(CODE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(CODE), "status", "--porcelain", "--", "scripts", "expertium", "rwkv"],
                           capture_output=True, text=True).stdout.strip()
    all_ok = all(v["ok"] for v in OUT.values())
    (HERE / "check-cpu-sm180.json").write_text(json.dumps({"all_ok": all_ok, "code_tree": str(CODE), "commit": commit,
                                                           "dirty": bool(dirty), "checks": OUT}, indent=1) + "\n",
                                               encoding="utf-8")
    print(f"[sm180-check] {'ALL OK' if all_ok else 'FAILED'} ({len(OUT)} checks, run tree {commit[:8]}, dirty {bool(dirty)})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
