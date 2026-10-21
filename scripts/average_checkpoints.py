#!/usr/bin/env python
"""SM51: uniform average of model snapshots (weight averaging along a decay) into a master.pth the scorer accepts.

  average  --inputs A.pth B.pth [...] --out DIR/master.pth   uniform mean of the floating-point tensors (float64
           accumulation, cast back to each tensor's dtype); non-float tensors must be equal in every input and are copied;
           keys, order, dtypes and shapes must match exactly. Writes DIR/average.json (inputs with sha256, method).
  decay    --case DIR --out-dir DIR [--last 2,3,4]             the snapshots of a decay case (snapshots/users-*/master.pth,
           in user order) plus its final master.pth; writes <out-dir>/avg<k>/master.pth = mean of the last k points.
  check    --input A.pth --out report.json                     CPU check: the mean of 2, 3 and 4 copies of A is A bit for
           bit; keys, order, dtypes and shapes intact; a 2-point mean of A and a perturbed A is their exact midpoint
           (within float rounding) and leaves non-float tensors untouched.

The scorer (scripts/small_model_heldout.py --model TAG=expertium:w64:<master.pth>) loads a plain state_dict, so the
output is a plain state_dict (torch.save of an ordered dict of tensors), like the trainer's master.pth.
"""
from argparse import ArgumentParser
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def average_states(states):
    if not states:
        raise ValueError("no inputs")
    first = states[0]
    for i, state in enumerate(states[1:], 1):
        if list(state) != list(first):
            raise ValueError(f"input {i}: keys or key order differ from input 0")
        for key, value in state.items():
            if value.dtype != first[key].dtype or value.shape != first[key].shape:
                raise ValueError(f"input {i}: {key} has {value.dtype}{tuple(value.shape)}, input 0 "
                                 f"{first[key].dtype}{tuple(first[key].shape)}")
    out = OrderedDict()
    for key, value in first.items():
        if value.is_floating_point():
            total = torch.zeros(value.shape, dtype=torch.float64)
            for state in states:
                total += state[key].detach().to(torch.float64)
            out[key] = (total / len(states)).to(value.dtype).contiguous()
        else:
            if any(not torch.equal(state[key], value) for state in states):
                raise ValueError(f"non-float tensor {key} differs between inputs")
            out[key] = value.clone()
    return out


def load(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"{path} is not a plain state_dict of tensors")
    return state


def average(inputs, out):
    state = average_states([load(p) for p in inputs])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)
    meta = {"method": "uniform mean, float64 accumulation, cast back to each tensor's dtype; non-float tensors copied (equal in all inputs)",
            "inputs": [{"path": str(p), "sha256": sha256(p)} for p in inputs], "output": str(out), "output_sha256": sha256(out),
            "tensors": len(state), "parameters": sum(v.numel() for v in state.values() if v.is_floating_point())}
    (out.parent / "average.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"[average] {len(inputs)} inputs -> {out} ({meta['parameters']:,} float values)")
    return meta


def decay_points(case):
    case = Path(case)
    snaps = sorted((case / "snapshots").glob("users-*/master.pth"), key=lambda p: int(p.parent.name.split("-")[1]))
    final = case / "master.pth"
    if not final.exists():
        raise FileNotFoundError(f"{final} missing (decay not finished)")
    return snaps + [final]


def check(path, out, arch="w64"):
    base = load(path)
    report = {"input": str(path), "checks": {}}

    def record(name, value, ok):
        report["checks"][name] = {"value": value, "ok": bool(ok)}
        print(("PASS " if ok else "FAIL ") + name + ": " + json.dumps(value))
    for k in (2, 3, 4):
        avg = average_states([base] * k)
        same = list(avg) == list(base) and all(avg[n].dtype == base[n].dtype and avg[n].shape == base[n].shape and torch.equal(avg[n], base[n])
                                               for n in base)
        record(f"mean of {k} identical copies == input bitwise (keys, order, dtypes, shapes)", len(base), same)
    gen = torch.Generator().manual_seed(0)
    other = OrderedDict((n, v + torch.randn(v.shape, generator=gen).to(v.dtype) * 1e-3 if v.is_floating_point() else v.clone())
                        for n, v in base.items())
    mid = average_states([base, other])
    exact = all(torch.equal(mid[n], ((base[n].double() + other[n].double()) / 2).to(base[n].dtype))
                for n in base if base[n].is_floating_point())
    err = max(float(((mid[n].double() - (base[n].double() + other[n].double()) / 2).abs()).max())
              for n in base if base[n].is_floating_point() and base[n].numel())
    record("2-point mean = the exact midpoint rounded once to the tensor dtype (bitwise)", {"max_abs_vs_float64": err}, exact)
    record("non-float tensors untouched", [n for n in base if not base[n].is_floating_point()],
           all(torch.equal(mid[n], base[n]) for n in base if not base[n].is_floating_point()))
    record("dtypes", sorted({str(v.dtype) for v in base.values()}), True)
    if arch:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from expertium.variants import build
        model = build(arch).float()
        model.load_state_dict(average_states([base, other]), strict=True)
        record(f"averaged state loads strictly into the {arch} model (what the scorer builds)", arch, True)
    tmp = Path(out).parent / "average-check-tmp" / "master.pth"
    try:
        average([path, path, path], tmp)
        record("file round trip: average([A, A, A]) file == A tensors", str(tmp), all(torch.equal(load(tmp)[n], base[n]) for n in base))
    finally:
        for f in tmp.parent.glob("*"):
            f.unlink()
        tmp.parent.rmdir()
    report["passed"] = all(c["ok"] for c in report["checks"].values())
    report["fails"] = [k for k, c in report["checks"].items() if not c["ok"]]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print("ALL OK" if report["passed"] else "FAILED " + str(report["fails"]))
    return 0 if report["passed"] else 1


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("average", "decay", "check"))
    parser.add_argument("--inputs", nargs="+", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--last", default="2,3,4")
    parser.add_argument("--arch", default="w64", help="check: architecture to load the average into ('' = skip)")
    args = parser.parse_args()
    if args.command == "check":
        return check(args.input, args.out, args.arch)
    if args.command == "average":
        average(args.inputs, args.out)
        return 0
    points = decay_points(args.case)
    print(f"[average] points: {[p.parent.name for p in points]}")
    for k in (int(x) for x in args.last.split(",")):
        if k > len(points):
            raise SystemExit(f"only {len(points)} points, cannot average the last {k}")
        average(points[-k:], args.out_dir / f"avg{k}" / "master.pth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
