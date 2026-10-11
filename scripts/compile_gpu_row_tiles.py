"""CPU-only NVRTC/PTXAS compilation of F09; no CUDA context or kernel launches."""

from argparse import ArgumentParser
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmarks/gpu-future/row-tiles-compile")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / "artifacts"):
        parser.error("Output must remain below workspace artifacts")
    provenance = process_provenance()
    import torch
    torch.set_num_threads(1)
    if os.name == "nt":
        directory = str(Path(torch.__file__).parent / "lib")
        os.environ["PATH"] = directory + os.pathsep + os.environ["PATH"]
        dll_handle = os.add_dll_directory(directory)
    os.environ["CUPY_CACHE_DIR"] = str(ROOT / "artifacts/cache/cupy-cache")
    from cupy.cuda.compiler import _NVRTCProgram
    from scripts.gpu_row_tiles import source_options
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "provenance": provenance,
              "scope": "CPU-only compilation for sm_120; not runtime parity, timing or quality",
              "cases": []}
    for dtype in (torch.float32, torch.bfloat16):
        source, options = source_options(dtype)
        scalar = "float" if dtype == torch.float32 else "__nv_bfloat16"
        for rows, tile in ((1, 8), (2, 8), (4, 8), (4, 16), (4, 32)):
            name = f"{str(dtype).split('.')[-1]}-rows{rows}-tile{tile}"
            names = tuple(f"rwkv_future::{stage}_kernel<{tile}, {rows}, {scalar}>" for stage in ("forward", "backward"))
            log = io.StringIO()
            program = _NVRTCProgram(source, name + ".cu", name_expressions=names)
            ptx, mapping = program.compile((*options, "--gpu-architecture=compute_120"), log_stream=log)
            ptx_path = args.output / (name + ".ptx")
            ptx_path.write_bytes(ptx)
            case = {"name": name, "rows": rows, "tile": tile, "nvrtc": "compiled", "symbols": mapping,
                    "ptx": ptx_path.relative_to(ROOT).as_posix(), "compiler_log": log.getvalue()}
            assembler = shutil.which("ptxas")
            if assembler:
                assembled = subprocess.run([assembler, "--gpu-name=sm_120", "--verbose", str(ptx_path), "-o",
                                            str(args.output / (name + ".cubin"))], capture_output=True, text=True)
                case.update(ptxas_exit_code=assembled.returncode, ptxas_log=assembled.stdout + assembled.stderr)
                if assembled.returncode:
                    raise RuntimeError(case["ptxas_log"])
            report["cases"].append(case)
            (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(name + ": compiled", flush=True)
    report["status"] = "complete"
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
