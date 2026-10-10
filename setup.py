"""Build the upstream CUDA extension only when explicitly requested."""

import os
from pathlib import Path

from setuptools import setup

extensions = []
commands = {}
if os.environ.get("RWKV_BUILD_CUDA") == "1":
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

    if CUDA_HOME is None or not torch.cuda.is_available():
        raise RuntimeError("Building RWKV requires a CUDA toolkit and CUDA-enabled PyTorch/GPU")
    sources = Path(__file__).parent / "rwkv" / "model" / "csrc"
    extensions = [
        CUDAExtension(
            "rwkv.model.RWKV_CUDA",
            sources=[str(path) for path in sorted(sources.glob("*.cpp"))]
            + [str(path) for path in sorted((sources / "cuda").glob("*.cu"))],
            extra_compile_args={"cxx": ["/O2"] if os.name == "nt" else ["-O3"], "nvcc": ["-O3"]},
        )
    ]
    commands["build_ext"] = BuildExtension

setup(ext_modules=extensions, cmdclass=commands)
