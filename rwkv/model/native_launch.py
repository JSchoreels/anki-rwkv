"""F65 native autograd/dispatcher transport for unchanged NVRTC function handles."""
from pathlib import Path
import hashlib
import json
import os
from functools import lru_cache
import torch

ROOT = Path(__file__).resolve().parents[2]
_schemas = torch.library.Library('rwkv_native','DEF')
_schemas.define('wkv(Tensor r, Tensor k, Tensor v, Tensor w, Tensor a, Tensor kd, Tensor skip, int forward_handle, int backward_handle, int rows, int tile) -> Tensor')
_loaded = None

BUILD_FLAGS = ['/nologo','/LD','/std:c++17','/O2','/MD','/EHsc','/bigobj','/D_CRT_SECURE_NO_WARNINGS','/DNOMINMAX']

@lru_cache(maxsize=1)
def build_identity():
    source = ROOT/'rwkv/model/csrc/native_launch.cpp'
    compiler_root = Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC')
    sdk_root = Path('C:/Program Files (x86)/Windows Kits/10/Include')
    compiler = sorted((path for path in compiler_root.glob('*') if path.is_dir()), key=lambda path:tuple(int(v) for v in path.name.split('.')))
    sdk = sorted((path for path in sdk_root.glob('10.*') if path.is_dir()), key=lambda path:tuple(int(v) for v in path.name.split('.')))
    cuda_header = Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/include/cuda.h')
    record = {'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'torch':torch.__version__,
              'torch_cuda':torch.version.cuda,'platform':os.name,'flags':BUILD_FLAGS,
              'msvc_version':compiler[-1].name if compiler else None,'windows_sdk_version':sdk[-1].name if sdk else None,
              'cuda_header_version':'13.4','cuda_header_sha256':hashlib.sha256(cuda_header.read_bytes()).hexdigest(),
              'torch_cudart_sha256':hashlib.sha256((Path(torch.__file__).parent/'lib/cudart64_12.dll').read_bytes()).hexdigest(),
              'libraries':['c10.lib','c10_cuda.lib','torch_cpu.lib','torch_cuda.lib','torch.lib','cuda.lib','torch_cudart.lib']}
    return record, hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()

def load():
    global _loaded
    identity,key=build_identity()
    controller=Path(os.environ.get('GPU_R01_CONTROLLER_ROOT',ROOT))
    folder=controller/'artifacts/cache/r01-native'/key
    metadata=json.loads((folder/'build.json').read_text(encoding='utf-8'))
    binary=folder/'native_launch.dll'
    if metadata.get('identity') != identity or metadata.get('status') != 'built' or hashlib.sha256(binary.read_bytes()).hexdigest() != metadata.get('binary_sha256'):
        raise RuntimeError('Native build/source mismatch; build/check before freezing R01')
    if _loaded is None:
        torch.ops.load_library(str(binary))
        _loaded=metadata
    return metadata

def configure(model,options):
    if not options.get('native_wkv'):
        return
    if options.get('row_tiles_exact') != 4 or options.get('recurrence_load_pipeline',0) or options.get('recurrence_prefetch') or options.get('wave_aware_backward') or options.get('compact_previous_state'):
        raise ValueError('Native WKV initially binds only accepted four-row/tile32 kernels')
    load()
    from scripts.gpu_row_tiles import compile_module
    from rwkv.model import nvrtc
    # The cached RawModule/RawKernel retains CUfunction ownership throughout forward/backward.
    forward,backward=compile_module(torch.device('cuda',torch.cuda.current_device()),torch.bfloat16,4,32,True,False,False,0)
    handles=(int(forward.kernel.ptr),int(backward.kernel.ptr))
    for module in model.modules():
        if hasattr(module,'native_wkv'):
            module.native_wkv=True
            module.native_forward_handle=handles[0]
            module.native_backward_handle=handles[1]
    return {'identity':_loaded['identity'],'binary_sha256':_loaded['binary_sha256'],'build_key':build_identity()[1]}
