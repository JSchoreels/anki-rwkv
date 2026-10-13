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
_schemas.define('group_norm(Tensor x, Tensor weight, Tensor bias, int groups, float eps, int chains, int tree) -> Tensor')
_schemas.define('shift_gather(Tensor x, Tensor select, Tensor order, Tensor starts, int handle) -> Tensor')
_schemas.define('padded_rows(Tensor x, Tensor index, int handle) -> Tensor')
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
    if not (options.get('native_wkv') or options.get('native_group_norm') or options.get('native_index')):
        return
    if options.get('native_wkv') and (options.get('row_tiles_exact') != 4 or options.get('recurrence_load_pipeline',0) or options.get('recurrence_prefetch') or options.get('wave_aware_backward') or options.get('compact_previous_state')):
        raise ValueError('Native WKV initially binds only accepted four-row/tile32 kernels')
    load()
    if options.get('native_wkv'):
        from scripts.gpu_row_tiles import compile_module
        # The cached RawModule/RawKernel retains CUfunction ownership through backward.
        forward,backward=compile_module(torch.device('cuda',torch.cuda.current_device()),torch.bfloat16,4,32,True,False,False,0)
        handles=(int(forward.kernel.ptr),int(backward.kernel.ptr))
        for module in model.modules():
            if hasattr(module,'native_wkv'):
                module.native_wkv=True
                module.native_forward_handle=handles[0]
                module.native_backward_handle=handles[1]
    if options.get('native_group_norm'):
        if not options.get('exact_group_norm'):
            raise ValueError('Native GroupNorm requires the accepted exact normalization path')
        from rwkv.model.exact_group_norm import kernel
        functions=kernel(torch.bfloat16)
        handles=[int(functions[name].kernel.ptr) for name in ('gamma_beta_chains','gamma_beta_tree')]
        for module in model.modules():
            if hasattr(module,'native_group_norm'):
                module.native_group_norm=True
                module.native_group_chains_handle=handles[0]
                module.native_group_tree_handle=handles[1]
    if options.get('native_index'):
        # F65d: same F26/F27 (F35 prefetch) kernels; orders still come from the F38 cache or the backward sort.
        if not (options.get('exact_shift') and options.get('exact_index')) or options.get('worker_scatter_orders') or options.get('fused_shift'):
            raise ValueError('Native index requires the accepted exact shift/row paths without F53 or fused shift')
        from rwkv.model.exact_index import kernel
        functions=kernel(torch.bfloat16)
        shift=functions['shift_backward_prefetch' if options.get('prefetch_shift') else 'shift_backward']
        rows=functions['rows_backward_prefetch' if options.get('prefetch_rows') else 'rows_backward']
        handles=[int(getattr(f,'kernel',f).ptr) for f in (shift,rows)]
        for module in model.modules():
            if hasattr(module,'native_shift'):
                module.native_shift=True
                module.native_shift_handle=handles[0]
            if hasattr(module,'native_rows'):
                module.native_rows=True
                module.native_rows_handle=handles[1]
        configured=[sum(bool(getattr(m,name,False)) for m in model.modules()) for name in ('native_shift','native_rows')]
        if not all(configured):
            raise ValueError(f'Native index found no shift/row modules to configure: {configured}')
    return {'identity':_loaded['identity'],'binary_sha256':_loaded['binary_sha256'],'build_key':build_identity()[1]}
