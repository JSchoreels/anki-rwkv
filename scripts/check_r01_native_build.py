"""Compile the host-only native dispatcher with scoped MSVC/SDK environment; no GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    from scripts.experiment_provenance import process_provenance
    provenance=process_provenance()
    from rwkv.model.native_launch import build_identity, BUILD_FLAGS
    from scripts.experiment_provenance import process_provenance
    import torch
    identity,key=build_identity()
    destination=ROOT/'artifacts/cache/r01-native'/key
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/'build.json').exists():
        metadata=json.loads((destination/'build.json').read_text(encoding='utf-8'))
        if metadata['identity']!=identity or hashlib.sha256((destination/'native_launch.dll').read_bytes()).hexdigest()!=metadata['binary_sha256']:
            raise RuntimeError('Existing native cache build differs; preserve it')
    else:
        candidates=[Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat')]
        vcvars=next((path for path in candidates if path.is_file()),None)
        if vcvars is None:raise RuntimeError('Authorized Build Tools installation is not yet available')
        # Pass CMD's own quotation syntax as raw command text; list2cmdline's backslash
        # escaping is for C runtimes and turns vcvars' quotes into literal backslashes.
        captured=subprocess.run(f'cmd.exe /d /s /c "call "{vcvars}" >nul && set"',capture_output=True,text=True)
        if captured.returncode:
            (args.output/'vcvars-error.log').write_text(captured.stderr,encoding='utf-8')
            raise RuntimeError('vcvars environment capture failed; inspect bounded stderr file')
        environment={name.upper():value for name,value in os.environ.items()}
        for line in captured.stdout.splitlines():
            name,sep,value=line.partition('=')
            if sep and name:environment[name.upper()]=value
        torch_root=Path(torch.__file__).parent
        cuda_root=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4')
        if environment.get('VCTOOLSVERSION','').rstrip('\\/') != identity['msvc_version'] or environment.get('WINDOWSSDKVERSION','').rstrip('\\/') != identity['windows_sdk_version']:
            raise RuntimeError('vcvars selected a different compiler/SDK from the frozen build identity')
        compiler=Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC')/identity['msvc_version']/'bin/Hostx64/x64/cl.exe'
        # CUDAGuard's inline vtable references runtime APIs. Link to the exact runtime
        # already loaded by Torch, rather than adding the CUDA13 toolkit's second runtime.
        symbols=['cudaDeviceSynchronize','cudaGetLastError','cudaGetErrorString','cudaStreamWaitEvent',
                 'cudaEventCreateWithFlags','cudaEventRecord','cudaEventQuery','cudaEventSynchronize',
                 'cudaEventDestroy','cudaEventElapsedTime']
        definition=destination/'torch_cudart.def'
        definition.write_text('LIBRARY cudart64_12.dll\nEXPORTS\n'+'\n'.join(symbols)+'\n',encoding='utf-8')
        with (args.output/'import-library.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([str(compiler.parent/'lib.exe'),'/nologo','/machine:x64',f'/def:{definition}',
                                   f'/out:{destination / "torch_cudart.lib"}'],cwd=destination,env=environment,
                                   stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Torch runtime import library generation failed')
        command=[str(compiler),*BUILD_FLAGS,f'/I{torch_root / "include"}',
                 f'/I{torch_root / "include/torch/csrc/api/include"}',f'/I{cuda_root / "include"}',
                 str(ROOT/'rwkv/model/csrc/native_launch.cpp'),f'/Fo{destination / "native_launch.obj"}',
                 '/link',f'/OUT:{destination / "native_launch.dll"}',f'/LIBPATH:{torch_root / "lib"}',
                 f'/LIBPATH:{cuda_root / "lib/x64"}','c10.lib','c10_cuda.lib','torch_cpu.lib','torch_cuda.lib','torch.lib','cuda.lib',str(destination/'torch_cudart.lib')]
        with (args.output/'compile.log').open('w',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=destination,env=environment,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Native host build failed; inspect {args.output / "compile.log"}')
        metadata={'status':'built','identity':identity,'build_key':key,'command':command,
                  'vcvars':str(vcvars),'VCToolsVersion':environment.get('VCTOOLSVERSION'),
                  'WindowsSDKVersion':environment.get('WINDOWSSDKVERSION'),'binary_sha256':hashlib.sha256((destination/'native_launch.dll').read_bytes()).hexdigest()}
        (destination/'build.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    record={'status':'passed','scope':'MSVC host-only dispatcher/autograd build; no GPU or numerical check',
            'build':metadata,'provenance':provenance}
    (args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','build_key':key,'binary_sha256':metadata['binary_sha256']}))

if __name__=='__main__':main()
