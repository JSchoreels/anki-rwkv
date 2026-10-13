"""Audit actual Torch CUDA-runtime exports and native importlib/DLL dependencies; no GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    from scripts.experiment_provenance import process_provenance
    provenance=process_provenance()
    import torch
    import psutil
    from rwkv.model.native_launch import build_identity,load
    from scripts.experiment_provenance import process_provenance
    identity,key=build_identity()
    folder=ROOT/'artifacts/cache/r01-native'/key
    tool=Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC')/identity['msvc_version']/'bin/Hostx64/x64/dumpbin.exe'
    runtime=Path(torch.__file__).parent/'lib/cudart64_12.dll'
    exports=subprocess.run([str(tool),'/exports',str(runtime)],capture_output=True,text=True,check=True).stdout
    imports=subprocess.run([str(tool),'/dependents',str(folder/'native_launch.dll')],capture_output=True,text=True,check=True).stdout
    (args.output/'runtime-exports.log').write_text(exports,encoding='utf-8')
    (args.output/'native-dependencies.log').write_text(imports,encoding='utf-8')
    symbols=(folder/'torch_cudart.def').read_text().split('EXPORTS\n')[1].splitlines()
    def runtimes():
        return sorted({row.path for row in psutil.Process().memory_maps() if 'cudart' in row.path.lower()})
    before=runtimes();load();after=runtimes()
    exact_exports=all(any(line.rstrip().endswith(' '+symbol) for line in exports.splitlines()) for symbol in symbols)
    passed=exact_exports and 'cudart64_12.dll' in imports and 'cudart64_13.dll' not in imports and not set(after)-set(before)
    record={'status':'passed' if passed else 'failed','required_runtime_exports':symbols,'required_exports_present':exact_exports,
            'torch_runtime_sha256':hashlib.sha256(runtime.read_bytes()).hexdigest(),
            'import_library_sha256':hashlib.sha256((folder/'torch_cudart.lib').read_bytes()).hexdigest(),
            'native_binary_sha256':hashlib.sha256((folder/'native_launch.dll').read_bytes()).hexdigest(),
            'loaded_runtime_paths_before':before,'loaded_runtime_paths_after':after,'build_identity':identity,'build_key':key,
            'scope':'actualDLLexport and linkerdependency audit plus before/after own-process runtimeDLL inventory; no GPU/parity claim',
            'provenance':provenance}
    (args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':record['status'],'exports':len(symbols),'required_exports_present':exact_exports,'added_runtime_libraries':sorted(set(after)-set(before))}))
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
