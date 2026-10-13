"""F65a argument width, non-default-stream, asynchronous lifetime and launch cost screen."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    import torch
    import numpy as np
    from rwkv.model import nvrtc
    from scripts.experiment_provenance import process_provenance
    nvrtc.register()
    nvrtc.set_stream_launch(True)
    k = nvrtc._cupy.RawKernel('''extern "C" __global__ void transport(unsigned long long* out,
        unsigned long long u, long long s, int n) { int i=threadIdx.x; if(i<4) out[i]=i==0?u:i==1?(unsigned long long)s:i==2?(unsigned long long)n:99; }''', 'transport')
    output = torch.zeros(4, dtype=torch.uint64, device='cuda')
    args = (np.uint64(output.data_ptr()), np.uint64(0xffff123456789abc), np.int64(-123456789123), np.int32(-2147483647))
    cases = []
    for stream in (torch.cuda.current_stream(), torch.cuda.Stream()):
        for mode in (False, True):
            with torch.cuda.stream(stream):
                output.zero_()
                nvrtc.set_driver_launch(mode)
                nvrtc.raw_launch(k, output.device, (1,), (32,), args)
                # Reuse slots before the previous GPU operation finishes: launch copies arguments.
                other = torch.empty_like(output)
                nvrtc.raw_launch(k, other.device, (1,), (32,), (np.uint64(other.data_ptr()), *args[1:]))
            stream.synchronize()
            values = output.cpu().tolist()
            expected = [int(args[1]), int(args[2]) % (1 << 64), int(args[3]) % (1 << 64), 99]
            cases.append({'driver':mode, 'nondefault_stream':stream != torch.cuda.default_stream(), 'passed':values == expected and torch.equal(output,other)})
    times = []
    for repeat in range(4):
        for mode in ((False,True) if repeat%2==0 else (True,False)):
            nvrtc.set_driver_launch(mode)
            torch.cuda.synchronize()
            began = time.perf_counter()
            for _ in range(3000):
                nvrtc.raw_launch(k, output.device, (1,), (32,), args)
            dispatch = time.perf_counter()-began
            torch.cuda.synchronize()
            times.append({'repeat':repeat,'driver':mode,'calls':3000,'dispatch_us_per_call':dispatch*1e6/3000})
    nvrtc.set_driver_launch(False)
    record = {'status':'passed' if all(c['passed'] for c in cases) else 'failed','cases':cases,'launch_cost':times,
              'scope':'same function pointer, exact scalar widths, Torch current device/streams; host enqueue includes command-buffer backpressure; not training throughput',
              'provenance':process_provenance()}
    (a.output/'result.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':record['status'],'launch_cost':times}))
    if record['status'] != 'passed':
        raise SystemExit(1)

if __name__ == '__main__':
    main()
