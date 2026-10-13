"""F34 pointwise glue; preserve dtype rounding at every original op boundary."""
import numpy as np
import torch
from rwkv.model import nvrtc

_kernels = {}

def kernel(dtype, name):
    key = dtype, name
    if key not in _kernels:
        from scripts.gpu_row_tiles import source_options
        prefix, options = source_options(dtype, exact=True)
        prefix = prefix.split('namespace rwkv_exact {')[0]
        scalar = 'float' if dtype == torch.float32 else '__nv_bfloat16'
        code = prefix + f'''
typedef {scalar} G;
__device__ float rd(G x) {{ return (float)x; }}
__device__ G wr(float x) {{ return (G)x; }}
extern "C" __global__ void gate_forward(const G* a,const G* b,const G* g,G* s,G* y,int n) {{
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
 G z=wr(__fadd_rn(rd(a[i]),rd(b[i]))); s[i]=z; y[i]=wr(__fmul_rn(rd(g[i]),rd(z)));
}}
extern "C" __global__ void gate_backward(const G* dy,const G* s,const G* g,G* ds,G* dg,int n) {{
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
 ds[i]=wr(__fmul_rn(rd(dy[i]),rd(g[i]))); dg[i]=wr(__fmul_rn(rd(dy[i]),rd(s[i])));
}}
extern "C" __global__ void channel_forward(const G* x,G* r,G* y,int n) {{
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
 float a=rd(x[i]); G z=wr(a<0.0f?0.0f:a); r[i]=z; y[i]=wr(__fmul_rn(rd(z),rd(z)));
}}
'''
        nvrtc.register()
        _kernels[key] = nvrtc._cupy.RawKernel(code, name, options=options)
    return _kernels[key]

def launch(name, inputs, outputs):
    x = inputs[0]
    args = tuple(np.uint64(t.data_ptr()) for t in (*inputs, *outputs)) + (np.int32(x.numel()),)
    nvrtc.raw_launch(kernel(x.dtype, name), x.device, ((x.numel()+255)//256,), (256,), args)

class ExactGateGlue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, g, fused_backward):
        a,b,g = a.contiguous(),b.contiguous(),g.contiguous()
        s,y = torch.empty_like(a),torch.empty_like(a)
        launch('gate_forward',(a,b,g),(s,y))
        ctx.save_for_backward(s,g)
        ctx.fused_backward = fused_backward
        return y
    @staticmethod
    def backward(ctx, dy):
        s,g = ctx.saved_tensors
        if ctx.fused_backward:
            dy=dy.contiguous()
            ds,dg=torch.empty_like(s),torch.empty_like(g)
            launch('gate_backward',(dy,s,g),(ds,dg))
        else:
            ds,dg=dy*g,dy*s
        return ds,ds,dg,None

class ExactChannelGlue(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        x=x.contiguous()
        r,y=torch.empty_like(x),torch.empty_like(x)
        launch('channel_forward',(x,),(r,y))
        ctx.save_for_backward(r)
        return y
    @staticmethod
    def backward(ctx,dy):
        r,=ctx.saved_tensors
        return torch.ops.aten.threshold_backward(dy*(2*r),r,0)
