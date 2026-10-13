#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda.h>
#include <array>
#include <limits>

using at::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

static void retain_on_current_stream(const Tensor& tensor) {
    c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(),
        c10::cuda::getCurrentCUDAStream(tensor.get_device()));
}

static void launch(int64_t handle, int b, int h, int rows, void** arguments, const Tensor& tensor) {
    TORCH_CHECK(handle != 0 && (rows == 1 || rows == 2 || rows == 4 || rows == 8), "Invalid native kernel handle/layout");
    auto stream = c10::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
    CUresult status = cuLaunchKernel(reinterpret_cast<CUfunction>(static_cast<uintptr_t>(handle)),
        b, h, 1, 32, 32 / rows, 1, 0, reinterpret_cast<CUstream>(stream), arguments, nullptr);
    TORCH_CHECK(status == CUDA_SUCCESS, "Native CUDA driver launch failed: ", static_cast<int>(status));
}

static void validate(const Tensor& r, const Tensor& k, const Tensor& v, const Tensor& w,
                     const Tensor& a, const Tensor& kd, const Tensor& skip) {
    TORCH_CHECK(r.is_cuda() && r.dim() == 4 && r.size(3) == 32 && r.is_contiguous(), "Invalid native WKV input");
    TORCH_CHECK(r.scalar_type() == at::kBFloat16, "Configured native WKV handles require BF16 payloads and FP32 decay");
    for (const auto& tensor : {k, v, a, kd})
        TORCH_CHECK(tensor.sizes() == r.sizes() && tensor.device() == r.device() && tensor.scalar_type() == r.scalar_type() && tensor.is_contiguous(), "Native WKV input mismatch");
    TORCH_CHECK(w.sizes() == r.sizes() && w.device() == r.device() && w.scalar_type() == at::kFloat && w.is_contiguous(), "Native WKV decay mismatch");
    TORCH_CHECK(skip.dim() == 2 && skip.size(0) == r.size(0) && skip.size(1) == r.size(1) && skip.device() == r.device() && skip.scalar_type() == at::kBool && skip.is_contiguous(), "Native WKV skip mismatch");
    for (int i = 0; i < 3; ++i) TORCH_CHECK(r.size(i) > 0 && r.size(i) <= std::numeric_limits<int>::max(), "Native WKV dimension overflow");
}

class NativeWKV : public torch::autograd::Function<NativeWKV> {
public:
    static Tensor forward(AutogradContext* ctx, Tensor r, Tensor k, Tensor v, Tensor w,
                          Tensor a, Tensor kd, Tensor skip, int64_t forward_handle,
                          int64_t backward_handle, int64_t rows, int64_t tile) {
        validate(r, k, v, w, a, kd, skip);
        TORCH_CHECK(tile == 8 || tile == 16 || tile == 32, "Invalid native checkpoint tile");
        c10::cuda::CUDAGuard guard(r.device());
        for (const auto& tensor : {r,k,v,w,a,kd,skip}) retain_on_current_stream(tensor);
        int b = static_cast<int>(r.size(0)), t = static_cast<int>(r.size(1)), h = static_cast<int>(r.size(2));
        int checkpoints = (t + static_cast<int>(tile) - 1) / static_cast<int>(tile);
        Tensor out = at::empty_like(r);
        Tensor states = at::empty({b, checkpoints, h, 32, 32}, r.options().dtype(at::kFloat));
        void *rp=r.data_ptr(), *kp=k.data_ptr(), *vp=v.data_ptr(), *wp=w.data_ptr(),
             *ap=a.data_ptr(), *kdp=kd.data_ptr(), *sp=skip.data_ptr(), *op=out.data_ptr(), *cp=states.data_ptr();
        std::array<void*,13> arguments = {&b,&t,&h,&rp,&kp,&vp,&wp,&ap,&kdp,&sp,&op,&checkpoints,&cp};
        launch(forward_handle,b,h,static_cast<int>(rows),arguments.data(),r);
        ctx->save_for_backward({r,k,v,w,a,kd,skip,states});
        ctx->saved_data["handle"] = backward_handle;
        ctx->saved_data["rows"] = rows;
        return out;
    }

    static variable_list backward(AutogradContext* ctx, variable_list output_gradients) {
        auto saved = ctx->get_saved_variables();
        Tensor r=saved[0], k=saved[1], v=saved[2], w=saved[3], a=saved[4], kd=saved[5], skip=saved[6], states=saved[7];
        Tensor grad = output_gradients[0].contiguous();
        TORCH_CHECK(grad.sizes() == r.sizes() && grad.scalar_type() == r.scalar_type() && grad.device() == r.device(), "Native WKV output gradient mismatch");
        c10::cuda::CUDAGuard guard(r.device());
        for (const auto& tensor : saved) retain_on_current_stream(tensor);
        retain_on_current_stream(grad);
        int b=static_cast<int>(r.size(0)), t=static_cast<int>(r.size(1)), h=static_cast<int>(r.size(2)), checkpoints=static_cast<int>(states.size(1));
        Tensor dr=at::empty_like(r), dk=at::empty_like(k), dv=at::empty_like(v), dw=at::empty_like(w), da=at::empty_like(a), dkd=at::empty_like(kd);
        void *rp=r.data_ptr(), *kp=k.data_ptr(), *vp=v.data_ptr(), *wp=w.data_ptr(), *ap=a.data_ptr(), *kdp=kd.data_ptr(),
             *sp=skip.data_ptr(), *gp=grad.data_ptr(), *cp=states.data_ptr(), *drp=dr.data_ptr(), *dkp=dk.data_ptr(),
             *dvp=dv.data_ptr(), *dwp=dw.data_ptr(), *dap=da.data_ptr(), *dkdp=dkd.data_ptr();
        std::array<void*,19> arguments = {&b,&t,&h,&rp,&kp,&vp,&wp,&ap,&kdp,&sp,&gp,&checkpoints,&cp,&drp,&dkp,&dvp,&dwp,&dap,&dkdp};
        launch(ctx->saved_data["handle"].toInt(),b,h,static_cast<int>(ctx->saved_data["rows"].toInt()),arguments.data(),r);
        return {dr,dk,dv,dw,da,dkd,Tensor(),Tensor(),Tensor(),Tensor(),Tensor()};
    }
};

static Tensor wkv(const Tensor& r, const Tensor& k, const Tensor& v, const Tensor& w,
                  const Tensor& a, const Tensor& kd, const Tensor& skip, int64_t forward_handle,
                  int64_t backward_handle, int64_t rows, int64_t tile) {
    return NativeWKV::apply(r,k,v,w,a,kd,skip,forward_handle,backward_handle,rows,tile);
}

TORCH_LIBRARY_IMPL(rwkv_native, CUDA, m) { m.impl("wkv", TORCH_FN(wkv)); }
TORCH_LIBRARY_IMPL(rwkv_native, Autograd, m) { m.impl("wkv", TORCH_FN(wkv)); }

// Draft addition to native_launch.cpp; not part of any running/frozen source.
#include <ATen/ops/native_group_norm.h>
#include <ATen/ops/native_group_norm_backward.h>

static void launch_norm(int64_t handle, unsigned int gx, unsigned int gy, void** args, const Tensor& tensor) {
    TORCH_CHECK(handle != 0, "Missing native GroupNorm function handle");
    auto stream=c10::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
    CUresult status=cuLaunchKernel(reinterpret_cast<CUfunction>(static_cast<uintptr_t>(handle)),
        gx,gy,1,32,1,1,0,reinterpret_cast<CUstream>(stream),args,nullptr);
    TORCH_CHECK(status==CUDA_SUCCESS,"Native GroupNorm launch failed: ",static_cast<int>(status));
}

class NativeGroupNorm : public torch::autograd::Function<NativeGroupNorm> {
public:
    static Tensor forward(AutogradContext* ctx, Tensor x, Tensor weight, Tensor bias,
                          int64_t groups, double eps, int64_t chains, int64_t tree) {
        TORCH_CHECK(x.is_cuda() && x.dim()==2 && x.size(1)==32 && x.is_contiguous() && x.scalar_type()==at::kBFloat16,"Native GroupNorm requires contiguous BF16 (N,32)");
        TORCH_CHECK(weight.device()==x.device() && bias.device()==x.device() && weight.scalar_type()==x.scalar_type() && bias.scalar_type()==x.scalar_type(),"Native GroupNorm requires BF16 affine parameters matching input");
        TORCH_CHECK(groups>0 && 32%groups==0,"Invalid native GroupNorm groups");
        c10::cuda::CUDAGuard guard(x.device());
        for (const auto& value:{x,weight,bias}) retain_on_current_stream(value);
        auto values=at::native_group_norm(x,weight,bias,x.size(0),32,1,groups,eps);
        Tensor mean=std::get<1>(values), rstd=std::get<2>(values);
        TORCH_CHECK(mean.scalar_type()==x.scalar_type() && rstd.scalar_type()==x.scalar_type(),"Native GroupNorm statistics must match input dtype");
        ctx->save_for_backward({x,weight,mean,rstd});
        ctx->saved_data["groups"]=groups;ctx->saved_data["chains"]=chains;ctx->saved_data["tree"]=tree;
        return std::get<0>(values);
    }
    static variable_list backward(AutogradContext* ctx,variable_list output_gradients) {
        auto saved=ctx->get_saved_variables();
        Tensor x=saved[0], weight=saved[1], mean=saved[2], rstd=saved[3], grad=output_gradients[0].contiguous();
        c10::cuda::CUDAGuard guard(x.device());
        for (const auto& value:saved) retain_on_current_stream(value);
        retain_on_current_stream(grad);
        int64_t n=x.size(0), groups=ctx->saved_data["groups"].toInt();
        Tensor dx,dgamma,dbeta;
        if (n<=128) {
            auto values=at::native_group_norm_backward(grad,x,mean,rstd,weight,n,32,1,groups,std::array<bool,3>{true,true,true});
            dx=std::get<0>(values);dgamma=std::get<1>(values);dbeta=std::get<2>(values);
        } else {
            auto values=at::native_group_norm_backward(grad,x,mean,rstd,weight,n,32,1,groups,std::array<bool,3>{true,false,false});
            dx=std::get<0>(values);
            Tensor partial=at::empty({2,32,32},x.options().dtype(at::kFloat));
            dgamma=at::empty({32},x.options());dbeta=at::empty({32},x.options());
            void *gp=grad.data_ptr(),*xp=x.data_ptr(),*mp=mean.data_ptr(),*rp=rstd.data_ptr(),*pp=partial.data_ptr();
            int32_t group_count=static_cast<int32_t>(groups);
            std::array<void*,7> arguments={&gp,&xp,&mp,&rp,&pp,&n,&group_count};
            launch_norm(ctx->saved_data["chains"].toInt(),32,2,arguments.data(),x);
            void *dgp=dgamma.data_ptr(),*dbp=dbeta.data_ptr();
            std::array<void*,3> tree_arguments={&pp,&dgp,&dbp};
            launch_norm(ctx->saved_data["tree"].toInt(),2,1,tree_arguments.data(),x);
        }
        return {dx,dgamma,dbeta,Tensor(),Tensor(),Tensor(),Tensor()};
    }
};

static Tensor native_group_norm(const Tensor& x,const Tensor& weight,const Tensor& bias,
                                int64_t groups,double eps,int64_t chains,int64_t tree) {
    return NativeGroupNorm::apply(x,weight,bias,groups,eps,chains,tree);
}
TORCH_LIBRARY_IMPL(rwkv_native,CUDA,m) { m.impl("group_norm",TORCH_FN(native_group_norm)); }
TORCH_LIBRARY_IMPL(rwkv_native,Autograd,m) { m.impl("group_norm",TORCH_FN(native_group_norm)); }

// F65d: exact shift-gather and padded row-select backward on the native path, same F26/F27 kernels.
#include <ATen/ops/gather.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/sort.h>
#include <ATen/ops/searchsorted.h>
#include <ATen/ops/arange.h>

static void launch_linear(int64_t handle, int64_t elements, void** args, const Tensor& tensor) {
    TORCH_CHECK(handle != 0, "Missing native index function handle");
    TORCH_CHECK(elements >= 0 && (elements + 255) / 256 <= std::numeric_limits<int>::max(), "Native index grid overflow");
    if (elements == 0) return;
    auto stream=c10::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
    CUresult status=cuLaunchKernel(reinterpret_cast<CUfunction>(static_cast<uintptr_t>(handle)),
        static_cast<unsigned int>((elements + 255) / 256),1,1,256,1,1,0,reinterpret_cast<CUstream>(stream),args,nullptr);
    TORCH_CHECK(status==CUDA_SUCCESS,"Native index launch failed: ",static_cast<int>(status));
}

class NativeShiftGather : public torch::autograd::Function<NativeShiftGather> {
public:
    static Tensor forward(AutogradContext* ctx, Tensor x, Tensor select, Tensor order, Tensor starts, int64_t handle) {
        TORCH_CHECK(x.is_cuda() && x.dim()==3 && select.dim()==2 && select.size(0)==x.size(0) && select.size(1)==x.size(1), "Native shift gather requires CUDA (B,T,C) input and (B,T) select");
        TORCH_CHECK(order.scalar_type()==at::kLong && starts.scalar_type()==at::kLong && order.is_contiguous() && starts.is_contiguous()
                    && order.numel()==select.numel() && starts.numel()==select.numel()+1, "Native shift gather order mismatch");
        ctx->save_for_backward({order, starts});
        ctx->saved_data["handle"]=handle;
        return at::gather(x, 1, select.unsqueeze(-1).expand({-1, -1, x.size(2)}));
    }
    static variable_list backward(AutogradContext* ctx, variable_list output_gradients) {
        auto saved=ctx->get_saved_variables();
        Tensor order=saved[0], starts=saved[1], grad=output_gradients[0].contiguous();
        TORCH_CHECK(grad.scalar_type()==at::kBFloat16 && grad.dim()==3, "Configured native shift handle requires BF16 gradients");
        c10::cuda::CUDAGuard guard(grad.device());
        for (const auto& value:{order,starts,grad}) retain_on_current_stream(value);
        Tensor out=at::empty_like(grad);
        int64_t rows=grad.size(0)*grad.size(1); int32_t C=static_cast<int32_t>(grad.size(2));
        void *gp=grad.data_ptr(), *op=order.data_ptr(), *sp=starts.data_ptr(), *outp=out.data_ptr();
        std::array<void*,6> arguments={&gp,&op,&sp,&outp,&rows,&C};
        launch_linear(ctx->saved_data["handle"].toInt(), rows*C, arguments.data(), grad);
        return {out, Tensor(), Tensor(), Tensor(), Tensor()};
    }
};

class NativePaddedRows : public torch::autograd::Function<NativePaddedRows> {
public:
    static Tensor forward(AutogradContext* ctx, Tensor x, Tensor index, int64_t handle) {
        TORCH_CHECK(x.is_cuda() && x.dim()==2 && x.size(1)<=32 && index.dim()==1, "Native row select requires 2-D CUDA rows of width <= 32");
        ctx->save_for_backward({index});
        ctx->saved_data["rows"]=x.size(0);
        ctx->saved_data["handle"]=handle;
        return at::index_select(x, 0, index.clamp(0));
    }
    static variable_list backward(AutogradContext* ctx, variable_list output_gradients) {
        auto saved=ctx->get_saved_variables();
        Tensor index=saved[0].contiguous().to(at::kLong), grad=output_gradients[0].contiguous();
        TORCH_CHECK(grad.scalar_type()==at::kBFloat16 && grad.dim()==2, "Configured native row handle requires BF16 gradients");
        c10::cuda::CUDAGuard guard(grad.device());
        int64_t rows=ctx->saved_data["rows"].toInt(); int32_t C=static_cast<int32_t>(grad.size(1));
        auto sorted=at::sort(index, true, -1, false);
        Tensor order=std::get<1>(sorted);
        Tensor starts=at::searchsorted(std::get<0>(sorted), at::arange(rows + 1, index.options()));
        for (const auto& value:{order,starts,grad}) retain_on_current_stream(value);
        Tensor out=at::empty({rows, grad.size(1)}, grad.options());
        void *gp=grad.data_ptr(), *op=order.data_ptr(), *sp=starts.data_ptr(), *outp=out.data_ptr();
        std::array<void*,6> arguments={&gp,&op,&sp,&outp,&rows,&C};
        launch_linear(ctx->saved_data["handle"].toInt(), rows*C, arguments.data(), grad);
        return {out, Tensor(), Tensor()};
    }
};

static Tensor shift_gather(const Tensor& x, const Tensor& select, const Tensor& order, const Tensor& starts, int64_t handle) {
    return NativeShiftGather::apply(x, select, order, starts, handle);
}
static Tensor padded_rows(const Tensor& x, const Tensor& index, int64_t handle) {
    return NativePaddedRows::apply(x, index, handle);
}
TORCH_LIBRARY_IMPL(rwkv_native,CUDA,m) { m.impl("shift_gather",TORCH_FN(shift_gather)); m.impl("padded_rows",TORCH_FN(padded_rows)); }
TORCH_LIBRARY_IMPL(rwkv_native,Autograd,m) { m.impl("shift_gather",TORCH_FN(shift_gather)); m.impl("padded_rows",TORCH_FN(padded_rows)); }
