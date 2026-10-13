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
