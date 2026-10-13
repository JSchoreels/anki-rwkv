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
        TORCH_CHECK(weight.device()==x.device() && bias.device()==x.device() && weight.scalar_type()==at::kFloat && bias.scalar_type()==at::kFloat,"Native GroupNorm requires FP32 affine parameters");
        TORCH_CHECK(groups>0 && 32%groups==0,"Invalid native GroupNorm groups");
        c10::cuda::CUDAGuard guard(x.device());
        for (const auto& value:{x,weight,bias}) retain_on_current_stream(value);
        auto values=at::native_group_norm(x,weight,bias,x.size(0),32,1,groups,eps);
        Tensor mean=std::get<1>(values), rstd=std::get<2>(values);
        TORCH_CHECK(mean.scalar_type()==at::kFloat && rstd.scalar_type()==at::kFloat,"Native GroupNorm statistics must be FP32");
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
