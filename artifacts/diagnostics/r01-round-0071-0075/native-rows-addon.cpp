// F65e: native transport for accepted padded row gather and F35 backward kernel.
class NativeRows : public torch::autograd::Function<NativeRows> {
public:
    static Tensor forward(AutogradContext* ctx,Tensor x,Tensor index,int64_t handle,bool padding) {
        TORCH_CHECK(x.is_cuda() && x.dim()==2 && x.size(1)>0 && x.size(1)<=32 && x.scalar_type()==at::kBFloat16,"Native row selection requires BF16 CUDA rows of width <=32");
        TORCH_CHECK(index.dim()==1 && index.device()==x.device() && index.scalar_type()==at::kLong,"Native row index mismatch");
        c10::cuda::CUDAGuard guard(x.device()); retain_on_current_stream(x);retain_on_current_stream(index);
        Tensor out=at::index_select(x,0,padding?index.clamp_min(0):index);
        ctx->save_for_backward({index});ctx->saved_data["rows"]=x.size(0);ctx->saved_data["handle"]=handle;
        return out;
    }
    static variable_list backward(AutogradContext* ctx,variable_list gradients) {
        Tensor grad=gradients[0].contiguous(),index=ctx->get_saved_variables()[0].contiguous();
        c10::cuda::CUDAGuard guard(grad.device());
        int64_t rows=ctx->saved_data["rows"].toInt();int c=static_cast<int>(grad.size(1));
        auto sorted=at::sort(index,std::optional<bool>(true),-1,false);
        Tensor order=std::get<1>(sorted),starts=at::searchsorted(std::get<0>(sorted),at::arange(rows+1,index.options()));
        Tensor out=at::empty({rows,c},grad.options());
        for(const auto& tensor:{grad,index,order,starts})retain_on_current_stream(tensor);
        void *gp=grad.data_ptr(),*op=order.data_ptr(),*sp=starts.data_ptr(),*yp=out.data_ptr();
        std::array<void*,6> args={&gp,&op,&sp,&yp,&rows,&c};
        int64_t handle=ctx->saved_data["handle"].toInt();TORCH_CHECK(handle!=0,"Missing native row kernel handle");
        auto stream=c10::cuda::getCurrentCUDAStream(grad.get_device()).stream();
        CUresult status=cuLaunchKernel(reinterpret_cast<CUfunction>(static_cast<uintptr_t>(handle)),
            static_cast<unsigned int>((rows*c+255)/256),1,1,256,1,1,0,reinterpret_cast<CUstream>(stream),args.data(),nullptr);
        TORCH_CHECK(status==CUDA_SUCCESS,"Native row launch failed: ",static_cast<int>(status));
        return {out,Tensor(),Tensor(),Tensor()};
    }
};
static Tensor rows(const Tensor& x,const Tensor& index,int64_t handle,bool padding){return NativeRows::apply(x,index,handle,padding);}
TORCH_LIBRARY_IMPL(rwkv_native,CUDA,m){m.impl("rows",TORCH_FN(rows));}
TORCH_LIBRARY_IMPL(rwkv_native,Autograd,m){m.impl("rows",TORCH_FN(rows));}
