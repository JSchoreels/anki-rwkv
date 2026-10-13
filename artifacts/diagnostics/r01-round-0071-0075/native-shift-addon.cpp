// F65d: unchanged stable shift orders and accepted backward kernel, native transport.
#include <ATen/ops/sort.h>
#include <ATen/ops/searchsorted.h>
#include <deque>
#include <mutex>
#include <unordered_set>

struct ShiftEntry {
    Tensor select, order, starts;
    uint32_t version;
    cudaStream_t stream;
};
static std::deque<ShiftEntry> shift_entries;
static std::mutex shift_mutex;

static std::pair<Tensor,Tensor> shift_order_native(const Tensor& select, bool shared) {
    int64_t b=select.size(0), t=select.size(1);
    auto stream=c10::cuda::getCurrentCUDAStream(select.get_device()).stream();
    auto version=select.unsafeGetTensorImpl()->version_counter().current_version();
    std::lock_guard<std::mutex> lock(shift_mutex);
    if (shared) for (const auto& entry:shift_entries) {
        // Strong Tensor ownership prevents allocator address reuse. Stream identity prevents reading orders before their producing stream.
        if (entry.select.unsafeGetTensorImpl()==select.unsafeGetTensorImpl() && entry.version==version &&
            entry.stream==stream && entry.select.sizes()==select.sizes() && entry.select.strides()==select.strides())
            return {entry.order,entry.starts};
    }
    Tensor keys=(at::arange(b,select.options()).unsqueeze(1)*t+select).reshape({-1});
    auto sorted=at::sort(keys,std::optional<bool>(true),-1,false);
    Tensor order=std::get<1>(sorted);
    Tensor starts=at::searchsorted(std::get<0>(sorted),at::arange(b*t+1,select.options()));
    if (shared) {
        if (shift_entries.size()>=64) shift_entries.pop_front();
        shift_entries.push_back({select,order,starts,version,stream});
    }
    return {order,starts};
}

class NativeShift : public torch::autograd::Function<NativeShift> {
public:
    static Tensor forward(AutogradContext* ctx,Tensor x,Tensor select,int64_t handle,bool shared) {
        TORCH_CHECK(x.is_cuda() && x.dim()==3 && x.scalar_type()==at::kBFloat16,"Native shift requires BF16 CUDA (B,T,C)");
        TORCH_CHECK(select.dim()==2 && select.size(0)==x.size(0) && select.size(1)==x.size(1) &&
            select.device()==x.device() && select.scalar_type()==at::kLong,"Native shift selection mismatch");
        c10::cuda::CUDAGuard guard(x.device());
        retain_on_current_stream(x); retain_on_current_stream(select);
        Tensor out=at::gather(x,1,select.unsqueeze(-1).expand({x.size(0),x.size(1),x.size(2)}));
        ctx->save_for_backward({select}); ctx->saved_data["handle"]=handle; ctx->saved_data["shared"]=shared;
        return out;
    }
    static variable_list backward(AutogradContext* ctx,variable_list gradients) {
        Tensor grad=gradients[0].contiguous(), select=ctx->get_saved_variables()[0].contiguous();
        c10::cuda::CUDAGuard guard(grad.device());
        auto orders=shift_order_native(select,ctx->saved_data["shared"].toBool());
        Tensor order=orders.first,starts=orders.second,out=at::empty_like(grad);
        for (const auto& tensor:{grad,select,order,starts}) retain_on_current_stream(tensor);
        int64_t rows=grad.size(0)*grad.size(1); int c=static_cast<int>(grad.size(2));
        TORCH_CHECK(grad.size(2)<=std::numeric_limits<int>::max(),"Native shift channel overflow");
        void *gp=grad.data_ptr(),*op=order.data_ptr(),*sp=starts.data_ptr(),*yp=out.data_ptr();
        std::array<void*,6> args={&gp,&op,&sp,&yp,&rows,&c};
        int64_t handle=ctx->saved_data["handle"].toInt(); TORCH_CHECK(handle!=0,"Missing native shift handle");
        auto stream=c10::cuda::getCurrentCUDAStream(grad.get_device()).stream();
        CUresult status=cuLaunchKernel(reinterpret_cast<CUfunction>(static_cast<uintptr_t>(handle)),
            static_cast<unsigned int>((rows*c+255)/256),1,1,256,1,1,0,reinterpret_cast<CUstream>(stream),args.data(),nullptr);
        TORCH_CHECK(status==CUDA_SUCCESS,"Native shift launch failed: ",static_cast<int>(status));
        return {out,Tensor(),Tensor(),Tensor()};
    }
};
static Tensor shift(const Tensor& x,const Tensor& select,int64_t handle,bool shared) {
    return NativeShift::apply(x,select,handle,shared);
}
static std::tuple<int64_t,int64_t,int64_t> shift_cache_stats(bool clear) {
    std::lock_guard<std::mutex> lock(shift_mutex);
    std::unordered_set<void*> owners;
    int64_t select_bytes=0,order_bytes=0;
    for (const auto& entry:shift_entries) {
        if (owners.insert(entry.select.storage().data_ptr().get()).second) select_bytes+=entry.select.storage().nbytes();
        order_bytes+=entry.order.nbytes()+entry.starts.nbytes();
    }
    auto result=std::make_tuple(static_cast<int64_t>(shift_entries.size()),select_bytes,order_bytes);
    if (clear) shift_entries.clear();
    return result;
}
TORCH_LIBRARY_IMPL(rwkv_native,CUDA,m) {m.impl("shift",TORCH_FN(shift));}
TORCH_LIBRARY_IMPL(rwkv_native,Autograd,m) {m.impl("shift",TORCH_FN(shift));}
TORCH_LIBRARY_IMPL(rwkv_native,CompositeExplicitAutograd,m) {m.impl("shift_cache_stats",TORCH_FN(shift_cache_stats));}
