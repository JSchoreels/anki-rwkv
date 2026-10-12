// F08 opt-in: GroupNorm(K=32), recurrence bonus and output gate.
// BF16 pointwise intermediates round at the original eager operation boundaries.
#ifdef USE_BF16
#include <cuda_bf16.h>
typedef __nv_bfloat16 F;
__device__ __forceinline__ float read(F value) { return __bfloat162float(value); }
__device__ __forceinline__ F write(float value) { return __float2bfloat16(value); }
#else
typedef float F;
__device__ __forceinline__ float read(F value) { return value; }
__device__ __forceinline__ F write(float value) { return value; }
#endif
__device__ __forceinline__ float rounded(float value) { return read(write(value)); }
__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return __shfl_sync(0xffffffffu, value, 0);
}

extern "C" __global__ void fused_post_forward(
    const F* x, const F* r, const F* k, const F* v, const F* g,
    const F* bonus, const F* gamma, const F* beta,
    F* output, float* means, float* inverses, int groups, int heads, float eps) {
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = thread / 32, lane = thread % 32;
    if (row >= groups) return;  // Entire warp exits together.
    const int i = row * 32 + lane, parameter = (row % heads) * 32 + lane;
    const float value = read(x[i]);
    const float unrounded_mean = warp_sum(value) / 32.0f;
    const float delta = value - unrounded_mean;
    // PyTorch CUDA GroupNorm stores moments and casts epsilon in input dtype.
    const float mean = rounded(unrounded_mean);
    const float inv = rounded(rsqrtf(warp_sum(delta * delta) / 32.0f + rounded(eps)));
    const float centered = value - mean;
    const float normalized = rounded(centered * inv * read(gamma[parameter]) + read(beta[parameter]));
    const float rk = rounded(rounded(read(r[i]) * read(bonus[parameter])) * read(k[i]));
    const float inner = rounded(warp_sum(rk));
    const float extra = rounded(inner * read(v[i]));
    const float sum = rounded(normalized + extra);
    output[i] = write(read(g[i]) * sum);
    if (lane == 0) { means[row] = mean; inverses[row] = inv; }
}

extern "C" __global__ void fused_post_backward(
    const F* gradient, const F* x, const F* r, const F* k, const F* v,
    const F* g, const F* bonus, const F* gamma, const F* beta,
    const float* means, const float* inverses,
    F* dx, F* dr, F* dk, F* dv, F* dg,
    float* dbonus, float* dgamma, float* dbeta, int groups, int heads) {
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = thread / 32, lane = thread % 32;
    if (row >= groups) return;
    const int i = row * 32 + lane, parameter = (row % heads) * 32 + lane;
    const float mean = means[row], inv = inverses[row];
    const float centered = read(x[i]) - mean, xhat = centered * inv;
    const float normalized = rounded(xhat * read(gamma[parameter]) + read(beta[parameter]));
    const float r_bonus = rounded(read(r[i]) * read(bonus[parameter]));
    const float inner = rounded(warp_sum(rounded(r_bonus * read(k[i]))));
    const float extra = rounded(inner * read(v[i]));
    const float dy = rounded(read(gradient[i]) * read(g[i]));
    dg[i] = write(read(gradient[i]) * rounded(normalized + extra));
    dv[i] = write(dy * inner);
    const float dinner = rounded(warp_sum(rounded(dy * read(v[i]))));
    dk[i] = write(dinner * r_bonus);
    const float d_r_bonus = rounded(dinner * read(k[i]));
    dr[i] = write(d_r_bonus * read(bonus[parameter]));
    atomicAdd(dbonus + parameter, rounded(d_r_bonus * read(r[i])));
    // Follow Torch 2.9.1's HxW=1 CUDA coefficient formulas. In its sum1,
    // dY*X is an input-dtype operation before multiplication by FP32 gamma.
    const float sum1 = warp_sum(rounded(dy * read(x[i])) * read(gamma[parameter]));
    const float sum2 = warp_sum(dy * read(gamma[parameter]));
    const float c2 = (sum2 * mean - sum1) * inv * inv * inv / 32.0f;
    const float c3 = -c2 * mean - sum2 * inv / 32.0f;
    dx[i] = write((inv * read(gamma[parameter])) * dy + c2 * read(x[i]) + c3);
    atomicAdd(dgamma + parameter, (dy * read(x[i]) - dy * mean) * inv);
    atomicAdd(dbeta + parameter, dy);
}
