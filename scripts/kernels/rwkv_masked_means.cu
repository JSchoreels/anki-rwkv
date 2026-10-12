// F08: nine masked FP32 means, four counts, and gradients of seven loss arrays.
// Partial sums and their final reduction have a fixed order; no global atomics.
__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}
__device__ __forceinline__ float block_sum(float value, float* shared) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    value = warp_sum(value);
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? shared[lane] : 0.0f;
    if (warp == 0) value = warp_sum(value);
    return value;
}
__device__ __forceinline__ int count_field(int field) {
    return field == 2 ? 10 : (field == 6 || field == 7) ? 11 : field == 8 ? 12 : 9;
}

extern "C" __global__ void masked_partial(
    const float* curve, const float* raw, const float* ratings, const float* kl,
    const float* magnitude, const float* difference, const float* binary,
    const int* ahead, const int* immediate, const int* equal_ahead, const int* equal_immediate,
    float* partials, int n, int chunks) {
    const int i = blockIdx.x * 256 + threadIdx.x, field = blockIdx.y;
    float value = 0;
    if (i < n) {
        switch (field) {
            case 0: value = curve[i] * ahead[i]; break;
            case 1: value = raw[i] * ahead[i]; break;
            case 2: value = ratings[i] * immediate[i]; break;
            case 3: value = kl[i] * ahead[i]; break;
            case 4: value = magnitude[i] * ahead[i]; break;
            case 5: value = difference[i] * ahead[i]; break;
            case 6: value = curve[i] * equal_ahead[i]; break;
            case 7: value = raw[i] * equal_ahead[i]; break;
            case 8: value = binary[i] * equal_immediate[i]; break;
            case 9: value = ahead[i]; break;
            case 10: value = immediate[i]; break;
            case 11: value = equal_ahead[i]; break;
            case 12: value = equal_immediate[i]; break;
        }
    }
    __shared__ float shared[8];
    value = block_sum(value, shared);
    if (threadIdx.x == 0) partials[field * chunks + blockIdx.x] = value;
}

extern "C" __global__ void masked_finish(const float* partials, float* output, int chunks) {
    const int field = blockIdx.x;
    float numerator = 0, denominator = 0;
    for (int i = threadIdx.x; i < chunks; i += 256) {
        numerator += partials[field * chunks + i];
        if (field < 9) denominator += partials[count_field(field) * chunks + i];
    }
    __shared__ float shared[8];
    numerator = block_sum(numerator, shared);
    __syncthreads();  // Do not overwrite shared memory until numerator is consumed.
    denominator = block_sum(denominator, shared);
    if (threadIdx.x == 0)
        output[field] = field < 9 ? numerator / (1e-8f + denominator) : numerator;
}

extern "C" __global__ void masked_backward(
    const float* gradient, const int* ahead, const int* immediate,
    const int* equal_ahead, const int* equal_immediate, const float* output,
    float* dcurve, float* draw, float* dratings, float* dkl,
    float* dmagnitude, float* ddifference, float* dbinary, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const float da = 1e-8f + output[9], di = 1e-8f + output[10];
    const float dea = 1e-8f + output[11], dei = 1e-8f + output[12];
    dcurve[i] = (gradient[0] / da) * ahead[i] + (gradient[6] / dea) * equal_ahead[i];
    draw[i] = (gradient[1] / da) * ahead[i] + (gradient[7] / dea) * equal_ahead[i];
    dratings[i] = (gradient[2] / di) * immediate[i];
    dkl[i] = (gradient[3] / da) * ahead[i];
    dmagnitude[i] = (gradient[4] / da) * ahead[i];
    ddifference[i] = (gradient[5] / da) * ahead[i];
    dbinary[i] = (gradient[8] / dei) * equal_immediate[i];
}
