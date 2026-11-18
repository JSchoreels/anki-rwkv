// S09 (docs/GPU_OPTIMIZATIONS.md) / SM32b: fused forward of the state-VQ QAT recurrence of one card/note WKV layer.
// One block of 16 x 16 threads per (row b, head h) runs the whole sequence of a split (thread (x, y) owns S[x][y]),
// so a split costs one launch instead of ~200 PyTorch ops per step. Per step, the same arithmetic as
// expertium/model/state_vq_hooks.py VQStateWKV.forward with scripts/state_vq.py StateCodec.wkv, factor "sq":
//   nxt = S diag(w) - (S kd)(a kd)^T + v k^T ;  out = nxt r  (the answer uses the float update)
//   skipped step: S stays;  else S = codec(nxt):
//     An = nxt / max|nxt|, G = An An^T, P = G / tr G, 12 squarings P <- P P (trace renormalised before every 4th),
//     p = G P[:, j] (j = argmax diag P, first on ties), u = p / |p|, ev = u^T G u, v_un = nxt^T u, sigma = |v_un|,
//     valid = finite & ev > 0 & sigma > 1e-20, v = v_un / sigma, sign canon on u's largest |entry| (first on ties),
//     joint code q = (u, v) (32 floats): residual VQ, stage s picks argmin_c |c|^2 - 2 <q - acc, c> (first on ties),
//     acc += c; norm n = sqrt(sigma) on the uniform log2 grid; S = valid ? q(n)^2 acc_u acc_v^T : 0.
// Writes out [B, T, H, 16] and the state BEFORE every step [B, T, H, 16, 16]: the backward is the plain WKV backward
// along that (quantized) trajectory = rwkv7_wkv_backward_kernel<1, float> of rwkv/model/nvrtc.py (checkpoint every step).
// SM32c (learned codebooks, carried state): four optional pointers, nullptr = the SM32b kernel exactly (same arithmetic,
// same writes). s0_ [B, H, 16, 16]: the state step 0 starts from (a carried, already coded state; nullptr = zeros).
// final_ [B, H, 16, 16]: the stored state after the last step. picks_ [B, T, H, stages] int32: stage picks of every
// coded step (untouched on skipped steps, the caller fills -1). cscale_ [B, T, H]: q(n)^2 of every coded step, 0 when
// the factorization is invalid (stored state 0); the codebook gradient is cscale (G acc_v, G^T acc_u) per pick.
// FT = factorization type (float, or double with -DFACTOR_DOUBLE); the VQ search is float32 as in the torch codec.
#define KD 16
#define NT 256
#define FULL_MASK 0xffffffffu
#ifndef SQ_SQUARINGS
#define SQ_SQUARINGS 12
#endif
#ifndef SQ_RENORM
#define SQ_RENORM 4
#endif
#ifdef FACTOR_DOUBLE
typedef double FT;
__device__ __forceinline__ double ft_sqrt(double x) { return sqrt(x); }
__device__ __forceinline__ double ft_abs(double x) { return fabs(x); }
#else
typedef float FT;
__device__ __forceinline__ float ft_sqrt(float x) { return sqrtf(x); }
__device__ __forceinline__ float ft_abs(float x) { return fabsf(x); }
#endif
typedef long long i64;

template <typename X> __device__ __forceinline__ bool finite_value(X x) { return (x - x) == (X)0; }

// Max over the block's 256 values; every thread gets the result. `red` holds NT / 32 floats.
__device__ __forceinline__ float block_max(float value, float* red, int tid) {
    for (int offset = 16; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, offset));
    if ((tid & 31) == 0) red[tid >> 5] = value;
    __syncthreads();
    float m = red[0];
    for (int i = 1; i < NT / 32; i++) m = fmaxf(m, red[i]);
    __syncthreads();
    return m;
}

// Index of the smallest (value, index) pair over the block (lowest index on equal values = torch.argmin).
__device__ __forceinline__ int block_argmin(float value, int index, float* redv, int* redi, int tid) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float ov = __shfl_xor_sync(FULL_MASK, value, offset);
        const int oi = __shfl_xor_sync(FULL_MASK, index, offset);
        if (ov < value || (ov == value && oi < index)) { value = ov; index = oi; }
    }
    if ((tid & 31) == 0) { redv[tid >> 5] = value; redi[tid >> 5] = index; }
    __syncthreads();
    float bv = redv[0];
    int bi = redi[0];
    for (int i = 1; i < NT / 32; i++) {
        const float ov = redv[i];
        const int oi = redi[i];
        if (ov < bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __syncthreads();
    return bi;
}

extern "C" __global__ void __launch_bounds__(NT) state_vq_wkv_forward(
    const int B, const int T, const int H,
    const float* __restrict__ r_, const float* __restrict__ k_, const float* __restrict__ v_,
    const float* __restrict__ w_, const float* __restrict__ a_, const float* __restrict__ kd_,
    const bool* __restrict__ skip_,
    float* __restrict__ out_,      // [B, T, H, KD]
    float* __restrict__ before_,   // [B, T, H, KD, KD]
    const float* __restrict__ books_,  // [stages, 2 KD, M]: codebooks transposed (dimension-major, coalesced reads)
    const float* __restrict__ cn2_,    // [stages, M]: squared centroid norms (the codec's wkv_n2)
    const int stages, const int M,
    const int norm_mode,           // 1 = uniform log2 grid (norm_lo, norm_span, norm_levels), 0 = fixed norm_fixed
    const float norm_lo, const float norm_span, const float norm_levels, const float norm_fixed,
    const float* __restrict__ s0_,     // [B, H, KD, KD] or nullptr (zeros)
    float* __restrict__ final_,        // [B, H, KD, KD] or nullptr
    int* __restrict__ picks_,          // [B, T, H, stages] or nullptr
    float* __restrict__ cscale_) {     // [B, T, H] or nullptr
    __shared__ FT sN[KD][KD + 1];   // nxt
    __shared__ FT sA[KD][KD + 1];   // nxt / amax
    __shared__ FT sG[KD][KD + 1];   // Gram
    __shared__ FT sP[KD][KD + 1];   // normalised powers of the Gram
    __shared__ FT sVec[KD], sU[KD], sGu[KD], sAv[KD];
    __shared__ float sQ[2 * KD], sAcc[2 * KD], sRes[2 * KD];
    __shared__ float sRedV[NT / 32];
    __shared__ int sRedI[NT / 32];
    const int b = blockIdx.x, h = blockIdx.y;
    const int x = threadIdx.y, y = threadIdx.x, tid = x * KD + y;
    float state = s0_ ? s0_[(((i64)b * H + h) * KD + x) * KD + y] : 0.0f;
    for (int t = 0; t < T; t++) {
        const i64 base = (((i64)b * T + t) * H + h) * KD;
        before_[(base + x) * KD + y] = state;
        const float r_y = r_[base + y], k_y = k_[base + y], v_x = v_[base + x];
        const float w_y = w_[base + y], a_y = a_[base + y], kd_y = kd_[base + y];
        const bool skip = skip_[(i64)b * T + t];
        float skd = state * kd_y;   // (S kd)[x], a 16-lane row reduction
        for (int offset = 8; offset > 0; offset >>= 1) skd += __shfl_xor_sync(FULL_MASK, skd, offset, 16);
        const float nxt = state * w_y - skd * (a_y * kd_y) + v_x * k_y;
        float o = nxt * r_y;
        for (int offset = 8; offset > 0; offset >>= 1) o += __shfl_xor_sync(FULL_MASK, o, offset, 16);
        if (y == 0) out_[base + x] = o;
        if (skip) continue;   // skip_[b, t] is the same for the whole block: the stored state stays S

        // ---- rank-1 factorization by Gram squaring (scripts/state_vq.py factorize_sq)
        const float amax = block_max(fabsf(nxt), sRedV, tid);
        const FT scale = (finite_value(amax) && amax > 1e-30f) ? (FT)amax : (FT)1;
        const FT nd = (FT)nxt;
        sN[x][y] = nd;
        sA[x][y] = nd / scale;
        __syncthreads();
        FT g = 0;
        for (int i = 0; i < KD; i++) g += sA[x][i] * sA[y][i];
        sG[x][y] = g;
        sP[x][y] = g;
        __syncthreads();
        for (int i = 0; i < SQ_SQUARINGS; i++) {
            if (i % SQ_RENORM == 0) {
                FT tr = 0;
                for (int j = 0; j < KD; j++) tr += sP[j][j];
                const FT p = sP[x][y] / (tr > 0 ? tr : (FT)1);
                __syncthreads();
                sP[x][y] = p;
                __syncthreads();
            }
            FT acc = 0;
            for (int j = 0; j < KD; j++) acc += sP[x][j] * sP[j][y];
            __syncthreads();
            sP[x][y] = acc;
            __syncthreads();
        }
        int jc = 0;
        FT best = sP[0][0];
        for (int j = 1; j < KD; j++) if (sP[j][j] > best) { best = sP[j][j]; jc = j; }
        if (tid < KD) {
            FT p = 0;
            for (int j = 0; j < KD; j++) p += sG[tid][j] * sP[j][jc];
            sVec[tid] = p;
        }
        __syncthreads();
        FT pn2 = 0;
        for (int j = 0; j < KD; j++) pn2 += sVec[j] * sVec[j];
        const FT pn = ft_sqrt(pn2);
        if (tid < KD) sU[tid] = sVec[tid] / (pn > 0 ? pn : (FT)1);
        __syncthreads();
        if (tid < KD) {
            FT gu = 0, av = 0;
            for (int j = 0; j < KD; j++) { gu += sG[tid][j] * sU[j]; av += sN[j][tid] * sU[j]; }
            sGu[tid] = gu;   // (G u)[tid]
            sAv[tid] = av;   // (nxt^T u)[tid]
        }
        __syncthreads();
        FT ev = 0, s2 = 0;
        for (int j = 0; j < KD; j++) { ev += sU[j] * sGu[j]; s2 += sAv[j] * sAv[j]; }
        const FT sigma = ft_sqrt(s2);
        const bool valid = finite_value(ev) && ev > 0 && finite_value(sigma) && sigma > (FT)1e-20;
        int lead = 0;
        FT lm = ft_abs(sU[0]);
        for (int j = 1; j < KD; j++) if (ft_abs(sU[j]) > lm) { lm = ft_abs(sU[j]); lead = j; }
        const FT sign = sU[lead] >= 0 ? (FT)1 : (FT)-1;
        const FT vdiv = valid ? sigma : (FT)1;
        if (tid < KD) sQ[tid] = (float)(sU[tid] * sign);
        else if (tid < 2 * KD) sQ[tid] = (float)((sAv[tid - KD] / vdiv) * sign);
        if (tid < 2 * KD) sAcc[tid] = 0.0f;
        __syncthreads();

        // ---- joint (u, v) residual VQ (scripts/state_vq.py rvq / nearest)
        for (int s = 0; s < stages; s++) {
            if (tid < 2 * KD) sRes[tid] = sQ[tid] - sAcc[tid];
            __syncthreads();
            const float* book = books_ + (i64)s * 2 * KD * M;
            const float* n2 = cn2_ + (i64)s * M;
            float bv = __int_as_float(0x7f800000);
            int bi = tid < M ? tid : 0;
            for (int c = tid; c < M; c += NT) {
                float dot = 0.0f;
                for (int d = 0; d < 2 * KD; d++) dot += sRes[d] * book[(i64)d * M + c];
                const float dist = n2[c] - 2.0f * dot;
                if (dist < bv) { bv = dist; bi = c; }
            }
            const int pick = block_argmin(bv, bi, sRedV, sRedI, tid);
            if (picks_ && tid == 0) picks_[((((i64)b * T + t) * H + h) * stages) + s] = pick;
            if (tid < 2 * KD) sAcc[tid] += book[(i64)tid * M + pick];
            __syncthreads();
        }

        // ---- norm (NormQuant) and reconstruction
        const FT n = ft_sqrt(valid ? sigma : (FT)1);
        float nq;
        if (norm_mode == 1) {
            const float tq = (log2f((float)n) - norm_lo) / norm_span;
            const float q = fminf(fmaxf(rintf(tq * norm_levels), 0.0f), norm_levels);
            nq = exp2f(norm_lo + q / norm_levels * norm_span);
        } else {
            nq = norm_fixed;
        }
        const FT nq2 = (FT)nq * (FT)nq;
        state = valid ? (float)(nq2 * (FT)sAcc[x] * (FT)sAcc[KD + y]) : 0.0f;
        if (cscale_ && tid == 0) cscale_[((i64)b * T + t) * H + h] = valid ? (float)nq2 : 0.0f;
        __syncthreads();   // the shared buffers are rewritten by the next coded step
    }
    if (final_) final_[(((i64)b * H + h) * KD + x) * KD + y] = state;
}
