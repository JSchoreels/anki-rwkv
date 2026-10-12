// F31: F09 row layout with every float operation pinned to the production kernel's SASS.
// rwkv/model/csrc/cuda/rwkv7_cuda.cu leaves contraction to ptxas, which fuses a
// different set of multiply/add pairs once the layout changes (the F09 prototype
// differs by one ulp in the carry update). Round-to-nearest intrinsics cannot be
// contracted, so each expression below reproduces the production instruction
// sequence for sm_120 (NVRTC 13.4): products, FFMA operands and shuffle-tree order.
namespace rwkv_exact {
__device__ inline float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2)
        value = __fadd_rn(value, __shfl_down_sync(FULL_MASK, value, offset));
    return value;
}

// state <- fma(v, k, fma(state, w, -(kd * (a * dot)))), as FFMA R, R, R, -R; FFMA.
__device__ inline float advance(float state, float dot, float wy, float ay, float kdy, float vx, float ky) {
    const float removed = __fmul_rn(kdy, __fmul_rn(ay, dot));
    return __fmaf_rn(vx, ky, __fmaf_rn(state, wy, -removed));
}

template <int TILE, int ROWS, typename F>
__global__ void forward_kernel(int B, int T, int H, const F* r, const F* k,
    const F* v, const float* w, const F* a, const F* kd, const bool* skip,
    F* out, int L, float* checkpoints) {
    const int b = blockIdx.x, h = blockIdx.y, y = threadIdx.x;
    const int WARPS = 32 / ROWS;
    float state[ROWS];
    #pragma unroll
    for (int q = 0; q < ROWS; ++q) state[q] = 0.0f;
    for (int t = 0; t < T; ++t) {
        const int64_t iy = get_index3(b, t, h, y, T, H, 32);
        const float ry = to_float<F>(r[iy]), ky = to_float<F>(k[iy]);
        const float wy = w[iy], ay = to_float<F>(a[iy]), kdy = to_float<F>(kd[iy]);
        const bool skipped = skip[get_index1(b, t, T)];
        #pragma unroll
        for (int q = 0; q < ROWS; ++q) {
            const int x = threadIdx.y + q * WARPS;
            const int64_t ix = get_index3(b, t, h, x, T, H, 32);
            if (t % TILE == 0)
                checkpoints[get_index4(b, t / TILE, h, x, y, L, H, 32, 32)] = state[q];
            const float previous = state[q];
            float dot = warp_sum(__fmul_rn(state[q], kdy));
            dot = __shfl_sync(FULL_MASK, dot, 0);
            state[q] = advance(state[q], dot, wy, ay, kdy, to_float<F>(v[ix]), ky);
            const float result = warp_sum(__fmul_rn(state[q], ry));
            if (y == 0) out[ix] = to_F<F>(result);
            if (skipped) state[q] = previous;
        }
    }
}

template <int TILE, int ROWS, typename F>
__global__ void backward_kernel(int B, int T, int H, const F* r, const F* k,
    const F* v, const float* w, const F* a, const F* kd, const bool* skip,
    const F* grad, int L, const float* checkpoints, F* dr, F* dk, F* dv,
    float* dw, F* da, F* dkd) {
    __shared__ float S[32 * 33], P[32 * 33], D[32 * 33], G[32 * 33];
    __shared__ float Kd[32], A[32];
    const int b = blockIdx.x, h = blockIdx.y, y = threadIdx.x;
    const int WARPS = 32 / ROWS;
    float chunk[ROWS][TILE], previous_chunk[ROWS][TILE];
    float carry[ROWS];
    #pragma unroll
    for (int q = 0; q < ROWS; ++q) carry[q] = 0.0f;
    if (threadIdx.y == 0) {
        da[get_index3(b, 0, h, y, T, H, 32)] = to_F<F>(0.0f);
        dkd[get_index3(b, 0, h, y, T, H, 32)] = to_F<F>(0.0f);
    }
    for (int l = L - 1; l >= 0; --l) {
        float state[ROWS];
        #pragma unroll
        for (int q = 0; q < ROWS; ++q) {
            const int x = threadIdx.y + q * WARPS;
            state[q] = checkpoints[get_index4(b, l, h, x, y, L, H, 32, 32)];
        }
        for (int c = 0; c < TILE && l * TILE + c < T; ++c) {
            const int t = l * TILE + c;
            const int64_t iy = get_index3(b, t, h, y, T, H, 32);
            const float ky = to_float<F>(k[iy]), wy = w[iy];
            const float ay = to_float<F>(a[iy]), kdy = to_float<F>(kd[iy]);
            const bool skipped = skip[get_index1(b, t, T)];
            #pragma unroll
            for (int q = 0; q < ROWS; ++q) {
                const int x = threadIdx.y + q * WARPS;
                previous_chunk[q][c] = state[q];
                float dot = warp_sum(__fmul_rn(state[q], kdy));
                dot = __shfl_sync(FULL_MASK, dot, 0);
                state[q] = advance(state[q], dot, wy, ay, kdy, to_float<F>(v[get_index3(b, t, h, x, T, H, 32)]), ky);
                chunk[q][c] = state[q];
                if (skipped) state[q] = previous_chunk[q][c];
            }
        }
        for (int t = nvrtc_min(T - 1, (l + 1) * TILE - 1); t >= l * TILE; --t) {
            const int c = t - l * TILE;
            const int64_t iy = get_index3(b, t, h, y, T, H, 32);
            const float ry = to_float<F>(r[iy]), ky = to_float<F>(k[iy]);
            const float vy = to_float<F>(v[iy]), wy = w[iy];
            const float ay = to_float<F>(a[iy]), kdy = to_float<F>(kd[iy]);
            const float gy = to_float<F>(grad[iy]);
            const bool skipped = skip[get_index1(b, t, T)];
            float ds_rows[ROWS], remove[ROWS], product[ROWS];
            if (threadIdx.y == 0) { Kd[y] = kdy; A[y] = ay; }
            #pragma unroll
            for (int q = 0; q < ROWS; ++q) {
                const int x = threadIdx.y + q * WARPS;
                const int64_t ix = get_index3(b, t, h, x, T, H, 32);
                float ds = __fmul_rn(to_float<F>(grad[ix]), ry);
                if (!skipped) { ds = __fadd_rn(ds, carry[q]); carry[q] = 0.0f; }
                ds_rows[q] = ds;
                remove[q] = __fmul_rn(kdy, __fmul_rn(ay, ds));
                S[x * 33 + y] = chunk[q][c];
                P[x * 33 + y] = previous_chunk[q][c];
                D[x * 33 + y] = ds;
            }
            __syncthreads();
            #pragma unroll
            for (int q = 0; q < ROWS; ++q) {
                const int x = threadIdx.y + q * WARPS;
                const int64_t ix = get_index3(b, t, h, x, T, H, 32);
                float p = 0.0f;
                for (int j = 0; j < 32; ++j) p = __fmaf_rn(P[j * 33 + x], D[j * 33 + y], p);
                product[q] = p;
                if (x == y) dw[ix] = p;
                G[x * 33 + y] = p;
                const float gr = warp_sum(__fmul_rn(S[y * 33 + x], gy));
                const float gv = warp_sum(__fmul_rn(ds_rows[q], ky));
                const float gk = warp_sum(__fmul_rn(D[y * 33 + x], vy));
                remove[q] = __shfl_sync(FULL_MASK, warp_sum(remove[q]), 0);
                if (y == 0) { dr[ix] = to_F<F>(gr); dv[ix] = to_F<F>(gv); dk[ix] = to_F<F>(gk); }
            }
            __syncthreads();
            #pragma unroll
            for (int q = 0; q < ROWS; ++q) {
                const int x = threadIdx.y + q * WARPS;
                const int64_t ix = get_index3(b, t, h, x, T, H, 32);
                const float transpose = G[y * 33 + x];
                const float ga = warp_sum(__fmul_rn(-transpose, Kd[y]));
                const float gkd1 = warp_sum(__fmul_rn(__fmul_rn(-product[q], A[y]), Kd[y]));
                const float gkd2 = warp_sum(__fmul_rn(__fmul_rn(-A[x], transpose), Kd[y]));
                if (y == 0) { da[ix] = to_F<F>(__fmul_rn(ga, Kd[x])); dkd[ix] = to_F<F>(__fadd_rn(gkd1, gkd2)); }
                // Production: FMUL kd*rem; FFMA w, ds, -(kd*rem); FADD with the reset carry.
                carry[q] = __fadd_rn(carry[q], __fmaf_rn(wy, ds_rows[q], -__fmul_rn(kdy, remove[q])));
            }
            __syncthreads();
        }
    }
}
}
