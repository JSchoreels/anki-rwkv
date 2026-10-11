// Experimental F09: multiple state rows per warp. Float32 state and reductions.
// Equations and checkpoint layout match rwkv/model/csrc/cuda/rwkv7_cuda.cu.
namespace rwkv_future {
template <int TILE, int ROWS, typename F>
__global__ void forward_kernel(int B, int T, int H, const F* r, const F* k,
    const F* v, const float* w, const F* a, const F* kd, const bool* skip,
    F* out, int L, float* checkpoints) {
    const int b = blockIdx.x, h = blockIdx.y, y = threadIdx.x;
    const int WARPS = 32 / ROWS;
    float state[ROWS] = {};
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
            float dot = state[q] * kdy;
            for (int offset = 16; offset > 0; offset /= 2)
                dot += __shfl_down_sync(FULL_MASK, dot, offset);
            dot = __shfl_sync(FULL_MASK, dot, 0);
            state[q] = state[q] * wy - dot * ay * kdy;
            state[q] += to_float<F>(v[ix]) * ky;
            float result = state[q] * ry;
            for (int offset = 16; offset > 0; offset /= 2)
                result += __shfl_down_sync(FULL_MASK, result, offset);
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
    float carry[ROWS] = {};
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
                float dot = state[q] * kdy;
                for (int offset = 16; offset > 0; offset /= 2)
                    dot += __shfl_down_sync(FULL_MASK, dot, offset);
                dot = __shfl_sync(FULL_MASK, dot, 0);
                state[q] = state[q] * wy - dot * ay * kdy;
                state[q] += to_float<F>(v[get_index3(b, t, h, x, T, H, 32)]) * ky;
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
            float decay[ROWS], remove[ROWS], product[ROWS];
            if (threadIdx.y == 0) { Kd[y] = kdy; A[y] = ay; }
            #pragma unroll
            for (int q = 0; q < ROWS; ++q) {
                const int x = threadIdx.y + q * WARPS;
                const int64_t ix = get_index3(b, t, h, x, T, H, 32);
                float ds = to_float<F>(grad[ix]) * ry;
                if (!skipped) { ds += carry[q]; carry[q] = 0.0f; }
                decay[q] = ds * wy;
                remove[q] = ds * ay * kdy;
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
                for (int j = 0; j < 32; ++j) p += P[j * 33 + x] * D[j * 33 + y];
                product[q] = p;
                if (x == y) dw[ix] = p;
                G[x * 33 + y] = p;
                float gr = S[y * 33 + x] * gy;
                float gv = D[x * 33 + y] * ky;
                float gk = D[y * 33 + x] * vy;
                float rem = remove[q];
                for (int offset = 16; offset > 0; offset /= 2) {
                    gr += __shfl_down_sync(FULL_MASK, gr, offset);
                    gv += __shfl_down_sync(FULL_MASK, gv, offset);
                    gk += __shfl_down_sync(FULL_MASK, gk, offset);
                    rem += __shfl_down_sync(FULL_MASK, rem, offset);
                }
                remove[q] = __shfl_sync(FULL_MASK, rem, 0);
                if (y == 0) { dr[ix] = to_F<F>(gr); dv[ix] = to_F<F>(gv); dk[ix] = to_F<F>(gk); }
            }
            __syncthreads();
            #pragma unroll
            for (int q = 0; q < ROWS; ++q) {
                const int x = threadIdx.y + q * WARPS;
                const int64_t ix = get_index3(b, t, h, x, T, H, 32);
                const float transpose = G[y * 33 + x];
                float ga = -transpose * Kd[y];
                float gkd1 = -product[q] * A[y] * Kd[y];
                float gkd2 = -A[x] * transpose * Kd[y];
                for (int offset = 16; offset > 0; offset /= 2) {
                    ga += __shfl_down_sync(FULL_MASK, ga, offset);
                    gkd1 += __shfl_down_sync(FULL_MASK, gkd1, offset);
                    gkd2 += __shfl_down_sync(FULL_MASK, gkd2, offset);
                }
                if (y == 0) { da[ix] = to_F<F>(ga * Kd[x]); dkd[ix] = to_F<F>(gkd1 + gkd2); }
                carry[q] += decay[q] - remove[q] * kdy;
            }
            __syncthreads();
        }
    }
}
}
