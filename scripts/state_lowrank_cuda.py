"""SM148: batched symmetric eigen-decomposition of small matrices (m <= 8) on CUDA by cyclic Jacobi, one thread per
matrix, float64 in registers, no host synchronisation (torch.linalg.eigh's cusolver path checks errors on the host
and refuses batches of about 32k matrices). Output as torch.linalg.eigh: eigenvalues ascending [N, m] and the
eigenvectors as columns [N, m, m], same dtype as the input. Used by scripts/state_lowrank._eigh on CUDA tensors.
Checked against torch.linalg.eigh in selftest() (eigenvalues, reconstruction, orthonormality).
"""
import torch

SWEEPS = 12   # cyclic sweeps (quadratic convergence: m = 5 converges to float64 rounding in about 5 to 6)

_SOURCE = r"""
extern "C" __global__ void jacobi_eigh(const float* A, float* W, float* V, int N, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    double a[8][8], v[8][8];
    const float* src = A + (long long)idx * m * m;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            a[i][j] = 0.5 * ((double)src[i * m + j] + (double)src[j * m + i]);
            v[i][j] = (i == j) ? 1.0 : 0.0;
        }
    for (int sweep = 0; sweep < SWEEPS; sweep++) {
        double off = 0.0, diag = 0.0;
        for (int p = 0; p < m; p++) {
            diag += a[p][p] * a[p][p];
            for (int q = p + 1; q < m; q++) off += a[p][q] * a[p][q];
        }
        if (off <= 1e-30 * diag || off == 0.0) break;
        for (int p = 0; p < m - 1; p++)
            for (int q = p + 1; q < m; q++) {
                double apq = a[p][q];
                if (apq == 0.0) continue;
                double theta = (a[q][q] - a[p][p]) / (2.0 * apq);
                double t = (theta >= 0.0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < m; k++) {          // A <- A J (columns p, q)
                    double akp = a[k][p], akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < m; k++) {          // A <- J^T A (rows p, q)
                    double apk = a[p][k], aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                a[p][q] = 0.0; a[q][p] = 0.0;
                for (int k = 0; k < m; k++) {          // V <- V J
                    double vkp = v[k][p], vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
    }
    int order[8];
    for (int i = 0; i < m; i++) order[i] = i;
    for (int i = 1; i < m; i++) {                      // insertion sort, ascending eigenvalues
        int o = order[i], j = i - 1;
        while (j >= 0 && a[order[j]][order[j]] > a[o][o]) { order[j + 1] = order[j]; j--; }
        order[j + 1] = o;
    }
    float* w = W + (long long)idx * m;
    float* vv = V + (long long)idx * m * m;
    for (int c2 = 0; c2 < m; c2++) {
        int o = order[c2];
        w[c2] = (float)a[o][o];
        for (int k = 0; k < m; k++) vv[k * m + c2] = (float)v[k][o];
    }
}
"""

_kernels = {}


def _kernel():
    if "k" not in _kernels:
        from scripts import small_model_wkv
        small_model_wkv._kernel(16)   # the DLL directory set-up (Windows) of the repo's cupy kernels
        import cupy
        _kernels["k"] = cupy.RawKernel(_SOURCE.replace("SWEEPS", str(SWEEPS)), "jacobi_eigh",
                                       options=("--std=c++14",))
    return _kernels["k"]


def eigh(A):
    """A [..., m, m] float32 CUDA, symmetric (symmetrised in the kernel), m <= 8 -> (eigenvalues ascending, vectors)."""
    if A.dtype != torch.float32 or not A.is_cuda or A.shape[-1] > 8:
        raise ValueError("state_lowrank_cuda.eigh: float32 CUDA [..., m, m] with m <= 8")
    import cupy
    shape = A.shape
    m = shape[-1]
    flat = A.reshape(-1, m, m).contiguous()
    N = flat.shape[0]
    W = torch.empty(N, m, dtype=torch.float32, device=A.device)
    V = torch.empty(N, m, m, dtype=torch.float32, device=A.device)
    if N:
        with cupy.cuda.Device(A.device.index), \
                cupy.cuda.ExternalStream(torch.cuda.current_stream(A.device).cuda_stream):
            _kernel()(((N + 127) // 128,), (128,),
                      (cupy.from_dlpack(flat), cupy.from_dlpack(W), cupy.from_dlpack(V), cupy.int32(N), cupy.int32(m)))
    return W.reshape(*shape[:-1]), V.reshape(shape)


# ------------------------------------------------------------------------------------------------ fused encode

# One warp per stored matrix (row, head): the rank tiers of state_lowrank.LowRankCodec.encode (factored truncation,
# rank r_eff, int8 factor vectors with float16 scales, product) in one launch instead of about 90 small kernels per
# step. Same math as state_lowrank.lowrank_factors (float32 throughout, Jacobi in float32, Gram eigenvalues below
# KEEP_TOL x the largest dropped); not bitwise equal to the torch path (summation order, Jacobi vs LAPACK).
_ENCODE = r"""
#include <cuda_fp16.h>
__device__ void jac(float a[8][8], float v[8][8], int m) {
    for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) v[i][j] = (i == j) ? 1.0f : 0.0f;
    for (int sweep = 0; sweep < SWEEPS; sweep++) {
        float off = 0.0, diag = 0.0f;
        for (int p = 0; p < m; p++) { diag += a[p][p] * a[p][p]; for (int q = p + 1; q < m; q++) off += a[p][q] * a[p][q]; }
        if (off <= 1e-14f * diag || off == 0.0f) break;
        for (int p = 0; p < m - 1; p++) for (int q = p + 1; q < m; q++) {
            float apq = a[p][q];
            if (apq == 0.0f) continue;
            float theta = (a[q][q] - a[p][p]) / (2.0f * apq);
            float t = (theta >= 0.0f ? 1.0f : -1.0f) / (fabsf(theta) + sqrtf(theta * theta + 1.0f));
            float c = 1.0f / sqrtf(t * t + 1.0f), s = t * c;
            for (int k = 0; k < m; k++) { float x = a[k][p], y = a[k][q]; a[k][p] = c * x - s * y; a[k][q] = s * x + c * y; }
            for (int k = 0; k < m; k++) { float x = a[p][k], y = a[q][k]; a[p][k] = c * x - s * y; a[q][k] = s * x + c * y; }
            a[p][q] = 0.0f; a[q][p] = 0.0f;
            for (int k = 0; k < m; k++) { float x = v[k][p], y = v[k][q]; v[k][p] = c * x - s * y; v[k][q] = s * x + c * y; }
        }
    }
}

__device__ float f16r(float x) { return __half2float(__float2half_rn(fminf(x, 65504.0f))); }
__device__ float wsum(float x) { for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o); return x; }
__device__ float wmax(float x) { for (int o = 16; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o)); return x; }
__device__ int wand(int x) { for (int o = 16; o > 0; o >>= 1) x &= __shfl_xor_sync(0xffffffff, x, o); return x; }

// One warp per stored matrix; lane c holds column c of S (and of B, the right factors, the output) and plays row
// k = c for P, Q and the left factors. Lanes c >= K hold zeros.
extern "C" __global__ void lowrank_encode(const float* S_, const float* Lp_, const float* v_, const int* reff_,
                                          float* out_, float* Lo_, int N, int H, int quant) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int c = threadIdx.x & 31;
    if (warp >= N) return;            // whole warps exit together
    const int m = RR + 1;
    const bool on = c < KK;
    const float* S = S_ + (long long)warp * KK * KK;
    const float* Lp = Lp_ + (long long)warp * KK * RR;
    const float* v = v_ + (long long)warp * KK;
    float* out = out_ + (long long)warp * KK * KK;
    float* Lo = Lo_ + (long long)warp * KK * RR;
    int reff = reff_[warp / H];
    float Sc[KK];
    int fin = 1;
    for (int k = 0; k < KK; k++) { Sc[k] = on ? S[k * KK + c] : 0.0f; fin &= isfinite(Sc[k]) ? 1 : 0; }
    float Pr[RR + 1];
    for (int j = 0; j < RR; j++) { Pr[j] = on ? Lp[c * RR + j] : 0.0f; fin &= isfinite(Pr[j]) ? 1 : 0; }
    Pr[RR] = on ? v[c] : 0.0f; fin &= isfinite(Pr[RR]) ? 1 : 0;
    fin = wand(fin);
    if (!fin) {
        if (on) { for (int k = 0; k < KK; k++) out[k * KK + c] = Sc[k]; for (int j = 0; j < RR; j++) Lo[c * RR + j] = 0.0f; }
        return;
    }
    for (int j = 0; j < m; j++) {
        float nrm = sqrtf(wsum(Pr[j] * Pr[j]));
        Pr[j] = nrm > 0.0f ? Pr[j] / nrm : 0.0f;
    }
    float G[8][8], W[8][8];
    for (int i = 0; i < m; i++) for (int j = i; j < m; j++) { float s = wsum(Pr[i] * Pr[j]); G[i][j] = s; G[j][i] = s; }
    jac(G, W, m);
    float top = 0.0f;
    for (int j = 0; j < m; j++) top = fmaxf(top, G[j][j]);
    float Qr[RR + 1];
    for (int j = 0; j < m; j++) {
        float lam = G[j][j];
        float inv = (lam > top * KEEP && lam > 0.0f) ? 1.0f / sqrtf(lam) : 0.0f;
        float s = 0.0f; for (int i = 0; i < m; i++) s += Pr[i] * W[i][j];
        Qr[j] = (s * inv);
    }
    float Bc[RR + 1];
    for (int j = 0; j < m; j++) {
        float s = 0.0f;
        for (int k = 0; k < KK; k++) s += __shfl_sync(0xffffffff, Qr[j], k) * Sc[k];
        Bc[j] = s;
    }
    float C[8][8], U[8][8];
    for (int i = 0; i < m; i++) for (int j = i; j < m; j++) { float s = wsum(Bc[i] * Bc[j]); C[i][j] = s; C[j][i] = s; }
    jac(C, U, m);
    int order[8];
    for (int i = 0; i < m; i++) order[i] = i;
    for (int i = 1; i < m; i++) {
        int o = order[i], j = i - 1;
        while (j >= 0 && C[order[j]][order[j]] < C[o][o]) { order[j + 1] = order[j]; j--; }
        order[j + 1] = o;
    }
    float lf[RR], rf[RR];
    for (int i = 0; i < RR; i++) {
        int o = order[i];
        float s = 0.0f; for (int j = 0; j < m; j++) s += U[j][o] * Bc[j];
        float r = s;
        float ss = wsum(r * r);
        float g = sqrtf(sqrtf(ss));
        float t = 0.0f; for (int j = 0; j < m; j++) t += Qr[j] * U[j][o];
        float l = t * g;
        r = g > 0.0f ? r / g : r;
        if (i >= reff) { l = 0.0f; r = 0.0f; }
        if (quant) {
            float sc = f16r(wmax(fabsf(l)) / 127.0f); if (!(sc > 0.0f)) sc = 1.0f;
            l = fminf(fmaxf(rintf(l / sc), -127.0f), 127.0f) * sc;
            sc = f16r(wmax(fabsf(r)) / 127.0f); if (!(sc > 0.0f)) sc = 1.0f;
            r = fminf(fmaxf(rintf(r / sc), -127.0f), 127.0f) * sc;
        }
        lf[i] = on ? l : 0.0f;
        rf[i] = on ? r : 0.0f;
    }
    for (int k = 0; k < KK; k++) {
        float s = 0.0f;
        for (int i = 0; i < RR; i++) s += __shfl_sync(0xffffffff, lf[i], k) * rf[i];
        if (on) out[k * KK + c] = s;
    }
    if (on) for (int i = 0; i < RR; i++) Lo[c * RR + i] = lf[i];
}
"""

KEEP_TOL = 1e-5   # Gram eigenvalue threshold relative to the largest (float32 data; as state_lowrank.KEEP_TOL)


def _encode_kernel(K, R):
    key = ("enc", K, R)
    if key not in _kernels:
        _kernel()
        import cupy
        src = (_ENCODE.replace("SWEEPS", str(SWEEPS)).replace("KEEP", repr(KEEP_TOL))
               .replace("KK", str(K)).replace("RR", str(R)))
        _kernels[key] = cupy.RawKernel(src, "lowrank_encode", options=("--std=c++14",))
    return _kernels[key]


def encode(nxt, left_prev, v, r_eff, quant):
    """nxt [N, H, K, K], left_prev [N, H, K, R], v [N, H, K] float32 CUDA, r_eff [N] (rank kept per row, <= R),
    quant = int8 factor vectors -> (stored [N, H, K, K], left [N, H, K, R])."""
    import cupy
    N, H, K, _ = nxt.shape
    R = left_prev.shape[-1]
    S = nxt.contiguous()
    Lp = left_prev.contiguous()
    vv = v.contiguous()
    re = r_eff.to(torch.int32).contiguous()
    out = torch.empty_like(S)
    Lo = torch.empty_like(Lp)
    M = N * H
    if M:
        with cupy.cuda.Device(S.device.index), \
                cupy.cuda.ExternalStream(torch.cuda.current_stream(S.device).cuda_stream):
            _encode_kernel(K, R)(((M + 3) // 4,), (128,),   # one warp per matrix, 4 per block
                                 (cupy.from_dlpack(S), cupy.from_dlpack(Lp), cupy.from_dlpack(vv),
                                  cupy.from_dlpack(re), cupy.from_dlpack(out), cupy.from_dlpack(Lo), cupy.int32(M),
                                  cupy.int32(H), cupy.int32(1 if quant else 0)))
    return out, Lo


def selftest(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = {}
    for m in (1, 2, 3, 5, 8):
        X = torch.randn(40000, m, m + 2, device="cuda", generator=g)
        A = X @ X.mT
        A[:5] = 0                                       # zero matrices
        A[5:10, :, -1] = 0
        A[5:10, -1, :] = 0                              # rank deficient
        w, V = eigh(A)
        w0 = torch.linalg.eigh(A[:16000].double())[0]
        scale = A.flatten(1).norm(dim=-1).clamp(min=1e-30)
        rec = ((V @ torch.diag_embed(w) @ V.mT) - A).flatten(1).norm(dim=-1) / scale
        ortho = (V.mT @ V - torch.eye(m, device="cuda")).flatten(1).abs().amax(-1)
        dw = ((w[:16000].double() - w0).abs().amax(-1) / scale[:16000].double()).max()
        assert float(rec[5:].max()) < 1e-5 and float(ortho.max()) < 1e-5 and float(dw) < 1e-5, (m, rec.max(), ortho.max(), dw)
        assert torch.equal(w[:5], torch.zeros_like(w[:5])) and bool((w[:, 1:] >= w[:, :-1]).all())
        out[m] = {"reconstruction": float(rec[5:].max()), "orthonormality": float(ortho.max()), "eigenvalues": float(dw)}
    print("state_lowrank_cuda selftest OK", out)
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    selftest()
