/*
 * bisheng 2026 - device-resident matrix-free CG for the constant potential method.
 * See cpm_gpu_cg.h. Arithmetic mirrors the host solvers in pme.cpp operation for
 * operation (float vectors, float products, double accumulation), so that the
 * device solve can be checked against the host solve at the level of the GPU
 * path's own run-to-run floor.
 */
#include "gmxpre.h"

#include "cpm_gpu_cg.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "gromacs/ewald/pme.h"
#include "gromacs/utility/fatalerror.h"

namespace
{

#define CPM_CU_CHECK(call)                                                                        \
    do                                                                                            \
    {                                                                                             \
        cudaError_t err_ = (call);                                                                \
        if (err_ != cudaSuccess)                                                                  \
        {                                                                                         \
            gmx_fatal(FARGS, "CPM GPU CG: %s failed: %s", #call, cudaGetErrorString(err_));       \
        }                                                                                         \
    } while (0)

constexpr int c_redThreads = 256; // single-block reductions (deterministic order); 256 beats 1024 on FP64-lean parts
constexpr int c_vecThreads = 256;

/* scalar slots on the device */
enum
{
    sDotOld = 0,
    sDAd    = 1,
    sDotNew = 2,
    sDelta  = 3,
    sDipole = 4,
    sSum1   = 5,
    sSum2   = 6,
    sCount  = 8
};

struct CgState
{
    int    n = 0, nAlloc = 0, nnzAlloc = 0;
    float *q = nullptr, *r = nullptr, *d = nullptr, *Ad = nullptr, *b = nullptr, *qBest = nullptr;
    float *z = nullptr, *self = nullptr, *tf = nullptr, *xtmp = nullptr;
    int *  rowPtr = nullptr, *colIdx = nullptr;
    float* coef   = nullptr;
    double* s     = nullptr; // device scalars
    double* hS    = nullptr; // pinned mirror
    bool    haveTf = false;
    float   pref3dc = 0.0f, evscale = 0.0f, potUnit = 0.0f;
    double  ms = 0.0;
};
CgState g;

__device__ __forceinline__ double warpSum(double v)
{
    for (int o = 16; o > 0; o >>= 1)
    {
        v += __shfl_down_sync(0xffffffffu, v, o);
    }
    return v;
}

/* Block-wide sum broadcast to every thread. sh: 32 doubles, bc: 1 double. */
__device__ double blockSumAll(double v, double* sh, double* bc)
{
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warpSum(v);
    if (lane == 0)
    {
        sh[wid] = v;
    }
    __syncthreads();
    if (wid == 0)
    {
        v = (lane < (blockDim.x >> 5)) ? sh[lane] : 0.0;
        v = warpSum(v);
        if (lane == 0)
        {
            *bc = v;
        }
    }
    __syncthreads();
    const double out = *bc;
    __syncthreads();
    return out;
}

/* dipole = sum_j z_j q_j (double accumulation; the host uses float here) */
__global__ void k_dipole(const float* z, const float* q, int n, double* s)
{
    __shared__ double sh[32];
    __shared__ double bc;
    double            acc = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        acc += static_cast<double>(z[i]) * static_cast<double>(q[i]);
    }
    acc = blockSumAll(acc, sh, &bc);
    if (threadIdx.x == 0)
    {
        s[sDipole] = acc;
    }
}

/* A x assembly, one warp per electrode atom: the lanes stride over the CSR row and
 * the partial sums are folded with a warp reduction (float, like the host's row sum,
 * but in a different order). The rest of the row mirrors gmx_pme_calc_Aq. */
__global__ void k_assemble(float*       Ax,
                           const float* phik,
                           const float* x,
                           const int*   rowPtr,
                           const int*   colIdx,
                           const float* coef,
                           const float* self,
                           const float* tf,
                           const float* z,
                           const double* s,
                           float        evscale,
                           float        potUnit,
                           float        pref3dc,
                           int          n)
{
    const int lane = threadIdx.x & 31;
    const int i    = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (i >= n)
    {
        return;
    }
    const int end = rowPtr[i + 1];
    float     sr  = 0.0f;
    for (int idx = rowPtr[i] + lane; idx < end; idx += 32)
    {
        sr += coef[idx] * x[colIdx[idx]];
    }
    for (int o = 16; o > 0; o >>= 1)
    {
        sr += __shfl_down_sync(0xffffffffu, sr, o);
    }
    if (lane != 0)
    {
        return;
    }
    const float ks  = phik[i] * evscale * potUnit;
    float       acc = ks + sr + self[i] * x[i];
    if (pref3dc != 0.0f)
    {
        acc += pref3dc * z[i] * static_cast<float>(s[sDipole]);
    }
    if (tf != nullptr)
    {
        acc += tf[i] * x[i];
    }
    Ax[i] = acc;
}

/* r = b - Ax */
__global__ void k_residual(float* r, const float* b, const float* Ax, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        r[i] = b[i] - Ax[i];
    }
}

/* Projection: shift group sums to targets. groups 1: whole vector to t1;
 * groups 2: first half to t1, second half to t2. Two passes like the host. */
__global__ void k_project(float* v, int n, int groups, double t1, double t2, int passes)
{
    __shared__ double sh[32];
    __shared__ double bc;
    const int         half = (groups == 2) ? n / 2 : n;
    for (int p = 0; p < passes; ++p)
    {
        double s1 = 0.0, s2 = 0.0;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
        {
            const double x = v[i];
            if (i < half)
            {
                s1 += x;
            }
            else
            {
                s2 += x;
            }
        }
        s1 = blockSumAll(s1, sh, &bc);
        if (groups == 2)
        {
            s2 = blockSumAll(s2, sh, &bc);
        }
        const double sh1 = (t1 - s1) / half;
        const double sh2 = (groups == 2) ? (t2 - s2) / (n - half) : 0.0;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
        {
            v[i] = static_cast<float>(static_cast<double>(v[i]) + (i < half ? sh1 : sh2));
        }
        __syncthreads();
    }
}

/* Up to two dot products (float product, double sum, like the host), plus an
 * optional scalar copy s[copyTo] = s[copyFrom] done before the results land. */
__global__ void k_dots(const float* a, const float* b, int outAB, const float* c, const float* d, int outCD,
                       int n, double* s, int copyFrom, int copyTo)
{
    __shared__ double sh[32];
    __shared__ double bc;
    double            x = 0.0, y = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        x += static_cast<double>(a[i] * b[i]);
        if (c != nullptr)
        {
            y += static_cast<double>(c[i] * d[i]);
        }
    }
    x = blockSumAll(x, sh, &bc);
    if (c != nullptr)
    {
        y = blockSumAll(y, sh, &bc);
    }
    if (threadIdx.x == 0)
    {
        if (copyFrom >= 0)
        {
            s[copyTo] = s[copyFrom];
        }
        s[outAB] = x;
        if (c != nullptr)
        {
            s[outCD] = y;
        }
    }
}

/* alpha = dot_old / dAd; q += alpha d; r -= alpha Ad (skipped when dAd ~ 0, host breaks there) */
__global__ void k_update(float* q, const float* d, float* r, const float* Ad, const double* s, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const double dAd = s[sDAd];
    if (fabs(dAd) < 1e-30)
    {
        return;
    }
    const double alpha = s[sDotOld] / dAd;
    q[i]               = static_cast<float>(static_cast<double>(q[i]) + alpha * static_cast<double>(d[i]));
    r[i]               = static_cast<float>(static_cast<double>(r[i]) - alpha * static_cast<double>(Ad[i]));
}

/* d = r + beta d, beta = restart ? 0 : dot_new / dot_old */
__global__ void k_direction(float* d, const float* r, const double* s, int restart, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const double beta = restart ? 0.0 : s[sDotNew] / s[sDotOld];
    d[i]              = static_cast<float>(static_cast<double>(r[i]) + beta * static_cast<double>(d[i]));
}

/* group sums of (b - Ax), for the realised-potential read-out of the constrained modes */
__global__ void k_groupSumDiff(const float* b, const float* Ax, int n, int groups, double* s)
{
    __shared__ double sh[32];
    __shared__ double bc;
    const int         half = (groups == 2) ? n / 2 : n;
    double            s1 = 0.0, s2 = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        const double v = static_cast<double>(b[i]) - static_cast<double>(Ax[i]);
        if (i < half)
        {
            s1 += v;
        }
        else
        {
            s2 += v;
        }
    }
    s1 = blockSumAll(s1, sh, &bc);
    s2 = blockSumAll(s2, sh, &bc);
    if (threadIdx.x == 0)
    {
        s[sSum1] = s1;
        s[sSum2] = s2;
    }
}

template<typename T>
void devAlloc(T** p, size_t count)
{
    if (*p)
    {
        CPM_CU_CHECK(cudaFree(*p));
    }
    CPM_CU_CHECK(cudaMalloc(reinterpret_cast<void**>(p), count * sizeof(T)));
}

void ensureVectors(int n)
{
    if (n <= g.nAlloc)
    {
        g.n = n;
        return;
    }
    devAlloc(&g.q, n);
    devAlloc(&g.r, n);
    devAlloc(&g.d, n);
    devAlloc(&g.Ad, n);
    devAlloc(&g.b, n);
    devAlloc(&g.qBest, n);
    devAlloc(&g.z, n);
    devAlloc(&g.self, n);
    devAlloc(&g.tf, n);
    devAlloc(&g.xtmp, n);
    devAlloc(&g.rowPtr, n + 1);
    if (g.s == nullptr)
    {
        devAlloc(&g.s, sCount);
        CPM_CU_CHECK(cudaMemset(g.s, 0, sCount * sizeof(double)));
        CPM_CU_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&g.hS), sCount * sizeof(double), cudaHostAllocDefault));
    }
    g.nAlloc = n;
    g.n      = n;
}

inline int vecBlocks(int n)
{
    return (n + c_vecThreads - 1) / c_vecThreads;
}

/* y = A x, x and y device vectors of length g.n, on the PME stream */
void deviceAq(gmx_pme_t* pme, const float* x, float* y, cudaStream_t st)
{
    const float* phik = pme_gpu_cpm_kspace_from_device(pme, x, g.n);
    if (g.pref3dc != 0.0f)
    {
        k_dipole<<<1, c_redThreads, 0, st>>>(g.z, x, g.n, g.s);
    }
    const int rowsPerBlock = c_vecThreads / 32;
    k_assemble<<<(g.n + rowsPerBlock - 1) / rowsPerBlock, c_vecThreads, 0, st>>>(y, phik, x, g.rowPtr, g.colIdx, g.coef, g.self,
                                                        g.haveTf ? g.tf : nullptr, g.z, g.s, g.evscale,
                                                        g.potUnit, g.pref3dc, g.n);
}

void projectQ(int mode, double Q1, double Q2, cudaStream_t st)
{
    switch (mode)
    {
        case 0: k_project<<<1, c_redThreads, 0, st>>>(g.q, g.n, 1, 0.0, 0.0, 2); break;
        case 1: k_project<<<1, c_redThreads, 0, st>>>(g.q, g.n, 2, Q1, Q2, 2); break;
        default: k_project<<<1, c_redThreads, 0, st>>>(g.q, g.n, 1, Q1, 0.0, 1); break;
    }
}

void projectZero(float* v, int mode, cudaStream_t st)
{
    k_project<<<1, c_redThreads, 0, st>>>(v, g.n, (mode == 1) ? 2 : 1, 0.0, 0.0, 2);
}

void readScalars(cudaStream_t st)
{
    CPM_CU_CHECK(cudaMemcpyAsync(g.hS, g.s, sCount * sizeof(double), cudaMemcpyDeviceToHost, st));
    CPM_CU_CHECK(cudaStreamSynchronize(st));
}

} // namespace

bool cpm_gpu_cg_available()
{
    return true;
}

void cpm_gpu_cg_set_nlist(const int* rowPtr, const int* colIdx, const real* coef, int n, int nnz)
{
    ensureVectors(n);
    if (nnz > g.nnzAlloc)
    {
        devAlloc(&g.colIdx, nnz);
        devAlloc(&g.coef, nnz);
        g.nnzAlloc = nnz;
    }
    CPM_CU_CHECK(cudaMemcpy(g.rowPtr, rowPtr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (nnz > 0)
    {
        CPM_CU_CHECK(cudaMemcpy(g.colIdx, colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice));
        CPM_CU_CHECK(cudaMemcpy(g.coef, coef, nnz * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void cpm_gpu_cg_set_step(const real* z, const real* selfCoef, const real* tfDiag, int n, real prefactor3dc, real evscale, real potUnit)
{
    ensureVectors(n);
    CPM_CU_CHECK(cudaMemcpy(g.z, z, n * sizeof(float), cudaMemcpyHostToDevice));
    CPM_CU_CHECK(cudaMemcpy(g.self, selfCoef, n * sizeof(float), cudaMemcpyHostToDevice));
    g.haveTf = (tfDiag != nullptr);
    if (g.haveTf)
    {
        CPM_CU_CHECK(cudaMemcpy(g.tf, tfDiag, n * sizeof(float), cudaMemcpyHostToDevice));
    }
    g.pref3dc = prefactor3dc;
    g.evscale = evscale;
    g.potUnit = potUnit;
}

void cpm_gpu_cg_Aq(gmx_pme_t* pme, const real* x, real* y, int n)
{
    ensureVectors(n);
    cudaStream_t st = static_cast<cudaStream_t>(pme_gpu_cpm_stream(pme));
    CPM_CU_CHECK(cudaMemcpyAsync(g.xtmp, x, n * sizeof(float), cudaMemcpyHostToDevice, st));
    deviceAq(pme, g.xtmp, g.Ad, st);
    CPM_CU_CHECK(cudaMemcpyAsync(y, g.Ad, n * sizeof(float), cudaMemcpyDeviceToHost, st));
    CPM_CU_CHECK(cudaStreamSynchronize(st));
}

int cpm_gpu_cg_solve(gmx_pme_t* pme, const real* b, real* q, int n, int mode, real Q1, real Q2, real tol, int maxiter, double* cellVoltage)
{
    const auto t0 = std::chrono::steady_clock::now();
    ensureVectors(n);
    cudaStream_t st = static_cast<cudaStream_t>(pme_gpu_cpm_stream(pme));
    const int    nb = vecBlocks(n);

    CPM_CU_CHECK(cudaMemcpyAsync(g.b, b, n * sizeof(float), cudaMemcpyHostToDevice, st));
    CPM_CU_CHECK(cudaMemcpyAsync(g.q, q, n * sizeof(float), cudaMemcpyHostToDevice, st));

    projectQ(mode, Q1, Q2, st);
    deviceAq(pme, g.q, g.Ad, st);
    k_residual<<<nb, c_vecThreads, 0, st>>>(g.r, g.b, g.Ad, n);
    projectZero(g.r, mode, st);
    CPM_CU_CHECK(cudaMemcpyAsync(g.d, g.r, n * sizeof(float), cudaMemcpyDeviceToDevice, st));
    k_dots<<<1, c_redThreads, 0, st>>>(g.r, g.d, sDotOld, nullptr, nullptr, -1, n, g.s, -1, -1);
    readScalars(st);
    double delta      = g.hS[sDotOld];
    double deltaBest  = delta;
    bool   trackBest  = (mode == 1);
    if (trackBest)
    {
        CPM_CU_CHECK(cudaMemcpyAsync(g.qBest, g.q, n * sizeof(float), cudaMemcpyDeviceToDevice, st));
    }

    int iter;
    for (iter = 0; iter < maxiter && delta > tol; iter++)
    {
        deviceAq(pme, g.d, g.Ad, st);
        k_dots<<<1, c_redThreads, 0, st>>>(g.d, g.Ad, sDAd, nullptr, nullptr, -1, n, g.s, -1, -1);
        k_update<<<nb, c_vecThreads, 0, st>>>(g.q, g.d, g.r, g.Ad, g.s, n);
        const bool restart = ((iter + 1) % 20 == 0);
        if (restart)
        {
            projectQ(mode, Q1, Q2, st);
            deviceAq(pme, g.q, g.Ad, st);
            k_residual<<<nb, c_vecThreads, 0, st>>>(g.r, g.b, g.Ad, n);
        }
        projectZero(g.r, mode, st);
        k_dots<<<1, c_redThreads, 0, st>>>(g.r, g.r, sDotNew, nullptr, nullptr, -1, n, g.s, -1, -1);
        k_direction<<<nb, c_vecThreads, 0, st>>>(g.d, g.r, g.s, restart ? 1 : 0, n);
        projectZero(g.d, mode, st);
        /* delta = r.d, then dot_old <- dot_new for the next iteration */
        k_dots<<<1, c_redThreads, 0, st>>>(g.r, g.d, sDelta, nullptr, nullptr, -1, n, g.s, sDotNew, sDotOld);
        readScalars(st);
        if (std::abs(g.hS[sDAd]) < 1e-30)
        {
            break; // q was left untouched by k_update, as on the host
        }
        delta = g.hS[sDelta];
        if (trackBest && delta < deltaBest)
        {
            deltaBest = delta;
            CPM_CU_CHECK(cudaMemcpyAsync(g.qBest, g.q, n * sizeof(float), cudaMemcpyDeviceToDevice, st));
        }
    }
    if (trackBest && delta > deltaBest)
    {
        CPM_CU_CHECK(cudaMemcpyAsync(g.q, g.qBest, n * sizeof(float), cudaMemcpyDeviceToDevice, st));
        delta = deltaBest;
    }
    projectQ(mode, Q1, Q2, st);
    if (mode != 0 && cellVoltage != nullptr)
    {
        deviceAq(pme, g.q, g.Ad, st);
        k_groupSumDiff<<<1, c_redThreads, 0, st>>>(g.b, g.Ad, n, (mode == 1) ? 2 : 1, g.s);
    }
    CPM_CU_CHECK(cudaMemcpyAsync(q, g.q, n * sizeof(float), cudaMemcpyDeviceToHost, st));
    readScalars(st);
    if (mode != 0 && cellVoltage != nullptr)
    {
        if (mode == 1)
        {
            const int    half = n / 2;
            const double psi1 = -g.hS[sSum1] / half, psi2 = -g.hS[sSum2] / (n - half);
            *cellVoltage      = psi1 - psi2;
        }
        else
        {
            *cellVoltage = -g.hS[sSum1] / n;
        }
    }
    CPM_CU_CHECK(cudaGetLastError());
    g.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (iter >= maxiter && delta >= tol)
    {
        return -iter;
    }
    return iter;
}

double cpm_gpu_cg_elapsed_ms()
{
    return g.ms;
}
