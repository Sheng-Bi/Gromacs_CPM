/*
 * bisheng 2026 - Constant-potential charge solve on the GPU.
 *
 * Dense y = A^{-1} x for the matrix-inversion solver. The inverse is stored
 * transposed on the device so that thread i, computing row i, reads
 * At[j*n + i] for j = 0..n-1: consecutive threads touch consecutive
 * addresses and the 8-byte reads coalesce. x[j] is a broadcast read served
 * from cache. The dense product is memory-bound.
 *
 * Reductions and the O(n) constraint algebra remain on the host.
 */
#include "gmxpre.h"

#include "cpm_gpu_solve.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace
{

double*       g_dA     = nullptr; // transposed inverse, double
float*        g_fA     = nullptr; // transposed inverse, float (optional)
double*       g_dx     = nullptr;
double*       g_dy     = nullptr;
float*        g_fx     = nullptr;
float*        g_fy     = nullptr;
int           g_n      = 0;
bool          g_single = false;
const double* g_key    = nullptr;
double        g_ms     = 0.0;
/* bisheng 2026 - a non-blocking stream and pinned staging allow the charge
   solve to overlap with the nonbonded force kernel in the rigid-electrode
   schedule. */
cudaStream_t  g_stream = nullptr;
double*       g_dpart  = nullptr; // chunk partials, double
float*        g_fpart  = nullptr; // chunk partials, float
double*       g_hx     = nullptr; // pinned staging, double
double*       g_hy     = nullptr;
float*        g_hxf    = nullptr; // pinned staging, float
float*        g_hyf    = nullptr;

bool cudaOk(cudaError_t stat, const char* what)
{
    if (stat != cudaSuccess)
    {
        fprintf(stderr, "CPM GPU solve: %s failed: %s\n", what, cudaGetErrorString(stat));
        return false;
    }
    return true;
}

/* y = A x with A stored transposed (At[j*n + i] = A[i][j]), so that the threads
   of a warp (consecutive i) read consecutive addresses for each j.
   bisheng 2026 - split the j range over gridDim.y chunks to expose more
   parallel work. A second kernel sums the partials in a fixed order. */
template<typename T>
__global__ void cpmGemvPartial(const T* __restrict__ At, const T* __restrict__ x, T* __restrict__ partial, int n, int chunk)
{
    const int i  = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * chunk;
    const int j1 = min(j0 + chunk, n);
    if (i >= n)
    {
        return;
    }
    T acc = 0;
    for (int j = j0; j < j1; ++j)
    {
        acc += At[static_cast<size_t>(j) * n + i] * x[j];
    }
    partial[static_cast<size_t>(blockIdx.y) * n + i] = acc;
}

template<typename T>
__global__ void cpmGemvReduce(const T* __restrict__ partial, T* __restrict__ y, int n, int nchunk)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    T acc = 0;
    for (int k = 0; k < nchunk; ++k)
    {
        acc += partial[static_cast<size_t>(k) * n + i];
    }
    y[i] = acc;
}

const int c_maxChunks = 64;

/* enough blocks to keep the memory system busy: ~512 in total */
int chunkCount(const int n, const int block)
{
    const int nblockRows = (n + block - 1) / block;
    int       nchunk     = (512 + nblockRows - 1) / nblockRows;
    nchunk               = nchunk < 1 ? 1 : (nchunk > c_maxChunks ? c_maxChunks : nchunk);
    return nchunk;
}

void releaseAll()
{
    if (g_dA) cudaFree(g_dA);
    if (g_fA) cudaFree(g_fA);
    if (g_dx) cudaFree(g_dx);
    if (g_dy) cudaFree(g_dy);
    if (g_fx) cudaFree(g_fx);
    if (g_fy) cudaFree(g_fy);
    if (g_dpart) cudaFree(g_dpart);
    if (g_fpart) cudaFree(g_fpart);
    g_dpart = nullptr; g_fpart = nullptr;
    if (g_hx) cudaFreeHost(g_hx);
    if (g_hy) cudaFreeHost(g_hy);
    if (g_hxf) cudaFreeHost(g_hxf);
    if (g_hyf) cudaFreeHost(g_hyf);
    if (g_stream) cudaStreamDestroy(g_stream);
    g_dA = nullptr; g_fA = nullptr; g_dx = nullptr; g_dy = nullptr; g_fx = nullptr; g_fy = nullptr;
    g_hx = nullptr; g_hy = nullptr; g_hxf = nullptr; g_hyf = nullptr; g_stream = nullptr;
    g_n = 0; g_key = nullptr;
}

} // namespace

bool cpm_gpu_solve_available()
{
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess)
    {
        return false;
    }
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

bool cpm_gpu_solve_init(const double* invA, const int n, const bool useSingle)
{
    if (invA == g_key && n == g_n && useSingle == g_single && (useSingle ? g_fA != nullptr : g_dA != nullptr))
    {
        return true;
    }
    releaseAll();
    g_single = useSingle;
    const size_t nn = static_cast<size_t>(n) * n;
    if (useSingle)
    {
        std::vector<float> At(nn);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                At[static_cast<size_t>(j) * n + i] = static_cast<float>(invA[static_cast<size_t>(i) * n + j]);
        if (!cudaOk(cudaMalloc(&g_fA, nn * sizeof(float)), "cudaMalloc(A, float)")
            || !cudaOk(cudaMemcpy(g_fA, At.data(), nn * sizeof(float), cudaMemcpyHostToDevice), "upload(A, float)")
            || !cudaOk(cudaMalloc(&g_fx, n * sizeof(float)), "cudaMalloc(x)")
            || !cudaOk(cudaMalloc(&g_fy, n * sizeof(float)), "cudaMalloc(y)")
            || !cudaOk(cudaMalloc(&g_fpart, static_cast<size_t>(c_maxChunks) * n * sizeof(float)), "cudaMalloc(partial)"))
        {
            releaseAll();
            return false;
        }
    }
    else
    {
        std::vector<double> At(nn);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                At[static_cast<size_t>(j) * n + i] = invA[static_cast<size_t>(i) * n + j];
        if (!cudaOk(cudaMalloc(&g_dA, nn * sizeof(double)), "cudaMalloc(A)")
            || !cudaOk(cudaMemcpy(g_dA, At.data(), nn * sizeof(double), cudaMemcpyHostToDevice), "upload(A)")
            || !cudaOk(cudaMalloc(&g_dx, n * sizeof(double)), "cudaMalloc(x)")
            || !cudaOk(cudaMalloc(&g_dy, n * sizeof(double)), "cudaMalloc(y)")
            || !cudaOk(cudaMalloc(&g_dpart, static_cast<size_t>(c_maxChunks) * n * sizeof(double)), "cudaMalloc(partial)"))
        {
            releaseAll();
            return false;
        }
    }
    /* Highest priority, like the PME stream: the solve runs while the nonbonded
       force kernel has the GPU full, and blocks of an equal-priority stream are
       only dispatched once the force kernel's remaining blocks have been. */
    int lowestPriority = 0, highestPriority = 0;
    cudaDeviceGetStreamPriorityRange(&lowestPriority, &highestPriority);
    if (!cudaOk(cudaStreamCreateWithPriority(&g_stream, cudaStreamNonBlocking, highestPriority), "cudaStreamCreate(solve)")
        || (useSingle
                    ? (!cudaOk(cudaMallocHost(&g_hxf, n * sizeof(float)), "cudaMallocHost(x)")
                       || !cudaOk(cudaMallocHost(&g_hyf, n * sizeof(float)), "cudaMallocHost(y)"))
                    : (!cudaOk(cudaMallocHost(&g_hx, n * sizeof(double)), "cudaMallocHost(x)")
                       || !cudaOk(cudaMallocHost(&g_hy, n * sizeof(double)), "cudaMallocHost(y)"))))
    {
        releaseAll();
        return false;
    }
    g_n   = n;
    g_key = invA;
    printf("CPM GPU solve: inverse matrix (%d x %d, %.0f MB, %s) resident on the device\n", n, n,
           nn * (useSingle ? sizeof(float) : sizeof(double)) / 1.0e6, useSingle ? "single" : "double");
    return true;
}

bool cpm_gpu_solve_gemv(const double* x, double* y, const int n)
{
    if (n != g_n || (g_single ? g_fA == nullptr : g_dA == nullptr))
    {
        return false;
    }
    const auto t0     = std::chrono::steady_clock::now();
    const int  block  = 128;
    const int  nblock = (n + block - 1) / block;
    const int  nchunk = chunkCount(n, block);
    const int  chunk  = (n + nchunk - 1) / nchunk;
    bool       ok;
    if (g_single)
    {
        for (int i = 0; i < n; ++i) g_hxf[i] = static_cast<float>(x[i]);
        ok = cudaOk(cudaMemcpyAsync(g_fx, g_hxf, n * sizeof(float), cudaMemcpyHostToDevice, g_stream), "upload(x)");
        if (ok)
        {
            cpmGemvPartial<float><<<dim3(nblock, nchunk), block, 0, g_stream>>>(g_fA, g_fx, g_fpart, n, chunk);
            cpmGemvReduce<float><<<nblock, block, 0, g_stream>>>(g_fpart, g_fy, n, nchunk);
            ok = cudaOk(cudaGetLastError(), "gemv kernel")
                 && cudaOk(cudaMemcpyAsync(g_hyf, g_fy, n * sizeof(float), cudaMemcpyDeviceToHost, g_stream), "download(y)")
                 && cudaOk(cudaStreamSynchronize(g_stream), "sync(solve)");
            for (int i = 0; i < n; ++i) y[i] = g_hyf[i];
        }
    }
    else
    {
        for (int i = 0; i < n; ++i) g_hx[i] = x[i];
        ok = cudaOk(cudaMemcpyAsync(g_dx, g_hx, n * sizeof(double), cudaMemcpyHostToDevice, g_stream), "upload(x)");
        if (ok)
        {
            cpmGemvPartial<double><<<dim3(nblock, nchunk), block, 0, g_stream>>>(g_dA, g_dx, g_dpart, n, chunk);
            cpmGemvReduce<double><<<nblock, block, 0, g_stream>>>(g_dpart, g_dy, n, nchunk);
            ok = cudaOk(cudaGetLastError(), "gemv kernel")
                 && cudaOk(cudaMemcpyAsync(g_hy, g_dy, n * sizeof(double), cudaMemcpyDeviceToHost, g_stream), "download(y)")
                 && cudaOk(cudaStreamSynchronize(g_stream), "sync(solve)");
            for (int i = 0; i < n; ++i) y[i] = g_hy[i];
        }
    }
    g_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return ok;
}

void cpm_gpu_solve_free()
{
    releaseAll();
}

double cpm_gpu_solve_elapsed_ms()
{
    return g_ms;
}
