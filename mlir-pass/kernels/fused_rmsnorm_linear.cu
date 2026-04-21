#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

#define THREADS 256

extern "C" __global__
void rmsnorm_kernel(
    const __half * __restrict__ x,
    const __half * __restrict__ w_norm,
          __half * __restrict__ out,
    int batch_seq, int d_in, float epsilon)
{
    int row = blockIdx.x;
    if (row >= batch_seq) return;

    __shared__ float smem[THREADS/32];
    const __half *xrow = x   + (int64_t)row * d_in;
          __half *orow = out  + (int64_t)row * d_in;

    float ss = 0.f;
    for (int k = threadIdx.x; k < d_in; k += THREADS)
        ss += __half2float(xrow[k]) * __half2float(xrow[k]);

    for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) smem[wid] = ss;
    __syncthreads();
    if (wid == 0) {
        ss = (lane < (THREADS/32)) ? smem[lane] : 0.f;
        for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
        if (lane == 0) smem[0] = rsqrtf(ss / (float)d_in + epsilon);
    }
    __syncthreads();
    float rms = smem[0];

    for (int k = threadIdx.x; k < d_in; k += THREADS)
        orow[k] = __float2half(__half2float(xrow[k]) * rms * __half2float(w_norm[k]));
}

static cublasHandle_t g_handle = nullptr;
static __half *norm_buf = nullptr;
static int norm_buf_size = 0;

extern "C" void fused_rmsnorm_linear_cuda(
    const __half *x, const __half *w_norm, const __half *w_proj,
    __half *y, int batch_seq, int d_in, int d_out, float epsilon)
{
    // Allocate temp buffer
    int needed = batch_seq * d_in;
    if (needed > norm_buf_size) {
        if (norm_buf) cudaFree(norm_buf);
        cudaMalloc(&norm_buf, needed * sizeof(__half));
        norm_buf_size = needed;
    }

    // Step 1: RMSNorm  →  norm_buf [batch_seq, d_in]
    rmsnorm_kernel<<<batch_seq, THREADS>>>(
        x, w_norm, norm_buf, batch_seq, d_in, epsilon);

    // Step 2: y [batch_seq, d_out] = norm_buf [batch_seq, d_in] @ w_proj.T [d_in, d_out]
    // cuBLAS is column-major. Treat matrices as transposed:
    //   norm_buf row-major [BS, DI]  =  col-major [DI, BS]  (matrix A^T)
    //   w_proj   row-major [DO, DI]  =  col-major [DI, DO]  (matrix B^T)
    //   y        row-major [BS, DO]  =  col-major [DO, BS]  (matrix C^T)
    // We want C = A @ B^T  →  C^T = B @ A^T
    // cuBLAS: C[DO,BS] = w_proj_colmaj[DI,DO]^T  *  norm_buf_colmaj[DI,BS]
    //       = GEMM(OP_T, OP_N, DO, BS, DI, w_proj, DI, norm_buf, DI, y, DO)
    if (!g_handle) {
        cublasCreate(&g_handle);
        cublasSetMathMode(g_handle, CUBLAS_TENSOR_OP_MATH);
    }

    const __half alpha = __float2half(1.f), beta = __float2half(0.f);
    cublasHgemm(g_handle,
        CUBLAS_OP_T,   // op on w_proj:   [DI,DO] -> [DO,DI]
        CUBLAS_OP_N,   // op on norm_buf: [DI,BS] stays
        d_out,         // rows of C^T
        batch_seq,     // cols of C^T
        d_in,          // inner dim
        &alpha,
        w_proj,   d_in,     // leading dim of w_proj col-major
        norm_buf, d_in,     // leading dim of norm_buf col-major
        &beta,
        y,        d_out);   // leading dim of y col-major

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "[fused_rmsnorm_linear_cuda] %s\n", cudaGetErrorString(err));
}
