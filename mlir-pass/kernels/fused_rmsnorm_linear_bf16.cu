#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include "cute/tensor.hpp"

using namespace cute;
#define THREADS 256

extern "C" __global__
void rmsnorm_cute_kernel_bf16(
    const __nv_bfloat16 * __restrict__ x,
    const __nv_bfloat16 * __restrict__ w_norm,
          __nv_bfloat16 * __restrict__ out,
    int batch_seq, int d_in, float epsilon)
{
    int row = blockIdx.x;
    if (row >= batch_seq) return;

    auto layout    = make_layout(make_shape(d_in), make_stride(Int<1>{}));
    auto x_tensor  = make_tensor(make_gmem_ptr(x   + (int64_t)row * d_in), layout);
    auto w_tensor  = make_tensor(make_gmem_ptr(w_norm), layout);
    auto out_tensor= make_tensor(make_gmem_ptr(out + (int64_t)row * d_in), layout);

    __shared__ float smem[THREADS/32];

    float ss = 0.f;
    for (int k = threadIdx.x; k < d_in; k += THREADS) {
        float v = __bfloat162float(x_tensor(k));
        ss += v * v;
    }
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

    for (int k = threadIdx.x; k < d_in; k += THREADS) {
        float v = __bfloat162float(x_tensor(k)) * rms * __bfloat162float(w_tensor(k));
        out_tensor(k) = __float2bfloat16(v);
    }
}

static cublasHandle_t g_handle_bf16 = nullptr;
static __nv_bfloat16 *norm_buf_bf16 = nullptr;
static int norm_buf_size_bf16 = 0;

extern "C" void fused_rmsnorm_linear_cuda_bf16(
    const __nv_bfloat16 *x, const __nv_bfloat16 *w_norm,
    const __nv_bfloat16 *w_proj, __nv_bfloat16 *y,
    int batch_seq, int d_in, int d_out, float epsilon)
{
    int needed = batch_seq * d_in;
    if (needed > norm_buf_size_bf16) {
        if (norm_buf_bf16) cudaFree(norm_buf_bf16);
        cudaMalloc(&norm_buf_bf16, needed * sizeof(__nv_bfloat16));
        norm_buf_size_bf16 = needed;
    }

    rmsnorm_cute_kernel_bf16<<<batch_seq, THREADS>>>(
        x, w_norm, norm_buf_bf16, batch_seq, d_in, epsilon);

    if (!g_handle_bf16) {
        cublasCreate(&g_handle_bf16);
        cublasSetMathMode(g_handle_bf16, CUBLAS_TENSOR_OP_MATH);
    }

    const float alpha = 1.f, beta = 0.f;
    cublasGemmEx(g_handle_bf16,
        CUBLAS_OP_T, CUBLAS_OP_N,
        d_out, batch_seq, d_in,
        &alpha,
        w_proj,      CUDA_R_16BF, d_in,
        norm_buf_bf16, CUDA_R_16BF, d_in,
        &beta,
        y,           CUDA_R_16BF, d_out,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "[fused_rmsnorm_linear_cuda_bf16] %s\n", cudaGetErrorString(err));
}
