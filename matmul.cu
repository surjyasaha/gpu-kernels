// CUDA matrix multiply: naive -> shared-memory tiled, compared against cuBLAS.
// Build:  nvcc -O3 matmul.cu -o matmul -lcublas
// Run:    ./matmul
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <functional>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

// ---------- Kernel 1: naive. One thread per output element. ----------
// Every thread re-reads a full row of A and column of B from global memory.
__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float s = 0.f;
        for (int k = 0; k < n; k++) s += A[row * n + k] * B[k * n + col];
        C[row * n + col] = s;
    }
}

// ---------- Kernel 2: tiled. Threads in a block cooperatively load TILExTILE ----------
// tiles of A and B into shared memory, so each global value is loaded once per
// tile and reused TILE times from fast on-chip memory.
#define TILE 32
__global__ void matmul_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float s = 0.f;
    for (int t = 0; t < (n + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < n && aCol < n) ? A[row * n + aCol] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < n && col < n) ? B[bRow * n + col] : 0.f;
        __syncthreads();                       // tile fully loaded before use
        for (int k = 0; k < TILE; k++) s += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();                       // done using tile before overwriting
    }
    if (row < n && col < n) C[row * n + col] = s;
}

// ---------- Helpers ----------
float time_ms(const std::function<void()>& fn, int iters = 10) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    fn(); CHECK(cudaDeviceSynchronize());      // warm-up
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) fn();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters;
}

double gflops(int n, float ms) { return 2.0 * n * n * (double)n / (ms * 1e-3) / 1e9; }

float max_abs_diff(const std::vector<float>& x, const std::vector<float>& y) {
    float d = 0.f;
    for (size_t i = 0; i < x.size(); i++) d = fmaxf(d, fabsf(x[i] - y[i]));
    return d;
}

int main() {
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    cublasHandle_t handle; cublasCreate(&handle);
    const float alpha = 1.f, beta = 0.f;

    printf("%6s | %10s %10s %10s | %8s %8s\n",
           "N", "naive", "tiled", "cuBLAS", "err(nv)", "err(tl)");
    printf("       |   (GFLOPS, higher is better)         | (max abs diff vs cuBLAS)\n");

    for (int n : {256, 512, 1024, 2048}) {
        size_t bytes = (size_t)n * n * sizeof(float);
        std::vector<float> hA(n * n), hB(n * n), hC(n * n), hRef(n * n);
        for (auto& v : hA) v = rand() / (float)RAND_MAX - 0.5f;
        for (auto& v : hB) v = rand() / (float)RAND_MAX - 0.5f;

        float *dA, *dB, *dC, *dRef;
        CHECK(cudaMalloc(&dA, bytes)); CHECK(cudaMalloc(&dB, bytes));
        CHECK(cudaMalloc(&dC, bytes)); CHECK(cudaMalloc(&dRef, bytes));
        CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

        dim3 block(TILE, TILE);
        dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

        // cuBLAS is column-major. Row-major C = A*B equals column-major C^T = B^T * A^T,
        // so we pass B first and A second with no transposes.
        auto run_cublas = [&] {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                        &alpha, dB, n, dA, n, &beta, dRef, n);
        };
        auto run_naive = [&] { matmul_naive<<<grid, block>>>(dA, dB, dC, n); };
        auto run_tiled = [&] { matmul_tiled<<<grid, block>>>(dA, dB, dC, n); };

        float t_cublas = time_ms(run_cublas);
        CHECK(cudaMemcpy(hRef.data(), dRef, bytes, cudaMemcpyDeviceToHost));

        float t_naive = time_ms(run_naive);
        CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
        float e_naive = max_abs_diff(hC, hRef);

        float t_tiled = time_ms(run_tiled);
        CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
        float e_tiled = max_abs_diff(hC, hRef);

        printf("%6d | %10.1f %10.1f %10.1f | %8.1e %8.1e\n", n,
               gflops(n, t_naive), gflops(n, t_tiled), gflops(n, t_cublas), e_naive, e_tiled);

        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    }
    cublasDestroy(handle);
    return 0;
}
