# GPU Kernels: CUDA and Triton (learning project)

My first GPU kernels, written to learn how memory access patterns drive performance.
Everything was benchmarked on a **Tesla T4** (Google Colab, CUDA 13.0, Triton 3.6.0).
The kernels follow standard tutorial patterns (shared-memory tiled matmul, fused softmax);
the goal was to implement them, verify them, and understand the results.

## Files

| File | What it is |
|---|---|
| `matmul.cu` | CUDA matmul: naive vs shared-memory tiled, checked against cuBLAS |
| `triton_kernels.py` | Triton vector-add and fused row-wise softmax vs PyTorch |
| `triton_matmul.py` | Triton tiled matmul (fp16, fp32 accumulate) vs `torch.matmul` |

## Run

```
nvcc -O3 -arch=sm_75 matmul.cu -o matmul -lcublas && ./matmul
python triton_kernels.py
python triton_matmul.py
```

## Results (Tesla T4)

### CUDA matmul (fp32, GFLOPS)

| N | naive | tiled | cuBLAS |
|---|---|---|---|
| 256 | 421 | 514 | 959 |
| 512 | 509 | 634 | 3,260 |
| 1024 | 518 | 665 | 4,261 |
| 2048 | 575 | 885 | 5,804 |

All outputs match cuBLAS to within ~4e-5 max absolute difference.

- Shared-memory tiling gives **1.2x to 1.5x** over the naive kernel.
- cuBLAS is still **~6.5x faster** than my tiled kernel at N=2048.
- Why: in my tiled kernel each thread computes a single output, so every multiply-add
  needs two shared-memory reads and the kernel is limited by shared-memory bandwidth
  and instruction overhead. cuBLAS uses register blocking (each thread computes a small
  block of outputs and reuses values from registers), vectorized loads, and double
  buffering. These are my next steps.

### Triton fused softmax (4096 rows, GB/s)

| cols | Triton | PyTorch | unfused naive |
|---|---|---|---|
| 256 | 219 | 216 | 55 |
| 1024 | 227 | 213 | 58 |
| 4096 | 228 | 242 | 60 |
| 8192 | 227 | 237 | 61 |

- Fusing the max, exp, sum, and divide into one kernel is **~4x faster** than the unfused
  version, because each row is read and written once instead of making several passes
  through global memory. It is roughly on par with PyTorch's built-in softmax.
- Vector add matches PyTorch (~240 GB/s) since both are limited by memory bandwidth
  (T4 peak is about 320 GB/s).

### Triton matmul: known limitation

| N | Triton (TFLOPS) | torch.matmul (TFLOPS) |
|---|---|---|
| 512 | 0.87 | 6.90 |
| 1024 | 1.13 | 34.34 |
| 2048 | 1.01 | 33.90 |
| 4096 | 0.99 | 21.79 |

Correct (max abs error 3e-2 in fp16) but about 30x slower than cuBLAS. A sweep of block
sizes, warps, and stages (64x64 up to 128x128) never exceeded ~1.1 TFLOPS, and larger
tiles got slower. My likely explanation is that `tl.dot` does not use tensor cores on
this Turing GPU in Triton 3.6.0 and falls back to scalar FMA, but I have not confirmed
this. Next step: rerun on an Ampere GPU (A100/L4) and check the generated PTX.

## What I learned

- Matmul performance is mostly about data reuse. Moving data into shared memory helps,
  but registers are what close the gap to cuBLAS.
- Kernel fusion matters most for memory-bound ops like softmax.
- A kernel being correct and a kernel being fast are separate problems.

## Next steps

- Register-blocked CUDA matmul (each thread computes a 4x4 or 8x8 block)
- Fused attention kernel in Triton
- Profile with Nsight Compute to confirm the bottlenecks above
