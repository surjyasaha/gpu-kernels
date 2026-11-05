"""
Triton tiled matmul (fp16 inputs, fp32 accumulation) vs torch.matmul (cuBLAS).
Run on a CUDA GPU:  python triton_matmul.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Each program computes one BM x BN tile of C.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    a_ptrs = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    b_ptrs = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        k_left = K - k * BK
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < k_left), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] < k_left) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)            # maps to tensor cores on supported GPUs
        a_ptrs += BK * sak
        b_ptrs += BK * sbk

    c_ptrs = c_ptr + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul(a, b, BM=64, BN=64, BK=32):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK)
    return c


def check():
    torch.manual_seed(0)
    a = torch.randn(512, 384, device="cuda", dtype=torch.float16)
    b = torch.randn(384, 640, device="cuda", dtype=torch.float16)
    ref = torch.matmul(a, b)
    out = matmul(a, b)
    err = (out.float() - ref.float()).abs().max().item()
    assert torch.allclose(out, ref, atol=1e-1, rtol=1e-2), f"mismatch, max err {err}"
    print(f"Correctness passed (max abs err {err:.3e}).")


def bench():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    print(f"{'N':>6} {'triton TFLOPS':>15} {'torch TFLOPS':>15}")
    for n in [512, 1024, 2048, 4096]:
        a = torch.randn(n, n, device="cuda", dtype=torch.float16)
        b = torch.randn(n, n, device="cuda", dtype=torch.float16)
        tflops = lambda ms: 2 * n ** 3 / (ms * 1e-3) / 1e12
        t_tr = triton.testing.do_bench(lambda: matmul(a, b))
        t_pt = triton.testing.do_bench(lambda: torch.matmul(a, b))
        print(f"{n:>6} {tflops(t_tr):>15.2f} {tflops(t_pt):>15.2f}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "Needs a CUDA GPU (Colab: Runtime > T4 GPU)"
    check()
    bench()
