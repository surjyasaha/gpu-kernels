"""
Triton kernels: vector add + fused row-wise softmax, benchmarked against PyTorch.
Run on a CUDA GPU (e.g. a free Colab T4):  pip install triton  (usually preinstalled)
    python triton_kernels.py
"""
import torch
import triton
import triton.language as tl


# ---------- 1. Vector add ----------
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out


# ---------- 2. Fused softmax (one program per row) ----------
@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols, x_stride, out_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)          # subtract max for numerical stability
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_stride + cols, num / den, mask=mask)


def softmax(x):
    rows, cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(cols)
    softmax_kernel[(rows,)](x, out, cols, x.stride(0), out.stride(0), BLOCK=BLOCK)
    return out


def naive_softmax(x):
    """Unfused PyTorch baseline: several kernel launches and memory round-trips."""
    z = x - x.max(dim=1, keepdim=True).values
    num = torch.exp(z)
    return num / num.sum(dim=1, keepdim=True)


# ---------- Correctness ----------
def check():
    torch.manual_seed(0)
    x = torch.randn(98432, device="cuda")
    y = torch.randn(98432, device="cuda")
    assert torch.allclose(add(x, y), x + y), "add mismatch"

    m = torch.randn(1823, 781, device="cuda")
    assert torch.allclose(softmax(m), torch.softmax(m, dim=1), atol=1e-5), "softmax mismatch"
    print("Correctness checks passed.")


# ---------- Benchmark ----------
def bench():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    print("Vector add (GB/s, higher is better)")
    print(f"{'N':>12} {'triton':>10} {'torch':>10}")
    for p in range(18, 27, 2):
        n = 2 ** p
        x = torch.rand(n, device="cuda")
        y = torch.rand(n, device="cuda")
        gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
        t_tr = triton.testing.do_bench(lambda: add(x, y))
        t_pt = triton.testing.do_bench(lambda: x + y)
        print(f"{n:>12} {gbps(t_tr):>10.1f} {gbps(t_pt):>10.1f}")

    print("\nSoftmax, 4096 rows (GB/s, higher is better)")
    print(f"{'cols':>12} {'triton':>10} {'torch':>10} {'naive':>10}")
    for cols in [256, 512, 1024, 2048, 4096, 8192]:
        x = torch.randn(4096, cols, device="cuda")
        gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
        t_tr = triton.testing.do_bench(lambda: softmax(x))
        t_pt = triton.testing.do_bench(lambda: torch.softmax(x, dim=1))
        t_nv = triton.testing.do_bench(lambda: naive_softmax(x))
        print(f"{cols:>12} {gbps(t_tr):>10.1f} {gbps(t_pt):>10.1f} {gbps(t_nv):>10.1f}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "Needs a CUDA GPU (try Google Colab: Runtime > T4 GPU)"
    check()
    bench()
