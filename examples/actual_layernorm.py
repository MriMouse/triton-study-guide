"""Run small LayerNorm and RMSNorm teaching kernels on a real GPU.

The LayerNorm kernel keeps the three-pass structure used in the official
tutorial but is simplified to one block per row so it can be read together
with the 2x3 numerical example in chapter 9.3. RMSNorm follows the book's
teaching rewrite and is not claimed to be an upstream tutorial kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(X, Y, W, B, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    base = row * N

    s = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, N, BLOCK):
        c = off + cols
        v = tl.load(X + base + c, mask=c < N, other=0.0).to(tl.float32)
        s += v
    mean = tl.sum(s, axis=0) / N

    var_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, N, BLOCK):
        c = off + cols
        v = tl.load(X + base + c, mask=c < N, other=0.0).to(tl.float32)
        centered = tl.where(c < N, v - mean, 0.0)
        var_acc += centered * centered
    var = tl.sum(var_acc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK):
        c = off + cols
        m = c < N
        v = tl.load(X + base + c, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + c, mask=m, other=0.0).to(tl.float32)
        bias = tl.load(B + c, mask=m, other=0.0).to(tl.float32)
        y = (v - mean) * rstd * w + bias
        tl.store(Y + base + c, y, mask=m)


@triton.jit
def rmsnorm_kernel(X, Y, W, N: tl.constexpr, B: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, B)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / N + EPS)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * N + cols, x * inv * w, mask=mask)


def main() -> None:
    print("triton_version", triton.__version__)
    print("triton_path", triton.__file__)
    print("device", torch.cuda.get_device_name())

    x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], device="cuda")
    w = torch.tensor([2.0, 1.0, 0.5], device="cuda")
    b = torch.tensor([0.0, 1.0, -1.0], device="cuda")
    y = torch.empty_like(x)
    block = triton.next_power_of_2(x.shape[1])
    layernorm_kernel[(x.shape[0],)](x, y, w, b, x.shape[1], 1e-5, BLOCK=block)
    torch.cuda.synchronize()

    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, unbiased=False, keepdim=True)
    reference = (x - mean) / torch.sqrt(var + 1e-5) * w + b
    print("layernorm_output", y.cpu().tolist())
    print("layernorm_reference", reference.cpu().tolist())
    print("layernorm_max_abs_diff", (y - reference).abs().max().item())

    yr = torch.empty_like(x)
    rmsnorm_kernel[(x.shape[0],)](x, yr, w, N=x.shape[1], B=block, EPS=1e-5)
    torch.cuda.synchronize()
    rms_ref = x * torch.rsqrt((x * x).mean(dim=1, keepdim=True) + 1e-5) * w
    print("rmsnorm_output", yr.cpu().tolist())
    print("rmsnorm_reference", rms_ref.cpu().tolist())
    print("rmsnorm_max_abs_diff", (yr - rms_ref).abs().max().item())


if __name__ == "__main__":
    main()
