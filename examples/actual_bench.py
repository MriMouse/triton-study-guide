"""Measure the vector-add kernel with Triton's do_bench on a real GPU.

This is a single-machine teaching measurement, not a benchmark suite. The
reported bandwidth uses the three effective FP32 streams (read x, read y,
write out) and excludes allocation and compilation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def main() -> None:
    print("triton_version", triton.__version__)
    print("triton_path", triton.__file__)
    print("device", torch.cuda.get_device_name())

    n = 1 << 20
    block = 1024
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y = torch.arange(n, device="cuda", dtype=torch.float32) + 1
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    def launch():
        return add_kernel[grid](x, y, out, n, BLOCK_SIZE=block)

    kernel = launch()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x + y, rtol=0, atol=0)
    print("n", n, "block", block, "grid", grid({"BLOCK_SIZE": block}))
    print("n_regs", kernel.n_regs, "n_spills", kernel.n_spills)

    ms = triton.testing.do_bench(launch, warmup=10, rep=50, return_mode="median")
    gbytes = 3 * n * 4 / 1e9
    print("median_ms", ms)
    print("effective_gbytes", gbytes)
    print("effective_GB_per_s", gbytes / (ms * 1e-3))


if __name__ == "__main__":
    main()
