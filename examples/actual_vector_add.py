"""Run the book's N=10, B=4 vector-add example on a real GPU.

This is the self-contained version of the code discussed in chapter 4.1 and
4.2. It prints the actual grid, output, compiled stage names and resource
counts so the book can quote a real run instead of a paper-only trace.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def main() -> None:
    print("triton_version", triton.__version__)
    print("triton_path", triton.__file__)
    print("torch_version", torch.__version__)
    print("cuda_version", torch.version.cuda)
    print("device", torch.cuda.get_device_name())

    n = 10
    block = 4
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y = torch.arange(1, n + 1, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    grid = (triton.cdiv(n, block),)
    kernel = add_kernel[grid](x, y, out, n, BLOCK_SIZE=block)
    torch.cuda.synchronize()

    print("n", n, "block", block, "grid", grid)
    print("x", x.cpu().tolist())
    print("y", y.cpu().tolist())
    print("out", out.cpu().tolist())
    print("reference", (x + y).cpu().tolist())
    print("asm_stages", sorted(kernel.asm.keys()))
    print("n_regs", kernel.n_regs)
    print("n_spills", kernel.n_spills)
    print("shared_bytes", kernel.metadata.shared)

    # The second call reuses the same compilation conditions.
    out2 = torch.empty_like(x)
    add_kernel[grid](x + 10, y + 10, out2, n, BLOCK_SIZE=block)
    torch.cuda.synchronize()
    print("second_call_out", out2.cpu().tolist())


if __name__ == "__main__":
    main()
