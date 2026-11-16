"""Run the book's 2x3 stable-softmax teaching kernel on a real GPU."""

import torch
import triton
import triton.language as tl


@triton.jit
def row_softmax(out_ptr, in_ptr, in_stride, out_stride, n_cols, BLOCK: tl.constexpr):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    row = tl.load(in_ptr + row_idx * in_stride + cols, mask=mask, other=-float("inf")).to(tl.float32)
    centered = row - tl.max(row, axis=0)
    numerator = tl.exp(centered)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator
    tl.store(out_ptr + row_idx * out_stride + cols, result, mask=mask)


def main() -> None:
    print("triton_version", triton.__version__)
    print("triton_path", triton.__file__)
    print("device", torch.cuda.get_device_name())

    x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]], device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    block = triton.next_power_of_2(x.shape[1])
    row_softmax[(x.shape[0],)](out, x, x.stride(0), out.stride(0), x.shape[1], BLOCK=block)
    torch.cuda.synchronize()

    reference = torch.softmax(x, dim=1)
    print("input", x.cpu().tolist())
    print("output", out.cpu().tolist())
    print("reference", reference.cpu().tolist())
    print("max_abs_diff", (out - reference).abs().max().item())
    print("block", block)


if __name__ == "__main__":
    main()
