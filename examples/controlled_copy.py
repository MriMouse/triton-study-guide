"""Controlled copy example for chapter 8.2.

Run only in a prepared environment with /research/runs/triton-copy created.
See ../chapters/08-backend-debug.md#sec-8-2 and ../CODE-NOTICE.md.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def copy_kernel(X, Y, N, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    v = tl.load(X + i, i < N, other=0)
    tl.store(Y + i, v, i < N)

def main():
    print("triton", triton.__version__, triton.__file__)
    print("torch", torch.__version__, "cuda", torch.version.cuda,
          "hip", torch.version.hip)
    print("device", torch.cuda.get_device_name())
    x = torch.arange(10, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    def launch():
        return copy_kernel[(triton.cdiv(x.numel(), 4),)](x, y, x.numel(), B=4)
    kernel = launch()                    # 首次可能包含编译与装载
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x, rtol=0, atol=0)
    for stage in ("ttir", "ttgir", "llir", "ptx"):
        if stage in kernel.asm:
            from pathlib import Path
            Path(f"/research/runs/triton-copy/{stage}.txt").write_text(kernel.asm[stage])
    ms = triton.testing.do_bench(launch, warmup=5, rep=20, return_mode="median")
    print("median kernel interval ms", ms)

if __name__ == "__main__":
    main()
