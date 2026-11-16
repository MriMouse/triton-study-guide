"""Run the book's small explicit-multiply GEMM teaching kernel on a real GPU.

The 2x3 @ 3x2 example is deliberately tiny so every product is easy to check.
It is not a performance-oriented GEMM and does not use tl.dot.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def teaching_gemm(A, B, C, M, N, K,
                  sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, BK)):
        ks = kb * BK + kk
        ap = A + rows[:, None] * sam + ks[None, :] * sak
        bp = B + ks[:, None] * sbk + cols[None, :] * sbn
        av = tl.load(ap, (rows[:, None] < M) & (ks[None, :] < K), other=0).to(tl.float32)
        bv = tl.load(bp, (ks[:, None] < K) & (cols[None, :] < N), other=0).to(tl.float32)
        products = av[:, :, None] * bv[None, :, :]
        acc = acc + tl.sum(products, axis=1)
    cp = C + rows[:, None] * scm + cols[None, :] * scn
    tl.store(cp, acc, (rows[:, None] < M) & (cols[None, :] < N))


def main() -> None:
    print("triton_version", triton.__version__)
    print("triton_path", triton.__file__)
    print("device", torch.cuda.get_device_name())

    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")
    b = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], device="cuda")
    c = torch.empty((2, 2), device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(a.shape[0], 2), triton.cdiv(b.shape[1], 2))
    teaching_gemm[grid](
        a, b, c, a.shape[0], b.shape[1], a.shape[1],
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1), BM=2, BN=2, BK=2,
    )
    torch.cuda.synchronize()

    reference = a @ b
    print("grid", grid)
    print("output", c.cpu().tolist())
    print("reference", reference.cpu().tolist())
    print("max_abs_diff", (c - reference).abs().max().item())


if __name__ == "__main__":
    main()
