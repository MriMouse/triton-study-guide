"""Compile a small vector-add kernel and print selected lines from each stage.

The full stage text is written under /research/runs/book-actual/stages so the
book can quote real TTIR/TTGIR/LLIR/PTX snippets with a reproducible command.
"""

import re
from pathlib import Path

import torch
import triton
import triton.language as tl

OUT = Path("/research/runs/book-actual/stages")


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def show(name: str, text: str, patterns: list[str], limit: int = 14) -> None:
    lines = text.splitlines()
    print(f"--- {name}: {len(lines)} lines, {len(text)} bytes ---")
    shown = 0
    for no, line in enumerate(lines, 1):
        if any(p in line for p in patterns):
            print(f"{no}: {line.strip()}")
            shown += 1
            if shown >= limit:
                break


def main() -> None:
    print("triton_version", triton.__version__)
    print("triton_path", triton.__file__)
    print("device", torch.cuda.get_device_name())

    n, block = 10, 4
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y = torch.arange(1, n + 1, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    kernel = add_kernel[(triton.cdiv(n, block),)](x, y, out, n, BLOCK_SIZE=block)
    torch.cuda.synchronize()

    OUT.mkdir(parents=True, exist_ok=True)
    for stage in ("ttir", "ttgir", "llir", "ptx"):
        text = kernel.asm.get(stage)
        if text is None:
            print(f"--- {stage}: missing ---")
            continue
        if isinstance(text, bytes):
            text = text.decode("utf-8", "replace")
        (OUT / f"{stage}.txt").write_text(text)
        if stage == "ttir":
            show(stage, text, ["tt.func", "tt.load", "tt.store", "arith.addf", "tt.make_range"])
        elif stage == "ttgir":
            show(stage, text, ["ttg.blocked", "ttg.convert_layout", "tt.load", "tt.store", "ttg.local_alloc"])
        elif stage == "llir":
            show(stage, text, ["define", "load", "store", "fadd", "getelementptr"])
        else:
            show(stage, text, [".visible .entry", "ld.global", "st.global"], limit=6)
            print("--- ptx selected instructions ---")
            for needle in (".visible .entry", "ld.global.b32", "add.rn.f32", "st.global.b32"):
                for line in text.splitlines():
                    if needle in line:
                        print(line.strip())
                        break

    print("n_regs", kernel.n_regs)
    print("n_spills", kernel.n_spills)
    print("shared_bytes", kernel.metadata.shared)
    print("num_warps", kernel.metadata.num_warps)
    print("stages_dir", OUT)


if __name__ == "__main__":
    main()
