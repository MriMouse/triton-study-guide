#!/usr/bin/env python3
"""Pure Python index exercise; does not import Triton or simulate GPU layouts."""


def trace(n_elements, block_size, verbose=False):
    if n_elements < 0 or block_size <= 0:
        raise ValueError("Require N >= 0 and B > 0")
    grid = (n_elements + block_size - 1) // block_size
    x = list(range(n_elements))
    y = [i + 1 for i in range(n_elements)]
    output = [None] * n_elements
    writes = [0] * n_elements
    for pid in range(grid):
        offsets = [pid * block_size + j for j in range(block_size)]
        mask = [offset < n_elements for offset in offsets]
        if verbose:
            print(f"pid={pid}: offsets={offsets}, mask={mask}")
        # Mask BEFORE indexing the Python lists, like the logical load contract.
        loaded_x = [x[i] if valid else 0 for i, valid in zip(offsets, mask)]
        loaded_y = [y[i] if valid else 0 for i, valid in zip(offsets, mask)]
        tile = [a + b for a, b in zip(loaded_x, loaded_y)]
        if verbose:
            print(f"  x_tile={loaded_x}, y_tile={loaded_y}, sum_tile={tile}")
        for i, valid, value in zip(offsets, mask, tile):
            if valid:
                output[i] = value
                writes[i] += 1
    if verbose:
        print(f"output={output}")
        print(f"write_counts={writes}")
    assert writes == [1] * n_elements, "Each valid output must be written exactly once"
    assert output == [a + b for a, b in zip(x, y)], "Numerical index exercise failed"
    return grid


def main():
    print("Logical index exercise, not a GPU/thread-layout simulator")
    print("Masked-out CPU slots use explicit zero fill (the other=0 teaching variant).")
    print("N=10, B=4:")
    assert trace(10, 4, verbose=True) == 3
    cases = [(0, 4), (1, 4), (3, 4), (4, 4), (5, 4), (1025, 1024)]
    for n_elements, block_size in cases:
        grid = trace(n_elements, block_size)
        print(f"PASS N={n_elements}, B={block_size}, grid={grid}")
    print("All valid indices covered exactly once; CPU teaching sums match.")
    print("N=0 here only validates the integer model; no empty Triton launch was tested.")


if __name__ == "__main__":
    main()
