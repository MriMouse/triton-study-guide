#!/usr/bin/env python3
"""CPU arithmetic companion: no torch, Triton, compiler or GPU simulation."""

import math


def stable_softmax(row):
    maximum = max(row)
    numerators = [math.exp(x - maximum) for x in row]
    total = sum(numerators)
    return [x / total for x in numerators]


def online_weighted_sum(scores, values, block_size):
    """Finite nonempty scores, scalar values; FP64 Python math, not tl.exp."""
    maximum = -math.inf
    denominator = 0.0
    numerator = 0.0
    for start in range(0, len(scores), block_size):
        score_block = scores[start:start + block_size]
        value_block = values[start:start + block_size]
        new_maximum = max(maximum, max(score_block))
        alpha = math.exp(maximum - new_maximum)
        weights = [math.exp(score - new_maximum) for score in score_block]
        numerator = alpha * numerator + sum(w * v for w, v in zip(weights, value_block))
        denominator = alpha * denominator + sum(weights)
        maximum = new_maximum
        print(f"  block={start // block_size}: max={maximum}, "
              f"denominator={denominator:.9f}, numerator={numerator:.9f}")
    return numerator / denominator


def main():
    print("CPU arithmetic examples; not GPU execution, IR output or timing")
    offset = 1 * 3 + 2 * 1
    assert offset == 5
    print(f"A[1,2], stride=(3,1), FP32: element offset={offset}, byte offset={4 * offset}")

    probabilities = stable_softmax([1, 2, 3])
    shifted = stable_softmax([1001, 1002, 1003])
    assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(probabilities, shifted))
    assert math.isclose(sum(probabilities), 1.0, rel_tol=1e-12)
    print("softmax([1,2,3]) =", [round(x, 9) for x in probabilities])

    a = [[1, 2, 3], [4, 5, 6]]
    b = [[7, 8], [9, 10], [11, 12]]
    c = [[sum(a[i][k] * b[k][j] for k in range(3))
          for j in range(2)] for i in range(2)]
    assert c == [[58, 64], [139, 154]]
    print("2x3 @ 3x2 =", c)

    scores = [1.0, 2.0, 3.0, 4.0]
    values = [10.0, 20.0, 30.0, 40.0]
    reference = sum(p * v for p, v in zip(stable_softmax(scores), values))
    online = online_weighted_sum(scores, values, block_size=2)
    assert math.isclose(online, reference, rel_tol=1e-12)
    print(f"online weighted sum={online:.9f}; full-row reference={reference:.9f}")
    print("PASS: address, softmax shift, GEMM and online rescaling arithmetic")


if __name__ == "__main__":
    main()
