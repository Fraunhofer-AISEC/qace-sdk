# Copyright 2026 Fraunhofer AISEC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Primitives for linear cryptanalysis of vectorial Boolean functions.

This module provides functions to compute quantities commonly used in linear
cryptanalysis, such as the mask equality count, mask equality probability,
Walsh transform, correlation, and bias, as well as sample-based approximations
for the mask equality count and bias.
"""

from collections.abc import Sequence

from qace.vbf import get_parity, VectorialBooleanFunction


def mask_equality_count(f: VectorialBooleanFunction, a: int, b: int) -> int:
    """Computes the so-called "mask equality count" for a function and input-/output-masks.

    Args:
        f: The function.
        a: Input mask.
        b: Output mask.

    Returns:
        The number of x with <x, a> + <f(x), b> = 0.
    """
    c = 0
    for x in range(1 << f.m):
        y = f.eval(x)
        e = get_parity(x & a) + get_parity(y & b)
        c += e % 2 == 0
    return c


def mask_equality_probability(
    f: VectorialBooleanFunction,
    a: int,
    b: int,
    mec: int | None = None,
) -> float:
    """Converts a mask equality count to its associated probability.

    Args:
        f: The function.
        a: Input mask.
        b: Output mask.
        mec: Optionally, the precomputed mask equality count.

    Returns:
        The mask_equality_count(f, a, b) divided by (1<<f.m).
    """
    if mec == None:
        p = mask_equality_count(f, a, b) / (1 << f.m)
    else:
        p = mec / (1 << f.m)
    return p


def walsh_transform(
    f: VectorialBooleanFunction,
    a: int,
    b: int,
    mec: int | None = None,
) -> int:
    """Computes the Walsh transform for given masks with respect to a function.

    The function initializes a variable to accumulate the Walsh transform
    result. It calculates the total number of possible inputs (M = 2 ** m) and
    iterates through each possible input x. For each input, it computes the
    parity of the bits specified by a and the output of the function f masked
    by b. The result is adjusted by raising -1 to the power of the computed
    parity, effectively contributing either +1 or -1 to the total Walsh
    transform value. The function returns the accumulated result.

    Args:
        f: The function.
        a: Input mask.
        b: Output mask.
        mec: Optionally, the precomputed mask equality count.

    Returns:
        The computed Walsh transform value, which indicates the correlation
        between the two masks with respect to the function `f`.
    """
    if mec == None:
        e = mask_equality_count(f, a, b)
    else:
        e = mec
    w = 2 * e - (1 << f.m)
    return w


def correlation(
    f: VectorialBooleanFunction,
    a: int,
    b: int,
    mec: int | None = None,
) -> float:
    """Computes the correlation of a function with respect to input and output masks.

    The function calculates the total number of possible inputs (N = 2^n) and
    then calls the walsh_transform function to compute the Walsh transform
    value for the given masks a and b. The result is normalized by dividing by
    N, providing a correlation value that indicates the strength of the
    relationship between the masks with respect to the function f. This
    correlation value is useful for analyzing the influence of specific bits
    on the output of the Boolean function.

    Args:
        f: The function.
        a: Input mask.
        b: Output mask.
        mec: Optionally, the precomputed mask equality count.

    Returns:
        The computed correlation value, normalized by the total number of
        inputs `1<<f.m`.
    """
    corr = walsh_transform(f, a, b, mec) / (1 << f.m)
    return corr


def bias(
    f: VectorialBooleanFunction,
    a: int,
    b: int,
    mec: int | None = None,
) -> float:
    """Computes the bias of a function with respect to input and output masks.

    Args:
        f: The function.
        a: Input mask.
        b: Output mask.
        mec: Optionally, the precomputed mask equality count.

    Returns:
        The computed bias.
    """
    return correlation(f, a, b, mec) / 2


def approximate_mec(
    sample_inputs: Sequence[int],
    sample_outputs: Sequence[int],
    a: int,
    b: int,
) -> int:
    """Computes the approximate mask equality count from sample data.

    Args:
        sample_inputs: Input values to use.
        sample_outputs: Output values corresponding to the sample inputs.
        a: Input mask.
        b: Output mask.

    Returns:
        The computed approximate mask equality count.
    """
    approx_mec = sum(
        [
            ((get_parity(a & x) ^ get_parity(b & y)) % 2) ^ 1
            for (x, y) in zip(sample_inputs, sample_outputs)
        ]
    )
    return approx_mec


def approximate_bias(
    sample_inputs: Sequence[int],
    sample_outputs: Sequence[int],
    a: int,
    b: int,
) -> float:
    """Computes the approximate bias from sample data.

    Args:
        sample_inputs: Input values to use.
        sample_outputs: Output values corresponding to the sample inputs.
        a: Input mask.
        b: Output mask.

    Returns:
        The computed approximate bias.
    """
    s = len(sample_inputs)
    approx_bias = (2 * approximate_mec(sample_inputs, sample_outputs, a, b) - s) / (
        2 * s
    )
    return approx_bias
