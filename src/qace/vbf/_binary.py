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
"""Utilities for working with binary vectors.

This module provides helper functions for binary operations such as
computing the parity of an integer and converting integers to fixed-width
binary string representations.
"""

##############################################
# qace                                       #
# _binary.py                                  #
# Utilities for working with binary vectors. #
##############################################

import numba as nb


@nb.jit
def get_parity(x: int) -> int:
    """Optimized parity computation.

    Args:
        x: The integer >= 0.

    Returns:
        Parity of the binary representation of the integer.
    """
    r = 0
    while x != 0:
        r ^= x & 1
        x >>= 1
    return r


def int_to_fixed_bin_str(x: int, n: int) -> str:
    """Get binary representation with a fixed number of bits of an integer.

    Args:
        x: The integer.
        n: Number of bits.

    Returns:
        `n`-bit binary representation of `x` as string without prefix.
    """
    return format(x, f"#0{n+2}b")[2:]


def int_to_twos_complement_repr(x: int, n: int) -> int:
    """Get twos complement representation with fixed number of bits of an integer.

    Args:
        x: The integer.
        n: Number of bits including the sign bit.

    Returns:
        Integer encoding the `n`-bit long twos complement of `x`.
    """
    s = int(x < 0)
    m = (1 << (n - 1)) - 1
    a = abs(x) & m
    if s == 1:
        a ^= m
        a += 1
        a &= m
    y = (s << (n - 1)) | a
    return y
