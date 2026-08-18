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
"""XOR cipher as a vectorial Boolean function.

This module provides a vectorial Boolean function that implements a simple
XOR cipher, combining two m-bit halves of the input via bitwise XOR.
"""

# qace-sdk
# vbf/_xor_cipher.py
# Fraunhofer AISEC

from qace.vbf import VectorialBooleanFunctionFull


class XORCipherVBF(VectorialBooleanFunctionFull):
    """A vectorial Boolean function implementing a XOR cipher.

    Takes a 2m-bit input, splits it into two m-bit halves, and produces
    their bitwise XOR as the m-bit output.

    Attributes:
        m: The number of input bits (2*m for the two halves).
        n: The number of output bits (m).
    """

    def __init__(self, m: int) -> None:
        """Initializes the XOR cipher vectorial Boolean function.

        Args:
            m: The bit width of each input half. The total input size is 2*m
                bits and the output size is m bits.
        """

        def xor(x: int) -> int:
            mask = (1 << m) - 1
            return ((x & (mask << m)) >> m) ^ (x & mask)

        VectorialBooleanFunctionFull.__init__(self, xor, 2 * m, m)
        del xor
