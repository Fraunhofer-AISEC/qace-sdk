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
"""Lookup table-based vectorial Boolean function implementation.

This module provides a vectorial Boolean function defined by an explicit
lookup table mapping inputs to outputs.
"""

# qace-sdk
# vbf/_lookup_table.py
# Fraunhofer AISEC

from qace.vbf import VectorialBooleanFunctionFull


class LookupTableVBF(VectorialBooleanFunctionFull):
    """A vectorial Boolean function defined by a lookup table.

    Attributes:
        m: The number of input bits.
        n: The number of output bits.
    """

    def __init__(self, m: int, n: int, lookup: dict[int, int]) -> None:
        """Initializes the instance from a lookup table.

        Args:
            m: The number of input bits.
            n: The number of output bits.
            lookup: A sequence mapping each input value to its corresponding
                output value.
        """
        VectorialBooleanFunctionFull.__init__(self, lambda x: lookup[x], m, n)
