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
"""Random vectorial Boolean function implementation.

This module provides a vectorial Boolean function whose output values are
generated pseudorandomly from a given seed.
"""

# qace-sdk
# vbf/_random_cipher.py
# Fraunhofer AISEC

from qace.vbf import VectorialBooleanFunctionFull
import gc
import random


# NOTE: Due to its construction, we do not think a test is necessary.
class RandomVBF(VectorialBooleanFunctionFull):
    """A vectorial Boolean function with pseudorandomly generated outputs.

    The function maps each input to a randomly chosen output value,
    determined by an optional random seed for reproducibility.

    Attributes:
        m: The number of input bits.
        n: The number of output bits.
    """

    def __init__(self, m: int, n: int, rng_seed: int | None = None) -> None:
        """Initializes the instance with random output values.

        Generates a random lookup table of output values using the provided
        seed. The global random state is preserved and restored after
        construction.

        Args:
            m: The number of input bits.
            n: The number of output bits.
            rng_seed: Optional seed for the random number generator to enable
                reproducibility.
        """
        state_backup = random.getstate()
        random.seed(rng_seed)

        c = random.choices(range(1 << n), k=1 << m)

        # Restore the rng state
        random.setstate(state_backup)

        def f(x: int) -> int:
            return c[x]

        VectorialBooleanFunctionFull.__init__(self, f, m, n)
        del f
        del c
        gc.collect()
